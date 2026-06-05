"""
core/dataset.py
===============
SE3AF dataset — loads CSV data, builds graphs/embeddings, caches results.
Merged from: se3af/dataset.py

FIXES CARRIED FORWARD
---------------------
BUG-01  CRITICAL  Missing SMILES → "C" methane fallback → scientific corruption.
        Fix: empty/invalid SMILES → _dummy_graph(), row flagged as dummy.
BUG-02  HIGH      _is_valid_graph() only checked graph is not None.
        Fix: full dimension/shape validation via preprocessing.validate_graph().
BUG-03  HIGH      Python hash() for cache keys → non-deterministic cross-session.
        Fix: MD5-based keys.
BUG-04  MEDIUM    No error on missing label column in supervised training.
        Fix: raise ValueError if label column absent when supervised=True.
BUG-05  MEDIUM    collate_fn moved device tensors inside DataLoader workers
        (multiprocessing) → CUDA init in worker → crash on Linux.
        Fix: device moves happen ONLY in the training loop, not here.
N05     MEDIUM    __import__() anti-pattern → proper top-level imports.
N06     MEDIUM    Duplicate/aliased hashlib import → single top-level import.

F01     HIGH      NEW BUG — RF stacker bug (fixed in trainer.py).
        PROTACDataset is often wrapped in torch.utils.data.Subset.
        Subset has no ._df attribute, so _build_fp_matrix() received None.
        Fix applied in trainer.py/_build_fp_matrix(): unwrap Subset automatically.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from core.preprocessing import (
    ESM_DIM,
    ESMEncoder,
    _zero_embedding,
    _dummy_graph,
    smiles_to_graph,
    validate_graph,
)
from core.utils import (
    COL_E3_SMILES,
    COL_LABEL,
    COL_LIG_SEQ,
    COL_LNK_SMILES,
    COL_TGT_SEQ,
    COL_TGT_SMILES,
    _parse_label,
    discover_datasets,
)

# Must match rebuild_cache.py CACHE_VERSION
_CACHE_VERSION = "v3.1"


# ===========================================================================
# Sample dataclass
# ===========================================================================

class PROTACSample:
    """One PROTAC training / inference sample.

    Uses __slots__ for memory efficiency (no __dict__ overhead per sample).

    v3.6.0 AF-02: tgt_af_feat / e3_af_feat store mean-pooled AlphaFold
    Cα features (shape (4,) or zeros).  They are appended to ESM embeddings
    in protac_collate_fn when af_extra_dim > 0.
    """
    __slots__ = [
        "tgt_graph", "e3_graph", "lnk_graph",
        "tgt_esm", "e3_esm",
        "tgt_af_feat", "e3_af_feat",   # v3.6.0 AF-02
        "label", "has_label",
        "idx", "is_corrupted",
    ]

    def __init__(
        self,
        tgt_graph:    Data,
        e3_graph:     Data,
        lnk_graph:    Data,
        tgt_esm:      np.ndarray,
        e3_esm:       np.ndarray,
        label:        float,
        has_label:    bool,
        idx:          int,
        is_corrupted: bool = False,
        tgt_af_feat:  Optional[np.ndarray] = None,   # v3.6.0 AF-02
        e3_af_feat:   Optional[np.ndarray] = None,   # v3.6.0 AF-02
    ) -> None:
        self.tgt_graph    = tgt_graph
        self.e3_graph     = e3_graph
        self.lnk_graph    = lnk_graph
        self.tgt_esm      = tgt_esm
        self.e3_esm       = e3_esm
        self.tgt_af_feat  = tgt_af_feat   # v3.6.0 AF-02: (4,) or None
        self.e3_af_feat   = e3_af_feat    # v3.6.0 AF-02: (4,) or None
        self.label        = label
        self.has_label    = has_label
        self.idx          = idx
        self.is_corrupted = is_corrupted


# ===========================================================================
# Dataset
# ===========================================================================

class PROTACDataset(Dataset):
    """PyTorch Dataset for PROTAC activity prediction.

    Parameters
    ----------
    data_root : str
        Path to a directory of CSV files or a single CSV file.
    cache_dir : str
        Directory for molecular graph cache.
    use_coords : bool
        Whether to include 3D conformer coordinates.
    max_atoms : int
        Skip molecule (return dummy graph) if it exceeds this atom count.
    esm_cache_dir : str | None
        If provided, ESM embeddings are cached here as .npy files.
    supervised : bool
        If True, raise ``ValueError`` when the label column is missing
        (BUG-04 fix — prevents silent training on zeros).
    """

    def __init__(
        self,
        data_root: str,
        cache_dir: str = ".cache",
        use_coords: bool = False,
        max_atoms: int = 256,
        esm_cache_dir: Optional[str] = None,
        supervised: bool = False,
        alphafold_store=None,   # v3.6.0 AF-02: optional AlphaFoldStore instance
        af_protein_id_col: Optional[str] = None,  # column name for protein ID lookup
    ) -> None:
        super().__init__()

        self.cache_dir  = Path(cache_dir)
        self.use_coords = use_coords
        self.max_atoms  = max_atoms
        self._af_store  = alphafold_store   # v3.6.0 AF-02
        self._af_id_col = af_protein_id_col

        _esm_cache = esm_cache_dir or str(self.cache_dir / "esm")
        self._esm_encoder = ESMEncoder(cache_dir=_esm_cache)

        dfs = discover_datasets(data_root)
        df  = pd.concat(dfs, ignore_index=True)

        # BUG-04 fix: fail loudly for supervised training without labels
        if supervised and COL_LABEL not in df.columns:
            raise ValueError(
                f"Label column '{COL_LABEL}' not found for supervised training. "
                f"Available columns: {list(df.columns)}"
            )

        self._df      = df
        self._samples: List[PROTACSample] = []
        self._build_samples()

    # ------------------------------------------------------------------
    def _build_samples(self) -> None:
        df  = self._df
        N   = len(df)
        bad = 0

        def _safe_smi(val) -> str:
            """Guard against pandas NaN leaking as literal 'nan' string."""
            if val is None:
                return ""
            s = str(val).strip()
            return "" if s.lower() in ("nan", "none", "", "null") else s

        def _safe_seq(val) -> str:
            if val is None:
                return ""
            s = str(val).strip()
            return "" if s.lower() in ("nan", "none", "", "null") else s

        for i, row in df.iterrows():
            tgt_smi = _safe_smi(row.get(COL_TGT_SMILES))
            e3_smi  = _safe_smi(row.get(COL_E3_SMILES))
            lnk_smi = _safe_smi(row.get(COL_LNK_SMILES))
            tgt_seq = _safe_seq(row.get(COL_TGT_SEQ))
            e3_seq  = _safe_seq(row.get(COL_LIG_SEQ))

            raw_label  = row.get(COL_LABEL, None)
            parsed_lbl = _parse_label(raw_label) if raw_label is not None else None
            has_label  = (parsed_lbl is not None)
            label      = float(parsed_lbl) if has_label else 0.0

            # BUG-01 fix: no methane fallback — invalid SMILES → dummy graph
            tgt_g = self._load_or_build_graph(tgt_smi)
            e3_g  = self._load_or_build_graph(e3_smi)
            lnk_g = self._load_or_build_graph(lnk_smi)

            # BUG-02 fix: full graph validation
            # A sample is "corrupted" when a graph is not dummy but also fails
            # structural validation (e.g. NaN features, wrong dimensions).
            # A graph that is IS_DUMMY is not "corrupted" — it's just missing SMILES.
            corrupted = any(
                (not getattr(g, "is_dummy", False)) and (not validate_graph(g))
                for g in [tgt_g, e3_g, lnk_g]
            )
            if any(getattr(g, "is_dummy", False) for g in [tgt_g, e3_g, lnk_g]):
                bad += 1

            tgt_esm = self._esm_encoder.encode(tgt_seq) if tgt_seq else _zero_embedding()
            e3_esm  = self._esm_encoder.encode(e3_seq)  if e3_seq  else _zero_embedding()

            # v3.6.0 AF-02 / V40-PHASE-9: look up AlphaFold Cα features
            # V40: pLDDT confidence-aware pooling replaces simple mean pooling
            tgt_af_feat: Optional[np.ndarray] = None
            e3_af_feat:  Optional[np.ndarray] = None
            if self._af_store is not None and getattr(self._af_store, "enabled", False):
                try:
                    from core.alphafold import af_residue_features, af_confidence_weighted_mean_pool
                    # V38-FIX-E3-AF-01: _e3_id was HARDCODED to "" which meant
                    # E3 ligase AlphaFold features were NEVER fetched, even when
                    # CRBN/VHL PDB files were present in data/alphafold/. Fix:
                    # use the same ID-column or sequence-hash approach for E3 too.
                    _id_col = self._af_id_col
                    _tgt_id = str(row.get(_id_col, "")).strip() if _id_col else ""
                    # For E3, check for dedicated e3_protein_id column first,
                    # then fall back to the ligase_seq hash.
                    _e3_id_col = None
                    if _id_col and "e3" not in str(_id_col).lower():
                        # Try an e3-specific column if one exists
                        for _candidate in ("e3_protein_id", "ligase_id", "e3_id"):
                            if _candidate in row.index if hasattr(row, "index") else _candidate in df.columns:
                                _e3_id_col = _candidate
                                break
                    _e3_id = str(row.get(_e3_id_col, "")).strip() if _e3_id_col else ""
                    # If no dedicated E3 ID column, try ligase_seq as hash key
                    if not _e3_id and e3_seq:
                        import hashlib as _hl
                        _e3_id = _hl.md5(e3_seq.encode("utf-8")).hexdigest()[:16]

                    if _tgt_id:
                        _coords, _plddt = self._af_store.get(_tgt_id)
                        if _coords is not None:
                            # V40-PHASE-9: pLDDT confidence-aware pooling
                            tgt_af_feat = af_confidence_weighted_mean_pool(
                                _coords, _plddt,
                                confidence_threshold=70.0,
                                low_confidence_weight=0.1,
                            )  # (4,)
                    if _e3_id:
                        _e3_coords, _e3_plddt = self._af_store.get(_e3_id)
                        if _e3_coords is not None:
                            # V40-PHASE-9: pLDDT confidence-aware pooling
                            e3_af_feat = af_confidence_weighted_mean_pool(
                                _e3_coords, _e3_plddt,
                                confidence_threshold=70.0,
                                low_confidence_weight=0.1,
                            )  # (4,)
                except Exception:
                    pass  # AF lookup failure is non-fatal

            self._samples.append(PROTACSample(
                tgt_graph=tgt_g,
                e3_graph=e3_g,
                lnk_graph=lnk_g,
                tgt_esm=tgt_esm,
                e3_esm=e3_esm,
                label=label,
                has_label=has_label,
                idx=i,
                is_corrupted=corrupted,
                tgt_af_feat=tgt_af_feat,   # v3.6.0 AF-02
                e3_af_feat=e3_af_feat,     # v3.6.0 AF-02
            ))

        if bad > 0:
            warnings.warn(
                f"[PROTACDataset] {bad}/{N} samples have at least one dummy graph "
                "(invalid/empty SMILES). They will be included but flagged."
            )

    # ------------------------------------------------------------------
    def _load_or_build_graph(self, smiles: str) -> Data:
        """Try loading from cache; build fresh if missing or stale.

        Cache key is MD5(smiles) to match rebuild_cache.py exactly.
        N05/N06 fix: proper top-level imports (no __import__()).
        """
        if smiles:
            md5_key = hashlib.md5(smiles.encode("utf-8")).hexdigest()
        else:
            md5_key = "empty"

        cache_path = self.cache_dir / "graphs" / f"{md5_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as fh:
                    payload = pickle.load(fh)
                if isinstance(payload, dict) and "data" in payload:
                    g = payload["data"]
                    if validate_graph(g) or getattr(g, "is_dummy", False):
                        return g
            except Exception:
                pass

        # Build fresh
        g = smiles_to_graph(smiles, use_coords=self.use_coords, max_atoms=self.max_atoms)

        # Atomic cache write
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload  = {"version": _CACHE_VERSION, "data": g, "smiles": smiles}
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(cache_path.parent), suffix=".tmp"
            )
            with os.fdopen(tmp_fd, "wb") as fh:
                pickle.dump(payload, fh, protocol=4)
            Path(tmp_path).replace(cache_path)
        except Exception:
            pass

        return g

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> PROTACSample:
        return self._samples[idx]


# ===========================================================================
# Collate function
# ===========================================================================

def protac_collate_fn(samples: List[PROTACSample]) -> dict:
    """Custom collate for :class:`PROTACSample` batches.

    BUG-05 fix: NO device moves here.  All tensors remain on CPU.
    The training loop is responsible for moving data to the correct device.

    v3.6.0 AF-02: If tgt_af_feat / e3_af_feat are present in samples,
    they are stacked and appended to tgt_esm / e3_esm.  This produces
    (B, esm_dim + af_extra_dim) tensors consumed by the updated esm_proj
    in CrossInteractionFusion.  When AF features are absent (all None),
    the ESM tensors are unchanged (B, esm_dim).
    """
    tgt_graphs = Batch.from_data_list([s.tgt_graph for s in samples])
    e3_graphs  = Batch.from_data_list([s.e3_graph  for s in samples])
    lnk_graphs = Batch.from_data_list([s.lnk_graph for s in samples])

    tgt_esm_arr = np.stack([s.tgt_esm for s in samples])
    e3_esm_arr  = np.stack([s.e3_esm  for s in samples])

    # v3.6.0 AF-02: Append AlphaFold mean-pooled features if available
    _has_tgt_af = any(getattr(s, "tgt_af_feat", None) is not None for s in samples)
    _has_e3_af  = any(getattr(s, "e3_af_feat",  None) is not None for s in samples)

    if _has_tgt_af:
        # Build (B, 4) matrix; use zeros for samples without AF
        _af_dim = 4
        tgt_af_mat = np.zeros((len(samples), _af_dim), dtype=np.float32)
        for idx, s in enumerate(samples):
            _feat = getattr(s, "tgt_af_feat", None)
            if _feat is not None:
                tgt_af_mat[idx, :min(len(_feat), _af_dim)] = _feat[:_af_dim]
        tgt_esm_arr = np.concatenate([tgt_esm_arr, tgt_af_mat], axis=1)

    if _has_e3_af:
        _af_dim = 4
        e3_af_mat = np.zeros((len(samples), _af_dim), dtype=np.float32)
        for idx, s in enumerate(samples):
            _feat = getattr(s, "e3_af_feat", None)
            if _feat is not None:
                e3_af_mat[idx, :min(len(_feat), _af_dim)] = _feat[:_af_dim]
        e3_esm_arr = np.concatenate([e3_esm_arr, e3_af_mat], axis=1)

    tgt_esm = torch.tensor(tgt_esm_arr, dtype=torch.float32)
    e3_esm  = torch.tensor(e3_esm_arr,  dtype=torch.float32)

    labels     = torch.tensor([s.label     for s in samples], dtype=torch.float32)
    has_labels = torch.tensor([s.has_label for s in samples], dtype=torch.bool)
    idxs       = torch.tensor([s.idx       for s in samples], dtype=torch.long)

    return {
        "tgt_graph":  tgt_graphs,
        "e3_graph":   e3_graphs,
        "lnk_graph":  lnk_graphs,
        "tgt_esm":    tgt_esm,
        "e3_esm":     e3_esm,
        "labels":     labels,
        "has_labels": has_labels,
        "idxs":       idxs,
    }
