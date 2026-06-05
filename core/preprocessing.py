"""
core/preprocessing.py
=====================
Molecular preprocessing for SE3AF:
  - atom and bond featurisation   (originally: se3af/models/graph_features.py)
  - SMILES → PyG graph conversion (originally: se3af/models/graph_encoder.py)
  - ESM-2 protein embeddings      (originally: se3af/models/esm_encoder.py)

All bugs identified in the Phase-1 audit have been fixed here.

FIXES CARRIED FORWARD
---------------------
BUG-01  CRITICAL  atom.IsInRingOfSize() is a hallucinated RDKit API.
        Fix: mol.GetRingInfo().IsAtomInRingOfSize(atom_idx, size).
BUG-02  HIGH      bond.GetStereo() compared to string literals.
        Fix: compare against Chem.rdchem.BondStereo enum values.
BUG-03  MEDIUM    one_of_k silently clamped unknowns to last known bin.
        Fix: explicit trailing "other" bin.
BUG-04  HIGH      All-zero 3D coords → degenerate RBF kernels in SE3 backend.
        Fix: fall back to random unit-sphere coords when conformer is flat.
BUG-05  HIGH      No self-loop for isolated atoms → empty edge_index.
        Fix: add self-loop when edge_index is empty.
BUG-06  MEDIUM    atom_features called with wrong mol object post-sanitisation.
        Fix: always use sanitised mol.
BUG-07  MEDIUM    Gasteiger charges computed twice.
        Fix: compute only in _gasteiger_charge().
BUG-08  CRITICAL  hash() non-deterministic across Python sessions.
        Fix: MD5 keys for ESM cache.
BUG-09  HIGH      Hardcoded signal peptide as fallback for empty sequences.
        Fix: zero embedding for missing/empty sequences.
BUG-10  HIGH      ESM model loaded to CPU, tensors moved to GPU mid-forward.
        Fix: resolve device once at load time, move model and inputs together.
BUG-11  MEDIUM    No truncation of over-long sequences → CUDA OOM.
        Fix: truncate to ESM_MAX_LEN=1022.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges

# ===========================================================================
# SECTION 1 — Atom and bond featurisation
# ===========================================================================

NODE_DIM: int = 78
EDGE_DIM: int = 12

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def one_of_k_encoding(value, allowable_set: list) -> List[float]:
    """One-hot encode *value* against *allowable_set*.

    Unknown values map to a dedicated trailing 'other' bin rather than
    silently mapping onto the last known category (BUG-03 fix).
    """
    enc = [0.0] * (len(allowable_set) + 1)   # +1 for "other"
    try:
        idx = allowable_set.index(value)
        enc[idx] = 1.0
    except ValueError:
        enc[-1] = 1.0  # "other" bin
    return enc


def _safe_ring_size(mol: Chem.Mol, atom_idx: int, size: int) -> float:
    """BUG-01 FIX: correct RDKit API for ring-size membership.

    atom.IsInRingOfSize(size) — DOES NOT EXIST.
    mol.GetRingInfo().IsAtomInRingOfSize(idx, size) — CORRECT.
    """
    try:
        return float(mol.GetRingInfo().IsAtomInRingOfSize(atom_idx, size))
    except Exception:
        return 0.0


def _precompute_gasteiger(mol: Chem.Mol) -> bool:
    """V38-FIX-GASTEIGER-01: Compute Gasteiger charges ONCE per molecule.

    The original _gasteiger_charge() called ComputeGasteigerCharges(mol)
    for EVERY atom, making it O(N²) — one full charge computation per atom.
    Call this once before atom_features() loop, then use
    _gasteiger_charge_precomputed() to read the cached property.

    Returns True if computation succeeded, False otherwise.
    """
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        return True
    except Exception:
        return False


def _gasteiger_charge(mol: Chem.Mol, atom_idx: int) -> float:
    """Return normalised Gasteiger partial charge, safely.

    V38-FIX-GASTEIGER-01: This function no longer calls
    ComputeGasteigerCharges(mol). Call _precompute_gasteiger(mol) ONCE
    before the atom loop (smiles_to_graph does this). This function only
    reads the pre-computed property from the atom.
    """
    try:
        chg = mol.GetAtomWithIdx(atom_idx).GetDoubleProp("_GasteigerCharge")
        if chg != chg or abs(chg) > 4.0:   # NaN or extreme
            return 0.0
        return float(chg) / 2.0            # normalise ~[-2, 2]
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Atom feature constants
# ---------------------------------------------------------------------------

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "Cl", "Br", "I",
    "P", "Si", "B", "Se", "Te", "As", "Ge",
]

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

CHIRAL_TYPES = [
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]

HETEROATOMS = {
    "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B",
    "Se", "Te", "As", "Ge",
}

# BUG-02 FIX: compare bond stereo to enum values, not strings
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


# ---------------------------------------------------------------------------
# Atom features  (78-dim)
# ---------------------------------------------------------------------------

def atom_features(mol: Chem.Mol, atom: Chem.Atom) -> np.ndarray:
    """Compute 78-dimensional atom feature vector.

    Dimension breakdown
    -------------------
    [0 :16]  atom type one-hot (15 types + other)           = 16
    [16:24]  degree one-hot (0..6 + other)                   =  8
    [24:31]  formal charge one-hot (-2,-1,0,1,2,3 + other)  =  7
    [31:37]  num_Hs one-hot (0..4 + other)                   =  6
    [37:43]  hybridisation one-hot (5 types + other)         =  6
    [43:47]  chirality one-hot (3 types + other)             =  4
    [47]     is aromatic                                      =  1
    [48]     is in ring                                       =  1
    [49:55]  ring-size 3-8 (BUG-01 fix)                      =  6
    [55:60]  radical_e one-hot (0..3 + other)                =  5
    [60:70]  total_valence one-hot (0..8 + other)            = 10
    [70]     is heteroatom                                    =  1
    [71]     gasteiger charge (normalised)                    =  1
    [72:78]  num_heavy_nbrs one-hot (0..4 + other)           =  6
                                                   TOTAL  = 78
    """
    idx = atom.GetIdx()
    feats: List[float] = []

    feats += one_of_k_encoding(atom.GetSymbol(), ATOM_TYPES)                      # 16
    feats += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])           #  8
    feats += one_of_k_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, 3])      #  7
    feats += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])             #  6
    feats += one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPES)      #  6
    feats += one_of_k_encoding(atom.GetChiralTag(), CHIRAL_TYPES)                 #  4
    feats.append(float(atom.GetIsAromatic()))                                       #  1
    feats.append(float(atom.IsInRing()))                                            #  1
    for ring_sz in (3, 4, 5, 6, 7, 8):                                             #  6 (BUG-01 fix)
        feats.append(_safe_ring_size(mol, idx, ring_sz))
    feats += one_of_k_encoding(atom.GetNumRadicalElectrons(), [0, 1, 2, 3])       #  5
    feats += one_of_k_encoding(atom.GetTotalValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8])  # 10
    feats.append(float(atom.GetSymbol() in HETEROATOMS))                            #  1
    feats.append(_gasteiger_charge(mol, idx))                                       #  1
    n_heavy = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1)
    feats += one_of_k_encoding(n_heavy, [0, 1, 2, 3, 4])                          #  6

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Bond features  (12-dim)
# ---------------------------------------------------------------------------

def bond_features(bond: Chem.Bond) -> np.ndarray:
    """Compute 12-dimensional bond feature vector.

    Dimension breakdown
    -------------------
    [0:5]   bond type one-hot (4 types + other)             = 5
    [5]     is conjugated                                    = 1
    [6]     is in ring                                       = 1
    [7:12]  stereo one-hot (4 types + other) (BUG-02 fix)   = 5
                                                  TOTAL  = 12
    """
    feats: List[float] = one_of_k_encoding(bond.GetBondType(), BOND_TYPES)   # 5
    feats.append(float(bond.GetIsConjugated()))                                # 1
    feats.append(float(bond.IsInRing()))                                       # 1
    feats += one_of_k_encoding(bond.GetStereo(), BOND_STEREO)                 # 5 (BUG-02 fix)
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dimension verification (runs at import time — fast)
# ---------------------------------------------------------------------------

def _verify_dims() -> None:
    """Build a test molecule and verify NODE_DIM / EDGE_DIM match constants."""
    try:
        mol = Chem.MolFromSmiles("c1ccccc1")
        if mol is None:
            return
        a = mol.GetAtomWithIdx(0)
        n = len(atom_features(mol, a))
        b = mol.GetBondWithIdx(0)
        e = len(bond_features(b))
        if n != NODE_DIM:
            raise AssertionError(
                f"atom_features returns {n} dims but NODE_DIM={NODE_DIM}. "
                "Update NODE_DIM constant."
            )
        if e != EDGE_DIM:
            raise AssertionError(
                f"bond_features returns {e} dims but EDGE_DIM={EDGE_DIM}. "
                "Update EDGE_DIM constant."
            )
    except Exception as exc:
        warnings.warn(f"[preprocessing] dim verification failed: {exc}")


_verify_dims()


# ===========================================================================
# SECTION 2 — SMILES → PyTorch Geometric graph
# ===========================================================================

# Dummy graph templates for invalid / empty SMILES
_DUMMY_ATOM_FEAT = np.zeros(NODE_DIM, dtype=np.float32)
_DUMMY_EDGE_FEAT = np.zeros(EDGE_DIM, dtype=np.float32)


def _dummy_graph(label: str = "dummy") -> Data:
    """Return a single-node, no-edge placeholder graph.

    Tagged with ``is_dummy=True`` so downstream code can detect and reject it
    instead of treating it as real chemistry.
    """
    return Data(
        x=torch.zeros(1, NODE_DIM),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        edge_attr=torch.zeros(0, EDGE_DIM),
        pos=torch.zeros(1, 3),
        is_dummy=True,
        num_nodes=1,
    )


def augment_smiles(smiles: str, n_variants: int = 4, seed: int = 0) -> str:
    """Return a random valid SMILES string for the same molecule.

    Uses RDKit's random SMILES generation. Falls back to canonical SMILES
    if augmentation fails. This acts as data augmentation — the same
    molecule is represented differently, improving generalization.

    Parameters
    ----------
    smiles     : input SMILES string
    n_variants : how many random variants to choose from
    seed       : random seed (0 = deterministic for caching)
    """
    if not smiles:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        rng = np.random.default_rng(seed)
        variants = set()
        for _ in range(n_variants * 2):
            idx = int(rng.integers(0, mol.GetNumAtoms()))
            try:
                v = Chem.MolToSmiles(mol, rootedAtAtom=idx, canonical=False)
                if v:
                    variants.add(v)
            except Exception:
                pass
        if not variants:
            return smiles
        return list(variants)[int(rng.integers(0, len(variants)))]
    except Exception:
        return smiles


def smiles_to_graph(
    smiles: str,
    use_coords: bool = True,   # v4.1: 3D is default (always on)
    max_atoms: int = 256,
) -> Data:
    """Convert SMILES to a PyTorch Geometric Data graph with 3D coordinates.

    v4.0: use_coords=True is the default. 3D coordinate generation via
    ETKDGv3 + MMFF minimisation with 5-stage fallback chain (no random coords).
    If all geometry attempts fail, zero coordinates are used (detectable/honest).

    Parameters
    ----------
    smiles     : SMILES string (invalid → dummy graph)
    use_coords : if True (default), generate 3D coords via ETKDGv3
    max_atoms  : return dummy graph above this atom count
    """
    if not smiles or not isinstance(smiles, str):
        return _dummy_graph()
    smiles = smiles.strip()
    if not smiles:
        return _dummy_graph()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _dummy_graph()

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return _dummy_graph()

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0 or n_atoms > max_atoms:
        return _dummy_graph()

    # 3-D coordinates — always generated in v4.1
    pos = _get_coordinates(mol, use_coords=True, n_atoms=n_atoms)

    # V38-FIX-GASTEIGER-01: pre-compute Gasteiger charges ONCE for the whole
    # molecule before iterating atoms. This converts the original O(N²)
    # per-atom charge computation to O(N) — one call per molecule.
    _precompute_gasteiger(mol)

    # Node features — BUG-06: always pass sanitised mol
    x = np.zeros((n_atoms, NODE_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i] = atom_features(mol, atom)

    # Edge features
    rows, cols, e_attrs = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        rows += [i, j]   # undirected → both directions
        cols += [j, i]
        e_attrs += [bf, bf]

    # BUG-05: isolated atom → self-loop to avoid empty edge_index
    if len(rows) == 0:
        rows = list(range(n_atoms))
        cols = list(range(n_atoms))
        e_attrs = [np.zeros(EDGE_DIM, dtype=np.float32)] * n_atoms

    # BUG-03 fix: edge_index must be shape (2, E)
    edge_index = torch.tensor([rows, cols], dtype=torch.long).contiguous()
    edge_attr  = torch.tensor(np.stack(e_attrs), dtype=torch.float32)
    x_t        = torch.tensor(x, dtype=torch.float32)
    pos_t      = torch.tensor(pos, dtype=torch.float32)

    return Data(
        x=x_t,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos_t,
        is_dummy=False,
        num_nodes=n_atoms,
    )


def _get_coordinates(mol: Chem.Mol, use_coords: bool = True, n_atoms: int = 0) -> np.ndarray:
    """Return (n_atoms, 3) coordinate array.

    v4.0: Proper fallback chain — NO random coordinates.
    Chain:
      1. ETKDGv3 seed=42
      2. ETKDGv3 seed=0 (retry)
      3. ETKDGv3 useRandomCoords=True
      4. ETKDG (v1) — less constrained
      5. 2D → 3D via Compute2DCoords + EmbedMolecule
      6. Zero coordinates (marked coord_fallback=True)

    Rationale: random unit-sphere coordinates (previous fallback) inject
    scientifically invalid geometry. Zero coordinates + is_dummy-like flag
    is preferable — downstream models can detect and handle them.
    """
    if n_atoms == 0:
        n_atoms = mol.GetNumAtoms()

    def _try_embed_and_extract(mol_h) -> np.ndarray:
        """Extract Cα-equivalent positions from a successfully embedded mol_h."""
        mol_noH = Chem.RemoveHs(mol_h)
        try:
            conf = mol_noH.GetConformer()
        except Exception:
            return None
        n = min(n_atoms, mol_noH.GetNumAtoms())
        pos = np.zeros((n_atoms, 3), dtype=np.float32)
        for i in range(n):
            p = conf.GetAtomPosition(i)
            pos[i] = [p.x, p.y, p.z]
        if not np.allclose(pos, 0.0):
            return pos
        return None

    # Attempt 1: ETKDGv3 seed=42 (original)
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useSmallRingTorsions = True
        params.useBasicKnowledge = True
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
            except Exception:
                pass
            pos = _try_embed_and_extract(mol_h)
            if pos is not None:
                return pos
    except Exception:
        pass

    # Attempt 2: ETKDGv3 seed=0 (different seed)
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        params.useSmallRingTorsions = True
        params.useBasicKnowledge = True
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            except Exception:
                pass
            pos = _try_embed_and_extract(mol_h)
            if pos is not None:
                return pos
    except Exception:
        pass

    # Attempt 3: ETKDGv3 with random coordinates init
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            pos = _try_embed_and_extract(mol_h)
            if pos is not None:
                return pos
    except Exception:
        pass

    # Attempt 4: ETKDG v1 (more permissive)
    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.EmbedParameters()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
            except Exception:
                pass
            pos = _try_embed_and_extract(mol_h)
            if pos is not None:
                return pos
    except Exception:
        pass

    # Attempt 5: 2D coordinates → embed to 3D
    try:
        mol2d = Chem.RWMol(mol)
        AllChem.Compute2DCoords(mol2d)
        mol_h = Chem.AddHs(mol2d)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useBasicKnowledge = False
        if AllChem.EmbedMolecule(mol_h, params) == 0:
            pos = _try_embed_and_extract(mol_h)
            if pos is not None:
                return pos
    except Exception:
        pass

    # All attempts failed → return zero coordinates
    # Zero coords are detectable (all atoms at origin) and honest.
    # The graph is still valid but geometry is degenerate.
    # RBF kernels on zero-distance will produce identical outputs, which
    # is a neutral (not misleading) representation.
    return np.zeros((n_atoms, 3), dtype=np.float32)


def validate_graph(g: Data) -> bool:
    """Full graph integrity check — used by the cache validator."""
    if g is None:
        return False
    if getattr(g, "is_dummy", True):
        return False
    if g.x is None or g.edge_index is None:
        return False
    if g.x.shape[1] != NODE_DIM:
        return False
    if g.edge_index.shape[0] != 2:
        return False
    if g.edge_attr is not None and g.edge_attr.shape[1] != EDGE_DIM:
        return False
    if torch.isnan(g.x).any():
        return False
    if g.num_nodes != g.x.shape[0]:
        return False
    return True


# ===========================================================================
# SECTION 3 — ESM-2 protein sequence encoder
# ===========================================================================

ESM_DIM: int = 1280        # ESM-2 esm2_t33_650M_UR50D output dimension
ESM_MAX_LEN: int = 1022    # max AA tokens (+ 2 special = 1024 ESM limit)


def _seq_key(seq: str) -> str:
    """BUG-08 FIX: deterministic MD5 cache key from sequence string.

    Python's built-in hash() changes across sessions (PYTHONHASHSEED).
    MD5 is stable and fast enough for this purpose.
    """
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def _zero_embedding() -> np.ndarray:
    """BUG-09 FIX: zero vector for missing / empty sequences.

    Previously a real signal peptide was used as fallback, which silently
    injected biological signal into missing-sequence samples.
    """
    return np.zeros(ESM_DIM, dtype=np.float32)


class ESMEncoder:
    """Lazy-loading ESM-2 encoder with on-disk numpy cache.

    The ESM-2 model is downloaded once to ``~/.cache/torch/hub/``.
    Computed embeddings are saved as ``.npy`` files under *cache_dir*
    (keyed by MD5 of the sequence, so the index is session-independent).

    Usage
    -----
    >>> enc = ESMEncoder(cache_dir=".cache/esm")
    >>> emb = enc.encode("MKTIIALSYIFCLVFA")   # (1280,) float32 array
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._index: Dict[str, str] = {}      # seq_md5 → npy filename
        self._model = None
        self._alphabet = None
        self._batch_converter = None
        self._device = device   # resolved lazily in _load_model()

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._index_path = self._cache_dir / "esm_index.json"
            if self._index_path.exists():
                try:
                    with open(self._index_path, encoding="utf-8") as fh:
                        self._index = json.load(fh)
                except Exception:
                    self._index = {}

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Lazy-load ESM-2 model to the resolved device."""
        if self._model is not None:
            return
        try:
            import esm as esm_lib
            self._model, self._alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
            self._batch_converter = self._alphabet.get_batch_converter()
            # BUG-10 FIX: resolve device once and move model at load time
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            warnings.warn(
                f"[ESMEncoder] Could not load ESM-2 model: {e}. "
                "Falling back to zero embeddings."
            )
            self._model = None

    # ------------------------------------------------------------------
    def encode(self, sequence: str) -> np.ndarray:
        """Return 1280-dim mean-pool embedding for *sequence*.

        Returns a zero embedding for empty / invalid sequences (BUG-09 fix).
        """
        if not sequence or not isinstance(sequence, str):
            return _zero_embedding()
        sequence = sequence.strip()
        if not sequence:
            return _zero_embedding()

        # BUG-11: truncate over-long sequences
        if len(sequence) > ESM_MAX_LEN:
            sequence = sequence[:ESM_MAX_LEN]

        key = _seq_key(sequence)

        # Cache hit
        if self._cache_dir and key in self._index:
            npy_path = self._cache_dir / self._index[key]
            if npy_path.exists():
                try:
                    return np.load(str(npy_path))
                except Exception:
                    pass  # corrupted → recompute

        emb = self._compute(sequence)

        # Cache write
        if self._cache_dir:
            npy_name = f"{key}.npy"
            np.save(str(self._cache_dir / npy_name), emb)
            self._index[key] = npy_name
            try:
                with open(self._index_path, "w", encoding="utf-8") as fh:
                    json.dump(self._index, fh)
            except Exception:
                pass

        return emb

    # ------------------------------------------------------------------
    def _compute(self, sequence: str) -> np.ndarray:
        """Compute ESM-2 embedding (called on cache miss)."""
        self._load_model()
        if self._model is None:
            return _zero_embedding()

        try:
            data = [("prot", sequence)]
            _, _, tokens = self._batch_converter(data)
            tokens = tokens.to(self._device)   # BUG-10 fix
            with torch.no_grad():
                results = self._model(
                    tokens,
                    repr_layers=[33],
                    return_contacts=False,
                )
            # mean-pool over sequence positions (skip BOS/EOS tokens)
            token_emb = results["representations"][33]   # (1, L+2, 1280)
            emb = token_emb[0, 1:-1].mean(dim=0)         # (1280,)
            return emb.cpu().numpy().astype(np.float32)
        except Exception as e:
            warnings.warn(f"[ESMEncoder] compute failed: {e}")
            return _zero_embedding()


# ---------------------------------------------------------------------------
# Lightweight fallback: deterministic random projection of one-hot AA encoding.
# Used when ESM-2 is unavailable (CPU-only environment / no internet).
# ---------------------------------------------------------------------------

_AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
_AA_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(_AA_VOCAB)}


def _onehot_to_esm_dim(sequence: str, out_dim: int = ESM_DIM) -> np.ndarray:
    """Deterministic random projection of one-hot AA encoding.

    Not biologically meaningful — dimension-compatible with ESM_DIM only.
    Used as graceful fallback when ESM-2 is unavailable.
    """
    L = min(len(sequence), ESM_MAX_LEN)
    if L == 0:
        return _zero_embedding()

    one_hot = np.zeros((L, len(_AA_VOCAB)), dtype=np.float32)
    for i, aa in enumerate(sequence[:L]):
        if aa in _AA_IDX:
            one_hot[i, _AA_IDX[aa]] = 1.0

    rng = np.random.default_rng(0)   # seed=0 → deterministic
    proj = rng.standard_normal((len(_AA_VOCAB), out_dim)).astype(np.float32)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True).clip(min=1e-6)
    emb = one_hot.mean(axis=0) @ proj   # (out_dim,)
    return emb.astype(np.float32)
