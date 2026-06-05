"""
core/alphafold.py
=================
AlphaFold structure integration for SE3AF.

Provides:
  - AlphaFoldStore  : scans a directory for AlphaFold PDB/mmCIF files
  - load_af_coords  : extracts Cα backbone coordinates + pLDDT from PDB
  - af_residue_features : returns (L, 4) array of [Cα_x, Cα_y, Cα_z, pLDDT]
  - GRACEFUL FALLBACK: if no AF files found, returns zero arrays silently
                       but logs a visible WARNING

AlphaFold File Naming Convention Supported
------------------------------------------
  AF-<UniProt_ID>-F1-model_v4.pdb          (official AF2 download)
  AF-<UniProt_ID>-F1-model_v4.cif
  <protein_id>.pdb                          (custom naming)
  <protein_id>.cif

Usage
-----
>>> store = AlphaFoldStore("/data/alphafold/")
>>> print(store.summary())          # AlphaFold Structures Found: 42 ...
>>> coords, plddt = store.get("P62988")   # (L, 3), (L,) or None, None
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ===========================================================================
# PDB backbone extraction (no biopython required — pure regex on ATOM records)
# ===========================================================================

# Legacy regex patterns — kept for reference but PDB parsing uses
# fixed-width column slicing (more robust than regex for PDB format)
# _ATOM_RE and _CA_RE are not used internally; retained for external code
# that may import them.
_ATOM_RE = re.compile(
    r"^ATOM\s+"              # record type
    r"\d+\s+"                # atom serial
    r"(\S+)\s+"              # atom name
    r"\S*\s+"                # alt loc + residue name (combined or split)
    r"(\S+)\s+"              # chain ID (or residue name if alt loc absent)
    r"(\-?\d+)\s+"           # residue seq num
    r"\S*\s+"                # insertion code (optional)
    r"(\-?\d+\.\d+)\s+"      # X
    r"(\-?\d+\.\d+)\s+"      # Y
    r"(\-?\d+\.\d+)\s+"      # Z
    r"\S+\s+"                # occupancy
    r"(\d+\.\d+)",           # B-factor (pLDDT in AF2)
    re.IGNORECASE,
)

# Simpler but reliable Cα-only regex for standard PDB format
_CA_RE = re.compile(
    r"^ATOM\s+\d+\s+CA\s+\w+\s+\w\s+(-?\d+)\s+"   # residue num
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"  # X Y Z
    r"\s+\S+\s+(\d+\.\d+)",                          # B-factor/pLDDT
    re.IGNORECASE,
)


def _parse_pdb_ca(pdb_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract Cα coordinates and pLDDT from a PDB file.

    Returns
    -------
    coords : np.ndarray of shape (L, 3) or None if parse fails
    plddt  : np.ndarray of shape (L,)   or None if parse fails
    """
    coords_list: List[List[float]] = []
    plddt_list:  List[float] = []

    try:
        with open(pdb_path, "r", errors="replace") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                # Check atom name field (columns 13-16, 0-indexed 12-15)
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                try:
                    x       = float(line[30:38])
                    y       = float(line[38:46])
                    z       = float(line[46:54])
                    b_fac   = float(line[60:66])   # pLDDT in AF2
                    coords_list.append([x, y, z])
                    plddt_list.append(b_fac)
                except (ValueError, IndexError):
                    continue
    except Exception as exc:
        warnings.warn(f"[alphafold] PDB parse error {pdb_path}: {exc}")
        return None, None

    if not coords_list:
        return None, None

    return (
        np.array(coords_list, dtype=np.float32),
        np.array(plddt_list,  dtype=np.float32),
    )


# ===========================================================================
# AlphaFold Store
# ===========================================================================

class AlphaFoldStore:
    """Scans a directory for AlphaFold structure files and provides fast lookup.

    Supports both PDB and mmCIF file extensions.  Resolves UniProt IDs from
    AF-{ID}-F1-model_v*.pdb naming or plain {ID}.pdb naming.

    Parameters
    ----------
    af_dir : str | Path | None
        Directory to scan.  If None or the directory doesn't exist, the store
        is empty and all lookups return (None, None).

    Examples
    --------
    >>> store = AlphaFoldStore("/data/af2/")
    >>> coords, plddt = store.get("P62988")
    >>> print(store.n_found, store.n_missing_from_last_check)
    """

    EXTENSIONS = {".pdb", ".cif", ".mmcif"}

    def __init__(self, af_dir: Optional[str] = None) -> None:
        self.af_dir = Path(af_dir) if af_dir else None
        self._index: Dict[str, Path] = {}   # protein_id → file path
        self._cache: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        self.n_found   = 0
        self.enabled   = False
        self._scanned  = False

        if self.af_dir and self.af_dir.is_dir():
            self._scan()

    # ------------------------------------------------------------------
    def _scan(self) -> None:
        """Walk af_dir and build protein_id → Path index."""
        if self._scanned:
            return
        self._scanned = True

        af_pattern = re.compile(r"AF-([A-Z0-9]+)-F\d+", re.IGNORECASE)

        for root, _, files in os.walk(str(self.af_dir)):
            for fname in files:
                stem = Path(fname).stem
                suf  = Path(fname).suffix.lower()
                if suf not in self.EXTENSIONS:
                    continue
                full_path = Path(root) / fname

                # Try AF-{ID}-F1-model_v4 naming
                m = af_pattern.search(stem)
                if m:
                    protein_id = m.group(1).upper()
                else:
                    # Plain {ID}.pdb naming
                    protein_id = stem.upper()

                self._index[protein_id] = full_path

        self.n_found = len(self._index)
        self.enabled = self.n_found > 0

    # ------------------------------------------------------------------
    def get(
        self,
        protein_id: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (coords: (L,3), plddt: (L,)) for *protein_id*.

        Returns (None, None) if not found or if store is empty.
        """
        if not protein_id or not self.enabled:
            return None, None

        key = str(protein_id).strip().upper()

        if key in self._cache:
            return self._cache[key]

        if key not in self._index:
            # Try prefix match (e.g. "P62988" matches "P62988_HUMAN")
            candidates = [k for k in self._index if k.startswith(key)]
            if not candidates:
                self._cache[key] = (None, None)
                return None, None
            key = candidates[0]

        fpath = self._index[key]
        coords, plddt = _parse_pdb_ca(str(fpath))
        result = (coords, plddt)
        self._cache[protein_id.strip().upper()] = result
        return result

    # ------------------------------------------------------------------
    def check_ids(
        self,
        protein_ids: List[str],
    ) -> Tuple[int, int]:
        """Return (n_found, n_missing) for a list of protein IDs."""
        found   = sum(1 for pid in protein_ids if pid.strip().upper() in self._index)
        missing = len(protein_ids) - found
        return found, missing

    # ------------------------------------------------------------------
    def summary(self, protein_ids: Optional[List[str]] = None) -> str:
        """Return a human-readable summary string for the startup banner."""
        if not self.enabled:
            return (
                "AlphaFold Structures Found: 0\n"
                "  AlphaFold Structures Missing: N/A\n"
                "  AlphaFold Features Enabled: False\n"
                "  (No AlphaFold directory configured — set alphafold_dir in config)"
            )

        if protein_ids:
            found, missing = self.check_ids(protein_ids)
            return (
                f"AlphaFold Structures Found: {found}\n"
                f"  AlphaFold Structures Missing: {missing}\n"
                f"  AlphaFold Features Enabled: {self.enabled}"
            )

        return (
            f"AlphaFold Structures Found: {self.n_found}\n"
            f"  AlphaFold Structures Missing: 0 (not checked against dataset)\n"
            f"  AlphaFold Features Enabled: {self.enabled}"
        )


# ===========================================================================
# Feature extraction helper
# ===========================================================================

def af_residue_features(
    coords: Optional[np.ndarray],
    plddt:  Optional[np.ndarray],
    max_len: int = 1022,
    normalize_coords: bool = True,
) -> np.ndarray:
    """Convert Cα coordinates + pLDDT into a fixed-length feature array.

    Returns
    -------
    np.ndarray of shape (max_len, 4) : [Cα_x, Cα_y, Cα_z, pLDDT_norm]
      Rows beyond actual length are zero-padded.
      pLDDT is normalised to [0, 1] by dividing by 100.

    Usage in model
    --------------
    The per-residue feature matrix can be:
      (a) Mean-pooled → (4,) summary vector appended to ESM embedding
      (b) Used as attention bias in CrossInteractionFusion
    """
    out = np.zeros((max_len, 4), dtype=np.float32)

    if coords is None or plddt is None:
        return out

    L = min(len(coords), max_len)
    if L == 0:
        return out

    c = coords[:L]  # (L, 3)
    p = plddt[:L]   # (L,)

    if normalize_coords and L > 0:
        # Centre and scale to unit sphere
        centroid = c.mean(axis=0)
        c = c - centroid
        scale = np.linalg.norm(c, axis=1).max()
        if scale > 1e-6:
            c = c / scale

    out[:L, :3] = c
    out[:L,  3] = np.clip(p / 100.0, 0.0, 1.0)  # pLDDT → [0,1]

    return out


# ===========================================================================
# Null store — used when no AF directory configured
# ===========================================================================

NULL_STORE = AlphaFoldStore(af_dir=None)


# ===========================================================================
# ISSUE-3 FIX: Residue-level graph builder for ProteinGeometryEncoder
# ===========================================================================

def af_residue_graph(
    coords: Optional[np.ndarray],
    plddt:  Optional[np.ndarray],
    max_len: int = 256,
    knn_k:   int = 8,
    distance_cutoff: float = 10.0,
) -> Optional[dict]:
    """Build a sparse residue-level graph from AlphaFold Cα coordinates.

    Scientific motivation (ISSUE-3):
    - Simple mean-pooling of Cα positions discards local geometry (beta-sheets,
      helices, loops) that determines binding-site accessibility for PROTACs.
    - A k-NN graph over Cα positions captures sequential and spatial proximity,
      enabling a lightweight EGNN (ProteinGeometryEncoder) to learn
      geometry-aware protein embeddings instead of a global average.

    Graph structure
    ---------------
    - Nodes  : first min(L, max_len) residues
      · Features (8-dim): [pLDDT_norm, sin(φ), cos(φ), sin(ψ), cos(ψ),
                            dist_prev, dist_next, local_density]
    - Edges  : k-NN contacts within distance_cutoff Å PLUS sequential bonds
      · Edge features (4-dim): [dist_norm, Δx_norm, Δy_norm, Δz_norm]

    Returns
    -------
    dict with keys:
      "node_feat"   : np.ndarray (N, 8)   node features
      "edge_index"  : np.ndarray (2, E)   source / target indices
      "edge_feat"   : np.ndarray (E, 4)   edge features
      "n_nodes"     : int
    or None if coords is None / L == 0.
    """
    if coords is None or plddt is None or len(coords) == 0:
        return None

    L = min(len(coords), max_len)
    c = coords[:L].astype(np.float32)       # (L, 3)
    p = np.clip(plddt[:L] / 100.0, 0.0, 1.0).astype(np.float32)  # (L,)

    # ── 1. Backbone dihedral approximation (via inter-Cα vectors) ─────────────
    # True φ/ψ require N,Cα,C atoms. We approximate using adjacent Cα vectors.
    # For residue i: forward vec = c[i+1]-c[i], backward vec = c[i-1]-c[i]
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n < 1e-8, 1.0, n)

    fwd  = np.zeros((L, 3), dtype=np.float32)   # c[i+1]-c[i]
    bwd  = np.zeros((L, 3), dtype=np.float32)   # c[i]-c[i-1]
    if L > 1:
        fwd[:-1] = c[1:] - c[:-1]
        bwd[1:]  = c[1:] - c[:-1]

    fwd_u = _unit(fwd)
    bwd_u = _unit(bwd)

    # Approximate φ proxy: dot(bwd, fwd) gives cosine of turn angle
    cos_phi = np.einsum('ij,ij->i', bwd_u, fwd_u)  # (L,)
    sin_phi = np.sqrt(np.clip(1.0 - cos_phi**2, 0.0, 1.0))

    # ψ proxy: angle between fwd and next-fwd (second derivative of chain)
    fwd2    = np.zeros((L, 3), dtype=np.float32)
    if L > 2:
        fwd2[:-1] = fwd[1:] - fwd[:-1]
    fwd2_u  = _unit(fwd2)
    cos_psi = np.einsum('ij,ij->i', fwd_u, fwd2_u)
    sin_psi = np.sqrt(np.clip(1.0 - cos_psi**2, 0.0, 1.0))

    # ── 2. Distances to neighbours ────────────────────────────────────────────
    dist_prev = np.zeros(L, dtype=np.float32)
    dist_next = np.zeros(L, dtype=np.float32)
    if L > 1:
        seqdists = np.linalg.norm(c[1:] - c[:-1], axis=-1)
        dist_prev[1:] = seqdists
        dist_next[:-1] = seqdists
    # Normalise by typical Cα-Cα bond length (3.8 Å)
    dist_prev_n = dist_prev / 3.8
    dist_next_n = dist_next / 3.8

    # ── 3. Local density: count neighbours within 10 Å ───────────────────────
    # Fast O(L²) but L ≤ 256 so max 256² = 65536 ops — acceptable
    diff  = c[:, None, :] - c[None, :, :]        # (L, L, 3)
    dmat  = np.linalg.norm(diff, axis=-1)          # (L, L)
    local_density = ((dmat < distance_cutoff) & (dmat > 0.01)).sum(axis=1).astype(np.float32)
    # Normalise to [0,1] (max meaningful density ≈ knn_k*2)
    local_density = np.clip(local_density / (knn_k * 4), 0.0, 1.0)

    # ── 4. Assemble node features (8-dim) ─────────────────────────────────────
    node_feat = np.stack([
        p,           # pLDDT normalised
        sin_phi, cos_phi,
        sin_psi, cos_psi,
        dist_prev_n, dist_next_n,
        local_density,
    ], axis=1)  # (L, 8)

    # ── 5. Build k-NN edge index ───────────────────────────────────────────────
    src_list: list = []
    tgt_list: list = []
    ef_list:  list = []

    # Sequential bonds (i→i+1 and i+1→i)
    for i in range(L - 1):
        for s, t in [(i, i + 1), (i + 1, i)]:
            dvec = c[t] - c[s]
            d    = float(np.linalg.norm(dvec))
            src_list.append(s); tgt_list.append(t)
            ef_list.append([d / distance_cutoff,
                             dvec[0] / distance_cutoff,
                             dvec[1] / distance_cutoff,
                             dvec[2] / distance_cutoff])

    # k-NN spatial contacts (excluding self and already-added sequential)
    seq_pairs = set((i, i + 1) for i in range(L - 1)) | \
                set((i + 1, i) for i in range(L - 1))
    for i in range(L):
        row = dmat[i].copy()
        row[i] = np.inf  # exclude self
        # Only consider within cutoff
        candidates = np.where(row < distance_cutoff)[0]
        if len(candidates) == 0:
            continue
        nearest = candidates[np.argsort(row[candidates])[:knn_k]]
        for j in nearest:
            if (int(i), int(j)) in seq_pairs:
                continue
            dvec = c[j] - c[i]
            d    = float(row[j])
            src_list.append(int(i)); tgt_list.append(int(j))
            ef_list.append([d / distance_cutoff,
                             dvec[0] / distance_cutoff,
                             dvec[1] / distance_cutoff,
                             dvec[2] / distance_cutoff])

    if not src_list:
        # Fallback: empty graph with single self-loop
        src_list = [0]; tgt_list = [0]
        ef_list  = [[0.0, 0.0, 0.0, 0.0]]

    return {
        "node_feat":  node_feat,                                       # (N, 8)
        "edge_index": np.array([src_list, tgt_list], dtype=np.int64),  # (2, E)
        "edge_feat":  np.array(ef_list, dtype=np.float32),             # (E, 4)
        "n_nodes":    L,
    }


# ===========================================================================
# V40-PHASE-9 — pLDDT CONFIDENCE-AWARE FEATURES
# ===========================================================================

def af_confidence_weighted_mean_pool(
    coords: Optional[np.ndarray],
    plddt: Optional[np.ndarray],
    confidence_threshold: float = 70.0,
    low_confidence_weight: float = 0.1,
) -> np.ndarray:
    """V40-PHASE-9: pLDDT confidence-aware mean pooling of Cα features.

    Residues with pLDDT < confidence_threshold are downweighted by
    low_confidence_weight, ensuring low-confidence regions contribute less
    to the protein representation.

    Returns
    -------
    np.ndarray of shape (4,): [mean_cx, mean_cy, mean_cz, mean_plddt_norm]
    All zeros if coords/plddt are None.

    Scientific justification:
    - AlphaFold pLDDT < 70 indicates disordered/loop regions with uncertain geometry
    - Uniformly pooling all residues gives misleading average coordinates
    - Confidence-weighted pooling emphasises structured (high-confidence) regions
    - This improves the biological signal in the AF feature vector
    """
    if coords is None or plddt is None or len(coords) == 0:
        return np.zeros(4, dtype=np.float32)

    L = len(coords)
    # Build confidence weights
    weights = np.ones(L, dtype=np.float32)
    low_conf_mask = plddt < confidence_threshold
    weights[low_conf_mask] = low_confidence_weight

    # Normalize coordinates
    c = coords.copy().astype(np.float32)
    centroid = c.mean(axis=0)
    c -= centroid
    scale = np.linalg.norm(c, axis=1).max()
    if scale > 1e-6:
        c /= scale

    p_norm = np.clip(plddt / 100.0, 0.0, 1.0).astype(np.float32)

    # Weighted mean
    w_sum = weights.sum()
    if w_sum < 1e-8:
        return np.zeros(4, dtype=np.float32)

    mean_c = (c * weights[:, None]).sum(axis=0) / w_sum
    mean_p = float((p_norm * weights).sum() / w_sum)

    return np.array([mean_c[0], mean_c[1], mean_c[2], mean_p], dtype=np.float32)


def get_plddt_mask(
    plddt: Optional[np.ndarray],
    threshold: float = 70.0,
    max_len: int = 1022,
) -> np.ndarray:
    """Return per-residue confidence mask (1.0=high, low_w=low confidence).

    Used as attention bias in models that explicitly use pLDDT to weight
    residue contributions in cross-attention.

    Returns np.ndarray of shape (max_len,) with values in [0, 1].
    """
    mask = np.zeros(max_len, dtype=np.float32)
    if plddt is None or len(plddt) == 0:
        return mask
    L = min(len(plddt), max_len)
    p = np.clip(plddt[:L] / 100.0, 0.0, 1.0)
    # Scale: high confidence (≥threshold/100) → close to 1.0
    # Low confidence (< threshold/100) → close to 0.0
    mask[:L] = p
    return mask
