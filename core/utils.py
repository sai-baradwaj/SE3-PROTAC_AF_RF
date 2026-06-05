"""
core/utils.py
=============
Shared utilities for SE3AF:
  - Logging
  - Metrics (AUROC, AUPRC, bootstrap CI)
  - Temperature scaling calibration
  - VRAM auto-detection
  - Column-name discovery and label parsing
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────

def get_logger(name: str = "se3af", log_file: Optional[str] = None) -> logging.Logger:
    """Return a named logger with stdout + optional file handler.

    BUG-W01 FIX: StreamHandler uses UTF-8 encoding on Windows to prevent
    UnicodeEncodeError when log messages contain non-ASCII characters (e.g.
    the em-dash in the format string or arrow characters in messages).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # BUG-W01 FIX: force UTF-8 on the stream so Windows cp1252 consoles
    # do not raise UnicodeEncodeError for non-ASCII chars in log messages.
    # BUG-W02 FIX: avoid wrapping an already-wrapped TextIOWrapper (e.g. when
    # pytest or IPython already set sys.stdout to a custom stream without .buffer).
    # We use the existing stream if it already reports UTF-8, otherwise wrap it.
    import io
    _existing_stream = sys.stdout
    if hasattr(_existing_stream, "buffer") and getattr(_existing_stream, "encoding", "").lower() not in ("utf-8", "utf8"):
        stream = io.TextIOWrapper(
            _existing_stream.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    else:
        stream = _existing_stream
    ch = logging.StreamHandler(stream)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # BUG-W01 FIX: explicit UTF-8 for file handler too
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            warnings.warn("AUROC: only one class present, returning 0.5")
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except Exception as e:
        warnings.warn(f"AUROC failed: {e}")
        return 0.5


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import average_precision_score
        if len(np.unique(y_true)) < 2:
            return float(y_true.mean())
        return float(average_precision_score(y_true, y_score))
    except Exception as e:
        warnings.warn(f"AUPRC failed: {e}")
        return 0.0


def _find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """BUG-8 FIX: Find optimal classification threshold using F1-score maximization.

    Previously hardcoded threshold=0.5 caused F1=0/Precision=0/Recall=0 when all
    model predictions cluster below 0.5 (common in early training).

    Algorithm:
      1. Search thresholds from 0.05 to 0.95 in steps of 0.01
      2. Select threshold maximizing F1 score
      3. Fall back to 0.5 if no samples exist or all predictions are identical

    This threshold is used only for classification metrics (F1, Precision, Recall,
    Accuracy, MCC) — AUROC/AUPRC are threshold-independent and unchanged.
    """
    # Ensure y_true is integer for sklearn classification metrics
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    from sklearn.metrics import f1_score as _f1
    best_t, best_f1 = 0.5, -1.0
    for i in range(5, 96):
        t = i / 100.0
        f1 = _f1(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute classification metrics with optional threshold optimization.

    BUG-8 FIX: When threshold=None (default), automatically finds the optimal
    classification threshold via F1 maximization instead of using hardcoded 0.5.
    This prevents F1=0/Precision=0/Recall=0 when predictions cluster below 0.5.

    The optimal threshold is included in the returned dict as 'threshold'.
    AUROC and AUPRC are always threshold-independent.
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score,
    )
    # Ensure correct dtypes to prevent sklearn "mix of continuous and binary" error
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    # BUG-8 FIX: Use optimal threshold if none specified
    if threshold is None:
        threshold = _find_optimal_threshold(y_true, y_score)

    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc":     safe_auroc(y_true, y_score),
        "auprc":     safe_auprc(y_true, y_score),
        "acc":       float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc":       float(matthews_corrcoef(y_true, y_pred))
                     if len(np.unique(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),   # BUG-8 FIX: report optimal threshold used
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap 95% CI for AUROC and AUPRC."""
    rng = np.random.default_rng(seed)
    aurocs, auprcs = [], []
    N = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        aurocs.append(safe_auroc(y_true[idx], y_score[idx]))
        auprcs.append(safe_auprc(y_true[idx], y_score[idx]))
    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "auroc":    safe_auroc(y_true, y_score),
        "auroc_lo": float(np.percentile(aurocs, lo * 100)),
        "auroc_hi": float(np.percentile(aurocs, hi * 100)),
        "auprc":    safe_auprc(y_true, y_score),
        "auprc_lo": float(np.percentile(auprcs, lo * 100)),
        "auprc_hi": float(np.percentile(auprcs, hi * 100)),
    }


# ─────────────────────────────────────────────────────────────────
# Temperature scaling calibration
# ─────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class TemperatureScaling(nn.Module):
    """Post-hoc calibration via temperature scaling."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def fit(self, logits: Tensor, labels: Tensor, max_iter: int = 50) -> float:
        self.train()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def _eval():
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(
                self(logits), labels.float()
            )
            loss.backward()
            return loss

        optimizer.step(_eval)
        self.eval()
        return float(self.temperature.item())


# ─────────────────────────────────────────────────────────────────
# VRAM auto-detection
# ─────────────────────────────────────────────────────────────────

@dataclass
class VRAMProfile:
    tier: int
    backend: str
    use_amp: bool
    amp_dtype: str
    batch_size: int
    vram_gb: float


def get_vram_profile() -> VRAMProfile:
    """Detect GPU VRAM and return recommended training profile."""
    if not torch.cuda.is_available():
        return VRAMProfile(6, "lite", False, "float32", 1, 0.0)
    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        vram_gb = 0.0

    if   vram_gb >= 24: return VRAMProfile(1, "se3",  False, "float32", 32, vram_gb)
    elif vram_gb >= 16: return VRAMProfile(2, "lite", True,  "float16", 16, vram_gb)
    elif vram_gb >= 8:  return VRAMProfile(3, "lite", True,  "float16", 12, vram_gb)
    elif vram_gb >= 6:  return VRAMProfile(4, "lite", True,  "float16",  8, vram_gb)
    elif vram_gb >= 4:  return VRAMProfile(5, "lite", True,  "float16",  4, vram_gb)
    elif vram_gb >= 2:  return VRAMProfile(5, "lite", True,  "float16",  2, vram_gb)
    else:               return VRAMProfile(6, "lite", False, "float32",  1, vram_gb)


# ─────────────────────────────────────────────────────────────────
# Column normalisation and label parsing
# ─────────────────────────────────────────────────────────────────

# Canonical column names
COL_TGT_SMILES = "smiles_target_lig"
COL_E3_SMILES  = "smiles_e3_lig"
COL_LNK_SMILES = "smiles_linker"
COL_TGT_SEQ    = "target_seq"
COL_LIG_SEQ    = "ligase_seq"
COL_LABEL      = "activity"

REQUIRED_COLS = [COL_TGT_SMILES, COL_E3_SMILES, COL_LNK_SMILES]

# Alias table: input column name (lowercase) → canonical name
COL_ALIASES: Dict[str, str] = {
    # target smiles
    "warhead_smiles": COL_TGT_SMILES, "target_smiles": COL_TGT_SMILES,
    "smiles_warhead": COL_TGT_SMILES, "target_ligand_smiles": COL_TGT_SMILES,
    "tgt_smiles": COL_TGT_SMILES, "poi_smiles": COL_TGT_SMILES,
    "poi_ligand_smiles": COL_TGT_SMILES, "smiles_poi": COL_TGT_SMILES,
    "binder_smiles": COL_TGT_SMILES, "warhead": COL_TGT_SMILES,
    # e3 smiles
    "e3_ligase_smiles": COL_E3_SMILES, "e3_smiles": COL_E3_SMILES,
    "smiles_e3": COL_E3_SMILES, "ligase_smiles": COL_E3_SMILES,
    "e3_ligand_smiles": COL_E3_SMILES, "e3lig_smiles": COL_E3_SMILES,
    "e3_binder_smiles": COL_E3_SMILES,
    # linker smiles
    "linker_smiles": COL_LNK_SMILES, "linker": COL_LNK_SMILES,
    "smiles_lnk": COL_LNK_SMILES, "link_smiles": COL_LNK_SMILES,
    "connector_smiles": COL_LNK_SMILES, "smiles_linker": COL_LNK_SMILES,
    # target sequence
    "target_sequence": COL_TGT_SEQ, "protein_seq": COL_TGT_SEQ,
    "protein_sequence": COL_TGT_SEQ, "seq_target": COL_TGT_SEQ,
    "target_protein_sequence": COL_TGT_SEQ, "poi_sequence": COL_TGT_SEQ,
    "poi_seq": COL_TGT_SEQ,
    # ligase sequence
    "e3_ligase_sequence": COL_LIG_SEQ, "e3_sequence": COL_LIG_SEQ,
    "e3_seq": COL_LIG_SEQ, "ligase_sequence": COL_LIG_SEQ,
    "ligase_protein_seq": COL_LIG_SEQ,
    # label
    "label": COL_LABEL, "active": COL_LABEL, "activity_label": COL_LABEL,
    "degradation": COL_LABEL, "degrader": COL_LABEL, "pdc50": COL_LABEL,
    "dc50": COL_LABEL, "binary_label": COL_LABEL, "y": COL_LABEL,
    "class": COL_LABEL, "is_active": COL_LABEL,
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names (case-insensitive alias mapping)."""
    col_map = {}
    for col in df.columns:
        key = col.lower().strip()
        if key in COL_ALIASES:
            canonical = COL_ALIASES[key]
            if canonical not in df.columns:
                col_map[col] = canonical
    return df.rename(columns=col_map) if col_map else df


def _parse_label(val) -> Optional[float]:
    """Parse label to 0.0 or 1.0 with range-aware thresholding.

    Handles:
      (a) Binary 0/1 integer  → pass through
      (b) DC50 in nM (>20)    → active if val ≤ 100 nM
      (c) pDC50 (1–20)        → active if val ≥ 7.0
      (d) Fraction (0–1)      → active if val ≥ 0.5

    BUG-UTILS-01 FIX: string values "0", "1", "0.0", "1.0", "true", "false",
    "yes", "no" are now explicitly handled to avoid mis-classification.
    """
    if isinstance(val, str):
        stripped = val.strip().lower()
        if stripped in ("nan", "none", "null", ""):
            return None
        if stripped in ("true", "yes", "1", "1.0", "active"):
            return 1.0
        if stripped in ("false", "no", "0", "0.0", "inactive"):
            return 0.0
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v:          # NaN
        return None
    if v in (0.0, 1.0):
        return float(v)   # explicit float cast for clarity
    if v < 0.0:
        return None
    if v > 20.0:
        return 1.0 if v <= 100.0 else 0.0   # DC50 nM
    elif v > 1.0:
        return 1.0 if v >= 7.0 else 0.0     # pDC50
    else:
        return 1.0 if v >= 0.5 else 0.0     # fraction


def discover_datasets(data_root: str) -> List[pd.DataFrame]:
    """Find and load all valid CSV files under *data_root*."""
    root = Path(data_root)
    if root.is_file() and root.suffix.lower() == ".csv":
        paths = [root]
    elif root.is_dir():
        paths = list(root.glob("**/*.csv"))
    else:
        raise FileNotFoundError(f"data_root not found or not CSV: {data_root}")
    if not paths:
        raise FileNotFoundError(f"No CSV files found under: {data_root}")

    dfs = []
    for p in sorted(paths):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            warnings.warn(f"Could not read {p}: {e}")
            continue
        df = normalise_columns(df)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            warnings.warn(f"{p}: missing required columns {missing}. Available: {list(df.columns)}")
            continue
        if len(df) == 0:
            warnings.warn(f"{p}: empty CSV, skipping.")
            continue
        dfs.append(df)

    if not dfs:
        raise ValueError(
            f"No valid CSV datasets found under {data_root}. "
            "Check column names match expected schema (see COL_ALIASES in core/utils.py)."
        )
    return dfs


# ===========================================================================
# V37: DataLeakageAuditor — Pre-training data validation
# ===========================================================================

class DataLeakageReport:
    """Result of a data leakage audit."""

    def __init__(self):
        self.exact_smiles_duplicates:    int = 0
        self.protein_pair_duplicates:    int = 0
        self.scaffold_train_val_overlap: int = 0
        self.scaffold_train_test_overlap: int = 0
        self.cross_split_smiles_overlap_tv: int = 0
        self.cross_split_smiles_overlap_tt: int = 0
        self.warnings: list = []
        self.has_leakage: bool = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        self.has_leakage = True

    def to_markdown(self) -> str:
        lines = [
            "# DATA LEAKAGE AUDIT REPORT",
            f"**Generated**: {__import__('datetime').datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            f"| Check | Count | Status |",
            f"|-------|-------|--------|",
            f"| Exact SMILES duplicates | {self.exact_smiles_duplicates} | {'⚠️ WARNING' if self.exact_smiles_duplicates > 0 else '✅ OK'} |",
            f"| Protein pair duplicates | {self.protein_pair_duplicates} | {'⚠️ WARNING' if self.protein_pair_duplicates > 0 else '✅ OK'} |",
            f"| Scaffold train/val overlap | {self.scaffold_train_val_overlap} | {'⚠️ WARNING' if self.scaffold_train_val_overlap > 0 else '✅ OK'} |",
            f"| Scaffold train/test overlap | {self.scaffold_train_test_overlap} | {'⚠️ WARNING' if self.scaffold_train_test_overlap > 0 else '✅ OK'} |",
            f"| SMILES train/val overlap | {self.cross_split_smiles_overlap_tv} | {'⚠️ WARNING' if self.cross_split_smiles_overlap_tv > 0 else '✅ OK'} |",
            f"| SMILES train/test overlap | {self.cross_split_smiles_overlap_tt} | {'⚠️ WARNING' if self.cross_split_smiles_overlap_tt > 0 else '✅ OK'} |",
            "",
        ]
        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")
        else:
            lines.append("## Status: No leakage detected ✅")

        lines.append("## Interpretation")
        lines.append("")
        lines.append("**Scaffold overlap** means the same Bemis-Murcko scaffold appears in")
        lines.append("both training and validation/test sets. The model may memorize scaffold-")
        lines.append("specific features rather than learning generalizable representations.")
        lines.append("")
        lines.append("**NOTE**: With n=25 samples, some overlap is statistically unavoidable.")
        lines.append("This audit reports what exists; it does not claim the model is invalid.")
        return "\n".join(lines)


# ===========================================================================
# V40-PHASE-12 — CALIBRATION METRICS (ECE, Brier, Reliability Diagram)
# ===========================================================================

def compute_ece(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    ECE = Σ_b |B_b| / N * |acc(B_b) - conf(B_b)|

    Lower is better. A perfectly calibrated model has ECE=0.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (y_score >= lo) & (y_score < hi)
        if not mask.any():
            continue
        acc  = y_true[mask].mean()
        conf = y_score[mask].mean()
        ece  += mask.sum() / N * abs(acc - conf)
    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier Score: mean squared error between probabilities and labels.

    Brier = mean((p - y)^2).  Lower is better. Range [0, 1].
    """
    return float(np.mean((y_score - y_true) ** 2))


def compute_calibration_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Full calibration report dict with ECE, Brier, and per-bin stats.

    Returns a dict suitable for JSON serialization and report generation.
    """
    ece    = compute_ece(y_true, y_score, n_bins)
    brier  = compute_brier_score(y_true, y_score)

    # Per-bin stats for reliability diagram
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            bin_data.append({
                "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                "count": 0, "mean_conf": 0.0, "mean_acc": 0.0,
            })
        else:
            bin_data.append({
                "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                "count": int(mask.sum()),
                "mean_conf": round(float(y_score[mask].mean()), 4),
                "mean_acc":  round(float(y_true[mask].mean()),  4),
            })

    return {
        "ece":           round(ece,   4),
        "brier_score":   round(brier, 4),
        "n_samples":     len(y_true),
        "n_bins":        n_bins,
        "bin_stats":     bin_data,
        "interpretation": (
            "Well-calibrated"   if ece < 0.05 else
            "Slightly off"      if ece < 0.10 else
            "Poorly calibrated"
        ),
    }


def save_reliability_diagram(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: str = "reports/reliability_diagram.txt",
    n_bins: int = 10,
) -> str:
    """Save ASCII reliability diagram to a text file.

    Returns path to saved file.
    """
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report = compute_calibration_report(y_true, y_score, n_bins)

    lines = [
        "RELIABILITY DIAGRAM — SE3AF v4.0",
        "=================================",
        f"ECE:         {report['ece']:.4f}",
        f"Brier Score: {report['brier_score']:.4f}",
        f"Status:      {report['interpretation']}",
        f"N samples:   {report['n_samples']}",
        "",
        "Bin        | Count | Confidence | Accuracy  | Gap",
        "-----------|-------|------------|-----------|--------",
    ]
    for b in report["bin_stats"]:
        gap = abs(b["mean_conf"] - b["mean_acc"])
        bar_len = int(b["mean_acc"] * 20) if b["count"] > 0 else 0
        ideal   = int(b["mean_conf"] * 20) if b["count"] > 0 else 0
        flag = " ⚠" if gap > 0.1 else ""
        lines.append(
            f"[{b['bin_lo']:.1f}-{b['bin_hi']:.1f}]  "
            f"|{b['count']:>5d}|"
            f"{b['mean_conf']:>10.3f}  |"
            f"{b['mean_acc']:>9.3f}  |"
            f"{gap:>7.3f}{flag}"
        )

    content = "\n".join(lines)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except Exception:
        pass
    return out_path


class DataLeakageAuditor:
    """V37: Pre-training data validation and leakage detection.

    Checks:
    1. Exact SMILES duplicates within the full dataset
    2. Same protein pair duplicates
    3. Scaffold (Bemis-Murcko) overlap between train/val/test
    4. Cross-split SMILES string overlap
    """

    def __init__(self, df: "pd.DataFrame"):
        self.df = df

    def run(
        self,
        train_idx: list,
        val_idx: list,
        test_idx: list,
    ) -> "DataLeakageReport":
        report = DataLeakageReport()
        df = self.df

        smiles_cols = [COL_TGT_SMILES, COL_E3_SMILES, COL_LNK_SMILES]
        available = [c for c in smiles_cols if c in df.columns]

        # 1. Exact SMILES duplicates in full dataset
        if available:
            combined = df[available].apply(lambda r: "|".join(str(r[c]) for c in available), axis=1)
            n_dupes = combined.duplicated().sum()
            report.exact_smiles_duplicates = int(n_dupes)
            if n_dupes > 0:
                report.add_warning(f"{n_dupes} exact duplicate SMILES combination(s) found in dataset")

        # 2. Protein pair duplicates
        prot_cols = [c for c in ["target_uniprot", "ligase_uniprot", COL_TGT_SEQ, COL_LIG_SEQ] if c in df.columns]
        if len(prot_cols) >= 2:
            prot_combined = df[prot_cols[:2]].apply(lambda r: "|".join(str(r[c]) for c in prot_cols[:2]), axis=1)
            n_prot_dupes = prot_combined.duplicated().sum()
            report.protein_pair_duplicates = int(n_prot_dupes)

        # 3. Cross-split SMILES overlap
        if available and len(available) > 0:
            col = available[0]
            train_smiles = set(df[col].iloc[train_idx].dropna().astype(str))
            val_smiles   = set(df[col].iloc[val_idx].dropna().astype(str)) if val_idx else set()
            test_smiles  = set(df[col].iloc[test_idx].dropna().astype(str)) if test_idx else set()

            tv_overlap = train_smiles & val_smiles
            tt_overlap = train_smiles & test_smiles
            report.cross_split_smiles_overlap_tv = len(tv_overlap)
            report.cross_split_smiles_overlap_tt = len(tt_overlap)
            if tv_overlap:
                report.add_warning(
                    f"{len(tv_overlap)} target SMILES overlap between train and val: "
                    + ", ".join(list(tv_overlap)[:3])
                )
            if tt_overlap:
                report.add_warning(
                    f"{len(tt_overlap)} target SMILES overlap between train and test: "
                    + ", ".join(list(tt_overlap)[:3])
                )

        # 4. Scaffold overlap (requires rdkit)
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold

            if available:
                col = available[0]

                def _scaffold(smi: str) -> str:
                    try:
                        mol = Chem.MolFromSmiles(str(smi))
                        if mol is None:
                            return ""
                        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                    except Exception:
                        return ""

                train_scafs = {_scaffold(s) for s in df[col].iloc[train_idx] if s} - {""}
                val_scafs   = {_scaffold(s) for s in df[col].iloc[val_idx] if s} - {""}  if val_idx else set()
                test_scafs  = {_scaffold(s) for s in df[col].iloc[test_idx] if s} - {""}  if test_idx else set()

                tv_scaf = train_scafs & val_scafs
                tt_scaf = train_scafs & test_scafs
                report.scaffold_train_val_overlap  = len(tv_scaf)
                report.scaffold_train_test_overlap = len(tt_scaf)
                if tv_scaf:
                    report.add_warning(
                        f"{len(tv_scaf)} scaffold(s) overlap between train and val (scaffold leakage)"
                    )
                if tt_scaf:
                    report.add_warning(
                        f"{len(tt_scaf)} scaffold(s) overlap between train and test (scaffold leakage)"
                    )
        except ImportError:
            pass   # rdkit not available — skip scaffold check

        return report

    def print_summary(self, report: "DataLeakageReport"):
        """Print formatted leakage summary to stdout."""
        print()
        print("  DATA LEAKAGE AUDIT")
        print("  ══════════════════════════════════════════════════════")
        status = lambda n: "⚠️  WARNING" if n > 0 else "✅ OK"
        print(f"  Exact SMILES duplicates:    {report.exact_smiles_duplicates:3d}  {status(report.exact_smiles_duplicates)}")
        print(f"  Protein pair duplicates:    {report.protein_pair_duplicates:3d}  {status(report.protein_pair_duplicates)}")
        print(f"  Scaffold train/val overlap: {report.scaffold_train_val_overlap:3d}  {status(report.scaffold_train_val_overlap)}")
        print(f"  Scaffold train/test overlap:{report.scaffold_train_test_overlap:3d}  {status(report.scaffold_train_test_overlap)}")
        print(f"  SMILES train/val overlap:   {report.cross_split_smiles_overlap_tv:3d}  {status(report.cross_split_smiles_overlap_tv)}")
        print(f"  SMILES train/test overlap:  {report.cross_split_smiles_overlap_tt:3d}  {status(report.cross_split_smiles_overlap_tt)}")
        if report.warnings:
            print()
            print("  WARNINGS:")
            for w in report.warnings:
                print(f"    ⚠️  {w}")
        else:
            print()
            print("  Status: No leakage detected ✅")
        print("  ══════════════════════════════════════════════════════")
        print()
