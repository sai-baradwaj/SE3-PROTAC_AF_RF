"""
GLOBAL_CONFIG.py  —  SE3AF v4.1  SINGLE SOURCE OF TRUTH
=========================================================
All runtime configuration lives here.  Every other module imports from
this file.  config.py is a thin backward-compatibility shim.

BACKEND
-------
"se3"  → SE3GraphTransformer  (SE(3)-equivariant, coordinate-based MP)
"lite" → Lite3DEncoder         (distance-bias 3D attention, faster)

Both backends REQUIRE 3D coordinates (use_coords=True is forced globally).
3D mode is not optional in v4.1 — the system is designed around geometry.

TRAINING MODE
-------------
"fresh"    → start from epoch 0, discard old checkpoints
"continue" → load best_model.pt, resume from saved epoch
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ===========================================================================
# VERSION
# ===========================================================================
SE3AF_VERSION = "3.9.0"
CONFIG_SOURCE  = __file__

# ===========================================================================
# SECTION 1 — BACKEND & COMPONENT SWITCHES
# ===========================================================================

# Backend selection — BOTH use 3D coordinates in v4.1
# "se3"  : SE(3)-equivariant message passing with learnable RBF distances
# "lite" : distance-bias multi-head attention (faster, same 3D awareness)
BACKEND: str = "se3"

USE_SE3:  bool = (BACKEND == "se3")
USE_LITE: bool = (BACKEND == "lite")

# Feature component toggles
USE_ESM:       bool = True   # ESM-2 protein language model embeddings
USE_RF:        bool = True   # Random Forest stacker ensemble
USE_ALPHAFOLD: bool = True   # AlphaFold structural Cα features
USE_GOSS:      bool = True   # GOSS pair-importance weighting
USE_EMA:       bool = True   # Exponential Moving Average of weights
USE_SWA:       bool = False  # SWA disabled — conflicts with EMA

# 3D pipeline — ALWAYS True in v4.1
USE_COORDS: bool = True      # 3D conformer coordinates (ETKDGv3 + MMFF)

# Training mode
TRAINING_MODE: str = "fresh"   # "fresh" | "continue"

# SWA (Stochastic Weight Averaging)
SWA_START_FRACTION: float = 0.75  # fraction of epochs to start SWA
SWA_LR: float = 5e-5              # SWA learning rate

# ===========================================================================
# SECTION 2 — TRAINING HYPERPARAMETERS
# ===========================================================================

EPOCHS:        int   = 100     # More epochs with better regularization
BATCH_SIZE:    int   = 8        # small dataset → small batches
LEARNING_RATE: float = 2e-4    # Slightly lower LR for stability
WEIGHT_DECAY:  float = 5e-4    # Increased weight decay (was 1e-4) to fight overfitting
GRAD_CLIP:     float = 1.0
WARMUP_EPOCHS: int   = 5
SEED:          int   = 42

# ─── Advanced regularisation ──────────────────────────────────────────────
DROPOUT:             float = 0.40    # Increased from 0.25 to fight severe overfitting
LABEL_SMOOTHING:     float = 0.12    # Increased label smoothing
LABEL_SMOOTHING_END: float = 0.04
FEATURE_NOISE_STD:   float = 0.05   # Increased Gaussian noise on node features
MIXUP_ALPHA:         float = 0.4    # Stronger mixup
STOCHASTIC_DEPTH_P:  float = 0.20   # Increased DropPath
COORD_JITTER_STD:    float = 0.10   # Increased Å jitter on 3D coords during training

# ─── EMA ─────────────────────────────────────────────────────────────────
EMA_DECAY:          float = 0.999
USE_AMP:            bool  = True    # Mixed precision ENABLED for RTX 4050 speed/precision
AMP_DTYPE:          str   = "bfloat16"

# ─── Gradient accumulation ───────────────────────────────────────────────
GRAD_ACCUM_STEPS: int = 4    # effective batch = BATCH_SIZE × GRAD_ACCUM_STEPS

# ===========================================================================
# SECTION 3 — LOSS FUNCTION
# ===========================================================================

FOCAL_GAMMA: float = 2.0
FOCAL_ALPHA: float = 0.5     # balanced for ~64% positive PROTAC datasets

STABILITY_LOSS_WEIGHT:   float = 0.25
INTERACTION_LOSS_WEIGHT: float = 0.25

# SSL / contrastive — reserved for future use
SSL_LOSS_WEIGHT:         float = 0.0
CONTRASTIVE_LOSS_WEIGHT: float = 0.0

# ===========================================================================
# SECTION 4 — MODEL ARCHITECTURE
# ===========================================================================

FUSION_DIM:       int = 128   # Reduced from 192 to fight overfitting (divisible by num_heads=4)
ESM_DIM:          int = 1280  # ESM-2 fixed output dimension
GRAPH_HIDDEN:     int = 128   # Reduced from 192 to fight overfitting
NUM_GRAPH_LAYERS: int = 4     # Reduced from 5 layers to reduce capacity
NUM_HEADS:        int = 4     # Reduced from 6 to match new FUSION_DIM (128/4=32)
NUM_RBF:          int = 16    # Slightly reduced
CUTOFF_ANGSTROM:  float = 8.0 # distance cutoff for RBF (Å)

GOSS_TOP_K: int = 6

# pLDDT confidence attention
USE_PLDDT_ATTENTION:       bool  = True
PLDDT_CONFIDENCE_THRESHOLD: float = 70.0
PLDDT_LOW_CONFIDENCE_WEIGHT: float = 0.1

# Gradient checkpointing (saves GPU memory, ~20% slower)
USE_GRADIENT_CHECKPOINT: bool = False

# ===========================================================================
# SECTION 5 — DATA SPLIT & AUGMENTATION
# ===========================================================================

VAL_FRACTION:  float = 0.15
TEST_FRACTION: float = 0.10
# ISSUE-10 FIX: Use 2 workers on Linux/macOS; 0 on Windows (avoids pickling errors).
# DataLoader with num_workers > 0 on Windows requires __main__ guard which
# many training scripts don't have → silent hangs. Safe default: 0 on Win.
NUM_WORKERS:   int   = 0 if sys.platform.startswith("win") else 2
PIN_MEMORY:    bool  = False

# SMILES augmentation via random valid SMILES enumeration
USE_SMILES_AUGMENT: bool  = True
SMILES_AUGMENT_P:   float = 0.5

# 3D coordinate jitter augmentation (applied during training only)
USE_COORD_JITTER: bool = True

# Leakage thresholds — training aborts if exceeded
# v4.0: Reduced to 0.05 (5%) — scaffold split produces 0% overlap
MAX_SCAFFOLD_OVERLAP_FRACTION: float = 0.05
MAX_SMILES_OVERLAP_FRACTION:   float = 0.05

# ===========================================================================
# SECTION 6 — CHECKPOINTING & EARLY STOPPING
# ===========================================================================

CHECKPOINT_DIR:          str = "checkpoints"
SAVE_EVERY_N_EPOCHS:     int = 5
KEEP_LAST_N:             int = 3
EARLY_STOPPING_PATIENCE: int = 25  # More patience for smaller model
OVERFIT_WINDOW:          int = 3

CHECKPOINT_METRIC_WEIGHTS: Dict[str, float] = {
    "auroc": 0.35,
    "auprc": 0.25,
    "mcc":   0.20,
    "f1":    0.15,
    "acc":   0.05,
}

CHECKPOINT_ARCH_KEYS: List[str] = [
    "backend", "fusion_dim", "graph_hidden", "num_graph_layers", "num_heads",
    "dropout", "af_extra_dim", "use_alphafold", "use_rf",
    "dataset_hash", "config_hash",
]

# ===========================================================================
# SECTION 7 — RANDOM FOREST STACKER
# ===========================================================================

RF_TREES:            int = 300
RF_DEPTH:            int = 12
RF_FP_BITS:          int = 2048   # larger Morgan FP for better coverage
RF_MIN_SAMPLES_LEAF: int = 2

# ===========================================================================
# SECTION 8 — ALPHAFOLD
# ===========================================================================

ALPHAFOLD_DIR: str = "data/alphafold"
AF_EXTRA_DIM:  int =  16  # mean-pooled [Cα_x, Cα_y, Cα_z, pLDDT_norm]

# ===========================================================================
# SECTION 9 — CALIBRATION
# ===========================================================================

CALIBRATION_ENABLED:  bool  = True
TEMPERATURE_INIT:     float = 1.0
CALIBRATION_MAX_ITER: int   = 50

# ===========================================================================
# SECTION 10 — HARDWARE
# ===========================================================================

GPU_MEMORY_LIMIT_GB: float = 8.0

# ===========================================================================
# SECTION 11 — PATHS
# ===========================================================================

DATA_DIR:   str = "data/split_scaffold"
DATA_FILE:  str = "data/split_scaffold/train.csv"
CACHE_DIR:  str = ".cache"
LOG_DIR:    str = "logs"
REPORTS_DIR: str = "reports"

# ===========================================================================
# COMPUTED PROPERTIES
# ===========================================================================

def get_af_extra_dim() -> int:
    """Return actual af_extra_dim based on USE_ALPHAFOLD and PDB presence."""
    if not USE_ALPHAFOLD:
        return 0
    af_path = Path(ALPHAFOLD_DIR)
    if not af_path.exists() or not any(af_path.glob("*.pdb")):
        return 0
    return AF_EXTRA_DIM


def get_config_dict() -> dict:
    """Return all configuration as a flat dict (for fingerprinting)."""
    return {
        "version":           SE3AF_VERSION,
        "backend":           BACKEND,
        "use_coords":        USE_COORDS,
        "use_esm":           USE_ESM,
        "use_rf":            USE_RF,
        "use_alphafold":     USE_ALPHAFOLD,
        "use_goss":          USE_GOSS,
        "use_ema":           USE_EMA,
        "training_mode":     TRAINING_MODE,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "learning_rate":     LEARNING_RATE,
        "weight_decay":      WEIGHT_DECAY,
        "grad_accum_steps":  GRAD_ACCUM_STEPS,
        "focal_gamma":       FOCAL_GAMMA,
        "focal_alpha":       FOCAL_ALPHA,
        "fusion_dim":        FUSION_DIM,
        "graph_hidden":      GRAPH_HIDDEN,
        "num_graph_layers":  NUM_GRAPH_LAYERS,
        "num_heads":         NUM_HEADS,
        "num_rbf":           NUM_RBF,
        "cutoff":            CUTOFF_ANGSTROM,
        "dropout":           DROPOUT,
        "stochastic_depth":  STOCHASTIC_DEPTH_P,
        "coord_jitter":      COORD_JITTER_STD,
        "val_fraction":      VAL_FRACTION,
        "test_fraction":     TEST_FRACTION,
        "rf_trees":          RF_TREES,
        "rf_fp_bits":        RF_FP_BITS,
        "af_extra_dim":      get_af_extra_dim(),
    }


def get_config_hash() -> str:
    """MD5 hash of config dict for checkpoint fingerprinting."""
    s = json.dumps(get_config_dict(), sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def validate_config() -> List[str]:
    """Return list of configuration warnings (empty = all good)."""
    warnings_out: List[str] = []

    if BACKEND not in ("se3", "lite"):
        warnings_out.append(f"BACKEND='{BACKEND}' invalid; must be 'se3' or 'lite'")

    if TRAINING_MODE not in ("fresh", "continue"):
        warnings_out.append(f"TRAINING_MODE='{TRAINING_MODE}' must be 'fresh'|'continue'")

    if FUSION_DIM % NUM_HEADS != 0:
        warnings_out.append(
            f"FUSION_DIM={FUSION_DIM} not divisible by NUM_HEADS={NUM_HEADS} "
            "— attention heads will error at runtime"
        )

    if GRAPH_HIDDEN % NUM_HEADS != 0:
        warnings_out.append(
            f"GRAPH_HIDDEN={GRAPH_HIDDEN} not divisible by NUM_HEADS={NUM_HEADS}"
        )

    if USE_ALPHAFOLD:
        af_path = Path(ALPHAFOLD_DIR)
        if not af_path.exists():
            warnings_out.append(f"USE_ALPHAFOLD=True but ALPHAFOLD_DIR '{ALPHAFOLD_DIR}' missing")
        else:
            n = len(list(af_path.glob("*.pdb")))
            if n == 0:
                warnings_out.append(f"USE_ALPHAFOLD=True but no .pdb files in '{ALPHAFOLD_DIR}'")

    if USE_SWA and USE_EMA:
        warnings_out.append("USE_SWA + USE_EMA simultaneously — disable USE_SWA")

    if BATCH_SIZE < 4:
        warnings_out.append(f"BATCH_SIZE={BATCH_SIZE} very small; consider ≥ 8")

    return warnings_out


def verify_backend(requested: str, loaded: str) -> None:
    """Abort if backend requested in config doesn't match loaded model encoder."""
    if requested != loaded:
        print(f"\n  ✗  BACKEND MISMATCH — ABORTING")
        print(f"     Requested : {requested}")
        print(f"     Loaded    : {loaded}")
        print(f"     Fix: set BACKEND='{loaded}' in GLOBAL_CONFIG.py\n")
        sys.exit(1)


def verify_checkpoint_arch(ckpt_arch: dict, model_arch: dict) -> bool:
    """Return False and print mismatch details if architectures differ."""
    keys = ["backend", "fusion_dim", "graph_hidden", "num_heads", "af_extra_dim"]
    mismatches = [
        f"  {k}: ckpt={ckpt_arch[k]}  current={model_arch[k]}"
        for k in keys
        if k in ckpt_arch and k in model_arch and ckpt_arch[k] != model_arch[k]
    ]
    if mismatches:
        print("\n  ╔════════════════════════════════════════════╗")
        print("  ║  CHECKPOINT ARCHITECTURE MISMATCH          ║")
        print("  ║  Rebuilding from checkpoint dims …         ║")
        print("  ╠════════════════════════════════════════════╣")
        for m in mismatches:
            print(f"  ║  {m:<44}║")
        print("  ╚════════════════════════════════════════════╝\n")
        return False
    return True


def print_startup_info() -> None:
    """Print config banner at startup (V40-02)."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║        SE3AF v{SE3AF_VERSION}  —  3D Equivariant PROTAC AI        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Config source:  GLOBAL_CONFIG.py                      ║")
    print(f"║  Config hash:    {get_config_hash():<38}║")
    print("╠══════════════════════════════════════════════════════════╣")
    _be = "SE3GraphTransformer" if USE_SE3 else "Lite3DEncoder"
    print(f"║  Backend:        {_be:<38}║")
    print(f"║  3D Coords:      ALWAYS ON ✓                           ║")
    print(f"║  Training mode:  {TRAINING_MODE:<38}║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  ACTIVE COMPONENTS                                       ║")
    print(f"║    ESM-2:        {'ON ✓' if USE_ESM else 'OFF':<38}║")
    print(f"║    AlphaFold:    {'ON ✓' if USE_ALPHAFOLD else 'OFF':<38}║")
    print(f"║    RF Stacker:   {'ON ✓' if USE_RF else 'OFF':<38}║")
    print(f"║    GOSS:         {'ON ✓' if USE_GOSS else 'OFF':<38}║")
    print(f"║    pLDDT Attn:   {'ON ✓' if USE_PLDDT_ATTENTION else 'OFF':<38}║")
    print(f"║    EMA (0.999):  {'ON ✓' if USE_EMA else 'OFF':<38}║")
    print(f"║    Calibration:  {'ON ✓' if CALIBRATION_ENABLED else 'OFF':<38}║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  ARCHITECTURE                                            ║")
    print(f"║    fusion_dim:   {FUSION_DIM:<38}║")
    print(f"║    graph_hidden: {GRAPH_HIDDEN:<38}║")
    print(f"║    num_layers:   {NUM_GRAPH_LAYERS:<38}║")
    print(f"║    num_heads:    {NUM_HEADS:<38}║")
    print(f"║    num_rbf:      {NUM_RBF:<38}║")
    print(f"║    cutoff:       {CUTOFF_ANGSTROM} Å{'':<35}║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  TRAINING                                                ║")
    print(f"║    epochs:       {EPOCHS:<38}║")
    print(f"║    batch_size:   {BATCH_SIZE} × grad_accum {GRAD_ACCUM_STEPS} = {BATCH_SIZE*GRAD_ACCUM_STEPS} eff{'':<14}║")
    print(f"║    lr:           {LEARNING_RATE:<38}║")
    print(f"║    dropout:      {DROPOUT:<38}║")
    print(f"║    stoch_depth:  {STOCHASTIC_DEPTH_P:<38}║")
    print(f"║    coord_jitter: {COORD_JITTER_STD} Å (train augment){'':<18}║")
    print("╚══════════════════════════════════════════════════════════╝")
    warns = validate_config()
    if warns:
        print("\n  ⚠  CONFIGURATION WARNINGS:")
        for w in warns:
            print(f"    - {w}")
    print()


def generate_dynamic_flowchart(
    backend: str = None,
    use_alphafold: bool = None,
    use_rf: bool = None,
) -> str:
    """Generate ASCII flowchart matching runtime state (V40-15)."""
    b  = backend      if backend      is not None else BACKEND
    af = use_alphafold if use_alphafold is not None else USE_ALPHAFOLD
    rf = use_rf        if use_rf        is not None else USE_RF

    enc = ("SE3GraphTransformer\n  (SE(3)-equivariant MP + learnable RBF)"
           if b == "se3" else
           "Lite3DEncoder\n  (distance-bias 3D attention)")
    lines = [
        "=" * 60,
        "  SE3AF v4.1 — Model Architecture (Runtime)",
        "=" * 60, "",
        "  SMILES × 3  +  3D Conformers (ETKDGv3 + MMFF)",
        "      │",
        "      ▼",
        f"  {enc}",
        "  [tgt_lig | e3_lig | linker] — ALL with 3D coords",
        "      │",
        "      ▼",
    ]
    if af:
        lines += [
            "  CrossInteractionFusion (C(5,2)=10 pairs)",
            "      ↑ ESM-2 (1280-dim) + AlphaFold (pLDDT-weighted Cα)",
        ]
    else:
        lines += [
            "  CrossInteractionFusion (C(5,2)=10 pairs)",
            "      ↑ ESM-2 (1280-dim)  [AlphaFold: DISABLED]",
        ]
    lines += [
        "      │",
        "      ├──→ Main classifier  →  PROTAC activity",
        "      ├──→ Stability head   →  ternary stability",
        "      └──→ Interaction head →  target engagement",
        "      │",
        "      ▼",
        "  DynamicLossBalancer (Kendall uncertainty weighting)",
    ]
    if rf:
        lines += [
            "      │", "      ▼",
            "  RF Stacker (neural × 3 | Morgan FP × 2048)",
            "      │", "      ▼",
            "  Final ensemble prediction",
        ]
    else:
        lines += ["      │", "      ▼", "  Final prediction (neural only)"]
    lines += ["", "=" * 60]
    return "\n".join(lines)


# ===========================================================================
# BACKWARD-COMPAT — keep 'import config' working in legacy code
# ===========================================================================
def _sync_to_config_module() -> None:
    try:
        import config as _cm
        _names = [
            "SE3AF_VERSION", "BACKEND", "USE_SE3", "USE_LITE", "USE_COORDS",
            "USE_ESM", "USE_RF", "USE_ALPHAFOLD", "USE_GOSS", "USE_EMA",
            "USE_SWA", "TRAINING_MODE", "EPOCHS", "BATCH_SIZE", "LEARNING_RATE",
            "WEIGHT_DECAY", "GRAD_CLIP", "WARMUP_EPOCHS", "SEED", "DROPOUT",
            "LABEL_SMOOTHING", "LABEL_SMOOTHING_END", "FEATURE_NOISE_STD",
            "MIXUP_ALPHA", "STOCHASTIC_DEPTH_P", "COORD_JITTER_STD",
            "EMA_DECAY", "USE_AMP", "AMP_DTYPE", "GRAD_ACCUM_STEPS",
            "FOCAL_GAMMA", "FOCAL_ALPHA", "STABILITY_LOSS_WEIGHT",
            "INTERACTION_LOSS_WEIGHT", "FUSION_DIM", "ESM_DIM", "GRAPH_HIDDEN",
            "NUM_GRAPH_LAYERS", "NUM_HEADS", "NUM_RBF", "CUTOFF_ANGSTROM",
            "GOSS_TOP_K", "VAL_FRACTION", "TEST_FRACTION", "NUM_WORKERS",
            "PIN_MEMORY", "USE_SMILES_AUGMENT", "SMILES_AUGMENT_P",
            "USE_COORD_JITTER", "CHECKPOINT_DIR", "SAVE_EVERY_N_EPOCHS",
            "KEEP_LAST_N", "EARLY_STOPPING_PATIENCE", "OVERFIT_WINDOW",
            "RF_TREES", "RF_DEPTH", "RF_FP_BITS", "RF_MIN_SAMPLES_LEAF",
            "ALPHAFOLD_DIR", "AF_EXTRA_DIM", "GPU_MEMORY_LIMIT_GB",
            "DATA_DIR", "DATA_FILE", "CACHE_DIR", "LOG_DIR",
            "MAX_SCAFFOLD_OVERLAP_FRACTION", "MAX_SMILES_OVERLAP_FRACTION",
            "CALIBRATION_ENABLED", "TEMPERATURE_INIT",
        ]
        for name in _names:
            if name in globals():
                setattr(_cm, name, globals()[name])
    except ImportError:
        pass


if __name__ == "__main__":
    print_startup_info()
    print(f"Config hash  : {get_config_hash()}")
    print(f"AF extra dim : {get_af_extra_dim()}")
    print()
    print(generate_dynamic_flowchart())
