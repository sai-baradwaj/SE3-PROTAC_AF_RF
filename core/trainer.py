"""
core/trainer.py  (SE3AF v3.7)
==============================
SE3AF training engine + RF stacker ensemble.

v3.7 CHANGES
------------
V37-01  Read defaults from config.py (single global control center)
V37-02  TRAINING_MODE support (fresh/continue) via config.py
V37-03  Combined score checkpoint selection (AUROC+AUPRC+MCC+F1+ACC)
V37-04  Architecture fingerprint stored in every checkpoint
V37-05  Multi-metric overfitting detection (no focal-loss false positives)
V37-06  Backend display before training
V37-07  AlphaFold coverage report (via DataLeakageAuditor)
V37-08  Focal alpha fixed: 0.5 (was 0.25 — wrong for 64% positive data)
Merged from: se3af/train.py + se3af/ensemble.py

FIXES CARRIED FORWARD
---------------------
(From train.py)
BUG-A   CRITICAL  focal_loss device arg missing → fixed via SE3AFLoss
BUG-B   HIGH      zero_grad() inside autocast block → moved before autocast
BUG-C   HIGH      Model assumed 2-tuple; crashed on 3-tuple → always unpack 3
BUG-D   HIGH      torch.load() missing weights_only=False → added
BUG-E   MEDIUM    EMA state dict copied to CPU before save → fixed
BUG-C03 CRITICAL  RF stacker never called → now called after neural training
BUG-C04 CRITICAL  PROTACDataset created without supervised=True → fixed (in main.py)
BUG-H02 HIGH      EMA.update() device mismatch → explicit v.to(device).float()
BUG-H04 HIGH      evaluate/predict/calibrate crash before setup() → guard added
BUG-H06 HIGH      _CosineWarmupScheduler used deprecated _LRScheduler base class
                  → version-compatible base class selection
BUG-H03 MEDIUM    drop_last with insufficient batches → only drop when n > 2*bs
BUG-M02 MEDIUM    pin_memory with num_workers=0 is a no-op → guard added

NEW BUGS FIXED IN THIS REFACTOR
--------------------------------
F01  HIGH  RF stacker bug: _build_fp_matrix() received a torch.utils.data.Subset
     (no ._df attribute) → silent zero fingerprints → RF effectively disabled.
     Fix: unwrap Subset transparently: when dataset is a Subset, use
     dataset.dataset._df.iloc[list(dataset.indices)].

F02  MEDIUM  Double checkpoint load: app.py called trainer.predict(ds, checkpoint=…)
     which called _load_checkpoint() again even though _init_model() had already
     loaded the weights.
     Fix: predict() now accepts checkpoint=None as "use currently loaded weights";
     _load_checkpoint() is skipped when checkpoint is None.

UI-UPDATE v3.3.0
----------------
UI01  Replaced bare logging.info() calls with professional tqdm progress bars
      (training / validation / test) matching PyTorch Lightning / Ultralytics style.
UI02  Added startup banner with data summary, hardware info, config summary.
UI03  Added per-epoch compact summary line with all key metrics.
UI04  RF stacker section now has its own progress display.
UI05  Early stopping now prints a visible warning box.

v3.4.0 NEW FEATURES
--------------------
V01  AlphaFoldStore integration: startup display of found/missing/enabled status.
V02  Backend status display: 'Lite3DEncoder active' / 'SE3 backend active'.
V03  Model Health Monitor: 10-metric per-epoch table (train+val loss, acc, AUROC,
     AUPRC, F1, MCC, Precision, Recall).
V04  Overfitting Detector: automatic detection via train_loss↓ + val_loss↑ pattern
     over 3 consecutive epochs; formatted WARNING box shown.
V05  Checkpoint Validation Display: shows prev/new AUROC/F1/MCC when best_model.pt saved.
V06  RF Improvements: class_weight='balanced', max_features='sqrt', min_samples_leaf=2.
V07  Dynamic pos_weight: computed from train set label distribution, passed to focal loss.
V08  force use_coords=True in dataset when backend='se3'.
V09  alphafold_dir field in TrainerConfig.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# V37-01: Import from single global control center
try:
    import config as _global_cfg
    from config import verify_backend, verify_checkpoint_arch
    _HAS_GLOBAL_CFG = True
except ImportError:
    _global_cfg = None
    _HAS_GLOBAL_CFG = False
    def verify_backend(requested, loaded):  # noqa: E302
        pass
    def verify_checkpoint_arch(ckpt_arch, model_arch):  # noqa: E302
        return True

# v3.6.0: SWA support
try:
    from torch.optim.swa_utils import AveragedModel, SWALR
    _HAS_SWA = True
except ImportError:
    _HAS_SWA = False

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split  # Subset used in setup() and pos_weight calc

from core.dataset import PROTACDataset, protac_collate_fn
from core.model import SE3AFLoss, SE3AFModel, build_from_config, get_encoder_display_name
from core.utils import (
    TemperatureScaling,
    bootstrap_ci,
    compute_metrics,
    get_logger,
    get_vram_profile,
    COL_E3_SMILES,
    COL_LNK_SMILES,
    COL_TGT_SMILES,
)
from core.ui import (
    print_startup_banner,
    print_training_header,
    print_epoch_summary,
    print_early_stop,
    print_rf_stacker_header,
    print_rf_stacker_result,
    print_training_complete,
    print_test_results,
    train_epoch_bar,
    val_bar,
    test_bar,
    rf_bar,
    # v3.4.0 new UI functions
    print_backend_status,
    print_alphafold_status,
    print_overfit_warning,
    print_checkpoint_update,
    print_health_monitor,
    # v3.5.0 new UI functions
    print_architecture_visualization,
)


# ===========================================================================
# SECTION 1 — TrainerConfig
# ===========================================================================

@dataclass
class TrainerConfig:
    """All hyperparameters for SE3AF v4.1 training.

    All defaults come from GLOBAL_CONFIG.py (single source of truth).
    """
    # Training loop
    epochs:            int   = 80
    batch_size:        int   = 8
    learning_rate:     float = 3e-4
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    warmup_epochs:     int   = 5
    seed:              int   = 42

    # Loss
    focal_gamma:               float = 2.0
    focal_alpha:               float = 0.5
    stability_loss_weight:     float = 0.25
    interaction_loss_weight:   float = 0.25
    ssl_loss_weight:           float = 0.0
    contrastive_loss_weight:   float = 0.0
    label_smoothing:           float = 0.08
    label_smoothing_end:       float = 0.02

    # Regularisation
    dropout:              float = 0.25
    ema_decay:            float = 0.999
    use_ema:              bool  = True
    stochastic_depth_p:   float = 0.15
    feature_noise_std:    float = 0.03
    mixup_alpha:          float = 0.3
    coord_jitter_std:     float = 0.05   # v4.1: 3D coordinate jitter (Å)

    # SWA — disabled (conflicts with EMA)
    use_swa:                bool  = False
    swa_start_fraction:     float = 0.75
    swa_lr:                 float = 5e-5

    # Augmentation
    use_smiles_augment:  bool  = True
    smiles_augment_p:    float = 0.5
    use_coord_jitter:    bool  = True

    # Data
    val_fraction:  float = 0.15
    test_fraction: float = 0.10
    num_workers:   int   = 0
    pin_memory:    bool  = False

    # 3D pipeline — always True in v4.1
    use_coords:    bool  = True

    # Checkpointing
    checkpoint_dir:      str  = "checkpoints"
    save_every_n_epochs: int  = 5
    keep_last_n:         int  = 3
    early_stop_patience: int  = 20
    early_stop_metric:   str  = "auroc"
    early_stop_mode:     str  = "max"
    overfit_window:      int  = 3

    # Precision
    use_amp:    bool = False
    amp_dtype:  str  = "bfloat16"

    # Model architecture
    backend:          str   = "se3"
    fusion_dim:       int   = 192
    esm_dim:          int   = 1280
    graph_hidden:     int   = 192
    num_graph_layers: int   = 5
    num_heads:        int   = 6
    num_rbf:          int   = 20
    cutoff:           float = 8.0

    # Gradient accumulation
    grad_accum_steps: int = 4

    # RF stacker
    use_rf_stacker:  bool = True
    rf_n_estimators: int  = 300
    rf_fp_bits:      int  = 2048
    rf_max_depth:    int  = 12

    # AlphaFold
    alphafold_dir: str  = "data/alphafold"
    af_extra_dim:  int  = 4

    # GOSS cross-attention
    use_goss:   bool = True
    goss_top_k: int  = 6

    # Training mode
    training_mode: str = "fresh"   # "fresh" | "continue"

    # SSL (stubbed)
    use_masked_graph:    bool = False
    use_contrastive:     bool = False
    ssl_pretrain_epochs: int  = 0

    @classmethod
    def from_global_config(cls) -> "TrainerConfig":
        """V37-01: Create TrainerConfig from GLOBAL_CONFIG.py (single authority)."""
        if not _HAS_GLOBAL_CFG:
            return cls()
        cfg = cls()
        _map = {
            "EPOCHS":                  "epochs",
            "BATCH_SIZE":              "batch_size",
            "LEARNING_RATE":           "learning_rate",
            "WEIGHT_DECAY":            "weight_decay",
            "GRAD_CLIP":               "grad_clip",
            "WARMUP_EPOCHS":           "warmup_epochs",
            "SEED":                    "seed",
            "FOCAL_GAMMA":             "focal_gamma",
            "FOCAL_ALPHA":             "focal_alpha",
            "STABILITY_LOSS_WEIGHT":   "stability_loss_weight",
            "INTERACTION_LOSS_WEIGHT": "interaction_loss_weight",
            "LABEL_SMOOTHING":         "label_smoothing",
            "LABEL_SMOOTHING_END":     "label_smoothing_end",
            "DROPOUT":                 "dropout",
            "EMA_DECAY":               "ema_decay",
            "USE_EMA":                 "use_ema",
            "USE_SWA":                 "use_swa",
            "SWA_START_FRACTION":      "swa_start_fraction",
            "SWA_LR":                  "swa_lr",
            "USE_SMILES_AUGMENT":      "use_smiles_augment",
            "SMILES_AUGMENT_P":        "smiles_augment_p",
            "FEATURE_NOISE_STD":       "feature_noise_std",
            "MIXUP_ALPHA":             "mixup_alpha",
            "STOCHASTIC_DEPTH_P":      "stochastic_depth_p",
            "COORD_JITTER_STD":        "coord_jitter_std",
            "USE_COORD_JITTER":        "use_coord_jitter",
            "VAL_FRACTION":            "val_fraction",
            "TEST_FRACTION":           "test_fraction",
            "NUM_WORKERS":             "num_workers",
            "PIN_MEMORY":              "pin_memory",
            "EARLY_STOPPING_PATIENCE": "early_stop_patience",
            "OVERFIT_WINDOW":          "overfit_window",
            "GRAD_ACCUM_STEPS":        "grad_accum_steps",
            "RF_TREES":                "rf_n_estimators",
            "RF_DEPTH":                "rf_max_depth",
            "RF_FP_BITS":              "rf_fp_bits",
            "USE_RF":                  "use_rf_stacker",
            "ALPHAFOLD_DIR":           "alphafold_dir",
            "AF_EXTRA_DIM":            "af_extra_dim",
            "USE_GOSS":                "use_goss",
            "GOSS_TOP_K":              "goss_top_k",
            "FUSION_DIM":              "fusion_dim",
            "ESM_DIM":                 "esm_dim",
            "GRAPH_HIDDEN":            "graph_hidden",
            "NUM_GRAPH_LAYERS":        "num_graph_layers",
            "NUM_HEADS":               "num_heads",
            "BACKEND":                 "backend",
            "TRAINING_MODE":           "training_mode",
            "USE_AMP":                 "use_amp",
            "AMP_DTYPE":               "amp_dtype",
            "USE_COORDS":              "use_coords",
        }
        for cfg_attr, tc_attr in _map.items():
            val = getattr(_global_cfg, cfg_attr, None)
            if val is not None and hasattr(cfg, tc_attr):
                setattr(cfg, tc_attr, val)
        # Force af_extra_dim=0 when USE_ALPHAFOLD=False
        if not getattr(_global_cfg, "USE_ALPHAFOLD", True):
            cfg.af_extra_dim = 0
        # Add num_rbf and cutoff
        nr = getattr(_global_cfg, "NUM_RBF", None)
        if nr: cfg.num_rbf = nr
        co = getattr(_global_cfg, "CUTOFF_ANGSTROM", None)
        if co: cfg.cutoff = co
        cfg.alphafold_dir = ""
        return cfg

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainerConfig":
        """Load from a JSON file, ignoring comment keys.
        V37: Starts from global config defaults, then applies JSON overrides.
        """
        # V37-01: Start from global config (single authority)
        cfg = cls.from_global_config() if _HAS_GLOBAL_CFG else cls()

        # Apply JSON overrides on top
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        d = {k: v for k, v in d.items()
             if not k.startswith("//")
             and not k.startswith("_")   # ignore _comment, _C01_fix_comment etc.
        }
        for k, v in d.items():
            if hasattr(cfg, k) and v != "":
                setattr(cfg, k, v)
        return cfg


# ===========================================================================
# SECTION 2 — EMA
# ===========================================================================

class EMA:
    """Exponential Moving Average of model weights.

    BUG-H02 FIX: update() explicitly aligns device and dtype before blending.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                shadow_v = self.shadow[k]
                # BUG-H02 FIX: align device and dtype explicitly
                v_aligned = v.to(shadow_v.device).float()
                self.shadow[k] = self.decay * shadow_v.float() + (1 - self.decay) * v_aligned

    def state_dict(self) -> Dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# ===========================================================================
# SECTION 3 — LR Scheduler
# ===========================================================================

# BUG-H06 FIX: version-compatible base class
try:
    _LRSchedulerBase = torch.optim.lr_scheduler.LRScheduler      # PyTorch >= 2.2
except AttributeError:
    _LRSchedulerBase = torch.optim.lr_scheduler._LRScheduler      # type: ignore[attr-defined]


class _CosineWarmupScheduler(_LRSchedulerBase):
    """Cosine decay LR scheduler with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


# ===========================================================================
# SECTION 4 — RF Stacker
# ===========================================================================

_ensemble_logger = logging.getLogger("se3af.ensemble")


def _mol_fingerprint(smiles: str, n_bits: int = 1024) -> np.ndarray:
    """Morgan fingerprint (radius=2) as float32 numpy array.

    Returns a zero array on invalid / missing SMILES.
    """
    arr = np.zeros(n_bits, dtype=np.float32)
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        if not smiles or not isinstance(smiles, str):
            return arr
        smiles = smiles.strip()
        if smiles.lower() in ("nan", "none", "null", ""):
            return arr
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return arr
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), arr)
    except Exception:
        pass
    return arr


def _build_fp_matrix(dataset: Dataset, fp_bits: int = 1024) -> np.ndarray:
    """Concatenated Morgan fingerprints: [tgt_lig | e3_lig | linker] × fp_bits.

    F01 FIX: when *dataset* is a ``torch.utils.data.Subset`` (which has no
    ``._df`` attribute), unwrap it to access the underlying DataFrame using
    the Subset's indices.  This was silently producing zero fingerprints for
    every sample because ``getattr(Subset_instance, '_df', None)`` always
    returned None.
    """
    # F01 FIX: unwrap Subset transparently
    if isinstance(dataset, Subset):
        base_df = getattr(dataset.dataset, "_df", None)
        if base_df is not None:
            df = base_df.iloc[list(dataset.indices)].reset_index(drop=True)
        else:
            df = None
    else:
        df = getattr(dataset, "_df", None)

    N = len(dataset)

    if df is None:
        _ensemble_logger.warning(
            "RF Stacker: dataset has no ._df attribute; using zero fingerprints"
        )
        return np.zeros((N, fp_bits * 3), dtype=np.float32)

    parts = []
    for col in [COL_TGT_SMILES, COL_E3_SMILES, COL_LNK_SMILES]:
        if col in df.columns:
            fps = np.vstack([_mol_fingerprint(str(s), fp_bits) for s in df[col]])
        else:
            fps = np.zeros((N, fp_bits), dtype=np.float32)
        parts.append(fps)

    return np.hstack(parts)   # (N, fp_bits * 3)


def _collect_neural_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    collate_fn,
    batch_size: int = 32,
    use_amp: bool = False,
    amp_dtype=None,
    desc: str = "Inference",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference in insertion order.

    Returns
    -------
    probs  : (N, 3) float32 — [main_prob, stab_prob, inter_prob]
    labels : (N,)  int64
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
    )

    if use_amp and device.type == "cuda" and amp_dtype is not None:
        try:
            ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
        except TypeError:
            ctx = torch.cuda.amp.autocast(dtype=amp_dtype)   # type: ignore[attr-defined]
    else:
        ctx = nullcontext()

    model.eval()
    all_probs:  list = []
    all_labels: list = []

    with torch.no_grad():
        pbar = rf_bar(len(loader), desc=f"  RF Stacker [{desc}]")
        for batch in loader:
            dev_batch = {
                k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
            with ctx:
                main, stab, inter = model(
                    tgt_graph=dev_batch["tgt_graph"],
                    e3_graph=dev_batch["e3_graph"],
                    lnk_graph=dev_batch["lnk_graph"],
                    tgt_esm=dev_batch["tgt_esm"],
                    e3_esm=dev_batch["e3_esm"],
                )
            probs = torch.stack([
                torch.sigmoid(main),
                torch.sigmoid(stab),
                torch.sigmoid(inter),
            ], dim=1).float().cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(batch["labels"].tolist())
            pbar.update(1)
        pbar.close()

    model.train()
    return np.vstack(all_probs), np.array(all_labels, dtype=np.int64)


class RFStackerResult:
    """Fitted RF stacker with a predict() convenience method."""

    def __init__(self, rf, threshold: float, fp_bits: int) -> None:
        self.rf        = rf
        self.threshold = threshold
        self.fp_bits   = fp_bits

    def predict(
        self,
        neural_probs: np.ndarray,   # (N, 3)
        dataset: Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (probabilities [N], binary predictions [N] at self.threshold)."""
        fps   = _build_fp_matrix(dataset, self.fp_bits)
        X     = np.hstack([neural_probs, fps]).astype(np.float32)
        probs = self.rf.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        return probs, preds

    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {"rf": self.rf, "threshold": self.threshold, "fp_bits": self.fp_bits},
            path,
        )
        _ensemble_logger.info(f"RF stacker saved -> {path}")

    @classmethod
    def load(cls, path: str) -> "RFStackerResult":
        import joblib
        d = joblib.load(path)
        return cls(rf=d["rf"], threshold=d["threshold"], fp_bits=d["fp_bits"])


def build_rf_stacker(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    collate_fn,
    device: torch.device,
    batch_size: int = 32,
    use_amp: bool = False,
    amp_dtype=None,
    n_estimators: int = 500,
    fp_bits: int = 1024,
    max_depth: int = 10,   # V38-FIX: now a real parameter (was hardcoded)
    save_path: Optional[str] = None,
) -> RFStackerResult:
    """Train an RF stacker on top of the neural model (no leakage).

    Workflow
    --------
    1. Collect neural predictions on TRAIN set with the already-trained model.
    2. Compute Morgan fingerprints for TRAIN set.
    3. X_train = [3 neural probs | 3072 FP bits] → fit RandomForestClassifier.
    4. Tune decision threshold on VAL set predictions.
    5. Optionally save {rf, threshold, fp_bits} to *save_path*.

    F01 FIX: _build_fp_matrix() now unwraps Subset datasets automatically.

    v3.4.0 RF IMPROVEMENTS:
    - class_weight='balanced'  : proper class balancing via sklearn (not manual)
    - max_features='sqrt'      : standard best practice for RF feature selection
    - min_samples_leaf=2       : reduce overfitting on small datasets
    - n_jobs=-1                : full CPU parallelism
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise ImportError(
            "scikit-learn is required for the RF stacker. "
            "Install with: pip install scikit-learn"
        )

    print_rf_stacker_header()

    print("  Collecting neural predictions (train)...")
    train_probs, train_labels = _collect_neural_predictions(
        model, train_dataset, device, collate_fn, batch_size, use_amp, amp_dtype,
        desc="train",
    )
    print("  Collecting neural predictions (val)...")
    val_probs, val_labels = _collect_neural_predictions(
        model, val_dataset, device, collate_fn, batch_size, use_amp, amp_dtype,
        desc="val",
    )

    print("  Computing Morgan fingerprints...")
    train_fps = _build_fp_matrix(train_dataset, fp_bits)   # F01 fix: Subset unwrap
    val_fps   = _build_fp_matrix(val_dataset, fp_bits)     # F01 fix: Subset unwrap

    X_train = np.hstack([train_probs, train_fps]).astype(np.float32)
    X_val   = np.hstack([val_probs,   val_fps  ]).astype(np.float32)

    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos

    print(
        f"  Fitting {n_estimators} trees  |  "
        f"{len(train_labels)} samples  |  "
        f"{X_train.shape[1]} features  |  "
        f"pos={n_pos}  neg={n_neg}  |  "
        f"class_weight=balanced  max_features=sqrt"
    )

    # v3.4.0 RF IMPROVEMENTS: class_weight='balanced', max_features='sqrt', min_samples_leaf=2
    try:
        from tqdm import tqdm as _tqdm

        with _tqdm(
            total=n_estimators,
            desc="  RF Fitting",
            unit="tree",
            colour="yellow",
            ncols=100,
        ) as pbar:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,         # V38-FIX: use parameter (was hardcoded 10)
                min_samples_leaf=2,          # v3.4.0: was 3, now 2 for better recall
                max_features="sqrt",         # v3.4.0: standard RF best practice
                class_weight="balanced",     # v3.4.0: proper sklearn balancing
                random_state=42,
                n_jobs=-1,
                warm_start=False,
            )
            rf.fit(X_train, train_labels)
            pbar.update(n_estimators)
    except ImportError:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,             # V38-FIX: use parameter (was hardcoded 10)
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, train_labels)

    # Tune threshold on validation set
    val_scores = rf.predict_proba(X_val)[:, 1]
    best_t, best_s = 0.5, -1.0
    for i in range(5, 96):
        t    = i / 100.0
        pred = (val_scores >= t).astype(int)
        tp   = int(((pred == 1) & (val_labels == 1)).sum())
        fp   = int(((pred == 1) & (val_labels == 0)).sum())
        tn   = int(((pred == 0) & (val_labels == 0)).sum())
        fn   = int(((pred == 0) & (val_labels == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        score = 0.5 * f1 + 0.5 * (0.5 * (rec + spec))
        if score > best_s:
            best_s, best_t = score, t

    if len(np.unique(val_labels)) > 1:
        val_auroc = roc_auc_score(val_labels, val_scores)
    else:
        val_auroc = 0.5

    print_rf_stacker_result(val_auroc, best_t)

    result = RFStackerResult(rf=rf, threshold=best_t, fp_bits=fp_bits)
    if save_path:
        result.save(save_path)

    return result


# ===========================================================================
# SECTION 5 — SE3AFTrainer
# ===========================================================================

class SE3AFTrainer:
    """Full training orchestrator for SE3AF.

    Usage
    -----
    >>> cfg     = TrainerConfig.from_json("configs/train_config.json")
    >>> trainer = SE3AFTrainer(cfg)
    >>> dataset = PROTACDataset("data/", supervised=True)
    >>> trainer.setup(dataset)
    >>> trainer.train()
    """

    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg    = cfg
        self.logger = get_logger("se3af.train", log_file="logs/train.log")
        self._setup_device()
        self._ckpt_dir = Path(cfg.checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Set in setup()
        self.model:        Optional[SE3AFModel] = None
        self._rf_stacker:  Optional[RFStackerResult] = None

        # v3.4.0: Training history for health monitor and overfitting detection
        self._train_loss_history: List[float] = []
        self._val_loss_history:   List[float] = []
        self._val_metrics_history: List[Dict[str, float]] = []

        # v3.4.0: Best metrics tracking for checkpoint validation display
        self._best_metrics_before_save: Dict[str, float] = {}

    def _setup_device(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    # ------------------------------------------------------------------
    def _make_model_cfg(self, cfg: "TrainerConfig", af_extra_dim: int) -> dict:
        """v4.1: Centralised model config assembly for all build_from_config() call sites.

        Ensures all v4.1 parameters (num_rbf, cutoff, jitter_std, stochastic_depth,
        use_coords) are consistently passed regardless of call site, eliminating
        the risk of omitting new parameters when the model is rebuilt mid-setup
        (AF detection, checkpoint mismatch recovery).

        Key renames vs v4.0:
          stochastic_depth_p  →  stochastic_depth   (model.py kwarg name)
          jitter_std          ←  coord_jitter_std    (TrainerConfig field)
        """
        # Resolve jitter_std: only inject during training if use_coord_jitter is True
        jitter = (
            getattr(cfg, "coord_jitter_std", 0.05)
            if getattr(cfg, "use_coord_jitter", True)
            else 0.0
        )
        return {
            # Core architecture
            "backend":          cfg.backend,
            "fusion_dim":       cfg.fusion_dim,
            "esm_dim":          cfg.esm_dim,
            "graph_hidden":     cfg.graph_hidden,
            "num_graph_layers": cfg.num_graph_layers,
            "num_heads":        cfg.num_heads,
            "dropout":          cfg.dropout,
            # v4.1 new — model.py expects "stochastic_depth" (not "stochastic_depth_p")
            "stochastic_depth": getattr(cfg, "stochastic_depth_p", 0.15),
            # v4.1 new — GaussianRBF / DistanceBiasRBF centres + cutoff
            "num_rbf":          getattr(cfg, "num_rbf",  20),
            "cutoff":           getattr(cfg, "cutoff",   8.0),
            # v4.1 new — per-layer coordinate jitter (Å)
            "jitter_std":       jitter,
            # AlphaFold extra dim (0 when AF disabled)
            "af_extra_dim":     af_extra_dim,
            # GOSS
            "use_goss":         getattr(cfg, "use_goss",   True),
            "goss_top_k":       getattr(cfg, "goss_top_k", 6),
        }

    # ------------------------------------------------------------------
    def setup(self, dataset: PROTACDataset) -> None:
        """Build model, dataloaders, loss, optimiser, and scheduler.

        v3.4.0 additions:
        - AlphaFoldStore initialized and displayed
        - Backend status displayed
        - Dynamic pos_weight computed from label distribution
        - force use_coords=True when backend='se3'
        """
        cfg = self.cfg

        # v4.1: backend='auto' is removed — SE3AF v4.1 requires an explicit backend
        # since 3D coordinates are mandatory and both backends now support them.
        # Keeping auto-resolution would silently differ between train/inference envs.
        if cfg.backend == "auto":
            self.logger.warning(
                "V4.1: backend='auto' is deprecated — defaulting to 'se3'. "
                "Set backend='lite' or backend='se3' explicitly in train_config.json."
            )
            print(
                "\n  ⚠  V4.1: backend='auto' is deprecated — defaulting to 'se3'.\n"
                "     Set backend='lite' or backend='se3' in train_config.json.\n",
                flush=True,
            )
            cfg.backend = "se3"
        if cfg.backend not in ("lite", "se3"):
            raise ValueError(
                f"backend must be 'lite', 'se3', or 'auto', got: '{cfg.backend}'. "
                "Set backend='se3' for SE(3)-equivariant message passing, "
                "or backend='lite' for distance-bias 3D attention."
            )

        # v4.1: Force use_coords=True for ALL backends — 3D coordinates are
        # mandatory throughout the entire pipeline (V4.1-3D-01)
        self.logger.info(
            "V4.1-3D-01: Forcing use_coords=True for ALL backends — "
            "3D coordinate-based message passing is mandatory in v4.1"
        )
        if hasattr(dataset, "use_coords"):
            dataset.use_coords = True
        cfg.use_coords = True   # patch cfg in-place for consistency

        self.model = build_from_config(
            self._make_model_cfg(cfg, getattr(cfg, "af_extra_dim", 0))
        ).to(self.device)

        # v3.4.0: Display backend status
        try:
            backend_display = get_encoder_display_name(self.model)
            print_backend_status(backend_display)
        except Exception as e:
            self.logger.warning(f"Backend display failed (non-fatal): {e}")

        # V40-03: Verify backend matches GLOBAL_CONFIG after model construction
        try:
            _global_backend = getattr(_global_cfg, "BACKEND", None)
            if _global_backend and _global_backend not in ("auto",):
                verify_backend(_global_backend, cfg.backend)
        except SystemExit:
            raise
        except Exception as _vb_err:
            self.logger.warning(f"V40-03: backend verify skipped: {_vb_err}")

        # v3.4.0 / v3.6.0 / v3.6.1: Initialize AlphaFold store, display status, wire into dataset
        #
        # FIX-AF-DIM-01 (v3.6.1): When alphafold_dir is empty / not a valid directory,
        # the AF store is disabled (NULL_STORE).  In that case NO AF features are appended
        # to ESM tensors in protac_collate_fn, so the model MUST be built with
        # af_extra_dim=0 to avoid a Linear shape mismatch at runtime.
        #
        # We do NOT rebuild the model here if AF is disabled — instead we patch
        # cfg.af_extra_dim = 0 and rely on the already-built model (af_extra_dim=0).
        # If AF IS enabled we rebuild with the real af_extra_dim.  The model's
        # CrossInteractionFusion._pad_esm() zero-pads at forward time for any
        # remaining edge cases (e.g. mixed batches).
        try:
            from core.alphafold import AlphaFoldStore, NULL_STORE
            if cfg.alphafold_dir and os.path.isdir(cfg.alphafold_dir):
                af_store = AlphaFoldStore(af_dir=cfg.alphafold_dir)
                # Get all protein IDs from the dataset if possible
                protein_ids = None
                base_df = getattr(dataset, "_df", None)
                if base_df is not None:
                    id_candidates = ["target_id", "protein_id", "uniprot_id", "target"]
                    for col in id_candidates:
                        if col in base_df.columns:
                            protein_ids = base_df[col].dropna().unique().tolist()
                            # v3.6.0 AF-02: wire AF store into dataset
                            if not getattr(dataset, "_af_store", None):
                                dataset._af_store  = af_store
                                dataset._af_id_col = col
                            break
                af_summary = af_store.summary(protein_ids)

                # v3.6.0 AF-01 / FIX-AF-DIM-01:
                # Only rebuild model with af_extra_dim if AF store is actually enabled.
                # If enabled=False (no PDBs found), fall through to the 'else' branch
                # so cfg.af_extra_dim is forced to 0.
                _af_extra = getattr(cfg, "af_extra_dim", 4)
                if af_store.enabled and _af_extra > 0:
                    self.logger.info(
                        f"AF-01: AlphaFold enabled — rebuilding model with "
                        f"af_extra_dim={_af_extra}"
                    )
                    self.model = build_from_config(
                        self._make_model_cfg(cfg, _af_extra)
                    ).to(self.device)
                else:
                    # AF dir found but no PDBs inside — force af_extra_dim=0
                    # FIX-AF-DIM-01: model was already built with af_extra_dim=0
                    # (from the first build_from_config call above); just patch cfg.
                    if getattr(cfg, "af_extra_dim", 0) != 0:
                        self.logger.warning(
                            "FIX-AF-DIM-01: AF store found but enabled=False "
                            "(no PDB files). Forcing af_extra_dim=0 to prevent "
                            "esm_proj shape mismatch."
                        )
                        cfg.af_extra_dim = 0
            else:
                af_store   = NULL_STORE
                af_summary = af_store.summary()
                if not cfg.alphafold_dir:
                    af_summary += "\n(No alphafold_dir configured — set in train_config.json)"
                    # FIX-AF-DIM-01: No AF dir → no AF features appended → model must use
                    # af_extra_dim=0.  Patch cfg so the already-built model is consistent.
                    if getattr(cfg, "af_extra_dim", 0) != 0:
                        self.logger.warning(
                            f"FIX-AF-DIM-01: af_extra_dim={cfg.af_extra_dim} in config "
                            "but no alphafold_dir set. Forcing af_extra_dim=0 to prevent "
                            "RuntimeError: mat1 and mat2 shapes cannot be multiplied "
                            f"({cfg.esm_dim}x{cfg.esm_dim + cfg.af_extra_dim})."
                        )
                        print(
                            f"\n  \u26a0  FIX-AF-DIM-01: af_extra_dim={cfg.af_extra_dim} in config "
                            "but alphafold_dir is empty.\n"
                            "     Forcing af_extra_dim=0. "
                            "Set alphafold_dir in train_config.json to enable AF features.\n",
                            flush=True,
                        )
                        cfg.af_extra_dim = 0
                        # Rebuild model with af_extra_dim=0 to match what collate_fn produces
                        self.model = build_from_config(
                            self._make_model_cfg(cfg, 0)   # FIX-AF-DIM-01: af_extra_dim=0
                        ).to(self.device)
                else:
                    af_summary += f"\n(Directory not found: {cfg.alphafold_dir})"
                    if getattr(cfg, "af_extra_dim", 0) != 0:
                        self.logger.warning(
                            f"FIX-AF-DIM-01: alphafold_dir='{cfg.alphafold_dir}' not found. "
                            "Forcing af_extra_dim=0."
                        )
                        cfg.af_extra_dim = 0
            print_alphafold_status(af_summary)
        except ImportError:
            print_alphafold_status(
                "AlphaFold Structures Found: 0\n"
                "AlphaFold Structures Missing: 0\n"
                "AlphaFold Features Enabled: False\n"
                "(core/alphafold.py not found)"
            )
        except Exception as e:
            self.logger.warning(f"AlphaFold status display failed (non-fatal): {e}")
            print_alphafold_status(
                "AlphaFold Structures Found: 0\n"
                "AlphaFold Structures Missing: 0\n"
                "AlphaFold Features Enabled: False\n"
                f"(Error: {e})"
            )

        self.ema = EMA(self.model, cfg.ema_decay) if cfg.use_ema else None

        N       = len(dataset)
        # Handle tiny datasets (e.g. single-molecule predict mode)
        if N <= 3:
            # All samples go to train; val/test are single-sample copies
            n_train, n_val, n_test = N, 0, 0
        else:
            n_test  = max(1, int(N * cfg.test_fraction))
            n_val   = max(1, int(N * cfg.val_fraction))
            n_train = N - n_val - n_test
            if n_train <= 0:
                n_train, n_val, n_test = max(1, N - 2), 1, 1
            # Ensure sum == N
            diff = N - (n_train + n_val + n_test)
            n_train += diff

        g = torch.Generator().manual_seed(cfg.seed)
        if n_val == 0 or n_test == 0:
            # Tiny dataset: duplicate some samples for val/test
            # (Subset already imported at top of module)
            indices = list(range(N))
            train_ds = Subset(dataset, indices)
            val_ds   = Subset(dataset, indices[-1:])
            test_ds  = Subset(dataset, indices[-1:])
        else:
            # DATA-SPLIT-01 FIX: Use stratified split to ensure both classes
            # appear in val/test sets.  Without stratification, random_split()
            # frequently places all negative samples in train with n=25,
            # leaving val/test all-positive → roc_auc_score warns "only one
            # class present" and returns 0.5 for ALL epochs.
            try:
                from sklearn.model_selection import train_test_split as sk_split
                # Gather labels from dataset
                _all_labels = []
                for _i in range(N):
                    _s = dataset[_i]
                    _all_labels.append(int(_s.label.item()) if hasattr(_s.label, 'item') else int(_s.label))
                _all_idx = list(range(N))
                # First split off test set with stratification
                _trainval_idx, _test_idx = sk_split(
                    _all_idx, test_size=n_test, random_state=cfg.seed,
                    stratify=_all_labels
                )
                _trainval_labels = [_all_labels[i] for i in _trainval_idx]
                # Then split train/val with stratification
                # Compute val size relative to trainval
                _val_size = n_val
                if _val_size >= len(_trainval_idx):
                    _val_size = max(1, len(_trainval_idx) // 5)
                _train_idx, _val_idx = sk_split(
                    _trainval_idx, test_size=_val_size, random_state=cfg.seed,
                    stratify=_trainval_labels
                )
                train_ds = Subset(dataset, _train_idx)
                val_ds   = Subset(dataset, _val_idx)
                test_ds  = Subset(dataset, _test_idx)
                self.logger.info(
                    f"DATA-SPLIT-01 FIX: Stratified split applied. "
                    f"Train={len(_train_idx)}, Val={len(_val_idx)}, Test={len(_test_idx)}. "
                    f"Val labels: {sorted([_all_labels[i] for i in _val_idx])}"
                )
            except Exception as _split_err:
                # Fallback to random_split if stratification fails (e.g., only 1 class)
                self.logger.warning(
                    f"DATA-SPLIT-01: Stratified split failed ({_split_err}), "
                    f"falling back to random_split. Val set may be class-imbalanced."
                )
                train_ds, val_ds, test_ds = random_split(
                    dataset, [n_train, n_val, n_test], generator=g
                )

        # BUG-M02 fix: pin_memory only when it can actually help
        _pin = cfg.pin_memory and (cfg.num_workers > 0) and (self.device.type == "cuda")
        # BUG-H03 fix: drop_last only when we have room for ≥ 2 full batches
        _drop_last = n_train > 2 * cfg.batch_size

        loader_kw = dict(
            collate_fn=protac_collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=_pin,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            drop_last=_drop_last, **loader_kw,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            drop_last=False, **loader_kw,
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            drop_last=False, **loader_kw,
        )

        # Keep Subset handles for RF stacker fingerprint access (F01 fix applied there)
        self._train_ds = train_ds
        self._val_ds   = val_ds
        self._test_ds  = test_ds

        # V37: Data Leakage Audit — run before training begins
        try:
            from core.utils import DataLeakageAuditor
            base_df_for_audit = getattr(dataset, "_df", None)
            if base_df_for_audit is not None:
                _auditor = DataLeakageAuditor(base_df_for_audit)
                # Get actual indices for leakage checking
                _train_idx_audit = list(_train_idx) if '_train_idx' in dir() else list(range(n_train))
                _val_idx_audit   = list(_val_idx)   if '_val_idx'   in dir() else []
                _test_idx_audit  = list(_test_idx)  if '_test_idx'  in dir() else []
                _leakage_report  = _auditor.run(_train_idx_audit, _val_idx_audit, _test_idx_audit)
                _auditor.print_summary(_leakage_report)
                # Save report to file
                import os as _os
                _os.makedirs("reports", exist_ok=True)
                with open("reports/DATA_LEAKAGE_AUDIT.md", "w", encoding="utf-8") as _fh:
                    _fh.write(_leakage_report.to_markdown())
                # V40-04: Abort training if scaffold overlap exceeds threshold
                try:
                    _max_scaffold_overlap = getattr(_global_cfg, "MAX_SCAFFOLD_OVERLAP_FRACTION", 0.3)
                    _scaffold_frac = getattr(_leakage_report, "scaffold_overlap_fraction", 0.0) or 0.0
                    if _scaffold_frac > _max_scaffold_overlap:
                        print(
                            f"\n  ✗  DATA LEAKAGE ABORT — scaffold overlap {_scaffold_frac:.1%} "
                            f"exceeds threshold {_max_scaffold_overlap:.1%}.\n"
                            f"     Fix your train/val/test split before training.\n",
                            flush=True,
                        )
                        import sys as _sys
                        _sys.exit(1)
                except SystemExit:
                    raise
                except Exception:
                    pass  # Non-fatal if leakage fraction not available
        except Exception as _audit_err:
            self.logger.warning(f"Data leakage audit failed (non-fatal): {_audit_err}")

        # ── dataset stats for UI banner ─────────────────────────────
        base_df    = getattr(dataset, "_df", None)
        dup_count  = 0
        raw_rows   = 0
        if base_df is not None:
            raw_rows   = len(base_df)
            unique_df  = base_df.drop_duplicates()
            dup_count  = raw_rows - len(unique_df)
        self._dataset_stats = {
            "Raw training rows":    n_train,
            "Raw validation rows":  n_val,
            "Raw test rows":        n_test,
            "Duplicate rows removed": dup_count,
            "Clean train rows":     n_train,
            "Clean validation rows": n_val,
            "Clean test rows":      n_test,
        }

        # v3.4.0: Compute dynamic pos_weight from training label distribution
        # This improves class imbalance handling beyond just focal_alpha
        _pos_weight_tensor = None
        try:
            # Collect labels from train split
            train_labels_list = []
            if isinstance(train_ds, Subset):
                base_df_for_labels = getattr(train_ds.dataset, "_df", None)
                if base_df_for_labels is not None:
                    # V07 FIX: search for label column using canonical name first,
                    # then common aliases, matching the normalise_columns() logic
                    from core.utils import COL_LABEL as _COL_LABEL, COL_ALIASES as _COL_ALIASES
                    label_col = None
                    # Check canonical name first, then aliases used in dataset discovery
                    _label_candidates = [_COL_LABEL] + [
                        k for k, v in _COL_ALIASES.items() if v == _COL_LABEL
                    ]
                    for lc in _label_candidates:
                        if lc in base_df_for_labels.columns:
                            label_col = lc
                            break
                    if label_col:
                        sub_labels = base_df_for_labels.iloc[
                            list(train_ds.indices)
                        ][label_col].values
                        train_labels_list = [int(v) for v in sub_labels if not np.isnan(float(v))]
            if train_labels_list:
                n_pos_lbl = sum(train_labels_list)
                n_neg_lbl = len(train_labels_list) - n_pos_lbl
                if n_pos_lbl > 0 and n_neg_lbl > 0:
                    pw = float(n_neg_lbl) / float(n_pos_lbl)
                    _pos_weight_tensor = torch.tensor([pw], dtype=torch.float32)
                    self.logger.info(
                        f"V07: Dynamic pos_weight={pw:.3f} "
                        f"(n_pos={n_pos_lbl}, n_neg={n_neg_lbl})"
                    )
        except Exception as e:
            self.logger.warning(f"pos_weight computation failed (using default): {e}")

        # V07 FIX: pass pos_weight directly to SE3AFLoss constructor
        self.criterion = SE3AFLoss(
            focal_gamma             = cfg.focal_gamma,
            focal_alpha             = cfg.focal_alpha,
            stability_loss_weight   = cfg.stability_loss_weight,
            interaction_loss_weight = cfg.interaction_loss_weight,
            ssl_loss_weight         = cfg.ssl_loss_weight,
            contrastive_loss_weight = cfg.contrastive_loss_weight,
            label_smoothing         = cfg.label_smoothing,
            pos_weight              = _pos_weight_tensor,  # V07: dynamic pos_weight
        ).to(self.device)
        if _pos_weight_tensor is not None:
            self.logger.info("V07: pos_weight applied to SE3AFLoss")

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # FIX-SCHEDULER-01: compute optimizer steps (not batch steps)
        # With grad_accum_steps=4, there are num_batches/4 optimizer steps per epoch.
        _grad_accum = max(1, getattr(cfg, "grad_accum_steps", 1))
        _batches_per_epoch = max(1, len(self.train_loader))
        _optimizer_steps_per_epoch = max(1, math.ceil(_batches_per_epoch / _grad_accum))
        total_steps = cfg.epochs * _optimizer_steps_per_epoch
        warm_steps  = cfg.warmup_epochs * _optimizer_steps_per_epoch
        self.scheduler = _CosineWarmupScheduler(self.optimizer, warm_steps, total_steps)

        self.scaler = None
        if cfg.use_amp and self.device.type == "cuda":
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except TypeError:
                self.scaler = torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def _move_batch(self, batch: Dict) -> Dict:
        dev = self.device
        return {
            k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }

    def _forward(self, batch: Dict) -> Tuple[Tensor, Tensor, Tensor]:
        """Run model forward; always unpack 3-tuple (BUG-C fix)."""
        assert self.model is not None, (
            "SE3AFTrainer.setup() must be called before _forward()"
        )
        return self.model(
            tgt_graph=batch["tgt_graph"],
            e3_graph=batch["e3_graph"],
            lnk_graph=batch["lnk_graph"],
            tgt_esm=batch["tgt_esm"],
            e3_esm=batch["e3_esm"],
        )

    # ------------------------------------------------------------------
    def _apply_feature_noise(self, batch: Dict) -> Dict:
        """Apply Gaussian noise to graph node features for regularization.

        v3.5.0 Generalization: Adding small noise to input features during
        training acts as a form of regularization (similar to Dropout but
        at the input level), improving generalization on small datasets.

        v3.6.0 GEN-01: Also applies stochastic coordinate jitter (for SE3 backend)
        """
        if self.cfg.feature_noise_std <= 0:
            return batch
        try:
            for key in ("tgt_graph", "e3_graph", "lnk_graph"):
                if key in batch:
                    g = batch[key]
                    if hasattr(g, "x") and g.x is not None:
                        noise = torch.randn_like(g.x) * self.cfg.feature_noise_std
                        g.x = g.x + noise
                    # v3.6.0 GEN-01: coordinate jitter for 3D robustness
                    if hasattr(g, "pos") and g.pos is not None:
                        coord_noise = torch.randn_like(g.pos) * (self.cfg.feature_noise_std * 0.1)
                        g.pos = g.pos + coord_noise
        except Exception:
            pass  # non-fatal
        return batch

    # ------------------------------------------------------------------
    def _train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train one epoch with tqdm progress bar.

        v3.4.0: Now returns (train_loss, train_metrics) for health monitor.
        v3.5.0: Adds feature noise regularization for improved generalization.

        FIX-GRAD-ACCUM-01 (v3.8.1): Implements true gradient accumulation.
          - Loss divided by grad_accum_steps before backward
          - optimizer.zero_grad() only every grad_accum_steps batches
          - optimizer.step() only every grad_accum_steps batches
          - scheduler.step() aligned with actual optimizer steps

        FIX-SCHEDULER-01 (v3.8.1): scheduler.step() called once per optimizer
          step (not every batch), ensuring cosine decay spans the full training.
        """
        assert self.model is not None
        self.model.train()
        total_loss    = 0.0
        amp_dtype     = torch.float16 if self.cfg.amp_dtype == "float16" else torch.bfloat16
        n_batches     = len(self.train_loader)
        grad_accum    = max(1, getattr(self.cfg, "grad_accum_steps", 1))

        # v3.4.0: collect logits+labels for train metrics
        all_train_logits: list = []
        all_train_labels: list = []

        pbar = train_epoch_bar(self.train_loader, epoch, self.cfg.epochs)

        # FIX-GRAD-ACCUM-01: zero_grad at start of accumulation window
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar, 1):
            batch  = self._move_batch(batch)
            # v3.5.0: Apply feature noise regularization for generalization
            if self.cfg.feature_noise_std > 0:
                batch = self._apply_feature_noise(batch)
            labels = batch["labels"]

            is_accum_step = (step % grad_accum == 0) or (step == n_batches)

            if self.scaler is not None:
                try:
                    autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
                except TypeError:
                    autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)  # type: ignore[attr-defined]
                with autocast_ctx:
                    main, stab, inter = self._forward(batch)
                    # FIX-GRAD-ACCUM-01: divide loss by accumulation steps
                    loss = self.criterion(main, stab, inter, labels) / grad_accum
                self.scaler.scale(loss).backward()
                if is_accum_step:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # FIX-SCHEDULER-01: step scheduler once per optimizer step
                    self.scheduler.step()
                    if self.ema:
                        self.ema.update(self.model)
                    self.optimizer.zero_grad()
            else:
                main, stab, inter = self._forward(batch)
                # FIX-GRAD-ACCUM-01: divide loss by accumulation steps
                loss = self.criterion(main, stab, inter, labels) / grad_accum
                loss.backward()
                if is_accum_step:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    # FIX-SCHEDULER-01: step scheduler once per optimizer step
                    self.scheduler.step()
                    if self.ema:
                        self.ema.update(self.model)
                    self.optimizer.zero_grad()

            # Scale loss back for logging (undo the /grad_accum scaling)
            total_loss += loss.item() * grad_accum

            # v3.4.0: collect for train metrics
            all_train_logits.append(main.detach().cpu().float())
            all_train_labels.append(labels.cpu())

            # Update tqdm postfix
            current_lr = self.scheduler.get_last_lr()[0] if hasattr(self.scheduler, 'get_last_lr') else self.cfg.learning_rate
            avg_loss = total_loss / step
            try:
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                    accum=f"{grad_accum}",
                )
            except Exception:
                pass

        try:
            pbar.close()
        except Exception:
            pass

        avg_loss = total_loss / max(1, n_batches)

        # v3.4.0: Compute train metrics
        train_metrics: Dict[str, float] = {}
        try:
            t_logits = torch.cat(all_train_logits).numpy()
            t_labels = torch.cat(all_train_labels).numpy()
            t_probs  = 1.0 / (1.0 + np.exp(-t_logits))
            train_metrics = compute_metrics(t_labels, t_probs)
        except Exception as e:
            self.logger.debug(f"Train metrics computation failed: {e}")
            train_metrics = {}

        return avg_loss, train_metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _eval_epoch(
        self, loader: DataLoader, epoch: int = 0, split: str = "val"
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate with tqdm progress bar.

        v3.4.0: Returns (val_loss, val_metrics) for health monitor.
        val_loss is approximated from sigmoid cross-entropy of main logits.
        """
        assert self.model is not None
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0
        n_batches  = 0

        if split == "val":
            pbar = val_bar(loader, epoch, self.cfg.epochs)
        else:
            pbar = test_bar(loader)

        for batch in pbar:
            batch = self._move_batch(batch)
            main, stab, inter = self._forward(batch)
            all_logits.append(main.cpu().float())
            all_labels.append(batch["labels"].cpu())

            # Approximate val loss for overfitting detection
            try:
                lbl_f = batch["labels"].float().cpu()
                prob  = torch.sigmoid(main.cpu().float())
                bce   = -(lbl_f * torch.log(prob.clamp(1e-7, 1-1e-7)) +
                          (1 - lbl_f) * torch.log((1 - prob).clamp(1e-7, 1-1e-7)))
                total_loss += bce.mean().item()
                n_batches  += 1
            except Exception:
                pass

        try:
            pbar.close()
        except Exception:
            pass

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs  = 1 / (1 + np.exp(-logits))
        metrics = compute_metrics(labels, probs)

        val_loss = total_loss / max(1, n_batches)
        return val_loss, metrics

    # ------------------------------------------------------------------
    def _check_overfitting(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        improved: bool,
    ) -> bool:
        """V37-05: Multi-metric overfitting detection.

        Requires ALL of the following to be true over `window` epochs:
        1. Train loss consistently decreasing
        2. Val loss consistently increasing OR val AUROC consistently decreasing
        3. Loss gap exceeds threshold

        This avoids false positives from focal loss negative values
        (BUG-V37-10 fix: was single-criterion loss-only check).

        Returns True if overfitting is detected.
        """
        # Never flag epochs where val improved (checkpoint saved)
        if improved:
            return False

        window = self.cfg.overfit_window
        if len(self._train_loss_history) < window:
            return False

        recent_train = self._train_loss_history[-window:]
        recent_val   = self._val_loss_history[-window:]

        # Condition 1: train loss consistently decreasing
        train_decreasing = all(
            recent_train[i] > recent_train[i+1]
            for i in range(len(recent_train) - 1)
        )

        # Condition 2a: val loss consistently increasing
        val_loss_increasing = all(
            recent_val[i] < recent_val[i+1]
            for i in range(len(recent_val) - 1)
        )

        # Condition 2b: val AUROC consistently decreasing (requires metric history)
        val_auroc_decreasing = False
        if len(self._val_metrics_history) >= window:
            recent_auroc = [
                m.get("auroc", 0.5)
                for m in self._val_metrics_history[-window:]
            ]
            val_auroc_decreasing = all(
                recent_auroc[i] > recent_auroc[i+1]
                for i in range(len(recent_auroc) - 1)
            )

        # V37-05: Require BOTH loss-based AND AUROC-based evidence
        # Prevents false positives when focal loss briefly goes negative
        val_worsening = val_loss_increasing and val_auroc_decreasing

        # Condition 3: significant gap (use absolute values for stability)
        abs_train = abs(train_loss)
        abs_val   = abs(val_loss)
        gap_significant = (abs_val - abs_train) > 0.05

        if train_decreasing and val_worsening and gap_significant:
            # Get AUROC for message
            curr_auroc = self._val_metrics_history[-1].get("auroc", 0) if self._val_metrics_history else 0
            reason = (
                f"Multi-metric overfitting signal detected over {window} epochs:\n"
                f"  Train loss:  {recent_train[0]:.4f} → {recent_train[-1]:.4f} (↓ decreasing)\n"
                f"  Val loss:    {recent_val[0]:.4f} → {recent_val[-1]:.4f} (↑ increasing)\n"
                f"  Val AUROC:   trending downward\n"
                f"  Loss gap:    {abs_val - abs_train:.4f}"
            )
            print_overfit_warning(
                reason=reason,
                train_loss=train_loss,
                val_loss=val_loss,
                suggested_actions=[
                    f"Increase dropout (current: {self.cfg.dropout:.2f})",
                    f"Increase weight_decay (current: {self.cfg.weight_decay:.1e})",
                    "Enable earlier stopping (reduce early_stop_patience)",
                    "Add SMILES augmentation or noise injection",
                    f"Reduce model capacity (current num_graph_layers={self.cfg.num_graph_layers})",
                    "Reduce batch_size to add gradient noise regularization",
                ],
            )
            return True

        return False

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_metric_key(metric_name: str) -> str:
        """BUG-ES01 FIX: strip 'val_' prefix before val_metrics dict lookup.

        compute_metrics() returns bare keys: 'auroc', 'auprc', 'f1', ...
        Config may use 'val_auroc' convention.  Strip the prefix here.

        Examples:  'val_auroc' -> 'auroc',  'auroc' -> 'auroc'
        """
        if metric_name.startswith("val_"):
            return metric_name[len("val_"):]
        return metric_name

    # ------------------------------------------------------------------
    def train(self, resume_from: Optional[str] = None) -> Dict[str, float]:
        """Run full training loop with early stopping and checkpoint management.

        v3.4.0 additions:
        - _train_epoch() returns (loss, train_metrics)
        - _eval_epoch() returns (val_loss, val_metrics)
        - Health Monitor: print_health_monitor() called each epoch
        - Overfitting Detector: _check_overfitting() called each epoch
        - Checkpoint Validation: print_checkpoint_update() on best_model.pt save

        v3.6.0 BUG-ES01 FIX:
        - _resolve_metric_key() strips 'val_' prefix so compute_metrics() keys match.
          Without this fix, score was always 0.0, best_model.pt never updated past ep.1,
          and early stopping fired on a bogus best_score of 0.0.

        v3.6.0 GEN additions:
        - SWA: averages model weights over final swa_start_fraction of epochs
        - Label smoothing ramp: decays label_smoothing from initial to label_smoothing_end
        """  # noqa: E501
        assert self.model is not None, "Call setup() before train()"
        cfg    = self.cfg
        logger = self.logger

        # ── Architecture Visualization (Phase 8) ────────────────────
        # Determine resolved backend (auto was resolved in setup())
        _resolved_backend = cfg.backend if cfg.backend in ("lite", "se3") else "lite"
        print_architecture_visualization(backend=_resolved_backend)

        # ── Professional startup banner ─────────────────────────────
        print_startup_banner(cfg, self._dataset_stats, self.model)
        print_training_header(cfg, str(self.device), self.model)

        best_score   = -float("inf") if cfg.early_stop_mode == "max" else float("inf")
        no_improve   = 0
        best_metrics: Dict = {}
        saved_ckpts: List[Path] = []

        # V37-02: Training mode management (fresh/continue)
        _training_mode = getattr(cfg, "training_mode", "fresh")
        _best_ckpt_path = self._ckpt_dir / "best_model.pt"

        if _training_mode == "fresh":
            print(f"\n  Training Mode: FRESH — Starting from epoch 0")
            if _best_ckpt_path.exists():
                print(f"  (Previous checkpoint exists but will be ignored in fresh mode)")
            print()
        elif _training_mode == "continue":
            if _best_ckpt_path.exists():
                logger.info(f"Training Mode: CONTINUE — Loading {_best_ckpt_path}")
                print(f"\n  Training Mode: CONTINUE")
                try:
                    self._load_checkpoint(str(_best_ckpt_path))
                    ckpt = torch.load(str(_best_ckpt_path), map_location="cpu", weights_only=False)
                    _resumed_epoch = ckpt.get("epoch", 0)
                    _resumed_metrics = ckpt.get("metrics", {})
                    _resumed_score = _resumed_metrics.get("combined_score",
                                     _resumed_metrics.get("auroc", 0.0))
                    best_score = _resumed_score
                    best_metrics = _resumed_metrics
                    print(f"  Resume Epoch:      {_resumed_epoch}")
                    print(f"  Checkpoint Loaded: {_best_ckpt_path}")
                    if "architecture_fingerprint" in ckpt:
                        fp = ckpt["architecture_fingerprint"]
                        print(f"  Config Hash:       {fp.get('config_hash', 'N/A')}")
                        print(f"  Saved Timestamp:   {fp.get('timestamp', 'N/A')}")
                    print()
                    resume_from = None   # already loaded above
                except Exception as e:
                    logger.warning(f"Could not load checkpoint for continue mode: {e}")
                    print(f"  WARNING: Could not load checkpoint ({e}); starting fresh")
            else:
                print(f"\n  Training Mode: CONTINUE requested but no best_model.pt found")
                print(f"  Falling back to FRESH mode")
                print()

        if resume_from:
            self._load_checkpoint(resume_from)

        # v3.6.0 GEN-02: Stochastic Weight Averaging (SWA) initialisation
        _swa_model  = None
        _swa_scheduler = None
        _swa_start_epoch = max(1, int(cfg.epochs * getattr(cfg, "swa_start_fraction", 0.75)))
        _use_swa = (
            getattr(cfg, "use_swa", True)
            and _HAS_SWA
            and cfg.epochs >= 4   # need at least a few epochs of SWA averaging
        )
        if _use_swa:
            try:
                _swa_model = AveragedModel(self.model)
                _swa_lr    = getattr(cfg, "swa_lr", 5e-5)
                _swa_scheduler = SWALR(
                    self.optimizer,
                    swa_lr=_swa_lr,
                    anneal_epochs=max(1, cfg.epochs - _swa_start_epoch),
                    anneal_strategy="cos",
                )
                self.logger.info(
                    f"GEN-02: SWA enabled — starts at epoch {_swa_start_epoch}, "
                    f"swa_lr={_swa_lr:.1e}, {cfg.epochs - _swa_start_epoch} averaging epochs"
                )
                print(
                    f"\n  SWA Enabled: starts epoch {_swa_start_epoch}/"
                    f"{cfg.epochs}  swa_lr={_swa_lr:.1e}",
                    flush=True,
                )
            except Exception as _swa_err:
                self.logger.warning(f"GEN-02: SWA init failed (non-fatal): {_swa_err}")
                _use_swa = False

        # v3.6.0 GEN-05: Label smoothing ramp — compute per-epoch smoothing
        _ls_start = cfg.label_smoothing
        _ls_end   = getattr(cfg, "label_smoothing_end", max(0.0, cfg.label_smoothing - 0.04))

        print(f"  Training: {cfg.epochs} epochs  |  "
              f"{len(self.train_loader)} batches/epoch  |  "
              f"batch_size={cfg.batch_size}", flush=True)
        print()

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # v3.6.0 GEN-05: Update label smoothing via ramp
            if cfg.epochs > 1:
                _progress = (epoch - 1) / max(1, cfg.epochs - 1)
                _cur_smoothing = _ls_start + (_ls_end - _ls_start) * _progress
                if hasattr(self.criterion, "label_smoothing"):
                    self.criterion.label_smoothing = float(_cur_smoothing)

            # v3.4.0: _train_epoch returns (loss, train_metrics)
            train_loss, train_metrics = self._train_epoch(epoch)

            # V38-FIX-EMA-01: Apply EMA weights BEFORE validation so the
            # model being evaluated is the averaged (smoother) version.
            # After validation, restore the training weights so gradient
            # updates continue on the live model (not the averaged copy).
            _ema_backup_state = None
            if self.ema is not None:
                _ema_backup_state = copy.deepcopy(self.model.state_dict())
                self.ema.apply(self.model)   # swap in EMA weights

            # v3.4.0: _eval_epoch returns (val_loss, val_metrics)
            val_loss, val_metrics = self._eval_epoch(
                self.val_loader, epoch=epoch, split="val"
            )

            # V38-FIX-EMA-01: Restore training weights after EMA-evaluated validation
            if _ema_backup_state is not None:
                self.model.load_state_dict(_ema_backup_state)

            elapsed = time.time() - t0

            # Update history for overfitting detection
            self._train_loss_history.append(train_loss)
            self._val_loss_history.append(val_loss)
            self._val_metrics_history.append(val_metrics)

            # V37-03: Combined score checkpoint selection
            # Use combined_score = 0.40*AUROC + 0.20*AUPRC + 0.15*MCC_norm + 0.15*F1 + 0.10*ACC
            # This replaces AUROC-only selection (V37-BUG-03 fix)
            _combined_score = self._compute_combined_score(val_metrics)
            _bare_metric = "combined_score"
            score = _combined_score

            # BUG-ES01 FIX (preserved): Also track individual AUROC for display
            _auroc = val_metrics.get("auroc", 0.0)

            # Get current LR
            try:
                current_lr = self.scheduler.get_last_lr()[0]
            except Exception:
                current_lr = cfg.learning_rate

            improved = (score > best_score)

            # Compact epoch summary line
            print_epoch_summary(
                epoch=epoch,
                total_epochs=cfg.epochs,
                train_loss=train_loss,
                val_metrics=val_metrics,
                lr=current_lr,
                elapsed=elapsed,
                best_score=best_score,
                improved=improved,
            )

            # v3.4.0: Full 10-metric health monitor table
            print_health_monitor(
                epoch=epoch,
                total_epochs=cfg.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

            # v3.4.0: Overfitting detection (NOT triggered on improved epochs)
            self._check_overfitting(epoch, train_loss, val_loss, improved)

            if improved:
                prev_best = dict(best_metrics)  # snapshot before update
                best_score   = score
                best_metrics = val_metrics
                no_improve   = 0

                # V37-03: Checkpoint display with combined score breakdown
                _reason = (
                    f"Combined Score improved to {score:.4f}  "
                    f"(AUROC={val_metrics.get('auroc',0):.4f} "
                    f"AUPRC={val_metrics.get('auprc',0):.4f} "
                    f"MCC={val_metrics.get('mcc',0):.4f} "
                    f"F1={val_metrics.get('f1',0):.4f})"
                )
                # v3.4.0: Checkpoint validation display
                print_checkpoint_update(
                    prev_metrics=prev_best,
                    new_metrics=val_metrics,
                    epoch=epoch,
                    reason=_reason,
                )

                self._save_checkpoint("best_model.pt", epoch, val_metrics)
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience:
                    # BUG-4 FIX + BUG-ES01 FIX: Display exact early stopping reason
                    _stop_reason = (
                        f"Stopped because validation {cfg.early_stop_metric} "
                        f"(resolved key: '{_bare_metric}') "
                        f"failed to improve for {no_improve} epochs.\n"
                        f"  Best {cfg.early_stop_metric}: {best_score:.4f} "
                        f"(achieved at the epoch of the last checkpoint save).\n"
                        f"  Current {cfg.early_stop_metric}: {score:.4f}\n"
                        f"  Mode: {cfg.early_stop_mode} (higher is better)"
                        if cfg.early_stop_mode == "max"
                        else f"Stopped because validation {cfg.early_stop_metric} "
                             f"(key: '{_bare_metric}') "
                             f"failed to improve for {no_improve} epochs."
                    )
                    print_early_stop(epoch, no_improve, reason=_stop_reason)
                    logger.info(
                        f"Early stopping at epoch {epoch}: {_stop_reason}"
                    )
                    break

            if epoch % cfg.save_every_n_epochs == 0:
                ckpt_name = f"epoch_{epoch:04d}.pt"
                self._save_checkpoint(ckpt_name, epoch, val_metrics)
                saved_ckpts.append(self._ckpt_dir / ckpt_name)
                while len(saved_ckpts) > cfg.keep_last_n:
                    old = saved_ckpts.pop(0)
                    if old.exists() and "best_model" not in old.name:
                        old.unlink()

            # v3.6.0 GEN-02: SWA update after swa_start_epoch
            # V38-FIX-SWA-01: When SWA is active, stop stepping the cosine
            # scheduler — SWALR takes over LR control from this epoch forward.
            # Previously both stepped simultaneously, causing conflicting LR
            # schedules and unstable final-phase training.
            if _use_swa and _swa_model is not None and epoch >= _swa_start_epoch:
                try:
                    _swa_model.update_parameters(self.model)
                    _swa_scheduler.step()   # SWA LR scheduler takes over
                    # Do NOT also call self.scheduler.step() for these epochs
                    # (the cosine scheduler was already stepped in _train_epoch
                    # per-batch, but the epoch-level step below should be skipped
                    # once SWA is active to avoid LR conflicts)
                except Exception as _swa_step_err:
                    self.logger.debug(f"SWA step failed (non-fatal): {_swa_step_err}")

        # v3.6.0 GEN-03: Apply SWA BatchNorm update + save swa_model.pt
        if _use_swa and _swa_model is not None:
            try:
                from torch.optim.swa_utils import update_bn
                print("\n  SWA: Updating BatchNorm statistics...", flush=True)
                update_bn(self.train_loader, _swa_model, device=self.device)
                _swa_path = str(self._ckpt_dir / "swa_model.pt")
                torch.save({"model_state": _swa_model.module.state_dict()}, _swa_path)
                self.logger.info(f"GEN-03: SWA model saved -> {_swa_path}")
                print(f"  SWA model saved -> {_swa_path}", flush=True)
                # Evaluate SWA model on val set
                _orig_state = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(_swa_model.module.state_dict())
                _, _swa_val_metrics = self._eval_epoch(
                    self.val_loader, epoch=cfg.epochs, split="val"
                )
                _swa_score = _swa_val_metrics.get(self._resolve_metric_key(cfg.early_stop_metric), 0.0)
                _best_score_for_compare = best_score if best_metrics else -float("inf")
                if _swa_score > _best_score_for_compare:
                    self.logger.info(
                        f"GEN-02: SWA model outperforms best checkpoint "
                        f"({_swa_score:.4f} > {_best_score_for_compare:.4f}) — "
                        f"replacing best_model.pt"
                    )
                    print(
                        f"\n  ✓ SWA Model is BEST: {cfg.early_stop_metric}={_swa_score:.4f} "
                        f"(prev best={_best_score_for_compare:.4f}) — saved as best_model.pt",
                        flush=True,
                    )
                    best_metrics = _swa_val_metrics
                    self._save_checkpoint("best_model.pt", cfg.epochs, _swa_val_metrics)
                else:
                    # Restore original best model weights
                    self.model.load_state_dict(_orig_state)
                    print(
                        f"\n  SWA score ({_swa_score:.4f}) \u2264 best checkpoint "
                        f"({_best_score_for_compare:.4f}) — keeping best_model.pt unchanged.",
                        flush=True,
                    )
            except Exception as _swa_final_err:
                self.logger.warning(
                    f"GEN-03: SWA final step failed (non-fatal): {_swa_final_err}"
                )

        # BUG-C03 FIX: build RF stacker after neural training completes
        if cfg.use_rf_stacker:
            self._build_rf_stacker()

        print_training_complete(best_metrics)
        return best_metrics


    # ------------------------------------------------------------------
    def _build_rf_stacker(self) -> None:
        """Fit and save RF stacker after neural training."""
        assert self.model is not None
        cfg    = self.cfg
        logger = self.logger

        best_ckpt = self._ckpt_dir / "best_model.pt"
        if best_ckpt.exists():
            self._load_checkpoint(str(best_ckpt))
            logger.info("RF Stacker: loaded best_model.pt weights")

        try:
            amp_dtype_str = cfg.amp_dtype if cfg.use_amp else "float32"
            use_amp       = cfg.use_amp and self.device.type == "cuda"
            if amp_dtype_str == "float16":
                amp_dtype_t = torch.float16
            elif amp_dtype_str in ("bfloat16", "bf16"):
                amp_dtype_t = torch.bfloat16
            else:
                amp_dtype_t = None
                use_amp     = False

            save_path = str(self._ckpt_dir / "rf_stacker.joblib")
            stacker = build_rf_stacker(
                model         = self.model,
                train_dataset = self._train_ds,   # Subset — F01 fix handles it
                val_dataset   = self._val_ds,     # Subset — F01 fix handles it
                collate_fn    = protac_collate_fn,
                device        = self.device,
                batch_size    = cfg.batch_size,
                use_amp       = use_amp,
                amp_dtype     = amp_dtype_t,
                n_estimators  = cfg.rf_n_estimators,
                fp_bits       = cfg.rf_fp_bits,
                max_depth     = cfg.rf_max_depth,    # V38-FIX: pass through rf_max_depth
                save_path     = save_path,
            )
            self._rf_stacker = stacker
            logger.info(f"RF stacker built and saved -> {save_path}")
        except Exception as exc:
            logger.warning(
                f"RF stacker build failed (non-fatal, neural model unaffected): {exc}"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_combined_score(metrics: Dict[str, float]) -> float:
        """V37-03: Combined score for checkpoint selection.

        COMBINED_SCORE = 0.40*AUROC + 0.20*AUPRC + 0.15*MCC_norm + 0.15*F1 + 0.10*ACC
        MCC is in [-1, 1] — normalized to [0, 1] via (mcc + 1) / 2

        This replaces AUROC-only selection (V37-BUG-03 fix).
        """
        auroc = float(metrics.get("auroc", 0.0))
        auprc = float(metrics.get("auprc", 0.0))
        mcc   = float(metrics.get("mcc",   0.0))
        f1    = float(metrics.get("f1",    0.0))
        acc   = float(metrics.get("acc",   metrics.get("accuracy", 0.0)))
        mcc_norm = (mcc + 1.0) / 2.0   # normalize [-1,1] → [0,1]
        return (0.40 * auroc + 0.20 * auprc + 0.15 * mcc_norm
                + 0.15 * f1 + 0.10 * acc)

    def _build_architecture_fingerprint(self, epoch: int, dataset_path: str = "") -> Dict:
        """V37-07: Build complete architecture fingerprint for checkpointing."""
        cfg = self.cfg
        fp: Dict[str, Any] = {
            "version":           "3.7",
            "backend":           cfg.backend,
            "use_alphafold":     bool(getattr(cfg, "alphafold_dir", "")),
            "af_extra_dim":      getattr(cfg, "af_extra_dim", 0),
            "use_rf":            getattr(cfg, "use_rf_stacker", True),
            "use_esm":           True,
            "use_goss":          getattr(cfg, "use_goss", True),
            "fusion_dim":        cfg.fusion_dim,
            "graph_hidden":      cfg.graph_hidden,
            "num_graph_layers":  cfg.num_graph_layers,
            "num_heads":         cfg.num_heads,
            "dropout":           cfg.dropout,
            "focal_alpha":       getattr(cfg, "focal_alpha", 0.5),
            "focal_gamma":       getattr(cfg, "focal_gamma", 2.0),
            "saved_epoch":       epoch,
            "timestamp":         datetime.now().isoformat(),
        }
        # Config hash (from all hyperparameters)
        cfg_dict = asdict(cfg)
        fp["config_hash"] = hashlib.md5(
            json.dumps(cfg_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        # Dataset hash (if path provided)
        if dataset_path and os.path.isfile(dataset_path):
            try:
                with open(dataset_path, "rb") as fh:
                    fp["dataset_hash"] = hashlib.md5(fh.read()).hexdigest()[:12]
            except Exception:
                fp["dataset_hash"] = "unknown"
        return fp

    def _save_checkpoint(self, name: str, epoch: int, metrics: Dict) -> None:
        """Save checkpoint with epoch, metrics, config, and architecture fingerprint.

        V37-03: combined_score added to metrics
        V37-07: architecture_fingerprint stored in every checkpoint
        BUG-3 FIX: best_model.pt is ONLY saved when val metric improves.
        """
        # V37-03: compute combined score and add to metrics
        combined = self._compute_combined_score(metrics)
        metrics_with_combined = dict(metrics)
        metrics_with_combined["combined_score"] = combined

        # V37-07: architecture fingerprint
        fingerprint = self._build_architecture_fingerprint(epoch)

        state = {
            "epoch":                epoch,
            "model_state":          self.model.state_dict(),
            "optimizer":            self.optimizer.state_dict(),
            "metrics":              metrics_with_combined,
            "cfg":                  asdict(self.cfg),
            "best_epoch":           epoch,
            "best_metrics":         metrics_with_combined,
            "architecture_fingerprint": fingerprint,  # V37-07
            "combined_score":       combined,          # V37-03
        }
        if self.ema:
            state["ema_state"] = self.ema.state_dict()
        path = self._ckpt_dir / name
        torch.save(state, str(path))
        self.logger.info(
            f"Checkpoint saved -> {path}  "
            f"(epoch={epoch}, "
            + ", ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
                if isinstance(v, float)
            ) + ")"
        )

    def _load_checkpoint(self, path: str) -> None:
        """Load model weights from checkpoint.

        BUG-D   FIX: weights_only=False required for custom objects in state.
        CKPT-BUG-01 FIX: If the checkpoint was saved with different architecture
        dims (fusion_dim, num_graph_layers, graph_hidden, num_heads), rebuild the
        model using those saved dims before loading weights.  This allows a
        checkpoint trained with reduced test dims (e.g. fusion_dim=64) to be
        loaded at inference time without a RuntimeError size mismatch.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # CKPT-BUG-01: reconcile architecture dims from checkpoint cfg
        saved_cfg: dict = ckpt.get("cfg", {})
        _arch_keys = ("fusion_dim", "num_graph_layers", "graph_hidden",
                      "num_heads", "dropout", "backend", "esm_dim",
                      "stochastic_depth", "stochastic_depth_p",
                      "num_rbf", "cutoff",
                      "af_extra_dim", "use_goss", "goss_top_k")
        mismatch = any(
            saved_cfg.get(k) is not None and saved_cfg.get(k) != getattr(self.cfg, k, None)
            for k in _arch_keys
        )
        if mismatch and saved_cfg:
            self.logger.warning(
                "CKPT-BUG-01: checkpoint architecture differs from current cfg — "
                "rebuilding model from checkpoint dims: "
                + ", ".join(f"{k}={saved_cfg.get(k)}" for k in _arch_keys
                            if saved_cfg.get(k) is not None)
            )
            # Patch live cfg with saved dims
            # BUG-TRAINER-01 FIX: use setattr() for non-frozen dataclass,
            # object.__setattr__() is unnecessarily low-level here.
            for k in _arch_keys:
                if saved_cfg.get(k) is not None:
                    setattr(self.cfg, k, saved_cfg[k])
            # Rebuild model with correct dims
            self.model = build_from_config(
                self._make_model_cfg(self.cfg, getattr(self.cfg, "af_extra_dim", 0))
            ).to(self.device)
            # Rebuild EMA with new model (EMA defined in this same module)
            if self.ema is not None:
                self.ema = EMA(self.model, self.cfg.ema_decay)

        # V40-04: Checkpoint architecture protection
        _ckpt_arch = {k: saved_cfg.get(k) for k in _arch_keys if saved_cfg.get(k) is not None}
        _live_arch = {k: getattr(self.cfg, k, None) for k in _arch_keys}
        if not verify_checkpoint_arch(_ckpt_arch, _live_arch):
            self.logger.warning(
                "V40-04: CHECKPOINT ARCHITECTURE MISMATCH detected — "
                "model rebuilt from checkpoint dims to ensure compatibility."
            )

        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                pass  # optimizer state may not match after arch rebuild
        if self.ema and "ema_state" in ckpt:
            self.ema.shadow = {
                k: v.to(self.device) for k, v in ckpt["ema_state"].items()
            }
        self.logger.info(f"Resumed from {path} (epoch {ckpt.get('epoch', '?')})")

    # ------------------------------------------------------------------
    def evaluate(
        self, checkpoint: str, bootstrap_n: int = 1000
    ) -> Dict[str, float]:
        """Evaluate on test set with bootstrap confidence intervals.

        BUG-H04 FIX: guard against calling before setup().
        """
        assert self.model is not None, "Call setup() before evaluate()"
        self._load_checkpoint(checkpoint)

        print("\n  Running test-set evaluation...", flush=True)
        y_true, y_score = self._collect_preds(self.test_loader, split="test")
        results = bootstrap_ci(y_true, y_score, n=bootstrap_n)
        print_test_results(results)

        # V40-PHASE-12: Calibration metrics
        try:
            from core.utils import compute_ece, compute_brier_score, save_reliability_diagram
            _ece   = compute_ece(y_true, y_score)
            _brier = compute_brier_score(y_true, y_score)
            print(f"\n  Calibration Metrics:")
            print(f"    ECE   (Expected Calibration Error): {_ece:.4f}  [lower is better]")
            print(f"    Brier Score:                        {_brier:.4f}  [lower is better]")
            import os as _os
            _os.makedirs("reports", exist_ok=True)
            _diag_path = save_reliability_diagram(y_true, y_score, "reports/reliability_diagram.txt")
            print(f"    Reliability diagram saved → {_diag_path}")
            results["ece"]   = _ece
            results["brier"] = _brier
        except Exception as _cal_err:
            self.logger.warning(f"V40-PHASE-12: calibration metrics failed (non-fatal): {_cal_err}")

        return results

    def _collect_preds(
        self, loader: DataLoader, split: str = "val"
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.model is not None
        self.model.eval()
        all_logits, all_labels = [], []

        pbar = test_bar(loader)
        with torch.no_grad():
            for batch in pbar:
                batch = self._move_batch(batch)
                main, _, _ = self._forward(batch)
                all_logits.append(main.cpu().float().numpy())
                all_labels.append(batch["labels"].cpu().numpy())
        try:
            pbar.close()
        except Exception:
            pass

        return (
            np.concatenate(all_labels),
            1 / (1 + np.exp(-np.concatenate(all_logits))),
        )

    # ------------------------------------------------------------------
    def predict(
        self,
        dataset: Dataset,
        checkpoint: Optional[str] = None,
        calibrator=None,
    ) -> np.ndarray:
        """Run inference; ensemble with RF stacker if available.

        F02 FIX: ``checkpoint=None`` means "use currently loaded weights".
        When app.py calls this after ``_init_model()`` has already loaded
        the checkpoint, passing ``checkpoint=None`` avoids the double load.

        Parameters
        ----------
        dataset    : PROTACDataset (or compatible)
        checkpoint : path to .pt file, or None to use weights already in model
        calibrator : optional TemperatureScaling instance
        """
        assert self.model is not None, "Call setup() before predict()"

        # F02 FIX: only load if explicitly requested
        if checkpoint is not None:
            self._load_checkpoint(checkpoint)

        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size,
            collate_fn=protac_collate_fn, num_workers=0,
        )
        self.model.eval()
        all_logits: list = []
        all_stab:   list = []
        all_inter:  list = []

        pbar = test_bar(loader)
        with torch.no_grad():
            for batch in pbar:
                batch = self._move_batch(batch)
                main, stab, inter = self._forward(batch)
                if calibrator is not None:
                    main = calibrator(main)
                all_logits.append(torch.sigmoid(main).cpu().float().numpy())
                all_stab.append(torch.sigmoid(stab).cpu().float().numpy())
                all_inter.append(torch.sigmoid(inter).cpu().float().numpy())
        try:
            pbar.close()
        except Exception:
            pass

        neural_probs_main = np.concatenate(all_logits)

        # Try RF stacker ensemble
        stacker_path = str(self._ckpt_dir / "rf_stacker.joblib")
        if os.path.exists(stacker_path):
            try:
                stacker   = RFStackerResult.load(stacker_path)
                neural_3  = np.stack([
                    neural_probs_main,
                    np.concatenate(all_stab),
                    np.concatenate(all_inter),
                ], axis=1)
                rf_probs, _ = stacker.predict(neural_3, dataset)
                final_probs = 0.5 * neural_probs_main + 0.5 * rf_probs
                self.logger.info(
                    f"RF stacker ensemble applied (loaded from {stacker_path})"
                )
                return final_probs
            except Exception as exc:
                self.logger.warning(
                    f"RF stacker load/apply failed (using neural only): {exc}"
                )

        return neural_probs_main

    # ------------------------------------------------------------------
    def calibrate(self) -> TemperatureScaling:
        """Fit temperature scaling on validation set.

        BUG-H04 FIX: guard against calling before setup().
        """
        assert self.model is not None, "Call setup() before calibrate()"
        self.model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch(batch)
                main, _, _ = self._forward(batch)
                logits_list.append(main.cpu())
                labels_list.append(batch["labels"].cpu())
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        cal = TemperatureScaling()
        t   = cal.fit(logits, labels)
        self.logger.info(f"Temperature scaling fitted: T={t:.4f}")
        return cal
