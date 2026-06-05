"""
core/ui.py
==========
SE3AF Professional Training UI
================================
Provides a PyTorch Lightning / HuggingFace Trainer style output experience.

Features
--------
- Startup banner with hardware/config summary
- tqdm progress bars for training, validation, and test evaluation
- Live loss / LR / ETA / iteration-speed display
- Rich-compatible fallback to plain terminal
- Zero dependency on external libraries beyond tqdm

v3.4.0 NEW ADDITIONS
--------------------
UI07  FEATURE: print_overfit_warning()    — formatted WARNING box for overfitting detection
UI08  FEATURE: print_checkpoint_update()  — display prev/new AUROC/F1/MCC on best_model.pt save
UI09  FEATURE: print_alphafold_status()   — AlphaFold structures found/missing banner
UI10  FEATURE: print_backend_status()     — Lite3DEncoder / SE3 backend startup message
UI11  FEATURE: print_health_monitor()     — 10-metric per-epoch health table
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

# Try tqdm; graceful no-op fallback if missing
try:
    from tqdm import tqdm as _tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False
    _tqdm = None  # type: ignore[misc]

# ── Terminal width helpers ──────────────────────────────────────────────────

def _term_width(default: int = 80) -> int:
    try:
        return os.get_terminal_size().columns
    except Exception:
        return default


def _box(text: str, char: str = "=", width: int = 0) -> str:
    w = width or max(_term_width(), len(text) + 4)
    return char * w + "\n" + text + "\n" + char * w


def _header(title: str, sub: str = "", width: int = 56) -> str:
    lines = ["=" * width, title]
    if sub:
        lines.append("-" * len(sub))
        lines.append(sub)
    lines.append("=" * width)
    return "\n".join(lines)


def _kv_table(rows: List[tuple], indent: int = 2) -> str:
    if not rows:
        return ""
    max_k = max(len(str(k)) for k, _ in rows)
    pad = " " * indent
    return "\n".join(f"{pad}{str(k):<{max_k+1}}: {v}" for k, v in rows)


# ── Hardware detection ──────────────────────────────────────────────────────

def _gpu_info() -> Dict[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            vram_gb = prop.total_memory / (1024 ** 3)
            return {
                "gpu_name":  prop.name,
                "vram_gb":   f"{vram_gb:.1f} GB",
                "device":    "cuda",
            }
    except Exception:
        pass
    return {"gpu_name": "N/A (CPU only)", "vram_gb": "N/A", "device": "cpu"}


def _count_params(model) -> str:
    try:
        n = sum(p.numel() for p in model.parameters())
        if n >= 1_000_000:
            return f"{n/1_000_000:.2f}M ({n:,})"
        return f"{n:,}"
    except Exception:
        return "unknown"


# ── Startup banner ──────────────────────────────────────────────────────────

def print_startup_banner(cfg, dataset_stats: Optional[Dict] = None,
                          model=None) -> None:
    """Print a professional startup banner to stdout.

    Parameters
    ----------
    cfg   : TrainerConfig-like object
    dataset_stats : optional dict with train/val/test row counts and dup counts
    model : optional nn.Module (for parameter count)
    """
    gpu  = _gpu_info()
    w    = 58

    banner_lines = [
        "",
        "=" * w,
        "  SE3AF  ─  SE(3)-Equivariant PROTAC Activity Predictor",
        "  SE3AF Model with Data Cleaning + SE(3) Backend        ",
        "=" * w,
        "",
    ]

    # ── Data summary ──
    if dataset_stats:
        banner_lines += [
            "  Preparing leak-resistant splits...",
            "  Loading datasets...",
            "",
            "  Data Summary:",
        ]
        for k, v in dataset_stats.items():
            banner_lines.append(f"    {k:<30}: {v}")
        banner_lines.append("")

    # ── Protein embedder / backend ──
    banner_lines += [
        "  Loading protein embedder...",
        "  Loading graph backend...",
        "",
    ]

    # ── Hardware / config ──
    device_str = gpu["device"].upper()
    banner_lines += [
        f"  Device                        : {device_str}",
        f"  GPU                           : {gpu['gpu_name']}",
        f"  VRAM                          : {gpu['vram_gb']}",
    ]

    amp_on = getattr(cfg, "use_amp", False)
    amp_dt = getattr(cfg, "amp_dtype", "float32") if amp_on else "disabled"

    try:
        bs   = cfg.batch_size
        ga   = getattr(cfg, "grad_accum_steps", 1)
        lr   = cfg.learning_rate
        eps  = cfg.epochs
        pat  = cfg.early_stop_patience
        fdim = cfg.fusion_dim
        nh   = cfg.num_heads
        dr   = cfg.dropout
        wd   = cfg.weight_decay
        gc   = cfg.grad_clip
    except Exception:
        bs = ga = lr = eps = pat = fdim = nh = dr = wd = gc = "?"

    banner_lines += [
        f"  Batch Size                    : {bs}",
        f"  Gradient Accumulation         : {ga}",
        f"  Learning Rate                 : {lr}",
        f"  Epochs                        : {eps}",
        f"  Early Stopping Patience       : {pat} epochs",
        f"  Model Dimension (fusion_dim)  : {fdim}",
        f"  Attention Heads               : {nh}",
        f"  Dropout                       : {dr}",
        f"  Weight Decay                  : {wd}",
        "",
        "=" * w,
    ]

    print("\n".join(banner_lines), flush=True)


def print_training_header(cfg, device_str: str, model=None) -> None:
    """Print the block that appears just before the training loop starts."""
    amp_on  = getattr(cfg, "use_amp", False)
    amp_dt  = getattr(cfg, "amp_dtype", "float32") if amp_on else "N/A"
    gc      = getattr(cfg, "grad_clip", 1.0)
    ga      = getattr(cfg, "focal_alpha", 0.25)
    scaler  = "enabled" if amp_on else "disabled"
    params  = _count_params(model) if model is not None else "?"

    lines = [
        "",
        f"{'=' * 58}",
        f"  Starting training on {device_str.upper()}...",
        f"{'=' * 58}",
        "",
        f"  Mixed Precision AMP           : {'ON (' + amp_dt + ')' if amp_on else 'OFF'}",
        f"  Gradient Scaler               : {scaler}",
        f"  Max Gradient Norm             : {gc}",
        f"  Focal Loss Alpha              : {ga}",
        f"  Parameter Count               : {params}",
        "",
    ]
    print("\n".join(lines), flush=True)


# ── v3.4.0 NEW: Backend status display ─────────────────────────────────────

def print_backend_status(display_name: str) -> None:
    """Print the encoder backend status line during startup.

    Parameters
    ----------
    display_name : str returned by get_encoder_display_name()
        e.g. 'Lite3DEncoder active — distance-bias 3D attention ENABLED'
             'SE3GraphTransformer active — coordinate-based message passing ENABLED'
    """
    w = 58
    lines = [
        "",
        "  " + "─" * 54,
        f"  Graph Encoder Backend:",
        f"    ▶  {display_name}",
        "  " + "─" * 54,
        "",
    ]
    print("\n".join(lines), flush=True)


# ── v3.4.0 NEW: AlphaFold status display ───────────────────────────────────

def print_alphafold_status(summary_str: str) -> None:
    """Print AlphaFold verification block during startup.

    Parameters
    ----------
    summary_str : multi-line string from AlphaFoldStore.summary()
        Contains 'AlphaFold Structures Found: X',
                 'AlphaFold Structures Missing: Y',
                 'AlphaFold Features Enabled: True/False'
    """
    w = 58
    lines_in = summary_str.strip().splitlines()
    lines = [
        "",
        "  " + "─" * 54,
        "  AlphaFold Structural Features:",
    ]
    for line in lines_in:
        lines.append(f"    {line.strip()}")
    lines += [
        "  " + "─" * 54,
        "",
    ]
    print("\n".join(lines), flush=True)


# ── v3.4.0 NEW: Overfitting warning ────────────────────────────────────────

def print_overfit_warning(
    reason: str,
    train_loss: float,
    val_loss: float,
    suggested_actions: Optional[List[str]] = None,
) -> None:
    """Print a formatted overfitting warning box.

    Parameters
    ----------
    reason  : human-readable explanation
    train_loss : current epoch training loss
    val_loss   : current epoch validation loss
    suggested_actions : list of suggested remediation steps
    """
    w = 58
    if suggested_actions is None:
        suggested_actions = [
            "Increase dropout rate (current may be too low)",
            "Increase weight decay for stronger L2 regularization",
            "Reduce model capacity (fewer layers or smaller hidden_dim)",
            "Add data augmentation (SMILES enumeration / noise injection)",
            "Reduce batch size to add gradient noise",
            "Enable or tighten early stopping",
        ]

    lines = [
        "",
        "=" * w,
        "  ⚠  WARNING: OVERFITTING DETECTED",
        "=" * w,
        "",
        f"  Reason:",
        f"    {reason}",
        "",
        f"  Train Loss : {train_loss:.4f}",
        f"  Val Loss   : {val_loss:.4f}",
        "",
        "  Suggested Actions:",
    ]
    for action in suggested_actions:
        lines.append(f"    • {action}")
    lines += [
        "",
        "=" * w,
        "",
    ]
    print("\n".join(lines), flush=True)


# ── v3.4.0 NEW: Checkpoint validation display ──────────────────────────────

def print_checkpoint_update(
    prev_metrics: Dict[str, float],
    new_metrics: Dict[str, float],
    epoch: int,
    reason: str = "Validation metric improved",
) -> None:
    """Print checkpoint save notification with metric comparison.

    Parameters
    ----------
    prev_metrics : metrics before this save (or empty dict on first save)
    new_metrics  : current epoch metrics
    epoch        : current epoch number
    reason       : why the checkpoint was saved
    """
    w = 58

    prev_auroc = prev_metrics.get("auroc", 0.0)
    new_auroc  = new_metrics.get("auroc",  0.0)
    prev_f1    = prev_metrics.get("f1",    0.0)
    new_f1     = new_metrics.get("f1",     0.0)
    prev_mcc   = prev_metrics.get("mcc",   0.0)
    new_mcc    = new_metrics.get("mcc",    0.0)
    prev_auprc = prev_metrics.get("auprc", 0.0)
    new_auprc  = new_metrics.get("auprc",  0.0)

    def _delta(a: float, b: float) -> str:
        d = b - a
        return f"{'+' if d >= 0 else ''}{d:.4f}"

    is_first = not bool(prev_metrics)

    lines = [
        "",
        "  " + "─" * 54,
        f"  ✓ New Best Model Found  (Epoch {epoch})",
        "  " + "─" * 54,
        "",
    ]

    if is_first:
        lines += [
            f"    First checkpoint — baseline established",
            f"    AUROC  : {new_auroc:.4f}",
            f"    AUPRC  : {new_auprc:.4f}",
            f"    F1     : {new_f1:.4f}",
            f"    MCC    : {new_mcc:.4f}",
        ]
    else:
        lines += [
            f"    {'Metric':<12}  {'Previous':>10}  {'New':>10}  {'Delta':>10}",
            f"    {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}",
            f"    {'AUROC':<12}  {prev_auroc:>10.4f}  {new_auroc:>10.4f}  {_delta(prev_auroc, new_auroc):>10}",
            f"    {'AUPRC':<12}  {prev_auprc:>10.4f}  {new_auprc:>10.4f}  {_delta(prev_auprc, new_auprc):>10}",
            f"    {'F1':<12}  {prev_f1:>10.4f}  {new_f1:>10.4f}  {_delta(prev_f1, new_f1):>10}",
            f"    {'MCC':<12}  {prev_mcc:>10.4f}  {new_mcc:>10.4f}  {_delta(prev_mcc, new_mcc):>10}",
        ]

    lines += [
        "",
        f"    Reason: {reason}",
        f"    Saved  : checkpoints/best_model.pt",
        "",
        "  " + "─" * 54,
        "",
    ]
    print("\n".join(lines), flush=True)


# ── v3.4.0 NEW: Model health monitor ───────────────────────────────────────

def print_health_monitor(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    """Print a full 10-metric health table after each epoch.

    Metrics covered
    ---------------
    Train Loss, Val Loss, Train Acc, Val Acc, AUROC, AUPRC, F1, MCC, Precision, Recall
    """
    w = 66
    lines = [
        "",
        f"  ╔{'═' * (w - 4)}╗",
        f"  ║  Model Health Monitor — Epoch {epoch}/{total_epochs:<4}{'':>{w-41}}║",
        f"  ╠{'═' * (w - 4)}╣",
        f"  ║  {'Metric':<22}  {'Train':>9}  {'Val':>9}  {'Delta':>8}  ║",
        f"  ╟{'─' * (w - 4)}╢",
    ]

    def _row(name: str, train_val: float, val_val: float) -> str:
        delta = val_val - train_val
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
        # Highlight suspicious delta (large gap = potential overfit)
        flag = "  ⚠" if delta < -0.15 else "   "
        return (
            f"  ║  {name:<22}  {train_val:>9.4f}  {val_val:>9.4f}"
            f"  {delta_str:>8}{flag}║"
        )

    # Loss rows (lower = better, so delta convention reversed for loss)
    train_loss_str = f"{train_loss:.4f}"
    val_loss_str   = f"{val_loss:.4f}"
    loss_delta     = val_loss - train_loss
    loss_delta_str = f"{'+' if loss_delta >= 0 else ''}{loss_delta:.4f}"
    loss_flag      = "  ⚠" if loss_delta > 0.15 else "   "
    lines.append(
        f"  ║  {'Train/Val Loss':<22}  {train_loss_str:>9}  {val_loss_str:>9}"
        f"  {loss_delta_str:>8}{loss_flag}║"
    )

    # Accuracy
    train_acc = train_metrics.get("acc", 0.0)
    val_acc   = val_metrics.get("acc", 0.0)
    lines.append(_row("Accuracy", train_acc, val_acc))

    # AUROC (val only from neural eval, train approx from train_metrics if available)
    train_auroc = train_metrics.get("auroc", float("nan"))
    val_auroc   = val_metrics.get("auroc", 0.0)
    if train_auroc != train_auroc:  # nan check
        lines.append(
            f"  ║  {'AUROC':<22}  {'  N/A':>9}  {val_auroc:>9.4f}  {'N/A':>8}   ║"
        )
    else:
        lines.append(_row("AUROC", train_auroc, val_auroc))

    # AUPRC
    train_auprc = train_metrics.get("auprc", float("nan"))
    val_auprc   = val_metrics.get("auprc", 0.0)
    if train_auprc != train_auprc:
        lines.append(
            f"  ║  {'AUPRC':<22}  {'  N/A':>9}  {val_auprc:>9.4f}  {'N/A':>8}   ║"
        )
    else:
        lines.append(_row("AUPRC", train_auprc, val_auprc))

    # F1
    train_f1 = train_metrics.get("f1", float("nan"))
    val_f1   = val_metrics.get("f1", 0.0)
    if train_f1 != train_f1:
        lines.append(
            f"  ║  {'F1 Score':<22}  {'  N/A':>9}  {val_f1:>9.4f}  {'N/A':>8}   ║"
        )
    else:
        lines.append(_row("F1 Score", train_f1, val_f1))

    # MCC
    train_mcc = train_metrics.get("mcc", float("nan"))
    val_mcc   = val_metrics.get("mcc", 0.0)
    if train_mcc != train_mcc:
        lines.append(
            f"  ║  {'MCC':<22}  {'  N/A':>9}  {val_mcc:>9.4f}  {'N/A':>8}   ║"
        )
    else:
        lines.append(_row("MCC", train_mcc, val_mcc))

    # Precision
    train_prec = train_metrics.get("precision", float("nan"))
    val_prec   = val_metrics.get("precision", 0.0)
    if train_prec != train_prec:
        lines.append(
            f"  ║  {'Precision':<22}  {'  N/A':>9}  {val_prec:>9.4f}  {'N/A':>8}   ║"
        )
    else:
        lines.append(_row("Precision", train_prec, val_prec))

    # Recall
    train_rec = train_metrics.get("recall", float("nan"))
    val_rec   = val_metrics.get("recall", 0.0)
    if train_rec != train_rec:
        lines.append(
            f"  ║  {'Recall':<22}  {'  N/A':>9}  {val_rec:>9.4f}  {'N/A':>8}   ║"
        )
    else:
        lines.append(_row("Recall", train_rec, val_rec))

    lines += [
        f"  ╚{'═' * (w - 4)}╝",
        "",
    ]
    print("\n".join(lines), flush=True)


# ── Progress bar wrappers ───────────────────────────────────────────────────

class _NoOpBar:
    """Dummy progress bar when tqdm is unavailable."""

    def __init__(self, iterable=None, **kw):
        self._it = iter(iterable) if iterable is not None else iter([])

    def __iter__(self):
        return self._it

    def __next__(self):
        return next(self._it)

    def set_postfix(self, *a, **kw):
        pass

    def set_description(self, *a, **kw):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _make_bar(
    iterable=None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    leave: bool = True,
    colour: Optional[str] = None,
    ncols: int = 100,
    **extra,
):
    """Create a tqdm bar or a no-op stub."""
    if not _HAVE_TQDM:
        return _NoOpBar(iterable)
    kw = dict(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        ncols=ncols,
        dynamic_ncols=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    if colour:
        kw["colour"] = colour
    kw.update(extra)
    if iterable is not None:
        return _tqdm(iterable, **kw)
    return _tqdm(**kw)


# ── v3.5.0: Architecture Visualization ─────────────────────────────────────

def print_architecture_visualization(backend: str = "lite") -> None:
    """Print ASCII architecture diagram before training starts.

    Phase 8 requirement: shows active backend encoder only.
    If backend='lite': shows Lite3DEncoder only.
    If backend='se3':  shows SE3 Encoder only.
    """
    w = 58
    if backend == "se3":
        graph_block = "SE3 Encoder (coordinate-based MP)"
    else:
        graph_block = "Lite3DEncoder (distance-bias 3D attention)"

    lines = [
        "",
        "=" * w,
        "  MODEL ARCHITECTURE",
        "=" * w,
        "",
        "  SMILES ×3",
        "     │",
        "     ▼",
        "  Graph Construction",
        "  (atom/bond featurisation → PyG Data)",
        "     │",
        "     ▼",
        f"  {graph_block}",
        "  (3 encoders: target_lig, e3_lig, linker)",
        "     │",
        "     ▼         ↑",
        "  CrossInteractionFusion   ESM-2 Protein Encoder",
        "  (C(5,2)=10 cross-attention pairs)  ↑",
        "     │               AlphaFold Features (opt.)",
        "     ▼",
        "  CLS + Pair Embeddings",
        "     │",
        "     ▼",
        "  Final Classifier (MLP → scalar logit)",
        "     │",
        "     ▼",
        "  Random Forest Stacker (neural+RF ensemble)",
        "     │",
        "     ▼",
        "  PROTAC Activity Probability ∈ [0, 1]",
        "",
        "=" * w,
        "",
    ]
    print("\n".join(lines), flush=True)


# ── Per-epoch display helpers ───────────────────────────────────────────────

def train_epoch_bar(loader, epoch: int, total_epochs: int):
    """Return a tqdm bar for one training epoch."""
    desc = f"  Epoch {epoch}/{total_epochs} [train]"
    return _make_bar(loader, desc=desc, unit="batch", colour="green")


def val_bar(loader, epoch: int, total_epochs: int):
    """Return a tqdm bar for one validation epoch."""
    desc = f"  Epoch {epoch}/{total_epochs} [val]  "
    return _make_bar(loader, desc=desc, unit="batch", colour="blue")


def test_bar(loader):
    """Return a tqdm bar for test-set evaluation."""
    return _make_bar(loader, desc="  Evaluating [test]         ", unit="batch", colour="cyan")


def rf_bar(steps: int, desc: str = "  RF Stacker"):
    """Return a tqdm bar for RF stacker steps."""
    return _make_bar(range(steps), desc=desc, unit="step", colour="yellow")


# ── Post-epoch summary ──────────────────────────────────────────────────────

def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_metrics: Dict[str, float],
    lr: float,
    elapsed: float,
    best_score: float,
    improved: bool,
) -> None:
    """Print a one-line epoch result summary."""
    auroc = val_metrics.get("auroc", 0.0)
    auprc = val_metrics.get("auprc", 0.0)
    acc   = val_metrics.get("acc",   0.0)
    f1    = val_metrics.get("f1",    0.0)
    mcc   = val_metrics.get("mcc",   0.0)
    prec  = val_metrics.get("precision", 0.0)
    rec   = val_metrics.get("recall",    0.0)
    flag  = " ✓ (best)" if improved else ""

    line = (
        f"  Epoch {epoch:>4}/{total_epochs}  "
        f"loss={train_loss:.4f}  "
        f"val_auroc={auroc:.4f}  "
        f"val_auprc={auprc:.4f}  "
        f"val_acc={acc:.4f}  "
        f"val_f1={f1:.4f}  "
        f"val_mcc={mcc:.4f}  "
        f"val_prec={prec:.4f}  "
        f"val_rec={rec:.4f}  "
        f"lr={lr:.2e}  "
        f"t={elapsed:.1f}s"
        f"{flag}"
    )
    print(line, flush=True)


def print_early_stop(epoch: int, patience: int, reason: Optional[str] = None) -> None:
    """Print early stopping message with optional detailed reason.

    BUG-4 FIX: Display exact stopping reason including metric name and values.
    """
    w = 58
    lines = [
        "",
        "=" * w,
        "  ⚠  EARLY STOPPING TRIGGERED",
        "=" * w,
        "",
        f"  Stopped at epoch: {epoch}",
        f"  Patience exhausted: {patience} epochs without improvement",
    ]
    if reason:
        lines.append("")
        lines.append("  Reason:")
        for r_line in reason.splitlines():
            lines.append(f"    {r_line.strip()}")
    lines += [
        "",
        "  Note: To continue training, increase early_stop_patience in train_config.json",
        "  Note: The best checkpoint is saved in checkpoints/best_model.pt",
        "",
        "=" * w,
        "",
    ]
    print("\n".join(lines), flush=True)


def print_rf_stacker_header() -> None:
    print(
        "\n"
        "  " + "─" * 54 + "\n"
        "  RF Stacker Training\n"
        "  " + "─" * 54,
        flush=True,
    )


def print_rf_stacker_result(val_auroc: float, threshold: float) -> None:
    print(
        f"  RF Stacker  Val AUROC={val_auroc:.4f}  "
        f"Optimal Threshold={threshold:.2f}",
        flush=True,
    )


def print_training_complete(best_metrics: Dict[str, float]) -> None:
    w = 58
    lines = [
        "",
        "=" * w,
        "  Training Complete",
        "=" * w,
        "",
    ]
    for k, v in best_metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<30}: {v:.4f}")
        else:
            lines.append(f"  {k:<30}: {v}")
    lines.append("")
    print("\n".join(lines), flush=True)


def print_test_results(metrics: Dict) -> None:
    w = 58
    lines = [
        "",
        "=" * w,
        "  Test Evaluation Results",
        "=" * w,
        "",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<30}: {v:.4f}")
        else:
            lines.append(f"  {k:<30}: {v}")
    lines.append("")
    print("\n".join(lines), flush=True)
