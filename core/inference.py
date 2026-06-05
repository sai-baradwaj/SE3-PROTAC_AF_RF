"""
core/inference.py
=================
Standalone inference helpers for SE3AF.

This module provides:
  - ``load_model_for_inference()``  — loads a trained checkpoint into an
    SE3AFModel ready for prediction (no DataLoader / trainer overhead).
  - ``predict_from_smiles()``       — single-molecule prediction from raw SMILES.
  - ``predict_from_csv()``          — batch prediction from a CSV file.
  - ``predict_from_dataset()``      — batch prediction from a PROTACDataset.

These helpers are used by:
  - ``predict.py`` standalone CLI entrypoint
  - ``app.py`` Flask server (via ``load_model_for_inference`` + direct forward)

F02 FIX (double checkpoint load)
---------------------------------
The original app.py called ``trainer.predict(ds, checkpoint=str(CHECKPOINT))``,
which loaded the checkpoint again even though ``_init_model()`` had already
called ``torch.load`` + ``model.load_state_dict()``.

In the new app.py, we:
  1. Call ``load_model_for_inference(checkpoint)`` once at startup → returns
     (model, device) with weights already loaded.
  2. At request time, call the model directly via ``predict_from_smiles()``
     or ``predict_from_dataset()`` — no second torch.load.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.dataset import PROTACDataset, protac_collate_fn
from core.model import SE3AFModel, build_from_config
from core.trainer import RFStackerResult


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_for_inference(
    checkpoint_path: str,
    model_cfg: Optional[Dict] = None,
    device: Optional[str] = None,
) -> Tuple[SE3AFModel, torch.device]:
    """Load a trained SE3AF checkpoint and return an eval-mode model.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` checkpoint file produced by :class:`SE3AFTrainer`.
    model_cfg : dict | None
        Override model architecture settings.  If None, settings are read
        from the checkpoint ``cfg`` dict.
    device : str | None
        Target device (``"cuda"`` / ``"cpu"``).  Auto-detected if None.

    Returns
    -------
    (model, device)
        ``model`` is already in eval mode with weights applied.
    """
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    if model_cfg is None:
        # Read architecture from stored config
        saved_cfg = ckpt.get("cfg", {})
        train_backend = saved_cfg.get("backend", "lite")
        # V37-FIX: Also read af_extra_dim from checkpoint so model architecture
        # matches exactly — previously this was missing, causing shape mismatch
        # when AlphaFold was enabled during training (esm_dim=1280+af_extra_dim=4).
        # Also try reading from architecture_fingerprint for robustness.
        fp = ckpt.get("architecture_fingerprint", {})
        af_extra_dim = saved_cfg.get(
            "af_extra_dim",
            fp.get("af_extra_dim", 0)
        )
        esm_dim_base = saved_cfg.get("esm_dim", 1280)
        # Effective esm_dim = base + af_extra_dim (as appended by AlphaFoldStore)
        esm_dim_effective = esm_dim_base + af_extra_dim
        model_cfg = {
            "backend":          train_backend,
            "fusion_dim":       saved_cfg.get("fusion_dim",       256),
            "esm_dim":          esm_dim_effective,
            "graph_hidden":     saved_cfg.get("graph_hidden",     256),
            "num_graph_layers": saved_cfg.get("num_graph_layers", 6),
            "num_heads":        saved_cfg.get("num_heads",        8),
            "dropout":          0.0,   # inference mode: no dropout
            # V37-FIX: pass use_goss so GOSS log-weights register in model state
            "use_goss":         saved_cfg.get("use_goss", fp.get("use_goss", True)),
        }
        import warnings
        warnings.warn(
            f"[inference] Loading checkpoint with backend='{train_backend}'. "
            "Training and inference use the same backend (BUG-6 fix: no mismatch).",
            stacklevel=2,
        )
    else:
        # BUG-6 FIX: Detect training/inference backend mismatch
        saved_cfg = ckpt.get("cfg", {})
        train_backend = saved_cfg.get("backend", "lite")
        infer_backend = model_cfg.get("backend", "lite")
        if train_backend and infer_backend and train_backend != infer_backend:
            import warnings
            warnings.warn(
                f"[inference] BUG-6 WARNING: Training backend='{train_backend}' "
                f"differs from inference backend='{infer_backend}'. "
                "This may cause incorrect predictions. "
                "Ensure training and inference use the same backend.",
                stacklevel=2,
            )

    model = build_from_config(model_cfg).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, dev


# ---------------------------------------------------------------------------
# Single-molecule inference
# ---------------------------------------------------------------------------

def predict_from_smiles(
    model: SE3AFModel,
    device: torch.device,
    smiles_target: str,
    smiles_e3: str,
    smiles_linker: str,
    target_seq: str = "",
    ligase_seq: str = "",
    cache_dir: str = ".cache",
) -> Dict[str, float]:
    """Predict PROTAC activity from three SMILES strings.

    Creates an in-memory single-row dataset, runs a forward pass, and
    returns a result dict with probability scores.

    Parameters
    ----------
    model        : SE3AFModel in eval mode (loaded by load_model_for_inference)
    device       : torch.device
    smiles_*     : SMILES strings for target ligand, E3 ligand, and linker
    target_seq   : optional target protein sequence
    ligase_seq   : optional E3 ligase sequence
    cache_dir    : cache directory (for graph + ESM caches)

    Returns
    -------
    dict with keys: probability, active, stability_prob, interaction_prob
    """
    row = {
        "smiles_target_lig": smiles_target,
        "smiles_e3_lig":     smiles_e3,
        "smiles_linker":     smiles_linker,
        "target_seq":        target_seq or "",
        "ligase_seq":        ligase_seq or "",
    }

    # Write to a temp CSV
    # BUG-INFER-01 FIX: do NOT delete the temp file inside a try/finally that
    # wraps dataset construction — PROTACDataset.discover_datasets() needs the
    # file to exist during construction.  Delete AFTER the dataset is built.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    try:
        pd.DataFrame([row]).to_csv(tmp_path, index=False)
        os.close(tmp_fd)
        ds = PROTACDataset(data_root=tmp_path, cache_dir=cache_dir)
    except Exception:
        raise
    finally:
        # Always clean up — dataset has already read/copied data into memory
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except (OSError, TypeError):
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    probs_dict = _run_inference(model, device, ds, batch_size=1)

    return {
        "probability":      float(probs_dict["main"][0]),
        "active":           bool(probs_dict["main"][0] >= 0.5),
        "stability_prob":   float(probs_dict["stab"][0]),
        "interaction_prob": float(probs_dict["inter"][0]),
    }


# ---------------------------------------------------------------------------
# CSV-based batch inference
# ---------------------------------------------------------------------------

def predict_from_csv(
    model: SE3AFModel,
    device: torch.device,
    csv_path: str,
    cache_dir: str = ".cache",
    batch_size: int = 32,
    rf_stacker_path: Optional[str] = None,
) -> np.ndarray:
    """Run batch inference on a CSV file.

    Parameters
    ----------
    model           : SE3AFModel in eval mode
    device          : torch.device
    csv_path        : path to input CSV
    cache_dir       : cache directory
    batch_size      : inference batch size
    rf_stacker_path : optional path to ``rf_stacker.joblib``; if provided
                      and the file exists, ensembles neural + RF probabilities.

    Returns
    -------
    np.ndarray of shape (N,) — activity probabilities in [0, 1]
    """
    ds = PROTACDataset(data_root=csv_path, cache_dir=cache_dir)
    return predict_from_dataset(
        model=model,
        device=device,
        dataset=ds,
        batch_size=batch_size,
        rf_stacker_path=rf_stacker_path,
    )


# ---------------------------------------------------------------------------
# Dataset-based batch inference (also used internally by app.py)
# ---------------------------------------------------------------------------

def predict_from_dataset(
    model: SE3AFModel,
    device: torch.device,
    dataset: PROTACDataset,
    batch_size: int = 32,
    rf_stacker_path: Optional[str] = None,
) -> np.ndarray:
    """Run batch inference on a PROTACDataset.

    F02 FIX: No checkpoint loading here — caller is responsible for calling
    ``load_model_for_inference()`` or otherwise ensuring weights are loaded.

    Parameters
    ----------
    model            : SE3AFModel in eval mode
    device           : torch.device
    dataset          : PROTACDataset (already constructed)
    batch_size       : inference batch size
    rf_stacker_path  : optional path to ``rf_stacker.joblib``

    Returns
    -------
    np.ndarray of shape (N,) — activity probabilities
    """
    probs_dict = _run_inference(model, device, dataset, batch_size=batch_size)
    neural_main = probs_dict["main"]   # (N,)

    # Optional RF stacker ensemble
    if rf_stacker_path and os.path.exists(rf_stacker_path):
        try:
            stacker  = RFStackerResult.load(rf_stacker_path)
            neural_3 = np.stack([
                neural_main,
                probs_dict["stab"],
                probs_dict["inter"],
            ], axis=1)
            rf_probs, _ = stacker.predict(neural_3, dataset)
            return 0.5 * neural_main + 0.5 * rf_probs
        except Exception as exc:
            import warnings
            warnings.warn(f"[inference] RF stacker failed, using neural only: {exc}")

    return neural_main


# ---------------------------------------------------------------------------
# Internal runner
# ---------------------------------------------------------------------------

def _run_inference(
    model: SE3AFModel,
    device: torch.device,
    dataset: PROTACDataset,
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """Run model forward pass over a full dataset.

    Returns dict with keys ``main``, ``stab``, ``inter`` — each shape (N,).

    V37-FIX: Automatically pads ESM embeddings with zeros when the model was
    trained with AlphaFold features (af_extra_dim > 0) but inference data
    has no AF features available (e.g. ad-hoc SMILES without UniProt ID).
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=protac_collate_fn, num_workers=0,
    )

    # Detect model's expected ESM dim (from esm_proj weight shape)
    expected_esm_dim: int = 1280
    try:
        esm_proj_w = model.fusion.esm_proj.weight   # shape [fusion_dim, esm_dim]
        expected_esm_dim = esm_proj_w.shape[1]
    except AttributeError:
        pass

    all_main:  list = []
    all_stab:  list = []
    all_inter: list = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            # Move to device
            dev_batch = {
                k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
                for k, v in batch.items()
            }
            # V37-FIX: Pad ESM tensors if model expects AF-augmented dim
            tgt_esm = dev_batch["tgt_esm"]
            e3_esm  = dev_batch["e3_esm"]
            if tgt_esm.shape[-1] < expected_esm_dim:
                pad_size = expected_esm_dim - tgt_esm.shape[-1]
                tgt_esm = torch.cat(
                    [tgt_esm, torch.zeros(*tgt_esm.shape[:-1], pad_size,
                                         dtype=tgt_esm.dtype, device=device)], dim=-1)
            if e3_esm.shape[-1] < expected_esm_dim:
                pad_size = expected_esm_dim - e3_esm.shape[-1]
                e3_esm = torch.cat(
                    [e3_esm, torch.zeros(*e3_esm.shape[:-1], pad_size,
                                        dtype=e3_esm.dtype, device=device)], dim=-1)
            main, stab, inter = model(
                tgt_graph=dev_batch["tgt_graph"],
                e3_graph=dev_batch["e3_graph"],
                lnk_graph=dev_batch["lnk_graph"],
                tgt_esm=tgt_esm,
                e3_esm=e3_esm,
            )
            all_main.append(torch.sigmoid(main).cpu().float().numpy())
            all_stab.append(torch.sigmoid(stab).cpu().float().numpy())
            all_inter.append(torch.sigmoid(inter).cpu().float().numpy())

    return {
        "main":  np.concatenate(all_main),
        "stab":  np.concatenate(all_stab),
        "inter": np.concatenate(all_inter),
    }
