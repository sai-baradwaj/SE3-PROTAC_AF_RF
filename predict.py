#!/usr/bin/env python3
"""
predict.py
==========
SE3AF — Standalone prediction entrypoint.

This script provides a lightweight CLI for PROTAC activity prediction that
does NOT require a full trainer/dataset setup.  It uses the inference
helpers in ``core/inference.py`` (``load_model_for_inference``,
``predict_from_smiles``, ``predict_from_csv``, ``predict_from_dataset``)
which load a checkpoint once and expose a clean forward-pass API.

Unlike ``main.py predict``, this script:
  • Does NOT instantiate SE3AFTrainer or set up dataloaders.
  • Does NOT rebuild the feature cache.
  • Loads the model in a single torch.load call and re-uses it for all rows.
  • Exits cleanly with return code 0 (success) or 1 (error).

Usage examples
--------------
Single PROTAC (three SMILES strings):
    python predict.py \\
        --checkpoint checkpoints/best_model.pt \\
        --smiles-target  "CC(=O)Nc1ccc(Cl)cc1" \\
        --smiles-e3      "O=C1CN(c2ccc(F)cc2)C(=O)c2ccccc21" \\
        --smiles-linker  "CCOCCOCCOCCN"

Batch from CSV (must have smiles_target_lig / smiles_e3_lig / smiles_linker columns):
    python predict.py \\
        --checkpoint checkpoints/best_model.pt \\
        --data data/protac_test.csv \\
        --out  predictions.csv

With protein sequences:
    python predict.py \\
        --checkpoint  checkpoints/best_model.pt \\
        --smiles-target  "CC(=O)Nc1ccc(Cl)cc1" \\
        --smiles-e3      "O=C1CN(c2ccc(F)cc2)C(=O)c2ccccc21" \\
        --smiles-linker  "CCOCCOCCOCCN" \\
        --target-seq  "MKTAYIAKQRQISFVKSHFSRQ" \\
        --ligase-seq  "MSSSHHHHHHHSSGLVPRGSH"

With RF stacker (if a stacker .pkl was saved after training):
    python predict.py \\
        --checkpoint  checkpoints/best_model.pt \\
        --rf-stacker  checkpoints/rf_stacker.pkl \\
        --data data/protac_test.csv

Options
-------
  --checkpoint    Path to trained .pt checkpoint (required)
  --smiles-target Target ligand SMILES
  --smiles-e3     E3 ligand SMILES
  --smiles-linker Linker SMILES
  --target-seq    Target protein sequence (optional)
  --ligase-seq    E3 ligase protein sequence (optional)
  --data          Input CSV for batch mode (mutually exclusive with --smiles-*)
  --out           Output CSV path  [default: predictions.csv]
  --cache-dir     Feature cache directory  [default: .cache]
  --config        train_config.json (optional; used to read architecture)
  --calibrate     Apply temperature-scaled calibration (requires val data)
  --rf-stacker    Path to saved RF stacker .pkl for ensemble prediction
  --device        Force cpu or cuda  [default: auto]
  --quiet         Suppress info messages
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict",
        description="SE3AF standalone PROTAC activity predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── checkpoint (required) ─────────────────────────────────────────
    p.add_argument(
        "--checkpoint", "-c",
        required=True,
        metavar="PATH",
        help="Trained SE3AF checkpoint (.pt file)",
    )

    # ── single-molecule SMILES mode ───────────────────────────────────
    smiles_grp = p.add_argument_group("single-molecule SMILES mode")
    smiles_grp.add_argument(
        "--smiles-target",
        default=None,
        metavar="SMILES",
        help="Target ligand SMILES",
    )
    smiles_grp.add_argument(
        "--smiles-e3",
        default=None,
        metavar="SMILES",
        help="E3 ligand SMILES",
    )
    smiles_grp.add_argument(
        "--smiles-linker",
        default=None,
        metavar="SMILES",
        help="Linker SMILES",
    )
    smiles_grp.add_argument(
        "--target-seq",
        default="",
        metavar="SEQ",
        help="Target protein amino-acid sequence (optional)",
    )
    smiles_grp.add_argument(
        "--ligase-seq",
        default="",
        metavar="SEQ",
        help="E3 ligase amino-acid sequence (optional)",
    )

    # ── batch CSV mode ────────────────────────────────────────────────
    batch_grp = p.add_argument_group("batch CSV mode")
    batch_grp.add_argument(
        "--data", "-d",
        default=None,
        metavar="CSV",
        help=(
            "Input CSV file containing smiles_target_lig, smiles_e3_lig, "
            "smiles_linker columns (and optionally target_seq, ligase_seq)"
        ),
    )
    batch_grp.add_argument(
        "--out", "-o",
        default="predictions.csv",
        metavar="CSV",
        help="Output CSV path for predictions  [default: predictions.csv]",
    )

    # ── optional settings ─────────────────────────────────────────────
    opt_grp = p.add_argument_group("optional settings")
    opt_grp.add_argument(
        "--cache-dir",
        default=".cache",
        metavar="DIR",
        help="Feature cache directory  [default: .cache]",
    )
    opt_grp.add_argument(
        "--config",
        default=None,
        metavar="JSON",
        help="train_config.json path (architecture overrides from checkpoint if omitted)",
    )
    opt_grp.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "Load a saved temperature-scaling calibrator from the checkpoint "
            "and apply it to output probabilities"
        ),
    )
    opt_grp.add_argument(
        "--rf-stacker",
        default=None,
        metavar="PKL",
        help="Path to a saved RF stacker .pkl for ensemble prediction",
    )
    opt_grp.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Force device (default: auto-detect)",
    )
    opt_grp.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational output",
    )

    return p


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def _print_single_result(result: dict, quiet: bool = False) -> None:
    """Pretty-print a single-molecule prediction result to stdout."""
    if quiet:
        print(json.dumps(result))
        return

    prob   = result.get("probability", 0.0)
    active = prob >= 0.5
    bar_w  = int(prob * 40)
    bar    = "█" * bar_w + "░" * (40 - bar_w)

    print()
    print("=" * 56)
    print("  SE3AF  - PROTAC Activity Prediction")
    print("=" * 56)
    print(f"  Probability  : {prob:.4f}  [{bar}]")
    print(f"  Classification: {'✅  ACTIVE' if active else '❌  INACTIVE'}")
    if "stability_prob" in result:
        print(f"  Stability    : {result['stability_prob']:.4f}")
    if "interaction_prob" in result:
        print(f"  Interaction  : {result['interaction_prob']:.4f}")
    print("-" * 56)
    inp = result.get("inputs", {})
    if inp.get("smiles_target_lig"):
        print(f"  Target ligand : {inp['smiles_target_lig']}")
    if inp.get("smiles_e3_lig"):
        print(f"  E3 ligand     : {inp['smiles_e3_lig']}")
    if inp.get("smiles_linker"):
        print(f"  Linker        : {inp['smiles_linker']}")
    print("=" * 56)
    print()


def _write_csv(out_path: Path, rows: list, quiet: bool) -> None:
    """Write a list of prediction dicts to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "probability", "active",
                         "stability_prob", "interaction_prob"])
        for i, row in enumerate(rows):
            prob = float(row.get("probability", 0.0))
            writer.writerow([
                i,
                f"{prob:.6f}",
                "true" if prob >= 0.5 else "false",
                f"{float(row.get('stability_prob',   0.0)):.6f}",
                f"{float(row.get('interaction_prob', 0.0)):.6f}",
            ])
    if not quiet:
        print(f"[predict] Predictions saved -> {out_path}  ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# RF stacker helper
# ---------------------------------------------------------------------------

def _load_rf_stacker(pkl_path: str):
    """Load a pickled RF stacker result (returns None if path is None or missing)."""
    if not pkl_path:
        return None
    p = Path(pkl_path)
    if not p.exists():
        print(f"[predict] WARNING: RF stacker not found at {p}  - skipping ensemble.",
              file=sys.stderr)
        return None
    import pickle
    with open(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    quiet = args.quiet

    # ── validate mutual exclusivity ───────────────────────────────────
    smiles_provided = any([args.smiles_target, args.smiles_e3, args.smiles_linker])
    csv_provided    = bool(args.data)

    if smiles_provided and csv_provided:
        parser.error("--data and --smiles-* are mutually exclusive.")

    if not smiles_provided and not csv_provided:
        parser.error(
            "Provide either --data CSV or all three "
            "--smiles-target / --smiles-e3 / --smiles-linker."
        )

    if smiles_provided:
        missing = [
            name for name, val in [
                ("--smiles-target",  args.smiles_target),
                ("--smiles-e3",      args.smiles_e3),
                ("--smiles-linker",  args.smiles_linker),
            ] if not val
        ]
        if missing:
            parser.error(
                f"Single-molecule mode requires all three SMILES args. "
                f"Missing: {', '.join(missing)}"
            )

    # ── validate checkpoint ───────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[predict] ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    # ── optional model_cfg override from train_config.json ───────────
    model_cfg = None
    if args.config:
        cfg_path = Path(args.config)
        if cfg_path.exists():
            try:
                import json as _json
                with open(cfg_path) as _f:
                    raw = _json.load(_f)
                model_cfg = {
                    "backend":          raw.get("backend",          "lite"),
                    "fusion_dim":       raw.get("fusion_dim",       256),
                    "esm_dim":          raw.get("esm_dim",          1280),
                    "graph_hidden":     raw.get("graph_hidden",     256),
                    "num_graph_layers": raw.get("num_graph_layers", 6),
                    "num_heads":        raw.get("num_heads",        8),
                    "dropout":          0.0,
                }
                if not quiet:
                    print(f"[predict] Loaded architecture from: {cfg_path}")
            except Exception as exc:
                print(f"[predict] WARNING: could not parse config {cfg_path}: {exc}",
                      file=sys.stderr)
        else:
            print(f"[predict] WARNING: config not found at {cfg_path}; "
                  f"using checkpoint defaults.", file=sys.stderr)

    # ── load model (single torch.load call — F02-style single load) ───
    if not quiet:
        print(f"[predict] Loading model from: {ckpt_path}")

    from core.inference import (
        load_model_for_inference,
        predict_from_smiles,
        predict_from_csv,
        _run_inference,
    )

    try:
        model, device = load_model_for_inference(
            checkpoint_path=str(ckpt_path),
            model_cfg=model_cfg,
            device=args.device,
        )
    except Exception as exc:
        print(f"[predict] ERROR loading checkpoint: {exc}", file=sys.stderr)
        return 1

    if not quiet:
        params = sum(p.numel() for p in model.parameters())
        print(f"[predict] Model ready  - device={device}  params={params:,}")

    # ── optional RF stacker ────────────────────────────────────────────
    rf_result = _load_rf_stacker(args.rf_stacker)

    # ── calibrator (loaded from checkpoint if --calibrate) ────────────
    calibrator = None
    if args.calibrate:
        import torch
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        calibrator = ckpt_data.get("calibrator", None)
        if calibrator is None:
            print("[predict] WARNING: --calibrate requested but no calibrator found "
                  "in checkpoint. Calibration skipped.", file=sys.stderr)
        elif not quiet:
            print(f"[predict] Calibrator loaded (temperature={getattr(calibrator, 'temperature', '?')})")

    # ── single-molecule mode ──────────────────────────────────────────
    if smiles_provided:
        try:
            result = predict_from_smiles(
                model=model,
                device=device,
                smiles_target=args.smiles_target,
                smiles_e3=args.smiles_e3,
                smiles_linker=args.smiles_linker,
                target_seq=args.target_seq or "",
                ligase_seq=args.ligase_seq or "",
                cache_dir=args.cache_dir,
            )
        except Exception as exc:
            print(f"[predict] ERROR during inference: {exc}", file=sys.stderr)
            return 1

        # Apply calibrator if loaded
        if calibrator is not None:
            import torch
            with torch.no_grad():
                logit = torch.logit(
                    torch.tensor(result["probability"]).clamp(1e-6, 1 - 1e-6)
                )
                cal_prob = float(torch.sigmoid(logit / calibrator.temperature))
            result["probability"] = cal_prob
            result["active"]      = cal_prob >= 0.5

        _print_single_result(result, quiet=quiet)

        # Also write to --out if user changed the default
        if args.out != "predictions.csv":
            _write_csv(Path(args.out), [result], quiet=quiet)

        return 0

    # ── batch CSV mode ────────────────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[predict] ERROR: data file not found: {data_path}", file=sys.stderr)
        return 1

    if not quiet:
        print(f"[predict] Running batch prediction on: {data_path}")

    try:
        # BUG-PREDICT-01 FIX: predict_from_csv returns np.ndarray of probabilities,
        # not a list of dicts. We use predict_from_dataset directly to get all 3
        # probability outputs (main, stab, inter) for RF stacker support.
        from core.dataset import PROTACDataset
        from core.inference import predict_from_dataset, _run_inference
        pred_ds = PROTACDataset(data_root=str(data_path), cache_dir=args.cache_dir)
        probs_dict = _run_inference(model, device, pred_ds, batch_size=32)
        # Convert to list of dicts for uniform downstream handling
        rows = [
            {
                "probability":      float(probs_dict["main"][i]),
                "stability_prob":   float(probs_dict["stab"][i]),
                "interaction_prob": float(probs_dict["inter"][i]),
                "active":           bool(float(probs_dict["main"][i]) >= 0.5),
            }
            for i in range(len(probs_dict["main"]))
        ]
    except Exception as exc:
        print(f"[predict] ERROR during batch inference: {exc}", file=sys.stderr)
        return 1

    # Apply calibrator row-wise if loaded
    if calibrator is not None:
        import torch
        for row in rows:
            with torch.no_grad():
                logit = torch.logit(
                    torch.tensor(row["probability"]).clamp(1e-6, 1 - 1e-6)
                )
                cal_prob = float(torch.sigmoid(logit / calibrator.temperature))
            row["probability"] = cal_prob
            row["active"]      = cal_prob >= 0.5

    # Apply RF stacker ensemble if loaded
    # BUG-PREDICT-02 FIX: use RFStackerResult.predict() API which handles
    # fingerprint computation internally. Pass 3 neural probs as required.
    if rf_result is not None and hasattr(rf_result, "rf") and rf_result.rf is not None:
        if not quiet:
            print("[predict] Applying RF stacker ensemble …")
        try:
            import numpy as np
            neural_3 = np.stack([
                [r["probability"]      for r in rows],
                [r.get("stability_prob",   0.0) for r in rows],
                [r.get("interaction_prob", 0.0) for r in rows],
            ], axis=1).astype(np.float32)  # (N, 3)
            # Use RFStackerResult.predict() which handles FP matrix internally
            if hasattr(rf_result, "predict"):
                ensemble_probs, _ = rf_result.predict(neural_3, pred_ds)
            else:
                # Fallback: just use neural_3 directly
                ensemble_probs = rf_result.rf.predict_proba(neural_3)[:, 1]
            for row, ep in zip(rows, ensemble_probs):
                row["probability"] = float(ep)
                row["active"]      = float(ep) >= 0.5
        except Exception as exc:
            print(f"[predict] WARNING: RF stacker failed ({exc}); "
                  f"using neural predictions only.", file=sys.stderr)

    _write_csv(Path(args.out), rows, quiet=quiet)

    if not quiet:
        n_active = sum(1 for r in rows if r.get("active", False))
        print(f"[predict] Summary: {n_active}/{len(rows)} predicted ACTIVE "
              f"({100*n_active/max(len(rows),1):.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
