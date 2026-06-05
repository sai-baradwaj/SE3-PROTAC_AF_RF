#!/usr/bin/env python3
"""
main.py  —  SE3AF command-line entry point  (v3.7)
====================================================
v3.7 CHANGES:
  - Reads from config.py (single global control center)
  - Displays config summary before every run
  - Backend switching via config.py BACKEND variable
  - TRAINING_MODE = "fresh" | "continue" support
  - AlphaFold coverage report before training
  - 'ablation' subcommand for component contribution analysis
  - Combined score checkpoint selection (replaces AUROC-only)
  - Architecture fingerprint in every checkpoint

v3.6.2 (carried):
  - Subcommands: train, evaluate, predict, cache, pretrain, export
  - All v3.6.2 bug fixes carried forward
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# V37: config is imported but banner is printed AFTER args are parsed
# so CLI --backend override is reflected correctly in the banner
try:
    import config as _cfg
except ImportError:
    _cfg = None  # type: ignore


def _print_config_banner(resolved_backend: str | None = None) -> None:
    """Print v3.7 global config banner.
    
    resolved_backend: if provided, temporarily override the backend shown in the
    banner (used when --backend CLI flag overrides config.py default).
    """
    if _cfg is None:
        return
    # Temporarily patch module globals so print_config_summary shows the
    # CLI-resolved backend, not the config.py default.
    orig_backend  = _cfg.BACKEND
    orig_use_se3  = _cfg.USE_SE3
    orig_use_lite = _cfg.USE_LITE
    if resolved_backend and resolved_backend in ("se3", "lite"):
        _cfg.BACKEND   = resolved_backend
        _cfg.USE_SE3   = resolved_backend == "se3"
        _cfg.USE_LITE  = resolved_backend == "lite"
    try:
        _cfg.print_config_summary()
        warnings_list = _cfg.validate()
        if warnings_list:
            print("  ⚠  Config warnings detected — see above")
    finally:
        # Restore originals so downstream imports see the file default
        _cfg.BACKEND   = orig_backend
        _cfg.USE_SE3   = orig_use_se3
        _cfg.USE_LITE  = orig_use_lite


# ---------------------------------------------------------------------------
# Lazy imports — keeps --help instantaneous
# ---------------------------------------------------------------------------

def _import_trainer():
    from core.trainer import SE3AFTrainer, TrainerConfig
    return SE3AFTrainer, TrainerConfig


def _import_dataset():
    from core.dataset import PROTACDataset
    return PROTACDataset


def _import_logger():
    from core.utils import get_logger
    return get_logger("se3af.main")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="se3af",
        description="SE3AF: SE(3)-equivariant PROTAC activity & degradation predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    from core import __version__ as _se3af_ver
    parser.add_argument("--version", action="version", version=f"SE3AF {_se3af_ver}")

    subs = parser.add_subparsers(dest="subcommand", required=False)

    # ── ablation ───────────────────────────────────────────────────────
    p_abl = subs.add_parser("ablation",
                             help="Run ablation study for component contribution analysis")
    p_abl.add_argument("--data",       "-d", required=True)
    p_abl.add_argument("--cache-dir",        default=".cache")
    p_abl.add_argument("--epochs",     type=int, default=20)
    p_abl.add_argument("--fast",       action="store_true",
                        help="Run only 5 key conditions")
    p_abl.add_argument("--out-dir",    default="reports")

    # ── train ──────────────────────────────────────────────────────────
    p_train = subs.add_parser("train", help="Train SE3AF on labelled PROTAC data")
    p_train.add_argument("--config", "-c", default="configs/train_config.json",
                         help="train_config.json path")
    p_train.add_argument("--data",   "-d", default=None,
                         help="Directory or CSV file with ALL data (will be split internally)")
    p_train.add_argument("--train-data", default=None,
                         help="Pre-split train CSV (use with --val-data and --test-data)")
    p_train.add_argument("--val-data",   default=None,
                         help="Pre-split val CSV")
    p_train.add_argument("--test-data",  default=None,
                         help="Pre-split test CSV (optional)")
    p_train.add_argument("--cache-dir", default=".cache")
    p_train.add_argument("--resume",  metavar="CHECKPOINT",
                         help="Resume from checkpoint .pt file")
    p_train.add_argument("--no-cache-rebuild", action="store_true",
                         help="Skip cache rebuild check")
    p_train.add_argument("--seed",       type=int,   default=None)
    p_train.add_argument("--epochs",     type=int,   default=None)
    p_train.add_argument("--batch-size", type=int,   default=None)
    p_train.add_argument("--lr",         type=float, default=None)
    p_train.add_argument("--backend", choices=["auto", "se3", "lite"], default=None,
                         help="Graph transformer backend (default: auto)")

    # ── evaluate ───────────────────────────────────────────────────────
    p_eval = subs.add_parser("evaluate",
                             help="Evaluate on test split with bootstrap CIs")
    p_eval.add_argument("--config",      "-c", default="configs/train_config.json")
    p_eval.add_argument("--data",        "-d", default="data/",
                        help="Data directory or CSV file (default: data/)")
    p_eval.add_argument("--cache-dir",         default=".cache")
    p_eval.add_argument("--checkpoint",        required=True,
                        help="Checkpoint .pt file to evaluate")
    p_eval.add_argument("--bootstrap-n", type=int, default=1000,
                        help="Bootstrap iterations for 95%% CI")
    p_eval.add_argument("--out", default=None, help="Save JSON results here")

    # ── predict ────────────────────────────────────────────────────────
    p_pred = subs.add_parser("predict",
                             help="Run inference on CSV or single SMILES")
    p_pred.add_argument("--config",     "-c", default="configs/train_config.json")
    p_pred.add_argument("--data",       "-d", default=None,
                        help="CSV file for batch prediction")
    p_pred.add_argument("--cache-dir",        default=".cache")
    p_pred.add_argument("--checkpoint",       required=True,
                        help="Checkpoint .pt file")
    p_pred.add_argument("--out",    default="predictions.csv")
    p_pred.add_argument("--calibrate", action="store_true",
                        help="Apply temperature scaling before prediction")
    # N01 FIX: single-SMILES args
    p_pred.add_argument("--smiles-target",  default=None, metavar="SMILES",
                        help="Target ligand SMILES (single-molecule mode)")
    p_pred.add_argument("--smiles-e3",      default=None, metavar="SMILES",
                        help="E3 ligand SMILES (single-molecule mode)")
    p_pred.add_argument("--smiles-linker",  default=None, metavar="SMILES",
                        help="Linker SMILES (single-molecule mode)")
    p_pred.add_argument("--target-seq",     default="", metavar="SEQ")
    p_pred.add_argument("--ligase-seq",     default="", metavar="SEQ")

    # ── cache ──────────────────────────────────────────────────────────
    p_cache = subs.add_parser("cache",
                              help="Rebuild or validate the feature cache")
    p_cache.add_argument("--data",          "-d", required=True)
    p_cache.add_argument("--cache-dir",           default=".cache")
    p_cache.add_argument("--validate-only",       action="store_true")
    p_cache.add_argument("--workers",       type=int, default=4)
    p_cache.add_argument("--max-atoms",     type=int, default=256)
    p_cache.add_argument("--use-coords",          action="store_true")
    p_cache.add_argument("--force",               action="store_true")

    # ── pretrain ───────────────────────────────────────────────────────
    p_pre = subs.add_parser("pretrain",
                            help="SSL pre-training (masked-graph + contrastive)")
    p_pre.add_argument("--config",    "-c", default="configs/train_config.json")
    p_pre.add_argument("--data",      "-d", required=True)
    p_pre.add_argument("--cache-dir",       default=".cache")
    p_pre.add_argument("--epochs",    type=int, default=None)
    p_pre.add_argument("--out",       default="checkpoints/pretrained.pt")

    # ── export ─────────────────────────────────────────────────────────
    p_exp = subs.add_parser("export",
                            help="Export model to ONNX or TorchScript")
    p_exp.add_argument("--checkpoint", required=True)
    p_exp.add_argument("--format", choices=["onnx", "torchscript"],
                       default="torchscript")
    p_exp.add_argument("--out",    default=None)
    p_exp.add_argument("--config", "-c", default="configs/train_config.json")

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def _cmd_train(args: argparse.Namespace) -> int:
    SE3AFTrainer, TrainerConfig = _import_trainer()
    PROTACDataset = _import_dataset()
    logger = _import_logger()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1
    cfg = TrainerConfig.from_json(config_path)

    if args.seed       is not None: cfg.seed          = args.seed
    if args.epochs     is not None: cfg.epochs        = args.epochs
    if args.batch_size is not None: cfg.batch_size    = args.batch_size
    if args.lr         is not None: cfg.learning_rate = args.lr
    if args.backend    is not None: cfg.backend       = args.backend

    # FIX-SPLIT-01: Support pre-split train/val/test files
    # When --train-data is provided, use those files directly and skip internal resplit
    _has_presplit = getattr(args, "train_data", None) is not None

    if _has_presplit:
        import pandas as pd
        # Use pre-split files — merge into one dataset and mark splits explicitly
        _train_data = args.train_data
        _val_data   = getattr(args, "val_data", None)
        _test_data  = getattr(args, "test_data", None)
        if not _val_data:
            logger.warning("--train-data provided without --val-data; will use internal val split")
            _has_presplit = False
        else:
            logger.info(f"Using pre-split files: train={_train_data}, val={_val_data}, test={_test_data}")
            data_root = _train_data
            if not args.no_cache_rebuild:
                _run_cache_rebuild(_train_data, args.cache_dir, workers=4, logger=logger)
    else:
        # Validate --data is provided when not using presplit
        data_root = getattr(args, "data", None)
        if not data_root:
            logger.error("Must provide either --data or --train-data/--val-data")
            return 1
        if not args.no_cache_rebuild:
            logger.info("Checking feature cache …")
            _run_cache_rebuild(data_root, args.cache_dir, workers=4, logger=logger)

    if _has_presplit:
        # Load pre-split datasets
        import pandas as pd
        from core.utils import discover_datasets
        _train_dfs = discover_datasets(_train_data)
        _val_dfs   = discover_datasets(_val_data)
        train_df   = pd.concat(_train_dfs, ignore_index=True)
        val_df     = pd.concat(_val_dfs,   ignore_index=True)
        if _test_data:
            _test_dfs = discover_datasets(_test_data)
            test_df   = pd.concat(_test_dfs, ignore_index=True)
        else:
            test_df = val_df.copy()

        # Create a combined dataset for PROTACDataset, then provide explicit split indices
        import tempfile, os
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        # Write to temp file so PROTACDataset can load it
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        all_df.to_csv(tmp_path, index=False)
        os.close(tmp_fd)
        try:
            dataset = PROTACDataset(
                data_root=tmp_path,
                cache_dir=args.cache_dir,
                supervised=True,
            )
            # Inject pre-split indices into trainer
            n_train = len(train_df)
            n_val   = len(val_df)
            n_test  = len(test_df)
            cfg._presplit_n_train = n_train
            cfg._presplit_n_val   = n_val
            cfg._presplit_n_test  = n_test
            logger.info(f"Pre-split: train={n_train}, val={n_val}, test={n_test}")
        finally:
            try: os.unlink(tmp_path)
            except: pass
    else:
        logger.info(f"Loading dataset from: {data_root}")
        dataset = PROTACDataset(
            data_root=data_root,
            cache_dir=args.cache_dir,
            supervised=True,   # N02 FIX
        )

    logger.info(f"Dataset size: {len(dataset)}")

    trainer = SE3AFTrainer(cfg)
    trainer.setup(dataset)
    best_metrics = trainer.train(resume_from=args.resume)

    logger.info("=== Training complete ===")
    for k, v in best_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    SE3AFTrainer, TrainerConfig = _import_trainer()
    PROTACDataset = _import_dataset()
    logger = _import_logger()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return 1

    cfg     = TrainerConfig.from_json(config_path)
    dataset = PROTACDataset(data_root=args.data, cache_dir=args.cache_dir)
    trainer = SE3AFTrainer(cfg)
    trainer.setup(dataset)

    results = trainer.evaluate(
        checkpoint=args.checkpoint,
        bootstrap_n=args.bootstrap_n,
    )

    logger.info("=== Evaluation results ===")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved -> {out_path}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import csv
    import tempfile

    import pandas as pd

    SE3AFTrainer, TrainerConfig = _import_trainer()
    PROTACDataset = _import_dataset()
    logger = _import_logger()

    cfg = TrainerConfig.from_json(args.config)

    # N01 FIX: single-SMILES mode → build temp CSV
    tmp_csv_path = None
    data_path    = args.data

    smiles_provided = any([args.smiles_target, args.smiles_e3, args.smiles_linker])

    if smiles_provided:
        if not (args.smiles_target and args.smiles_e3 and args.smiles_linker):
            logger.error(
                "Single-SMILES mode requires all three: "
                "--smiles-target, --smiles-e3, --smiles-linker"
            )
            return 1
        if data_path:
            logger.warning("--data ignored; --smiles-* args take precedence.")
        row = {
            "smiles_target_lig": args.smiles_target,
            "smiles_e3_lig":     args.smiles_e3,
            "smiles_linker":     args.smiles_linker,
            "target_seq":        getattr(args, "target_seq", "") or "",
            "ligase_seq":        getattr(args, "ligase_seq", "") or "",
        }
        tmp_fd, tmp_csv_path = tempfile.mkstemp(suffix=".csv")
        pd.DataFrame([row]).to_csv(tmp_csv_path, index=False)
        os.close(tmp_fd)
        data_path = tmp_csv_path
        logger.info(f"Single-SMILES predict  - temp CSV: {tmp_csv_path}")
    elif not data_path:
        logger.error(
            "Provide --data CSV or all three --smiles-target/--smiles-e3/--smiles-linker."
        )
        return 1

    try:
        dataset    = PROTACDataset(data_root=data_path, cache_dir=args.cache_dir)
        trainer    = SE3AFTrainer(cfg)
        trainer.setup(dataset)

        calibrator = trainer.calibrate() if args.calibrate else None
        probs = trainer.predict(
            dataset,
            checkpoint=args.checkpoint,
            calibrator=calibrator,
        )

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "probability", "active"])
            for i, p in enumerate(probs):
                prob = float(p)
                writer.writerow([i, f"{prob:.6f}", "true" if prob >= 0.5 else "false"])

        logger.info(f"Predictions saved -> {out_path}  ({len(probs)} rows)")

        if smiles_provided:
            prob = float(probs[0])
            print(f"\n{'='*50}")
            print("SE3AF Prediction Result")
            print(f"{'='*50}")
            print(f"  Probability:  {prob:.4f}")
            print(f"  Prediction:   {'ACTIVE' if prob >= 0.5 else 'INACTIVE'}")
            print(f"  SMILES (tgt): {args.smiles_target}")
            print(f"  SMILES (e3):  {args.smiles_e3}")
            print(f"  SMILES (lnk): {args.smiles_linker}")
            print(f"{'='*50}\n")
    finally:
        if tmp_csv_path:
            try:
                os.unlink(tmp_csv_path)
            except OSError:
                pass
    return 0


def _run_cache_rebuild(
    data_path:     str,
    cache_dir:     str,
    workers:       int  = 4,
    force:         bool = False,
    validate_only: bool = False,
    max_atoms:     int  = 256,
    use_coords:    bool = False,
    logger=None,
) -> int:
    import subprocess
    cmd = [
        sys.executable, "rebuild_cache.py",
        "--data",      data_path,
        "--cache-dir", cache_dir,
        "--workers",   str(workers),
        "--max-atoms", str(max_atoms),
    ]
    if force:         cmd.append("--force")
    if validate_only: cmd.append("--validate-only")
    if use_coords:    cmd.append("--coords")
    return subprocess.call(cmd)


def _cmd_cache(args: argparse.Namespace) -> int:
    logger = _import_logger()
    logger.info(
        f"{'Validating' if args.validate_only else 'Rebuilding'} cache — "
        f"data={args.data}  cache-dir={args.cache_dir}"
    )
    return _run_cache_rebuild(
        data_path=args.data,
        cache_dir=args.cache_dir,
        workers=args.workers,
        force=args.force,
        validate_only=args.validate_only,
        max_atoms=args.max_atoms,
        use_coords=args.use_coords,
        logger=logger,
    )


def _cmd_pretrain(args: argparse.Namespace) -> int:
    import torch
    SE3AFTrainer, TrainerConfig = _import_trainer()
    PROTACDataset = _import_dataset()
    logger = _import_logger()

    cfg = TrainerConfig.from_json(args.config)
    if args.epochs is not None:
        cfg.ssl_pretrain_epochs = args.epochs
    if cfg.ssl_pretrain_epochs <= 0:
        logger.error("ssl_pretrain_epochs must be > 0. Set in config or pass --epochs N.")
        return 1

    cfg.epochs           = cfg.ssl_pretrain_epochs
    cfg.use_masked_graph = True
    cfg.use_contrastive  = True

    dataset = PROTACDataset(data_root=args.data, cache_dir=args.cache_dir)
    trainer = SE3AFTrainer(cfg)
    trainer.setup(dataset)

    trainer._ckpt_dir = Path(args.out).parent
    trainer._ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.train()

    out_path = Path(args.out)
    torch.save(
        {"model_state": trainer.model.state_dict(), "pretrain": True},
        str(out_path),
    )
    logger.info(f"Pre-trained weights saved -> {out_path}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """V38-FIX-EXPORT-01: Use load_model_for_inference (reads af_extra_dim
    from checkpoint) instead of hardcoded model_cfg. Falls back to state-dict
    export when TorchScript fails (itertools.combinations is not scriptable).
    """
    import torch
    logger = _import_logger()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1

    suffix   = ".onnx" if args.format == "onnx" else ".pt"
    out_path = Path(args.out) if args.out else ckpt_path.with_suffix(suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # V38-FIX-EXPORT-01: use inference loader (reads arch from checkpoint)
    from core.inference import load_model_for_inference
    model, _device = load_model_for_inference(str(ckpt_path))
    model.eval()

    if args.format == "torchscript":
        try:
            scripted = torch.jit.script(model)
            scripted.save(str(out_path))
            logger.info(f"TorchScript model saved -> {out_path}")
        except RuntimeError as ts_err:
            # SE3AF uses itertools.combinations and dynamic Python which are
            # not TorchScript-compatible. Fall back to state-dict export.
            logger.warning(
                f"TorchScript failed ({ts_err.__class__.__name__}): {ts_err}\n"
                "Falling back to state-dict .pt export (compatible with torch.load)."
            )
            ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            torch.save({
                "model_state": model.state_dict(),
                "cfg":         ckpt_data.get("cfg", {}),
                "export_note": "torchscript_fallback_state_dict",
            }, str(out_path))
            logger.info(f"State-dict export saved -> {out_path}")
    elif args.format == "onnx":
        logger.warning("ONNX export not implemented; saving state-dict instead.")
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        torch.save({
            "model_state": model.state_dict(),
            "cfg":         ckpt_data.get("cfg", {}),
            "export_note": "onnx_fallback_state_dict",
        }, str(out_path))
        logger.info(f"State-dict export saved -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _cmd_ablation(args: argparse.Namespace) -> int:
    """Run ablation study."""
    sys.path.insert(0, str(Path(__file__).parent))
    from ablation.run_ablation import main as _ablation_main
    argv = ["--data", args.data, "--cache-dir", args.cache_dir,
            "--epochs", str(args.epochs), "--out-dir", args.out_dir]
    if args.fast:
        argv.append("--fast")
    return _ablation_main(argv)


_DISPATCH = {
    "train":    _cmd_train,
    "evaluate": _cmd_evaluate,
    "predict":  _cmd_predict,
    "cache":    _cmd_cache,
    "pretrain": _cmd_pretrain,
    "export":   _cmd_export,
    "ablation": _cmd_ablation,
}


def _print_help_menu() -> None:
    """V38-FIX-HELP-01: Print friendly help menu instead of 'usage error'."""
    from core import __version__ as _ver
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║   SE3AF v{_ver}  — SE(3)-Equivariant PROTAC Activity Predictor    ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  Available Commands                                                  ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  train      Train SE3AF on labelled PROTAC data                      ║")
    print("║    python main.py train --config configs/train_config.json           ║")
    print("║                        --data data/                                  ║")
    print("╠──────────────────────────────────────────────────────────────────────╣")
    print("║  evaluate   Evaluate model with bootstrap confidence intervals       ║")
    print("║    python main.py evaluate --checkpoint checkpoints/best_model.pt    ║")
    print("║                            --data data/                              ║")
    print("╠──────────────────────────────────────────────────────────────────────╣")
    print("║  predict    Run inference on CSV or single SMILES                    ║")
    print("║    python main.py predict --checkpoint checkpoints/best_model.pt     ║")
    print("║                           --data data/predict.csv                    ║")
    print("║    python main.py predict --checkpoint checkpoints/best_model.pt     ║")
    print("║                           --smiles-target \"...\" --smiles-e3 \"...\"║")
    print("║                           --smiles-linker \"...\"                    ║")
    print("╠──────────────────────────────────────────────────────────────────────╣")
    print("║  cache      Rebuild or validate the molecular graph feature cache    ║")
    print("║    python main.py cache --data data/                                 ║")
    print("║    python main.py cache --data data/ --validate-only                 ║")
    print("╠──────────────────────────────────────────────────────────────────────╣")
    print("║  pretrain   SSL pre-training (masked-graph + contrastive)            ║")
    print("║    python main.py pretrain --config configs/train_config.json        ║")
    print("║                            --data data/                              ║")
    print("╠──────────────────────────────────────────────────────────────────────╣")
    print("║  export     Export model to TorchScript or ONNX                      ║")
    print("║    python main.py export --checkpoint checkpoints/best_model.pt      ║")
    print("║                         --format torchscript                         ║")
    print("╠──────────────────────────────────────────────────────────────────────╣")
    print("║  ablation   Run ablation study for component contribution analysis   ║")
    print("║    python main.py ablation --data data/ --epochs 20                  ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  Quick Start:                                                        ║")
    print("║    python rebuild_cache.py                                           ║")
    print("║    python main.py train --config configs/train_config.json --data data/ ║")
    print("║    python app.py                  # Web UI on http://localhost:5000  ║")
    print("║    python api.py                  # REST API on http://localhost:5001║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()


def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # V38-FIX-HELP-01: Show friendly help menu when no subcommand given
    if not args.subcommand:
        _print_help_menu()
        return 0

    # V37: Print config banner NOW — after args parsed so --backend is known
    resolved_backend = getattr(args, "backend", None)  # None for non-train cmds
    _print_config_banner(resolved_backend)

    fn     = _DISPATCH.get(args.subcommand)
    if fn is None:
        _print_help_menu()
        return 0
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
