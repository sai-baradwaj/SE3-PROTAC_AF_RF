#!/usr/bin/env python3
"""
rebuild_cache.py  —  SE3AF v3.8.1
===================================
Rebuild the molecular graph feature cache (.cache/graphs/) for all CSV files
in a data directory.

This must be run before training if:
  1. The cache is empty or missing
  2. The source CSV files have changed
  3. The featurisation code has been updated

Usage
-----
  python rebuild_cache.py                        # default: data/ → .cache/
  python rebuild_cache.py --data data/ --cache-dir .cache/ --coords --workers 4
  python rebuild_cache.py --validate-only        # check cache without rebuilding
  python rebuild_cache.py --force                # force rebuild even if cache exists

Options
-------
  --data         data directory or CSV file (default: data/)
  --cache-dir    cache directory (default: .cache/)
  --coords       include 3D conformer coordinates (default: True in v4.1)
  --workers      parallel workers (default: 4; 0=sequential)
  --max-atoms    skip molecules with more than N atoms (default: 256)
  --validate-only only check existing cache validity
  --force        delete existing cache before rebuilding
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

# ── Cache version must match dataset.py ──────────────────────────────────────
CACHE_VERSION = "v3.1"


def _worker_init():
    """Suppress warnings in worker processes."""
    warnings.filterwarnings("ignore")


def _build_one(args: Tuple[str, str, bool, int]) -> Tuple[str, str, bool]:
    """Build graph for a single SMILES string.

    Returns (smiles, md5_key, success).
    """
    smiles, cache_dir_str, use_coords, max_atoms = args
    cache_dir = Path(cache_dir_str) / "graphs"

    if not smiles or str(smiles).strip().lower() in ("nan", "none", "", "null"):
        return smiles, "empty", True   # skip empty SMILES

    md5_key    = hashlib.md5(smiles.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{md5_key}.pkl"

    # Check if already cached and valid
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as fh:
                payload = pickle.load(fh)
            if isinstance(payload, dict) and payload.get("version") == CACHE_VERSION:
                return smiles, md5_key, True   # cache hit
        except Exception:
            pass   # stale / corrupt → rebuild

    # Build fresh
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from core.preprocessing import smiles_to_graph, validate_graph
        g = smiles_to_graph(smiles, use_coords=use_coords, max_atoms=max_atoms)
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload   = {"version": CACHE_VERSION, "data": g, "smiles": smiles}
        tmp_path  = cache_dir / f"{md5_key}.tmp"
        with open(tmp_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=4)
        tmp_path.replace(cache_path)
        return smiles, md5_key, True
    except Exception as exc:
        return smiles, md5_key, False


def collect_unique_smiles(data_path: str) -> List[str]:
    """Discover all CSV files and return list of unique SMILES."""
    import pandas as pd
    from core.utils import discover_datasets, COL_TGT_SMILES, COL_E3_SMILES, COL_LNK_SMILES

    dfs = discover_datasets(data_path)
    smiles_set = set()
    for df in dfs:
        for col in [COL_TGT_SMILES, COL_E3_SMILES, COL_LNK_SMILES]:
            if col in df.columns:
                smiles_set.update(s for s in df[col].dropna() if s and str(s).strip())
    return list(smiles_set)


def validate_cache(cache_dir: str) -> Tuple[int, int, int]:
    """Validate existing cache entries.

    Returns (n_valid, n_invalid, n_total).
    """
    graph_dir = Path(cache_dir) / "graphs"
    if not graph_dir.exists():
        return 0, 0, 0

    n_valid   = 0
    n_invalid = 0

    for pkl_file in graph_dir.glob("*.pkl"):
        try:
            with open(pkl_file, "rb") as fh:
                payload = pickle.load(fh)
            if (isinstance(payload, dict)
                    and payload.get("version") == CACHE_VERSION
                    and "data" in payload):
                n_valid += 1
            else:
                n_invalid += 1
        except Exception:
            n_invalid += 1

    return n_valid, n_invalid, n_valid + n_invalid


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rebuild_cache",
        description="Rebuild SE3AF molecular graph feature cache",
    )
    parser.add_argument("--data",          default="data/",
                        help="Data directory or CSV file")
    parser.add_argument("--cache-dir",     default=".cache/",
                        help="Cache directory")
    parser.add_argument("--coords",        action="store_true", default=True,
                        help="Include 3D conformer coordinates (default: True)")
    parser.add_argument("--no-coords",     action="store_true",
                        help="Disable 3D coordinates")
    parser.add_argument("--workers",       type=int, default=4,
                        help="Parallel workers (0=sequential)")
    parser.add_argument("--max-atoms",     type=int, default=256,
                        help="Skip molecules with more than N atoms")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate existing cache")
    parser.add_argument("--force",         action="store_true",
                        help="Delete existing cache before rebuilding")
    args = parser.parse_args(argv)

    use_coords = args.coords and not args.no_coords

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       SE3AF v3.8.1 — Cache Builder                  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Data:       {args.data:<38}║")
    print(f"║  Cache:      {args.cache_dir:<38}║")
    print(f"║  3D coords:  {'Yes' if use_coords else 'No':<38}║")
    print(f"║  Workers:    {args.workers:<38}║")
    print(f"║  Max atoms:  {args.max_atoms:<38}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Force rebuild: delete existing graphs
    if args.force and not args.validate_only:
        graph_dir = Path(args.cache_dir) / "graphs"
        if graph_dir.exists():
            import shutil
            shutil.rmtree(graph_dir)
            print(f"  Deleted existing graph cache: {graph_dir}")

    # Validate-only mode
    if args.validate_only:
        n_valid, n_invalid, n_total = validate_cache(args.cache_dir)
        print(f"  Cache validation: {n_valid}/{n_total} valid, {n_invalid} invalid")
        if n_invalid > 0:
            print(f"  ⚠  {n_invalid} invalid entries — run without --validate-only to rebuild")
        else:
            print(f"  ✓  All {n_valid} cache entries are valid")
        return 0

    # Collect SMILES
    try:
        all_smiles = collect_unique_smiles(args.data)
    except Exception as exc:
        print(f"  ERROR collecting SMILES: {exc}")
        return 1

    print(f"  Unique SMILES to process: {len(all_smiles)}")

    # Build cache
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.cache_dir) / "graphs").mkdir(parents=True, exist_ok=True)

    worker_args = [
        (smi, args.cache_dir, use_coords, args.max_atoms)
        for smi in all_smiles
    ]

    n_ok  = 0
    n_err = 0

    try:
        from tqdm import tqdm as _tqdm
        _pbar = _tqdm(total=len(worker_args), desc="  Building cache", unit="mol", ncols=100)
    except ImportError:
        _pbar = None

    if args.workers > 0 and len(worker_args) > 10:
        # Parallel build
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_worker_init) as ex:
            futures = {ex.submit(_build_one, w): w for w in worker_args}
            for fut in as_completed(futures):
                try:
                    _, _, ok = fut.result()
                    if ok:
                        n_ok += 1
                    else:
                        n_err += 1
                except Exception:
                    n_err += 1
                if _pbar:
                    _pbar.update(1)
    else:
        # Sequential build
        for w in worker_args:
            _, _, ok = _build_one(w)
            if ok:
                n_ok += 1
            else:
                n_err += 1
            if _pbar:
                _pbar.update(1)

    if _pbar:
        _pbar.close()

    # Write metadata
    meta = {
        "version":     CACHE_VERSION,
        "n_ok":        n_ok,
        "n_err":       n_err,
        "use_coords":  use_coords,
        "max_atoms":   args.max_atoms,
    }
    meta_path = Path(args.cache_dir) / "cache_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print()
    print(f"  ✓  Cache complete: {n_ok} ok, {n_err} errors")
    print(f"  Metadata: {meta_path}")
    print()
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
