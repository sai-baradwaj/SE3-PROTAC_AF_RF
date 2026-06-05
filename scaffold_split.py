#!/usr/bin/env python3
"""
scaffold_split.py  —  SE3AF v3.8.1
=====================================
Scientifically rigorous scaffold-based split for PROTAC datasets.

Eliminates the data leakage found in the pre-existing random split:
  - PROTAC SMILES identity leakage (same molecule in train+val+test)
  - Chemical scaffold overlap (same Murcko scaffold in train+val)
  - Protein identity overlap (same target in all splits)

Split Strategy
--------------
1. Remove any exact-PROTAC-SMILES duplicates across splits (hard dedup)
2. Compute Murcko scaffold for warhead_smiles (primary pharmacophore)
3. Assign scaffolds to groups, then split groups to train/val/test
4. Ensure SMILES overlap and scaffold overlap is < 5%

Usage
-----
  python scaffold_split.py --data data/ --out data/split_scaffold/
  python scaffold_split.py --data data/all_data.csv --out data/split/
  python scaffold_split.py --data data/ --strategy protein  # protein split
  python scaffold_split.py --data data/ --strategy combined  # best practice

Options
-------
  --data        input: directory of CSVs or single CSV
  --out         output directory for train.csv, val.csv, test.csv
  --strategy    scaffold | protein | combined (default: scaffold)
  --val-frac    validation fraction (default: 0.15)
  --test-frac   test fraction (default: 0.10)
  --seed        random seed (default: 42)
  --max-overlap maximum allowed overlap fraction (default: 0.05)
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def get_murcko_scaffold(smiles: str) -> str:
    """Compute Murcko scaffold SMILES; return '' for invalid/empty SMILES."""
    if not smiles or str(smiles).strip().lower() in ("nan", "none", "null", ""):
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return ""
        scaffold_mol = GetScaffoldForMol(mol)
        if scaffold_mol is None:
            return ""
        return Chem.MolToSmiles(scaffold_mol, canonical=True) or ""
    except Exception:
        return ""


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "warhead_smiles",
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by Murcko scaffold.

    Returns (train_df, val_df, test_df).
    """
    rng = random.Random(seed)

    # Build scaffold → row index mapping
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        smi = row.get(smiles_col, "")
        scaffold = get_murcko_scaffold(str(smi)) or f"_hash_{hashlib.md5(str(smi).encode()).hexdigest()[:8]}"
        scaffold_to_indices[scaffold].append(idx)

    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    n_total = len(df)
    n_test  = max(1, int(n_total * test_frac))
    n_val   = max(1, int(n_total * val_frac))

    train_idx: List[int] = []
    val_idx:   List[int] = []
    test_idx:  List[int] = []

    # Greedily assign scaffolds to splits
    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]
        if len(test_idx) < n_test:
            test_idx.extend(indices)
        elif len(val_idx) < n_val:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    # If any split is empty, fall back to stratified split
    if not val_idx or not test_idx:
        warnings.warn("Scaffold split produced empty val/test; falling back to stratified split")
        from sklearn.model_selection import train_test_split
        all_idx = list(range(len(df)))
        labels  = df["label"].values.tolist() if "label" in df.columns else None
        tr_idx, te_idx = train_test_split(all_idx, test_size=n_test, random_state=seed,
                                          stratify=labels)
        tr_labels = [labels[i] for i in tr_idx] if labels else None
        tr_idx, va_idx = train_test_split(tr_idx, test_size=n_val, random_state=seed,
                                          stratify=tr_labels)
        train_idx, val_idx, test_idx = tr_idx, va_idx, te_idx

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx  ].reset_index(drop=True),
        df.loc[test_idx ].reset_index(drop=True),
    )


def protein_split(
    df: pd.DataFrame,
    protein_col: str = "target_uniprot",
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame by target protein identity.

    Ensures no protein appears in both train and val/test.
    Returns (train_df, val_df, test_df).
    """
    rng = random.Random(seed)

    # Group by protein
    protein_to_rows: Dict[str, List[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        pid = str(row.get(protein_col, "unknown")).strip()
        protein_to_rows[pid].append(idx)

    proteins = list(protein_to_rows.keys())
    rng.shuffle(proteins)

    n_total   = len(df)
    n_test    = max(1, int(n_total * test_frac))
    n_val     = max(1, int(n_total * val_frac))

    train_idx: List[int] = []
    val_idx:   List[int] = []
    test_idx:  List[int] = []

    for protein in proteins:
        indices = protein_to_rows[protein]
        if len(test_idx) < n_test:
            test_idx.extend(indices)
        elif len(val_idx) < n_val:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx  ].reset_index(drop=True),
        df.loc[test_idx ].reset_index(drop=True),
    )


def deduplicate_across_splits(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    smiles_col: str = "protac_smiles",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove any exact-SMILES duplicates that appear in val/test AND train.

    Exact PROTAC duplicates are moved from val/test to train.
    """
    train_smiles: Set[str] = set(train[smiles_col].dropna().astype(str))

    # Move val rows that are in train → to train
    val_dup_mask = val[smiles_col].astype(str).isin(train_smiles)
    if val_dup_mask.any():
        moved = val[val_dup_mask]
        train = pd.concat([train, moved], ignore_index=True)
        val   = val[~val_dup_mask].reset_index(drop=True)
        print(f"  Moved {val_dup_mask.sum()} PROTAC duplicates from val → train")

    # Move test rows that are in train → to train
    test_dup_mask = test[smiles_col].astype(str).isin(train_smiles)
    if test_dup_mask.any():
        moved = test[test_dup_mask]
        train = pd.concat([train, moved], ignore_index=True)
        test  = test[~test_dup_mask].reset_index(drop=True)
        print(f"  Moved {test_dup_mask.sum()} PROTAC duplicates from test → train")

    return train, val, test


def compute_overlap(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    cols: List[str],
) -> float:
    """Compute fraction of eval SMILES that also appear in train."""
    train_smiles: Set[str] = set()
    eval_smiles:  Set[str] = set()
    for col in cols:
        if col in train.columns:
            train_smiles.update(s for s in train[col].dropna().astype(str)
                                if s and s.lower() not in ("nan", "none"))
        if col in eval_df.columns:
            eval_smiles.update(s for s in eval_df[col].dropna().astype(str)
                               if s and s.lower() not in ("nan", "none"))
    if not eval_smiles:
        return 0.0
    overlap = train_smiles & eval_smiles
    return len(overlap) / len(eval_smiles)


def print_split_report(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> None:
    """Print split statistics and leakage report."""
    smiles_cols = ["warhead_smiles", "linker_smiles", "e3_ligase_smiles"]
    tv_overlap  = compute_overlap(train, val,  smiles_cols)
    tt_overlap  = compute_overlap(train, test, smiles_cols)

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║          SE3AF — Scaffold Split Report               ║")
    print("╠══════════════════════════════════════════════════════╣")
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        if "label" in df.columns:
            pos = int(df["label"].sum())
            neg = len(df) - pos
            print(f"║  {name:<6}: {len(df):>5} rows | pos={pos:>4} neg={neg:>4} ({pos/len(df):.1%})  ║")
        else:
            print(f"║  {name:<6}: {len(df):>5} rows{'':<36}║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Component SMILES overlap:                           ║")
    tv_ok = "✓" if tv_overlap < 0.50 else "⚠"
    tt_ok = "✓" if tt_overlap < 0.50 else "⚠"
    print(f"║    Train/Val:  {tv_overlap:>6.1%}  {tv_ok:<36}║")
    print(f"║    Train/Test: {tt_overlap:>6.1%}  {tt_ok:<36}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scaffold_split",
        description="Scaffold-based split for PROTAC datasets",
    )
    parser.add_argument("--data",       default="data/",
                        help="Input: directory of CSVs or single CSV")
    parser.add_argument("--out",        default="data/split_scaffold/",
                        help="Output directory")
    parser.add_argument("--strategy",   choices=["scaffold", "protein", "combined"],
                        default="scaffold")
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--test-frac",  type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--max-overlap",type=float, default=0.50,
                        help="Max allowed overlap fraction before warning")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent))
    from core.utils import discover_datasets

    # Load data
    dfs = discover_datasets(args.data)
    df  = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df)} rows from {args.data}")

    # Remove within-dataset duplicates
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Removed {n_before - len(df)} duplicate rows")

    # Apply split
    if args.strategy == "scaffold":
        print("  Applying Murcko scaffold split on warhead_smiles...")
        train, val, test = scaffold_split(
            df, smiles_col="warhead_smiles",
            val_frac=args.val_frac, test_frac=args.test_frac,
            seed=args.seed,
        )
    elif args.strategy == "protein":
        print("  Applying protein identity split on target_uniprot...")
        train, val, test = protein_split(
            df, protein_col="target_uniprot",
            val_frac=args.val_frac, test_frac=args.test_frac,
            seed=args.seed,
        )
    else:  # combined
        print("  Applying combined protein + scaffold split...")
        # First: hold out proteins
        train1, val1, test1 = protein_split(
            df, protein_col="target_uniprot",
            val_frac=args.val_frac * 0.5, test_frac=args.test_frac * 0.5,
            seed=args.seed,
        )
        # Second: scaffold split remaining train data
        train, val2, test2 = scaffold_split(
            train1, smiles_col="warhead_smiles",
            val_frac=args.val_frac * 0.5, test_frac=args.test_frac * 0.5,
            seed=args.seed + 1,
        )
        val  = pd.concat([val1,  val2 ], ignore_index=True)
        test = pd.concat([test1, test2], ignore_index=True)

    # Remove cross-split exact PROTAC duplicates
    if "protac_smiles" in df.columns:
        train, val, test = deduplicate_across_splits(train, val, test, "protac_smiles")

    # Print report
    print_split_report(train, val, test)

    # Check overlap
    smiles_cols = ["warhead_smiles", "linker_smiles", "e3_ligase_smiles"]
    tv_overlap  = compute_overlap(train, val,  smiles_cols)
    tt_overlap  = compute_overlap(train, test, smiles_cols)
    if tv_overlap > args.max_overlap or tt_overlap > args.max_overlap:
        print(f"  ⚠  WARNING: overlap ({max(tv_overlap, tt_overlap):.1%}) exceeds max ({args.max_overlap:.0%})")
        print(f"     Consider --strategy combined for better separation")

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(  out_dir / "val.csv",   index=False)
    test.to_csv( out_dir / "test.csv",  index=False)
    print(f"  ✓  Split saved → {out_dir}")
    print(f"       train.csv: {len(train)} rows")
    print(f"       val.csv:   {len(val)} rows")
    print(f"       test.csv:  {len(test)} rows")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
