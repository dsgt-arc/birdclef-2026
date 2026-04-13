#!/usr/bin/env python3
"""Pre-registered lambda_fam ablation on S22 (decision fold).

Loads all data once, then trains 5 models in sequence with lambda_fam
from the pre-registered set {0, 1e-3, 1e-2, 1e-1, 1}.  Decision rule
from proposal: drop L_family in v2 if no lambda_fam > 0 beats
lambda_fam = 0 on amphibian subset by >= 0.5 AUROC.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from birdclef_107.data import (
    load_all_focal,
    load_all_unlabeled,
    load_labeled_soundscape,
)
from birdclef_107.family_utils import build_family_index

# Reuse the training helper from scripts/train.py by importing it dynamically.
import importlib.util
_train_path = Path(__file__).parent / "train.py"
_spec = importlib.util.spec_from_file_location("train_mod", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
train_one_fold = _train_mod.train_one_fold
set_seed = _train_mod.set_seed


LAMBDA_FAM_VALUES = [0.0, 1e-3, 1e-2, 1e-1, 1.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=str, default="S22")
    p.add_argument("--lambda-site", type=float, default=0.01)
    p.add_argument("--lambda-xattn", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--unsup-warmup-start", type=int, default=200)
    p.add_argument("--unsup-warmup-end", type=int, default=500)
    p.add_argument("--unsup-batch-recordings", type=int, default=16)
    p.add_argument("--unsup-batch-windows-per-recording", type=int, default=8)
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default="~/scratch/birdclef-2026/checkpoints/exp107-sweep",
    )
    p.add_argument(
        "--features-dir",
        type=str,
        default="~/scratch/birdclef-2026/features/exp106",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load all data ONCE, reuse across all 5 sweep runs.
    print("Loading labeled soundscapes...")
    sc_data = load_labeled_soundscape(args.features_dir)
    print("Loading focal recordings...")
    focal_data = load_all_focal(args.features_dir)
    print("Loading unlabeled soundscapes...")
    unsup_data = load_all_unlabeled(args.features_dir, sc_data.site_to_idx)
    print(
        f"Data: {len(sc_data.embeddings)} labeled SC, "
        f"{len(focal_data.embeddings)} focal, "
        f"{len(unsup_data.embeddings)} unlabeled SC"
    )

    species_list = sc_data.species_list
    family_idx = build_family_index(species_list)

    results = []
    t0 = time.time()

    for lambda_fam in LAMBDA_FAM_VALUES:
        print(f"\n{'#'*60}")
        print(f"# Sweep run: lambda_fam = {lambda_fam}")
        print(f"{'#'*60}")

        # Re-seed each run so sampler/init are reproducible per lambda value.
        set_seed(args.seed)

        ckpt_dir = Path(args.checkpoint_root).expanduser() / f"lfam_{lambda_fam}"
        run_args = SimpleNamespace(
            fold=args.fold,
            lambda_site=args.lambda_site,
            lambda_xattn=args.lambda_xattn,
            lambda_fam=lambda_fam,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
            unsup_warmup_start=args.unsup_warmup_start,
            unsup_warmup_end=args.unsup_warmup_end,
            unsup_batch_recordings=args.unsup_batch_recordings,
            unsup_batch_windows_per_recording=args.unsup_batch_windows_per_recording,
            val_every_steps=100,
            checkpoint_dir=str(ckpt_dir),
            features_dir=args.features_dir,
            seed=args.seed,
            log_dir=str(ckpt_dir / "runs"),
        )

        result = train_one_fold(
            fold_site=args.fold,
            sc_data=sc_data,
            focal_data=focal_data,
            unsup_data=unsup_data,
            species_list=species_list,
            family_idx=family_idx,
            args=run_args,
            device=device,
            writer=None,  # sweep: one TB dir per run via train_one_fold logs
            global_prefix="",
        )
        result["lambda_fam"] = lambda_fam
        results.append(result)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Sweep complete in {elapsed/60:.1f}m")
    print(f"{'='*60}")
    print(f"{'lambda_fam':>12} | {'best_auroc':>10} | {'best_step':>10}")
    print("-" * 42)
    for r in results:
        print(
            f"{r['lambda_fam']:>12.4g} | {r['best_auroc']:>10.4f} | {r['best_step']:>10}"
        )

    # Save sweep summary
    out_path = Path(args.checkpoint_root).expanduser() / "sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
