#!/usr/bin/env python3
"""Training script for experiment 107: Hopfield prototype retrieval.

Usage:
    python scripts/train.py --fold S22 --lambda-fam 1e-2
    python scripts/train.py --fold all --lambda-fam 1e-2

Runs one LOSO fold (or all folds sequentially). Each fold trains on all
labeled + focal + unlabeled data except the held-out site's labeled
windows, then evaluates on the held-out site.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from birdclef_107.data import (
    EmbeddingDataset,
    UnsupervisedDataset,
    build_focal_sampler,
    collate_triple_batch,
    load_all_focal,
    load_all_unlabeled,
    load_labeled_soundscape,
    loso_folds,
)
from birdclef_107.eval import (
    compute_auroc,
    evaluate_loso,
    format_results,
    predict_fold,
)
from birdclef_107.family_utils import build_family_index
from birdclef_107.losses import compute_loss
from birdclef_107.model import PrototypeRetrievalModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train exp 107 LOSO CV")
    p.add_argument(
        "--fold",
        type=str,
        default="S22",
        help="Site ID to hold out (e.g. S22), or 'all' for full LOSO CV",
    )
    p.add_argument("--lambda-site", type=float, default=0.1)
    p.add_argument("--lambda-fam", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--val-every-steps",
        type=int,
        default=100,
        help="Run validation every N training steps (not just end of epoch)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="~/scratch/birdclef-2026/checkpoints/exp107",
    )
    p.add_argument(
        "--features-dir",
        type=str,
        default="~/scratch/birdclef-2026/features/exp106",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log dir (default: checkpoint-dir/runs/)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(
    fold_site: str,
    sc_data,
    focal_data,
    unsup_data,
    species_list: list[str],
    family_idx: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    writer: SummaryWriter | None = None,
    global_prefix: str = "",
) -> dict:
    """Train and evaluate a single LOSO fold.

    Returns dict with fold results including best val AUROC and checkpoint path.
    """
    # Build LOSO split on labeled soundscapes
    folds = loso_folds(sc_data.sites)
    fold = None
    for f in folds:
        if f.left_out_site == fold_site:
            fold = f
            break
    if fold is None:
        raise ValueError(
            f"Site {fold_site} not found in labeled data. "
            f"Available: {[f.left_out_site for f in folds]}"
        )

    # Split labeled soundscape data
    train_sc_emb = sc_data.embeddings[fold.train_mask]
    train_sc_labels = sc_data.labels[fold.train_mask]
    train_sc_sites_str = sc_data.sites[fold.train_mask]
    train_sc_site_idx = np.array(
        [sc_data.site_to_idx[str(s)] for s in train_sc_sites_str], dtype=np.int64
    )

    test_emb = sc_data.embeddings[fold.test_mask]
    test_labels = sc_data.labels[fold.test_mask]
    test_sites_str = sc_data.sites[fold.test_mask]
    test_site_idx = np.array(
        [sc_data.site_to_idx[str(s)] for s in test_sites_str], dtype=np.int64
    )

    # Datasets
    strong_ds = EmbeddingDataset(train_sc_emb, train_sc_labels, train_sc_site_idx)
    focal_ds = EmbeddingDataset(focal_data.embeddings, focal_data.labels)
    unsup_ds = UnsupervisedDataset(unsup_data.embeddings, unsup_data.site_indices)

    # Dataloaders
    strong_loader = DataLoader(
        strong_ds,
        batch_size=min(args.batch_size, len(strong_ds)),
        shuffle=True,
        drop_last=False,
    )
    focal_sampler = build_focal_sampler(focal_data.labels)
    focal_loader = DataLoader(
        focal_ds,
        batch_size=args.batch_size,
        sampler=focal_sampler,
        drop_last=True,
    )
    unsup_loader = DataLoader(
        unsup_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Model
    n_sites = len(sc_data.site_to_idx)
    model = PrototypeRetrievalModel(
        n_species=len(species_list),
        n_sites=n_sites,
    ).to(device)

    family_idx_dev = family_idx.to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Steps per epoch = one full pass through the focal loader (largest stream).
    # This gives focal's 265K windows meaningful coverage — the proposal's
    # thesis relies on focal shaping M_species, which requires enough
    # gradient exposure.  Strong SC (262 windows) and unsup SC (127K) cycle
    # independently inside each epoch.
    steps_per_epoch = max(1, len(focal_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)

    # Checkpoint directory
    ckpt_dir = Path(args.checkpoint_dir).expanduser() / fold_site
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = -1.0
    best_step = -1
    patience_counter = 0
    global_step = 0

    def run_validation() -> float:
        """Validate on held-out site and return AUROC."""
        model.eval()
        preds = predict_fold(model, test_emb, test_site_idx, device=str(device))
        auroc = compute_auroc(test_labels, preds)
        model.train()
        return auroc

    print(f"\n{'='*60}")
    print(f"Fold: {fold_site} | train={fold.train_mask.sum()}, test={fold.test_mask.sum()}")
    print(f"Strong: {len(strong_ds)}, Focal: {len(focal_ds)}, Unsup: {len(unsup_ds)}")
    print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"{'='*60}")

    early_stop = False
    for epoch in range(args.epochs):
        if early_stop:
            break
        model.train()
        epoch_losses = {"labeled": 0.0, "site": 0.0, "family": 0.0, "total": 0.0}
        n_steps = 0

        strong_iter = iter(strong_loader)
        unsup_iter = iter(unsup_loader)

        for focal_batch in focal_loader:
            # Get strong batch (cycle — strong is tiny, ~2 batches per epoch
            # means strong SC windows are seen many times per epoch)
            try:
                strong_batch = next(strong_iter)
            except StopIteration:
                strong_iter = iter(strong_loader)
                strong_batch = next(strong_iter)

            # Get unsup batch (cycle if exhausted)
            try:
                unsup_batch = next(unsup_iter)
            except StopIteration:
                unsup_iter = iter(unsup_loader)
                unsup_batch = next(unsup_iter)

            # Move to device
            for batch in [strong_batch, focal_batch, unsup_batch]:
                for k, v in batch.items():
                    batch[k] = v.to(device)

            triple = collate_triple_batch(strong_batch, focal_batch, unsup_batch)

            optimizer.zero_grad()
            total_loss, components = compute_loss(
                model,
                triple,
                family_idx_dev,
                lambda_site=args.lambda_site,
                lambda_fam=args.lambda_fam,
            )
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses["total"] += total_loss.item()
            for k in ["labeled", "site", "family"]:
                epoch_losses[k] += components[k]
            n_steps += 1
            global_step += 1

            # Intra-epoch validation
            if global_step % args.val_every_steps == 0:
                val_auroc = run_validation()
                if writer is not None:
                    prefix = f"{global_prefix}/" if global_prefix else ""
                    writer.add_scalar(f"{prefix}val/auroc", val_auroc, global_step)
                    writer.add_scalar(f"{prefix}loss/total_step", total_loss.item(), global_step)
                    writer.add_scalar(f"{prefix}loss/labeled_step", components["labeled"], global_step)
                    writer.add_scalar(f"{prefix}loss/site_step", components["site"], global_step)
                    writer.add_scalar(f"{prefix}lr", scheduler.get_last_lr()[0], global_step)

                print(
                    f"  step {global_step:5d} (ep {epoch}) | "
                    f"loss={total_loss.item():.4f} "
                    f"(lab={components['labeled']:.4f} "
                    f"site={components['site']:.4f} "
                    f"fam={components['family']:.4f}) | "
                    f"val_auroc={val_auroc:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

                # Early stopping + checkpointing (step-level)
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_step = global_step
                    patience_counter = 0
                    torch.save(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_auroc": val_auroc,
                            "fold_site": fold_site,
                            "args": vars(args),
                        },
                        ckpt_dir / "best.pt",
                    )
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(
                        f"  Early stopping at step {global_step} "
                        f"(best={best_step}, auroc={best_auroc:.4f})"
                    )
                    early_stop = True
                    break

        # End-of-epoch averaged losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_steps, 1)
        print(
            f"  === Epoch {epoch} done | "
            f"avg_loss={epoch_losses['total']:.4f} "
            f"(lab={epoch_losses['labeled']:.4f} "
            f"site={epoch_losses['site']:.4f}) ==="
        )

    # Save predictions from best model for pooled evaluation
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()
    best_preds = predict_fold(model, test_emb, test_site_idx, device=str(device))
    np.save(ckpt_dir / "preds.npy", best_preds)
    np.save(ckpt_dir / "labels.npy", test_labels)

    print(f"  Best: step={best_step}, val_auroc={best_auroc:.4f}")

    return {
        "fold_site": fold_site,
        "best_step": best_step,
        "best_auroc": best_auroc,
        "n_train": int(fold.train_mask.sum()),
        "n_test": int(fold.test_mask.sum()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load all data once (shared across folds)
    print("Loading labeled soundscapes...")
    sc_data = load_labeled_soundscape(args.features_dir)
    species_list = sc_data.species_list

    print("Loading focal recordings...")
    focal_data = load_all_focal(args.features_dir)

    print("Loading unlabeled soundscapes...")
    unsup_data = load_all_unlabeled(args.features_dir, sc_data.site_to_idx)

    print(
        f"Data: {len(sc_data.embeddings)} labeled SC, "
        f"{len(focal_data.embeddings)} focal, "
        f"{len(unsup_data.embeddings)} unlabeled SC"
    )

    # Family index
    family_idx = build_family_index(species_list)

    # Determine which folds to run
    folds = loso_folds(sc_data.sites)
    if args.fold == "all":
        fold_sites = [f.left_out_site for f in folds]
    else:
        fold_sites = [args.fold]

    # TensorBoard
    log_dir = args.log_dir or str(
        Path(args.checkpoint_dir).expanduser() / "runs"
    )
    run_name = f"lfam{args.lambda_fam}_lsite{args.lambda_site}"
    writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")

    # Focal species for subset evaluation
    from birdclef_106.data_loading import list_focal_species
    focal_species_set = set(list_focal_species(args.features_dir))

    # Train each fold
    all_results = []
    fold_y_true = []
    fold_y_pred = []

    t0 = time.time()
    for fold_site in fold_sites:
        result = train_one_fold(
            fold_site=fold_site,
            sc_data=sc_data,
            focal_data=focal_data,
            unsup_data=unsup_data,
            species_list=species_list,
            family_idx=family_idx,
            args=args,
            device=device,
            writer=writer,
            global_prefix=fold_site,
        )
        all_results.append(result)

        # Load saved predictions for pooled evaluation
        ckpt_dir = Path(args.checkpoint_dir).expanduser() / fold_site
        fold_y_pred.append(np.load(ckpt_dir / "preds.npy"))
        fold_y_true.append(np.load(ckpt_dir / "labels.npy"))

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Pooled LOSO evaluation
    if len(fold_sites) > 1:
        loso_results = evaluate_loso(
            folds=[f for f in folds if f.left_out_site in fold_sites],
            fold_y_true=fold_y_true,
            fold_y_pred=fold_y_pred,
            species_list=species_list,
            focal_species=focal_species_set,
        )
        print(f"\n{'='*60}")
        print("LOSO Evaluation Results")
        print(f"{'='*60}")
        print(format_results(loso_results))

        # Save results JSON
        results_path = Path(args.checkpoint_dir).expanduser() / "loso_results.json"
        with open(results_path, "w") as f:
            json.dump(loso_results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    else:
        # Single fold — just report
        r = all_results[0]
        print(f"\nSingle fold {r['fold_site']}: val_auroc={r['best_auroc']:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
