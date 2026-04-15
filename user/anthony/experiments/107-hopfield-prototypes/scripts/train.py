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
    FocalData,
    LabeledSoundscapeData,
    RecordingGroupedBatchSampler,
    UnlabeledSoundscapeData,
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
        help="Site ID to hold out (e.g. S22), comma-separated holdout sites for a custom dev split (e.g. S13,S15,S19), or 'all' for full LOSO CV",
    )
    p.add_argument(
        "--split-name",
        type=str,
        default=None,
        help="Optional output name for a custom split (defaults to --fold with commas replaced by '+').",
    )
    p.add_argument("--lambda-site", type=float, default=0.01,
                   help="Weight on L_site_contrastive (InfoNCE on M_site)")
    p.add_argument("--lambda-xattn", type=float, default=0.01,
                   help="Weight on L_xattn_contrastive (site-InfoNCE on cross-attn profiles)")
    p.add_argument("--lambda-fam", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=20,
                   help="v4 default 20 (scheduler gets full LR range; regularization lets peak appear later than epoch 2).")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="AdamW LR. Default 5e-4 (v3); v2 used 1e-3 but contrastive losses need slower LR.")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="AdamW weight decay. v4 default 1e-2 (was 1e-4); attacks labeled overfitting.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=10,
                   help="Val-AUROC patience checks. Default 10 (v3); contrastive objectives take longer to find non-trivial optima than v2's softplus.")
    p.add_argument("--strong-stride", type=int, default=4,
                   help="v4: compute strong-SC BCE only every N steps (default 4). Strong loader has ~2 batches, so steps_per_epoch/strong_stride ~= 260 strong passes/epoch instead of 1038. Attacks strong-SC memorization.")
    p.add_argument("--attn-dropout", type=float, default=0.1,
                   help="v4: dropout on cross-attention. Directly regularizes the strong-SC path.")
    p.add_argument("--proj-dropout", type=float, default=0.1,
                   help="v4: dropout inside PerchProjection MLP. Regularizes the shared trunk.")
    p.add_argument("--label-smoothing", type=float, default=0.01,
                   help="v4: BCE label smoothing ε. Small (0.01) because multi-hot binary labels.")
    p.add_argument("--unsup-warmup-start", type=int, default=200,
                   help="Step to begin ramping unsup lambdas from 0.")
    p.add_argument("--unsup-warmup-end", type=int, default=500,
                   help="Step at which unsup lambdas reach their target values.")
    p.add_argument("--unsup-batch-recordings", type=int, default=16,
                   help="R in RecordingGroupedBatchSampler — recordings per unsup batch.")
    p.add_argument("--unsup-batch-windows-per-recording", type=int, default=8,
                   help="N in RecordingGroupedBatchSampler — windows per recording per batch.")
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
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional directory for a local NPZ cache of the dev subset (e.g. /tmp/birdclef-107-devcache).",
    )
    p.add_argument(
        "--focal-cache-size",
        type=int,
        default=0,
        help="If >0, cache/load only this many focal windows using rare-species-aware weighted sampling without replacement.",
    )
    p.add_argument(
        "--unsup-cache-size",
        type=int,
        default=0,
        help="If >0, cache/load only this many unlabeled soundscape windows using stratified-by-site sampling.",
    )
    p.add_argument(
        "--cache-force-rebuild",
        action="store_true",
        help="Rebuild the local cache even if NPZ files already exist.",
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


def unsup_warmup_factor(step: int, start: int, end: int) -> float:
    """Linear ramp from 0 → 1 over [start, end].

    Used to schedule unsupervised lambdas: zero for steps < start (let
    labeled BCE place the projection alone), then linear ramp to target
    by step `end`. Without warmup, the v2 contrastive losses can dominate
    early gradient and prevent labeled BCE from finding useful structure.
    """
    if step < start:
        return 0.0
    if step >= end:
        return 1.0
    return (step - start) / max(1, end - start)


def _weighted_focal_subset(labels: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Rare-species-aware focal subset without replacement."""
    n = len(labels)
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n, dtype=np.int64)

    species_counts = labels.sum(axis=0).astype(np.float64)
    species_counts = np.maximum(species_counts, 1.0)
    species_weights = 1.0 / species_counts
    per_row = np.maximum((labels * species_weights[None, :]).max(axis=1), 1e-12)
    probs = per_row / per_row.sum()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_rows, replace=False, p=probs))


def _stratified_unsup_subset(site_indices: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    """Site-stratified unlabeled subset for fast local dev runs."""
    n = len(site_indices)
    if max_rows <= 0 or max_rows >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    unique_sites, counts = np.unique(site_indices, return_counts=True)
    raw = counts.astype(np.float64) * (max_rows / counts.sum())
    quotas = np.floor(raw).astype(int)
    quotas = np.minimum(quotas, counts)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, counts)

    total = int(quotas.sum())
    if total < max_rows:
        frac_order = np.argsort(-(raw - np.floor(raw)))
        for idx in frac_order:
            if total >= max_rows:
                break
            if quotas[idx] < counts[idx]:
                quotas[idx] += 1
                total += 1
    elif total > max_rows:
        frac_order = np.argsort(raw - np.floor(raw))
        for idx in frac_order:
            if total <= max_rows:
                break
            if quotas[idx] > 1:
                quotas[idx] -= 1
                total -= 1
        if total > max_rows:
            for idx in range(len(quotas)):
                if total <= max_rows:
                    break
                while quotas[idx] > 1 and total > max_rows:
                    quotas[idx] -= 1
                    total -= 1

    picks: list[np.ndarray] = []
    for site, quota in zip(unique_sites, quotas):
        pool = np.flatnonzero(site_indices == site)
        picks.append(rng.choice(pool, size=int(quota), replace=False))
    return np.sort(np.concatenate(picks))


def _save_dev_cache(
    cache_dir: Path,
    sc_data: LabeledSoundscapeData,
    focal_data: FocalData,
    unsup_data: UnlabeledSoundscapeData,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_dir / "labeled_soundscape.npz",
        embeddings=sc_data.embeddings,
        labels=sc_data.labels,
        sites=sc_data.sites.astype("U"),
        filenames=np.asarray(sc_data.filenames, dtype="U"),
        species_list=np.asarray(sc_data.species_list, dtype="U"),
        site_keys=np.asarray(list(sc_data.site_to_idx.keys()), dtype="U"),
        site_vals=np.asarray(list(sc_data.site_to_idx.values()), dtype=np.int64),
    )
    np.savez(
        cache_dir / "focal.npz",
        embeddings=focal_data.embeddings,
        labels=focal_data.labels,
        species_list=np.asarray(focal_data.species_list, dtype="U"),
    )
    np.savez(
        cache_dir / "unlabeled_soundscape.npz",
        embeddings=unsup_data.embeddings,
        site_indices=unsup_data.site_indices,
        recording_indices=unsup_data.recording_indices,
        site_keys=np.asarray(list(unsup_data.site_to_idx.keys()), dtype="U"),
        site_vals=np.asarray(list(unsup_data.site_to_idx.values()), dtype=np.int64),
        recording_keys=np.asarray(list(unsup_data.recording_to_idx.keys()), dtype="U"),
        recording_vals=np.asarray(list(unsup_data.recording_to_idx.values()), dtype=np.int64),
    )


def _load_dev_cache(cache_dir: Path) -> tuple[LabeledSoundscapeData, FocalData, UnlabeledSoundscapeData]:
    sc_npz = np.load(cache_dir / "labeled_soundscape.npz", allow_pickle=False)
    focal_npz = np.load(cache_dir / "focal.npz", allow_pickle=False)
    unsup_npz = np.load(cache_dir / "unlabeled_soundscape.npz", allow_pickle=False)

    sc_data = LabeledSoundscapeData(
        embeddings=sc_npz["embeddings"],
        labels=sc_npz["labels"],
        sites=sc_npz["sites"],
        filenames=sc_npz["filenames"].tolist(),
        species_list=sc_npz["species_list"].tolist(),
        site_to_idx={k: int(v) for k, v in zip(sc_npz["site_keys"].tolist(), sc_npz["site_vals"].tolist())},
    )
    focal_data = FocalData(
        embeddings=focal_npz["embeddings"],
        labels=focal_npz["labels"],
        species_list=focal_npz["species_list"].tolist(),
    )
    unsup_data = UnlabeledSoundscapeData(
        embeddings=unsup_npz["embeddings"],
        site_indices=unsup_npz["site_indices"],
        recording_indices=unsup_npz["recording_indices"],
        site_to_idx={k: int(v) for k, v in zip(unsup_npz["site_keys"].tolist(), unsup_npz["site_vals"].tolist())},
        recording_to_idx={k: int(v) for k, v in zip(unsup_npz["recording_keys"].tolist(), unsup_npz["recording_vals"].tolist())},
    )
    return sc_data, focal_data, unsup_data


def load_training_data(args: argparse.Namespace) -> tuple[LabeledSoundscapeData, FocalData, UnlabeledSoundscapeData]:
    """Load full data or materialize/load a fast local dev cache."""
    if args.cache_dir is None:
        print("Loading labeled soundscapes...")
        sc_data = load_labeled_soundscape(args.features_dir)
        print("Loading focal recordings...")
        focal_data = load_all_focal(args.features_dir)
        print("Loading unlabeled soundscapes...")
        unsup_data = load_all_unlabeled(args.features_dir, sc_data.site_to_idx)
        return sc_data, focal_data, unsup_data

    cache_dir = Path(args.cache_dir).expanduser()
    manifest_path = cache_dir / "manifest.json"
    cache_ready = (
        manifest_path.exists()
        and (cache_dir / "labeled_soundscape.npz").exists()
        and (cache_dir / "focal.npz").exists()
        and (cache_dir / "unlabeled_soundscape.npz").exists()
    )

    if cache_ready and not args.cache_force_rebuild:
        manifest = json.loads(manifest_path.read_text())
        expected = {
            "features_dir": args.features_dir,
            "seed": args.seed,
            "focal_cache_size": int(args.focal_cache_size),
            "unsup_cache_size": int(args.unsup_cache_size),
        }
        if all(manifest.get(k) == v for k, v in expected.items()):
            print(f"Loading dev cache from {cache_dir}")
            return _load_dev_cache(cache_dir)
        print(f"Cache config changed; rebuilding {cache_dir}")

    print(f"Building dev cache at {cache_dir} (first run is slow; later runs reuse NPZ files)")
    print("Loading labeled soundscapes...")
    sc_data = load_labeled_soundscape(args.features_dir)
    print("Loading focal recordings...")
    focal_full = load_all_focal(args.features_dir)
    print("Loading unlabeled soundscapes...")
    unsup_full = load_all_unlabeled(args.features_dir, sc_data.site_to_idx)

    focal_idx = _weighted_focal_subset(focal_full.labels, args.focal_cache_size, args.seed)
    unsup_idx = _stratified_unsup_subset(unsup_full.site_indices, args.unsup_cache_size, args.seed)

    focal_data = FocalData(
        embeddings=focal_full.embeddings[focal_idx],
        labels=focal_full.labels[focal_idx],
        species_list=focal_full.species_list,
    )
    unsup_data = UnlabeledSoundscapeData(
        embeddings=unsup_full.embeddings[unsup_idx],
        site_indices=unsup_full.site_indices[unsup_idx],
        recording_indices=unsup_full.recording_indices[unsup_idx],
        site_to_idx=unsup_full.site_to_idx,
        recording_to_idx=unsup_full.recording_to_idx,
    )

    _save_dev_cache(cache_dir, sc_data, focal_data, unsup_data)
    manifest = {
        "features_dir": args.features_dir,
        "seed": args.seed,
        "focal_cache_size": int(args.focal_cache_size),
        "unsup_cache_size": int(args.unsup_cache_size),
        "n_labeled": int(len(sc_data.embeddings)),
        "n_focal": int(len(focal_data.embeddings)),
        "n_unsup": int(len(unsup_data.embeddings)),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return sc_data, focal_data, unsup_data


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
    """Train and evaluate a single LOSO fold or custom multi-site holdout.

    Returns dict with fold results including best val AUROC and checkpoint path.
    """
    held_out_sites = [s.strip() for s in fold_site.split(",") if s.strip()]
    split_name = args.split_name or "+".join(held_out_sites)

    # Build LOSO split on labeled soundscapes, or a custom multi-site holdout.
    folds = loso_folds(sc_data.sites)
    available_sites = {f.left_out_site for f in folds}
    missing_sites = [s for s in held_out_sites if s not in available_sites]
    if missing_sites:
        raise ValueError(
            f"Site(s) {missing_sites} not found in labeled data. "
            f"Available: {sorted(available_sites)}"
        )

    if len(held_out_sites) == 1:
        fold = next(f for f in folds if f.left_out_site == held_out_sites[0])
        train_mask = fold.train_mask
        test_mask = fold.test_mask
    else:
        test_mask = np.isin(sc_data.sites, held_out_sites)
        train_mask = ~test_mask

    # Split labeled soundscape data
    train_sc_emb = sc_data.embeddings[train_mask]
    train_sc_labels = sc_data.labels[train_mask]
    train_sc_sites_str = sc_data.sites[train_mask]
    train_sc_site_idx = np.array(
        [sc_data.site_to_idx[str(s)] for s in train_sc_sites_str], dtype=np.int64
    )

    test_emb = sc_data.embeddings[test_mask]
    test_labels = sc_data.labels[test_mask]
    test_sites_str = sc_data.sites[test_mask]
    test_site_idx = np.array(
        [sc_data.site_to_idx[str(s)] for s in test_sites_str], dtype=np.int64
    )

    # Datasets
    strong_ds = EmbeddingDataset(train_sc_emb, train_sc_labels, train_sc_site_idx)
    focal_ds = EmbeddingDataset(focal_data.embeddings, focal_data.labels)
    unsup_ds = UnsupervisedDataset(
        unsup_data.embeddings,
        unsup_data.site_indices,
        recording_indices=unsup_data.recording_indices,
    )

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
    # v3: recording-grouped batches for the unsup stream — gives the
    # contrastive losses ~7 same-site positives + ~112 cross-site
    # negatives per anchor at default R=16, N=8.
    unsup_batch_sampler = RecordingGroupedBatchSampler(
        unsup_data.recording_indices,
        recordings_per_batch=args.unsup_batch_recordings,
        windows_per_recording=args.unsup_batch_windows_per_recording,
        seed=args.seed,
    )
    unsup_loader = DataLoader(
        unsup_ds,
        batch_sampler=unsup_batch_sampler,
    )

    # Model
    n_sites = len(sc_data.site_to_idx)
    model = PrototypeRetrievalModel(
        n_species=len(species_list),
        n_sites=n_sites,
        proj_dropout=args.proj_dropout,
        attn_dropout=args.attn_dropout,
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
    ckpt_dir = Path(args.checkpoint_dir).expanduser() / split_name
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
    print(f"Fold: {split_name} | holdout_sites={held_out_sites} | train={train_mask.sum()}, test={test_mask.sum()}")
    print(f"Strong: {len(strong_ds)}, Focal: {len(focal_ds)}, Unsup: {len(unsup_ds)}")
    print(f"Steps per epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"{'='*60}")

    early_stop = False
    for epoch in range(args.epochs):
        if early_stop:
            break
        model.train()
        epoch_losses = {
            "labeled": 0.0,
            "site": 0.0,
            "site_xattn": 0.0,
            "family": 0.0,
            "total": 0.0,
        }
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

            warmup = unsup_warmup_factor(
                global_step, args.unsup_warmup_start, args.unsup_warmup_end
            )
            # v4: cap strong-SC oversampling. strong_loader has ~2 batches
            # cycled every focal step → each window seen ~519×/epoch. Only
            # compute the strong BCE every Nth step so total passes per
            # window drop by N.
            include_strong = (global_step % args.strong_stride == 0)
            optimizer.zero_grad()
            total_loss, components = compute_loss(
                model,
                triple,
                family_idx_dev,
                lambda_site=args.lambda_site * warmup,
                lambda_xattn=args.lambda_xattn * warmup,
                lambda_fam=args.lambda_fam,
                label_smoothing=args.label_smoothing,
                include_strong=include_strong,
            )
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses["total"] += total_loss.item()
            for k in ["labeled", "site", "site_xattn", "family"]:
                epoch_losses[k] += components[k]
            n_steps += 1
            global_step += 1

            # Intra-epoch validation
            if global_step % args.val_every_steps == 0:
                val_auroc = run_validation()
                # Ratio of weighted unsup loss to labeled loss — flag if
                # unsup dominates >3× during the peak labeled phase.
                weighted_unsup = (
                    args.lambda_site * warmup * components["site"]
                    + args.lambda_xattn * warmup * components["site_xattn"]
                )
                ratio = weighted_unsup / max(components["labeled"], 1e-8)
                if writer is not None:
                    prefix = f"{global_prefix}/" if global_prefix else ""
                    writer.add_scalar(f"{prefix}val/auroc", val_auroc, global_step)
                    writer.add_scalar(f"{prefix}loss/total_step", total_loss.item(), global_step)
                    writer.add_scalar(f"{prefix}loss/labeled_step", components["labeled"], global_step)
                    writer.add_scalar(f"{prefix}loss/site_step", components["site"], global_step)
                    writer.add_scalar(f"{prefix}loss/site_xattn_step", components["site_xattn"], global_step)
                    writer.add_scalar(f"{prefix}loss/unsup_to_labeled_ratio", ratio, global_step)
                    writer.add_scalar(f"{prefix}sched/unsup_warmup", warmup, global_step)
                    writer.add_scalar(f"{prefix}lr", scheduler.get_last_lr()[0], global_step)

                print(
                    f"  step {global_step:5d} (ep {epoch}) | "
                    f"loss={total_loss.item():.4f} "
                    f"(lab={components['labeled']:.4f} "
                    f"site={components['site']:.4f} "
                    f"xattn={components['site_xattn']:.4f} "
                    f"fam={components['family']:.4f}) | "
                    f"unsup/lab={ratio:.2f} warmup={warmup:.2f} | "
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
                            "fold_site": split_name,
                            "held_out_sites": held_out_sites,
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
            f"site={epoch_losses['site']:.4f} "
            f"xattn={epoch_losses['site_xattn']:.4f}) ==="
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
        "fold_site": split_name,
        "held_out_sites": held_out_sites,
        "best_step": best_step,
        "best_auroc": best_auroc,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
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
    sc_data, focal_data, unsup_data = load_training_data(args)
    species_list = sc_data.species_list

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
    run_name = f"{(args.split_name or args.fold).replace(',', '+')}_lfam{args.lambda_fam}_lsite{args.lambda_site}"
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
            global_prefix=(args.split_name or fold_site).replace(",", "+"),
        )
        all_results.append(result)

        # Load saved predictions for pooled evaluation
        ckpt_dir = Path(args.checkpoint_dir).expanduser() / result["fold_site"]
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
        # Single fold — overall + per-subset AUROC.
        # Subset breakdown is the diagnostic we need to decide whether
        # bird-only signals (e.g. Perch pseudo-labels) would target the
        # right bottleneck or whether the gap is on amphibians/insects
        # (in which case bird-only SSL won't help).
        r = all_results[0]
        from birdclef_107.eval import build_subset_masks, compute_auroc

        y_true = fold_y_true[0]
        y_pred = fold_y_pred[0]
        masks = build_subset_masks(species_list, focal_species=focal_species_set)
        subset_aurocs = {
            name: compute_auroc(y_true, y_pred, mask)
            for name, mask in masks.items()
            if mask.sum() > 0
        }

        print(f"\n{'='*60}")
        print(f"Single fold {r['fold_site']} | best step {r['best_step']}")
        print(f"{'='*60}")
        print(f"  Overall AUROC : {r['best_auroc']:.4f}")
        if r.get("held_out_sites") == ["S22"]:
            print(f"  Ridge baseline: 0.7490  (Δ = {r['best_auroc'] - 0.749:+.4f})")
            print(f"  v1-no-gate    : 0.7160  (Δ = {r['best_auroc'] - 0.716:+.4f})")
        print()
        print("Per-subset AUROC:")
        for name in ["bird", "amphibian", "insect", "with_focal", "without_focal"]:
            if name in subset_aurocs:
                n = int(masks[name].sum())
                print(f"  {name:>15s} ({n:3d} sp): {subset_aurocs[name]:.4f}")

        # Save subset breakdown JSON for downstream analysis.
        subset_path = Path(args.checkpoint_dir).expanduser() / r['fold_site'] / "subset_aurocs.json"
        with open(subset_path, "w") as f:
            json.dump(
                {
                    "fold_site": r["fold_site"],
                    "best_step": r["best_step"],
                    "held_out_sites": r.get("held_out_sites", [r["fold_site"]]),
                    "overall_auroc": r["best_auroc"],
                    "n_windows": int(len(y_true)),
                    "subset_auroc": subset_aurocs,
                    "subset_n_species": {
                        name: int(masks[name].sum()) for name in subset_aurocs
                    },
                    "args": vars(args),
                },
                f,
                indent=2,
            )
        print(f"\nSubset breakdown saved to {subset_path}")

    writer.close()


if __name__ == "__main__":
    main()
