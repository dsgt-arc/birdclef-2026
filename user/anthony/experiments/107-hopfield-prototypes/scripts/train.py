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
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from birdclef_107.cache import load_training_data_cached
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
    stratified_unsup_subset,
    weighted_focal_subset,
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
    p.add_argument("--k-species", type=int, default=8,
                   help="Number of prototypes per species in the Hopfield species bank.")
    p.add_argument("--k-site", type=int, default=8,
                   help="Number of prototypes per site in the Hopfield site bank.")
    p.add_argument("--d-hidden", type=int, default=1024,
                   help="Hidden width of the shared Perch projection MLP.")
    p.add_argument("--d-proj", type=int, default=512,
                   help="Shared projection / prototype dimension for species, site, and cross-attention modules.")
    p.add_argument("--cross-attention-heads", type=int, default=4,
                   help="Number of heads in species↔site cross-attention.")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="AdamW LR. Default 5e-4 (v3); v2 used 1e-3 but contrastive losses need slower LR.")
    p.add_argument("--weight-decay", type=float, default=1e-2,
                   help="AdamW weight decay. v4 default 1e-2 (was 1e-4); attacks labeled overfitting.")
    p.add_argument(
        "--site-blend-lr-mult",
        type=float,
        default=1.0,
        help="LR multiplier for the site-aware shared-fallback blend parameters (site_blend_logit).",
    )
    p.add_argument(
        "--site-blend-anchor-lambda",
        type=float,
        default=0.0,
        help="L2 penalty on site_blend_logit toward the 18% fallback prior. Useful when site_blend_lr_mult > 1.",
    )
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
    p.add_argument("--strong-site-dropout", type=float, default=0.0,
                   help="Probability of routing a strong labeled SC window through the site-free species path instead of the site-conditioned path.")
    p.add_argument("--strong-site-dropout-start", type=int, default=0,
                   help="Training step at which strong-site dropout turns on. Use >0 to delay this regularizer until after the early scaffold forms.")
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
        help="Persistent cache root (contains base-v1/ + idx-v1/). Omit to bypass the cache and read parquet directly.",
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
        help="Recompute the idx-v1/*.npy files even if they exist. Leaves base shards alone.",
    )
    p.add_argument(
        "--cache-force-rebuild-base",
        action="store_true",
        help="Rebuild the full base-v1/ from parquet. Use if a source parquet changed without its mtime advancing.",
    )
    p.add_argument(
        "--cache-subset-seed",
        type=int,
        default=None,
        help="Optional seed for focal/unlabeled cache subsampling. Defaults to --seed so training init and subset draw stay coupled unless explicitly separated.",
    )
    p.add_argument(
        "--include-perch-logits",
        action="store_true",
        help="Hydrate the optional perch_logits field on each dataclass. Triggers sidecar build on first use.",
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


def load_training_data(
    args: argparse.Namespace,
) -> tuple[LabeledSoundscapeData, FocalData, UnlabeledSoundscapeData]:
    """Load full data or go through the persistent sharded cache.

    With ``--cache-dir`` unset, read parquet directly (uncached path,
    useful for unit tests and ad-hoc full-corpus runs). Otherwise
    delegate to :func:`birdclef_107.cache.load_training_data_cached`,
    which keeps a sharded base on scratch and per-``(size, seed)``
    index files. See ``docs/cache-usage.md``.
    """
    if args.cache_dir is None:
        print("Loading labeled soundscapes...")
        sc_data = load_labeled_soundscape(args.features_dir)
        print("Loading focal recordings...")
        focal_data = load_all_focal(args.features_dir)
        print("Loading unlabeled soundscapes...")
        unsup_data = load_all_unlabeled(args.features_dir, sc_data.site_to_idx)
        return sc_data, focal_data, unsup_data

    cache_subset_seed = (
        int(args.seed) if args.cache_subset_seed is None else int(args.cache_subset_seed)
    )
    return load_training_data_cached(
        features_dir=args.features_dir,
        cache_root=args.cache_dir,
        focal_size=int(args.focal_cache_size),
        unsup_size=int(args.unsup_cache_size),
        seed=cache_subset_seed,
        include_perch_logits=bool(args.include_perch_logits),
        force_rebuild_base=bool(args.cache_force_rebuild_base),
        force_rebuild_index=bool(args.cache_force_rebuild),
        focal_subset_fn=weighted_focal_subset,
        unsup_subset_fn=stratified_unsup_subset,
    )


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
    train_sc_perch = None
    if sc_data.perch_logits is not None:
        train_sc_perch = sc_data.perch_logits[train_mask]
    train_sc_sites_str = sc_data.sites[train_mask]
    train_sc_site_idx = np.array(
        [sc_data.site_to_idx[str(s)] for s in train_sc_sites_str], dtype=np.int64
    )

    test_emb = sc_data.embeddings[test_mask]
    test_labels = sc_data.labels[test_mask]
    test_perch = None
    if sc_data.perch_logits is not None:
        test_perch = sc_data.perch_logits[test_mask]
    test_sites_str = sc_data.sites[test_mask]
    test_site_idx = np.array(
        [sc_data.site_to_idx[str(s)] for s in test_sites_str], dtype=np.int64
    )

    # Datasets
    strong_ds = EmbeddingDataset(
        train_sc_emb,
        train_sc_labels,
        train_sc_site_idx,
        perch_logits=train_sc_perch,
    )
    focal_ds = EmbeddingDataset(
        focal_data.embeddings,
        focal_data.labels,
        perch_logits=focal_data.perch_logits,
    )
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
        K_species=args.k_species,
        K_site=args.k_site,
        d_hidden=args.d_hidden,
        d_proj=args.d_proj,
        cross_attention_heads=args.cross_attention_heads,
        proj_dropout=args.proj_dropout,
        attn_dropout=args.attn_dropout,
        use_perch_logits=bool(args.include_perch_logits),
    ).to(device)

    family_idx_dev = family_idx.to(device)

    if args.site_blend_lr_mult == 1.0:
        optimizer = AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        blend_params = [model.site_blend_logit]
        if hasattr(model, "site_blend_delta_coef"):
            blend_params.append(model.site_blend_delta_coef)
        blend_param_ids = {id(p) for p in blend_params}
        base_params = [
            p for p in model.parameters() if id(p) not in blend_param_ids
        ]
        optimizer = AdamW(
            [
                {
                    "params": base_params,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                },
                {
                    "params": blend_params,
                    "lr": args.lr * args.site_blend_lr_mult,
                    "weight_decay": args.weight_decay,
                },
            ]
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
    site_blend_anchor_logit = math.log(0.18 / (1.0 - 0.18))

    def run_validation() -> float:
        """Validate on held-out site and return AUROC."""
        model.eval()
        preds = predict_fold(
            model,
            test_emb,
            test_site_idx,
            perch_logits=test_perch,
            device=str(device),
        )
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
            strong_site_dropout = (
                args.strong_site_dropout
                if global_step >= args.strong_site_dropout_start
                else 0.0
            )
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
                strong_site_dropout=strong_site_dropout,
            )
            if args.site_blend_anchor_lambda > 0.0:
                total_loss = total_loss + args.site_blend_anchor_lambda * (
                    (model.site_blend_logit - site_blend_anchor_logit) ** 2
                ).mean()
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
    best_preds = predict_fold(
        model,
        test_emb,
        test_site_idx,
        perch_logits=test_perch,
        device=str(device),
    )
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
