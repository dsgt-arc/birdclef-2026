#!/usr/bin/env python3
"""Diagnostic: explain why exp 032 val/auroc plateaus around 0.08-0.11
while Tucker's published trajectory reaches 0.62 by epoch 1.

Hypothesis: torchmetrics 1.9 `MultilabelAUROC(average="macro")` includes
0-positive classes in the macro average (assigning them 0.0 or near),
dragging the mean down. Tucker's `compute_macro_auc` (notebook lines 369-384)
explicitly skips classes with `col.sum() == 0 or col.sum() == len(col)`.

Two diagnostics:
    1) Synthetic: build a y_true/y_pred pair matching exp 032's val stats
       (168 windows, 234 classes, sparse positives) and compute both metrics.
    2) Real (optional --ckpt): load a checkpoint, run inference on the val
       dataloader, dump per-class AUROC for evaluable classes and report
       both the torchmetrics number and Tucker's number side-by-side.

Run without args for synthetic-only:
    python scripts/diag_val_auroc.py --config configs/distilled_sed_control.yaml

Run with --ckpt for real-data verification:
    python scripts/diag_val_auroc.py \\
        --config configs/distilled_sed_control.yaml \\
        --ckpt /storage/project/.../checkpoints/2026-exp032-distilled-sed-control/best-epochepoch=05.ckpt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelAUROC

EXP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_ROOT.parents[3]
SRC_ROOT = EXP_ROOT.parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tucker's compute_macro_auc (verbatim port from notebook lines 369-384)
# ---------------------------------------------------------------------------


def tucker_compute_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """Returns (macro_auc, n_evaluable_classes).

    Skips classes with col.sum() == 0 OR col.sum() == len(col), i.e. classes
    where all targets are 0 or all are 1 (AUROC undefined in both cases).
    """
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


# ---------------------------------------------------------------------------
# Synthetic diagnostic
# ---------------------------------------------------------------------------


def synthetic_diagnostic(num_samples: int = 168, num_classes: int = 234) -> None:
    """Reproduce the bug with synthetic data matching exp 032's val statistics.

    The exp 032 val pool is 168 SS windows (per fast-dev-run logs). Each
    window has ~1-2 positive labels among 234 classes — so the multi-hot
    sparsity is ~1.5/234 ≈ 0.6%.
    """
    print("=" * 70)
    print("SYNTHETIC DIAGNOSTIC")
    print("=" * 70)
    print(f"Simulating exp 032 val: {num_samples} samples × {num_classes} classes")
    print("  Average positives per sample: ~1.5 (matches Tucker's val pool)")
    print()

    rng = np.random.default_rng(42)

    # Build y_true: sparse multi-hot. Roughly 1.5 positives per sample.
    p_positive = 1.5 / num_classes
    y_true = (rng.random((num_samples, num_classes)) < p_positive).astype(np.float32)

    n_positives_per_class = y_true.sum(axis=0)
    n_evaluable = int(((n_positives_per_class > 0) & (n_positives_per_class < num_samples)).sum())
    n_zero_positive = int((n_positives_per_class == 0).sum())

    print("Class coverage in synthetic val:")
    print(f"  Classes with ≥1 positive: {num_classes - n_zero_positive} / {num_classes}")
    print(f"  Classes with 0 positives: {n_zero_positive} / {num_classes}")
    print(f"  Evaluable for AUROC:      {n_evaluable}")
    print()

    # Build a y_pred that's mildly informative: y_true + noise. A real model
    # at epoch 0 of training is much weaker than this, but this is a charitable
    # case — if even an informative predictor scores low under torchmetrics
    # then the metric itself is mismeasuring.
    noise = rng.normal(0, 1.0, y_true.shape).astype(np.float32)
    logits = y_true * 2.0 + noise  # rough "trained" predictor: pos shifted by 2σ
    y_pred = 1 / (1 + np.exp(-logits))  # sigmoid

    # Tucker's metric (sklearn-based, skip empty)
    tucker_auc, tucker_n = tucker_compute_macro_auc(y_true, y_pred)

    # Torchmetrics MultilabelAUROC with average='macro'
    tm_metric = MultilabelAUROC(num_labels=num_classes, average="macro")
    tm_metric.update(torch.from_numpy(y_pred), torch.from_numpy(y_true).int())
    tm_auc = tm_metric.compute().item()

    # Torchmetrics with average='none' (per-class) — diagnose what it does
    # with 0-positive classes
    tm_metric_per_class = MultilabelAUROC(num_labels=num_classes, average="none")
    tm_metric_per_class.update(torch.from_numpy(y_pred), torch.from_numpy(y_true).int())
    per_class_auc = tm_metric_per_class.compute().cpu().numpy()

    nan_count = int(np.isnan(per_class_auc).sum())
    zero_count = int((per_class_auc == 0).sum())
    finite = per_class_auc[~np.isnan(per_class_auc)]

    print("Tucker's compute_macro_auc (skips 0-positive classes):")
    print(f"  macro_auc      = {tucker_auc:.4f}")
    print(f"  n_evaluable    = {tucker_n}")
    print()
    print("torchmetrics MultilabelAUROC(average='macro'):")
    print(f"  macro          = {tm_auc:.4f}")
    print()
    print("torchmetrics MultilabelAUROC(average='none'):")
    print(f"  total classes  = {num_classes}")
    print(f"  NaN classes    = {nan_count}")
    print(f"  zero-exact     = {zero_count}")
    print(f"  finite mean    = {finite.mean():.4f}")
    print("  → torchmetrics treats 0-positive classes as: ", end="")
    if zero_count > nan_count:
        print("0.0 (DRAGS macro DOWN)")
    elif nan_count > 0:
        print("NaN (should be excluded)")
    else:
        print("see distribution above")
    print()
    print("Verdict:")
    diff = tucker_auc - tm_auc
    if diff > 0.1:
        print(f"  ✓ HYPOTHESIS CONFIRMED — torchmetrics undercounts by {diff:+.4f}")
        print(f"    Tucker:     {tucker_auc:.4f}")
        print(f"    Our metric: {tm_auc:.4f}  ← what W&B shows")
    else:
        print(f"  Δ = {diff:+.4f} (small; metric implementations agree)")
    print()


# ---------------------------------------------------------------------------
# Real-data diagnostic (requires --ckpt)
# ---------------------------------------------------------------------------


def real_diagnostic(config_path: Path, ckpt_path: Path, num_workers: int = 4) -> None:
    """Load a checkpoint, run inference on val_dataloader, compute both metrics."""
    from birdclef.data.distilled_sed_datamodule import (
        DistilledSEDDataModule,
        build_sc_cache_meta,
    )
    from birdclef.data.soundscape import build_full_species_list
    from birdclef.encoders.efficientnet import EfficientNetEncoder
    from birdclef.heads.sed_head import AttHeadLogitSpace
    from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule
    from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

    print("=" * 70)
    print("REAL-DATA DIAGNOSTIC")
    print(f"  config: {config_path}")
    print(f"  ckpt:   {ckpt_path}")
    print("=" * 70)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Build species_to_idx the SAME way train.py does.
    data_dir = Path(cfg["data"]["data_dir"])
    species_path = data_dir / "taxonomy.csv"
    species_list = build_full_species_list(species_path)
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)
    print(f"Species: {num_classes} classes (from {species_path})")

    # Build encoder + sed_head + mel_transform exactly as train.py does, so the
    # checkpoint's state_dict drops in cleanly via load_from_checkpoint kwargs.
    encoder = EfficientNetEncoder(
        model_name=cfg["encoder"]["model_name"],
        pretrained=False,  # we'll overwrite with checkpoint weights
        drop_path_rate=cfg["encoder"].get("drop_path_rate", 0.0),
        in_chans=cfg["encoder"].get("in_chans", 1),
        use_post_head=cfg["encoder"].get("use_post_head", False),
    )
    sed_head = AttHeadLogitSpace(
        in_channels=encoder.embed_dim,
        num_classes=num_classes,
        hidden_dim=cfg["sed_head"]["hidden_dim"],
        dropout=cfg["sed_head"]["dropout"],
        gem_p=cfg["sed_head"]["gem_p"],
    )
    mel_cfg = cfg["mel"]
    sample_rate = cfg["data"]["sample_rate"]
    window_duration = cfg["data"]["window_duration"]
    target_frames = int(window_duration * sample_rate / mel_cfg["hop_length"])
    mel_transform = NNAudioMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=mel_cfg["n_mels"],
        target_frames=target_frames,
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        fmax=mel_cfg["fmax"],
        norm_type=mel_cfg.get("norm_type", "audioset"),
        top_db=mel_cfg["top_db"],
    )

    print("Loading checkpoint...")
    module = DistilledSEDFinetuneModule.load_from_checkpoint(
        ckpt_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        encoder=encoder,
        sed_head=sed_head,
        mel_transform=mel_transform,
        spec_augment=None,  # eval-only — SpecAugment is training-only anyway
    )
    module.eval()
    device = next(module.parameters()).device
    print(f"  loaded onto: {device}")

    # Build val dataloader (same logic as train.py lines 467-519)
    from birdclef.data.distilled_sed_dataset import _default_audio_loader as _audio_loader

    focal_df = pd.read_csv(data_dir / "train.csv")
    sc_labels_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    labeled_sc_files = sorted(sc_labels_df["filename"].unique().tolist())
    sc_df = build_sc_cache_meta(labeled_sc_files)

    focal_audio_dir = data_dir / "train_audio"
    sc_audio_dir = data_dir / "train_soundscapes"

    def _focal_loader(path, n_samples, start=0, sr=sample_rate):
        return _audio_loader(focal_audio_dir / path, n_samples, start, sr)

    def _sc_loader(path, n_samples, start=0, sr=sample_rate):
        return _audio_loader(sc_audio_dir / path, n_samples, start, sr)

    mixup_cfg = cfg.get("mixup", {}) or {}
    focal_ds_kwargs = {
        "mixup_prob": mixup_cfg.get("focal_focal_prob", 0.5),
        "mixup_alpha": mixup_cfg.get("alpha", 0.4),
        "mixup_hard": mixup_cfg.get("hard", True),
        "focal_sc_mixup_prob": mixup_cfg.get("focal_sc_prob", 0.5),
        "focal_sc_mixup_alpha": mixup_cfg.get("focal_sc_alpha", 0.4),
        "aug_prob": mixup_cfg.get("aug_prob", 0.5),
    }
    window_samples = int(window_duration * sample_rate)
    data_cfg = cfg["data"]
    dm = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        fold_k=0,
        n_folds=data_cfg.get("n_folds", 5),
        focal_seed=data_cfg.get("seed", 42),
        min_sample=data_cfg.get("min_sample", 20),
        window_samples=window_samples,
        sample_rate=sample_rate,
        batch_size=data_cfg["batch_size"],
        focal_share=data_cfg.get("focal_share", 0.9),
        sc_share=data_cfg.get("sc_share", 0.1),
        steps_per_epoch=data_cfg.get("steps_per_epoch"),
        num_workers=num_workers,
        focal_audio_loader=_focal_loader,
        sc_audio_loader=_sc_loader,
        focal_ds_kwargs=focal_ds_kwargs,
        seed_epoch_shift=data_cfg.get("seed_epoch_shift", True),
        base_seed=data_cfg.get("base_seed", 42),
    )
    dm.setup("fit")
    val_dl = dm.val_dataloader()
    print(f"Val dataloader: {len(val_dl.dataset)} samples")

    # Inference loop — collect y_true and y_pred (probability space, post-blend)
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dl:
            wav = batch[0].to(device)
            lb = batch[1]
            mel = module._compute_mel(wav, training=False)
            clip_logits, frame_logits = module(mel)
            frame_max = frame_logits.max(dim=-1).values
            probs = module._blended_probs(clip_logits, frame_max)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(lb.numpy())
    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    print(f"Collected: y_true={y_true.shape}, y_pred={y_pred.shape}")
    print()

    # Class coverage
    n_pos_per_class = y_true.sum(axis=0)
    n_zero = int((n_pos_per_class == 0).sum())
    n_singleton = int((n_pos_per_class == 1).sum())
    n_evaluable = int(((n_pos_per_class > 0) & (n_pos_per_class < len(y_true))).sum())
    print("Class coverage in real val (fold 0):")
    print(f"  Classes with 0 positives:     {n_zero}")
    print(f"  Classes with 1 positive:      {n_singleton}")
    print(f"  Classes evaluable for AUROC:  {n_evaluable}")
    print(f"  Total positive labels:        {int(n_pos_per_class.sum())}")
    print()

    # Both metrics
    tucker_auc, tucker_n = tucker_compute_macro_auc(y_true, y_pred)

    tm_metric = MultilabelAUROC(num_labels=num_classes, average="macro")
    tm_metric.update(torch.from_numpy(y_pred), torch.from_numpy(y_true).int())
    tm_auc = tm_metric.compute().item()

    tm_per_class = MultilabelAUROC(num_labels=num_classes, average="none")
    tm_per_class.update(torch.from_numpy(y_pred), torch.from_numpy(y_true).int())
    per_class = tm_per_class.compute().cpu().numpy()

    nan_count = int(np.isnan(per_class).sum())
    zero_exact = int((per_class == 0).sum())

    print("Tucker's compute_macro_auc (skips 0-positive classes):")
    print(f"  macro_auc      = {tucker_auc:.4f}")
    print(f"  n_evaluable    = {tucker_n}")
    print()
    print("torchmetrics MultilabelAUROC(average='macro'):")
    print(f"  macro          = {tm_auc:.4f}  ← what W&B shows")
    print()
    print("torchmetrics MultilabelAUROC(average='none') diagnostic:")
    print(f"  total classes  = {num_classes}")
    print(f"  NaN entries    = {nan_count}")
    print(f"  zero entries   = {zero_exact}")
    finite = per_class[~np.isnan(per_class)]
    print(f"  finite mean    = {finite.mean():.4f}")
    print()
    print("Verdict:")
    diff = tucker_auc - tm_auc
    if diff > 0.1:
        print(f"  ✓ HYPOTHESIS CONFIRMED — torchmetrics undercounts by {diff:+.4f}")
        print("    Use Tucker's compute_macro_auc instead for an apples-to-apples")
        print("    comparison against his published trajectory (0.62 at ep 1).")
    elif np.isnan(tm_auc) or np.isinf(tm_auc):
        print("  ! torchmetrics returned NaN/Inf — different failure mode")
    else:
        print(f"  Δ = {diff:+.4f} (small; metrics agree → look elsewhere)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="YAML config (required for real-data check)"
    )
    parser.add_argument(
        "--ckpt", type=Path, default=None, help="checkpoint .ckpt path (omit for synthetic-only)"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Synthetic always runs (it's the cheap, deterministic diagnostic)
    synthetic_diagnostic(num_samples=168, num_classes=234)

    if args.ckpt is not None:
        if args.config is None:
            parser.error("--config is required when --ckpt is provided")
        real_diagnostic(args.config, args.ckpt, num_workers=args.num_workers)
    else:
        print("(skipping real-data diagnostic — pass --ckpt to enable)")


if __name__ == "__main__":
    main()
