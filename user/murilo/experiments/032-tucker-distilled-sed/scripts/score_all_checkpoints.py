#!/usr/bin/env python3
"""Rank all saved epoch checkpoints in a directory by Tucker's macro AUROC.

Companion to `diag_val_auroc.py`. Used after a paired training run completes
to pick the actual best epoch for LB submission, since the in-training
`val/cmap` monitor is broken by torchmetrics' 0-positive-class behavior (see
`src/birdclef/metrics/tucker_macro.py` for the fix; `scripts/diag_val_auroc.py`
for the diagnostic that proved it).

For each `best-epoch*.ckpt` in --ckpt-dir:
    1. Load state_dict into a freshly-constructed module
    2. Run inference over the val DataLoader
    3. Compute Tucker's macro_auc (skipping 0-positive classes)
    4. Record (epoch, macro_auc, n_evaluable, cmap_broken)

Outputs:
    - Ranked table to stdout
    - Optional --output TSV file
    - The top checkpoint's path is the recommended LB submission target

Usage:
    python scripts/score_all_checkpoints.py \\
        --config configs/distilled_sed_control.yaml \\
        --ckpt-dir /storage/project/.../checkpoints/2026-exp032-distilled-sed-control/ \\
        --output outputs/checkpoint_ranking_control.tsv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

EXP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_ROOT.parents[3]
SRC_ROOT = EXP_ROOT.parent.parent / "src"
sys.path.insert(0, str(SRC_ROOT))


# ----------------------------------------------------------------------
# Tucker reference metrics (verbatim from notebook lines 369-384)
# ----------------------------------------------------------------------


def tucker_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
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


def tucker_macro_ap(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aps = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aps.append(average_precision_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return float(np.mean(aps)) if aps else float("nan")


# ----------------------------------------------------------------------
# Helpers (mirrors train.py setup)
# ----------------------------------------------------------------------


_EPOCH_RE = re.compile(r"best-epochepoch=(\d+)")


def parse_epoch_from_ckpt(ckpt_path: Path) -> int | None:
    m = _EPOCH_RE.search(ckpt_path.name)
    return int(m.group(1)) if m else None


def build_module_and_dataloader(cfg: dict, num_workers: int):
    """Construct an empty DistilledSEDFinetuneModule + val DataLoader matching cfg.
    Returns (module, val_dl). The module's encoder + head will be filled by
    load_state_dict for each checkpoint."""
    from birdclef.data.distilled_sed_datamodule import (
        DistilledSEDDataModule,
        build_sc_cache_meta,
    )
    from birdclef.data.distilled_sed_dataset import _default_audio_loader as _audio_loader
    from birdclef.data.soundscape import build_full_species_list
    from birdclef.encoders.efficientnet import EfficientNetEncoder
    from birdclef.heads.sed_head import AttHeadLogitSpace
    from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule
    from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

    data_dir = Path(cfg["data"]["data_dir"])
    species_list = build_full_species_list(data_dir / "taxonomy.csv")
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)

    encoder = EfficientNetEncoder(
        model_name=cfg["encoder"]["model_name"],
        pretrained=False,
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

    training_cfg = cfg["training"]
    module = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        mel_transform=mel_transform,
        spec_augment=None,
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=training_cfg["max_epochs"],
        warmup_epochs=training_cfg["warmup_epochs"],
        warmup_start_factor=training_cfg.get("warmup_start_factor", 1.0 / 25),
        eta_min=training_cfg.get("eta_min", 1e-6),
        clipwise_weight=training_cfg.get("clipwise_weight", 0.5),
        framewise_weight=training_cfg.get("framewise_weight", 0.5),
        blend_clip=training_cfg.get("blend_clip", 0.5),
        blend_frame=training_cfg.get("blend_frame", 0.5),
        threshold=training_cfg.get("threshold", 0.5),
        per_sample_mel_norm=training_cfg.get("per_sample_mel_norm", True),
        use_perch_distill=False,  # we don't need the KD branch for inference
        perch_onnx_path=None,
        perch_embed_dim=training_cfg.get("perch_embed_dim", 1536),
        alpha_distill=training_cfg.get("alpha_distill", 1.0),
    )

    # Build val dataloader
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
    return module, dm.val_dataloader()


@torch.no_grad()
def score_one(
    module, val_dl, ckpt_path: Path, device: torch.device
) -> tuple[float, float, int, float]:
    """Load state from ckpt_path, run val inference, return Tucker AUC, AP, n_eval, broken-metric.

    The "broken-metric" value reproduces torchmetrics' MultilabelAUROC(average='macro')
    output so we can show the gap relative to Tucker's metric in the report.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # strict=False: KD-trained checkpoints carry extra `distill_head.proj.*`
    # keys, but the ranker builds the module with use_perch_distill=False
    # (the distill head isn't part of the inference forward). Surface any
    # unexpected-key set that ISN'T just the distill head, so a real schema
    # drift would still fail loudly.
    missing, unexpected = module.load_state_dict(state["state_dict"], strict=False)
    unexpected_non_distill = [k for k in unexpected if not k.startswith("distill_head.")]
    if missing or unexpected_non_distill:
        raise RuntimeError(
            f"Unexpected schema drift in {ckpt_path.name}: "
            f"missing={missing}, unexpected_non_distill={unexpected_non_distill}"
        )
    module.to(device)
    module.eval()

    all_probs, all_labels = [], []
    for batch in val_dl:
        x = batch[0].to(device)
        labels = batch[1]
        mel = module._compute_mel(x, training=False)
        clip_logits, frame_logits = module(mel)
        frame_max = frame_logits.max(dim=-1).values
        probs = module._blended_probs(clip_logits, frame_max)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    auc, n_eval = tucker_macro_auc(y_true, y_pred)
    ap = tucker_macro_ap(y_true, y_pred)

    # Reproduce the broken metric for the comparison column
    n_zero_classes = int((y_true.sum(axis=0) == 0).sum())
    broken_auc = (
        (n_eval * auc + n_zero_classes * 0.0) / y_true.shape[1]
        if not np.isnan(auc)
        else float("nan")
    )
    return auc, ap, n_eval, broken_auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="optional TSV output path")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pattern", type=str, default="best-epochepoch=*.ckpt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ckpts = sorted(args.ckpt_dir.glob(args.pattern), key=lambda p: parse_epoch_from_ckpt(p) or -1)
    if not ckpts:
        print(f"No checkpoints found in {args.ckpt_dir} matching {args.pattern!r}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(ckpts)} checkpoints to score:")
    for p in ckpts:
        print(f"  {p.name}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building module + val dataloader (device={device})...")
    module, val_dl = build_module_and_dataloader(cfg, num_workers=args.num_workers)
    print(f"Val dataloader: {len(val_dl.dataset)} samples")
    print()

    print(f"Scoring {len(ckpts)} checkpoints...")
    rows = []
    for i, ckpt in enumerate(ckpts, 1):
        epoch = parse_epoch_from_ckpt(ckpt)
        auc, ap, n_eval, broken = score_one(module, val_dl, ckpt, device)
        rows.append(
            {
                "epoch": epoch if epoch is not None else -1,
                "ckpt": ckpt.name,
                "tucker_macro_auc": round(auc, 4),
                "tucker_macro_ap": round(ap, 4),
                "n_evaluable": n_eval,
                "broken_torchmetrics_auc": round(broken, 4),
            }
        )
        print(
            f"  [{i}/{len(ckpts)}] epoch={epoch} "
            f"AUC={auc:.4f} AP={ap:.4f} n_eval={n_eval} (broken={broken:.4f})"
        )

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("tucker_macro_auc", ascending=False).reset_index(drop=True)

    print()
    print("=" * 90)
    print(f"Ranking by Tucker's macro AUROC (val fold 0, {args.ckpt_dir.name}):")
    print("=" * 90)
    print(df_sorted.to_string(index=False))
    print()
    best = df_sorted.iloc[0]
    print(f"Best epoch: {int(best['epoch'])}  →  AUC={best['tucker_macro_auc']:.4f}")
    print(f"Recommended LB submission target: {args.ckpt_dir / best['ckpt']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_sorted.to_csv(args.output, sep="\t", index=False)
        print(f"\nWrote ranking to: {args.output}")


if __name__ == "__main__":
    main()
