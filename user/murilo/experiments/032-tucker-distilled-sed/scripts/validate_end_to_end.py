#!/usr/bin/env python3
"""End-to-end pre-flight validation for exp 032 checkpoints (Phase D § D.5).

Runs the EXACT same audio→mel→model→post-processing path as the Kaggle
kernel (`compute_mel_tucker` + per-window LOGIT-space blend + 5-tap
Gaussian smoothing + single sigmoid) over the held-out validation fold,
then computes Tucker macro AUC against `train_soundscapes_labels.csv`.

The result is directly comparable to the training-side `val/auroc` that
`DistilledSEDFinetuneModule.validation_step` reports during training:

  same checkpoint, same val windows, same labels
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     TRAINING-SIDE val/auroc      THIS SCRIPT val/auroc
     uses training mel path       uses KERNEL mel + LB-style smoothing

If the two numbers agree within ~0.01, the training pipeline is end-to-end
consistent with the inference kernel and an LB submission is safe to run.

If they DIVERGE by more than ~0.01, there is a training/inference
mismatch (mel pipeline, audio loading, smoothing, anywhere). This is
exactly what would have caught the Phase 1 mel mismatch BEFORE we
burned an LB submission:
  - training-side val/auroc on Phase 1 ckpts: ~0.89 (Tucker macro)
  - this script on Phase 1 ckpts (predicted): ~0.52 (matching LB)

This script makes the offline-vs-online sanity check a one-command
pre-flight, not an after-the-fact diagnostic.

Usage:
    # Default config + ckpt (post-Phase-D retrain, control)
    ~/clef/birdclef-2026/.venv/bin/python user/murilo/experiments/032-tucker-distilled-sed/scripts/validate_end_to_end.py \\
        --config user/murilo/experiments/032-tucker-distilled-sed/configs/distilled_sed_control.yaml \\
        --ckpt /storage/project/.../checkpoints/.../last.ckpt

    # Optionally write predictions to TSV for offline inspection
    [...] --output user/murilo/experiments/032-tucker-distilled-sed/outputs/e2e_val_control.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

EXP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_ROOT.parents[3]
SRC_ROOT = REPO_ROOT / "user/murilo/src"
sys.path.insert(0, str(SRC_ROOT))

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins. Mirrors exp 033 train.py:_deep_merge."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_with_common(config_path: Path) -> dict:
    """yaml.safe_load + optional sibling `_common.yaml` deep-merge.

    No-op for experiments that don't use `_common.yaml` (e.g., exp 032's flat
    configs); enables exp 033's diff-only per-ablation YAMLs to load correctly
    without forking this script.
    """
    with open(config_path) as f:
        override = yaml.safe_load(f) or {}
    common_p = config_path.parent / "_common.yaml"
    if common_p.exists() and config_path.name != "_common.yaml":
        with open(common_p) as f:
            base = yaml.safe_load(f) or {}
        merged = _deep_merge(base, override)
        logger.info(f"Loaded config: {config_path.name} (merged with _common.yaml)")
        return merged
    return override


# ----------------------------------------------------------------------
# Tucker's correct macro AUC / AP — skip 0-positive classes
# (verbatim port of notebook lines 369-384; also implemented in
# src/birdclef/metrics/tucker_macro.py)
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
# Kernel-faithful audio loading + mel
# (mirror of submission.py:load_audio_chunk + compute_mel_tucker)
# ----------------------------------------------------------------------


def kernel_load_audio_chunk(
    audio_path: Path, sr: int, start_s: float, duration_s: float
) -> np.ndarray:
    """Mirror of `submission.py:load_audio_chunk` for the no-pre-pad case.

    Returns exactly `sr * duration_s` samples at the target SR.
    """
    target_samples = int(sr * duration_s)
    info = sf.info(str(audio_path))
    source_sr = info.samplerate
    total_duration = info.frames / source_sr

    actual_duration = min(duration_s, total_duration - start_s)
    if actual_duration <= 0:
        return np.zeros(target_samples, dtype=np.float32)

    audio, _ = sf.read(
        str(audio_path),
        dtype="float32",
        start=int(start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return audio.astype(np.float32)


def kernel_compute_mel_tucker(audio_np: np.ndarray, mel_cfg: dict) -> np.ndarray:
    """Mirror of `submission.py:compute_mel_tucker`. Returns (1, 1, n_mels, T)."""
    sr = mel_cfg["sr"]
    wave = torch.from_numpy(audio_np).float().unsqueeze(0)  # (1, samples)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=mel_cfg.get("n_fft", 2048),
        hop_length=mel_cfg.get("hop_length", 512),
        n_mels=mel_cfg.get("n_mels", 256),
        f_min=mel_cfg.get("fmin", 20),
        f_max=mel_cfg.get("fmax", 16000),
        power=mel_cfg.get("power", 2.0),
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=mel_cfg.get("top_db", 80.0)
    )

    mel = mel_transform(wave)
    mel = db_transform(mel)

    mean = mel.mean()
    std = mel.std().clamp(min=1e-6)
    mel = (mel - mean) / std

    target_frames = mel_cfg.get("target_frames", 313)
    T = mel.shape[-1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel = mel[..., start : start + target_frames]
    elif T < target_frames:
        mel = torch.nn.functional.pad(mel, (0, target_frames - T))

    return mel.unsqueeze(0).numpy().astype(np.float32)  # (1, 1, n_mels, T)


def kernel_smooth_and_sigmoid(window_logits: np.ndarray) -> np.ndarray:
    """5-tap Gaussian smoothing over windows in LOGIT space + single sigmoid.

    Mirrors `submission.py:predict_soundscape_tucker`. Input shape
    (n_windows, n_classes); output same shape, in [0,1].
    """
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    kernel = kernel / kernel.sum()
    if window_logits.shape[0] < 3:
        smoothed = window_logits
    else:
        pad = len(kernel) // 2
        padded = np.pad(window_logits, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.zeros_like(window_logits)
        for c in range(window_logits.shape[1]):
            smoothed[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
    # Single sigmoid at the end (numerically stable)
    return 1.0 / (1.0 + np.exp(-smoothed))


# ----------------------------------------------------------------------
# Build module + load checkpoint
# (mirrors score_all_checkpoints.py:build_module_and_dataloader, but only
# the module construction — we run the validation OURSELVES through the
# kernel path rather than the training-side DataLoader)
# ----------------------------------------------------------------------


def build_module(
    cfg: dict, num_classes: int, sample_rate: int, window_duration: float, mel_cfg: dict
):
    """Construct an empty DistilledSEDFinetuneModule matching cfg.

    The mel_transform stub matches the training-time mel transform so the
    checkpoint's state_dict loads cleanly. The mel_transform itself is
    NEVER called in this script — we go through `kernel_compute_mel_tucker`
    instead, and feed pre-computed mel directly into encoder + head via
    a custom forward path.
    """
    from birdclef.encoders.efficientnet import EfficientNetEncoder
    from birdclef.heads.sed_head import AttHeadLogitSpace
    from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule

    encoder = EfficientNetEncoder(
        model_name=cfg["encoder"]["model_name"],
        pretrained=False,
        drop_path_rate=0.0,
        in_chans=cfg["encoder"].get("in_chans", 1),
        use_post_head=cfg["encoder"].get("use_post_head", True),
    )
    sed_head = AttHeadLogitSpace(
        in_channels=encoder.embed_dim,
        num_classes=num_classes,
        hidden_dim=cfg["sed_head"]["hidden_dim"],
        dropout=cfg["sed_head"]["dropout"],
        gem_p=cfg["sed_head"].get("gem_p", 3.0),
    )

    # Stub mel_transform matching training config so state_dict loads cleanly
    mel_type_train = mel_cfg.get("type", "nnaudio")
    if mel_type_train == "tucker":
        from birdclef.preprocessing.tucker_mel import TuckerMelSpectrogram

        stub_mel = TuckerMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=mel_cfg.get("n_fft", 2048),
            hop_length=mel_cfg.get("hop_length", 512),
            n_mels=mel_cfg.get("n_mels", 256),
            fmin=mel_cfg.get("fmin", 20),
            fmax=mel_cfg.get("fmax", 16000),
            top_db=mel_cfg.get("top_db", 80.0),
            power=mel_cfg.get("power", 2.0),
        )
    else:
        from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

        target_frames = int(window_duration * sample_rate / mel_cfg["hop_length"])
        stub_mel = NNAudioMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=mel_cfg.get("n_mels", 128),
            target_frames=target_frames,
            n_fft=mel_cfg.get("n_fft"),
            hop_length=mel_cfg.get("hop_length"),
            fmax=mel_cfg.get("fmax"),
            norm_type=mel_cfg.get("norm_type", "audioset"),
            top_db=mel_cfg.get("top_db", 80.0),
        )

    training_cfg = cfg["training"]
    module = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        mel_transform=stub_mel,
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
        use_perch_distill=False,
        perch_onnx_path=None,
        perch_embed_dim=training_cfg.get("perch_embed_dim", 1536),
        alpha_distill=training_cfg.get("alpha_distill", 1.0),
    )
    return module


def load_checkpoint(module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    unexpected_non_distill = [k for k in unexpected if not k.startswith("distill_head.")]
    real_missing = [
        k
        for k in missing
        if not any(m in k for m in ["cmap", "auroc", "f1_macro", "f1_micro", "top1", "top5"])
    ]
    if real_missing or unexpected_non_distill:
        raise RuntimeError(
            f"Schema drift loading {ckpt_path.name}: "
            f"missing={real_missing}, unexpected_non_distill={unexpected_non_distill}"
        )


# ----------------------------------------------------------------------
# Build val-fold windows + label matrix
# ----------------------------------------------------------------------


def build_val_windows(
    data_dir: Path, cfg: dict, num_classes: int, species_to_idx: dict[str, int]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the val-fold (filename, start_sec, label_vector) table.

    Mirrors `DistilledSEDDataModule.setup` for the SS side only: load
    train_soundscapes_labels.csv, build the 66 × 12 = 792-row cache table,
    apply the same GroupKFold(n_splits=5) on filename, then filter to
    fold_k = cfg.data.fold_k (default 0).
    """
    from birdclef.data.distilled_sed_datamodule import (
        build_sc_cache_meta,
        build_sc_folds,
        build_sc_label_matrix,
    )

    sc_labels_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    labeled_sc_files = sorted(sc_labels_df["filename"].unique().tolist())
    sc_df = build_sc_cache_meta(labeled_sc_files)
    sc_df_folded = build_sc_folds(sc_df, n_folds=cfg["data"].get("n_folds", 5))
    y_full = build_sc_label_matrix(sc_df_folded, sc_labels_df, species_to_idx, num_classes)

    fold_k = cfg["data"].get("fold_k", 0)
    val_mask = (sc_df_folded["fold"] == fold_k).values
    val_df = sc_df_folded[val_mask].reset_index(drop=True)
    y_val = y_full[val_mask]

    logger.info(
        f"Val fold {fold_k}: {len(val_df)} windows, "
        f"{int(y_val.sum())} positive labels, "
        f"{int((y_val.sum(axis=0) > 0).sum())} evaluable classes"
    )
    return val_df, y_val


# ----------------------------------------------------------------------
# End-to-end inference
# ----------------------------------------------------------------------


@torch.no_grad()
def run_kernel_inference(
    module, val_df: pd.DataFrame, data_dir: Path, mel_cfg: dict, device: torch.device
) -> np.ndarray:
    """Run the kernel inference path over val_df windows.

    For each unique soundscape file in val_df:
      - load all 12 × 5s audio chunks
      - compute_mel_tucker on each
      - module forward → (clip_logits, framewise_logits)
      - frame_max_logits = framewise.max(dim=-1).values
      - per-window blend in LOGIT space (0.5 / 0.5, Tucker default)
      - stack 12 window logits per file
      - 5-tap Gauss smooth across windows in LOGIT space
      - sigmoid at end → per-window probabilities

    Returns y_pred of shape (n_windows, n_classes) aligned with val_df.
    """
    module.eval()
    module.to(device)

    sr = mel_cfg["sr"]
    segment_duration = 5.0
    n_classes = module.num_classes

    sc_dir = data_dir / "train_soundscapes"
    files = val_df["filename"].unique().tolist()
    logger.info(f"Running kernel inference over {len(files)} val files...")

    # Build a (filename, start_sec) → row index lookup so we write predictions
    # back in val_df's order
    val_idx = {(row["filename"], row["start_sec"]): i for i, row in val_df.iterrows()}
    y_pred = np.zeros((len(val_df), n_classes), dtype=np.float32)

    t0 = time.time()
    for fi, fn in enumerate(files):
        # Gather all val windows for this file (preserve start_sec order)
        file_rows = val_df[val_df["filename"] == fn].sort_values("start_sec")
        start_secs = file_rows["start_sec"].tolist()

        # Run kernel path per window
        file_window_logits = []
        for ss in start_secs:
            audio_np = kernel_load_audio_chunk(sc_dir / fn, sr, float(ss), segment_duration)
            mel_np = kernel_compute_mel_tucker(audio_np, mel_cfg)  # (1,1,n_mels,T)
            mel_t = torch.from_numpy(mel_np).to(device)

            features = module.encoder.forward_features(mel_t)  # (B, C, F, T)
            clip_logits, frame_logits = module.sed_head(features)
            frame_max_logits = frame_logits.max(dim=-1).values  # (B, num_classes)
            # Per-window blend in LOGIT space — matches the export wrapper's
            # pre-blend and the kernel's smoothing input
            blend_clip = mel_cfg.get("blend_clip", 0.5)
            blend_frame = mel_cfg.get("blend_frame", 0.5)
            blended_logits = (
                (blend_clip * clip_logits + blend_frame * frame_max_logits).cpu().numpy()
            )
            file_window_logits.append(blended_logits.squeeze(0))

        # Smooth across this file's windows in LOGIT space + sigmoid
        file_logits_matrix = np.stack(file_window_logits)  # (n_windows_in_file, n_classes)
        file_probs = kernel_smooth_and_sigmoid(file_logits_matrix)

        for i, ss in enumerate(start_secs):
            y_pred[val_idx[(fn, ss)]] = file_probs[i]

        if (fi + 1) % 5 == 0 or fi == len(files) - 1:
            logger.info(f"  {fi + 1}/{len(files)} files done ({time.time() - t0:.1f}s elapsed)")

    return y_pred


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional TSV with per-class metrics"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument(
        "--training-side-auroc",
        type=float,
        default=None,
        help="Optional: the training-side val/auroc value to compare against. "
        "If provided, the script will report the absolute gap and a PASS/FAIL.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Acceptable absolute gap between training-side and end-to-end "
        "val/auroc (default: 0.02). Outside this, FAIL.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Override data.fold_k from the config. Required when evaluating "
        "a 5-fold ckpt trained with `train.py --fold N` (Phase 2.A 5-fold KD).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config_with_common(args.config)

    if args.fold is not None:
        if not 0 <= args.fold < 5:
            raise ValueError(f"--fold must be in [0, 4], got {args.fold}")
        cfg["data"]["fold_k"] = args.fold
        logger.info(f"Override: data.fold_k = {args.fold} (from --fold CLI)")

    data_dir = Path(cfg["data"]["data_dir"])
    sample_rate = cfg["data"]["sample_rate"]
    window_duration = cfg["data"]["window_duration"]

    from birdclef.data.soundscape import build_full_species_list

    species_list = build_full_species_list(data_dir / "taxonomy.csv")
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    num_classes = len(species_list)

    # Build (filename, start_sec) → label_vector for the val fold
    val_df, y_true = build_val_windows(data_dir, cfg, num_classes, species_to_idx)

    # Build mel_cfg in the kernel-side shape expected by kernel_compute_mel_tucker
    mel_cfg_train = cfg["mel"]
    hop_length = mel_cfg_train["hop_length"]
    target_frames = int(window_duration * sample_rate / hop_length) + 1
    kernel_mel_cfg = {
        "sr": sample_rate,
        "n_fft": mel_cfg_train["n_fft"],
        "hop_length": hop_length,
        "n_mels": mel_cfg_train["n_mels"],
        "fmin": mel_cfg_train.get("fmin", 20),
        "fmax": mel_cfg_train.get("fmax", 16000),
        "top_db": mel_cfg_train.get("top_db", 80.0),
        "power": mel_cfg_train.get("power", 2.0),
        "target_frames": target_frames,
        "blend_clip": cfg["training"].get("blend_clip", 0.5),
        "blend_frame": cfg["training"].get("blend_frame", 0.5),
    }

    # Build module + load checkpoint
    logger.info(f"Building module from config: {args.config}")
    module = build_module(cfg, num_classes, sample_rate, window_duration, mel_cfg_train)
    logger.info(f"Loading checkpoint: {args.ckpt}")
    load_checkpoint(module, args.ckpt)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    logger.info(f"Device: {device}")

    # End-to-end inference
    t0 = time.time()
    y_pred = run_kernel_inference(module, val_df, data_dir, kernel_mel_cfg, device)
    logger.info(f"End-to-end inference complete in {time.time() - t0:.1f}s")

    # Metrics
    e2e_auc, n_eval = tucker_macro_auc(y_true, y_pred)
    e2e_ap = tucker_macro_ap(y_true, y_pred)

    print()
    print("=" * 78)
    print("End-to-end validation — kernel inference path on training val fold")
    print("=" * 78)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Config:     {args.config}")
    print(f"Val fold:   fold {cfg['data'].get('fold_k', 0)}  ({len(val_df)} windows)")
    print()
    print(f"  Tucker macro AUC (end-to-end):  {e2e_auc:.4f}  (n_evaluable={n_eval})")
    print(f"  Tucker macro AP  (end-to-end):  {e2e_ap:.4f}")

    if args.training_side_auroc is not None:
        gap = abs(args.training_side_auroc - e2e_auc)
        status = "PASS ✅" if gap <= args.tolerance else "FAIL ❌"
        print()
        print(f"  Training-side val/auroc (provided): {args.training_side_auroc:.4f}")
        print(f"  End-to-end val/auroc:               {e2e_auc:.4f}")
        print(f"  Absolute gap:                       {gap:.4f}  (tolerance {args.tolerance})")
        print(f"  Status:                             {status}")
        if gap > args.tolerance:
            print()
            print("  ⚠️  End-to-end metric diverges from training-side. There is a")
            print("      training/inference pipeline mismatch. Do NOT submit to LB")
            print("      until this is resolved.")
            print("      Common causes: mel pipeline drift, audio loading SR mismatch,")
            print("      smoothing/blend mismatch, or species_mapping bug.")
    else:
        print()
        print("  (No --training-side-auroc provided. Compare manually against the W&B")
        print("   `val/auroc` for the matching epoch from training. Expected: same to")
        print(f"   within ±{args.tolerance:.3f} if pipelines are consistent.)")

    if args.output is not None:
        # Per-class AUCs for offline inspection
        rows = []
        for c in range(y_true.shape[1]):
            col = y_true[:, c]
            if col.sum() == 0 or col.sum() == len(col):
                continue
            try:
                rows.append(
                    {
                        "class_idx": c,
                        "species": species_list[c],
                        "n_pos": int(col.sum()),
                        "auc": float(roc_auc_score(col, y_pred[:, c])),
                        "ap": float(average_precision_score(col, y_pred[:, c])),
                    }
                )
            except ValueError:
                continue
        df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, sep="\t", index=False)
        print(f"\nPer-class metrics written to: {args.output}")


if __name__ == "__main__":
    main()
