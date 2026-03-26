#!/usr/bin/env python3
"""Validate ONNX export matches PyTorch checkpoint predictions.

Loads a few real audio clips, runs through both PyTorch model and ONNX model,
and compares the outputs. Also compares the mel spectrogram from nnAudio (training)
vs librosa (Kaggle inference) to check for mel mismatches.

Usage:
    python scripts/validate_onnx.py \
        --checkpoint logs/lora-attention/checkpoints/best.ckpt \
        --config configs/run-c-lora-attention.yaml \
        --onnx-dir artifacts/weights/lora-attention/
"""

import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml
from nnAudio.features import MelSpectrogram

logger = logging.getLogger(__name__)


def compute_mel_librosa(audio, mel_cfg):
    """Compute mel the same way as the Kaggle submission kernel."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=mel_cfg["sr"], n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"], n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg["fmin"], fmax=mel_cfg["fmax"], power=2.0,
    )
    mel = np.log(np.clip(mel, a_min=1e-10, a_max=None))
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)
    target_frames = mel_cfg.get("target_frames", 625)
    T = mel.shape[1]
    if T > target_frames:
        mel = mel[:, :target_frames]
    elif T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")
    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def compute_mel_nnaudio(audio_tensor, mel_spec, norm_mean, norm_std, target_frames):
    """Compute mel the same way as training (nnAudio on CPU)."""
    with torch.no_grad():
        mel = mel_spec(audio_tensor.unsqueeze(0))  # (1, n_mels, T)
        mel = torch.log(mel.clamp(min=1e-10))
        mel = (mel - norm_mean) / (norm_std * 2)
        T = mel.shape[2]
        if T > target_frames:
            mel = mel[:, :, :target_frames]
        elif T < target_frames:
            mel = torch.nn.functional.pad(mel, (0, target_frames - T))
        return mel.unsqueeze(1).numpy()  # (1, 1, n_mels, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    lora_cfg = cfg.get("lora", {})
    onnx_dir = Path(args.onnx_dir)

    # Load mel config from ONNX export
    with open(onnx_dir / "mel_config.json") as f:
        mel_cfg = json.load(f)
    with open(onnx_dir / "species_list.json") as f:
        species_list = json.load(f)

    logger.info(f"Mel config: {mel_cfg}")
    logger.info(f"Species: {len(species_list)}")

    # Load ONNX model
    import onnxruntime as ort
    onnx_session = ort.InferenceSession(
        str(onnx_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    onnx_input_name = onnx_session.get_inputs()[0].name
    logger.info(f"ONNX loaded: input={onnx_input_name}")

    # Build nnAudio mel (same as training)
    mel_spec = MelSpectrogram(
        sr=data_cfg["sample_rate"],
        n_fft=data_cfg.get("n_fft", 2048),
        win_length=data_cfg.get("n_fft", 2048),
        hop_length=data_cfg.get("hop_length", 256),
        n_mels=data_cfg["n_mels"],
        fmin=50, fmax=14000,
        trainable_mel=False, trainable_STFT=False,
    )

    # Build PyTorch model
    from birdclef_102.model import EfficientNetClassifier
    from birdclef_102.data import build_species_list
    num_classes = len(species_list)
    backbone_mode = model_cfg.get("backbone_mode", "frozen")
    pooling_mode = model_cfg.get("pooling_mode", "avg")

    model = EfficientNetClassifier(
        model_name=model_cfg["model_name"],
        num_classes=num_classes,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        backbone_mode=backbone_mode,
        pooling_mode=pooling_mode,
        head_hidden_dim=model_cfg.get("head_hidden_dim", 0),
        head_dropout=model_cfg.get("head_dropout", 0.3),
        lora_rank=lora_cfg.get("rank", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.1),
        lora_targets=lora_cfg.get("target_modules"),
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    clean_state = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()
    logger.info("PyTorch model loaded")

    # Find some audio files
    import pandas as pd
    train_df = pd.read_csv(data_cfg["metadata_csv"])
    audio_dir = Path(data_cfg["audio_dir"])
    samples = train_df.sample(n=args.n_samples, random_state=42)

    print("\n=== Mel Comparison (nnAudio vs librosa) ===")
    print(f"{'file':40s} {'mel_mae':>10s} {'mel_max_diff':>12s}")

    print("\n=== Output Comparison (PyTorch vs ONNX) ===")
    print(f"{'file':40s} {'pt_top1':>20s} {'onnx_top1':>20s} {'mae':>10s} {'max_diff':>10s} {'match':>6s}")

    for _, row in samples.iterrows():
        fpath = audio_dir / row["filename"]
        if not fpath.exists():
            continue

        # Load 5s audio
        audio, sr = sf.read(str(fpath), dtype="float32", frames=int(5 * 32000) + 2)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 32000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
        if len(audio) < 160000:
            audio = np.pad(audio, (0, 160000 - len(audio)))
        audio = audio[:160000]

        # Mel comparison
        mel_librosa = compute_mel_librosa(audio, mel_cfg)
        mel_nnaudio = compute_mel_nnaudio(
            torch.from_numpy(audio).float(), mel_spec,
            -4.268, 4.569, data_cfg["target_frames"],
        )
        mel_mae = np.abs(mel_librosa - mel_nnaudio).mean()
        mel_max = np.abs(mel_librosa - mel_nnaudio).max()
        fname = Path(row["filename"]).name[:40]
        print(f"{fname:40s} {mel_mae:10.4f} {mel_max:12.4f}")

        # ONNX inference (uses librosa mel, like Kaggle kernel)
        onnx_out = onnx_session.run(None, {onnx_input_name: mel_librosa})[0]

        # PyTorch inference (uses nnAudio mel, like training)
        with torch.no_grad():
            pt_out = model(torch.from_numpy(audio).float().unsqueeze(0))
            pt_out = pt_out.numpy()

        # Handle attention pooling (output is probs, not logits)
        if mel_cfg.get("output_is_probs", False):
            onnx_probs = onnx_out[0]
            pt_probs = pt_out[0]
        else:
            onnx_probs = 1 / (1 + np.exp(-onnx_out[0]))
            pt_probs = 1 / (1 + np.exp(-pt_out[0]))

        mae = np.abs(onnx_probs - pt_probs).mean()
        max_diff = np.abs(onnx_probs - pt_probs).max()
        pt_top = species_list[pt_probs.argmax()]
        onnx_top = species_list[onnx_probs.argmax()]
        match = "YES" if pt_top == onnx_top else "NO"

        print(f"{fname:40s} {pt_top:>20s} {onnx_top:>20s} {mae:10.4f} {max_diff:10.4f} {match:>6s}")


if __name__ == "__main__":
    main()
