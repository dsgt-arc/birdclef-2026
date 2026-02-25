#!/usr/bin/env python3
"""Export fine-tuned EfficientNet-B1 classifier to ONNX.

Wraps the frozen backbone + linear head in a MelInputWrapper that takes
mel spectrograms (B, 1, 128, 1000) and outputs clip-level logits (B, 206).
No custom ONNX ops needed (unlike SSAMBA's mamba-onnx).

Also exports species_list.json and mel_config.json alongside the .onnx file.

Usage:
    python scripts/export_onnx.py \
        --checkpoint logs/finetune/checkpoints/best.ckpt \
        --config configs/finetune.yaml \
        --metadata-csv ~/scratch/birdclef-2026/data/birdclef-2025/train.csv \
        --output-dir artifacts/weights/
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import EfficientNetModel

from birdclef_009.data.finetune import build_species_list

logger = logging.getLogger(__name__)


class MelInputWrapper(nn.Module):
    """Wraps EfficientNet backbone + linear head for ONNX export.

    Input: (B, 1, 128, 1000) mel spectrogram (already log-normalized)
    Output: (B, C) clip-level logits

    No mel computation or SpecAugment — those are handled outside.
    """

    def __init__(self, backbone: EfficientNetModel, head: nn.Sequential):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=mel).last_hidden_state  # (B, 1280, H', W')
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, 1280)
        logits = self.head(pooled)  # (B, C)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Export EfficientNet classifier to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Lightning checkpoint path")
    parser.add_argument("--config", type=str, required=True, help="Finetune config YAML")
    parser.add_argument("--metadata-csv", type=str, required=True, help="train.csv for species list")
    parser.add_argument("--output-dir", type=str, default="artifacts/weights/", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # Build species list
    species_list = build_species_list(args.metadata_csv)
    num_classes = len(species_list)
    logger.info(f"Number of classes: {num_classes}")

    # Build backbone
    backbone = EfficientNetModel.from_pretrained(model_cfg["model_name"])
    backbone.eval()
    embed_dim = backbone.config.hidden_dim  # 1280 for B1

    # Build head (matching EfficientNetClassifier architecture)
    head = nn.Sequential(
        nn.Dropout(0.0),  # Set to 0 for inference
        nn.Linear(embed_dim, num_classes),
    )

    # Load trained weights from Lightning checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip 'model.' prefix from Lightning state dict
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value
        else:
            model_state[key] = value

    # Load head weights (backbone weights come from pretrained HuggingFace model)
    head_state = {}
    for key, value in model_state.items():
        if key.startswith("head."):
            head_state[key[len("head."):]] = value
    missing, unexpected = head.load_state_dict(head_state, strict=True)
    logger.info(f"Head weights loaded (missing={missing}, unexpected={unexpected})")

    # Optionally load backbone weights if they were saved (shouldn't change from pretrained)
    backbone_state = {}
    for key, value in model_state.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone."):]] = value
    if backbone_state:
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        if missing:
            real_missing = [k for k in missing if not k.startswith("embeddings")]
            if real_missing:
                logger.warning(f"Missing backbone keys: {len(real_missing)}")
        logger.info(f"Backbone weights loaded from checkpoint ({len(backbone_state)} keys)")

    # Wrap for export
    wrapper = MelInputWrapper(backbone, head)
    wrapper.eval()

    # Dummy input: (B, 1, n_mels, target_frames)
    n_mels = data_cfg["n_mels"]
    target_frames = data_cfg["target_frames"]
    dummy_input = torch.randn(1, 1, n_mels, target_frames)

    # Test forward pass
    with torch.no_grad():
        test_out = wrapper(dummy_input)
    logger.info(f"Test forward output shape: {test_out.shape}")  # (1, C)

    # Export ONNX
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "efficientnet_classifier.onnx"
    logger.info(f"Exporting ONNX to {onnx_path}")

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["clip_logits"],
        dynamic_axes={
            "mel": {0: "batch"},
            "clip_logits": {0: "batch"},
        },
        dynamo=False,
    )
    logger.info(f"ONNX model saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Export species list
    species_path = output_dir / "species_list.json"
    with open(species_path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(f"Species list saved: {species_path} ({num_classes} species)")

    # Export mel config (matching BirdSet preprocessing)
    sample_rate = data_cfg["sample_rate"]
    mel_config = {
        "sr": sample_rate,
        "n_fft": data_cfg.get("n_fft", 2048),
        "hop_length": data_cfg.get("hop_length", 256),
        "n_mels": n_mels,
        "fmin": 50,
        "fmax": 14000,
        "norm_mean": -4.268,
        "norm_std": 4.569,
    }
    mel_config_path = output_dir / "mel_config.json"
    with open(mel_config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved: {mel_config_path}")

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
