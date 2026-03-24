#!/usr/bin/env python3
"""Export fine-tuned EfficientNet-B1 to ONNX.

Handles both frozen and LoRA checkpoints:
- Frozen: loads head weights only (backbone from pretrained HuggingFace)
- LoRA: merges adapters into backbone, then exports merged model

Output: model.onnx + species_list.json + mel_config.json

Usage:
    python scripts/export_onnx.py \
        --checkpoint logs/frozen/checkpoints/best.ckpt \
        --config configs/run-a-frozen.yaml \
        --output-dir artifacts/weights/frozen/

    python scripts/export_onnx.py \
        --checkpoint logs/lora/checkpoints/best.ckpt \
        --config configs/run-b-lora.yaml \
        --output-dir artifacts/weights/lora/
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

from birdclef_102.data import build_species_list

logger = logging.getLogger(__name__)


class MelInputWrapper(nn.Module):
    """Wraps backbone + head for ONNX export.

    Input: (B, 1, n_mels, target_frames) mel spectrogram
    Output: (B, C) clip-level logits

    For attention pooling, always uses framewise max at export (inference mode).
    """

    def __init__(self, backbone, head, pooling_mode="avg"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.pooling_mode = pooling_mode

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=mel).last_hidden_state  # (B, 1280, H, W)

        if self.pooling_mode == "avg":
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, 1280)
            logits = self.head(pooled)
        elif self.pooling_mode == "attention":
            # At inference: collapse freq, get frame features, use frame_head + max
            frame_features = features.mean(dim=2).permute(0, 2, 1)  # (B, T, 1280)
            frame_logits = self.head.frame_head(frame_features)  # (B, T, C)
            logits = frame_logits.max(dim=1).values  # (B, C)
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        return logits


def main():
    parser = argparse.ArgumentParser(description="Export EfficientNet to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/weights/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    lora_cfg = cfg.get("lora", {})
    backbone_mode = model_cfg.get("backbone_mode", "frozen")

    species_list = build_species_list(data_cfg["metadata_csv"])
    num_classes = len(species_list)
    logger.info(f"Classes: {num_classes}, Backbone mode: {backbone_mode}")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip 'model.' prefix from Lightning state dict
    model_state = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("model.")
        model_state[clean_key] = value

    # Build backbone
    backbone = EfficientNetModel.from_pretrained(model_cfg["model_name"])
    embed_dim = backbone.config.hidden_dim

    if backbone_mode == "lora":
        # Apply LoRA, load adapter weights, then merge
        from peft import LoraConfig, get_peft_model

        lora_targets = lora_cfg.get("target_modules", [
            "project_conv", "expand_conv", "reduce", "expand",
        ])
        lora_config = LoraConfig(
            r=lora_cfg.get("rank", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=0.0,  # No dropout for inference
            target_modules=lora_targets,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)

        # Load LoRA adapter weights from checkpoint
        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in model_state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        logger.info(f"LoRA backbone loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        # Merge LoRA adapters into base weights for clean ONNX export
        backbone = backbone.merge_and_unload()
        logger.info("LoRA adapters merged into backbone")
    else:
        # Frozen: backbone weights are pretrained, no checkpoint loading needed
        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in model_state.items()
            if k.startswith("backbone.")
        }
        if backbone_state:
            backbone.load_state_dict(backbone_state, strict=False)
            logger.info(f"Backbone weights loaded ({len(backbone_state)} keys)")

    backbone.eval()

    # Build head
    pooling_mode = model_cfg.get("pooling_mode", "avg")
    if pooling_mode == "avg":
        head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(embed_dim, num_classes),
        )
    elif pooling_mode == "attention":
        from birdclef_102.model import AttentionPoolingHead
        head = AttentionPoolingHead(embed_dim, num_classes, dropout=0.0)
    else:
        raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

    head_state = {
        k.removeprefix("head."): v
        for k, v in model_state.items()
        if k.startswith("head.")
    }
    head.load_state_dict(head_state, strict=True)
    logger.info(f"Head weights loaded (pooling_mode={pooling_mode})")

    # Wrap and test
    wrapper = MelInputWrapper(backbone, head, pooling_mode=pooling_mode)
    wrapper.eval()

    n_mels = data_cfg["n_mels"]
    target_frames = data_cfg["target_frames"]
    dummy_input = torch.randn(1, 1, n_mels, target_frames)

    with torch.no_grad():
        test_out = wrapper(dummy_input)
    logger.info(f"Test forward: {test_out.shape}")

    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_input,),
        str(onnx_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["clip_logits"],
        dynamic_axes={"mel": {0: "batch"}, "clip_logits": {0: "batch"}},
    )
    logger.info(f"ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Save species list
    with open(output_dir / "species_list.json", "w") as f:
        json.dump(species_list, f, indent=2)

    # Save mel config
    mel_config = {
        "sr": data_cfg["sample_rate"],
        "n_fft": data_cfg.get("n_fft", 2048),
        "hop_length": data_cfg.get("hop_length", 256),
        "n_mels": n_mels,
        "fmin": 50,
        "fmax": 14000,
        "norm_mean": -4.268,
        "norm_std": 4.569,
    }
    with open(output_dir / "mel_config.json", "w") as f:
        json.dump(mel_config, f, indent=2)

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
