#!/usr/bin/env python3
"""Export trained model to ONNX.

Supports EfficientNet-B1 and ConvNeXt-Base backbones (auto-detected from config).
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
from transformers import AutoModel

from birdclef_102.data import build_species_list
from birdclef_102.model import _get_backbone_type, _get_embed_dim, LORA_TARGETS, SEDHead

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
        features = self.backbone(pixel_values=mel).last_hidden_state  # (B, D, H, W)

        if self.pooling_mode == "avg":
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)  # (B, D)
            logits = self.head(pooled)
        elif self.pooling_mode == "attention":
            frame_features = features.mean(dim=2).permute(0, 2, 1)  # (B, T, D)
            if self.head.projection is not None:
                frame_features = self.head.projection(frame_features)
            frame_logits = self.head.frame_head(frame_features)
            logits = torch.sigmoid(frame_logits).max(dim=1).values  # (B, C)
        elif self.pooling_mode == "sed":
            # SED head in inference mode returns blended clip logits
            self.head.eval()
            result = self.head(features)
            logits = result["clipwise_logits"]
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        return logits


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
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
    model_name = model_cfg["model_name"]

    species_list = build_species_list(data_cfg["metadata_csv"])
    num_classes = len(species_list)

    backbone_type = _get_backbone_type(model_name)
    logger.info(f"Classes: {num_classes}, Backbone: {backbone_type}, Mode: {backbone_mode}")

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
    backbone = AutoModel.from_pretrained(model_name)
    embed_dim = _get_embed_dim(backbone, backbone_type)

    if backbone_mode == "lora":
        from peft import LoraConfig, get_peft_model

        default_targets = LORA_TARGETS.get(backbone_type, LORA_TARGETS["efficientnet"])
        lora_targets = lora_cfg.get("target_modules", default_targets)
        lora_config = LoraConfig(
            r=lora_cfg.get("rank", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=0.0,
            target_modules=lora_targets,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)

        backbone_state = {
            k.removeprefix("backbone."): v
            for k, v in model_state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        logger.info(f"LoRA backbone loaded (missing={len(missing)}, unexpected={len(unexpected)})")

        backbone = backbone.merge_and_unload()
        logger.info("LoRA adapters merged into backbone")
    else:
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
    head_hidden_dim = model_cfg.get("head_hidden_dim", 0)
    if pooling_mode == "avg":
        head = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(embed_dim, num_classes),
        )
    elif pooling_mode == "attention":
        from birdclef_102.model import AttentionPoolingHead
        head = AttentionPoolingHead(embed_dim, num_classes, hidden_dim=head_hidden_dim, dropout=0.0)
    elif pooling_mode == "sed":
        head = SEDHead(
            embed_dim, num_classes,
            hidden_dim=head_hidden_dim if head_hidden_dim > 0 else 512,
            dropout=0.0,
            gem_p=model_cfg.get("gem_p", 3.0),
            blend_clip=model_cfg.get("blend_clip", 0.75),
            blend_frame=model_cfg.get("blend_frame", 0.25),
        )
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
    # avg/sed pooling outputs logits (need sigmoid in kernel)
    # attention pooling outputs probabilities (sigmoid already in ONNX graph)
    output_is_probs = pooling_mode == "attention"

    mel_config = {
        "sr": data_cfg["sample_rate"],
        "n_fft": data_cfg.get("n_fft", 2048),
        "hop_length": data_cfg.get("hop_length", 256),
        "n_mels": n_mels,
        "target_frames": target_frames,
        "fmin": 50,
        "fmax": 14000,
        "norm_mean": -4.268,
        "norm_std": 4.569,
        "output_is_probs": output_is_probs,
        "mel_norm": model_cfg.get("mel_norm", "audioset"),
    }
    with open(output_dir / "mel_config.json", "w") as f:
        json.dump(mel_config, f, indent=2)

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
