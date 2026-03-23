#!/usr/bin/env python3
"""Export ConvNeXt LoRA fine-tuned model to ONNX.

Loads the PEFT LoRA checkpoint, merges adapters into base weights, wraps
backbone + head for mel input, and exports to ONNX with opset 17.

Usage:
    python scripts/export_convnext.py \
        --checkpoint /path/to/best-epochepoch=12.ckpt \
        --config config/convnext_lora_mlp_mlphead.yaml \
        --output-dir artifacts/weights/convnext/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

EXPERIMENT_007_DIR = Path(__file__).resolve().parent.parent.parent / "007-convnext-finetune"
USER_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


def main():
    parser = argparse.ArgumentParser(description="Export ConvNeXt to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Lightning checkpoint path")
    parser.add_argument(
        "--config",
        type=str,
        default=str(EXPERIMENT_007_DIR / "config" / "convnext_lora_mlp_mlphead.yaml"),
        help="ConvNeXt LoRA config YAML",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2025/train.csv",
        help="BirdCLEF 2025 train.csv for species list",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/weights/convnext/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sys.path.insert(0, str(USER_SRC_DIR))

    from peft import LoraConfig, get_peft_model

    from birdclef.data.dataset import build_species_list
    from birdclef.encoders.convnext import ConvNeXtEncoder
    from birdclef.lightning.lora_finetune import LoRAFinetuneModule

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build species list
    species_list = build_species_list(args.train_csv)
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes}")

    # Build encoder
    encoder_cfg = cfg["encoder"]
    encoder = ConvNeXtEncoder(
        model_name=encoder_cfg["model_name"],
        pretrained=encoder_cfg.get("pretrained", True),
    )

    # Apply PEFT LoRA to backbone (not entire encoder)
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    encoder.backbone = get_peft_model(encoder.backbone, lora_config)
    logger.info("LoRA applied to ConvNeXt backbone")

    # Build full module (encoder + head)
    training_cfg = cfg["training"]
    module = LoRAFinetuneModule(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=encoder_cfg["embed_dim"],
        hidden_dim=training_cfg.get("hidden_dim", 0),
        dropout=training_cfg.get("dropout", 0.1),
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded successfully")

    # Merge LoRA weights into backbone
    module.encoder.backbone = module.encoder.backbone.merge_and_unload()
    logger.info("LoRA merged into backbone")

    # Create ONNX export wrapper
    class ConvNeXtWrapper(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, mel):
            # mel: (B, 1, 128, 500)
            features = self.backbone(mel)  # (B, 1024)
            return self.head(features)

    wrapper = ConvNeXtWrapper(module.encoder.backbone, module.head)
    wrapper.eval()

    # Test forward pass
    dummy_mel = torch.randn(1, 1, 128, 500)
    with torch.no_grad():
        ref_out = wrapper(dummy_mel)
    logger.info(f"Test forward output shape: {ref_out.shape}")

    # Export ONNX
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info(f"Exporting ONNX to {onnx_path}")
    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(onnx_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["logits"],
        dynamic_axes={
            "mel": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )
    logger.info(f"ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Verify ONNX output matches PyTorch
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"mel": dummy_mel.numpy()})[0]
    max_diff = np.max(np.abs(ref_out.numpy() - onnx_out))
    logger.info(f"ONNX verification: max diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"ONNX output mismatch: {max_diff}"

    # Save species list
    species_path = output_dir / "species_list.json"
    with open(species_path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(f"Species list saved: {species_path}")

    # Save mel config (matching nnAudio training pipeline)
    # hop_length = int(10ms / 1000 * 32000) = 320
    mel_config = {
        "type": "mel",
        "sr": 32000,
        "n_fft": 1024,
        "hop_length": 320,
        "n_mels": 128,
        "fmin": 0,
        "fmax": None,
        "htk": True,
        "target_frames": 500,
        "norm_mean": -4.268,
        "norm_std": 4.569,
    }
    mel_config_path = output_dir / "mel_config.json"
    with open(mel_config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved: {mel_config_path}")

    logger.info("ConvNeXt export complete!")


if __name__ == "__main__":
    main()
