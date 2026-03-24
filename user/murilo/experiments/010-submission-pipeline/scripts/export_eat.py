#!/usr/bin/env python3
"""Export EAT LoRA fine-tuned model to ONNX.

Loads the PEFT LoRA checkpoint, merges adapters into base weights, wraps
encoder + head for mel input, and exports to ONNX with opset 17.

Usage:
    python scripts/export_eat.py \
        --checkpoint /path/to/best-epochepoch=16.ckpt \
        --config config/eat_lora_attn_mlphead.yaml \
        --output-dir artifacts/weights/eat/
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

# Experiment root for imports
EXPERIMENT_006_DIR = Path(__file__).resolve().parent.parent.parent / "006-lora-finetune"
USER_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


def main():
    parser = argparse.ArgumentParser(description="Export EAT to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Lightning checkpoint path")
    parser.add_argument(
        "--config",
        type=str,
        default=str(EXPERIMENT_006_DIR / "config" / "eat_lora_attn_mlphead.yaml"),
        help="EAT LoRA config YAML",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2025/train.csv",
        help="BirdCLEF 2025 train.csv for species list",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/weights/eat/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Add src to path for imports
    sys.path.insert(0, str(USER_SRC_DIR))

    from peft import LoraConfig, get_peft_model

    from birdclef.data.dataset import build_species_list
    from birdclef.encoders.eat import EATEncoder
    from birdclef.lightning.lora_finetune import LoRAFinetuneModule

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build species list
    species_list = build_species_list(args.train_csv)
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes}")

    # Build encoder (precomputed_features=True: takes fbank input, not raw waveform)
    encoder_cfg = cfg["encoder"]
    encoder = EATEncoder(
        model_name=encoder_cfg["model_name"],
        target_length=encoder_cfg.get("target_length", 512),
        precomputed_features=True,
    )

    # Apply PEFT LoRA
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    encoder.model = get_peft_model(encoder.model, lora_config)
    logger.info("LoRA applied to EAT encoder")

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

    # Merge LoRA weights into base model
    module.encoder.model = module.encoder.model.merge_and_unload()
    logger.info("LoRA merged into base model")

    # Create ONNX export wrapper
    class EATWrapper(nn.Module):
        def __init__(self, eat_model, head):
            super().__init__()
            self.eat_model = eat_model
            self.head = head

        def forward(self, fbank):
            # fbank: (B, 1, 512, 128) — Kaldi fbank features
            features = self.eat_model.extract_features(fbank)
            features = features[:, 1:].mean(dim=1)  # mean pool excl CLS
            return self.head(features)

    wrapper = EATWrapper(module.encoder.model, module.head)
    wrapper.eval()

    # Test forward pass
    dummy_fbank = torch.randn(1, 1, 512, 128)
    with torch.no_grad():
        ref_out = wrapper(dummy_fbank)
    logger.info(f"Test forward output shape: {ref_out.shape}")

    # Export ONNX
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info(f"Exporting ONNX to {onnx_path}")
    torch.onnx.export(
        wrapper,
        (dummy_fbank,),
        str(onnx_path),
        opset_version=17,
        input_names=["fbank"],
        output_names=["logits"],
        dynamic_axes={
            "fbank": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )
    logger.info(f"ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Verify ONNX output matches PyTorch
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"fbank": dummy_fbank.numpy()})[0]
    max_diff = np.max(np.abs(ref_out.numpy() - onnx_out))
    logger.info(f"ONNX verification: max diff = {max_diff:.2e}")
    assert max_diff < 1e-4, f"ONNX output mismatch: {max_diff}"

    # Save species list
    species_path = output_dir / "species_list.json"
    with open(species_path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(f"Species list saved: {species_path}")

    # Save mel config (Kaldi fbank params matching training)
    mel_config = {
        "type": "kaldi_fbank",
        "sr": 16000,
        "num_mel_bins": 128,
        "frame_shift": 10,
        "target_length": 512,
        "norm_mean": -4.268,
        "norm_std": 4.569,
    }
    mel_config_path = output_dir / "mel_config.json"
    with open(mel_config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved: {mel_config_path}")

    logger.info("EAT export complete!")


if __name__ == "__main__":
    main()
