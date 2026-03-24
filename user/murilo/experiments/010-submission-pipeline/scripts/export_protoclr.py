#!/usr/bin/env python3
"""Export ProtoCLR (CvT-13) LoRA fine-tuned model to ONNX.

Loads the PEFT LoRA checkpoint, merges adapters into base weights, wraps
CvT model + head for mel input, and exports to ONNX with opset 17.

Note: CvT may use einops internally which can cause ONNX tracing issues.
If ONNX export fails, falls back to TorchScript (torch.jit.trace).

Usage:
    python scripts/export_protoclr.py \
        --checkpoint /path/to/best-epochepoch=20.ckpt \
        --config config/protoclr_lora_attn.yaml \
        --output-dir artifacts/weights/protoclr/
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

EXPERIMENT_009_DIR = Path(__file__).resolve().parent.parent.parent / "009-protoclr-finetune"
USER_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


def main():
    parser = argparse.ArgumentParser(description="Export ProtoCLR to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Lightning checkpoint path")
    parser.add_argument(
        "--config",
        type=str,
        default=str(EXPERIMENT_009_DIR / "config" / "protoclr_lora_attn.yaml"),
        help="ProtoCLR LoRA config YAML",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2025/train.csv",
        help="BirdCLEF 2025 train.csv for species list",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/weights/protoclr/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sys.path.insert(0, str(USER_SRC_DIR))

    from peft import LoraConfig, get_peft_model

    from birdclef.data.dataset import build_species_list
    from birdclef.encoders.protoclr import ProtoCLREncoder
    from birdclef.lightning.lora_finetune import LoRAFinetuneModule

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build species list
    species_list = build_species_list(args.train_csv)
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes}")

    # Build encoder (downloads CvT from HuggingFace)
    encoder_cfg = cfg["encoder"]
    encoder = ProtoCLREncoder(
        model_name=encoder_cfg.get("model_name", "ilyassmoummad/ProtoCLR"),
        checkpoint=encoder_cfg.get("checkpoint", "protoclr.pth"),
    )

    # Apply PEFT LoRA to encoder.model (CvT)
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    encoder.model = get_peft_model(encoder.model, lora_config)
    logger.info("LoRA applied to ProtoCLR model")

    # Build full module (encoder + head)
    training_cfg = cfg["training"]
    module = LoRAFinetuneModule(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=encoder_cfg["embed_dim"],
        hidden_dim=training_cfg.get("hidden_dim", 0),
        dropout=training_cfg.get("dropout", 0.0),
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=True)
    logger.info("Checkpoint loaded successfully")

    # Merge LoRA weights into CvT model
    module.encoder.model = module.encoder.model.merge_and_unload()
    logger.info("LoRA merged into CvT model")

    # Create ONNX export wrapper
    class ProtoCLRWrapper(nn.Module):
        def __init__(self, cvt_model, head):
            super().__init__()
            self.cvt_model = cvt_model
            self.head = head

        def forward(self, mel):
            # mel: (B, 1, 128, 250)
            features = self.cvt_model.forward_features(mel)  # (B, 384)
            return self.head(features)

    wrapper = ProtoCLRWrapper(module.encoder.model, module.head)
    wrapper.eval()

    # Test forward pass
    dummy_mel = torch.randn(1, 1, 128, 250)
    with torch.no_grad():
        ref_out = wrapper(dummy_mel)
    logger.info(f"Test forward output shape: {ref_out.shape}")

    # Export ONNX (with fallback to TorchScript if CvT has tracing issues)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    onnx_success = False
    try:
        logger.info(f"Attempting ONNX export to {onnx_path}")
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

        # Verify ONNX output
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_out = session.run(None, {"mel": dummy_mel.numpy()})[0]
        max_diff = np.max(np.abs(ref_out.numpy() - onnx_out))
        logger.info(f"ONNX verification: max diff = {max_diff:.2e}")
        assert max_diff < 1e-4, f"ONNX output mismatch: {max_diff}"
        onnx_success = True

    except Exception as e:
        logger.warning(f"ONNX export failed: {e}")
        logger.info("Falling back to TorchScript export...")

        ts_path = output_dir / "model.pt"
        traced = torch.jit.trace(wrapper, dummy_mel)
        traced.save(str(ts_path))
        logger.info(f"TorchScript saved: {ts_path} ({ts_path.stat().st_size / 1e6:.1f} MB)")

        # Verify TorchScript output
        ts_out = traced(dummy_mel)
        max_diff = np.max(np.abs(ref_out.numpy() - ts_out.detach().numpy()))
        logger.info(f"TorchScript verification: max diff = {max_diff:.2e}")

    # Copy CvT source files to artifacts (needed for TorchScript fallback on Kaggle)
    from huggingface_hub import snapshot_download

    hf_dir = snapshot_download(
        encoder_cfg.get("model_name", "ilyassmoummad/ProtoCLR"),
        allow_patterns=["*.py", "*.json"],
    )
    for fname in ["cvt.py", "config.json"]:
        src = Path(hf_dir) / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            logger.info(f"Copied {fname} to artifacts")

    # Save species list
    species_path = output_dir / "species_list.json"
    with open(species_path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(f"Species list saved: {species_path}")

    # Save mel config (matching ProtoCLR training pipeline)
    mel_config = {
        "type": "protoclr_mel",
        "sr": 16000,
        "n_fft": 1024,
        "hop_length": 320,
        "n_mels": 128,
        "fmin": 50,
        "fmax": 8000,
        "target_frames": 250,
        "protoclr_mean": 0.5347,
        "protoclr_std": 0.0772,
        "export_format": "onnx" if onnx_success else "torchscript",
    }
    mel_config_path = output_dir / "mel_config.json"
    with open(mel_config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved: {mel_config_path}")

    logger.info("ProtoCLR export complete!")


if __name__ == "__main__":
    main()
