#!/usr/bin/env python3
"""Export EfficientNet SED model to ONNX.

Loads the checkpoint, wraps encoder + SED head with blended sigmoid output,
and exports to ONNX. The ONNX model outputs blended probabilities directly
(no sigmoid needed in the inference kernel).

Usage:
    python scripts/export_efficientnet_sed.py \
        --checkpoint /path/to/best-epoch24.ckpt \
        --config config/efficientnet_b0_sed.yaml \
        --output-dir artifacts/weights/effnet-b0-sed-v1/
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

USER_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


def main():
    parser = argparse.ArgumentParser(description="Export EfficientNet SED to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv",
        help="Taxonomy CSV for 234-species list",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/weights/effnet-b0-sed-v1/")
    parser.add_argument(
        "--blend-clip",
        type=float,
        default=0.75,
        help="Blending weight for clipwise predictions",
    )
    parser.add_argument(
        "--blend-frame",
        type=float,
        default=0.25,
        help="Blending weight for framewise-max predictions",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sys.path.insert(0, str(USER_SRC_DIR))

    from birdclef.data.soundscape import build_full_species_list
    from birdclef.encoders.efficientnet import EfficientNetEncoder
    from birdclef.heads.sed_head import AttHead
    from birdclef.lightning.sed_finetune import SEDFinetuneModule

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Build species list (234 from taxonomy.csv)
    species_list = build_full_species_list(args.train_csv)
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes}")

    # Build encoder
    encoder = EfficientNetEncoder(
        model_name=cfg["encoder"]["model_name"],
        pretrained=cfg["encoder"].get("pretrained", True),
    )
    embed_dim = encoder.embed_dim
    logger.info(f"Encoder: {cfg['encoder']['model_name']}, embed_dim={embed_dim}")

    # Build SED head
    sed_head = AttHead(
        in_channels=embed_dim,
        num_classes=num_classes,
        hidden_dim=cfg["sed_head"]["hidden_dim"],
        dropout=cfg["sed_head"]["dropout"],
        gem_p=cfg["sed_head"]["gem_p"],
    )

    # Build full module and load checkpoint
    training_cfg = cfg["training"]
    module = SEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        lr=training_cfg["lr"],
        head_lr=training_cfg["head_lr"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=training_cfg["max_epochs"],
        warmup_epochs=training_cfg["warmup_epochs"],
        threshold=training_cfg["threshold"],
        clipwise_weight=training_cfg["clipwise_weight"],
        framewise_weight=training_cfg["framewise_weight"],
        blend_clip=training_cfg["blend_clip"],
        blend_frame=training_cfg["blend_frame"],
    )

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    # Filter metric state mismatches
    real_missing = [
        k
        for k in missing
        if not any(m in k for m in ["cmap", "auroc", "f1_macro", "f1_micro", "top1", "top5"])
    ]
    if real_missing:
        logger.warning(f"Missing keys (non-metric): {real_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:10]}")
    logger.info("Checkpoint loaded")

    # Create ONNX export wrapper that outputs blended probabilities
    class SEDWrapper(nn.Module):
        """Wraps encoder + SED head → blended sigmoid probabilities."""

        def __init__(self, encoder, sed_head, blend_clip=0.75, blend_frame=0.25):
            super().__init__()
            self.encoder = encoder
            self.sed_head = sed_head
            self.blend_clip = blend_clip
            self.blend_frame = blend_frame

        def forward(self, mel):
            # mel: (B, 1, 128, T)
            features = self.encoder.forward_features(mel)
            clipwise_logits, framewise_logits = self.sed_head(features)
            framewise_max_logits = framewise_logits.max(dim=-1).values

            # Blended sigmoid probabilities
            probs = self.blend_clip * torch.sigmoid(
                clipwise_logits
            ) + self.blend_frame * torch.sigmoid(framewise_max_logits)
            return probs

    wrapper = SEDWrapper(
        module.encoder,
        module.sed_head,
        blend_clip=args.blend_clip,
        blend_frame=args.blend_frame,
    )
    wrapper.eval()

    # Test forward pass with 10s input at 32kHz
    n_mels = cfg.get("mel", {}).get("n_mels", 128)
    hop_length_cfg = cfg.get("mel", {}).get("hop_length", 320)
    window_duration = cfg["data"]["window_duration"]
    sample_rate = cfg["data"]["sample_rate"]
    target_frames = int(window_duration * sample_rate / hop_length_cfg)
    dummy_mel = torch.randn(1, 1, n_mels, target_frames)
    with torch.no_grad():
        ref_out = wrapper(dummy_mel)
    logger.info(
        f"Test output shape: {ref_out.shape}, range: [{ref_out.min():.4f}, {ref_out.max():.4f}]"
    )

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
        output_names=["probs"],
        dynamic_axes={
            "mel": {0: "batch", 3: "time"},
            "probs": {0: "batch"},
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
    if max_diff > 1e-4:
        logger.warning(
            f"Max diff {max_diff:.2e} > 1e-4 (expected for larger inputs, e.g. n_mels=224)"
        )
    assert max_diff < 1e-2, f"ONNX output mismatch too large: {max_diff}"

    # Save species list
    species_path = output_dir / "species_list.json"
    with open(species_path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(f"Species list saved: {species_path}")

    # Save mel config (matching NNAudioMelSpectrogram params from training config)
    inference_window = window_duration  # match training window at inference

    mel_cfg = cfg.get("mel", {})
    mel_config = {
        "type": "mel",
        "sr": sample_rate,
        "n_fft": mel_cfg.get("n_fft", 1024),
        "hop_length": hop_length_cfg,
        "n_mels": n_mels,
        "fmin": 0,
        "fmax": mel_cfg.get("fmax", None),
        "htk": True,
        "target_frames": target_frames,
        "norm_mean": -4.268,
        "norm_std": 4.569,
        "output_type": "probs",
        "architecture": "sed",
        "blend_clip": args.blend_clip,
        "blend_frame": args.blend_frame,
        "window_duration": window_duration,
        "inference_window": inference_window,
    }
    mel_config_path = output_dir / "mel_config.json"
    with open(mel_config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved: {mel_config_path}")

    logger.info("EfficientNet SED export complete!")


if __name__ == "__main__":
    main()
