#!/usr/bin/env python3
"""Export fine-tuned SSAMBAClassifier to ONNX with mamba-onnx custom ops.

Wraps the backbone + classification head in a TokenLogitsWrapper that outputs
per-token logits (B, T, C). The Kaggle inference script then splits into
windows, mean-pools, and applies sigmoid.

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
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.mamba import BidirectionalMamba
from ssamba_hf.model import SSAMBAModel

from birdclef_009.data.finetune import build_species_list
from birdclef_009.models.finetune import SSAMBAClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ONNX custom op for selective scan (reused from exp 005)
# ---------------------------------------------------------------------------

class SelectiveScanONNXExport(torch.autograd.Function):
    """Emit a single mamba::SelectiveScanFusedExact node during ONNX export.

    forward() returns a shape stub (zeros) -- actual compute is done by
    ONNX Runtime's custom op at inference time.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, z):
        return torch.zeros_like(u)

    @staticmethod
    def symbolic(g, u, delta, A, B, C, D, z):
        return g.op(
            "mamba::SelectiveScanFusedExact",
            u, delta, A, B, C, D, z,
        )


def _ssm_forward_onnx(
    self,
    x: torch.Tensor,
    z: torch.Tensor,
    conv1d: nn.Conv1d,
    x_proj: nn.Linear,
    dt_proj: nn.Linear,
    A_log: nn.Parameter,
    D: nn.Parameter,
) -> torch.Tensor:
    """ONNX-exportable replacement for BidirectionalMamba._ssm_forward_pytorch.

    Standard ops (conv1d, silu, linear projections) trace normally.
    The selective scan recurrence becomes a single custom op node.
    """
    B, L, d_inner = x.shape

    # Conv1d: (B, L, D) -> (B, D, L) -> conv -> truncate -> silu
    x = x.transpose(1, 2)
    x = conv1d(x)[:, :, :L]
    x = F.silu(x)

    # Project to dt, B_mat, C_mat via standard linear ops
    x_dbl = x_proj(x.transpose(1, 2))  # (B, L, dt_rank + 2*d_state)
    dt, B_mat, C_mat = x_dbl.split(
        [self.dt_rank, self.d_state, self.d_state], dim=-1
    )

    # dt_proj WITHOUT softplus -- the fused op handles softplus internally.
    dt = F.linear(dt, dt_proj.weight, dt_proj.bias)  # (B, L, d_inner)
    dt = dt.transpose(1, 2)  # (B, d_inner, L)

    # A from log-parameterization
    A = -torch.exp(A_log.float())  # (d_inner, d_state)

    # Prepare z: (B, L, d_inner) -> (B, d_inner, L)
    z = z.transpose(1, 2).contiguous()

    # u is already (B, d_inner, L) from the conv1d path
    u = x.contiguous()

    # Single custom op node replaces the entire sequential scan + gating
    y = SelectiveScanONNXExport.apply(u, dt, A, B_mat, C_mat, D.float(), z)

    # Output: (B, d_inner, L) -> (B, L, d_inner)
    return y.transpose(1, 2)


# ---------------------------------------------------------------------------
# TokenLogitsWrapper: backbone + head, outputs per-token logits
# ---------------------------------------------------------------------------

class TokenLogitsWrapper(nn.Module):
    """Wraps SSAMBAClassifier for ONNX export.

    Input: (B, 1, 128, 1000) mel spectrogram
    Output: (B, T, C) per-token logits (head applied per-token, no pooling)

    The Kaggle script handles windowed mean-pooling and sigmoid.
    """

    def __init__(self, classifier: SSAMBAClassifier):
        super().__init__()
        self.backbone = classifier.backbone
        self.head = classifier.head

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Get all tokens including CLS
        all_tokens = self.backbone.forward_features(mel, return_all_tokens=True)

        # Remove CLS token
        cls_pos = self.backbone._cls_token_position
        patch_tokens = torch.cat(
            [all_tokens[:, :cls_pos, :], all_tokens[:, cls_pos + 1:, :]],
            dim=1,
        )

        # Apply head per-token: (B, T, D) -> (B, T, C)
        logits = self.head(patch_tokens)
        return logits


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def export_onnx(wrapper, dummy_input, output_path, opset_version=17):
    """Export TokenLogitsWrapper to ONNX with mamba-onnx custom ops.

    Monkey-patches BidirectionalMamba._ssm_forward_pytorch during export.
    """
    original_method = BidirectionalMamba._ssm_forward_pytorch

    try:
        BidirectionalMamba._ssm_forward_pytorch = _ssm_forward_onnx

        torch.onnx.export(
            wrapper,
            (dummy_input,),
            output_path,
            opset_version=opset_version,
            custom_opsets={"mamba": 1},
            input_names=["mel"],
            output_names=["token_logits"],
            dynamic_axes={
                "mel": {0: "batch"},
                "token_logits": {0: "batch"},
            },
            dynamo=False,
        )
    finally:
        BidirectionalMamba._ssm_forward_pytorch = original_method


def main():
    parser = argparse.ArgumentParser(description="Export SSAMBAClassifier to ONNX")
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

    # Build SSAMBA config (no mel layer needed -- we take mel as input)
    config = SSAMBAConfig(
        img_size=tuple(model_cfg["img_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
    )

    # Build a dummy mel layer (not used in ONNX -- just needed to construct SSAMBAClassifier)
    # We use a simple identity since we only extract backbone + head
    mel_spec_placeholder = nn.Identity()

    model = SSAMBAClassifier(
        config=config,
        num_classes=num_classes,
        mel_spec_layer=mel_spec_placeholder,
        target_frames=data_cfg["target_frames"],
        head_dropout=0.0,  # No dropout at inference
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

    # Load weights (strict=False to skip mel_spec if present)
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        # Filter out mel_spec keys which are expected to be missing
        real_missing = [k for k in missing if not k.startswith("mel_spec")]
        if real_missing:
            logger.warning(f"Missing keys: {real_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Set dropout to 0 in head for export
    for module in model.head.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0

    model.eval()
    logger.info("Model loaded and set to eval mode")

    # Wrap for export
    wrapper = TokenLogitsWrapper(model)
    wrapper.eval()

    # Dummy input: (B, 1, n_mels, target_frames)
    n_mels = data_cfg["n_mels"]
    target_frames = data_cfg["target_frames"]
    dummy_input = torch.randn(1, 1, n_mels, target_frames)

    # Trigger forward once to set _cls_token_position
    with torch.no_grad():
        test_out = wrapper(dummy_input)
    logger.info(f"Test forward output shape: {test_out.shape}")  # (1, T, C)

    # Export
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "ssamba_classifier.onnx"
    logger.info(f"Exporting ONNX to {onnx_path}")
    export_onnx(wrapper, dummy_input, str(onnx_path))
    logger.info(f"ONNX model saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Export species list
    species_path = output_dir / "species_list.json"
    with open(species_path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(f"Species list saved: {species_path} ({num_classes} species)")

    # Export mel config
    sample_rate = data_cfg["sample_rate"]
    hop_length = int(sample_rate * data_cfg["chunk_duration"] / target_frames)
    mel_config = {
        "sr": sample_rate,
        "n_fft": 1024,
        "hop_length": hop_length,
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
