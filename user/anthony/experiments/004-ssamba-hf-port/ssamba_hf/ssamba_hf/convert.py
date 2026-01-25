"""Weight conversion utilities for SSAMBA."""

import re
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .config import SSAMBAConfig
from .model import SSAMBAModel


def convert_ssamba_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert original SSAMBA checkpoint weights to HuggingFace format.

    The original SSAMBA uses mamba_ssm which has slightly different parameter names
    than HuggingFace's MambaMixer.

    Key mappings:
        module.v.patch_embed.proj.{weight,bias} -> patch_embed.proj.{weight,bias}
        module.v.pos_embed -> pos_embed
        module.v.cls_token -> cls_token
        module.v.layers.{i}.mixer.* -> layers.layers.{i}.mixer.*
        module.v.layers.{i}.norm.weight -> layers.layers.{i}.norm.weight
        module.v.norm_f.weight -> norm_f.weight
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Remove DataParallel wrapper prefix
        if new_key.startswith("module."):
            new_key = new_key[7:]

        # Remove VisionMamba wrapper prefix
        if new_key.startswith("v."):
            new_key = new_key[2:]

        # Skip pretraining-specific parameters
        if any(skip in new_key for skip in [
            "cpredlayer", "gpredlayer", "mask_embed", "mlp_head",
            "p_input_fdim", "p_input_tdim", "unfold"
        ]):
            continue

        # Map layer keys
        # Original: layers.{i}.mixer.* -> layers.layers.{i}.mixer.*
        # Original: layers.{i}.norm.* -> layers.layers.{i}.norm.*
        layer_match = re.match(r"layers\.(\d+)\.(.*)", new_key)
        if layer_match:
            layer_idx = layer_match.group(1)
            rest = layer_match.group(2)
            new_key = f"layers.layers.{layer_idx}.{rest}"

        new_state_dict[new_key] = value

    return new_state_dict


def verify_weight_shapes(
    model: SSAMBAModel,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[list, list, list]:
    """Verify that converted weights match model's expected shapes.

    Returns:
        Tuple of (matched_keys, missing_keys, unexpected_keys)
    """
    model_state = model.state_dict()

    matched = []
    mismatched = []
    missing = []

    for key, param in model_state.items():
        if key in state_dict:
            if param.shape == state_dict[key].shape:
                matched.append(key)
            else:
                mismatched.append(
                    f"{key}: model={param.shape}, checkpoint={state_dict[key].shape}"
                )
        else:
            missing.append(key)

    unexpected = [k for k in state_dict.keys() if k not in model_state]

    return matched, missing, unexpected, mismatched


def load_from_ssamba_checkpoint(
    checkpoint_path: str,
    config: SSAMBAConfig = None,
    device: str = "cpu",
    strict: bool = False,
) -> Tuple[SSAMBAModel, SSAMBAConfig]:
    """Load SSAMBA model from original checkpoint.

    Args:
        checkpoint_path: Path to original SSAMBA checkpoint
        config: Optional config (will be inferred from checkpoint if not provided)
        device: Device to load model on
        strict: If True, raise error on missing/unexpected keys

    Returns:
        Tuple of (model, config)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer config if not provided
    if config is None:
        config = SSAMBAConfig.from_checkpoint(checkpoint_path, device=device)

    # Create model
    model = SSAMBAModel(config)

    # Convert weights
    converted_weights = convert_ssamba_weights(checkpoint)

    # Verify shapes
    matched, missing, unexpected, mismatched = verify_weight_shapes(model, converted_weights)

    if mismatched:
        raise ValueError(f"Shape mismatches: {mismatched}")

    if strict and (missing or unexpected):
        raise ValueError(f"Missing keys: {missing}, Unexpected keys: {unexpected}")

    # Load weights (non-strict to handle HuggingFace Mamba internal state)
    result = model.load_state_dict(converted_weights, strict=False)

    print(f"Loaded {len(matched)} weights successfully")
    if result.missing_keys:
        # Filter out expected missing keys (HuggingFace internal state)
        actual_missing = [k for k in result.missing_keys
                         if not any(skip in k for skip in ["conv_state", "ssm_state"])]
        if actual_missing:
            print(f"Missing keys: {actual_missing}")
    if result.unexpected_keys:
        print(f"Unexpected keys: {result.unexpected_keys}")

    return model.to(device), config
