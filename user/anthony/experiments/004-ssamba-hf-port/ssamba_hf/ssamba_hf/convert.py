"""Weight conversion utilities for SSAMBA."""

import re
from typing import Dict, Tuple, List

import torch
import torch.nn as nn

from .config import SSAMBAConfig
from .model import SSAMBAModel


def analyze_checkpoint(checkpoint_path: str, device: str = "cpu") -> Dict:
    """Analyze checkpoint structure to understand weight organization.

    Returns dict with:
        - layer_count: Number of Mamba layers
        - has_bidirectional: Whether layers have _b suffix weights
        - weight_groups: Categorized weight names
    """
    sd = torch.load(checkpoint_path, map_location=device)

    # Find all layer indices
    layer_indices = set()
    forward_weights = []
    backward_weights = []
    other_weights = []

    for key in sd.keys():
        # Check for layer weights
        match = re.search(r"layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))

            # Check if it's a backward weight
            if "_b" in key.split(".")[-1] or ".conv1d_b" in key:
                backward_weights.append(key)
            elif any(x in key for x in ["mixer.", "norm."]):
                forward_weights.append(key)
            else:
                other_weights.append(key)
        else:
            other_weights.append(key)

    return {
        "layer_count": max(layer_indices) + 1 if layer_indices else 0,
        "layer_indices": sorted(layer_indices),
        "has_bidirectional": len(backward_weights) > 0,
        "forward_weights": sorted(forward_weights)[:10],
        "backward_weights": sorted(backward_weights)[:10],
        "other_weights": sorted(other_weights),
        "total_weights": len(sd),
    }


def convert_ssamba_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert original SSAMBA checkpoint weights to HuggingFace format.

    The original SSAMBA uses mamba_ssm which has slightly different parameter names
    than our BidirectionalMamba implementation.

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
) -> Tuple[List, List, List, List]:
    """Verify that converted weights match model's expected shapes.

    Returns:
        Tuple of (matched_keys, missing_keys, unexpected_keys, mismatched)
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
    verbose: bool = True,
) -> Tuple[SSAMBAModel, SSAMBAConfig]:
    """Load SSAMBA model from original checkpoint.

    Args:
        checkpoint_path: Path to original SSAMBA checkpoint
        config: Optional config (will be inferred from checkpoint if not provided)
        device: Device to load model on
        strict: If True, raise error on missing/unexpected keys
        verbose: If True, print detailed loading info

    Returns:
        Tuple of (model, config)
    """
    # Analyze checkpoint first
    if verbose:
        print("Analyzing checkpoint structure...")
        analysis = analyze_checkpoint(checkpoint_path, device=device)
        print(f"  Total weights: {analysis['total_weights']}")
        print(f"  Layer count: {analysis['layer_count']}")
        print(f"  Has bidirectional weights: {analysis['has_bidirectional']}")
        if analysis['forward_weights']:
            print(f"  Sample forward weights: {analysis['forward_weights'][:3]}")
        if analysis['backward_weights']:
            print(f"  Sample backward weights: {analysis['backward_weights'][:3]}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer config if not provided
    if config is None:
        config = SSAMBAConfig.from_checkpoint(checkpoint_path, device=device)

    if verbose:
        print(f"\nModel config:")
        print(f"  embed_dim: {config.embed_dim}")
        print(f"  depth: {config.depth}")
        print(f"  d_state: {config.d_state}")
        print(f"  d_conv: {config.d_conv}")

    # Create model
    model = SSAMBAModel(config)

    if verbose:
        model_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel created with {model_params:,} parameters")

    # Convert weights
    converted_weights = convert_ssamba_weights(checkpoint)

    # Verify shapes
    matched, missing, unexpected, mismatched = verify_weight_shapes(model, converted_weights)

    if verbose:
        print(f"\nWeight loading summary:")
        print(f"  Matched: {len(matched)}")
        print(f"  Missing from checkpoint: {len(missing)}")
        print(f"  Unexpected in checkpoint: {len(unexpected)}")
        print(f"  Shape mismatches: {len(mismatched)}")

        if missing:
            # Group missing by type
            missing_b = [k for k in missing if "_b" in k.split(".")[-1] or ".conv1d_b" in k]
            missing_other = [k for k in missing if k not in missing_b]
            if missing_b:
                print(f"\n  Missing backward weights ({len(missing_b)}):")
                for k in missing_b[:5]:
                    print(f"    - {k}")
                if len(missing_b) > 5:
                    print(f"    ... and {len(missing_b) - 5} more")
            if missing_other:
                print(f"\n  Missing other weights ({len(missing_other)}):")
                for k in missing_other[:5]:
                    print(f"    - {k}")

        if unexpected:
            print(f"\n  Unexpected weights (first 5):")
            for k in unexpected[:5]:
                print(f"    - {k}")

        if mismatched:
            print(f"\n  Shape mismatches:")
            for m in mismatched:
                print(f"    - {m}")

    if mismatched:
        raise ValueError(f"Shape mismatches: {mismatched}")

    if strict and (missing or unexpected):
        raise ValueError(f"Missing keys: {missing}, Unexpected keys: {unexpected}")

    # Load weights (non-strict to handle any architecture differences)
    result = model.load_state_dict(converted_weights, strict=False)

    if verbose:
        print(f"\nLoaded {len(matched)} weights successfully")
        actual_missing = [k for k in result.missing_keys
                         if not any(skip in k for skip in ["conv_state", "ssm_state"])]
        if actual_missing:
            print(f"Actual missing after load: {len(actual_missing)}")

    return model.to(device), config
