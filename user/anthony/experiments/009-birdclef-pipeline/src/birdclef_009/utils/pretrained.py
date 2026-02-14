"""Load pre-trained weights into SSAMBA models.

Supports two sources:
1. Original SSAMBA .pth checkpoints (via ssamba_hf.convert)
2. Our own MSPM pretraining Lightning checkpoints
"""

import logging

import torch

logger = logging.getLogger(__name__)


def load_pretrained_backbone(classifier, checkpoint_path, device="cpu"):
    """Load pre-trained SSAMBA weights into classifier backbone.

    Uses ssamba_hf.convert.convert_ssamba_weights() for key mapping from
    original SSAMBA .pth checkpoints. Automatically handles matching vs
    mismatched shapes.

    Args:
        classifier: Model with .backbone attribute (SSAMBAModel)
        checkpoint_path: Path to original SSAMBA .pth checkpoint
        device: Device to load checkpoint on

    Returns:
        Dict with loading stats
    """
    from ssamba_hf.convert import convert_ssamba_weights

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    converted = convert_ssamba_weights(checkpoint)

    return _load_weights_into_backbone(classifier.backbone, converted)


def load_mspm_backbone(classifier, checkpoint_path, device="cpu"):
    """Load weights from our MSPM pretraining checkpoint into classifier backbone.

    MSPM checkpoints are Lightning checkpoints with state_dict keys like
    'model.backbone.*'. We extract backbone weights and load them.

    Args:
        classifier: Model with .backbone attribute (SSAMBAModel)
        checkpoint_path: Path to MSPM Lightning checkpoint
        device: Device to load checkpoint on

    Returns:
        Dict with loading stats
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Extract backbone weights from Lightning state_dict
    prefix = "model.backbone."
    converted = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            converted[key[len(prefix):]] = value

    if not converted:
        raise ValueError(
            f"No backbone weights found with prefix '{prefix}' in checkpoint. "
            f"Available keys: {list(state_dict.keys())[:10]}"
        )

    return _load_weights_into_backbone(classifier.backbone, converted)


def _load_weights_into_backbone(backbone, weight_dict):
    """Load converted weight dict into backbone, handling shape mismatches.

    Args:
        backbone: SSAMBAModel instance
        weight_dict: Dict of {key: tensor} already mapped to backbone key names

    Returns:
        Dict with loading stats
    """
    model_state = backbone.state_dict()
    loaded_keys = []
    skipped_keys = []
    special_keys = []

    # Handle pos_embed specially: center-crop if shapes are close
    if "pos_embed" in weight_dict and "pos_embed" in model_state:
        pre_shape = weight_dict["pos_embed"].shape
        model_shape = model_state["pos_embed"].shape

        if pre_shape == model_shape:
            model_state["pos_embed"].copy_(weight_dict["pos_embed"])
            loaded_keys.append("pos_embed")
        elif pre_shape[2] == model_shape[2]:
            pre_tokens = pre_shape[1]
            model_tokens = model_shape[1]
            cls_embed = weight_dict["pos_embed"][:, :1, :]
            patch_embed_pre = weight_dict["pos_embed"][:, 1:, :]
            model_patch_count = model_tokens - 1
            pre_patch_count = pre_tokens - 1

            if model_patch_count <= pre_patch_count:
                pre_f_dim = backbone.patch_embed.f_dim
                pre_t_dim = pre_patch_count // pre_f_dim
                model_t_dim = backbone.patch_embed.t_dim

                patch_2d = patch_embed_pre.reshape(1, pre_f_dim, pre_t_dim, -1)
                t_start = (pre_t_dim - model_t_dim) // 2
                cropped = patch_2d[:, :, t_start:t_start + model_t_dim, :]
                cropped_flat = cropped.reshape(1, model_patch_count, -1)

                new_pos = torch.cat([cls_embed, cropped_flat], dim=1)
                model_state["pos_embed"].copy_(new_pos)
                special_keys.append(
                    f"pos_embed: center-cropped ({pre_tokens} -> {model_tokens} tokens)"
                )
            else:
                skipped_keys.append(
                    f"pos_embed: model needs more tokens ({model_tokens}) "
                    f"than checkpoint ({pre_tokens}), re-initialized"
                )
        else:
            skipped_keys.append(
                f"pos_embed: embed_dim mismatch {pre_shape} vs {model_shape}, re-initialized"
            )

    # Load all other weights
    for key in model_state:
        if key == "pos_embed":
            continue
        if key in weight_dict:
            if model_state[key].shape == weight_dict[key].shape:
                model_state[key].copy_(weight_dict[key])
                loaded_keys.append(key)
            else:
                skipped_keys.append(
                    f"{key}: shape mismatch model={model_state[key].shape} "
                    f"vs ckpt={weight_dict[key].shape}, re-initialized"
                )

    backbone.load_state_dict(model_state)

    total_params = sum(p.numel() for p in backbone.parameters())
    loaded_params = sum(model_state[k].numel() for k in loaded_keys)
    if any("pos_embed" in s for s in special_keys):
        loaded_params += model_state["pos_embed"].numel()

    pct = 100.0 * loaded_params / total_params if total_params > 0 else 0

    logger.info(f"Pre-trained weight loading summary:")
    logger.info(f"  Loaded: {len(loaded_keys)} params ({loaded_params:,} / {total_params:,} = {pct:.1f}%)")
    logger.info(f"  Special handling: {len(special_keys)}")
    for s in special_keys:
        logger.info(f"    {s}")
    logger.info(f"  Skipped (shape mismatch): {len(skipped_keys)}")
    for s in skipped_keys:
        logger.info(f"    {s}")

    return {
        "matched": loaded_keys,
        "skipped": skipped_keys,
        "special": special_keys,
        "loaded_params": loaded_params,
        "total_params": total_params,
        "pct": pct,
    }
