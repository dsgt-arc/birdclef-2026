"""Load pre-trained SSAMBA weights into an SSAMBAClassifier.

Uses ssamba_hf.convert for key mapping. Automatically handles matching vs
mismatched shapes (e.g. patch_embed/pos_embed differ for condition B 128x1).
"""

import logging

import torch

from ssamba_hf.convert import convert_ssamba_weights, verify_weight_shapes

logger = logging.getLogger(__name__)


def load_pretrained_backbone(classifier, checkpoint_path, device="cpu"):
    """Load pre-trained SSAMBA weights into classifier backbone.

    Uses ssamba_hf.convert.convert_ssamba_weights() for key mapping.
    Automatically handles matching vs mismatched shapes:
    - Matching: direct copy (Mamba layers, norms, cls_token)
    - Mismatched: skip and log (patch_embed.proj, pos_embed when shapes differ)

    For condition A (16x16 with 10s):
      - pos_embed: center-crop from pre-trained (513 -> 497 tokens)
      - patch_embed.proj: identical shape, direct copy

    For condition B (128x1):
      - patch_embed.proj: shape mismatch, skip (random init)
      - pos_embed: shape mismatch, skip (random init)
      - Mamba layers: direct copy (~93% of params)

    Args:
        classifier: SSAMBAClassifier instance (backbone + head)
        checkpoint_path: Path to original SSAMBA .pth checkpoint
        device: Device to load checkpoint on

    Returns:
        Dict with loading stats: {matched, skipped, special}
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    converted = convert_ssamba_weights(checkpoint)

    backbone = classifier.backbone
    model_state = backbone.state_dict()

    matched, missing, unexpected, mismatched = verify_weight_shapes(backbone, converted)

    loaded_keys = []
    skipped_keys = []
    special_keys = []

    # Handle pos_embed specially: center-crop if shapes are close
    if "pos_embed" in converted and "pos_embed" in model_state:
        pre_shape = converted["pos_embed"].shape  # (1, pre_tokens, embed_dim)
        model_shape = model_state["pos_embed"].shape  # (1, model_tokens, embed_dim)

        if pre_shape == model_shape:
            model_state["pos_embed"].copy_(converted["pos_embed"])
            loaded_keys.append("pos_embed")
        elif pre_shape[2] == model_shape[2]:
            # Same embed_dim, different token count -> center-crop
            pre_tokens = pre_shape[1]
            model_tokens = model_shape[1]

            # Separate CLS token (first position in pre-trained)
            # Pre-trained SSAMBA stores pos_embed as (1, num_patches+1, embed_dim)
            # where the first slot is for the CLS token
            cls_embed = converted["pos_embed"][:, :1, :]
            patch_embed_pre = converted["pos_embed"][:, 1:, :]

            model_cls = model_state["pos_embed"][:, :1, :]
            model_patch_count = model_tokens - 1
            pre_patch_count = pre_tokens - 1

            if model_patch_count <= pre_patch_count:
                # Center-crop patch embeddings
                # Reshape to 2D grid for spatial cropping
                # Pre-trained: 8x64=512, Fine-tune: 8x62=496
                pre_f_dim = backbone.patch_embed.f_dim
                # Infer pre-trained t_dim from patch count
                pre_t_dim = pre_patch_count // pre_f_dim
                model_t_dim = backbone.patch_embed.t_dim

                patch_2d = patch_embed_pre.reshape(1, pre_f_dim, pre_t_dim, -1)
                # Center-crop along time dimension
                t_start = (pre_t_dim - model_t_dim) // 2
                cropped = patch_2d[:, :, t_start:t_start + model_t_dim, :]
                cropped_flat = cropped.reshape(1, model_patch_count, -1)

                new_pos = torch.cat([cls_embed, cropped_flat], dim=1)
                model_state["pos_embed"].copy_(new_pos)
                special_keys.append(
                    f"pos_embed: center-cropped ({pre_tokens} -> {model_tokens} tokens, "
                    f"time {pre_t_dim} -> {model_t_dim})"
                )
            else:
                skipped_keys.append(
                    f"pos_embed: model needs more tokens ({model_tokens}) "
                    f"than pre-trained ({pre_tokens}), re-initialized"
                )
        else:
            skipped_keys.append(
                f"pos_embed: embed_dim mismatch {pre_shape} vs {model_shape}, re-initialized"
            )

    # Load all other weights
    for key in model_state:
        if key == "pos_embed":
            continue  # Already handled above

        if key in converted:
            if model_state[key].shape == converted[key].shape:
                model_state[key].copy_(converted[key])
                loaded_keys.append(key)
            else:
                skipped_keys.append(
                    f"{key}: shape mismatch model={model_state[key].shape} "
                    f"vs ckpt={converted[key].shape}, re-initialized"
                )
        # Keys not in converted are left at their random init

    backbone.load_state_dict(model_state)

    # Count params
    total_params = sum(p.numel() for p in backbone.parameters())
    loaded_params = sum(
        model_state[k].numel() for k in loaded_keys
    )
    # Include special keys (like center-cropped pos_embed) in loaded count
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

    missing_from_ckpt = [k for k in model_state if k not in converted and k != "pos_embed"]
    if missing_from_ckpt:
        logger.info(f"  Not in checkpoint (random init): {len(missing_from_ckpt)}")
        for k in missing_from_ckpt[:10]:
            logger.info(f"    {k}")
        if len(missing_from_ckpt) > 10:
            logger.info(f"    ... and {len(missing_from_ckpt) - 10} more")

    return {
        "matched": loaded_keys,
        "skipped": skipped_keys,
        "special": special_keys,
        "loaded_params": loaded_params,
        "total_params": total_params,
        "pct": pct,
    }
