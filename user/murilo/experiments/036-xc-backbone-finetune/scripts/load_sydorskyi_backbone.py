"""Load a Sydorskyi 2025 pretrained backbone ckpt into a timm features_only model.

Usage (Python API):
    from scripts.load_sydorskyi_backbone import load_sydorskyi_backbone
    backbone = load_sydorskyi_backbone("tf_efficientnetv2_s_in21k")
    # → timm features_only model with Sydorskyi 2024 pretrained weights, in_chans=1

Usage (CLI sanity check):
    python scripts/load_sydorskyi_backbone.py --backbone tf_efficientnetv2_s_in21k

Confirms:
- timm features_only mode strips conv_head/bn2/classifier (matches Sydorskyi's stripped ckpt key set)
- model.load_state_dict(ckpt, strict=True) succeeds with no missing/unexpected keys
- All conv weights have non-zero norms after load (rules out silent all-zeros load)
- backbone.num_features matches expected (256 for EffNetV2-S, 2304 for NFNet-L0)

Per docs/sydorskyi-config-verified.md, the ckpt mapping is:
    tf_efficientnetv2_s_in21k → models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth
    eca_nfnet_l0              → models_2025/eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt

(NOT the 2025 SWA EffNetV2-S — verified absent from all final selected configs.)
"""

import argparse
from pathlib import Path

import timm
import torch

# Default ckpt root on PACE shared storage. Override via SYDORSKYI_CKPT_ROOT env var
# or the `ckpt_root` arg to load_sydorskyi_backbone.
DEFAULT_CKPT_ROOT = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/pretrained-models/sydorskyi-2025"
)

# Map: timm backbone name → relative ckpt path (under ckpt_root)
SYDORSKYI_CKPT_MAP = {
    "tf_efficientnetv2_s_in21k": "models_2024/tf_efficientnetv2_s_in21k_Pretrainversion1.pth",
    "eca_nfnet_l0": "models_2025/eca_nfnet_l0_pretrain_from_bigXCV2_best.ckpt",
}

# Expected output channels (timm features_only mode) — verified by Phase A.1.4 ckpt inspection.
EXPECTED_OUTPUT_CHANNELS = {
    "tf_efficientnetv2_s_in21k": 256,
    "eca_nfnet_l0": 2304,
}


def load_sydorskyi_backbone(
    backbone_name: str,
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    exportable: bool = True,
    verbose: bool = True,
) -> torch.nn.Module:
    """Instantiate a timm features_only backbone and load Sydorskyi's pretrained ckpt.

    Args:
        backbone_name: timm model name. Must be a key of SYDORSKYI_CKPT_MAP.
        ckpt_root: Path to the directory containing models_2024/ and models_2025/.
        exportable: pass-through to timm.create_model (matches Sydorskyi exportable=True).
        verbose: print sanity-check info.

    Returns:
        timm.Model in features_only mode with Sydorskyi pretrained weights loaded.

    Raises:
        ValueError: unknown backbone or output dim mismatch.
        RuntimeError: ckpt load failed strict check, or all-zeros conv weights detected.
    """
    if backbone_name not in SYDORSKYI_CKPT_MAP:
        raise ValueError(f"Unknown backbone {backbone_name!r}. Known: {list(SYDORSKYI_CKPT_MAP)}")

    ckpt_path = Path(ckpt_root) / SYDORSKYI_CKPT_MAP[backbone_name]
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Pretrained ckpt not found: {ckpt_path}")

    # Build backbone in features_only mode (strips conv_head/bn2/classifier
    # automatically — matches Sydorskyi's stripped ckpt).
    backbone = timm.create_model(
        backbone_name,
        features_only=True,
        pretrained=False,
        exportable=exportable,
        in_chans=1,
    )

    # Load the ckpt (flat OrderedDict, no wrapping — see Phase A.1.4 findings).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Expected ckpt to be a dict/OrderedDict; got {type(ckpt).__name__}. "
            f"This is unexpected — Sydorskyi's create_pretrain_backbone.py should produce a flat dict."
        )

    # Verify no Lightning-style wrapping (defensive — Phase A.1.4 confirmed this).
    if "state_dict" in ckpt:
        if verbose:
            print("  [WARN] ckpt has 'state_dict' wrapping — unwrapping")
        ckpt = ckpt["state_dict"]

    # Strip any leftover 'model.backbone.' / 'backbone.' prefix (defensive).
    cleaned = {}
    for k, v in ckpt.items():
        new_k = k
        for prefix in ("model.backbone.", "backbone."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix) :]
                break
        cleaned[new_k] = v

    # Strict load. This must NOT silently skip keys — Sydorskyi ckpts match the
    # features_only key set exactly.
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Ckpt key mismatch:\n"
            f"  missing in ckpt  ({len(missing)}): {missing[:10]}\n"
            f"  extra in ckpt    ({len(unexpected)}): {unexpected[:10]}"
        )

    # Sanity: assert no all-zero conv weights (catches silent-load failures).
    n_zero_layers = 0
    for name, param in backbone.named_parameters():
        if "conv" in name and "weight" in name and param.dim() == 4:
            if param.detach().abs().mean().item() < 1e-6:
                n_zero_layers += 1
    if n_zero_layers > 0:
        raise RuntimeError(
            f"Detected {n_zero_layers} conv layers with near-zero weights post-load — "
            f"silent-load failure suspected."
        )

    # Sanity: assert backbone output channels match expectation.
    out_chans = backbone.feature_info.channels()[-1]
    expected = EXPECTED_OUTPUT_CHANNELS[backbone_name]
    if out_chans != expected:
        raise ValueError(
            f"Backbone {backbone_name} produced {out_chans} output channels; expected {expected}."
        )

    if verbose:
        print(f"  Loaded {backbone_name} from {ckpt_path.name}")
        print(f"  features_only output channels: {out_chans}")
        print(f"  num params: {sum(p.numel() for p in backbone.parameters()):,}")
        print(
            f"  conv-weight mean (1st conv): {next(backbone.parameters()).detach().abs().mean().item():.4e}"
        )

    return backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        default="tf_efficientnetv2_s_in21k",
        choices=list(SYDORSKYI_CKPT_MAP),
        help="Which Sydorskyi pretrained backbone to load.",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=DEFAULT_CKPT_ROOT,
        help=f"Root dir containing models_2024/ and models_2025/ (default: {DEFAULT_CKPT_ROOT})",
    )
    args = parser.parse_args()

    backbone = load_sydorskyi_backbone(args.backbone, ckpt_root=args.ckpt_root, verbose=True)

    # Forward-pass smoke test
    print("\n  Smoke test: random (B=2, 1, 128, 313) mel → backbone")
    x = torch.randn(2, 1, 128, 313)
    with torch.no_grad():
        out_list = backbone(x)
    print(f"  Got {len(out_list)} feature maps from features_only mode")
    print(
        f"  Last stage shape: {tuple(out_list[-1].shape)} (expected channels: {backbone.feature_info.channels()[-1]})"
    )
    print("\n  PASS")


if __name__ == "__main__":
    main()
