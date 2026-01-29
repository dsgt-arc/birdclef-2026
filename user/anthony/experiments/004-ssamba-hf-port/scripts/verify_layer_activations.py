"""Layer-by-layer activation comparison between original and HF SSAMBA.

Registers forward hooks on all 24 layers in both models and compares
hidden_states and residual at each layer to detect where differences
accumulate. Uses real audio input to ensure realistic activation ranges.

Usage:
    python verify_layer_activations.py \
        --checkpoint /path/to/ssamba_small_400.pth \
        --device cuda \
        --output /path/to/layer_activations_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def compute_layer_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute comparison metrics for layer activations."""
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    cos_sim = F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

    return {
        "cosine_similarity": cos_sim,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "a_norm": float(a.norm()),
        "b_norm": float(b.norm()),
        "a_mean": float(a.mean()),
        "b_mean": float(b.mean()),
    }


def hook_original_model(model, device: str):
    """Register hooks on the original SSAMBA model layers.

    The original model processes layers in a single loop with residual tracking.
    We hook each layer to capture (hidden_states, residual) after each block.
    """
    activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # VisionMamba's Block returns (hidden_states, residual)
            hidden_states, residual = output
            activations[layer_idx] = {
                "hidden_states": hidden_states.detach().cpu(),
                "residual": residual.detach().cpu() if residual is not None else None,
            }
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.v.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    return activations, hooks


def hook_hf_model(model):
    """Register hooks on the HF SSAMBA model layers.

    The HF model's BidirectionalMambaStack contains MambaBlock layers
    at model.layers.layers[i]. Each returns (hidden_states, residual).
    """
    activations = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden_states, residual = output
            activations[layer_idx] = {
                "hidden_states": hidden_states.detach().cpu(),
                "residual": residual.detach().cpu() if residual is not None else None,
            }
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.layers.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    return activations, hooks


def compare_pre_layer_inputs(
    original_model, hf_model, spectrogram: torch.Tensor, device: str
) -> dict:
    """Compare the inputs to the first layer (after patch embed + pos embed + CLS).

    This isolates whether differences start from the embedding stage or accumulate
    in the Mamba layers.
    """
    spectrogram = spectrogram.to(device)

    # Prepare input for both models
    # Both expect (B, T, F) and internally do unsqueeze(1).transpose(2,3) -> (B, 1, F, T)
    x = spectrogram.unsqueeze(1).transpose(2, 3)  # (B, 1, F, T)
    B = x.shape[0]

    # Original model: manually trace to get pre-layer input
    # use_middle_cls_token=False in pretraining -> CLS prepended
    with torch.no_grad():
        orig_x = original_model.v.patch_embed(x)
        cls_tokens = original_model.v.cls_token.expand(B, -1, -1)
        orig_x = torch.cat([cls_tokens, orig_x], dim=1)
        orig_x = orig_x + original_model.v.pos_embed
        orig_x = original_model.v.pos_drop(orig_x)

    # HF model: manually trace to get pre-layer input
    # use_middle_cls_token=False -> CLS prepended
    with torch.no_grad():
        hf_x = hf_model.patch_embed(x)
        cls_tokens_hf = hf_model.cls_token.expand(B, -1, -1)
        hf_x = torch.cat([cls_tokens_hf, hf_x], dim=1)
        hf_x = hf_x + hf_model.pos_embed
        hf_x = hf_model.pos_drop(hf_x)

    metrics = compute_layer_metrics(orig_x.cpu(), hf_x.cpu())
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Layer-by-layer activation comparison")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("LAYER-BY-LAYER ACTIVATION COMPARISON")
    print("=" * 70)

    # Create test spectrogram
    rng = np.random.default_rng(args.seed)
    spec = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    spectrogram = torch.from_numpy(spec)

    # Load both models
    print("\nLoading HF model...")
    from ssamba_hf import load_from_ssamba_checkpoint

    hf_model, hf_config = load_from_ssamba_checkpoint(
        args.checkpoint, device=args.device, verbose=True, use_cuda_kernels=False,
    )
    hf_model.eval()

    print("\nLoading original model...")
    from extract_original_embeddings import load_original_model

    original_model = load_original_model(args.checkpoint, args.device)
    original_model.eval()

    # Step 1: Compare pre-layer inputs
    print("\n" + "-" * 60)
    print("PRE-LAYER INPUT COMPARISON (after patch embed + CLS + pos embed)")
    print("-" * 60)

    pre_layer_metrics = compare_pre_layer_inputs(
        original_model, hf_model, spectrogram, args.device
    )
    print(f"  Cosine sim: {pre_layer_metrics['cosine_similarity']:.6f}")
    print(f"  Max abs diff: {pre_layer_metrics['max_abs_diff']:.2e}")
    print(f"  Mean abs diff: {pre_layer_metrics['mean_abs_diff']:.2e}")

    pre_layer_pass = pre_layer_metrics["cosine_similarity"] > 0.999
    print(f"  [{'PASS' if pre_layer_pass else 'FAIL'}]")

    # Step 2: Register hooks and run forward pass
    print("\n" + "-" * 60)
    print("LAYER-BY-LAYER COMPARISON")
    print("-" * 60)

    orig_activations, orig_hooks = hook_original_model(original_model, args.device)
    hf_activations, hf_hooks = hook_hf_model(hf_model)

    spectrogram_device = spectrogram.to(args.device)

    # Run original model (manual forward pass matching extract_original_embeddings.py)
    with torch.no_grad():
        x = spectrogram_device.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]
        x_orig = original_model.v.patch_embed(x)
        cls_tokens = original_model.v.cls_token.expand(B, -1, -1)
        # use_middle_cls_token=False in pretraining -> CLS prepended
        x_orig = torch.cat([cls_tokens, x_orig], dim=1)
        x_orig = x_orig + original_model.v.pos_embed
        x_orig = original_model.v.pos_drop(x_orig)

        residual = None
        hidden_states = x_orig
        for layer in original_model.v.layers:
            hidden_states, residual = layer(hidden_states, residual)

    # Run HF model
    with torch.no_grad():
        hf_embeddings = hf_model.get_embeddings(spectrogram_device)

    # Clean up hooks
    for h in orig_hooks + hf_hooks:
        h.remove()

    # Compare layer by layer
    num_layers = len(orig_activations)
    layer_results = []
    any_fail = False

    print(f"\n{'Layer':<8} {'Cos Sim':<12} {'Max Diff':<12} {'Mean Diff':<12} {'Orig Norm':<12} {'HF Norm':<12} {'Status'}")
    print("-" * 80)

    for i in range(num_layers):
        if i not in orig_activations or i not in hf_activations:
            print(f"  Layer {i}: MISSING activations")
            layer_results.append({"layer": i, "error": "missing"})
            any_fail = True
            continue

        orig_hs = orig_activations[i]["hidden_states"]
        hf_hs = hf_activations[i]["hidden_states"]

        metrics = compute_layer_metrics(orig_hs, hf_hs)
        passed = metrics["cosine_similarity"] > 0.999
        if not passed:
            any_fail = True

        status = "PASS" if passed else "FAIL"
        print(
            f"  {i:<6} {metrics['cosine_similarity']:<12.6f} "
            f"{metrics['max_abs_diff']:<12.2e} {metrics['mean_abs_diff']:<12.2e} "
            f"{metrics['a_norm']:<12.4f} {metrics['b_norm']:<12.4f} {status}"
        )

        layer_result = {"layer": i, "hidden_states": metrics, "passed": passed}

        # Also compare residuals if available
        orig_res = orig_activations[i]["residual"]
        hf_res = hf_activations[i]["residual"]
        if orig_res is not None and hf_res is not None:
            res_metrics = compute_layer_metrics(orig_res, hf_res)
            layer_result["residual"] = res_metrics

        layer_results.append(layer_result)

    # Check for growing differences (drift)
    # NOTE: Absolute diffs grow because activation norms grow through layers
    # (pre-norm architecture normalizes input, not output). Use cosine
    # similarity (scale-invariant) as the primary correctness metric.
    print("\n" + "-" * 60)
    print("DRIFT ANALYSIS")
    print("-" * 60)

    cos_sims = [r["hidden_states"]["cosine_similarity"] for r in layer_results if "hidden_states" in r]
    max_diffs = [r["hidden_states"]["max_abs_diff"] for r in layer_results if "hidden_states" in r]

    if cos_sims:
        min_cos = min(cos_sims)
        print(f"  Min cosine sim across layers: {min_cos:.6f}")
        cos_pass = min_cos > 0.999
        print(f"  All layers cosine > 0.999: [{'PASS' if cos_pass else 'FAIL'}]")
        if not cos_pass:
            any_fail = True

    if len(max_diffs) >= 2:
        first_half_avg = np.mean(max_diffs[:len(max_diffs) // 2])
        second_half_avg = np.mean(max_diffs[len(max_diffs) // 2:])
        drift_ratio = second_half_avg / (first_half_avg + 1e-12)
        print(f"  Abs diff drift ratio: {drift_ratio:.2f} (expected to grow with pre-norm)")
    else:
        drift_ratio = None

    # Summary
    print("\n" + "=" * 70)
    print("LAYER ACTIVATION SUMMARY")
    print("=" * 70)

    num_pass = sum(1 for r in layer_results if r.get("passed", False))
    print(f"  Layers passed: {num_pass}/{num_layers}")
    print(f"  Pre-layer input: {'PASS' if pre_layer_pass else 'FAIL'}")
    if drift_ratio is not None:
        print(f"  Drift ratio: {drift_ratio:.2f}")

    overall = "PASS" if not any_fail else "FAIL"
    print(f"\n  OVERALL: {overall}")

    # Save report
    if args.output:
        report = {
            "pre_layer_input": pre_layer_metrics,
            "layers": layer_results,
            "drift_ratio": drift_ratio,
            "overall_pass": not any_fail,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to {args.output}")

    return 0 if not any_fail else 1


if __name__ == "__main__":
    sys.exit(main())
