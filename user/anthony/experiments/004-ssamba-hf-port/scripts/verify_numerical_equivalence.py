"""Verify numerical equivalence between original and ported SSAMBA implementations."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def compute_metrics(original: torch.Tensor, ported: torch.Tensor) -> dict:
    """Compute comparison metrics between two embedding tensors."""
    # Ensure same dtype for comparison
    original = original.float()
    ported = ported.float()

    # Element-wise differences
    diff = original - ported
    abs_diff = torch.abs(diff)

    metrics = {
        # Basic statistics
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "std_abs_diff": float(abs_diff.std()),

        # Relative differences (avoid division by zero)
        "max_rel_diff": float((abs_diff / (torch.abs(original) + 1e-8)).max()),
        "mean_rel_diff": float((abs_diff / (torch.abs(original) + 1e-8)).mean()),

        # Cosine similarity (per sample, then averaged)
        "cosine_similarity": float(F.cosine_similarity(
            original.flatten().unsqueeze(0),
            ported.flatten().unsqueeze(0)
        ).item()),

        # Additional statistics
        "original_norm": float(original.norm()),
        "ported_norm": float(ported.norm()),
        "norm_diff": float(abs(original.norm() - ported.norm())),

        # Shape info
        "shape": list(original.shape),
    }

    return metrics


def evaluate_success(metrics: dict) -> dict:
    """Evaluate whether the porting was successful based on metrics."""
    criteria = {
        "cosine_similarity_pass": metrics["cosine_similarity"] > 0.999,
        "max_diff_pass": metrics["max_abs_diff"] < 1e-4,
        "mean_diff_pass": metrics["mean_abs_diff"] < 1e-5,
    }

    criteria["overall_pass"] = all(criteria.values())

    return criteria


def main():
    parser = argparse.ArgumentParser(description="Verify numerical equivalence")
    parser.add_argument("--original", type=str, required=True,
                        help="Path to original embeddings")
    parser.add_argument("--ported", type=str, required=True,
                        help="Path to ported embeddings")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for comparison report")
    args = parser.parse_args()

    print(f"Loading original embeddings from {args.original}")
    original_data = torch.load(args.original, map_location="cpu")
    original_embeddings = original_data["embeddings"]

    print(f"Loading ported embeddings from {args.ported}")
    ported_data = torch.load(args.ported, map_location="cpu")
    ported_embeddings = ported_data["embeddings"]

    print(f"Original shape: {original_embeddings.shape}")
    print(f"Ported shape: {ported_embeddings.shape}")

    # Verify same test input was used
    original_spec = original_data["spectrogram"]
    ported_spec = ported_data["spectrogram"]
    if not torch.allclose(original_spec, ported_spec):
        print("WARNING: Input spectrograms differ!")

    print("\nComputing metrics...")
    metrics = compute_metrics(original_embeddings, ported_embeddings)
    criteria = evaluate_success(metrics)

    report = {
        "metrics": metrics,
        "success_criteria": criteria,
        "original_checkpoint": original_data.get("checkpoint", "unknown"),
        "ported_checkpoint": ported_data.get("checkpoint", "unknown"),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(f"\nCosine Similarity: {metrics['cosine_similarity']:.6f}")
    print(f"Max Absolute Diff: {metrics['max_abs_diff']:.2e}")
    print(f"Mean Absolute Diff: {metrics['mean_abs_diff']:.2e}")
    print(f"Original Norm: {metrics['original_norm']:.4f}")
    print(f"Ported Norm: {metrics['ported_norm']:.4f}")

    print("\n" + "-" * 60)
    print("SUCCESS CRITERIA")
    print("-" * 60)
    for criterion, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {criterion}: {status}")

    print("\n" + "=" * 60)
    overall = "SUCCESS" if criteria["overall_pass"] else "FAILURE"
    print(f"OVERALL: {overall}")
    print("=" * 60)

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")

    # Return exit code based on success
    return 0 if criteria["overall_pass"] else 1


if __name__ == "__main__":
    exit(main())
