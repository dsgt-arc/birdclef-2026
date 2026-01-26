"""Unified benchmark runner for encoder comparison on ESC-50."""

import argparse
import json
import time
from pathlib import Path

import torch

from .data import ESC50Dataset
from .encoders import (
    Wav2Vec2Encoder,
    BEATsEncoder,
    EATEncoder,
    ASTEncoder,
    SSASTEncoder,
    SSAMBAEncoder,
    SSAMBAHFEncoder,
)
from .train import evaluate_5fold_cv

# All supported encoders
ENCODER_NAMES = [
    "wav2vec2",
    "beats",
    "eat",
    "ast",
    "ssast",
    "ssamba-tiny",
    "ssamba-small",
    "ssamba-base",
    "ssamba-hf-tiny",
    "ssamba-hf-small",
    "ssamba-hf-base",
]


def get_encoder(encoder_name: str, checkpoint_dir: Path, device: str):
    """Load encoder by name.

    Args:
        encoder_name: One of the supported encoder names
        checkpoint_dir: Directory containing model checkpoints
        device: Device to use

    Returns:
        Initialized encoder
    """
    if encoder_name == "wav2vec2":
        return Wav2Vec2Encoder(device=device)
    elif encoder_name == "beats":
        ckpt_path = checkpoint_dir / "beats" / "BEATs_iter3_plus_AS2M.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"BEATs checkpoint not found at {ckpt_path}. "
                "Please download from OneDrive (see tasks.md)."
            )
        return BEATsEncoder(checkpoint_path=ckpt_path, device=device)
    elif encoder_name == "eat":
        # Use HuggingFace model - will download if needed
        return EATEncoder(
            model_name="worstchan/EAT-base_epoch30_pretrain",
            target_length=512,  # 5 seconds for ESC-50
            device=device,
        )
    elif encoder_name == "ast":
        # Use HuggingFace model - will download if needed
        return ASTEncoder(
            model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
            device=device,
        )
    elif encoder_name == "ssast":
        if SSASTEncoder is None:
            raise ImportError(
                "SSASTEncoder not available. Please ensure timm==0.4.5 is installed."
            )
        ckpt_path = checkpoint_dir / "ssast" / "SSAST-Base-Patch-400.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SSAST checkpoint not found at {ckpt_path}. "
                "Please download from Dropbox (see tasks.md)."
            )
        return SSASTEncoder(
            checkpoint_path=str(ckpt_path),
            model_size="base",
            input_tdim=512,
            device=device,
        )
    elif encoder_name.startswith("ssamba-hf-"):
        # Check ssamba-hf- before ssamba- since "ssamba-hf-small" matches both
        if SSAMBAHFEncoder is None:
            raise ImportError(
                "SSAMBAHFEncoder not available. Please check that the ssamba_hf "
                "module from experiment 004 is accessible."
            )
        size = encoder_name.split("-")[2]  # tiny, small, or base
        ckpt_path = checkpoint_dir / "ssamba" / f"ssamba_{size}_400.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SSAMBA checkpoint not found at {ckpt_path}. "
                "Please download from Google Drive (see tasks.md)."
            )
        return SSAMBAHFEncoder(
            checkpoint_path=str(ckpt_path),
            model_size=size,
            input_tdim=512,
            device=device,
        )
    elif encoder_name.startswith("ssamba-"):
        if SSAMBAEncoder is None:
            raise ImportError(
                "SSAMBAEncoder not available. Please ensure mamba-ssm is installed "
                "and the Vim repo is cloned in vendor/ssamba/."
            )
        size = encoder_name.split("-")[1]  # tiny, small, or base
        ckpt_path = checkpoint_dir / "ssamba" / f"ssamba_{size}_400.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SSAMBA checkpoint not found at {ckpt_path}. "
                "Please download from Google Drive (see tasks.md)."
            )
        return SSAMBAEncoder(
            checkpoint_path=str(ckpt_path),
            model_size=size,
            input_tdim=512,
            device=device,
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Available: {ENCODER_NAMES}")


def benchmark_encoder(
    encoder_name: str,
    *,
    esc50_root: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    skip_existing: bool = True,
) -> dict | None:
    """Run benchmark for a single encoder.

    Args:
        encoder_name: Encoder to benchmark
        esc50_root: Path to ESC-50 dataset
        checkpoint_dir: Path to model checkpoints
        output_dir: Directory to save results
        device: Device to use
        batch_size: Batch size for feature extraction
        epochs: Training epochs
        lr: Learning rate
        skip_existing: Skip if results file already exists

    Returns:
        Results dictionary, or None if skipped
    """
    # Check for existing results
    output_path = output_dir / f"{encoder_name}_results.json"
    if skip_existing and output_path.exists():
        print(f"\n{'='*60}")
        print(f"Skipping {encoder_name}: results already exist at {output_path}")
        print(f"{'='*60}")
        with open(output_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"Benchmarking: {encoder_name}")
    print(f"{'='*60}")

    # Load encoder
    start_time = time.time()
    encoder = get_encoder(encoder_name, checkpoint_dir, device)
    load_time = time.time() - start_time
    print(f"Encoder loaded in {load_time:.2f}s")
    print(f"Embedding dimension: {encoder.embed_dim}")

    # Load dataset
    dataset = ESC50Dataset(esc50_root)
    print(f"Dataset: {len(dataset)} samples, {dataset.num_classes} classes")

    # Run 5-fold CV
    start_time = time.time()
    results = evaluate_5fold_cv(
        encoder,
        dataset,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    eval_time = time.time() - start_time

    # Add metadata
    results["encoder"] = encoder_name
    results["embed_dim"] = encoder.embed_dim
    results["load_time_s"] = load_time
    results["eval_time_s"] = eval_time
    results["epochs"] = epochs
    results["lr"] = lr
    results["batch_size"] = batch_size

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{encoder_name}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark audio encoders on ESC-50")
    parser.add_argument(
        "--encoder",
        type=str,
        choices=ENCODER_NAMES + ["all", "all-part-c"],
        default="all",
        help="Encoder to benchmark. 'all' runs original 3, 'all-part-c' runs new encoders",
    )
    parser.add_argument(
        "--esc50-root",
        type=Path,
        default=Path(__file__).parents[4] / "vendor" / "ESC-50",
        help="Path to ESC-50 dataset",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.home() / "scratch" / "birdclef-2026" / "checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "scratch" / "birdclef-2026" / "outputs" / "esc50-benchmark",
        help="Directory to save results",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if results exist",
    )
    args = parser.parse_args()

    # Determine which encoders to run
    if args.encoder == "all":
        encoders = ["wav2vec2", "beats", "eat"]
    elif args.encoder == "all-part-c":
        encoders = ["ast", "ssast", "ssamba-tiny", "ssamba-small", "ssamba-base"]
    else:
        encoders = [args.encoder]

    # Run benchmarks
    all_results = {}
    for encoder_name in encoders:
        try:
            results = benchmark_encoder(
                encoder_name=encoder_name,
                esc50_root=args.esc50_root,
                checkpoint_dir=args.checkpoint_dir,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                skip_existing=not args.force,
            )
            all_results[encoder_name] = results
        except Exception as e:
            print(f"Error benchmarking {encoder_name}: {e}")
            all_results[encoder_name] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Encoder':<16} {'Mean Acc':<12} {'Embed Dim':<12} {'Time (s)':<12}")
    print("-" * 52)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<16} {'ERROR':<12}")
        else:
            print(
                f"{name:<16} {res['mean_accuracy']:.4f}       "
                f"{res['embed_dim']:<12} {res['eval_time_s']:.1f}"
            )

    # Save combined results
    combined_path = args.output_dir / "combined_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
