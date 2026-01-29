"""Verify SSAMBA-HF vs original model with multiple inputs.

Extends the existing single-tensor verification to multiple random seeds,
real ESC-50 spectrograms, and edge cases. Requires both original SSAMBA
(via mamba-ssm CUDA kernels) and the HF port.

Usage:
    python verify_multi_input.py \
        --checkpoint /path/to/ssamba_small_400.pth \
        --device cuda \
        --esc50-dir /path/to/ESC-50-master/audio \
        --output /path/to/multi_input_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute comparison metrics between two tensors."""
    a = a.float().flatten()
    b = b.float().flatten()

    diff = (a - b).abs()
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

    return {
        "cosine_similarity": cos_sim,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "a_norm": float(a.norm()),
        "b_norm": float(b.norm()),
    }


def load_both_models(checkpoint_path: str, device: str):
    """Load both original and HF models."""
    # Load HF model
    from ssamba_hf import load_from_ssamba_checkpoint

    hf_model, hf_config = load_from_ssamba_checkpoint(
        checkpoint_path, device=device, verbose=True, use_cuda_kernels=False,
    )
    hf_model.eval()

    # Load original model
    from extract_original_embeddings import load_original_model

    original_model = load_original_model(checkpoint_path, device=device)
    original_model.eval()

    return hf_model, original_model, hf_config


def extract_original_embeddings(model, spectrogram: torch.Tensor, device: str) -> torch.Tensor:
    """Extract embeddings from the original SSAMBA model."""
    from extract_original_embeddings import extract_embeddings
    return extract_embeddings(model, spectrogram, device)


def extract_hf_embeddings(model, spectrogram: torch.Tensor, device: str) -> torch.Tensor:
    """Extract embeddings from the HF SSAMBA model."""
    with torch.no_grad():
        return model.get_embeddings(spectrogram.to(device))


def create_random_spectrogram(seed: int, tdim: int = 1024, fdim: int = 128) -> torch.Tensor:
    """Create a deterministic random spectrogram."""
    rng = np.random.default_rng(seed)
    spec = rng.standard_normal((1, tdim, fdim)).astype(np.float32)
    return torch.from_numpy(spec)


def create_zeros_spectrogram(tdim: int = 1024, fdim: int = 128) -> torch.Tensor:
    """Create an all-zeros spectrogram (edge case)."""
    return torch.zeros(1, tdim, fdim)


def create_short_padded_spectrogram(
    n_frames: int = 100, tdim: int = 1024, fdim: int = 128, seed: int = 99
) -> torch.Tensor:
    """Create a heavily-padded short audio spectrogram (edge case)."""
    rng = np.random.default_rng(seed)
    spec = rng.standard_normal((1, n_frames, fdim)).astype(np.float32)
    spec = np.pad(spec, ((0, 0), (0, tdim - n_frames), (0, 0)), mode="constant")
    return torch.from_numpy(spec)


def load_esc50_spectrogram(
    wav_path: str, target_tdim: int = 1024, target_fdim: int = 128
) -> torch.Tensor:
    """Load and preprocess a real ESC-50 audio file."""
    waveform, sr = torchaudio.load(wav_path)

    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Mean subtraction
    waveform = waveform - waveform.mean()

    # Compute fbank
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=target_fdim,
        dither=0.0,
        frame_shift=10,
    )

    # Pad/truncate
    n_frames = fbank.shape[0]
    if n_frames < target_tdim:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, target_tdim - n_frames))
    elif n_frames > target_tdim:
        fbank = fbank[:target_tdim, :]

    # Normalize
    norm_mean = -4.2677393
    norm_std = 4.5689974
    fbank = (fbank - norm_mean) / (norm_std * 2)

    return fbank.unsqueeze(0)  # (1, T, F)


def main():
    parser = argparse.ArgumentParser(description="Multi-input SSAMBA verification")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--esc50-dir", type=str, default=None,
                        help="Path to ESC-50 audio directory")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-INPUT SSAMBA VERIFICATION")
    print("=" * 70)

    # Load both models
    print("\nLoading models...")
    hf_model, original_model, hf_config = load_both_models(
        args.checkpoint, args.device
    )

    # Build test cases
    test_cases = {}

    # Multiple random seeds
    for seed in [0, 1, 2, 3, 42]:
        test_cases[f"random_seed_{seed}"] = create_random_spectrogram(seed)

    # Edge cases
    test_cases["all_zeros"] = create_zeros_spectrogram()
    test_cases["short_padded_100frames"] = create_short_padded_spectrogram(n_frames=100)
    test_cases["short_padded_50frames"] = create_short_padded_spectrogram(n_frames=50, seed=77)

    # Real ESC-50 files (if available)
    if args.esc50_dir:
        esc50_path = Path(args.esc50_dir)
        wav_files = sorted(esc50_path.glob("*.wav"))
        if wav_files:
            # Pick 5 files from different parts of the dataset
            indices = np.linspace(0, len(wav_files) - 1, min(5, len(wav_files)), dtype=int)
            for idx in indices:
                wav_file = wav_files[idx]
                try:
                    spec = load_esc50_spectrogram(str(wav_file))
                    test_cases[f"esc50_{wav_file.stem}"] = spec
                except Exception as e:
                    print(f"  WARNING: Failed to load {wav_file.name}: {e}")
        else:
            print(f"  WARNING: No .wav files found in {args.esc50_dir}")

    # Run comparisons
    results = {}
    any_fail = False

    for name, spectrogram in test_cases.items():
        print(f"\n--- {name} (shape: {spectrogram.shape}) ---")

        try:
            original_emb = extract_original_embeddings(
                original_model, spectrogram, args.device
            )
            hf_emb = extract_hf_embeddings(hf_model, spectrogram, args.device)

            # Move to CPU for comparison
            original_emb = original_emb.cpu()
            hf_emb = hf_emb.cpu()

            metrics = compute_metrics(original_emb, hf_emb)
            passed = metrics["cosine_similarity"] > 0.999
            status = "PASS" if passed else "FAIL"

            if not passed:
                any_fail = True

            print(f"  Cosine sim: {metrics['cosine_similarity']:.6f}")
            print(f"  Max abs diff: {metrics['max_abs_diff']:.2e}")
            print(f"  Mean abs diff: {metrics['mean_abs_diff']:.2e}")
            print(f"  Norms: original={metrics['a_norm']:.4f}, hf={metrics['b_norm']:.4f}")
            print(f"  [{status}]")

            results[name] = {"metrics": metrics, "passed": passed}

        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e), "passed": False}
            any_fail = True

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-INPUT VERIFICATION SUMMARY")
    print("=" * 70)

    num_pass = sum(1 for r in results.values() if r.get("passed", False))
    num_total = len(results)
    print(f"\n  Passed: {num_pass}/{num_total}")

    for name, result in results.items():
        if "error" in result:
            print(f"  {name}: ERROR - {result['error']}")
        else:
            status = "PASS" if result["passed"] else "FAIL"
            cos_sim = result["metrics"]["cosine_similarity"]
            print(f"  {name}: cos_sim={cos_sim:.6f} [{status}]")

    overall = "PASS" if not any_fail else "FAIL"
    print(f"\n  OVERALL: {overall}")

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Report saved to {args.output}")

    return 0 if not any_fail else 1


if __name__ == "__main__":
    sys.exit(main())
