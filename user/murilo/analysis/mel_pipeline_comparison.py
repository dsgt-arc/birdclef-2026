"""Mel-pipeline equivalence diagnostic for exp 031 Phase 0.

Numerically compares our v5j-style mel (NNAudio + amplitude_to_db + min-max) against
Tucker's bc2026-distilled-sed mel (torchaudio + AmplitudeToDB + per-sample standardize)
on a fixed batch of 100 training waveforms.

Decision gate (proposal § Procedure 0a):
- max abs diff < 1.0 dB AND same frame count → keep NNAudio, proceed
- max abs diff > 5.0 dB OR different frame count → switch student mel to torchaudio
- in between → proceed with NNAudio, flag as leading hypothesis if Phase 1 underperforms

Usage:
    cd user/murilo
    source ~/clef/birdclef-2026/.venv/bin/activate  # or interactive venv per CLAUDE.md
    python analysis/mel_pipeline_comparison.py

Output:
- Console summary with all numerical metrics
- experiments/031-perch-kd-distillation/phase0_mel_equivalence.md (decision doc)
- experiments/031-perch-kd-distillation/phase0_mel_equivalence_samples.png (sanity viz)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy.signal import resample_poly

# Path constants
REPO_ROOT = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026")
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
EXP_DIR = REPO_ROOT / "user/murilo/experiments/031-perch-kd-distillation"

# Audio constants
SR = 32_000
DURATION_SEC = 5
N_SAMPLES = SR * DURATION_SEC  # 160_000

# v5j mel config (from experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml)
V5J_N_FFT = 2048
V5J_HOP = 313
V5J_N_MELS = 256
V5J_FMIN = 0
V5J_FMAX = SR // 2  # Nyquist (16000)
V5J_TOP_DB = 80.0

# Tucker mel config (from vendor/.../bc2026-distilled-sed.ipynb cell 4)
TUCKER_N_FFT = 2048
TUCKER_HOP = 512
TUCKER_N_MELS = 256
TUCKER_FMIN = 20
TUCKER_FMAX = 16_000
TUCKER_TOP_DB = 80


# ---------------------------------------------------------------------------
# Mel pipelines
# ---------------------------------------------------------------------------


def v5j_mel(waveforms: torch.Tensor) -> torch.Tensor:
    """Reproduce v5j's NNAudio mel + amplitude_to_db + per-sample standardize + min-max.

    Mirrors birdclef.preprocessing.nnaudio_mel.NNAudioMelSpectrogram with
    norm_type="amplitude_to_db" and v5j hyperparameters.

    Args:
        waveforms: (B, n_samples) float32

    Returns:
        (B, 1, n_mels, T) mel in [0, 1]
    """
    from nnAudio.features import MelSpectrogram as NNMel

    # Match NNAudioMelSpectrogram defaults: htk=True, norm=None, win_length=n_fft
    mel = NNMel(
        sr=SR,
        n_fft=V5J_N_FFT,
        win_length=V5J_N_FFT,
        hop_length=V5J_HOP,
        n_mels=V5J_N_MELS,
        fmin=V5J_FMIN,
        fmax=V5J_FMAX,
        htk=True,
        norm=None,
        trainable_mel=False,
        trainable_STFT=False,
    )(waveforms)  # (B, n_mels, T) linear power

    # amplitude_to_db: 10 * log10(mel + 1e-10), clipped to top_db
    mel = 10.0 * torch.log10(mel + 1e-10)
    mel_max = mel.amax(dim=(-2, -1), keepdim=True)
    mel = torch.clamp(mel, min=mel_max - V5J_TOP_DB)

    # Per-sample standardize
    mean = mel.mean(dim=(-2, -1), keepdim=True)
    std = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    mel = (mel - mean) / std

    # Per-sample min-max scale to [0, 1]
    mn = mel.amin(dim=(-2, -1), keepdim=True)
    mx = mel.amax(dim=(-2, -1), keepdim=True)
    mel = (mel - mn) / (mx - mn + 1e-6)

    return mel.unsqueeze(1)  # (B, 1, n_mels, T)


def tucker_mel(waveforms: torch.Tensor) -> torch.Tensor:
    """Reproduce Tucker's torchaudio mel + AmplitudeToDB + per-sample standardize.

    Mirrors MelSpecTransform from cell 10 + the per-sample standardize loop
    from cell 14.

    Args:
        waveforms: (B, n_samples) float32

    Returns:
        (B, 1, n_mels, T) mel — standardized only, NO min-max
    """
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=TUCKER_N_FFT,
        hop_length=TUCKER_HOP,
        n_mels=TUCKER_N_MELS,
        f_min=TUCKER_FMIN,
        f_max=TUCKER_FMAX,
        power=2.0,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=TUCKER_TOP_DB)

    mel = db_transform(mel_spec(waveforms))  # (B, n_mels, T)

    # Per-sample standardize (Tucker's loop, vectorized)
    mean = mel.mean(dim=(-2, -1), keepdim=True)
    std = mel.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    mel = (mel - mean) / std

    return mel.unsqueeze(1)  # (B, 1, n_mels, T)


def tucker_mel_pre_norm(waveforms: torch.Tensor) -> torch.Tensor:
    """Tucker's pipeline up to AmplitudeToDB only (no normalization).

    For decomposing the diff into "STFT/lib numerics" vs "post-norm" sources.
    """
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=TUCKER_N_FFT,
        hop_length=TUCKER_HOP,
        n_mels=TUCKER_N_MELS,
        f_min=TUCKER_FMIN,
        f_max=TUCKER_FMAX,
        power=2.0,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=TUCKER_TOP_DB)
    return db_transform(mel_spec(waveforms))  # (B, n_mels, T)


def v5j_mel_pre_norm(waveforms: torch.Tensor) -> torch.Tensor:
    """v5j NNAudio mel up to amplitude_to_db only (no normalize/min-max).

    For decomposing the diff into "STFT/lib numerics" vs "post-norm" sources.
    Uses Tucker's hop=512 to make the comparison apples-to-apples on STFT
    library — see also v5j_mel_native_hop for the hop=313 case.
    """
    from nnAudio.features import MelSpectrogram as NNMel

    mel = NNMel(
        sr=SR,
        n_fft=V5J_N_FFT,
        win_length=V5J_N_FFT,
        hop_length=TUCKER_HOP,  # ⭐ match Tucker's hop for STFT-only comparison
        n_mels=V5J_N_MELS,
        fmin=TUCKER_FMIN,  # also match fmin/fmax to isolate STFT
        fmax=TUCKER_FMAX,
        htk=True,
        norm=None,
        trainable_mel=False,
        trainable_STFT=False,
    )(waveforms)
    mel = 10.0 * torch.log10(mel + 1e-10)
    mel_max = mel.amax(dim=(-2, -1), keepdim=True)
    mel = torch.clamp(mel, min=mel_max - V5J_TOP_DB)
    return mel  # (B, n_mels, T)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_audio_32k(path: Path) -> np.ndarray:
    """Decode + force mono + resample to 32 kHz."""
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        g = math.gcd(int(file_sr), SR)
        audio = resample_poly(audio, SR // g, int(file_sr) // g).astype(np.float32)
    return audio


def extract_5s(audio: np.ndarray, seed: int = 0) -> np.ndarray:
    """Deterministic 5s crop (random offset based on seed)."""
    if len(audio) <= N_SAMPLES:
        return np.pad(audio, (N_SAMPLES - len(audio), 0))
    rng = np.random.default_rng(seed)
    start = rng.integers(0, len(audio) - N_SAMPLES + 1)
    return audio[start : start + N_SAMPLES]


def load_fixed_batch(n_samples: int = 100, seed: int = 42) -> torch.Tensor:
    """Load N deterministically-chosen training waveforms cropped to 5s.

    Uses train.csv to pick filenames so the choice is reproducible across runs.
    """
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(train_df), size=n_samples, replace=False)

    waveforms = []
    paths_used = []
    for i, row_idx in enumerate(pick_idx):
        row = train_df.iloc[int(row_idx)]
        path = DATA_DIR / "train_audio" / row["filename"]
        try:
            wav = load_audio_32k(path)
            crop = extract_5s(wav, seed=i)
            waveforms.append(crop)
            paths_used.append(str(path))
        except Exception as e:
            print(f"  skip {path}: {e}")
    print(f"Loaded {len(waveforms)} waveforms")
    return torch.from_numpy(np.stack(waveforms)).float(), paths_used


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def compare(name: str, mel_a: torch.Tensor, mel_b: torch.Tensor):
    """Numerically compare two mel tensors. Returns dict of metrics."""
    print(f"\n--- {name} ---")
    print(f"  shapes: A={tuple(mel_a.shape)}  B={tuple(mel_b.shape)}")

    if mel_a.shape != mel_b.shape:
        print("  ⚠️  shape mismatch — interpolating B to A's time axis for delta metrics")
        # Resize B to match A's time dim along the last axis
        mel_b_resized = torch.nn.functional.interpolate(
            mel_b.unsqueeze(0) if mel_b.ndim == 3 else mel_b,
            size=(mel_b.shape[-2], mel_a.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        if mel_b_resized.ndim == 4 and mel_b.ndim == 3:
            mel_b_resized = mel_b_resized.squeeze(0)
        mel_b_cmp = mel_b_resized
    else:
        mel_b_cmp = mel_b

    diff = (mel_a - mel_b_cmp).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    # Per-sample cosine similarity (flatten to 1D)
    a_flat = mel_a.flatten(start_dim=1)
    b_flat = mel_b_cmp.flatten(start_dim=1)
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=1)
    cos_mean = cos.mean().item()
    cos_min = cos.min().item()

    print(f"  per-pixel max abs diff:  {max_abs:.4f}")
    print(f"  per-pixel mean abs diff: {mean_abs:.4f}")
    print(f"  cosine sim (mean):       {cos_mean:.4f}")
    print(f"  cosine sim (min):        {cos_min:.4f}")

    return {
        "name": name,
        "shape_a": tuple(mel_a.shape),
        "shape_b": tuple(mel_b.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "cos_sim_mean": cos_mean,
        "cos_sim_min": cos_min,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Phase 0.0 — Mel pipeline equivalence diagnostic")
    print("=" * 70)
    print(f"\nLoading 100 fixed training waveforms from {DATA_DIR}")
    waveforms, paths = load_fixed_batch(n_samples=100, seed=42)

    print(
        f"\nv5j config:    n_fft={V5J_N_FFT} hop={V5J_HOP} n_mels={V5J_N_MELS} "
        f"fmin={V5J_FMIN} fmax={V5J_FMAX} (NNAudio)"
    )
    print(
        f"Tucker config: n_fft={TUCKER_N_FFT} hop={TUCKER_HOP} n_mels={TUCKER_N_MELS} "
        f"fmin={TUCKER_FMIN} fmax={TUCKER_FMAX} (torchaudio)"
    )

    print("\nComputing mels...")
    mel_v5j = v5j_mel(waveforms)
    mel_tucker = tucker_mel(waveforms)
    mel_v5j_prenorm_at_tucker_hop = v5j_mel_pre_norm(waveforms)
    mel_tucker_prenorm = tucker_mel_pre_norm(waveforms)

    results = []

    # Headline: full v5j vs full Tucker
    results.append(
        compare(
            "FULL: v5j (NNAudio + amp2db + std + min-max, hop=313) vs Tucker (torchaudio + amp2db + std, hop=512)",
            mel_v5j.squeeze(1),
            mel_tucker.squeeze(1),
        )
    )

    # Decompose: STFT/lib numerics only (matched hop/fmin/fmax, no normalization)
    results.append(
        compare(
            "STFT-LIB ONLY: NNAudio vs torchaudio at matched config (hop=512, no norm)",
            mel_v5j_prenorm_at_tucker_hop,
            mel_tucker_prenorm,
        )
    )

    # Decision gate
    print("\n" + "=" * 70)
    print("DECISION GATE (proposal § Procedure 0a)")
    print("=" * 70)
    headline = results[0]
    max_abs = headline["max_abs_diff"]
    same_t = headline["shape_a"][-1] == headline["shape_b"][-1]
    if max_abs < 1.0 and same_t:
        verdict = "KEEP NNAUDIO — proceed to Phase 0.1 without refactor"
    elif max_abs > 5.0 or not same_t:
        verdict = "REFACTOR STUDENT MEL — switch to torchaudio + hop=512 (Phase 0.0a tasks)"
    else:
        verdict = "PROCEED WITH NNAUDIO BUT FLAG — log mismatch, leading hypothesis if Phase 1 underperforms"
    print(f"  Headline max abs diff:  {max_abs:.4f}")
    print(
        f"  Frame counts match:     {same_t} (v5j={headline['shape_a'][-1]}, tucker={headline['shape_b'][-1]})"
    )
    print(f"  Verdict: {verdict}")

    # Write decision doc
    out_md = EXP_DIR / "phase0_mel_equivalence.md"
    rows = []
    for r in results:
        rows.append(
            f"| {r['name']} | {r['shape_a']} | {r['shape_b']} | "
            f"{r['max_abs_diff']:.4f} | {r['mean_abs_diff']:.4f} | "
            f"{r['cos_sim_mean']:.4f} | {r['cos_sim_min']:.4f} |"
        )
    md = f"""# Phase 0.0 — Mel Pipeline Equivalence Diagnostic

**Date**: 2026-05-09
**Script**: `user/murilo/analysis/mel_pipeline_comparison.py`
**Input**: 100 deterministically-chosen training waveforms (5s @ 32kHz, seed=42)

## Headline

- **v5j config**: NNAudio, n_fft={V5J_N_FFT}, hop={V5J_HOP}, n_mels={V5J_N_MELS},
  fmin={V5J_FMIN}, fmax={V5J_FMAX}, AmpToDB(top_db={V5J_TOP_DB}), per-sample standardize, min-max [0, 1]
- **Tucker config**: torchaudio, n_fft={TUCKER_N_FFT}, hop={TUCKER_HOP}, n_mels={TUCKER_N_MELS},
  fmin={TUCKER_FMIN}, fmax={TUCKER_FMAX}, AmpToDB(top_db={TUCKER_TOP_DB}), per-sample standardize (NO min-max)

## Numerical comparison

| Comparison | Shape A | Shape B | Max abs diff | Mean abs diff | Cos sim (mean) | Cos sim (min) |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Decision

**Verdict**: {verdict}

Decision-gate criteria from `proposal.md` § Procedure 0a:
- max abs diff < 1.0 dB AND same frame count → keep NNAudio
- max abs diff > 5.0 dB OR different frame count → refactor student mel to torchaudio + hop=512
- otherwise → proceed with NNAudio, flag as leading hypothesis if Phase 1 underperforms

## Notes

- The "FULL" comparison includes both library/hop differences AND post-norm differences
  (Tucker omits the min-max step we apply). It's the actual "what does the backbone see"
  comparison.
- The "STFT-LIB ONLY" comparison fixes hop, fmin, fmax to be identical between the two
  libraries. This isolates pure STFT/filter-bank numerics. If this is small, NNAudio and
  torchaudio agree on STFT and we don't need to switch libraries — only the hop_length
  + post-norm choices remain.
- Frame count for hop=313 is ceil(160000/313)+1 ≈ 512; for hop=512 is ceil(160000/512)+1 ≈ 313.
- Phase 1 operates on the FULL pipeline; the headline (first row) is the decision driver.
"""
    out_md.write_text(md)
    print(f"\nWrote: {out_md}")
    print("\nDone.")


if __name__ == "__main__":
    main()
