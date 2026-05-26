#!/usr/bin/env python3
"""Diagnose the training/inference mel-pipeline mismatch (Phase D § D.2).

Feeds the SAME 5s waveform through three pipelines and compares element-wise:

1. **TUCKER** (reference) — verbatim port of Tucker's `MelSpecTransform`
   (vendored at `bc2026-distilled-sed.py` lines 400–407) + per-sample
   `(mel - mean) / (std + eps)` (line 949).
   Pipeline: `torchaudio.MelSpectrogram → AmplitudeToDB(80) → z-score`.

2. **OURS-TRAINING** — exp 032 training pipeline.
   `NNAudioMelSpectrogram(norm_type="amplitude_to_db")` + module's
   `per_sample_mel_norm=True`.
   Pipeline: `nnAudio mel → 10*log10 → top_db clamp → z-score → min-max [0,1]
              → z-score`.

3. **OURS-KERNEL** — the Kaggle inference path.
   `submission.py:compute_mel_tucker`. Should match TUCKER reference
   (it was written to).

Expected outcomes:
- TUCKER vs OURS-KERNEL: ≤ 1e-5 element-wise (both faithful to Tucker;
  any drift is fp/torch-vs-numpy noise).
- TUCKER vs OURS-TRAINING: LARGE diff (this is the bug — training did 3
  normalizations, Tucker does 1).
- OURS-TRAINING vs OURS-KERNEL: ALSO large diff, equivalent magnitude.

If both hypothesis checks fire, the mel mismatch is confirmed as the
primary cause of the LB collapse.

Usage:
    python user/murilo/experiments/032-tucker-distilled-sed/scripts/diag_mel_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

EXP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_ROOT.parents[3]
SRC_ROOT = REPO_ROOT / "user/murilo/src"
sys.path.insert(0, str(SRC_ROOT))

# ----------------------------------------------------------------------
# Tucker's mel pipeline — verbatim port from vendored notebook
# ----------------------------------------------------------------------

# Constants from `bc2026-distilled-sed.py` lines 127-135 + 158
SR = 32000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 256
FMIN = 20
FMAX = 16000
TOP_DB = 80


class TuckerMelSpecTransform(torch.nn.Module):
    """Verbatim port of `bc2026-distilled-sed.py` lines 400-407."""

    def __init__(self):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=2.0,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=TOP_DB)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.db_transform(self.mel_spec(waveform))


def tucker_per_sample_norm(mel: torch.Tensor) -> torch.Tensor:
    """Verbatim port of training loop lines 945-950."""
    mel = mel.clone()
    B = mel.size(0)
    for i in range(B):
        mel[i] = (mel[i] - mel[i].mean()) / (mel[i].std() + 1e-6)
    return mel


def pipeline_tucker(wav_1d: np.ndarray) -> np.ndarray:
    """Run the full Tucker training-side mel pipeline on a 1D waveform."""
    wav = torch.from_numpy(wav_1d).float().unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    transform = TuckerMelSpecTransform()
    mel = transform(wav)  # (1, 1, n_mels, T)
    mel = tucker_per_sample_norm(mel)
    return mel.squeeze().numpy()  # (n_mels, T)


# ----------------------------------------------------------------------
# Our training pipeline — NNAudio amplitude_to_db + module per-sample norm
# ----------------------------------------------------------------------


def pipeline_ours_training(wav_1d: np.ndarray) -> np.ndarray:
    """Run our exp 032 training mel pipeline on a 1D waveform.

    Mirrors `train.py` mel construction + `DistilledSEDFinetuneModule._compute_mel`
    when `per_sample_mel_norm=True`.
    """
    from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

    # Match `configs/distilled_sed_control.yaml` mel block
    target_frames = int(5.0 * SR / HOP_LENGTH)  # 312 (Tucker math: 160000//512 = 312)
    transform = NNAudioMelSpectrogram(
        sample_rate=SR,
        n_mels=N_MELS,
        target_frames=target_frames,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmax=FMAX,
        norm_type="amplitude_to_db",
        top_db=float(TOP_DB),
    )
    wav = torch.from_numpy(wav_1d).float().unsqueeze(0)  # (1, samples)
    mel = transform(wav)  # (1, 1, n_mels, T) — NNAudio's transform shape
    # Module's _compute_mel per-sample norm (lines 204-208 of distilled_sed_finetune.py)
    mean = mel.mean(dim=(1, 2, 3), keepdim=True)
    std = mel.std(dim=(1, 2, 3), keepdim=True)
    mel = (mel - mean) / (std + 1e-6)
    return mel.squeeze().detach().numpy()  # (n_mels, T)


# ----------------------------------------------------------------------
# Kernel pipeline — submission.py:compute_mel_tucker (Tucker-faithful)
# ----------------------------------------------------------------------


def pipeline_ours_kernel(wav_1d: np.ndarray) -> np.ndarray:
    """Mirror `compute_mel_tucker` from submission.py."""
    wave = torch.from_numpy(wav_1d).float().unsqueeze(0)  # (1, samples)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=2.0,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=float(TOP_DB))
    mel = mel_transform(wave)
    mel = db_transform(mel)
    mean = mel.mean()
    std = mel.std().clamp(min=1e-6)
    mel = (mel - mean) / std
    return mel.squeeze().numpy()  # (n_mels, T)


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


def compare(name_a: str, name_b: str, a: np.ndarray, b: np.ndarray) -> None:
    # Crop both to the same frame count (NNAudio uses target_frames; torchaudio
    # output depends on n_fft padding behaviour — usually one extra frame).
    min_T = min(a.shape[-1], b.shape[-1])
    a_c = a[..., :min_T]
    b_c = b[..., :min_T]
    diff = a_c - b_c
    print(f"\n{'=' * 70}")
    print(f"{name_a}  vs  {name_b}")
    print(f"{'=' * 70}")
    print(f"  shape {name_a}: {a.shape}, {name_b}: {b.shape}  (cropped to T={min_T})")
    print(
        f"  {name_a} range: [{a.min():.4f}, {a.max():.4f}]  mean={a.mean():.4f}  std={a.std():.4f}"
    )
    print(
        f"  {name_b} range: [{b.min():.4f}, {b.max():.4f}]  mean={b.mean():.4f}  std={b.std():.4f}"
    )
    print(
        f"  diff stats: max|d|={np.abs(diff).max():.4e}  mean|d|={np.abs(diff).mean():.4e}  rms_d={np.sqrt((diff**2).mean()):.4e}"
    )
    print(f"  pearson r: {np.corrcoef(a_c.ravel(), b_c.ravel())[0, 1]:.6f}")


def main() -> None:
    # Load a deterministic 5s slice of a labeled soundscape
    audio_path = (
        "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/"
        "train_soundscapes/BC2026_Train_0001_S08_20250606_030007.ogg"
    )
    print(f"Test audio: {audio_path}")

    audio, sr = sf.read(audio_path, dtype="float32", start=0, frames=int(SR * 5))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = audio[: int(SR * 5)].astype(np.float32)
    print(f"Loaded {len(audio)} samples @ {SR} Hz = {len(audio) / SR:.2f}s")

    print("\nRunning all three pipelines on the same waveform...")
    mel_tucker = pipeline_tucker(audio)
    mel_ours_train = pipeline_ours_training(audio)
    mel_ours_kernel = pipeline_ours_kernel(audio)

    compare("TUCKER (ref)", "OURS-KERNEL", mel_tucker, mel_ours_kernel)
    compare("TUCKER (ref)", "OURS-TRAINING", mel_tucker, mel_ours_train)
    compare("OURS-TRAINING", "OURS-KERNEL", mel_ours_train, mel_ours_kernel)

    print(f"\n{'=' * 70}")
    print("VERDICT")
    print(f"{'=' * 70}")

    def status(diff_max: float, tol: float) -> str:
        return "✅ MATCH" if diff_max < tol else "❌ MISMATCH"

    d_tk = np.abs(mel_tucker[..., :310] - mel_ours_kernel[..., :310]).max()
    d_tt = np.abs(mel_tucker[..., :310] - mel_ours_train[..., :310]).max()
    d_kt = np.abs(mel_ours_kernel[..., :310] - mel_ours_train[..., :310]).max()
    print(f"  TUCKER vs OURS-KERNEL:    max|d| = {d_tk:.4e}  {status(d_tk, 1e-3)}")
    print(f"  TUCKER vs OURS-TRAINING:  max|d| = {d_tt:.4e}  {status(d_tt, 1e-3)}")
    print(f"  OURS-KERNEL vs TRAINING:  max|d| = {d_kt:.4e}  {status(d_kt, 1e-3)}")
    if d_tt > 1e-3 and d_kt > 1e-3 and d_tk < 1e-3:
        print("\n💥 CONFIRMED: training/inference mel-pipeline mismatch is the bug.")
        print("   Inference matches Tucker, but TRAINING does NOT.")
        print("   Model was trained on a different distribution than it sees at inference.")
    elif d_tt < 1e-3 and d_kt < 1e-3:
        print("\n🤔 ALL THREE MATCH. The mel pipeline is fine; look elsewhere.")
    else:
        print("\n⚠️  AMBIGUOUS. Inspect intermediate stats above.")


if __name__ == "__main__":
    main()
