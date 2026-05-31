#!/usr/bin/env python3
"""Unified BirdCLEF 2026 Kaggle inference script.

Self-contained Kaggle CPU kernel for BirdCLEF 2026 submission.
Auto-detects which model (EAT, ConvNeXt, ProtoCLR) is available in the
weights directory and runs the appropriate preprocessing pipeline.

Supports models trained on either BirdCLEF 2025 or 2026 data:
  - 2025-trained: 41/206 model outputs map to 2026 columns (rest get 0.0)
  - 2026-trained: 206/206 model outputs map to 2026 columns

Pipeline per soundscape:
  for each 5s window:
    1. Load audio at model-specific sample rate
    2. Compute mel spectrogram (model-specific preprocessing)
    3. ONNX session.run() -> logits
    4. sigmoid -> probs
    5. Map model probs to 234 BirdCLEF-2026 columns via species_mapping.json
    6. Unmapped species get 0.0

Supports three mel types:
  - kaldi_fbank (EAT): torchaudio Kaldi fbank features
  - mel (ConvNeXt): librosa log-mel spectrogram + AudioSet normalization
  - protoclr_mel (ProtoCLR): librosa mel + dB + min-max + ProtoCLR standardization
"""

# ---------------------------------------------------------------------------
# Offline dependency installation (runs on Kaggle before imports)
# Wheels are in a separate Kaggle dataset: birdclef-deps
# ---------------------------------------------------------------------------
import os
import subprocess
import sys

KAGGLE_USER = "murilogustineli"


def _find_input(slug: str) -> str:
    """Resolve Kaggle input path, supporting both old and new directory layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]  # fallback to old-style


DEPS_DIR = _find_input("birdclef-deps")

# Debug: show what we see in /kaggle/input
if os.path.exists("/kaggle/input"):
    print(f"Kaggle input contents: {os.listdir('/kaggle/input')}")
print(f"DEPS_DIR resolved to: {DEPS_DIR} (exists={os.path.exists(DEPS_DIR)})")

if os.path.exists(DEPS_DIR):
    print(f"Installing deps from {DEPS_DIR}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            DEPS_DIR,
            "onnxruntime",
        ]
    )
else:
    print(f"WARNING: deps dir not found at {DEPS_DIR}, onnxruntime may not be available")

# ---------------------------------------------------------------------------
# Imports (after offline install, must use noqa: E402)
# ---------------------------------------------------------------------------
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

# ---------------------------------------------------------------------------
# Kaggle path resolution
# ---------------------------------------------------------------------------
KAGGLE_USER = "murilogustineli"

# Model slug names — the kernel metadata picks one of these as a dataset source
MODEL_SLUGS = [
    # Exp 036: XC-backbone faithful fine-tune (a1-lower-lr). Same XCPretrain graph as
    # exp 034 (routed by mel_config type=xcpretrain_audio_onnx → predict_soundscape_xcpretrain),
    # gentler LR 5e-5 fixes the full-FT overfitting collapse → stable 0.92 plateau offline.
    # Lead with SWA (avg of ep45-49) over the plain last.ckpt. Each kernel attaches only its
    # own weights dataset, so listing order between these two is moot; preferred → first.
    "birdclef-2026-xcfinetune-effv2s-lowlr-swa-weights",
    "birdclef-2026-xcfinetune-effv2s-lowlr-last-weights",
    # Exp 034: pretrained-backbone yoink (Sydorskyi EffNetV2-S). Full-model audio→probs
    # ONNX (nnAudio mel in-graph); routed by mel_config type=xcpretrain_audio_onnx →
    # predict_soundscape_xcpretrain (raw audio in, clipwise probs out).
    # Phase K KD (joint Perch v2 distillation) is the preferred member → listed first.
    "birdclef-2026-xcpretrain-effv2s-kd-joint-weights",
    "birdclef-2026-xcpretrain-effv2s-fullft-weights",
    "birdclef-2026-xcpretrain-effv2s-frozen-weights",
    # Exp 033 Phase 5 Sydorskyi-divergence diagnostic — Test 1 (chunk discard + trim).
    # Same canonical Antoine cell (a2-kd: random init + Uniform sampler + T2 teacher
    # + 25 ep) pointed at v3 PL parquet (Sydorskyi 0.5 chunk-discard + 0.1 per-class trim).
    # Same Tucker ONNX schema as exp 032 distilled-sed-kd / exp 033 v2 cells.
    "birdclef-2026-distilled-sed-pl-a2-kd-v3-weights",
    # Exp 033 own-teacher PL ablations on distilled-sed-kd (10 cells = 5 protocols × 2 teachers).
    # Same Tucker inference path as exp 032 distilled-sed-kd (mel_config["type"]=tucker_mel,
    # output_type=tucker_logits → per-window inference + 5-tap Gaussian smooth in logit
    # space + single sigmoid). PL is training-time only; ONNX schema is identical to
    # distilled-sed-kd. T1=v5j-4fold teacher, T2=distilled-sed-kd-5fold teacher.
    "birdclef-2026-distilled-sed-pl-a4-kd-weights",  # rank 1 by D.5 e2e val/auroc
    "birdclef-2026-distilled-sed-pl-a3-kd-weights",
    "birdclef-2026-distilled-sed-pl-a2-kd-weights",  # canonical Antoine + T2
    "birdclef-2026-distilled-sed-pl-a4-v5j-weights",
    "birdclef-2026-distilled-sed-pl-a3-v5j-weights",
    "birdclef-2026-distilled-sed-pl-a1-kd-weights",
    "birdclef-2026-distilled-sed-pl-a2-v5j-weights",  # canonical Antoine + T1
    "birdclef-2026-distilled-sed-pl-a1-v5j-weights",
    "birdclef-2026-distilled-sed-pl-a5-kd-weights",
    "birdclef-2026-distilled-sed-pl-a5-v5j-weights",
    # Exp 037 Noisy-Student soft-PL round-1 LB picks (ensemble-teacher PL on distilled-sed-kd,
    # best-by-val/auroc epoch, fold 0). Same Tucker ONNX schema as exp 032/033 (training-time
    # PL only). Each kernel attaches exactly one of these → first (only) match wins.
    "birdclef-2026-exp037-r1-a-raw-augf-weights",
    "birdclef-2026-exp037-r1-b-raw-augf-weights",
    "birdclef-2026-exp037-r1-c-raw-augf-weights",
    "birdclef-2026-exp037-r1-a-raw-augt-softauc-weights",
    "birdclef-2026-exp037-r1-c-raw-augt-softauc-weights",
    # Exp 032 Phase 2: 5-fold KD ensemble + HGNet backbone ablation
    "birdclef-2026-distilled-sed-kd-5fold-weights",
    "birdclef-2026-distilled-sed-kd-hgnet-weights",
    # Exp 032 Phase 2.A individual-fold diagnostics (folds 1-4; fold 0 is `distilled-sed-kd`).
    # Single-fold ckpts; route through `predict_soundscape_tucker` (single-model path)
    # via mel_config["type"] == "tucker_mel" + "output_type" == "tucker_logits".
    "birdclef-2026-distilled-sed-kd-fold1-weights",
    "birdclef-2026-distilled-sed-kd-fold2-weights",
    "birdclef-2026-distilled-sed-kd-fold3-weights",
    "birdclef-2026-distilled-sed-kd-fold4-weights",
    # Exp 032 Phase D: single-fold KD + control (Tucker-faithful mel)
    # EffNet-B0 NS-JFT + AttHeadLogitSpace + Perch v2 KD + 25 ep cosine.
    # ONNX outputs pre-blended LOGITS; kernel does per-window inference →
    # 5-tap Gaussian smooth in LOGIT space → single sigmoid at end.
    # Routed by mel_config["type"] == "tucker_mel" + "output_type" == "tucker_logits".
    # 5-fold variant uses `predict_soundscape_tucker_ensemble` (logit-space fold mean).
    "birdclef-2026-distilled-sed-kd-weights",
    "birdclef-2026-distilled-sed-control-weights",
    # Codebase regression diagnostic: v5j-era code (commit e420633) retrained
    # v5j_control on today's env. Tests whether HEAD codebase IS the cause of
    # the 0.906→0.897 LB regression. Offline val/cmap matches v5j-original
    # exactly; LB confirms.
    "birdclef-2026-hgnet-b0-sed-v5j-eracode-weights",
    # Exp 031 best-vs-last diagnostic: ep 19 weights from save_top_k=-1 retrains.
    # control slug drops "-2026-" to fit Kaggle's 50-char dataset slug limit.
    "birdclef-2026-hgnet-b0-sed-v5j-kd-last-weights",
    "birdclef-hgnet-b0-sed-v5j-control-last-weights",
    # Exp 031: Perch v2 knowledge distillation on v5j (Tucker recipe, Phase 1)
    "birdclef-2026-hgnet-b0-sed-v5j-kd-weights",
    # Exp 030 V3 (seed=43) diagnostic retrain — disambiguates the seed=42 EffNet ×
    # Sydorskyi +0.008 LB asymmetry from seed noise. Slugs drop "-2026-" prefix to
    # fit Kaggle's 50-char dataset slug limit.
    "birdclef-effnetv2-b0-sed-sydorskyi-seed43-weights",
    "birdclef-effnetv2-b0-sed-control-seed43-weights",
    "birdclef-hgnet-b0-sed-v5j-sydorskyi-seed43-weights",
    "birdclef-hgnet-b0-sed-v5j-control-seed43-weights",
    # Diagnostic: re-export of original v5j ckpt (LB 0.906) on today's submission.py
    # to separate training-pipeline drift from inference-pipeline drift.
    "birdclef-2026-hgnet-b0-sed-v5j-resubmit-weights",
    # Exp 030: Sydorskyi-faithful MixUp 2x2 (HGNet × EffNetV2-B0 × control × sydorskyi)
    "birdclef-2026-effnetv2-b0-sed-sydorskyi-weights",
    "birdclef-2026-effnetv2-b0-sed-control-weights",
    "birdclef-2026-hgnet-b0-sed-v5j-sydorskyi-weights",
    "birdclef-2026-hgnet-b0-sed-v5j-control-weights",
    # Exp 029 path A retry: 2025-pretrained → 2026 split-LR fine-tune
    "birdclef-2026-hgnet-b0-sed-v5j-ft-splitlr-weights",
    # Exp 029: 2025-pretrained → 2026 fine-tuned (Ali Ozan path)
    "birdclef-2026-hgnet-b0-sed-v5j-finetune-weights",
    # Exp 027 Phase 4: Salman-derived changes
    "birdclef-2026-hgnet-b0-sed-v11b-exp027-weights",
    # Exp 027: v5j with 10s overlapping window inference
    "birdclef-2026-hgnet-b0-sed-v5j-10s-weights",
    # Exp 027: additive MixUp ablations
    "birdclef-2026-hgnet-b0-sed-v10b-exp027-weights",
    "birdclef-2026-hgnet-b0-sed-v10a-exp027-weights",
    # Exp 026 Phase 3: AttHead + window length ablation
    "birdclef-2026-hgnet-b0-sed-v9c-20s-att-weights",
    "birdclef-2026-hgnet-b0-sed-v9b-15s-att-weights",
    "birdclef-2026-hgnet-b0-sed-v9a-10s-att-weights",
    # Exp 026: v5j + dual validation (soundscape checkpoint selection)
    "birdclef-2026-hgnet-b0-sed-v5j-dualval-weights",
    # Exp 025 Phase C: v3-waveform-mix (Mario-style PL waveform mixing)
    "birdclef-2026-hgnet-b0-sed-v3-waveform-mix-weights",
    # Exp 025: v5j + pseudo-label training (round 1)
    "birdclef-2026-hgnet-b0-sed-v5j-pl-v1-weights",
    # Exp 022: PCEN normalization + pipeline verification
    "birdclef-2026-hgnet-b0-sed-v5j-repro-weights",
    "birdclef-2026-hgnet-b0-sed-v10a-weights",
    "birdclef-2026-hgnet-b0-sed-v10b-weights",
    "birdclef-2026-hgnet-b0-sed-v10c-weights",
    # Exp 021 Phase G: manifold mixup
    "birdclef-2026-hgnet-b0-sed-v6l-weights",
    # Exp 021 Phase F: competition-standard AttHead + blend ablations
    "birdclef-2026-hgnet-b0-atthead-v6k-weights",
    "birdclef-2026-hgnet-b0-sed-v5j-fmax-weights",
    "birdclef-2026-hgnet-b0-sed-v5j-clip-weights",
    # Exp 021 Phase E: LSEHead multi-context
    "birdclef-2026-hgnet-b0-mc-lse-sed-v8b-weights",
    "birdclef-2026-hgnet-b0-mc-lse-sed-v8a-weights",
    # Exp 021 Phase B: waveform MixUp
    "birdclef-2026-hgnet-b0-sed-v6b-weights",
    # Exp 021 Phase C: AttHead + backbone scaling
    "birdclef-2026-hgnet-b2-sed-v6h-weights",
    "birdclef-2026-effnet-b0-ns-sed-v6j-weights",
    # Exp 021 Phase A: v5j + site/hour priors (pp2)
    "birdclef-2026-hgnet-b0-sed-v5j-4fold-pp2-weights",
    "birdclef-2026-hgnet-b0-sed-v5j-pp2-weights",
    # SED ablations, 234 classes (exp 019)
    "birdclef-2026-effnet-b0-sed-v5a-weights",
    "birdclef-2026-effnet-b0-sed-v5b-weights",
    "birdclef-2026-effnet-b0-sed-v5c-weights",
    "birdclef-2026-effnet-b0-sed-v5d-weights",
    "birdclef-2026-hgnet-b0-sed-v5e-weights",
    # SED v5l: v5j + n_fft=4096 (exp 019 step 4)
    "birdclef-2026-hgnet-b0-sed-v5l-weights",
    # SED v5j 4-fold ensemble (exp 019 step 4 — best single-fold 0.906 LB)
    "birdclef-2026-hgnet-b0-sed-v5j-4fold-weights",
    # SED v5g recipe ablations (exp 019 step 4)
    "birdclef-2026-hgnet-b0-sed-v5h-weights",
    "birdclef-2026-hgnet-b0-sed-v5i-weights",
    "birdclef-2026-hgnet-b0-sed-v5j-weights",
    "birdclef-2026-hgnet-b0-sed-v5k-weights",
    # SED 4-fold ensemble (exp 019 step 3)
    "birdclef-2026-hgnet-b0-sed-v5g-4fold-weights",
    # SED embed_dim hypothesis test (exp 019 step 4)
    "birdclef-2026-effnet-b0-sed-v5n-weights",
    "birdclef-2026-effnet-b0-sed-v5m-weights",
    # SED ttahara recipe reproduction, 234 classes (exp 019 step 2)
    "birdclef-2026-effnet-b0-sed-v5f-weights",
    "birdclef-2026-hgnet-b0-sed-v5g-weights",
    # Mel overhaul SED models, 234 classes (exp 017)
    # Step 5: loss + augmentation ablations
    "birdclef-2026-effnet-b0-sed-v4j-weights",
    # Step 3: combined winners + re-ablations
    "birdclef-2026-effnet-b0-sed-v4h-weights",
    "birdclef-2026-effnet-b0-sed-v4g-weights",
    "birdclef-2026-effnet-b0-sed-v4f-weights",
    # Step 4: fulldata
    "birdclef-2026-effnet-b0-sed-v4b-fulldata-weights",
    # Step 2 ablations + pp2
    "birdclef-2026-effnet-b0-sed-v4b-pp2-weights",
    "birdclef-2026-effnet-b0-sed-v4e-weights",
    "birdclef-2026-effnet-b0-sed-v4d-weights",
    "birdclef-2026-effnet-b0-sed-v4c-weights",
    "birdclef-2026-effnet-b0-sed-v4b-weights",
    # Step 1 baseline (v4a)
    "birdclef-2026-effnet-b0-sed-v4a-pp2-weights",
    "birdclef-2026-effnet-b0-sed-v4a-weights",
    # SED models, 234 classes (exp 016)
    "birdclef-2026-effnet-b0-sed-v1-10s-pp2-weights",
    "birdclef-2026-effnet-b0-sed-v1-10s-weights",
    "birdclef-2026-effnet-b0-sed-v3d-weights",
    "birdclef-2026-effnet-b0-sed-v3c-weights",
    "birdclef-2026-effnet-b0-sed-v3b-weights",
    "birdclef-2026-effnet-b0-sed-v3a-weights",
    "birdclef-2026-effnet-b0-sed-v2-weights",
    "birdclef-2026-effnet-b0-sed-v1-weights",
    # Student models trained on focal + soundscape + pseudo-labels, 234 classes (exp 015)
    "birdclef-2026-protoclr-student-v1-weights",
    "birdclef-2026-eat-student-v1-weights",
    # Models fine-tuned on focal + labeled soundscapes, 234 classes (exp 012)
    "birdclef-2026-protoclr-soundscape-weights",
    "birdclef-2026-eat-soundscape-weights",
    "birdclef-2026-convnext-soundscape-weights",
    # Models trained on BirdCLEF 2026 focal data, 100% fulldata (206/234 species mapped)
    "birdclef-2026-protoclr-fulldata-weights",
    "birdclef-2026-eat-fulldata-weights",
    "birdclef-2026-convnext-fulldata-weights",
    # Models trained on BirdCLEF 2026 focal data, 80/10/10 split (206/234 species mapped)
    "birdclef-2026-protoclr-weights",
    "birdclef-2026-eat-weights",
    "birdclef-2026-convnext-weights",
    # Models trained on BirdCLEF 2025 data (41/234 species mapped)
    "birdclef-2025-eat-weights",
    "birdclef-2025-convnext-weights",
    "birdclef-2025-protoclr-weights",
]


def _find_input(slug: str) -> str | None:
    """Resolve Kaggle input path, supporting both old and new layouts."""
    candidates = [
        f"/kaggle/input/{slug}",
        f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def find_weights_dir() -> Path:
    """Auto-detect which model weights are available."""
    for slug in MODEL_SLUGS:
        path = _find_input(slug)
        if path and os.path.exists(path):
            return Path(path)

    # Local testing fallback
    local = Path("artifacts/weights")
    for subdir in [
        "protoclr-2026-soundscape",
        "eat-2026-soundscape",
        "convnext-2026-soundscape",
        "protoclr-2026-fulldata",
        "eat-2026-fulldata",
        "convnext-2026-fulldata",
        "protoclr-2026",
        "eat-2026",
        "convnext-2026",
        "eat-2025",
        "convnext-2025",
        "protoclr-2025",
    ]:
        if (local / subdir / "model.onnx").exists():
            return local / subdir
    if local.exists():
        return local

    raise FileNotFoundError("No model weights found")


def find_competition_dir() -> Path:
    """Resolve competition data directory."""
    candidates = [
        "/kaggle/input/competitions/birdclef-2026",
        "/kaggle/input/birdclef-2026",
    ]
    for c in candidates:
        if os.path.exists(c):
            return Path(c)
    # Local fallback
    return Path(os.environ.get("COMPETITION_DIR", "data/birdclef-2026"))


# ---------------------------------------------------------------------------
# Mel spectrogram computation
# ---------------------------------------------------------------------------


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_mel_standard(audio, mel_cfg):
    """Compute log-mel spectrogram with AudioSet normalization (ConvNeXt).

    Matches NNAudioMelSpectrogram: log(mel + 1e-6), normalize by AudioSet stats.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    # Log scale (matching nnAudio: additive epsilon)
    mel = np.log(mel + 1e-6)

    # Normalize: (mel - mean) / (std * 2)
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)

    # Pad/trim to target frames
    target_frames = mel_cfg.get("target_frames", 500)
    T = mel.shape[1]
    if T > target_frames:
        # Center crop
        start = (T - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, 128, 500)


def compute_mel_amplitude_to_db(audio, mel_cfg):
    """Compute mel spectrogram with AmplitudeToDB + standardize + min-max [0,1].

    Matches Tom's 28 experiments and 2025 1st place normalization pipeline.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    # Power-to-dB: 10 * log10(mel + eps)
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))

    # top_db clipping
    top_db = mel_cfg.get("top_db", 80.0)
    mel_max = mel_db.max()
    mel_db = np.maximum(mel_db, mel_max - top_db)

    # Per-sample standardize (zero mean, unit variance)
    mean = mel_db.mean()
    std = max(mel_db.std(), 1e-6)
    mel_norm = (mel_db - mean) / std

    # Per-sample min-max scale to [0, 1]
    mel_min = mel_norm.min()
    mel_max = mel_norm.max()
    mel_norm = (mel_norm - mel_min) / (mel_max - mel_min + 1e-6)

    # Pad/trim to target frames
    target_frames = mel_cfg.get("target_frames", 1000)
    T = mel_norm.shape[1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel_norm = mel_norm[:, start : start + target_frames]
    elif T < target_frames:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)


def compute_mel_simple_range(audio, mel_cfg):
    """Compute mel spectrogram with simple range normalization: (X + 80) / 80.

    Maps AmplitudeToDB output from [-80, 0] to [0, 1]. Used by Salman Ahmed
    (BirdCLEF 2025). Simpler than global mean/std — no dataset statistics needed.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    # Power-to-dB: 10 * log10(mel + eps)
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))

    # top_db clipping
    top_db = mel_cfg.get("top_db", 80.0)
    mel_max = mel_db.max()
    mel_db = np.maximum(mel_db, mel_max - top_db)

    # Simple range normalization: [-top_db, 0] → [0, 1]
    mel_norm = (mel_db + top_db) / top_db

    # Pad/trim to target frames
    target_frames = mel_cfg.get("target_frames", 1000)
    T = mel_norm.shape[1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel_norm = mel_norm[:, start : start + target_frames]
    elif T < target_frames:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)


def compute_mel_ttahara(audio, mel_cfg):
    """Compute mel spectrogram with ttahara's pipeline: torchaudio + resize + min-max.

    Matches ttahara's HGNetV2-B0 baseline (0.884 LB). Per-segment processing:
    torchaudio MelSpectrogram → AmplitudeToDB → Resize(256,256) → per-sample min-max [0,1].
    """
    import torch
    import torchaudio

    sr = mel_cfg["sr"]
    wave = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=mel_cfg.get("n_fft", 2048),
        win_length=mel_cfg.get("win_length", 626),
        hop_length=mel_cfg.get("hop_length", 313),
        f_min=mel_cfg.get("fmin", 20),
        n_mels=mel_cfg.get("n_mels", 256),
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm="slaney",
        mel_scale="htk",
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=mel_cfg.get("top_db", 80.0)
    )

    mel_spec = mel_transform(wave)  # (1, n_mels, time)
    lms = db_transform(mel_spec)  # (1, n_mels, time)

    # Resize to target shape
    target_shape = mel_cfg.get("target_shape", [256, 256])
    from torchvision.transforms import v2 as tvt_v2

    resize = tvt_v2.Resize(size=target_shape)
    lms = resize(lms)  # (1, target_h, target_w)

    # Per-sample min-max normalization to [0, 1]
    lms_flat = lms.reshape(1, -1)
    lms_min = lms_flat.min(dim=1)[0][:, None, None]
    lms_max = lms_flat.max(dim=1)[0][:, None, None]
    lms = (lms - lms_min) / (lms_max - lms_min + 1e-7)

    result = lms[:, None, :, :].numpy().astype(np.float32)  # (1, 1, H, W)
    in_chans = mel_cfg.get("in_chans", 1)
    if in_chans > 1:
        result = np.repeat(result, in_chans, axis=1)  # (1, C, H, W)
    return result


def compute_mel_tucker(audio, mel_cfg):
    """Compute Tucker distilled-SED mel (exp 032, audit § 7.1).

    Pipeline (matches `MelSpecTransform` in `bc2026-distilled-sed.ipynb` cell 10):
        torchaudio MelSpectrogram (n_fft=2048, hop=512, n_mels=256,
            fmin=20, fmax=16000, power=2.0)
        → AmplitudeToDB(top_db=80)
        → per-sample (mel - mean) / (std + 1e-6)

    Returns (1, 1, n_mels, target_frames) float32. Tucker's training transform
    uses torchaudio (not librosa); we match that for numerical alignment.
    """
    import torch
    import torchaudio

    sr = mel_cfg["sr"]
    wave = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=mel_cfg.get("n_fft", 2048),
        hop_length=mel_cfg.get("hop_length", 512),
        n_mels=mel_cfg.get("n_mels", 256),
        f_min=mel_cfg.get("fmin", 20),
        f_max=mel_cfg.get("fmax", 16000),
        power=mel_cfg.get("power", 2.0),
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=mel_cfg.get("top_db", 80.0)
    )

    mel = mel_transform(wave)  # (1, n_mels, T)
    mel = db_transform(mel)  # (1, n_mels, T)

    # Per-sample mean/std normalization (Tucker cell 14 inner loop)
    mean = mel.mean()
    std = mel.std().clamp(min=1e-6)
    mel = (mel - mean) / std

    # Pad/trim to target_frames
    target_frames = mel_cfg.get("target_frames", 313)
    T = mel.shape[-1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel = mel[..., start : start + target_frames]
    elif T < target_frames:
        mel = torch.nn.functional.pad(mel, (0, target_frames - T))

    return mel.unsqueeze(0).numpy().astype(np.float32)  # (1, 1, n_mels, target_frames)


def compute_mel_kaldi_fbank(audio, mel_cfg):
    """Compute Kaldi fbank features for EAT.

    Uses torchaudio.compliance.kaldi.fbank (available on Kaggle).
    """
    import torch
    import torchaudio

    waveform = torch.from_numpy(audio).float()
    waveform = waveform - waveform.mean()

    mel = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=mel_cfg["sr"],
        use_energy=False,
        window_type="hanning",
        num_mel_bins=mel_cfg["num_mel_bins"],
        dither=0.0,
        frame_shift=mel_cfg["frame_shift"],
    )  # (time, 128)

    # Pad/truncate to target length
    target_length = mel_cfg.get("target_length", 512)
    n_frames = mel.shape[0]
    if n_frames < target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - n_frames))
    elif n_frames > target_length:
        mel = mel[:target_length, :]

    # Normalize
    mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)

    # Shape: (1, 1, target_length, 128)
    return mel.unsqueeze(0).unsqueeze(0).numpy().astype(np.float32)


def compute_mel_protoclr(audio, mel_cfg):
    """Compute ProtoCLR-compatible mel spectrogram.

    Pipeline: mel spectrogram -> dB scale -> min-max norm -> standardize.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 50),
        fmax=mel_cfg.get("fmax", 8000),
        power=2.0,
    )

    # Convert to dB (matches torchaudio.transforms.AmplitudeToDB default: ref=1.0, power)
    # AmplitudeToDB with stype="power": 10 * log10(mel / ref)
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))

    # Per-sample min-max normalization
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

    # Standardize with ProtoCLR training set statistics
    protoclr_mean = mel_cfg.get("protoclr_mean", 0.5347)
    protoclr_std = mel_cfg.get("protoclr_std", 0.0772)
    mel_std = (mel_norm - protoclr_mean) / protoclr_std

    # Pad/trim to target frames
    target_frames = mel_cfg.get("target_frames", 250)
    T = mel_std.shape[1]
    if T > target_frames:
        mel_std = mel_std[:, :target_frames]
    elif T < target_frames:
        mel_std = np.pad(mel_std, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel_std[np.newaxis, np.newaxis, :, :].astype(np.float32)  # (1, 1, 128, 250)


def _apply_pcen_numpy(mel, pcen_cfg):
    """Apply PCEN normalization to mel spectrogram (numpy).

    Args:
        mel: (n_mels, T) linear power mel spectrogram.
        pcen_cfg: Dict with keys: s, alpha, delta, r, eps.
            s/alpha/delta/r can be scalars or lists (per-channel).
    """
    s = np.array(pcen_cfg.get("s", 0.025), dtype=np.float32)
    alpha = np.array(pcen_cfg.get("alpha", 0.98), dtype=np.float32)
    delta = np.array(pcen_cfg.get("delta", 2.0), dtype=np.float32)
    r = np.array(pcen_cfg.get("r", 0.5), dtype=np.float32)
    eps = pcen_cfg.get("eps", 1e-6)

    # Reshape per-channel params: (n_mels,) → (n_mels, 1)
    if s.ndim == 1:
        s = s[:, np.newaxis]
        alpha = alpha[:, np.newaxis]
        delta = delta[:, np.newaxis]
        r = r[:, np.newaxis]

    # IIR smoothing: M_t = (1-s) * M_{t-1} + s * E_t
    n_mels, T = mel.shape
    M = np.zeros_like(mel)
    M[:, 0] = mel[:, 0]
    for t in range(1, T):
        M[:, t] = (1 - s.ravel()) * M[:, t - 1] + s.ravel() * mel[:, t]

    # PCEN: (E / (eps + M)^alpha + delta)^r - delta^r
    smooth = np.power(eps + M, alpha)
    pcen_out = np.power(mel / smooth + delta, r) - np.power(delta, r)
    return pcen_out


def compute_mel_pcen(audio, mel_cfg):
    """Compute mel spectrogram with PCEN normalization + min-max [0,1]."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    pcen_cfg = mel_cfg.get("pcen", {})
    mel = _apply_pcen_numpy(mel, pcen_cfg)

    # Per-sample min-max scale to [0, 1]
    mel_min, mel_max = mel.min(), mel.max()
    mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)

    # Pad/trim to target frames
    target_frames = mel_cfg.get("target_frames", 1000)
    T = mel.shape[1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif T < target_frames:
        mel = np.pad(mel, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel[np.newaxis, np.newaxis, :, :].astype(np.float32)


def compute_mel_pcen_amp2db(audio, mel_cfg):
    """Compute mel with PCEN → AmplitudeToDB → standardize → min-max [0,1]."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    pcen_cfg = mel_cfg.get("pcen", {})
    mel = _apply_pcen_numpy(mel, pcen_cfg)

    # AmplitudeToDB on PCEN output
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    top_db = mel_cfg.get("top_db", 80.0)
    mel_db = np.maximum(mel_db, mel_db.max() - top_db)

    # Standardize + min-max
    mean, std = mel_db.mean(), max(mel_db.std(), 1e-6)
    mel_norm = (mel_db - mean) / std
    mel_min, mel_max = mel_norm.min(), mel_norm.max()
    mel_norm = (mel_norm - mel_min) / (mel_max - mel_min + 1e-6)

    # Pad/trim
    target_frames = mel_cfg.get("target_frames", 1000)
    T = mel_norm.shape[1]
    if T > target_frames:
        start = (T - target_frames) // 2
        mel_norm = mel_norm[:, start : start + target_frames]
    elif T < target_frames:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, target_frames - T)), mode="constant")

    return mel_norm[np.newaxis, np.newaxis, :, :].astype(np.float32)


def compute_mel(audio, mel_cfg):
    """Dispatch to model-specific mel computation."""
    mel_type = mel_cfg.get("type", "mel")
    norm_type = mel_cfg.get("norm_type", "audioset")
    if mel_type == "tucker_mel":
        return compute_mel_tucker(audio, mel_cfg)
    elif mel_type == "ttahara":
        return compute_mel_ttahara(audio, mel_cfg)
    elif mel_type == "kaldi_fbank":
        return compute_mel_kaldi_fbank(audio, mel_cfg)
    elif mel_type == "protoclr_mel":
        return compute_mel_protoclr(audio, mel_cfg)
    elif norm_type == "pcen":
        return compute_mel_pcen(audio, mel_cfg)
    elif norm_type == "pcen_amp2db":
        return compute_mel_pcen_amp2db(audio, mel_cfg)
    elif norm_type == "amplitude_to_db":
        return compute_mel_amplitude_to_db(audio, mel_cfg)
    elif norm_type == "simple_range":
        return compute_mel_simple_range(audio, mel_cfg)
    else:
        return compute_mel_standard(audio, mel_cfg)


def compute_full_signal_mel(audio, mel_cfg):
    """Compute mel spectrogram for an entire audio signal (e.g. 60s).

    Returns the raw normalized mel as (n_mels, total_frames) WITHOUT
    batch/channel dims or frame truncation. Callers slice frames as needed.
    This avoids redundant STFT computation for overlapping segments.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=mel_cfg["sr"],
        n_fft=mel_cfg["n_fft"],
        hop_length=mel_cfg["hop_length"],
        n_mels=mel_cfg["n_mels"],
        fmin=mel_cfg.get("fmin", 0),
        fmax=mel_cfg.get("fmax"),
        htk=mel_cfg.get("htk", True),
        power=2.0,
    )
    norm_type = mel_cfg.get("norm_type", "audioset")
    if norm_type == "pcen":
        # PCEN on full signal → min-max [0,1]
        pcen_cfg = mel_cfg.get("pcen", {})
        mel = _apply_pcen_numpy(mel, pcen_cfg)
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
    elif norm_type == "pcen_amp2db":
        # PCEN → AmplitudeToDB → standardize → min-max [0,1]
        pcen_cfg = mel_cfg.get("pcen", {})
        mel = _apply_pcen_numpy(mel, pcen_cfg)
        top_db = mel_cfg.get("top_db", 80.0)
        mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel = np.maximum(mel, mel.max() - top_db)
        mean, std = mel.mean(), max(mel.std(), 1e-6)
        mel = (mel - mean) / std
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
    elif norm_type == "amplitude_to_db":
        # AmplitudeToDB → standardize → min-max [0,1] on the full signal.
        # Training normalizes per 10s window, but full-signal normalization gives
        # a consistent view across the file (same approach as 2025 1st place).
        top_db = mel_cfg.get("top_db", 80.0)
        mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel = np.maximum(mel, mel.max() - top_db)
        mean, std = mel.mean(), max(mel.std(), 1e-6)
        mel = (mel - mean) / std
        mel_min, mel_max = mel.min(), mel.max()
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
    elif norm_type == "simple_range":
        # Simple range: (X + top_db) / top_db → [0, 1]. No dataset stats needed.
        top_db = mel_cfg.get("top_db", 80.0)
        mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        mel = np.maximum(mel, mel.max() - top_db)
        mel = (mel + top_db) / top_db
    else:
        mel = np.log(mel + 1e-6)
        mel = (mel - mel_cfg["norm_mean"]) / (mel_cfg["norm_std"] * 2)
    return mel.astype(np.float32)  # (n_mels, total_frames)


def _slice_mel_window(full_mel, start_frame, target_frames):
    """Slice a window from a full-signal mel, padding if needed.

    Returns (1, 1, n_mels, target_frames) ready for ONNX.
    """
    total_frames = full_mel.shape[1]

    # Handle negative start (before audio begins)
    if start_frame < 0:
        pad_left = -start_frame
        start_frame = 0
    else:
        pad_left = 0

    end_frame = start_frame + target_frames - pad_left
    if end_frame > total_frames:
        end_frame = total_frames

    window = full_mel[:, start_frame:end_frame]

    # Pad if needed (left for negative start, right for end of file)
    pad_right = target_frames - pad_left - window.shape[1]
    if pad_left > 0 or pad_right > 0:
        window = np.pad(window, ((0, 0), (pad_left, max(0, pad_right))), mode="constant")

    return window[np.newaxis, np.newaxis, :, :]  # (1, 1, n_mels, target_frames)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio_chunk(audio_path, sr, start_s, duration_s):
    """Load a chunk of audio at target sample rate.

    Handles negative start times by zero-padding the beginning (needed for
    10s overlapping inference where the first segments need context before t=0).
    """
    info = sf.info(audio_path)
    source_sr = info.samplerate
    total_duration = info.frames / source_sr
    target_samples = int(sr * duration_s)

    # Handle negative start: zero-pad the beginning
    pre_pad_samples = 0
    read_start_s = start_s
    if start_s < 0:
        pre_pad_s = -start_s
        pre_pad_samples = int(sr * pre_pad_s)
        read_start_s = 0.0
        duration_s = duration_s - pre_pad_s

    actual_duration = min(duration_s, total_duration - read_start_s)
    if actual_duration <= 0:
        return np.zeros(target_samples, dtype=np.float32)

    audio, _ = sf.read(
        audio_path,
        dtype="float32",
        start=int(read_start_s * source_sr),
        frames=int(actual_duration * source_sr) + 2,
    )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr)

    # Prepend zero-padding for negative start
    if pre_pad_samples > 0:
        audio = np.concatenate([np.zeros(pre_pad_samples, dtype=np.float32), audio])

    # Pad/trim to exact target length
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _run_onnx(session, mel, mel_cfg):
    """Run ONNX inference and return probability vector."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: mel})[0]
    output = output.flatten()

    # SED models output blended probabilities directly; clip-level models output logits
    if mel_cfg.get("output_type") == "probs":
        return output
    else:
        return sigmoid(output)


def _predict_segment_with_tta(
    session, audio_path, sr, window_start_s, window_duration, mel_cfg, use_tta=True
):
    """Predict one segment, optionally with temporal TTA (2.5s shifts).

    TTA: run 3 predictions per segment (center, -2.5s, +2.5s) and average.
    Confirmed +0.012 LB from competition analysis.
    The shift is always 2.5s (half a competition segment), regardless of
    inference window size.
    """
    audio = load_audio_chunk(audio_path, sr, window_start_s, window_duration)
    mel = compute_mel(audio, mel_cfg)
    probs = _run_onnx(session, mel, mel_cfg)

    if not use_tta:
        return probs

    # TTA: shifted predictions (always 2.5s = half of 5s competition segment)
    shift_s = 2.5
    tta_preds = [probs]

    for shift in [-shift_s, shift_s]:
        shifted_start = window_start_s + shift
        # load_audio_chunk handles negative start times via zero-padding
        audio_shifted = load_audio_chunk(audio_path, sr, shifted_start, window_duration)
        mel_shifted = compute_mel(audio_shifted, mel_cfg)
        tta_preds.append(_run_onnx(session, mel_shifted, mel_cfg))

    # Average: 50% center + 25% left + 25% right
    return 0.5 * tta_preds[0] + 0.25 * tta_preds[1] + 0.25 * tta_preds[2]


def _run_onnx_batch(session, mel_batch, mel_cfg):
    """Run batched ONNX inference. mel_batch: (B, 1, n_mels, T)."""
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: mel_batch})[0]  # (B, n_classes)
    if mel_cfg.get("output_type") == "probs":
        return output
    else:
        return sigmoid(output)


def predict_soundscape_fullsignal(
    session, audio_path, mel_cfg, species_mapping, taxa_mapping=None, metadata_priors=None
):
    """Process one soundscape using full-signal mel (compute STFT once, slice frames).

    ~10-30x faster than per-segment mel computation for n_fft=4096 models.
    Only works for SED models with type="mel" normalization.
    """
    sr = mel_cfg["sr"]
    hop_length = mel_cfg["hop_length"]
    segment_duration = 5.0
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2
    target_frames = mel_cfg.get("target_frames", 1000)
    # For wider inference windows, compute the actual frame count to slice
    inference_frames = (
        int(inference_window * sr / hop_length)
        if inference_window > segment_duration
        else target_frames
    )
    soundscape_id = Path(audio_path).stem

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]

    # 1. Load entire audio file once
    audio, file_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

    total_duration = len(audio) / sr

    # 2. Pad audio for context (negative start / end overflow)
    context_samples = int(context_pad * sr)
    tta_shift_samples = int(2.5 * sr)  # TTA shift = 2.5s
    # Pad enough for context + TTA shifts on both sides
    pad_samples = context_samples + tta_shift_samples
    audio_padded = np.pad(audio, (pad_samples, pad_samples), mode="constant")

    # 3. Compute full-signal mel ONCE on the padded audio
    full_mel = compute_full_signal_mel(audio_padded, mel_cfg)
    # full_mel shape: (n_mels, total_frames)

    # Frame offset corresponding to t=0 in the original audio
    origin_frame = int(pad_samples / hop_length)

    # 4. Build all mel slices for all segments + TTA shifts
    segment_times = []
    segment_start_s = 0.0
    all_slices = []  # list of (segment_idx, tta_weight, mel_slice)
    segment_idx = 0

    while segment_start_s < total_duration:
        # Center of inference window on 5s segment
        window_center_s = segment_start_s - context_pad

        if ENABLE_TTA:
            shifts = [
                (window_center_s, 0.50),
                (window_center_s - 2.5, 0.25),
                (window_center_s + 2.5, 0.25),
            ]
        else:
            shifts = [(window_center_s, 1.0)]

        for shift_start_s, weight in shifts:
            start_frame = origin_frame + int(shift_start_s * sr / hop_length)
            mel_slice = _slice_mel_window(full_mel, start_frame, inference_frames)
            all_slices.append((segment_idx, weight, mel_slice))

        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration
        segment_idx += 1

    n_segments = segment_idx

    # 5. Batched ONNX inference — run all slices at once
    mel_batch = np.concatenate([s[2] for s in all_slices], axis=0)  # (N, 1, n_mels, T)
    all_probs = _run_onnx_batch(session, mel_batch, mel_cfg)  # (N, n_classes)

    # 6. Weighted average for TTA
    probs_matrix = np.zeros((n_segments, all_probs.shape[1]), dtype=np.float32)
    for i, (seg_idx, weight, _) in enumerate(all_slices):
        probs_matrix[seg_idx] += weight * all_probs[i]

    # 7. Post-processing pipeline (same as per-segment path)
    temperature = mel_cfg.get("temperature", 1.0)
    if temperature != 1.0:
        probs_matrix = _apply_temperature(probs_matrix, temperature)

    if ENABLE_TEMPORAL_SMOOTHING:
        probs_matrix = _apply_temporal_smoothing(probs_matrix, taxa_mapping)

    if ENABLE_TOPN:
        probs_matrix = _apply_topn(probs_matrix)

    site, hour = _parse_soundscape_metadata(audio_path)
    if metadata_priors is not None and (site is not None or hour is not None):
        probs_matrix = _apply_metadata_priors(
            probs_matrix, metadata_priors, site, hour, taxa_mapping, idx_map
        )

    # 8. Build prediction rows
    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}
        for sp in species_2026:
            row[sp] = 0.0
        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(probs_matrix[i, int(idx_str)])
        predictions.append(row)

    return predictions


def _apply_temporal_smoothing(probs_matrix, taxa_mapping=None):
    """Taxa-aware temporal smoothing across consecutive segments.

    When taxa_mapping is provided, applies different strategies by taxa:
    - Aves/Mammalia/Reptilia (events): local-max propagation (alpha=0.15)
      x_new = (1-a)*x + a*max(prev, next) — preserves transient peaks
    - Amphibia/Insecta (textures): average-neighbor smoothing (alpha=0.35)
      x_new = (1-a)*x + a*mean(prev, next) — solidifies continuous signals

    Fallback (no taxa_mapping): Gaussian kernel [0.1, 0.2, 0.4, 0.2, 0.1].
    """
    n_segments, n_species = probs_matrix.shape
    if n_segments < 3:
        return probs_matrix

    smoothed = np.copy(probs_matrix)

    if taxa_mapping is None:
        # Fallback: original uniform Gaussian kernel
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        for c in range(n_species):
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")
        return smoothed

    # Build index sets by ecological behavior
    event_indices = set()  # transient calls
    texture_indices = set()  # continuous sounds
    for taxa in ["aves", "mammalia", "reptilia"]:
        event_indices.update(taxa_mapping.get(taxa, []))
    for taxa in ["amphibia", "insecta"]:
        texture_indices.update(taxa_mapping.get(taxa, []))

    alpha_event = 0.15
    alpha_texture = 0.35

    for c in range(n_species):
        if c in event_indices:
            # Local-max propagation: preserves transient bird call peaks
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha_event) * probs_matrix[t, c] + alpha_event * max(
                    prev_val, next_val
                )
        elif c in texture_indices:
            # Average-neighbor: solidifies continuous frog/insect textures
            for t in range(n_segments):
                prev_val = probs_matrix[max(0, t - 1), c]
                next_val = probs_matrix[min(n_segments - 1, t + 1), c]
                smoothed[t, c] = (1 - alpha_texture) * probs_matrix[t, c] + alpha_texture * (
                    (prev_val + next_val) / 2
                )
        else:
            # Unknown taxa: use Gaussian kernel
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smoothed[:, c] = np.convolve(probs_matrix[:, c], kernel, mode="same")

    return smoothed


def _apply_topn(probs_matrix):
    """TopN postprocessing: multiply each segment by file-level max per species.

    Boosts species that are consistently detected across the file and suppresses
    spurious single-segment detections. Confirmed +0.011 LB.
    """
    file_max = probs_matrix.max(axis=0, keepdims=True)
    return probs_matrix * file_max


def _apply_temperature(probs_matrix, temperature):
    """Temperature scaling in logit space.

    Converts probabilities to logits, scales by 1/T, converts back.
    T > 1 softens (reduces overconfidence on dominant species),
    T < 1 sharpens. Perch notebooks use T=1.10.
    """
    if temperature is None or temperature == 1.0:
        return probs_matrix

    eps = 1e-7
    p = np.clip(probs_matrix, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    scaled_logits = logits / temperature
    return sigmoid(scaled_logits)


def _parse_soundscape_metadata(filename):
    """Parse site and hour from soundscape filename.

    Format: BC2026_{Train|Test}_NNNN_SXX_YYYYMMDD_HHMMSS.ogg
    Returns: (site, hour) or (None, None) if parsing fails.
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 6:
        return None, None

    site = parts[3]  # e.g. "S08"
    try:
        hour = int(parts[5][:2])  # first 2 digits of HHMMSS
    except (ValueError, IndexError):
        return site, None

    return site, hour


def _apply_metadata_priors(probs_matrix, metadata_priors, site, hour, taxa_mapping, idx_map):
    """Apply site/hour metadata priors multiplicatively.

    probs *= prior ** lambda, where lambda varies by taxa:
    - Aves: 0.3 (birds somewhat mobile, moderate prior)
    - Amphibia/Insecta: 0.5 (strongly site/time dependent)
    - Mammalia/Reptilia: 0.2 (most mobile)

    Fallback hierarchy: site+hour -> site -> hour -> global.
    """
    if metadata_priors is None:
        return probs_matrix

    n_segments, n_species = probs_matrix.shape
    lambda_map = {"aves": 0.3, "amphibia": 0.5, "insecta": 0.5, "mammalia": 0.2, "reptilia": 0.2}

    # Build index-to-taxa lookup
    idx_to_taxa = {}
    if taxa_mapping:
        for taxa_class, indices in taxa_mapping.items():
            for idx in indices:
                idx_to_taxa[idx] = taxa_class

    # Prior lookup tables
    site_hour_priors = metadata_priors.get("site_hour_species", {})
    site_priors = metadata_priors.get("site_species", {})
    hour_priors = metadata_priors.get("hour_species", {})
    global_priors = metadata_priors.get("global_species", {})

    site_str = site if site else None
    hour_str = str(hour).zfill(2) if hour is not None else None

    # Build per-species prior multiplier vector
    prior_vector = np.ones(n_species, dtype=np.float32)
    for idx_str, species_code in idx_map.items():
        idx = int(idx_str)
        if idx >= n_species:
            continue

        # Fallback hierarchy: site+hour -> site -> hour -> global
        prior_val = None
        if site_str and hour_str:
            prior_val = site_hour_priors.get(site_str, {}).get(hour_str, {}).get(species_code)
        if prior_val is None and site_str:
            prior_val = site_priors.get(site_str, {}).get(species_code)
        if prior_val is None and hour_str:
            prior_val = hour_priors.get(hour_str, {}).get(species_code)
        if prior_val is None:
            prior_val = global_priors.get(species_code)
        if prior_val is None:
            continue  # No prior data; leave at 1.0

        # Scale lambda by taxa class
        taxa = idx_to_taxa.get(idx, "aves")
        lam = lambda_map.get(taxa, 0.3)
        prior_vector[idx] = prior_val**lam

    # Apply multiplicatively to all segments
    return probs_matrix * prior_vector[np.newaxis, :]


# Post-processing flags (set to True to enable)
ENABLE_TTA = True
ENABLE_TEMPORAL_SMOOTHING = True
ENABLE_TOPN = True


def predict_soundscape(
    session, audio_path, mel_cfg, species_mapping, taxa_mapping=None, metadata_priors=None
):
    """Process one soundscape file and return prediction rows.

    Each 5s segment produces one row with 234 BirdCLEF 2026 species columns.

    If mel_cfg["inference_window"] > 5s (e.g., 10s), loads a wider audio
    window centered on each 5s target segment. This matches the training
    window size and provides temporal context (2.5s before + after).
    2024 1st place confirmed +0.015 LB from this approach.

    Post-processing pipeline:
      TTA → temperature scaling → taxa-aware smoothing → TopN → metadata priors
    """
    sr = mel_cfg["sr"]
    segment_duration = 5.0  # competition: predictions every 5s
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2  # 2.5s for 10s window
    soundscape_id = Path(audio_path).stem

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]

    # Collect all segment probabilities as a matrix for post-processing
    segment_probs = []
    segment_times = []
    segment_start_s = 0.0

    while segment_start_s < total_duration:
        # Center the inference window on the 5s target segment
        # For 10s inference: audio_start = segment_start - 2.5s
        audio_start = segment_start_s - context_pad
        probs = _predict_segment_with_tta(
            session,
            audio_path,
            sr,
            audio_start,
            inference_window,
            mel_cfg,
            use_tta=ENABLE_TTA,
        )
        segment_probs.append(probs)
        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration

    # Stack into (n_segments, n_model_outputs) matrix
    probs_matrix = np.stack(segment_probs)  # (n_segments, n_model_outputs)

    # Post-processing pipeline (order matters):
    # 1. Temperature scaling (calibration)
    temperature = mel_cfg.get("temperature", 1.0)
    if temperature != 1.0:
        probs_matrix = _apply_temperature(probs_matrix, temperature)

    # 2. Taxa-aware temporal smoothing
    if ENABLE_TEMPORAL_SMOOTHING:
        probs_matrix = _apply_temporal_smoothing(probs_matrix, taxa_mapping)

    # 3. TopN (file-level consistency)
    if ENABLE_TOPN:
        probs_matrix = _apply_topn(probs_matrix)

    # 4. Metadata priors (site/hour conditioning)
    site, hour = _parse_soundscape_metadata(audio_path)
    if metadata_priors is not None and (site is not None or hour is not None):
        probs_matrix = _apply_metadata_priors(
            probs_matrix, metadata_priors, site, hour, taxa_mapping, idx_map
        )

    # Build prediction rows mapped to 234 submission columns
    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}

        for sp in species_2026:
            row[sp] = 0.0

        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(probs_matrix[i, int(idx_str)])

        predictions.append(row)

    return predictions


def predict_soundscape_tucker(
    session, audio_path, mel_cfg, species_mapping, taxa_mapping=None, metadata_priors=None
):
    """Tucker distilled-SED inference (exp 032, audit § 7.3).

    Per audit § 7.3 (Tucker's `bc2026-distilled-sed` cell 24): everything stays
    in LOGIT space until the very end.

        Per 5s window:
            mel = compute_mel_tucker(audio[start:start+5s])
            blended_logits = ONNX(mel)  # ← already 0.5·clip + 0.5·frame_max, exported wrapper
        Stack 12 windows → (12, 234) logits
        Gaussian-smooth across windows in LOGIT space ([0.1, 0.2, 0.4, 0.2, 0.1], sums to 1)
        Sigmoid once at the end → (12, 234) probs
        Map to 234 BirdCLEF-2026 columns

    No TTA, no temperature, no TopN, no metadata priors in Phase 1 — matches
    Tucker's published recipe verbatim. The shared post-processing pipeline is
    DELIBERATELY skipped here so the LB read isolates the Tucker recipe lift.
    Site/hour priors layered on top in a separate Phase 2 kernel if Phase 1 lands.
    `taxa_mapping`, `metadata_priors` accepted only to match the dispatch signature.
    """
    del taxa_mapping, metadata_priors  # unused, intentional

    sr = mel_cfg["sr"]
    segment_duration = 5.0
    soundscape_id = Path(audio_path).stem

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    # Per-window ONNX inference; collect blended logits (LOGIT space)
    window_logits = []
    segment_times = []
    segment_start_s = 0.0
    input_name = session.get_inputs()[0].name

    while segment_start_s < total_duration:
        audio = load_audio_chunk(audio_path, sr, segment_start_s, segment_duration)
        mel = compute_mel(audio, mel_cfg)  # (1, 1, n_mels, T)
        out = session.run(None, {input_name: mel})[0]  # (1, 234) — pre-blended logits
        window_logits.append(out.flatten())
        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration

    if not window_logits:
        return []

    logits_matrix = np.stack(window_logits)  # (n_windows, 234)

    # 5-tap Gaussian smoothing across windows in LOGIT space
    if mel_cfg.get("gaussian_smooth_logit_space", True) and logits_matrix.shape[0] >= 3:
        kernel = np.array(
            mel_cfg.get("gaussian_kernel", [0.1, 0.2, 0.4, 0.2, 0.1]), dtype=np.float32
        )
        kernel = kernel / kernel.sum()  # safety: re-normalize in case kernel cfg drifts
        # mode='same' returns same length; boundaries use zero-padding by default.
        # Tucker's reference uses scipy.ndimage.convolve1d(mode='nearest') (audit § 7.4).
        # We approximate by padding with edge values:
        pad = len(kernel) // 2
        padded = np.pad(logits_matrix, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.zeros_like(logits_matrix)
        for c in range(logits_matrix.shape[1]):
            smoothed[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
        logits_matrix = smoothed

    # Single sigmoid at the end (LOGIT → prob)
    probs_matrix = sigmoid(logits_matrix)

    # Map model probs → 234 BirdCLEF-2026 species columns
    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}
        for sp in species_2026:
            row[sp] = 0.0
        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(probs_matrix[i, int(idx_str)])
        predictions.append(row)

    return predictions


def predict_soundscape_xcpretrain(
    session, audio_path, mel_cfg, species_mapping, taxa_mapping=None, metadata_priors=None
):
    """Exp 034 XCPretrain inference — feed RAW AUDIO to the full-model ONNX.

    The ONNX is the FULL pipeline (raw audio → nnAudio mel → backbone → SED head
    → clipwise_pred_long PROBABILITIES), so there is NO kernel-side mel computation
    → zero mel-mismatch risk. Per non-overlapping 5s segment: load audio →
    ONNX(audio) → (1, 234) probs → stack windows → 5-tap Gaussian temporal smooth
    in LOGIT space → map to 234 columns.

    PP (added 2026-05-28): 5-tap Gaussian smoothing across windows, mirroring exp
    032's Tucker 0.906 recipe (`predict_soundscape_tucker`). The ONNX returns
    prob-space `clipwise_pred_long`, so we inverse-sigmoid → smooth → sigmoid back;
    logit-space matches the winning recipe + the SED logit-space-aggregation finding.
    Toggled by `mel_cfg["gaussian_smooth_logit_space"]` (default True). Still no TTA
    / TopN / priors. `taxa_mapping` / `metadata_priors` accepted only to match the
    dispatch signature.
    """
    del taxa_mapping, metadata_priors  # unused, intentional

    sr = mel_cfg["sr"]
    segment_duration = mel_cfg.get("window_duration", 5.0)
    soundscape_id = Path(audio_path).stem

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate
    input_name = session.get_inputs()[0].name  # "audio"

    window_probs = []
    segment_times = []
    segment_start_s = 0.0
    while segment_start_s < total_duration:
        # load_audio_chunk pads/trims to exactly sr*segment_duration (= 160000) samples.
        audio = load_audio_chunk(audio_path, sr, segment_start_s, segment_duration)
        audio = np.asarray(audio, dtype=np.float32).reshape(1, -1)  # (1, T)
        out = session.run(None, {input_name: audio})[0]  # (1, 234) probs (already sigmoid'd)
        window_probs.append(out.flatten())
        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration

    if not window_probs:
        return []
    probs_matrix = np.stack(window_probs)  # (n_windows, 234) — already probabilities

    # 5-tap Gaussian temporal smoothing in LOGIT space (mirrors exp 032's Tucker
    # 0.906 recipe — see predict_soundscape_tucker). ONNX returns prob-space
    # clipwise_pred_long, so inverse-sigmoid → smooth across windows → sigmoid back.
    if mel_cfg.get("gaussian_smooth_logit_space", True) and probs_matrix.shape[0] >= 3:
        eps = 1e-7
        p = np.clip(probs_matrix, eps, 1 - eps)
        logits_matrix = np.log(p / (1 - p))  # inverse-sigmoid (pseudo-logit)
        kernel = np.array(
            mel_cfg.get("gaussian_kernel", [0.1, 0.2, 0.4, 0.2, 0.1]), dtype=np.float32
        )
        kernel = kernel / kernel.sum()  # safety re-normalize if cfg drifts
        pad = len(kernel) // 2
        padded = np.pad(logits_matrix, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.zeros_like(logits_matrix)
        for c in range(logits_matrix.shape[1]):
            smoothed[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
        probs_matrix = sigmoid(smoothed)

    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}
        for sp in species_2026:
            row[sp] = 0.0
        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(probs_matrix[i, int(idx_str)])
        predictions.append(row)
    return predictions


def predict_soundscape_tucker_ensemble(
    sessions, audio_path, mel_cfg, species_mapping, taxa_mapping=None, metadata_priors=None
):
    """Tucker 5-fold ensemble inference (audit § 7.3 cell 24, verbatim).

    For each 5s window, for each fold session, run ONNX → blended logits
    (LOGIT space, pre-blended via our export wrapper). Average across folds
    in LOGIT space → stack 12 windows per file → 5-tap Gauss smooth in
    LOGIT space → single sigmoid.

    Matches Tucker's cell 24 verbatim except the per-window clip+frame_max
    blend is already baked into the ONNX (single output, 0.5/0.5).

    No TTA, no temperature, no TopN, no priors — same Tucker-faithful
    inference as `predict_soundscape_tucker` (single-fold). Site/hour
    priors can be layered on top later if needed.
    """
    del taxa_mapping, metadata_priors  # unused, intentional

    sr = mel_cfg["sr"]
    segment_duration = 5.0
    soundscape_id = Path(audio_path).stem

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    n_folds = len(sessions)
    input_name = sessions[0].get_inputs()[0].name

    window_logits = []
    segment_times = []
    segment_start_s = 0.0

    while segment_start_s < total_duration:
        audio = load_audio_chunk(audio_path, sr, segment_start_s, segment_duration)
        mel = compute_mel(audio, mel_cfg)  # (1, 1, n_mels, T)
        fold_logits_sum = None
        for sess in sessions:
            out = sess.run(None, {input_name: mel})[0].flatten().astype(np.float32)
            fold_logits_sum = out if fold_logits_sum is None else fold_logits_sum + out
        window_logits.append(fold_logits_sum / n_folds)  # logit-space mean
        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration

    if not window_logits:
        return []

    logits_matrix = np.stack(window_logits)  # (n_windows, 234)

    # 5-tap Gaussian smoothing across windows in LOGIT space (same as single-fold path)
    if mel_cfg.get("gaussian_smooth_logit_space", True) and logits_matrix.shape[0] >= 3:
        kernel = np.array(
            mel_cfg.get("gaussian_kernel", [0.1, 0.2, 0.4, 0.2, 0.1]), dtype=np.float32
        )
        kernel = kernel / kernel.sum()
        pad = len(kernel) // 2
        padded = np.pad(logits_matrix, ((pad, pad), (0, 0)), mode="edge")
        smoothed = np.zeros_like(logits_matrix)
        for c in range(logits_matrix.shape[1]):
            smoothed[:, c] = np.convolve(padded[:, c], kernel, mode="valid")
        logits_matrix = smoothed

    probs_matrix = sigmoid(logits_matrix)

    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}
        for sp in species_2026:
            row[sp] = 0.0
        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(probs_matrix[i, int(idx_str)])
        predictions.append(row)

    return predictions


def predict_soundscape_ensemble(
    sessions, audio_path, mel_cfg, species_mapping, taxa_mapping=None, metadata_priors=None
):
    """4-fold ensemble: compute mel once per segment, run all models, average probs.

    Same interface as predict_soundscape() but takes a list of ONNX sessions.
    Computes mel spectrogram once per segment (the expensive step), then runs
    each fold model on the same mel input. Averages probabilities across folds
    before applying post-processing.
    """
    sr = mel_cfg["sr"]
    segment_duration = 5.0
    inference_window = mel_cfg.get("inference_window", segment_duration)
    context_pad = (inference_window - segment_duration) / 2
    soundscape_id = Path(audio_path).stem

    info = sf.info(audio_path)
    total_duration = info.frames / info.samplerate

    species_2026 = species_mapping["birdclef2026_species"]
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    n_model_outputs = max(int(k) for k in idx_map.keys()) + 1

    # Collect per-segment mel spectrograms (with TTA shifts)
    segment_mels = []  # list of [(mel, weight), ...] per segment
    segment_times = []
    segment_start_s = 0.0

    while segment_start_s < total_duration:
        audio_start = segment_start_s - context_pad

        if ENABLE_TTA:
            shifts = [
                (audio_start, 0.50),
                (audio_start - 2.5, 0.25),
                (audio_start + 2.5, 0.25),
            ]
        else:
            shifts = [(audio_start, 1.0)]

        mels_with_weights = []
        for shift_start, weight in shifts:
            audio = load_audio_chunk(audio_path, sr, shift_start, inference_window)
            mel = compute_mel(audio, mel_cfg)
            mels_with_weights.append((mel, weight))
        segment_mels.append(mels_with_weights)

        segment_times.append(int(segment_start_s + segment_duration))
        segment_start_s += segment_duration

    n_segments = len(segment_mels)
    n_folds = len(sessions)

    # Run all fold models on all segments (mel computed once, ONNX runs N_folds times)
    fold_probs = np.zeros((n_folds, n_segments, n_model_outputs), dtype=np.float32)
    for fold_idx, sess in enumerate(sessions):
        for seg_idx, mels_with_weights in enumerate(segment_mels):
            seg_prob = np.zeros(n_model_outputs, dtype=np.float32)
            for mel, weight in mels_with_weights:
                prob = _run_onnx(sess, mel, mel_cfg)
                seg_prob += weight * prob
            fold_probs[fold_idx, seg_idx] = seg_prob

    # Average across folds
    avg_probs = fold_probs.mean(axis=0)  # (n_segments, n_model_outputs)

    # Post-processing on averaged probabilities
    temperature = mel_cfg.get("temperature", 1.0)
    if temperature != 1.0:
        avg_probs = _apply_temperature(avg_probs, temperature)

    if ENABLE_TEMPORAL_SMOOTHING:
        avg_probs = _apply_temporal_smoothing(avg_probs, taxa_mapping)

    if ENABLE_TOPN:
        avg_probs = _apply_topn(avg_probs)

    site, hour = _parse_soundscape_metadata(audio_path)
    if metadata_priors is not None and (site is not None or hour is not None):
        avg_probs = _apply_metadata_priors(
            avg_probs, metadata_priors, site, hour, taxa_mapping, idx_map
        )

    # Build prediction rows
    predictions = []
    for i, end_time in enumerate(segment_times):
        row_id = f"{soundscape_id}_{end_time}"
        row = {"row_id": row_id}
        for sp in species_2026:
            row[sp] = 0.0
        for idx_str, name_2026 in idx_map.items():
            row[name_2026] = float(avg_probs[i, int(idx_str)])
        predictions.append(row)

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    start_time = time.time()

    # Resolve paths
    weights_dir = find_weights_dir()
    competition_dir = find_competition_dir()
    test_dir = competition_dir / "test_soundscapes"
    sample_sub_path = competition_dir / "sample_submission.csv"
    output_path = Path("/kaggle/working/submission.csv")

    # Fallback for local testing
    if not test_dir.exists():
        test_dir = Path(os.environ.get("TEST_SOUNDSCAPES_DIR", "test_soundscapes"))
    if not sample_sub_path.exists():
        sample_sub_path = Path(os.environ.get("SAMPLE_SUBMISSION", "sample_submission.csv"))
    if not output_path.parent.exists():
        output_path = Path("predictions/submission.csv")

    # Detect ensemble mode (4-fold: model_fold0.onnx, ..., model_fold3.onnx)
    fold_onnx_paths = sorted(weights_dir.glob("model_fold*.onnx"))
    is_ensemble = len(fold_onnx_paths) > 0

    # Load model artifacts
    onnx_path = weights_dir / "model.onnx"
    mel_config_path = weights_dir / "mel_config.json"
    species_mapping_path = weights_dir / "species_mapping.json"

    print(f"Model weights: {weights_dir}")
    if is_ensemble:
        print(f"Ensemble mode: {len(fold_onnx_paths)} fold models")
    else:
        print(f"ONNX model: {onnx_path}")

    with open(mel_config_path) as f:
        mel_cfg = json.load(f)
    with open(species_mapping_path) as f:
        species_mapping = json.load(f)

    mel_type = mel_cfg.get("type", "mel")
    print(f"Mel type: {mel_type}, SR: {mel_cfg['sr']}")
    idx_map = species_mapping.get("idx_to_name_2026") or species_mapping["idx_2025_to_name_2026"]
    print(f"Species mapping: {len(idx_map)} mapped to 2026 columns")

    # Load optional pp2 artifacts (taxa mapping, metadata priors)
    taxa_mapping = None
    taxa_path = weights_dir / "taxa_mapping.json"
    if taxa_path.exists():
        with open(taxa_path) as f:
            taxa_mapping = json.load(f)
        n_taxa_species = sum(len(v) for v in taxa_mapping.values())
        print(f"Taxa mapping: {n_taxa_species} species across {len(taxa_mapping)} classes")

    metadata_priors = None
    priors_path = weights_dir / "metadata_priors.json"
    if priors_path.exists():
        with open(priors_path) as f:
            metadata_priors = json.load(f)
        print(
            f"Metadata priors: {len(metadata_priors.get('sites', []))} sites, "
            f"{metadata_priors.get('n_segments', 0)} segments"
        )

    if mel_cfg.get("temperature", 1.0) != 1.0:
        print(f"Temperature scaling: T={mel_cfg['temperature']}")

    # Create ONNX session(s) (CPU only, optimized)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count() or 4
    sess_opts.inter_op_num_threads = 1

    if is_ensemble:
        sessions = []
        for fold_path in fold_onnx_paths:
            s = ort.InferenceSession(
                str(fold_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
            )
            sessions.append(s)
        session = sessions[0]  # for compatibility with non-ensemble code paths
        print(
            f"{len(sessions)} ONNX sessions created "
            f"(threads={sess_opts.intra_op_num_threads}, {time.time() - start_time:.1f}s)"
        )
    else:
        session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
        sessions = [session]
        print(
            f"ONNX session created (threads={sess_opts.intra_op_num_threads}, "
            f"{time.time() - start_time:.1f}s)"
        )

    # Find soundscape files
    audio_files = sorted(
        list(test_dir.glob("*.ogg")) + list(test_dir.glob("*.wav")) + list(test_dir.glob("*.flac"))
    )

    # Fallback: if test_soundscapes is empty (public run), use train_soundscapes
    # so we can validate the full inference pipeline in visible logs
    if len(audio_files) == 0:
        train_dir = competition_dir / "train_soundscapes"
        if train_dir.exists():
            audio_files = sorted(train_dir.glob("*.ogg"))[:2]
            print(f"No test soundscapes found, using {len(audio_files)} train files for validation")

    print(f"Processing {len(audio_files)} soundscapes from {test_dir}")

    # Choose inference strategy
    use_tucker = mel_cfg.get("type") == "tucker_mel"
    use_tucker_ensemble = use_tucker and is_ensemble
    use_fullsignal = mel_cfg.get("architecture") == "sed" and mel_type == "mel"
    use_xcpretrain = (
        mel_cfg.get("type") == "xcpretrain_audio_onnx"
    )  # exp 034: full-model audio→probs ONNX
    if use_xcpretrain:
        print(
            "Using XCPretrain full-model audio-input ONNX (raw audio → in-graph "
            "nnAudio mel → backbone → SED head → probs) → 5-tap LOGIT-space Gauss "
            "smooth. No kernel mel, no TTA/priors."
        )
    elif use_tucker_ensemble:
        print(
            f"Using Tucker {len(sessions)}-fold ensemble inference (per-window "
            "LOGIT-space fold mean → 5-tap Gauss smooth → single sigmoid). "
            "No TTA / no priors."
        )
    elif use_tucker:
        print(
            "Using Tucker distilled-SED inference (per-window LOGIT-space blend "
            "→ 5-tap Gauss smooth → single sigmoid at end). No TTA / no priors."
        )
    elif is_ensemble:
        print(
            f"Using {len(sessions)}-fold ensemble inference (compute mel once, run {len(sessions)} models)"
        )
    elif use_fullsignal:
        print("Using full-signal mel inference (compute STFT once per file)")
    else:
        print("Using per-segment mel inference")

    # Run inference
    all_predictions = []
    for i, audio_path in enumerate(audio_files):
        t0 = time.time()
        if use_xcpretrain:
            preds = predict_soundscape_xcpretrain(
                session, str(audio_path), mel_cfg, species_mapping
            )
        elif use_tucker_ensemble:
            preds = predict_soundscape_tucker_ensemble(
                sessions,
                str(audio_path),
                mel_cfg,
                species_mapping,
                taxa_mapping=taxa_mapping,
                metadata_priors=metadata_priors,
            )
        elif use_tucker:
            preds = predict_soundscape_tucker(
                session,
                str(audio_path),
                mel_cfg,
                species_mapping,
                taxa_mapping=taxa_mapping,
                metadata_priors=metadata_priors,
            )
        elif is_ensemble:
            preds = predict_soundscape_ensemble(
                sessions,
                str(audio_path),
                mel_cfg,
                species_mapping,
                taxa_mapping=taxa_mapping,
                metadata_priors=metadata_priors,
            )
        elif use_fullsignal:
            preds = predict_soundscape_fullsignal(
                session,
                str(audio_path),
                mel_cfg,
                species_mapping,
                taxa_mapping=taxa_mapping,
                metadata_priors=metadata_priors,
            )
        else:
            preds = predict_soundscape(
                session,
                str(audio_path),
                mel_cfg,
                species_mapping,
                taxa_mapping=taxa_mapping,
                metadata_priors=metadata_priors,
            )
        elapsed = time.time() - t0
        print(
            f"  [{i + 1}/{len(audio_files)}] {audio_path.name}: {len(preds)} rows ({elapsed:.1f}s)"
        )
        all_predictions.extend(preds)

    print(f"Total predictions: {len(all_predictions)} rows")

    # Build submission merged with sample_submission.csv
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        expected_columns = list(sample_sub.columns)
        species_columns = [c for c in expected_columns if c != "row_id"]

        if all_predictions:
            pred_df = pd.DataFrame(all_predictions)
            submission = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
            for col in species_columns:
                if col not in submission.columns:
                    submission[col] = 0.0
                else:
                    submission[col] = submission[col].fillna(0.0)
        else:
            # No test soundscapes (public test) — fill with zeros
            submission = sample_sub.copy()
            for col in species_columns:
                submission[col] = 0.0
        submission = submission[expected_columns]
    else:
        submission = pd.DataFrame(all_predictions) if all_predictions else pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    elapsed_total = time.time() - start_time
    print(f"Submission saved to {output_path} ({len(submission)} rows)")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total / 60:.1f}min)")


if __name__ == "__main__":
    main()
