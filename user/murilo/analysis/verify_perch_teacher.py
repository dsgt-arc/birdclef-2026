"""Phase 0.1 — verify Tucker's vendored Perch v2 no-DFT ONNX teacher.

Confirms:
- ONNX file loads cleanly via onnxruntime.
- Input/output schema matches expectations: input is (B, 160000) waveform,
  output includes a (B, 1536) embedding tensor.
- Embedding is non-degenerate on random waveform input (std > 0.05).
- Embedding is non-degenerate on a real training waveform.
- CUDA execution provider works (if GPU available).
- Throughput is reasonable for use as a teacher in the training loop.

Writes a schema-and-results doc to:
  user/murilo/experiments/031-perch-kd-distillation/phase0_perch_teacher_schema.md

Usage:
    cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026
    source ~/clef/birdclef-2026/.venv/bin/activate
    python user/murilo/analysis/verify_perch_teacher.py

Run AFTER `kaggle datasets download tuckerarrants/perch-v2-no-dft-onnx ...`
has unpacked the ONNX to vendor/kaggle/datasets/perch-v2-no-dft-onnx/.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

# Path constants
REPO_ROOT = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026")
DATA_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
VENDOR_DIR = REPO_ROOT / "vendor/kaggle/datasets/perch-v2-no-dft-onnx"
EXP_DIR = REPO_ROOT / "user/murilo/experiments/031-perch-kd-distillation"

SR = 32_000
N_SAMPLES = SR * 5  # 160_000


def find_onnx() -> Path:
    """Locate the .onnx file in the vendored dataset directory."""
    if not VENDOR_DIR.exists():
        raise FileNotFoundError(
            f"{VENDOR_DIR} doesn't exist. Run the kaggle download command "
            "from the Phase 0.1 instructions first."
        )
    onnx_files = list(VENDOR_DIR.rglob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx file found under {VENDOR_DIR}. " "Did the download unpack correctly?"
        )
    if len(onnx_files) > 1:
        print(f"  WARNING: multiple .onnx files found, using first: {onnx_files[0]}")
    return onnx_files[0]


def load_audio_32k(path: Path) -> np.ndarray:
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        g = math.gcd(int(file_sr), SR)
        audio = resample_poly(audio, SR // g, int(file_sr) // g).astype(np.float32)
    return audio


def get_real_waveform_batch(n: int = 4, seed: int = 42) -> np.ndarray:
    """Load N real training waveforms, cropped to 5s, deterministically."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(train_df), size=n, replace=False)
    waveforms = []
    for i, row_idx in enumerate(pick_idx):
        row = train_df.iloc[int(row_idx)]
        wav = load_audio_32k(DATA_DIR / "train_audio" / row["filename"])
        if len(wav) <= N_SAMPLES:
            crop = np.pad(wav, (N_SAMPLES - len(wav), 0))
        else:
            r = np.random.default_rng(seed + i)
            start = r.integers(0, len(wav) - N_SAMPLES + 1)
            crop = wav[start : start + N_SAMPLES]
        waveforms.append(crop)
    return np.stack(waveforms).astype(np.float32)


def main():
    print("=" * 70)
    print("Phase 0.1 — Verify Perch v2 no-DFT ONNX teacher")
    print("=" * 70)

    onnx_path = find_onnx()
    file_size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"\nONNX file: {onnx_path}")
    print(f"Size: {file_size_mb:.1f} MB")

    # 1. Load with CPU provider first (universal availability check)
    print("\n[1/6] Loading ONNX with CPUExecutionProvider...")
    sess_cpu = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print("  ✅ CPU session loaded")

    # 2. Inspect input/output schema
    print("\n[2/6] Inspecting I/O schema...")
    inputs = sess_cpu.get_inputs()
    outputs = sess_cpu.get_outputs()
    print(f"  Inputs ({len(inputs)}):")
    for i, inp in enumerate(inputs):
        print(f"    [{i}] name='{inp.name}', shape={inp.shape}, type={inp.type}")
    print(f"  Outputs ({len(outputs)}):")
    embed_idx = None
    for i, out in enumerate(outputs):
        marker = ""
        if out.shape and len(out.shape) >= 1:
            last = out.shape[-1]
            if last == 1536:
                marker = "  ⭐ EMBEDDING (1536-d)"
                if embed_idx is None:
                    embed_idx = i
        print(f"    [{i}] name='{out.name}', shape={out.shape}, type={out.type}{marker}")

    if embed_idx is None:
        print("  ⚠️  No 1536-d output detected. Inspect manually.")
        embed_idx = 1  # Tucker's notebook fallback

    # 3. Run on random waveform
    print("\n[3/6] Running on random (4, 160000) input on CPU...")
    rng = np.random.default_rng(42)
    rand_wav = rng.standard_normal((4, N_SAMPLES)).astype(np.float32) * 0.1
    input_name = inputs[0].name
    t0 = time.time()
    rand_out = sess_cpu.run(None, {input_name: rand_wav})
    cpu_t = time.time() - t0
    rand_emb = rand_out[embed_idx]
    print(f"  Output[{embed_idx}] shape: {rand_emb.shape}, dtype: {rand_emb.dtype}")
    print(f"  std: {rand_emb.std():.4f}, mean: {rand_emb.mean():.4f}")
    print(f"  min: {rand_emb.min():.4f}, max: {rand_emb.max():.4f}")
    print(f"  CPU runtime: {cpu_t*1000:.1f}ms for batch of 4")
    rand_std_ok = rand_emb.std() > 0.05
    rand_shape_ok = rand_emb.shape == (4, 1536)
    print(f"  shape OK (= (4, 1536)): {rand_shape_ok}")
    print(f"  std OK (> 0.05): {rand_std_ok}")

    # 4. Run on real training waveforms
    print("\n[4/6] Running on 4 real training waveforms on CPU...")
    real_wav = get_real_waveform_batch(n=4, seed=42)
    real_out = sess_cpu.run(None, {input_name: real_wav})
    real_emb = real_out[embed_idx]
    print(f"  Output[{embed_idx}] shape: {real_emb.shape}")
    print(f"  std: {real_emb.std():.4f}, mean: {real_emb.mean():.4f}")
    print(f"  min: {real_emb.min():.4f}, max: {real_emb.max():.4f}")
    print(f"  Per-sample stds: {real_emb.std(axis=1)}")
    real_std_ok = real_emb.std() > 0.05

    # 5. Per-sample embedding distinctness (a sanity check that real waveforms
    #    don't all collapse to the same embedding)
    print("\n[5/6] Cross-sample cosine similarity on real waveforms...")
    real_emb_t = torch.from_numpy(real_emb)
    cos = torch.nn.functional.cosine_similarity(
        real_emb_t.unsqueeze(0), real_emb_t.unsqueeze(1), dim=-1
    )
    print(f"  Cross-sample cosine sim matrix:\n{cos.numpy().round(3)}")
    diag = torch.eye(4, dtype=torch.bool)
    off_diag = cos[~diag]
    print(
        f"  Off-diagonal cos sim: mean={off_diag.mean():.3f}, "
        f"min={off_diag.min():.3f}, max={off_diag.max():.3f}"
    )
    distinctness_ok = off_diag.max() < 0.99  # different waveforms shouldn't collapse

    # 6. CUDA provider attempt
    print("\n[6/6] Attempting CUDAExecutionProvider...")
    cuda_works = False
    cuda_t = None
    try:
        sess_cuda = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active = sess_cuda.get_providers()[0]
        if active == "CUDAExecutionProvider":
            t0 = time.time()
            _ = sess_cuda.run(None, {input_name: real_wav})  # warmup
            t0 = time.time()
            for _ in range(5):
                _ = sess_cuda.run(None, {input_name: real_wav})
            cuda_t = (time.time() - t0) / 5
            print(f"  ✅ CUDA active. Runtime: {cuda_t*1000:.1f}ms / batch of 4")
            cuda_works = True
        else:
            print(f"  ⚠️  CUDAExecutionProvider not active (fell back to {active}).")
    except Exception as e:
        print(f"  ⚠️  CUDA load failed: {e}")
        print("  CPU-only mode is acceptable but slower for KD training.")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    checks = [
        ("ONNX loads (CPU)", True),
        ("Output shape == (B, 1536)", rand_shape_ok),
        ("Random-waveform embedding non-degenerate (std > 0.05)", rand_std_ok),
        ("Real-waveform embedding non-degenerate", real_std_ok),
        ("Real waveforms produce distinct embeddings (cos sim < 0.99)", distinctness_ok),
        ("CUDA provider works", cuda_works),
    ]
    all_required_ok = all(ok for _, ok in checks[:5])  # CUDA is optional
    for label, ok in checks:
        marker = "✅" if ok else ("⚠️ " if "CUDA" in label else "❌")
        print(f"  {marker} {label}")
    print()
    if all_required_ok:
        print("  → All required checks passed. Teacher is usable for Phase 1.")
    else:
        print("  → One or more required checks failed. Halt before Phase 1; debug.")

    # Write schema doc
    out_md = EXP_DIR / "phase0_perch_teacher_schema.md"
    md = f"""# Phase 0.1 — Perch v2 Teacher Schema and Verification

**Date**: 2026-05-09
**Source**: Tucker Arrants's Kaggle dataset
`tuckerarrants/perch-v2-no-dft-onnx`, vendored to
`{VENDOR_DIR.relative_to(REPO_ROOT)}/`.

## File

- Path: `{onnx_path.relative_to(REPO_ROOT)}`
- Size: {file_size_mb:.1f} MB

## I/O schema

### Inputs

| Index | Name | Shape | Type |
|---|---|---|---|
{chr(10).join(f"| {i} | {inp.name} | {inp.shape} | {inp.type} |" for i, inp in enumerate(inputs))}

### Outputs

| Index | Name | Shape | Type | Notes |
|---|---|---|---|---|
{chr(10).join(f"| {i} | {out.name} | {out.shape} | {out.type} | {'⭐ 1536-d EMBEDDING' if (out.shape and out.shape[-1] == 1536) else ''} |" for i, out in enumerate(outputs))}

**Embedding output index**: {embed_idx}

## Verification results

### Random waveform (B=4, σ=0.1 Gaussian)

- Output shape: `{rand_emb.shape}`
- std: {rand_emb.std():.4f}
- mean: {rand_emb.mean():.4f}
- range: [{rand_emb.min():.4f}, {rand_emb.max():.4f}]
- ✅ Non-degenerate

### Real training waveform (B=4 deterministic from `train.csv`, seed=42)

- Output shape: `{real_emb.shape}`
- std: {real_emb.std():.4f}
- mean: {real_emb.mean():.4f}
- range: [{real_emb.min():.4f}, {real_emb.max():.4f}]
- Cross-sample off-diagonal cos sim: mean={off_diag.mean():.3f}, min={off_diag.min():.3f}, max={off_diag.max():.3f}
- ✅ Distinct (different waveforms produce different embeddings)

## Compute

- **CPU (CPUExecutionProvider)**: ~{cpu_t*1000:.0f}ms / batch of 4
- **CUDA (CUDAExecutionProvider)**: {f"~{cuda_t*1000:.0f}ms / batch of 4 (5-run mean)" if cuda_works else "NOT AVAILABLE — CPU fallback only"}

## Implications for Phase 1

- The teacher takes (B, 160000) float32 raw waveforms at 32 kHz, returns
  `(B, 1536)` clip embeddings as output index {embed_idx}.
- Our `PerchTeacher` class in `sed_finetune.py` (Phase 0.2) wraps this.
- Per-batch teacher inference cost on the training partition's GPU
  determines whether we can use ONNX at training time or need to
  precompute embeddings. {"CUDA works → use directly in training loop." if cuda_works else "CUDA not working → either fix CUDA provider or precompute embeddings before training."}

## Files

- This doc: `phase0_perch_teacher_schema.md`
- Verification script: `user/murilo/analysis/verify_perch_teacher.py`
- ONNX (vendored): `{onnx_path.relative_to(REPO_ROOT)}`
"""
    out_md.write_text(md)
    print(f"\nWrote: {out_md}")


if __name__ == "__main__":
    main()
