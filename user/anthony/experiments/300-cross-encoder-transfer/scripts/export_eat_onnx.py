"""Export EAT (cls-pooled, AudioSet-finetuned) to ONNX for offline CPU Kaggle inference.

EAT's ONLY deployment blocker is `trust_remote_code=True` — the kernel can't pull the
remote modeling `.py` offline. Tracing `model.extract_features(mel)` to ONNX captures the
transformer as standard graph ops, so the kernel needs neither the remote code nor the HF
weights at inference (just the ONNX graph + torchaudio for the mel). Unlike AST, EAT is
cheap on CPU (5 s/512-frame mel -> ~257 tokens vs AST's ~1200), so FP32 is plenty fast and
gives EXACT cache parity (passes the strict 0.999 consistency gate — no INT8 needed).

This single job:
  1. Build the deployed EATEncoder (worstchan/EAT-base_epoch30_finetune_AS2M, target_length
     512, pooling=cls) — the SAME wrapper that built the frozen-cls-ft cache.
  2. Wrap so forward(mel (B,1,512,128)) -> extract_features -> CLS token [:, 0] (B, 768).
  3. torch.onnx.export FP32 (dynamic batch, opset 17, legacy exporter).
  4. Parity vs the eager EATEncoder on real focal windows: cosine(eager, ORT-fp32) must be
     ~1.0 (same math) — this is what the in-kernel consistency check will later confirm.
  5. Benchmark end-to-end win/s @ 4 threads (Kaggle's vCPU); project against the 90-min wall.

The mel is torchaudio kaldi-fbank (EATEncoder._waveform_to_mel) — NOT an HF feature
extractor — so the kernel replicates it directly; only the transformer is exported.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

EXP = Path(__file__).resolve().parents[1]
# The deployed EAT cache uses the AudioSet-finetuned checkpoint, frozen CLS, 5 s geometry.
EAT_CKPT = "worstchan/EAT-base_epoch30_finetune_AS2M"
EAT_FOCAL = Path("~/scratch/birdclef-2026/features/exp300-deploy/eat").expanduser()
SR = 16000
WINDOW_SAMPLES = SR * 5  # 80000, 5 s @ 16 kHz — the exp-300 window geometry
TARGET_FRAMES = 512  # 5 s at 10 ms hop
WALL_MIN = 90.0  # Kaggle CPU rerun limit

# birdclef.encoders lives under user/anthony/src (same path the extractor imports from).
sys.path.insert(0, str(EXP.parents[2] / "src"))


class EatClsWrapper(nn.Module):
    """forward(mel (B,1,512,128)) -> CLS-pooled embedding (B,768).

    extract_features returns [CLS, patch_0 ... patch_n]; CLS at position 0 is EAT's
    utterance summary — identical to EATEncoder(pooling='cls'), so the exported graph
    reproduces the trained-on embeddings."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        feats = self.model.extract_features(mel)  # (B, 1+patches, 768)
        return feats[:, 0]  # (B, 768)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_real_windows(enc, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mels (n,1,512,128), eager_emb (n,768)) for the first focal file with
    >= `limit` windows — real bird audio so parity reflects the deployed inputs."""
    import polars as pl
    import soundfile as sf
    import torchaudio.functional as taF

    cfg = yaml.safe_load((EXP / "configs/exp300.yaml").read_text())
    audio_dir = Path(cfg["focal"]["audio_dir"]).expanduser()
    shard = sorted((EAT_FOCAL / "focal").glob("shard-*.parquet"))[0]
    df = pl.read_parquet(shard, columns=["filename"])
    counts = df["filename"].value_counts(sort=True).filter(pl.col("count") >= limit)
    fn = counts["filename"][0]
    audio, sr = sf.read(str(audio_dir / fn), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = taF.resample(torch.from_numpy(audio).float(), sr, SR).numpy()
    n = max(1, int(np.ceil(len(audio) / WINDOW_SAMPLES)))
    audio = np.pad(audio, (0, n * WINDOW_SAMPLES - len(audio)))
    windows = audio.reshape(n, WINDOW_SAMPLES)[:limit]
    wav = torch.from_numpy(windows).float().unsqueeze(1)  # (n,1,samples)
    mels = enc._waveform_to_mel(wav.squeeze(1)).cpu()  # (n,1,512,128)
    eager = enc.extract_features(wav).cpu().numpy()  # (n,768) — the cache readout
    print(f"parity file={fn} windows={len(windows)} mel={tuple(mels.shape)}")
    return mels.numpy().astype(np.float32), eager.astype(np.float64)


def ort_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def run_ort(sess, name, mels: np.ndarray, batch: int = 8) -> np.ndarray:
    out = []
    for i in range(0, len(mels), batch):
        out.append(sess.run(None, {name: mels[i : i + batch].astype(np.float32)})[0])
    return np.concatenate(out)


def benchmark(sess, name, enc, n_windows: int, threads: int, batch: int) -> dict:
    """End-to-end win/s: kaldi-fbank mel (per window) + batched ORT forward. Random
    audio is fine for timing (compute is input-independent)."""
    torch.set_num_threads(threads)
    rng = np.random.default_rng(0)
    audio = torch.from_numpy(
        rng.standard_normal((n_windows, WINDOW_SAMPLES)).astype(np.float32)
    )
    warm = enc._waveform_to_mel(audio[:1]).cpu().numpy().astype(np.float32)
    run_ort(sess, name, warm, batch)
    t0 = time.perf_counter()
    mels = enc._waveform_to_mel(audio).cpu().numpy().astype(np.float32)
    run_ort(sess, name, mels, batch)
    dt = time.perf_counter() - t0
    wps = n_windows / dt
    return {
        "n_windows": n_windows,
        "threads": threads,
        "batch": batch,
        "win_per_sec": round(wps, 3),
        "proj_min_8400": round(8400 / wps / 60, 1),
        "proj_min_12000": round(12000 / wps / 60, 1),
        "clears_wall_12000": bool(12000 / wps / 60 < WALL_MIN),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=EAT_CKPT, help="HF id or local snapshot dir")
    ap.add_argument("--out-dir", default=str(EXP / "artifacts" / "eat_onnx"))
    ap.add_argument("--summary", default=str(EXP / "artifacts" / "eat_onnx_export.json"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=4, help="match Kaggle's 4 vCPU")
    ap.add_argument("--parity-windows", type=int, default=8)
    ap.add_argument("--bench-windows", type=int, default=64)
    ap.add_argument("--bench-batch", type=int, default=8)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "eat_cls.onnx"

    from birdclef.encoders.eat import EATEncoder

    print(f"loading EAT from {args.checkpoint} (CPU, trust_remote_code)")
    enc = EATEncoder(
        model_name=args.checkpoint,
        target_length=TARGET_FRAMES,
        device="cpu",
        pooling="cls",
    )
    enc.model.eval()
    wrapper = EatClsWrapper(enc.model).eval()

    # 1+2. export FP32 — fixed (1,1,512,128) mel, dynamic batch
    dummy = torch.randn(1, 1, TARGET_FRAMES, 128)
    torch.onnx.export(
        wrapper, dummy, str(fp32),
        export_params=True, opset_version=args.opset, do_constant_folding=True,
        input_names=["mel"], output_names=["embedding"],
        dynamic_axes={"mel": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,
    )
    print(f"exported FP32 -> {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)")

    # 3. parity vs eager on real windows
    mels, eager = load_real_windows(enc, args.parity_windows)
    s32, n32 = ort_session(fp32, args.threads)
    o32 = run_ort(s32, n32, mels, args.bench_batch).astype(np.float64)
    cos32 = [cos(eager[i], o32[i]) for i in range(len(eager))]
    parity = {
        "fp32_min_cos": round(min(cos32), 6),
        "fp32_mean_cos": round(float(np.mean(cos32)), 6),
        "fp32_passes_0999": bool(min(cos32) > 0.999),
    }
    print(f"parity {json.dumps(parity)}")

    # 4. benchmark @ 4 threads
    bench = {}
    if not args.skip_bench:
        bench["fp32"] = benchmark(s32, n32, enc, args.bench_windows, args.threads, args.bench_batch)
        print(f"bench[fp32] {json.dumps(bench['fp32'])}")

    summary = {
        "checkpoint": args.checkpoint,
        "opset": args.opset,
        "target_frames": TARGET_FRAMES,
        "fp32_onnx": str(fp32),
        "parity": parity,
        "benchmark": bench,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"summary -> {args.summary}")


if __name__ == "__main__":
    main()
