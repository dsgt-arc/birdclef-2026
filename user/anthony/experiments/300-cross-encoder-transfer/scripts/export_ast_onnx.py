"""Export AST (cls-pooled) to ONNX + INT8 for fast CPU Kaggle inference.

AST is the one exp-300 encoder that fails to deploy: its eager transformers forward
runs at ~1.72 win/s on Kaggle's 4 vCPU, so the hidden test (~8.4K-12K windows) blows
the 90-min CPU wall -> blank score. The fix (results.md "Deploy: AST cls"): a cheaper
forward via ONNX (+ optional INT8), re-validated to cache parity.

This single CPU job does the whole export + evidence loop:
  1. Wrap ASTModel so forward(input_values) -> cls-pooled (B, 768) embedding
     (mean of the [CLS]+[DIST] tokens) -- the SAME readout as kaggle_kernel AstEncoder
     and birdclef.encoders.ASTEncoder(pooling="cls").
  2. torch.onnx.export FP32 (fixed 1024x128 input_values, dynamic batch, opset 17,
     legacy exporter -- dynamo breaks on attention; mirrors birdclef.onnx_export).
  3. onnxruntime.quantization.quantize_dynamic -> INT8 (weight-only, no calibration).
  4. Parity: real focal windows -> cosine(eager, ORT-fp32) and cosine(eager, ORT-int8).
     ORT-fp32 must be ~1.0 (same math); INT8 reported so we know if it clears 0.999.
  5. Benchmark: end-to-end (ASTFeatureExtractor mel + model) win/s @ 4 threads for both
     graphs; project minutes for 8.4K / 12K windows against the 90-min wall.

ONNX files land in --out-dir; a JSON summary lands in --summary. The mel stays in
ASTFeatureExtractor (cheap, ~500 frames padded to 1024) -- only the transformer is
exported, so the kernel keeps the exact training mel and head_ast.npz transfers as-is.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

EXP = Path(__file__).resolve().parents[1]
AST_HF_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_FOCAL = Path("~/scratch/birdclef-2026/features/exp300-ast").expanduser()
SR = 16000
WINDOW_SAMPLES = SR * 5  # 80000, 5 s @ 16 kHz -- the exp-300 window geometry
WALL_MIN = 90.0  # Kaggle CPU rerun limit


class AstClsWrapper(nn.Module):
    """forward(input_values (B,1024,128)) -> cls-pooled embedding (B,768).

    cls = mean of last_hidden_state[:, :2] ([CLS]+[DIST]) -- identical to the kernel's
    AstEncoder readout, so the exported graph reproduces the trained-on embeddings."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        h = self.model(input_values).last_hidden_state  # (B, 1+1+patches, 768)
        return h[:, :2].mean(dim=1)  # (B, 768)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_real_windows(fe, limit: int) -> np.ndarray:
    """Mels (n, 1024, 128) for the first focal file with >= `limit` windows -- real
    inputs so INT8 parity + timing reflect actual bird audio, not random noise."""
    import polars as pl
    import soundfile as sf
    import torchaudio.functional as taF

    cfg = yaml.safe_load((EXP / "configs/exp300.yaml").read_text())
    audio_dir = Path(cfg["focal"]["audio_dir"]).expanduser()
    shard = sorted((AST_FOCAL / "focal").glob("shard-*.parquet"))[0]
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
    mels = [fe(w, sampling_rate=SR, return_tensors="np")["input_values"][0] for w in windows]
    print(f"parity file={fn} windows={len(mels)}")
    return np.stack(mels).astype(np.float32)  # (limit, 1024, 128)


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


def benchmark(sess, name, fe, n_windows: int, threads: int, batch: int) -> dict:
    """End-to-end win/s: ASTFeatureExtractor mel (per window) + batched ORT forward.
    Random audio is fine for timing (compute is input-independent)."""
    torch.set_num_threads(threads)
    rng = np.random.default_rng(0)
    audio = [rng.standard_normal(WINDOW_SAMPLES).astype(np.float32) for _ in range(n_windows)]
    # warmup
    warm = fe(audio[0], sampling_rate=SR, return_tensors="np")["input_values"]
    run_ort(sess, name, warm.astype(np.float32), batch)
    t0 = time.perf_counter()
    mels = np.stack(
        [fe(a, sampling_rate=SR, return_tensors="np")["input_values"][0] for a in audio]
    ).astype(np.float32)
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
    ap.add_argument("--snapshot", default=AST_HF_ID, help="HF id or local snapshot dir")
    ap.add_argument("--out-dir", default=str(EXP / "artifacts" / "ast_onnx"))
    ap.add_argument("--summary", default=str(EXP / "artifacts" / "ast_onnx_export.json"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--threads", type=int, default=4, help="match Kaggle's 4 vCPU")
    ap.add_argument("--parity-windows", type=int, default=8)
    ap.add_argument("--bench-windows", type=int, default=64)
    ap.add_argument("--bench-batch", type=int, default=8)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32 = out_dir / "ast_cls.onnx"
    int8 = out_dir / "ast_cls.int8.onnx"

    from transformers import ASTFeatureExtractor, ASTModel

    print(f"loading AST from {args.snapshot}")
    fe = ASTFeatureExtractor.from_pretrained(args.snapshot)
    model = ASTModel.from_pretrained(args.snapshot).eval()
    wrapper = AstClsWrapper(model).eval()
    # Drop the heavy HF model snapshot from the kernel: ship only the mel preprocessor
    # (preprocessor_config.json) beside the ONNX graph so out_dir is a self-contained
    # Kaggle dataset (onnx + mel config), no ASTModel weights needed at inference.
    fe.save_pretrained(str(out_dir))

    # 1+2. export FP32
    dummy = torch.randn(1, 1024, 128)
    torch.onnx.export(
        wrapper, dummy, str(fp32),
        export_params=True, opset_version=args.opset, do_constant_folding=True,
        input_names=["input_values"], output_names=["embedding"],
        dynamic_axes={"input_values": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,
    )
    print(f"exported FP32 -> {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)")

    # 3. INT8 dynamic quant (weight-only, no calibration set)
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"quantized INT8 -> {int8} ({int8.stat().st_size / 1e6:.1f} MB)")

    # 4. parity vs eager on real windows
    mels = load_real_windows(fe, args.parity_windows)
    with torch.no_grad():
        eager = wrapper(torch.from_numpy(mels)).numpy()
    s32, n32 = ort_session(fp32, args.threads)
    s8, n8 = ort_session(int8, args.threads)
    o32 = run_ort(s32, n32, mels, args.bench_batch)
    o8 = run_ort(s8, n8, mels, args.bench_batch)
    cos32 = [cos(eager[i], o32[i]) for i in range(len(eager))]
    cos8 = [cos(eager[i], o8[i]) for i in range(len(eager))]
    parity = {
        "fp32_min_cos": round(min(cos32), 6),
        "fp32_mean_cos": round(float(np.mean(cos32)), 6),
        "int8_min_cos": round(min(cos8), 6),
        "int8_mean_cos": round(float(np.mean(cos8)), 6),
        "int8_passes_0999": bool(min(cos8) > 0.999),
    }
    print(f"parity {json.dumps(parity)}")

    # 5. benchmark @ 4 threads
    bench = {}
    if not args.skip_bench:
        bench["fp32"] = benchmark(s32, n32, fe, args.bench_windows, args.threads, args.bench_batch)
        bench["int8"] = benchmark(s8, n8, fe, args.bench_windows, args.threads, args.bench_batch)
        for k, v in bench.items():
            print(f"bench[{k}] {json.dumps(v)}")

    summary = {
        "snapshot": args.snapshot,
        "opset": args.opset,
        "fp32_onnx": str(fp32),
        "int8_onnx": str(int8),
        "parity": parity,
        "benchmark": bench,
        "eager_baseline_win_per_sec": 1.72,  # results.md job 9957035, for reference
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"summary -> {args.summary}")


if __name__ == "__main__":
    main()
