"""Decide the SSAMBA CPU forward path: eager pure-PyTorch vs the unrolled ONNX.

The unrolled FP32 ONNX (513-token x 24-layer x 2-dir scan -> ~25K nodes, 203 MB) timed out
under ORT on CPU — the graph-optimization passes at session init are superlinear in node
count. This script measures, at 4 threads (Kaggle's vCPU allotment), the actual win/s of:
  A. eager SSAMBAHFEncoder.extract_features (pure-PyTorch scan, no graph) — likely faster
  B. the existing ONNX via ORT with graph_optimization_level=DISABLE_ALL (skip the slow
     passes; measure session-init AND forward separately)
Run UNBUFFERED (python -u) so partial results survive a kill. The Kaggle budget: ~12K
windows (5 s grid) or ~6K (10 s geometry) in 90 min -> need ~2.2 win/s (5 s) / ~1.1 (10 s).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
SSAMBA_CKPT = Path("~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth").expanduser()
SR = 16000
WINDOW_SAMPLES = SR * 5
sys.path.insert(0, str(EXP.parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(EXP / "artifacts" / "ssamba_onnx" / "ssamba_mean.onnx"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--batches", default="8,16,32,64", help="comma batch sizes to sweep")
    ap.add_argument("--int8", action="store_true", help="also test dynamic-INT8 quantized Linears")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    audio = torch.from_numpy(
        rng.standard_normal((args.windows, WINDOW_SAMPLES)).astype(np.float32)
    )
    wav = audio.unsqueeze(1)  # (n,1,samples)
    batches = [int(b) for b in args.batches.split(",")]

    from birdclef.encoders.ssamba_hf import SSAMBAHFEncoder

    print(f"[build] loading eager SSAMBAHFEncoder on CPU (threads={args.threads})", flush=True)
    enc = SSAMBAHFEncoder(checkpoint_path=str(SSAMBA_CKPT), model_size="small", device="cpu")
    enc.model.eval()
    for m in enc.model.modules():
        if hasattr(m, "_use_cuda_kernels"):
            m._use_cuda_kernels = False

    def bench(tag: str) -> None:
        """Sweep batch sizes; the 513-step Python scan loop has fixed per-step overhead
        amortized across the batch, so win/s should rise with batch until ops dominate."""
        with torch.no_grad():
            enc.extract_features(wav[:1])  # warm
            for bs in batches:
                t0 = time.perf_counter()
                for i in range(0, len(wav), bs):
                    enc.extract_features(wav[i : i + bs])
                dt = time.perf_counter() - t0
                wps = args.windows / dt
                print(f"[{tag} bs={bs:>3}] {args.windows} win in {dt:.1f}s -> {wps:.3f} win/s "
                      f"| proj 12K={12000/wps/60:.1f}min  6K={6000/wps/60:.1f}min", flush=True)

    bench("EAGER")

    # Dynamic INT8 on the Linear-heavy projections (in/out/x/dt). The scan recurrence is
    # elementwise (not Linear), so this only helps if the projections are a real fraction.
    if args.int8:
        import torch.ao.quantization as tq

        print("[build] quantize_dynamic Linear -> qint8", flush=True)
        enc.model = tq.quantize_dynamic(enc.model, {torch.nn.Linear}, dtype=torch.qint8)
        bench("INT8")


if __name__ == "__main__":
    main()
