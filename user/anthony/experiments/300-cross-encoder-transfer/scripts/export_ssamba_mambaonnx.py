"""Export SSAMBA at native 10 s geometry with the mamba-onnx fused custom op,
verify ONNX==eager parity, and benchmark CPU throughput on Kaggle's vCPU budget.

The selective-scan recurrence is the dominant cost; the stock exporter unrolled
it (513 steps x 24 layers x 2 dir -> ~25K-node graph that ORT could not init).
mamba-onnx (acmiyaguchi/mamba-onnx) replaces it with a SINGLE
`mamba::SelectiveScanFusedExact` node backed by an AVX2 + threaded kernel.

Native 16x16 geometry at 1024 frames (10 s) -> 513 tokens, which is EXACTLY the
geometry the deployed head_ssamba.npz was refit on, so the head is reused as-is:
this swaps only the forward (eager -> ONNX custom op), not the representation.

GATE: parity cosine > 0.999 vs eager, AND projected wall < 90 min for ~6000
windows (6 win/clip x ~1000 clips, full coverage).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ["REPO_DIR"])
EXP = REPO / "user/anthony/experiments/300-cross-encoder-transfer"
PORT = REPO / "user/anthony/experiments/004-ssamba-hf-port/ssamba_hf"
EXP005 = REPO / "user/anthony/experiments/005-ssamba-token-context/scripts"
SRC = REPO / "user/anthony/src"
for p in (str(PORT), str(EXP005), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from birdclef.encoders.ssamba_hf import SSAMBAHFEncoder  # noqa: E402  (after sys.path)

CKPT = Path(os.path.expanduser(
    "~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth"))
OUT = EXP / "artifacts/ssamba_onnx"
OUT.mkdir(parents=True, exist_ok=True)
ONNX_PATH = OUT / "ssamba_small_10s_mambaonnx.onnx"
RESULT_JSON = OUT / "export_bench.json"

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
WINDOW_SAMPLES = 160_000  # 10 s @ 16 kHz
N_THREADS = int(os.environ.get("BENCH_THREADS", "4"))  # Kaggle vCPU budget


def real_mel_windows(enc, n_windows=24):
    """N real 10 s mel windows (1024, 128) from a soundscape, via the encoder front-end.

    Loads with soundfile + librosa (the kernel's exact path; avoids torchcodec).
    """
    import librosa
    import soundfile as sf
    oggs = sorted(SHARED.glob("train_soundscapes/*.ogg"))[:4]
    wav_chunks = []
    for f in oggs:
        audio, sr = sf.read(str(f))
        if audio.ndim > 1:
            audio = audio.mean(1)
        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        for s in range(0, wav.numel() - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
            wav_chunks.append(wav[s:s + WINDOW_SAMPLES])
            if len(wav_chunks) >= n_windows:
                break
        if len(wav_chunks) >= n_windows:
            break
    batch = torch.stack(wav_chunks)  # (N, 160000)
    mel = enc._waveform_to_fbank(batch)  # (N, 1024, 128)
    return mel


def main():
    torch.set_num_threads(N_THREADS)
    print(f"[build] loading {CKPT.name} at native 10 s geometry")
    enc = SSAMBAHFEncoder(str(CKPT), model_size="small",
                          input_fdim=128, input_tdim=1024, device="cpu")
    for m in enc.model.modules():
        if hasattr(m, "_use_cuda_kernels"):
            m._use_cuda_kernels = False
    tdim, fdim = int(enc._model_tdim), int(enc._model_fdim)
    print(f"[build] geometry tdim={tdim} fdim={fdim} num_patches={enc.config.num_patches}")

    # --- export with the fused custom op ---
    from ssamba_onnx_export import export_ssamba
    dummy = torch.randn(1, tdim, fdim)
    print("[export] exporting model.forward with mamba::SelectiveScanFusedExact ...")
    export_ssamba(enc.model, dummy, str(ONNX_PATH))
    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"[export] wrote {ONNX_PATH.name} ({size_mb:.1f} MB)")

    # --- real windows + eager reference ---
    mel = real_mel_windows(enc, n_windows=24)  # (N, 1024, 128)
    print(f"[parity] {mel.shape[0]} real windows")
    with torch.no_grad():
        eager = enc.model.forward(mel).cpu().numpy().astype(np.float32)  # (N, 384)

    # --- ONNX custom-op session (zig backend, no runtime deps) ---
    import onnxruntime as ort
    import mamba_onnx
    print("[onnx] available backends:", mamba_onnx.available_backends())
    so = ort.SessionOptions()
    so.intra_op_num_threads = N_THREADS
    so.inter_op_num_threads = 1
    lib = mamba_onnx.register_custom_ops(so, backend="zig")
    print("[onnx] registered:", Path(lib).name)
    sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": mel.numpy().astype(np.float32)})[0]  # (N, 384)

    # --- parity ---
    cos = np.array([
        float(np.dot(eager[i], onnx_out[i]) /
              (np.linalg.norm(eager[i]) * np.linalg.norm(onnx_out[i]) + 1e-9))
        for i in range(len(eager))
    ])
    print(f"[parity] cosine eager-vs-onnx  min={cos.min():.6f}  mean={cos.mean():.6f}")

    # --- benchmark throughput (Kaggle vCPU budget) ---
    bench = {}
    for bs in (1, 8, 16):
        feed_mel = mel[:bs].numpy().astype(np.float32)
        for _ in range(2):  # warmup
            sess.run(None, {"input": feed_mel})
        reps = 5
        t0 = time.perf_counter()
        for _ in range(reps):
            sess.run(None, {"input": feed_mel})
        dt = (time.perf_counter() - t0) / reps
        win_s = bs / dt
        bench[f"batch{bs}"] = {"sec_per_call": round(dt, 4), "win_per_s": round(win_s, 3)}
        print(f"[bench] batch{bs:>2}: {dt*1000:7.1f} ms/call  -> {win_s:6.2f} win/s")

    best = max(b["win_per_s"] for b in bench.values())
    n_windows_full = 6 * 1000  # 6 win/clip x ~1000 clips, full coverage
    proj_min = n_windows_full / best / 60
    print(f"[gate] best {best:.2f} win/s -> ~{proj_min:.1f} min for {n_windows_full} windows "
          f"({'CLEARS' if proj_min < 90 else 'OVER'} 90-min wall)")

    RESULT_JSON.write_text(json.dumps({
        "onnx_mb": round(size_mb, 1),
        "geometry": {"tdim": tdim, "fdim": fdim, "num_patches": enc.config.num_patches},
        "threads": N_THREADS,
        "parity_cosine_min": float(cos.min()),
        "parity_cosine_mean": float(cos.mean()),
        "bench": bench,
        "best_win_per_s": best,
        "proj_min_6000_windows": round(proj_min, 1),
        "clears_wall": bool(proj_min < 90),
    }, indent=2))
    print(f"[done] wrote {RESULT_JSON}")


if __name__ == "__main__":
    main()
