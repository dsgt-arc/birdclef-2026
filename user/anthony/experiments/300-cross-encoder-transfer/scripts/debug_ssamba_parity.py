"""Isolate the NaN in the mamba-onnx fused-op SSAMBA forward.

Checks, in order: mel finite? eager finite? onnx (zig) finite? onnx (cpp) finite?
Then element-wise max-abs-diff eager-vs-onnx on the finite rows, and reports the
first NaN column index in the onnx output so we can tell op-blowup from layout bug.
"""
import os
import sys
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

from birdclef.encoders.ssamba_hf import SSAMBAHFEncoder

CKPT = Path(os.path.expanduser("~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_small_400.pth"))
ONNX_PATH = EXP / "artifacts/ssamba_onnx/ssamba_small_10s_mambaonnx.onnx"
SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
WIN = 160_000


def one_real_window(enc):
    import librosa
    import soundfile as sf
    f = sorted(SHARED.glob("train_soundscapes/*.ogg"))[0]
    audio, sr = sf.read(str(f))
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != 16000:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    wav = torch.from_numpy(np.asarray(audio[:WIN], dtype=np.float32))
    return enc._waveform_to_fbank(wav.unsqueeze(0))  # (1, 1024, 128)


def stat(name, arr):
    a = np.asarray(arr)
    nan = int(np.isnan(a).sum())
    inf = int(np.isinf(a).sum())
    print(f"  {name:18s} shape={tuple(a.shape)} nan={nan} inf={inf} "
          f"min={np.nanmin(a):.4g} max={np.nanmax(a):.4g} norm={np.linalg.norm(np.nan_to_num(a)):.4g}")
    return nan + inf


def main():
    torch.set_num_threads(4)
    enc = SSAMBAHFEncoder(str(CKPT), model_size="small", input_fdim=128, input_tdim=1024, device="cpu")
    for m in enc.model.modules():
        if hasattr(m, "_use_cuda_kernels"):
            m._use_cuda_kernels = False

    mel = one_real_window(enc)  # (1,1024,128)
    print("[mel]"); stat("mel", mel.numpy())

    with torch.no_grad():
        eager = enc.model.forward(mel).cpu().numpy()
    print("[eager]"); bad_eager = stat("eager", eager)

    import onnxruntime as ort
    import mamba_onnx
    for backend in ("zig", "cpp"):
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            mamba_onnx.register_custom_ops(so, backend=backend)
            sess = ort.InferenceSession(str(ONNX_PATH), so, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"input": mel.numpy().astype(np.float32)})[0]
            print(f"[onnx-{backend}]"); bad = stat(f"onnx-{backend}", out)
            if not bad and not bad_eager:
                diff = np.abs(eager - out)
                cos = float(np.dot(eager[0], out[0]) /
                            (np.linalg.norm(eager[0]) * np.linalg.norm(out[0]) + 1e-9))
                print(f"  -> max|diff|={diff.max():.4g} cosine={cos:.6f}")
            elif bad:
                first = np.argwhere(~np.isfinite(out[0]))
                print(f"  -> first non-finite col: {first[:5].ravel().tolist()}")
        except Exception as e:
            print(f"[onnx-{backend}] ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
