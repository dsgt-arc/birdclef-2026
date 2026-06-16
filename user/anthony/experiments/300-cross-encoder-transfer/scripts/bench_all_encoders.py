"""Controlled CPU inference benchmark for all 6 deploy encoders, on equal footing.

Pins threads to 4 (Kaggle's vCPU allotment) by monkeypatching os.cpu_count() — every
kernel encoder reads it (ORT intra_op at submission.py:139/224/275, torch at 352, plus
the global torch.set_num_threads below covers BEATs which doesn't self-set). Then runs
each encoder's DEPLOYED encode_file over the same real 60 s soundscapes and reports the
deploy-true cost: s/clip and projected minutes for 1000 clips. Encoders forward different
window counts/clip (Perch/BEATs/EAT 12, AST 6, SSAMBA 3-via-stride) so s/clip — not raw
win/s — is the apples-to-apples wall metric; raw window throughput is reported too.

One encoder per invocation (fresh 4-thread process); results appended to bench_all.json.

    python scripts/bench_all_encoders.py perch       # .venv-300
    python scripts/bench_all_encoders.py birdnet      # .venv-300-birdnet
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.cpu_count = lambda: 4  # the single thread-pinning lever for every encoder

import numpy as np  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "kaggle_kernel"))
sys.path.insert(0, str(EXP / "scripts"))
import submission as sub  # noqa: E402

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026")
N_CLIPS = int(os.environ.get("BENCH_CLIPS", "8"))
RESULT = EXP / "artifacts" / "bench_all.json"
PERCH_GLOB = "~/scratch/.cache/huggingface/hub/models--justinchuby--Perch-onnx/snapshots/*/"


def build(enc: str):
    if enc == "perch":
        d = sorted(Path(os.path.expanduser("~")).glob(
            ".cache/huggingface/hub/models--justinchuby--Perch-onnx/snapshots/*"))
        # fall back to scratch cache
        if not d:
            d = sorted(Path(os.path.expanduser(
                "~/scratch/.cache/huggingface/hub/models--justinchuby--Perch-onnx/snapshots")).glob("*"))
        return sub.PerchEncoder(d[0])
    # reuse consistency_check's local constructors for the other five
    from consistency_check import build_kernel_encoder
    return build_kernel_encoder("ssamba_hf" if enc == "ssamba" else enc)


def main(enc: str) -> None:
    try:
        import torch
        torch.set_num_threads(4)
    except ImportError:
        pass

    clips = sorted(SHARED.glob("train_soundscapes/*.ogg"))[:N_CLIPS]
    assert clips, "no train soundscapes found"

    t_build = time.perf_counter()
    e = build(enc)
    build_s = time.perf_counter() - t_build

    e.encode_file(str(clips[0]))  # warmup (model graph, mel kernels, JIT)

    t0 = time.perf_counter()
    total_rows = 0
    for f in clips:
        emb, _ = e.encode_file(str(f))
        total_rows += len(emb)
    dt = time.perf_counter() - t0

    n = len(clips)
    repeat = int(getattr(e, "repeat", 1))
    rows_per_clip = total_rows / n
    forwards_per_clip = rows_per_clip / repeat  # actual model calls/clip (tiling removed)
    s_per_clip = dt / n
    rtf = 60.0 / s_per_clip                     # realtime factor (60 s audio / wall s)
    est_min_1000 = s_per_clip * 1000 / 60
    fwd_per_s = (forwards_per_clip * n) / dt

    rec = {
        "encoder": enc,
        "build_s": round(build_s, 1),
        "clips": n,
        "rows_per_clip": round(rows_per_clip, 1),
        "forwards_per_clip": round(forwards_per_clip, 1),
        "s_per_clip": round(s_per_clip, 2),
        "forwards_per_s": round(fwd_per_s, 2),
        "realtime_factor": round(rtf, 1),
        "est_min_1000_clips": round(est_min_1000, 1),
    }
    print(f"[bench] {enc}: build={build_s:.1f}s  {s_per_clip:.2f} s/clip  "
          f"{forwards_per_clip:.0f} fwd/clip  {fwd_per_s:.2f} fwd/s  RTF={rtf:.1f}x  "
          f"-> ~{est_min_1000:.1f} min/1000 clips")

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if RESULT.exists():
        data = [r for r in json.loads(RESULT.read_text()) if r["encoder"] != enc]
    data.append(rec)
    RESULT.write_text(json.dumps(data, indent=2))
    print(f"[bench] wrote {RESULT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "perch")
