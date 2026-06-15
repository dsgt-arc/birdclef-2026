"""Risk #1 verification: the in-kernel encoder must reproduce the embeddings the head
was trained on. For one shared focal file, run the kernel's `encode_file` and compare —
per-window cosine — against the stored training embedding. Cosine > 0.999 = no drift.

Usage (inside the matching venv):
    python scripts/consistency_check.py beats
    python scripts/consistency_check.py birdnet
    python scripts/consistency_check.py ast       # needs transformers in the venv
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "kaggle_kernel"))
import submission as sub  # noqa: E402

CACHE = Path("~/scratch/birdclef-2026/features/exp300-deploy").expanduser()
CKPT_BEATS = Path("~/scratch/birdclef-2026/checkpoints/beats").expanduser()
VENDOR_BEATS = Path("~/arc/birdclef-2026/vendor/unilm/beats").expanduser()
# AST was extracted under a different flat dir (its cls focal cache); the rest live
# under CACHE/{encoder}/focal.
FOCAL_DIRS = {"ast": Path("~/scratch/birdclef-2026/features/exp300-ast").expanduser()}


def focal_dir(encoder: str) -> Path:
    return FOCAL_DIRS.get(encoder, CACHE / encoder)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_kernel_encoder(encoder: str):
    if encoder == "beats":
        md = Path(tempfile.mkdtemp(prefix="beats_md_"))
        (md / "beats").symlink_to(VENDOR_BEATS)
        for pt in CKPT_BEATS.glob("*.pt"):
            (md / pt.name).symlink_to(pt)
        return sub.BeatsEncoder(md)
    if encoder == "birdnet":
        # staged .tflite + labels (local-path load, no download)
        md = Path(os.environ.get(
            "BIRDNET_MODEL_DIR",
            str(EXP / "kaggle_kernel" / "birdnet_model_dataset"),
        ))
        return sub.BirdnetEncoder(md)
    if encoder == "ast":
        # AST now runs through ONNX: model_dir holds <name>.onnx + preprocessor_config.json
        # (built by scripts/export_ast_onnx.py). Validates the deployed ONNX forward, not
        # eager torch, against the eager-extracted cache (so this is the real gate).
        d = os.environ.get("AST_ONNX_DIR", str(EXP / "artifacts" / "ast_onnx"))
        return sub.AstEncoder(Path(d))
    raise SystemExit(f"unknown encoder {encoder}")


def main(encoder: str) -> None:
    cfg = yaml.safe_load((EXP / "configs/exp300.yaml").read_text())
    audio_dir = Path(cfg["focal"]["audio_dir"]).expanduser()

    shard = sorted((focal_dir(encoder) / "focal").glob("shard-*.parquet"))[0]
    df = pl.read_parquet(shard)
    print(f"[{encoder}] shard={shard.name} cols={df.columns} rows={len(df)}")

    # one file with several windows; keep parquet (= window) order
    fn = df["filename"].value_counts(sort=True).filter(pl.col("count") >= 3)["filename"][0]
    rows = df.filter(pl.col("filename") == fn)
    stored = np.stack(rows["embedding"].to_list()).astype(np.float64)
    audio_path = audio_dir / fn
    print(f"[{encoder}] file={fn} windows_stored={len(stored)} dim={stored.shape[1]} exists={audio_path.exists()}")

    enc = build_kernel_encoder(encoder)
    emb, _ = enc.encode_file(str(audio_path))
    emb = np.asarray(emb, dtype=np.float64)
    print(f"[{encoder}] kernel windows={len(emb)} dim={emb.shape[1]}")

    n = min(len(stored), len(emb))
    coss = np.array([cos(stored[i], emb[i]) for i in range(n)])
    print(f"[{encoder}] per-window cosine: min={coss.min():.6f} mean={coss.mean():.6f} max={coss.max():.6f} (n={n})")
    mean_cos = cos(stored[:n].mean(0), emb[:n].mean(0))
    print(f"[{encoder}] file-mean cosine: {mean_cos:.6f}")
    ok = coss.min() > 0.999
    print(f"[{encoder}] {'PASS' if ok else 'FAIL'} (threshold 0.999)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "beats")
