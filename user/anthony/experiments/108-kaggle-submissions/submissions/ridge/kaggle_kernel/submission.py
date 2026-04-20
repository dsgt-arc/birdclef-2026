#!/usr/bin/env python3
"""Kaggle kernel entrypoint: BirdCLEF 2026 ridge submission (exp 106).

Runs offline on a Kaggle CPU kernel (no internet, no GPU). Pipeline per
60-second test soundscape:

  1. load_clip -> (12, 160000) float32 @ 32 kHz
  2. Perch v2 ONNX session -> (12, 1536) embedding
  3. ridge: scores = embedding @ W, W = (1536, 234)
  4. emit 12 rows with row_id = f"{stem}_{end_s}" for end_s in 5..60

Inputs (Kaggle datasets, attached via kernel-metadata.json):
  - /kaggle/input/birdclef-2026-deps/              wheels: onnxruntime, soundfile, numpy, pandas
  - /kaggle/input/birdclef-2026-weights-ridge/     perch_v2_no_dft.onnx, ridge_model.npz,
                                                   perch_species_mapping.json,
                                                   birdclef_kaggle/*.py (vendored)

Competition data (mounted read-only by Kaggle):
  - /kaggle/input/birdclef-2026/test_soundscapes/*.ogg
  - /kaggle/input/birdclef-2026/sample_submission.csv

Output:
  - /kaggle/working/submission.csv
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

KAGGLE_USER = "acmiyaguchi"


def _find_input(slug: str) -> str:
    for c in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}"):
        if os.path.exists(c):
            return c
    return f"/kaggle/input/{slug}"


DEPS_DIR = _find_input("birdclef-2026-deps")
WEIGHTS_DIR = _find_input("birdclef-2026-weights-ridge")
COMPETITION_DIR = (
    "/kaggle/input/competitions/birdclef-2026"
    if os.path.exists("/kaggle/input/competitions/birdclef-2026")
    else "/kaggle/input/birdclef-2026"
)

if os.path.exists(DEPS_DIR):
    print(f"Installing deps from {DEPS_DIR}", flush=True)
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--no-index", "--find-links", DEPS_DIR,
            "onnxruntime", "soundfile", "numpy", "pandas",
        ]
    )

sys.path.insert(0, WEIGHTS_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from birdclef_kaggle.audio import load_clip  # noqa: E402
from birdclef_kaggle.perch_session import PerchSession  # noqa: E402
from birdclef_kaggle.submission import build_submission  # noqa: E402


def _load_ridge_weights(path: str | Path) -> tuple[np.ndarray, list[str]]:
    d = np.load(path)
    W = d["W"].astype(np.float32)
    species = [str(s) for s in d["species_list"]]
    if W.shape != (1536, 234):
        raise ValueError(f"Unexpected ridge W shape: {W.shape}")
    if len(species) != 234:
        raise ValueError(f"Unexpected species count: {len(species)}")
    return W, species


def main() -> None:
    perch_path = Path(WEIGHTS_DIR) / "perch_v2_no_dft.onnx"
    ridge_path = Path(WEIGHTS_DIR) / "ridge_model.npz"
    sample_submission = Path(COMPETITION_DIR) / "sample_submission.csv"
    test_dir = Path(COMPETITION_DIR) / "test_soundscapes"
    out_path = Path("/kaggle/working/submission.csv")

    print(f"Perch ONNX: {perch_path}", flush=True)
    print(f"Ridge weights: {ridge_path}", flush=True)
    print(f"Test dir: {test_dir}", flush=True)

    W, species = _load_ridge_weights(ridge_path)
    perch = PerchSession(model_path=perch_path)

    audio_files = sorted(test_dir.glob("*.ogg"))
    print(f"Found {len(audio_files)} test soundscapes", flush=True)

    batch_size = 12
    row_ids: list[str] = []
    scores_chunks: list[np.ndarray] = []

    for i, audio_path in enumerate(audio_files):
        clip = load_clip(audio_path)
        emb = perch.run(clip.windows, outputs=["embedding"])["embedding"]
        scores = emb @ W
        scores_chunks.append(scores)
        stem = audio_path.stem
        row_ids.extend(f"{stem}_{end_s}" for end_s in clip.end_seconds)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1} / {len(audio_files)}", flush=True)

    predictions = np.concatenate(scores_chunks, axis=0) if scores_chunks else np.zeros((0, 234), dtype=np.float32)

    df = build_submission(
        row_ids=row_ids,
        predictions=predictions,
        species=species,
        sample_submission_path=sample_submission if sample_submission.exists() else None,
        output_path=out_path,
    )
    print(f"Wrote submission: {df.shape} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
