"""Ridge (exp 106) Kaggle inference.

Loads the `ridge_model.npz` trained in exp 106, runs Perch v2 to extract
1536-d embeddings per 5s window, and applies `scores = embedding @ W` to
produce 234-species per-window predictions.

The ridge weight matrix `W` has shape (1536, 234) and no bias term. Species
ordering in the npz matches `birdclef_kaggle.species.build_species_list()`
(sorted, 234 entries). We do not apply sigmoid: Kaggle macro AUROC is
rank-based per species, and the OOF AUROC of 0.749 was measured on raw
`X @ W` scores in exp 106.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from birdclef_kaggle.audio import WINDOWS_PER_CLIP, load_clip
from birdclef_kaggle.perch_session import PerchSession
from birdclef_kaggle.submission import build_submission


def _iter_audio_files(audio_dir: Path) -> Iterable[Path]:
    for ext in ("*.ogg", "*.wav", "*.flac"):
        yield from sorted(audio_dir.glob(ext))


def load_ridge_weights(path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Return (W, species_list). W has shape (1536, 234)."""
    d = np.load(path)
    W = d["W"].astype(np.float32)
    species = [str(s) for s in d["species_list"]]
    if W.shape != (1536, 234):
        raise ValueError(f"Unexpected ridge W shape: {W.shape}")
    if len(species) != 234:
        raise ValueError(f"Unexpected species count: {len(species)}")
    return W, species


def predict(
    audio_dir: Path,
    ridge_weights_path: Path,
    output_path: Path,
    sample_submission_path: Path | None = None,
    perch_model_path: Path | None = None,
    perch_batch_size: int = 32,
) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    files = list(_iter_audio_files(audio_dir))
    if not files:
        raise FileNotFoundError(f"No audio files under {audio_dir}")

    W, species = load_ridge_weights(ridge_weights_path)
    perch = PerchSession(model_path=perch_model_path)

    all_row_ids: list[str] = []
    all_scores: list[np.ndarray] = []

    for audio_path in files:
        clip = load_clip(audio_path)  # (12, 160000) float32
        stem = audio_path.stem

        emb_batches = []
        for start in range(0, clip.windows.shape[0], perch_batch_size):
            batch = clip.windows[start : start + perch_batch_size]
            out = perch.run(batch, outputs=["embedding"])
            emb_batches.append(out["embedding"])
        embeddings = np.concatenate(emb_batches, axis=0)  # (12, 1536)

        scores = embeddings @ W  # (12, 234) float32
        all_scores.append(scores)
        all_row_ids.extend(f"{stem}_{end_s}" for end_s in clip.end_seconds)

        if len(all_scores) % 25 == 0:
            print(f"  processed {len(all_scores)} / {len(files)} files", flush=True)

    predictions = np.concatenate(all_scores, axis=0)  # (N*12, 234)
    if predictions.shape[0] != len(all_row_ids):
        raise RuntimeError(
            f"Row/score count mismatch: rows={len(all_row_ids)} scores={predictions.shape[0]}"
        )
    if predictions.shape[0] != len(files) * WINDOWS_PER_CLIP:
        raise RuntimeError(
            f"Expected {len(files) * WINDOWS_PER_CLIP} rows, got {predictions.shape[0]}"
        )

    return build_submission(
        row_ids=all_row_ids,
        predictions=predictions,
        species=species,
        sample_submission_path=sample_submission_path,
        output_path=output_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--ridge-weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, default=None)
    p.add_argument("--perch-model", type=Path, default=None, help="Optional local Perch ONNX path")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    df = predict(
        audio_dir=args.audio_dir,
        ridge_weights_path=args.ridge_weights,
        output_path=args.output,
        sample_submission_path=args.sample_submission,
        perch_model_path=args.perch_model,
        perch_batch_size=args.batch_size,
    )
    print(f"submission shape: {df.shape}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
