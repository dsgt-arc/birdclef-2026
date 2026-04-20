"""Submission CSV writer aligned against Kaggle's sample_submission.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_submission(
    row_ids: list[str],
    predictions: np.ndarray,
    species: list[str],
    sample_submission_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Assemble a submission DataFrame and optionally write it to disk.

    Args:
        row_ids: Length-N list of row IDs (same order as predictions).
        predictions: Shape (N, 234) float probabilities in the species order.
        species: Length-234 ordered species list (columns of the submission).
        sample_submission_path: Optional sample_submission.csv. If provided,
            the output is reindexed to match the sample's row_id order and
            column order, with any missing rows filled by 1/234.
        output_path: If provided, write the final DataFrame to this path.

    Returns:
        The submission DataFrame indexed by row_id with 234 species columns.
    """
    if predictions.ndim != 2:
        raise ValueError(f"predictions must be 2-D, got shape {predictions.shape}")
    if predictions.shape[0] != len(row_ids):
        raise ValueError(
            f"row_ids length {len(row_ids)} != predictions rows {predictions.shape[0]}"
        )
    if predictions.shape[1] != len(species):
        raise ValueError(
            f"species length {len(species)} != predictions cols {predictions.shape[1]}"
        )

    df = pd.DataFrame(predictions, columns=species)
    df.insert(0, "row_id", row_ids)

    if sample_submission_path is not None:
        sample = pd.read_csv(sample_submission_path)
        sample_species = [c for c in sample.columns if c != "row_id"]
        if set(sample_species) != set(species):
            missing = set(sample_species) - set(species)
            extra = set(species) - set(sample_species)
            raise ValueError(
                f"species set mismatch with sample_submission. "
                f"missing in predictions: {sorted(missing)[:5]}..., "
                f"extra in predictions: {sorted(extra)[:5]}..."
            )
        default = 1.0 / len(sample_species)
        out = sample[["row_id"]].merge(df, on="row_id", how="left")
        out[sample_species] = out[sample_species].fillna(default)
        out = out[["row_id"] + sample_species]
        df = out

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
