"""Format predictions into Kaggle submission CSV."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_submission(
    predictions: list[dict],
    sample_submission_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """Build submission CSV matching sample_submission.csv format.

    Args:
        predictions: List of dicts from predict_soundscape, each with row_id + species probs
        sample_submission_csv: Path to Kaggle sample_submission.csv (defines expected rows/columns)
        output_csv: Path to write output submission CSV

    Returns:
        Submission DataFrame
    """
    sample_sub = pd.read_csv(sample_submission_csv)
    expected_columns = list(sample_sub.columns)
    species_columns = [c for c in expected_columns if c != "row_id"]

    logger.info(f"Sample submission: {len(sample_sub)} rows, {len(species_columns)} species")

    pred_df = pd.DataFrame(predictions)

    # Merge with sample submission to ensure all expected rows exist
    submission = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")

    # Fill missing predictions with 0 (no detection)
    for col in species_columns:
        if col not in submission.columns:
            submission[col] = 0.0
        else:
            submission[col] = submission[col].fillna(0.0)

    # Reorder columns to match sample
    submission = submission[expected_columns]

    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    logger.info(f"Submission saved to {output_path} ({len(submission)} rows)")

    # Sanity checks
    missing = set(sample_sub["row_id"]) - set(pred_df.get("row_id", []))
    if missing:
        logger.warning(f"  {len(missing)} expected row_ids have no predictions (filled with 0)")

    extra = set(pred_df.get("row_id", [])) - set(sample_sub["row_id"])
    if extra:
        logger.warning(f"  {len(extra)} prediction row_ids not in sample_submission (ignored)")

    return submission
