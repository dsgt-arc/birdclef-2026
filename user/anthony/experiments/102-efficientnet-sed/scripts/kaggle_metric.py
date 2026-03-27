"""BirdCLEF ROC-AUC metric — reference implementation from Kaggle.

Source: https://www.kaggle.com/code/metric/birdclef-roc-auc

Key: macro-averaged ROC-AUC that IGNORES classes with no positive labels.
Only species that actually appear in the ground truth contribute to the score.
"""

import numpy as np
import sklearn.metrics


def birdclef_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute BirdCLEF macro-averaged ROC-AUC, excluding absent species.

    Args:
        y_true: (N, C) binary ground truth labels
        y_score: (N, C) predicted probabilities

    Returns:
        Macro-averaged ROC-AUC over species with at least one positive sample.
    """
    # Only score columns (species) that have at least one positive label
    col_sums = y_true.sum(axis=0)
    scored_cols = np.where(col_sums > 0)[0]

    if len(scored_cols) == 0:
        return 0.0

    return sklearn.metrics.roc_auc_score(
        y_true[:, scored_cols],
        y_score[:, scored_cols],
        average="macro",
    )
