"""Data reshaping and metadata utilities for file-level operations.

Cross-ref: ref_yuriy_protossm_v5.py lines 1598-1658.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_WINDOWS = 12


def reshape_to_files(
    flat_array: np.ndarray,
    meta_df: pd.DataFrame,
    n_windows: int = N_WINDOWS,
) -> tuple[np.ndarray, list[str]]:
    """Reshape flat window-level arrays to (n_files, n_windows, ...) grouped by filename.

    Args:
        flat_array: (N, ...) window-level data
        meta_df: DataFrame with 'filename' column, ordered by file then window
        n_windows: windows per file (12)

    Returns:
        file_array: (n_files, n_windows, ...) reshaped data
        file_list: ordered list of unique filenames
    """
    assert len(flat_array) % n_windows == 0
    file_list = list(dict.fromkeys(meta_df["filename"].tolist()))
    n_files = len(file_list)
    assert n_files * n_windows == len(
        flat_array
    ), f"Expected {n_files}*{n_windows}={n_files * n_windows} rows, got {len(flat_array)}"

    rest_shape = flat_array.shape[1:]
    return flat_array.reshape(n_files, n_windows, *rest_shape), file_list


def build_site_mapping(
    meta_df: pd.DataFrame,
    n_sites_max: int = 20,
) -> tuple[dict[str, int], int]:
    """Map unique sites to integer indices.

    Returns:
        site_to_idx: mapping from site name to integer index
        n_sites: number of unique sites (capped at n_sites_max)
    """
    sites = meta_df["site"].dropna().unique().tolist() if "site" in meta_df.columns else []
    sites = sorted({str(s) for s in sites})
    site_to_idx = {s: min(i, n_sites_max - 1) for i, s in enumerate(sites)}
    return site_to_idx, min(len(sites), n_sites_max)


def get_file_metadata(
    meta_df: pd.DataFrame,
    file_list: list[str],
    site_to_idx: dict[str, int],
    n_sites_max: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-file site_ids and hour_ids arrays.

    Returns:
        site_ids: (n_files,) integer site indices
        hour_ids: (n_files,) integer hours (0-23)
    """
    # Build lookup from filename to first row
    file_to_row = {}
    for i, row in meta_df.iterrows():
        fn = row["filename"]
        if fn not in file_to_row:
            file_to_row[fn] = i

    sites = meta_df["site"].values if "site" in meta_df.columns else np.full(len(meta_df), None)
    hours = meta_df["hour"].values if "hour" in meta_df.columns else np.zeros(len(meta_df))

    site_ids = np.zeros(len(file_list), dtype=np.int64)
    hour_ids = np.zeros(len(file_list), dtype=np.int64)

    for fi, fname in enumerate(file_list):
        row_idx = file_to_row.get(fname)
        if row_idx is not None:
            s = sites[row_idx]
            sid = site_to_idx.get(str(s), 0) if s is not None else 0
            site_ids[fi] = min(sid, n_sites_max - 1)
            h = hours[row_idx]
            hour_ids[fi] = (
                int(h) % 24 if h is not None and not (isinstance(h, float) and np.isnan(h)) else 0
            )

    return site_ids, hour_ids


def build_taxonomy_groups(
    taxonomy_df: pd.DataFrame,
    primary_labels: list[str],
) -> tuple[int, list[int], dict[str, int]]:
    """Build family grouping for auxiliary taxonomic head.

    Returns:
        n_families: number of unique families
        class_to_family: list mapping class index to family index
        family_to_idx: mapping from family name to integer index
    """
    # Get family for each species
    label_to_family = {}
    if "primary_label" in taxonomy_df.columns:
        for _, row in taxonomy_df.iterrows():
            label = row.get("primary_label", "")
            sci_name = str(row.get("scientific_name", ""))
            # Use genus (first word of scientific name) as family proxy
            family = sci_name.split()[0] if sci_name else "unknown"
            label_to_family[label] = family

    # Build family index
    families = sorted({label_to_family.get(lbl, "unknown") for lbl in primary_labels})
    family_to_idx = {f: i for i, f in enumerate(families)}

    class_to_family = [
        family_to_idx.get(label_to_family.get(lbl, "unknown"), 0) for lbl in primary_labels
    ]

    return len(families), class_to_family, family_to_idx


def build_file_families(
    labels_files: np.ndarray,
    class_to_family: list[int],
    n_families: int,
) -> np.ndarray:
    """Build per-file multi-hot family label matrix.

    Args:
        labels_files: (n_files, n_windows, n_classes)
        class_to_family: list mapping class index to family index
        n_families: total number of families

    Returns:
        file_families: (n_files, n_families) multi-hot family labels
    """
    n_files = labels_files.shape[0]
    # Aggregate labels across windows: any positive window counts
    file_labels = labels_files.max(axis=1)  # (n_files, n_classes)

    file_families = np.zeros((n_files, n_families), dtype=np.float32)
    for cls_idx, fam_idx in enumerate(class_to_family):
        mask = file_labels[:, cls_idx] > 0.5
        file_families[mask, fam_idx] = 1.0

    return file_families
