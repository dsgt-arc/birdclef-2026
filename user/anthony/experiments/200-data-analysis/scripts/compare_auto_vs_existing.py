#!/usr/bin/env python
"""A/B compare auto motif mining against the manual Marbled Frog pipeline.

Reads ``consolidated_events.csv`` from both pipelines plus the curated boxes,
and writes ``ab_vs_existing.csv`` to the auto pipeline's artifact tree with:
- per-recording event counts (auto vs manual) and absolute difference;
- pairwise nearest-peak match within ``--tolerance`` seconds;
- per-component curated recovery from both pipelines (only species with curated
  ground truth — defaults to Marbled Frog).

Run via::

    python scripts/compare_auto_vs_existing.py --label 22967
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from data_analysis.common.curated_eval import curated_recovery
from data_analysis.auto_motif.mining import LABEL_TO_DIR

DEFAULT_LABEL = "22967"
DEFAULT_CURATED = Path("artifacts/marbled_frog/motif_boxes/curated/tables/curated_motif_boxes.csv")
DEFAULT_AUTO = "artifacts/{species_dir}/motif_mining_auto/tables/consolidated_events.csv"
DEFAULT_MANUAL = "artifacts/{species_dir}/motif_template_calibration/tables/consolidated_events.csv"


def _agreement(auto: pl.DataFrame, manual: pl.DataFrame, tolerance: float) -> pl.DataFrame:
    """Per-recording event counts + nearest-peak match rates."""
    files = sorted(set(auto["filename"].to_list()) | set(manual["filename"].to_list()))
    rows: list[dict] = []
    for filename in files:
        a = auto.filter(pl.col("filename") == filename).sort("peak_time_sec")
        m = manual.filter(pl.col("filename") == filename).sort("peak_time_sec")
        a_times = a["peak_time_sec"].to_numpy()
        m_times = m["peak_time_sec"].to_numpy()
        a_matched = 0
        for t in a_times:
            if m_times.size and float(np.min(np.abs(m_times - t))) <= tolerance:
                a_matched += 1
        m_matched = 0
        for t in m_times:
            if a_times.size and float(np.min(np.abs(a_times - t))) <= tolerance:
                m_matched += 1
        rows.append({
            "filename": filename,
            "n_auto": int(a.height),
            "n_manual": int(m.height),
            "abs_diff": int(abs(a.height - m.height)),
            "n_auto_matched_by_manual": a_matched,
            "n_manual_matched_by_auto": m_matched,
            "auto_precision_vs_manual": float(a_matched / a.height) if a.height else float("nan"),
            "auto_recall_vs_manual": float(m_matched / m.height) if m.height else float("nan"),
        })
    return pl.DataFrame(rows)


def _per_component_recovery(events: pl.DataFrame, curated_path: Path, label: str) -> pl.DataFrame:
    recovery = curated_recovery(events, curated_path)
    if recovery.is_empty() or "component" not in recovery.columns:
        return pl.DataFrame()
    summary = recovery.group_by("component").agg(
        pl.len().alias("n_curated_boxes"),
        pl.col("recovered").sum().alias("n_recovered"),
        pl.col("recovered").mean().alias("recovery_rate"),
    ).sort("component")
    return summary.with_columns(pl.lit(label).alias("pipeline"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--curated", type=Path, default=DEFAULT_CURATED)
    parser.add_argument("--auto", type=Path, default=None,
                        help="Auto consolidated_events.csv (default: by label)")
    parser.add_argument("--manual", type=Path, default=None,
                        help="Manual consolidated_events.csv (default: by label)")
    parser.add_argument("--tolerance", type=float, default=0.25,
                        help="Seconds for nearest-peak match")
    args = parser.parse_args()

    species_dir = LABEL_TO_DIR.get(args.label, args.label)
    auto_path = args.auto or Path(DEFAULT_AUTO.format(species_dir=species_dir))
    manual_path = args.manual or Path(DEFAULT_MANUAL.format(species_dir=species_dir))
    auto = pl.read_csv(auto_path)
    manual = pl.read_csv(manual_path)

    agreement = _agreement(auto, manual, args.tolerance)
    auto_recovery = _per_component_recovery(auto, args.curated, "auto")
    manual_recovery = _per_component_recovery(manual, args.curated, "manual")
    recovery_combined = (
        pl.concat([auto_recovery, manual_recovery], how="diagonal_relaxed")
        if not auto_recovery.is_empty() and not manual_recovery.is_empty()
        else (auto_recovery if not auto_recovery.is_empty() else manual_recovery)
    )

    out_dir = auto_path.parent
    agreement.write_csv(out_dir / "ab_vs_existing.csv")
    if not recovery_combined.is_empty():
        recovery_combined.write_csv(out_dir / "ab_curated_recovery_by_component.csv")

    auto_total_recovery = float(auto_recovery["recovery_rate"].mean()) if not auto_recovery.is_empty() else float("nan")
    manual_total_recovery = float(manual_recovery["recovery_rate"].mean()) if not manual_recovery.is_empty() else float("nan")

    print(f"A/B summary for label {args.label}:")
    print(f"  Auto events:           {auto.height}")
    print(f"  Manual events:         {manual.height}")
    print(f"  Median |abs_diff|:     {float(agreement['abs_diff'].median())}")
    print(f"  Max  |abs_diff|:       {int(agreement['abs_diff'].max())}")
    print(f"  Auto precision:        {float(agreement['auto_precision_vs_manual'].mean()):.3f}")
    print(f"  Auto recall:           {float(agreement['auto_recall_vs_manual'].mean()):.3f}")
    print(f"  Auto curated recovery: {auto_total_recovery:.3f}")
    print(f"  Manual recovery:       {manual_total_recovery:.3f}")
    print(f"  Wrote: {out_dir / 'ab_vs_existing.csv'}")
    if not recovery_combined.is_empty():
        print(f"  Wrote: {out_dir / 'ab_curated_recovery_by_component.csv'}")


if __name__ == "__main__":
    main()
