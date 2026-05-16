"""Curated motif-box recovery evaluation, shared across motif pipelines.

Lifted from ``marbled_frog.motif_template_calibration`` so the manual and auto
pipelines compute curated recovery with identical semantics. The manual
pipeline re-exports these names for backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return float(inter / union) if union > 0 else 0.0


def curated_recovery(events: pl.DataFrame, curated_path: Path) -> pl.DataFrame:
    """Compare motif-bank detections against curated motif-box annotations."""
    if not curated_path.exists() or events.is_empty():
        return pl.DataFrame()
    curated = pl.read_csv(curated_path)
    rows = []
    for idx, cur in enumerate(curated.iter_rows(named=True), start=1):
        ev = events.filter(pl.col("filename") == cur["filename"])
        best = None
        for det in ev.iter_rows(named=True):
            time_iou = interval_iou(cur["start_sec"], cur["end_sec"], det["start_sec"], det["end_sec"])
            freq_iou = interval_iou(cur["fmin_hz"], cur["fmax_hz"], det["fmin_hz"], det["fmax_hz"])
            peak_inside = cur["start_sec"] <= det["peak_time_sec"] <= cur["end_sec"]
            score = max(time_iou, 1.0 if peak_inside else 0.0) * max(freq_iou, 0.05)
            if best is None or score > best[0]:
                best = (score, time_iou, freq_iou, peak_inside, det)
        if best is None:
            # Keep the schema stable so an all-miss frame (0 recovery) still
            # carries the best_* columns the recovery summary aggregates.
            rows.append(
                {
                    "curated_row_id": idx,
                    **cur,
                    "recovered": False,
                    "best_motif_cluster": None,
                    "best_detection_peak_time_sec": None,
                    "best_detection_score": None,
                    "best_time_iou": None,
                    "best_freq_iou": None,
                    "best_peak_inside_curated": None,
                }
            )
        else:
            _score, time_iou, freq_iou, peak_inside, det = best
            recovered = bool((peak_inside or time_iou >= 0.20) and freq_iou >= 0.10)
            rows.append(
                {
                    "curated_row_id": idx,
                    **cur,
                    "recovered": recovered,
                    "best_motif_cluster": det["motif_cluster"],
                    "best_detection_peak_time_sec": det["peak_time_sec"],
                    "best_detection_score": det["score"],
                    "best_time_iou": time_iou,
                    "best_freq_iou": freq_iou,
                    "best_peak_inside_curated": peak_inside,
                }
            )
    return pl.DataFrame(rows)


def detection_stage_metrics(
    raw_events: pl.DataFrame,
    nms_events_df: pl.DataFrame,
    consolidated: pl.DataFrame,
    raw_recovery: pl.DataFrame,
    nms_recovery: pl.DataFrame,
    consolidated_recovery: pl.DataFrame,
) -> pl.DataFrame:
    """Quantify whether NMS/consolidation reduces duplicates while preserving recovery."""
    def recovery_rate(df: pl.DataFrame) -> float:
        return float(df["recovered"].mean()) if not df.is_empty() and "recovered" in df.columns else float("nan")

    def median_components(df: pl.DataFrame) -> float:
        return float(df["n_component_detections"].median()) if not df.is_empty() and "n_component_detections" in df.columns else float("nan")

    raw_n = raw_events.height
    nms_n = nms_events_df.height
    cons_n = consolidated.height
    return pl.DataFrame(
        [
            {
                "stage": "raw_template_peaks",
                "n_boxes_or_events": raw_n,
                "reduction_vs_raw": 0.0,
                "curated_recovery_rate": recovery_rate(raw_recovery),
                "median_components_per_event": float("nan"),
            },
            {
                "stage": "within_family_nms",
                "n_boxes_or_events": nms_n,
                "reduction_vs_raw": 1.0 - nms_n / raw_n if raw_n else float("nan"),
                "curated_recovery_rate": recovery_rate(nms_recovery),
                "median_components_per_event": float("nan"),
            },
            {
                "stage": "cross_family_consolidated_events",
                "n_boxes_or_events": cons_n,
                "reduction_vs_raw": 1.0 - cons_n / raw_n if raw_n else float("nan"),
                "curated_recovery_rate": recovery_rate(consolidated_recovery),
                "median_components_per_event": median_components(consolidated),
            },
        ]
    )
