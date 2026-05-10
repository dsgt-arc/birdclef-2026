"""Recording-level motif review plots for Marbled White-lipped Frog.

Runs a small fixed detector bank over every `22967` train recording and renders
one standardized overlay plot per recording. This is for visual review of whether
primary/secondary/chip proposals help, not for final ground-truth annotation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import polars as pl

from data_analysis.common.paths import TRAIN_AUDIO_DIR, artifact_dir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)
from data_analysis.marbled_frog.motif_boxes import MotifBox, detect_custom_band_boxes

TARGET_LABEL = "22967"


@dataclass(frozen=True)
class ReviewDetector:
    """A detector configuration for review plots."""

    name: str
    motif_family_hint: str
    band: tuple[float, float]
    color: str
    linestyle: str
    peak_height: float = 0.40
    peak_prominence: float = 0.12
    lower_pad_hz: float = 0.0
    upper_pad_hz: float = 0.0


DETECTORS = [
    ReviewDetector(
        name="low_primary_600_1300",
        motif_family_hint="low_primary_or_low_tail",
        band=(600.0, 1300.0),
        color="lime",
        linestyle="-",
        lower_pad_hz=70.0,
    ),
    ReviewDetector(
        name="mid_chip_1000_1800",
        motif_family_hint="rising_chip",
        band=(1000.0, 1800.0),
        color="yellow",
        linestyle="-",
        lower_pad_hz=80.0,
        upper_pad_hz=80.0,
    ),
    ReviewDetector(
        name="broad_upper_1000_3000",
        motif_family_hint="secondary_or_broad_upper",
        band=(1000.0, 3000.0),
        color="cyan",
        linestyle="--",
        lower_pad_hz=120.0,
    ),
]

# Only use count priors for recordings already inspected manually. Everything
# else is left as unconstrained proposal generation.
EXPECTED_COUNTS: dict[tuple[str, str], int] = {
    ("22967/iNat1448422.ogg", "low_primary_600_1300"): 4,
    ("22967/iNat1448422.ogg", "broad_upper_1000_3000"): 4,
    ("22967/iNat1697350.ogg", "low_primary_600_1300"): 10,
    ("22967/iNat1697350.ogg", "broad_upper_1000_3000"): 10,
    ("22967/iNat1698104.ogg", "mid_chip_1000_1800"): 5,
}

# Per-recording review policy from visual inspection. Detectors are proposal
# channels, not biological classes; once a channel is visually irrelevant for a
# recording, suppress it to keep review plots interpretable.
DETECTOR_POLICY: dict[str, set[str]] = {
    # Visual review: green low-primary is good, cyan broad-upper/secondary is
    # fine, yellow mid-chip proposals are not a real motif family here.
    "22967/iNat1697350.ogg": {"low_primary_600_1300", "broad_upper_1000_3000"},
}


def recording_files() -> list[str]:
    """Return all Marbled Frog train recording filenames."""
    return [f"{TARGET_LABEL}/{path.name}" for path in sorted((TRAIN_AUDIO_DIR / TARGET_LABEL).glob("*.ogg"))]


def boxes_to_rows(filename: str, detector: ReviewDetector, boxes: list[MotifBox]) -> list[dict]:
    """Convert detector boxes to table rows."""
    rows = []
    for box in boxes:
        rows.append(
            {
                "filename": filename,
                "detector": detector.name,
                "motif_family_hint": detector.motif_family_hint,
                "motif_id": box.motif_id,
                "component": box.component,
                "start_sec": box.start_sec,
                "end_sec": box.end_sec,
                "duration_sec": box.end_sec - box.start_sec,
                "fmin_hz": box.fmin_hz,
                "fmax_hz": box.fmax_hz,
                "bandwidth_hz": box.fmax_hz - box.fmin_hz,
                "peak_time_sec": box.peak_time_sec,
                "peak_score": box.peak_score,
                "expected_count_used": EXPECTED_COUNTS.get((filename, detector.name)),
            }
        )
    return rows


def active_detectors(filename: str) -> list[ReviewDetector]:
    """Return detectors enabled for a recording."""
    allowed = DETECTOR_POLICY.get(filename)
    if allowed is None:
        return DETECTORS
    return [detector for detector in DETECTORS if detector.name in allowed]


def detect_recording(filename: str) -> tuple[list[dict], object, object, object]:
    """Run the detector bank for one recording."""
    audio, _ = librosa.load(TRAIN_AUDIO_DIR / filename, sr=MEL_SAMPLE_RATE, mono=True)
    db_band, z, freqs, times = robust_excess_spectrogram(audio)
    rows = []
    for detector in active_detectors(filename):
        boxes = detect_custom_band_boxes(
            z,
            freqs,
            times,
            band=detector.band,
            component=detector.name,
            peak_height=detector.peak_height,
            peak_prominence=detector.peak_prominence,
            lower_pad_hz=detector.lower_pad_hz,
            upper_pad_hz=detector.upper_pad_hz,
            expected_count=EXPECTED_COUNTS.get((filename, detector.name)),
        )
        rows.extend(boxes_to_rows(filename, detector, boxes))
    return rows, db_band, freqs, times


def plot_recording_review(
    filename: str,
    rows: list[dict],
    db_band,
    freqs,
    times,
    output_path: Path,
) -> None:
    """Plot all detector proposals over a full-recording spectrogram."""
    width = max(11.0, min(20.0, 0.38 * float(times[-1]) + 7.5))
    fig, ax = plt.subplots(figsize=(width, 6.2), dpi=180)
    librosa.display.specshow(
        db_band,
        x_coords=times,
        y_coords=freqs,
        x_axis="time",
        y_axis="linear",
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        ax=ax,
        cmap="magma",
    )

    detectors = active_detectors(filename)
    detector_map = {detector.name: detector for detector in detectors}
    for detector in detectors:
        ax.axhspan(
            detector.band[0],
            detector.band[1],
            color=detector.color,
            alpha=0.045,
            label=f"{detector.name} band",
        )

    for row in rows:
        detector = detector_map[row["detector"]]
        ax.add_patch(
            Rectangle(
                (row["start_sec"], row["fmin_hz"]),
                row["end_sec"] - row["start_sec"],
                row["fmax_hz"] - row["fmin_hz"],
                fill=False,
                edgecolor=detector.color,
                linewidth=1.8,
                linestyle=detector.linestyle,
            )
        )
        ax.text(
            row["start_sec"],
            row["fmax_hz"] + 35,
            f"{row['motif_id']}",
            color=detector.color,
            fontsize=7,
            weight="bold",
        )

    counts = pl.DataFrame(rows).group_by("detector").len().sort("detector") if rows else pl.DataFrame()
    count_text = ", ".join(
        f"{name.replace('_', ' ')}={n}" for name, n in zip(counts["detector"].to_list(), counts["len"].to_list())
    ) if not counts.is_empty() else "no boxes"
    ax.set_ylim(DISPLAY_BAND)
    ax.set_title(
        f"Marbled Frog motif review: {filename}\n"
        f"{count_text}; green=low primary, yellow=mid chip, cyan=broad upper/secondary"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run(output_dir: Path | None = None) -> Path:
    """Run review plot generation."""
    output_dir = output_dir or artifact_dir("marbled_frog/motif_review")
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for filename in recording_files():
        rows, db_band, freqs, times = detect_recording(filename)
        all_rows.extend(rows)
        plot_recording_review(
            filename,
            rows,
            db_band,
            freqs,
            times,
            plots_dir / f"{Path(filename).stem}_motif_review.png",
        )

    boxes = pl.DataFrame(all_rows)
    boxes.write_csv(tables_dir / "review_boxes.csv")
    summary = boxes.group_by("filename", "detector", "motif_family_hint").agg(
        pl.len().alias("n_boxes"),
        pl.col("duration_sec").median().alias("median_duration_sec"),
        pl.col("fmin_hz").median().alias("median_fmin_hz"),
        pl.col("fmax_hz").median().alias("median_fmax_hz"),
        pl.col("peak_score").median().alias("median_peak_score"),
    ).sort("filename", "detector")
    summary.write_csv(tables_dir / "review_summary.csv")

    (output_dir / "summary.md").write_text(
        f"""# Marbled Frog Motif Review

Generated standardized full-recording review plots for all `{TARGET_LABEL}` train recordings using a fixed detector bank:

- `low_primary_600_1300` — green boxes, low primary/tail proposals.
- `mid_chip_1000_1800` — yellow boxes, rising chip proposals.
- `broad_upper_1000_3000` — cyan dashed boxes, secondary/broad upper proposals.

Known count priors are only applied to already inspected recordings:

```text
{EXPECTED_COUNTS}
```

Artifacts:

- `plots/*_motif_review.png` — one review overlay per recording.
- `tables/review_boxes.csv` — all proposal boxes.
- `tables/review_summary.csv` — counts and median box features by recording/detector.

Summary:

```text
{summary.write_csv(separator='|').strip()}
```
"""
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = run(args.output_dir)
    print(f"Wrote Marbled Frog motif review to {output_dir}")


if __name__ == "__main__":
    main()
