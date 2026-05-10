"""Export current-best manual Marbled Frog motif annotations.

This copies the visually selected/current-best motif-box tables from the
motif-box workflow into a curated review folder, builds a combined CSV, and
renders standardized validation plots with primary/secondary bands.

Run after regenerating:

    python -m data_analysis.marbled_frog.motif_boxes
    python -m data_analysis.marbled_frog.manual_annotations
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from data_analysis.common.paths import TRAIN_AUDIO_DIR, artifact_dir, fresh_subdir
from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    robust_excess_spectrogram,
)


@dataclass(frozen=True)
class ManualAnnotationSet:
    """A curated motif annotation set to export."""

    annotation_set: str
    motif_family: str
    status: str
    notes: str
    source_table: Path
    source_plot: Path
    primary_band: tuple[float, float]
    secondary_band: tuple[float, float] | None
    output_plot_name: str


def annotation_sets(source_dir: Path) -> list[ManualAnnotationSet]:
    """Return the current curated manual motif annotation sets."""
    return [
        ManualAnnotationSet(
            annotation_set="iNat1448422_0-5s_low_click_secondary",
            motif_family="low_click_with_secondary",
            status="manual_current_best",
            notes=(
                "Four short calls in 0-5s seed window; primary around "
                "600-1300 Hz with secondary validation band around 1500-2300 Hz."
            ),
            source_table=source_dir / "tables" / "iNat1448422_0-5s_motif_boxes.csv",
            source_plot=source_dir / "plots" / "iNat1448422_0-5s_motif_boxes.png",
            primary_band=(600.0, 1300.0),
            secondary_band=(1500.0, 2300.0),
            output_plot_name="iNat1448422_0-5s_manual_motif_boxes.png",
        ),
        ManualAnnotationSet(
            annotation_set="iNat1697350_independent_broad_sweep",
            motif_family="broad_upper_sweep_with_low_tail",
            status="manual_current_best",
            notes=(
                "Ten paired-looking events; independent primary lower boxes plus "
                "broad secondary 1000-3000 Hz boxes. Treat as motif-event evidence, "
                "not precise boxes."
            ),
            source_table=source_dir
            / "template_matches"
            / "tables"
            / "iNat1697350_independent_band_motif_boxes.csv",
            source_plot=source_dir
            / "template_matches"
            / "plots"
            / "iNat1697350_independent_band_motif_boxes.png",
            primary_band=(600.0, 1300.0),
            secondary_band=(1000.0, 3000.0),
            output_plot_name="iNat1697350_manual_motif_boxes.png",
        ),
        ManualAnnotationSet(
            annotation_set="iNat1698104_1k_1p8k_rising_chips",
            motif_family="rising_chip_1k_1p8k",
            status="manual_current_best",
            notes=(
                "Five rising chips around 1-1.8 kHz; separate motif family not "
                "matched by the 1448422 seed template."
            ),
            source_table=source_dir
            / "template_matches"
            / "tables"
            / "iNat1698104_1k_1p8k_chip_boxes.csv",
            source_plot=source_dir
            / "template_matches"
            / "plots"
            / "iNat1698104_1k_1p8k_chip_boxes.png",
            primary_band=(1000.0, 1800.0),
            secondary_band=None,
            output_plot_name="iNat1698104_manual_motif_boxes.png",
        ),
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dictionaries."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    """Parse a float from a row."""
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def is_primary_component(component: str) -> bool:
    """Return whether a component should be styled as primary."""
    return "primary" in component or "chip" in component


def is_secondary_component(component: str) -> bool:
    """Return whether a component should be styled as secondary."""
    return "secondary" in component


def render_standard_plot(item: ManualAnnotationSet, rows: list[dict[str, str]], output_path: Path) -> None:
    """Render a standardized primary/secondary validation-band plot."""
    if not rows:
        return
    filename = rows[0]["filename"]
    window_start = min(row_float(row, "window_start_sec") for row in rows)
    window_end = max(row_float(row, "window_end_sec", row_float(row, "end_sec")) for row in rows)
    duration = max(0.1, window_end - window_start)

    audio, _ = librosa.load(
        TRAIN_AUDIO_DIR / filename,
        sr=MEL_SAMPLE_RATE,
        mono=True,
        offset=window_start,
        duration=duration,
    )
    db_band, _z, freqs, times = robust_excess_spectrogram(audio)
    display_times = times + window_start

    fig, ax = plt.subplots(figsize=(12.0, 6.2), dpi=180)
    librosa.display.specshow(
        db_band,
        x_coords=display_times,
        y_coords=freqs,
        x_axis="time",
        y_axis="linear",
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        ax=ax,
        cmap="magma",
    )

    ax.axhspan(
        item.primary_band[0],
        item.primary_band[1],
        color="lime",
        alpha=0.075,
        label=f"primary validation band {item.primary_band[0]:.0f}-{item.primary_band[1]:.0f} Hz",
    )
    if item.secondary_band is not None:
        ax.axhspan(
            item.secondary_band[0],
            item.secondary_band[1],
            color="cyan",
            alpha=0.07,
            label=f"secondary validation band {item.secondary_band[0]:.0f}-{item.secondary_band[1]:.0f} Hz",
        )

    for row in rows:
        component = row.get("component", "")
        if is_primary_component(component):
            color = "lime"
            linestyle = "-"
            linewidth = 2.0
            text_suffix = "p"
        elif is_secondary_component(component):
            color = "cyan"
            linestyle = "--"
            linewidth = 1.9
            text_suffix = "s"
        else:
            color = "white"
            linestyle = ":"
            linewidth = 1.2
            text_suffix = "?"

        start_sec = row_float(row, "start_sec")
        end_sec = row_float(row, "end_sec")
        fmin_hz = row_float(row, "fmin_hz")
        fmax_hz = row_float(row, "fmax_hz")
        ax.add_patch(
            Rectangle(
                (start_sec, fmin_hz),
                end_sec - start_sec,
                fmax_hz - fmin_hz,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )
        )
        ax.text(
            start_sec,
            fmax_hz + 35,
            f"{row.get('motif_id', '')}:{text_suffix}",
            color=color,
            fontsize=8,
            weight="bold",
        )

    ax.set_ylim(DISPLAY_BAND)
    ax.set_xlim(window_start, window_end)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(
        f"Manual motif annotations: {item.annotation_set}\n"
        f"{item.motif_family}; green=primary boxes, cyan=secondary boxes; shaded bands are validation bands"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def copy_and_combine(sets: list[ManualAnnotationSet], output_dir: Path) -> int:
    """Copy curated tables, render plots, and build manifest + combined CSV."""
    plots_dir = fresh_subdir(output_dir, "plots")
    source_plots_dir = fresh_subdir(output_dir, "source_plots")
    tables_dir = fresh_subdir(output_dir, "tables")

    manifest_rows = []
    combined_rows = []
    source_fieldnames: list[str] | None = None

    for item in sets:
        if not item.source_table.exists():
            raise FileNotFoundError(f"Missing source table: {item.source_table}")
        if not item.source_plot.exists():
            raise FileNotFoundError(f"Missing source plot: {item.source_plot}")

        table_dst = tables_dir / item.source_table.name
        source_plot_dst = source_plots_dir / item.source_plot.name
        plot_dst = plots_dir / item.output_plot_name
        shutil.copy2(item.source_table, table_dst)
        shutil.copy2(item.source_plot, source_plot_dst)
        rows = read_rows(item.source_table)
        render_standard_plot(item, rows, plot_dst)

        manifest_rows.append(
            {
                "annotation_set": item.annotation_set,
                "motif_family": item.motif_family,
                "status": item.status,
                "source_table": str(item.source_table),
                "source_plot": str(item.source_plot),
                "curated_table": str(table_dst),
                "curated_plot": str(plot_dst),
                "source_plot_copy": str(source_plot_dst),
                "primary_band_hz": f"{item.primary_band[0]:.0f}-{item.primary_band[1]:.0f}",
                "secondary_band_hz": "" if item.secondary_band is None else f"{item.secondary_band[0]:.0f}-{item.secondary_band[1]:.0f}",
                "notes": item.notes,
            }
        )

        with item.source_table.open(newline="") as f:
            reader = csv.DictReader(f)
            if source_fieldnames is None:
                source_fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row = dict(row)
                row.update(
                    {
                        "annotation_set": item.annotation_set,
                        "motif_family": item.motif_family,
                        "annotation_status": item.status,
                        "annotation_notes": item.notes,
                    }
                )
                combined_rows.append(row)

    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    combined_fields = [
        "annotation_set",
        "motif_family",
        "annotation_status",
        "annotation_notes",
        *(source_fieldnames or []),
    ]
    with (tables_dir / "manual_motif_boxes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({field: row.get(field, "") for field in combined_fields})

    return len(combined_rows)


def write_readme(output_dir: Path) -> None:
    """Write a README explaining provenance and regeneration."""
    (output_dir / "README.md").write_text(
        """# Marbled Frog Manual Motif Annotations

This folder collects the current manually selected/current-best motif-box annotations for Marbled White-lipped Frog (`22967`).

These are **review artifacts**, not ground-truth boxes. They preserve the boxes that currently look useful for three motif families:

1. `low_click_with_secondary` from `iNat1448422` 0-5s.
2. `broad_upper_sweep_with_low_tail` from `iNat1697350`.
3. `rising_chip_1k_1p8k` from `iNat1698104`.

## Files

- `manifest.csv` — source and curated paths for each annotation set.
- `tables/manual_motif_boxes.csv` — combined table with annotation family/status columns.
- `tables/*_motif_boxes.csv` — per-recording annotation tables copied from the motif-box workflow.
- `plots/*.png` — standardized primary/secondary validation plots.
- `source_plots/*.png` — original generated plots copied for provenance.

## Regeneration

From `user/anthony/experiments/200-data-analysis`:

```bash
python -m data_analysis.marbled_frog.motif_boxes
python -m data_analysis.marbled_frog.manual_annotations
```

Source code:

- `src/data_analysis/marbled_frog/motif_boxes.py` generates the motif boxes and source plots.
- `src/data_analysis/marbled_frog/manual_annotations.py` copies the curated/current-best tables and renders standardized validation plots.

## Caveat

Use these as motif evidence / crop hints. Do not treat them as final biological ground-truth boxes without further manual review and negative calibration.
"""
    )


def run(output_dir: Path | None = None, *, source_dir: Path | None = None) -> Path:
    """Export manual annotations and return the output directory."""
    source_dir = source_dir or artifact_dir("marbled_frog/motif_boxes")
    output_dir = output_dir or artifact_dir("marbled_frog/manual_annotations")
    n_rows = copy_and_combine(annotation_sets(source_dir), output_dir)
    write_readme(output_dir)
    print(f"Exported {n_rows} motif-box rows to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.output_dir, source_dir=args.source_dir)


if __name__ == "__main__":
    main()
