"""Full-band power spectra of labeled train_soundscape segments.

Generates a reference spectrum for what's actually present in BirdCLEF 2026
soundscapes across 0-16 kHz, stratified by labeled-segment taxon class
(Aves, Amphibia, Insecta, Mammalia, Reptilia). Useful for telling whether
high-frequency activity (e.g. 4-6 kHz energy in Marbled Frog non-call
windows) is plausibly an insect chorus, a bird, etc.

Inputs are 5-second windows extracted from `train_soundscapes/*.ogg` at
locations given by `train_soundscapes_labels.csv`. Per-class samples are
restricted to single-class segments (segments where every label maps to
the same taxonomic class) so each line represents a relatively clean
class signature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from data_analysis.common.paths import (
    TRAIN_SOUNDSCAPES_DIR,
    artifact_dir,
)
from data_analysis.soundscapes.labeled_soundscapes import (
    explode_segment_labels,
    load_segment_labels,
    load_taxonomy,
)

SAMPLE_RATE = 32_000
N_FFT = 2048
HOP_LENGTH = 512
DEFAULT_PER_CLASS = 200
DEFAULT_SEED = 42

# Linear bands from sub-bird through ultrasonic-edge — wider than the
# Marbled-Frog-focused bands in frequency_perch so the high-frequency
# soundscape content is visible.
BANDS_HZ: list[tuple[int, int]] = [
    (50, 250),
    (250, 500),
    (500, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 6000),
    (6000, 8000),
    (8000, 12000),
    (12000, 16000),
]

CLASS_COLORS = {
    "Aves": "#1f77b4",
    "Amphibia": "#2ca02c",
    "Insecta": "#d62728",
    "Mammalia": "#9467bd",
    "Reptilia": "#8c564b",
}


def read_soundscape_window(filename: str, start_sec: float, end_sec: float) -> np.ndarray:
    """Read a mono 32 kHz window from train_soundscapes."""
    audio, _ = librosa.load(
        TRAIN_SOUNDSCAPES_DIR / filename,
        sr=SAMPLE_RATE,
        mono=True,
        offset=float(start_sec),
        duration=float(end_sec - start_sec),
    )
    if audio.size == 0:
        return audio.astype(np.float32)
    audio = audio.astype(np.float32, copy=False)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return audio


def window_power_spectrum(audio: np.ndarray) -> np.ndarray:
    """Time-averaged FFT power for one window. Returns shape (n_freq,)."""
    if audio.size == 0:
        return np.full(N_FFT // 2 + 1, np.nan, dtype=np.float64)
    spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    power = np.abs(spec) ** 2
    return power.mean(axis=1)


def fft_frequencies() -> np.ndarray:
    """Linear FFT bin centers for our STFT settings."""
    return librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)


def select_single_class_segments(labels: pl.DataFrame) -> pl.DataFrame:
    """Keep only segments where every label belongs to a single taxonomic class.

    The labels frame has one row per (segment, label). Within a segment, if
    all rows share the same `class_name`, the segment is "single-class" and
    safe to attribute to that class.
    """
    return (
        labels.drop_nulls("class_name")
        .group_by("filename", "start_s", "end_s")
        .agg(
            pl.col("class_name").n_unique().alias("n_classes"),
            pl.col("class_name").first().alias("class_name"),
            pl.len().alias("n_labels"),
        )
        .filter(pl.col("n_classes") == 1)
        .drop("n_classes")
    )


def sample_segments(
    segments: pl.DataFrame,
    *,
    per_class: int,
    seed: int,
) -> pl.DataFrame:
    """Take up to `per_class` segments per class, deterministically per seed."""
    parts = []
    for (cls,), group in segments.sort("filename", "start_s").group_by(
        "class_name", maintain_order=True
    ):
        if group.height <= per_class:
            parts.append(group)
        else:
            parts.append(group.sample(n=per_class, seed=seed, shuffle=True))
    return pl.concat(parts).sort("class_name", "filename", "start_s")


def collect_class_spectra(
    segments: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray, dict[str, int]]:
    """Read each sampled segment and average the FFT power per class.

    Returns a (class_spectra_df, freqs, n_per_class) tuple. The DataFrame has
    one row per FFT bin and one column per class plus an `all` column.
    """
    classes = sorted(segments["class_name"].unique().to_list())
    freqs = fft_frequencies()
    sums: dict[str, np.ndarray] = {cls: np.zeros_like(freqs, dtype=np.float64) for cls in classes}
    counts: dict[str, int] = {cls: 0 for cls in classes}
    overall = np.zeros_like(freqs, dtype=np.float64)
    overall_count = 0

    for row in segments.iter_rows(named=True):
        try:
            audio = read_soundscape_window(row["filename"], row["start_s"], row["end_s"])
        except Exception as exc:  # noqa: BLE001 — log + continue, don't kill the run
            print(f"skip {row['filename']} [{row['start_s']:.0f}-{row['end_s']:.0f}]: {exc}")
            continue
        spectrum = window_power_spectrum(audio)
        if not np.all(np.isfinite(spectrum)):
            continue
        sums[row["class_name"]] += spectrum
        counts[row["class_name"]] += 1
        overall += spectrum
        overall_count += 1

    columns = {"freq_hz": pl.Series(freqs)}
    for cls in classes:
        if counts[cls] > 0:
            columns[cls] = pl.Series(sums[cls] / counts[cls])
        else:
            columns[cls] = pl.Series(np.full_like(freqs, np.nan))
    columns["all"] = pl.Series(overall / max(overall_count, 1))
    spectra = pl.DataFrame(columns)
    return spectra, freqs, counts


def to_db_relative(power: np.ndarray, reference_max: float) -> np.ndarray:
    """Convert linear power to dB normalized to a shared reference max."""
    safe = np.maximum(power, 1e-20)
    return 10.0 * np.log10(safe / max(reference_max, 1e-20))


def band_energy_fractions(
    spectra: pl.DataFrame,
    freqs: np.ndarray,
    classes: list[str],
) -> pl.DataFrame:
    """For each class, compute fraction of total power per frequency band."""
    rows: list[dict[str, object]] = []
    for cls in [*classes, "all"]:
        power = spectra[cls].to_numpy()
        total = float(np.nansum(power))
        if total <= 0:
            continue
        for lo_hz, hi_hz in BANDS_HZ:
            mask = (freqs >= lo_hz) & (freqs < hi_hz)
            rows.append(
                {
                    "class_name": cls,
                    "lo_hz": lo_hz,
                    "hi_hz": hi_hz,
                    "band_hz": f"{lo_hz}-{hi_hz}",
                    "energy_frac": float(np.nansum(power[mask])) / total,
                }
            )
    return pl.DataFrame(rows).sort("class_name", "lo_hz")


def plot_class_spectra(
    spectra: pl.DataFrame,
    freqs: np.ndarray,
    counts: dict[str, int],
    output_path: Path,
) -> None:
    """Plot per-class mean spectra plus an overall reference."""
    classes = [c for c in CLASS_COLORS if c in counts and counts[c] > 0]

    reference_max = max(
        np.nanmax(spectra["all"].to_numpy()),
        *(np.nanmax(spectra[c].to_numpy()) for c in classes if counts.get(c, 0)),
    )

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=180)
    ax.plot(
        freqs,
        to_db_relative(spectra["all"].to_numpy(), reference_max),
        color="black",
        linewidth=1.6,
        label=f"all single-class segments (n={sum(counts.values())})",
        zorder=10,
    )
    for cls in classes:
        ax.plot(
            freqs,
            to_db_relative(spectra[cls].to_numpy(), reference_max),
            color=CLASS_COLORS[cls],
            linewidth=1.4,
            label=f"{cls} (n={counts[cls]})",
            alpha=0.9,
        )

    ax.set_xlim(50, SAMPLE_RATE / 2)
    ax.set_ylim(-70, 2)
    ax.set_xscale("linear")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean power (dB, relative to global max)")
    ax.set_title("Train soundscape full-band mean power spectrum by labeled class")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_summary(
    counts: dict[str, int],
    band_table: pl.DataFrame,
    sample: pl.DataFrame,
    *,
    per_class: int,
    output_dir: Path,
) -> str:
    """Render a Markdown summary of the spectrum reference."""
    counts_table = pl.DataFrame(
        {
            "class_name": list(counts.keys()),
            "n_segments_used": list(counts.values()),
        }
    ).sort("class_name")

    pivot = (
        band_table.with_columns(pl.col("energy_frac").round(4))
        .pivot(values="energy_frac", index="band_hz", on="class_name", sort_columns=True)
    )
    return f"""# Train Soundscape Full-Band Spectrum Reference

Mean FFT power spectrum across labeled `train_soundscapes/` 5-second segments,
stratified by single-class taxonomic group. Sampled up to **{per_class}** segments
per class with seed-based deterministic selection.

## Segments used per class

```text
{counts_table.write_csv(separator='|').strip()}
```

## Energy fraction per band

```text
{pivot.write_csv(separator='|').strip()}
```

## Files

- `class_spectra.csv` — one row per FFT bin, columns are mean linear power per class plus `all`.
- `class_band_energy.csv` — long-form energy fractions per (class, band).
- `sampled_segments.csv` — the segments that were read, with class assignment.
- `full_band_spectra.png` — the rendered reference plot.

Sample window count: {sample.height}

Reading: 5 s windows from `train_soundscapes/*.ogg` at 32 kHz, STFT n_fft={N_FFT},
hop={HOP_LENGTH}, time-averaged power per window then mean-aggregated per class.

Output directory: `{output_dir}`
"""


def run(
    output_dir: Path | None = None,
    *,
    per_class: int = DEFAULT_PER_CLASS,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Run the soundscape spectrum reference and return the output directory."""
    output_dir = output_dir or artifact_dir("soundscapes/spectra")
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = load_segment_labels()
    taxonomy = load_taxonomy()
    labels = explode_segment_labels(segments, taxonomy)
    single_class = select_single_class_segments(labels)
    sample = sample_segments(single_class, per_class=per_class, seed=seed)
    sample.write_csv(output_dir / "sampled_segments.csv")
    print(
        f"Reading {sample.height} segments across "
        f"{sample['class_name'].n_unique()} classes..."
    )

    spectra, freqs, counts = collect_class_spectra(sample)
    spectra.write_csv(output_dir / "class_spectra.csv")

    classes_present = [c for c in CLASS_COLORS if counts.get(c, 0) > 0]
    band_table = band_energy_fractions(spectra, freqs, classes_present)
    band_table.write_csv(output_dir / "class_band_energy.csv")

    plot_class_spectra(spectra, freqs, counts, output_dir / "full_band_spectra.png")

    summary = render_summary(
        counts,
        band_table,
        sample,
        per_class=per_class,
        output_dir=output_dir,
    )
    (output_dir / "summary.md").write_text(summary)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/soundscapes/spectra/.",
    )
    parser.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    output_dir = run(args.output_dir, per_class=args.per_class, seed=args.seed)
    print(f"Wrote soundscape spectrum reference to {output_dir}")


if __name__ == "__main__":
    main()
