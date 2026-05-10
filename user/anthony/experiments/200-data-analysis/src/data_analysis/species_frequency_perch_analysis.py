"""Frequency-band and Perch-logit analysis for one train-audio species.

Default target is Marbled White-lipped Frog (`22967`, Leptodactylus elenae).
The analysis compares train-audio window mel spectra against Perch target/genus
logits to estimate which frequency bands are most supported by the data.
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

from data_analysis.labeled_species_training_coverage import TRAIN_SCHEMA_OVERRIDES
from data_analysis.paths import PERCH_TRAIN_AUDIO_LOGITS, TAXONOMY_CSV, TRAIN_AUDIO_DIR, TRAIN_CSV, artifact_dir

PERCH_LABELS_CSV = Path(
    "/mnt/data/research/arc/birdclef-2026/user/anthony/experiments/106-perch-bayesian/artifacts/perch_v2_labels.csv"
)

DEFAULT_LABEL = "22967"
DEFAULT_GENUS = "Leptodactylus"

MEL_SAMPLE_RATE = 32_000
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 128
MEL_FMIN = 50
MEL_FMAX = 8_000

BANDS_HZ = [
    (0, 250),
    (250, 500),
    (500, 750),
    (750, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 3000),
    (3000, 4000),
    (4000, 6000),
    (6000, 8000),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def load_taxon(label: str) -> dict[str, str]:
    """Load taxonomy row for a BirdCLEF label."""
    row = (
        pl.read_csv(TAXONOMY_CSV, schema_overrides={"primary_label": pl.Utf8})
        .filter(pl.col("primary_label") == label)
        .row(0, named=True)
    )
    return {k: str(v) for k, v in row.items() if v is not None}


def load_perch_labels() -> pl.DataFrame:
    """Load Perch v2 label names with logit indices."""
    return pl.read_csv(PERCH_LABELS_CSV).with_row_index("perch_index").rename(
        {"inat2024_fsd50k": "perch_label"}
    )


def load_train_files(label: str) -> pl.DataFrame:
    """Return train_audio files where label is primary or secondary."""
    train = pl.read_csv(TRAIN_CSV, schema_overrides=TRAIN_SCHEMA_OVERRIDES).select(
        "filename", "primary_label", "secondary_labels", "rating", "common_name", "scientific_name"
    )
    primary = train.filter(pl.col("primary_label") == label).select(
        "filename", "rating", pl.lit("primary").alias("label_role")
    )
    secondary = train.filter(pl.col("secondary_labels").str.contains(label)).select(
        "filename", "rating", pl.lit("secondary").alias("label_role")
    )
    return pl.concat([primary, secondary]).unique(["filename", "label_role"]).sort("filename")


def logit_column(index: int) -> str:
    return f"logit_{index}"


def load_logits_for_files(files: pl.DataFrame, label: str, genus: str) -> tuple[pl.DataFrame, dict[str, object]]:
    """Load selected files' logits and derive target/genus scores."""
    perch_labels = load_perch_labels()
    target_taxon = load_taxon(label)
    scientific_name = target_taxon["scientific_name"]
    target_match = perch_labels.filter(pl.col("perch_label") == scientific_name)
    if target_match.is_empty():
        raise ValueError(f"Could not find {scientific_name!r} in Perch labels")
    target_index = int(target_match["perch_index"][0])
    target_col = logit_column(target_index)

    genus_matches = perch_labels.filter(pl.col("perch_label").str.starts_with(f"{genus} "))
    genus_indices = [int(i) for i in genus_matches["perch_index"].to_list()]
    genus_cols = [logit_column(i) for i in genus_indices]

    base_cols = ["filename", "chunk_idx", "start_sec", "end_sec"]
    other_genus_cols = [col for col in genus_cols if col != target_col]
    logits = (
        pl.scan_parquet(PERCH_TRAIN_AUDIO_LOGITS)
        .join(files.lazy(), on="filename", how="inner")
        .select(*base_cols, "label_role", "rating", target_col, *other_genus_cols)
        .collect()
    )

    genus_matrix = logits.select(genus_cols).to_numpy().astype(np.float32)
    target_values = logits[target_col].to_numpy().astype(np.float32)
    max_idx = genus_matrix.argmax(axis=1)
    max_genus_logit = genus_matrix[np.arange(genus_matrix.shape[0]), max_idx]
    max_genus_index = [genus_indices[i] for i in max_idx]
    max_genus_label = [perch_labels["perch_label"][i] for i in max_genus_index]

    logits = logits.with_columns(
        pl.Series("target_perch_index", np.full(logits.height, target_index, dtype=np.int64)),
        pl.Series("target_logit", target_values),
        pl.Series("target_prob", sigmoid(target_values)),
        pl.Series("max_genus_logit", max_genus_logit),
        pl.Series("max_genus_prob", sigmoid(max_genus_logit)),
        pl.Series("max_genus_perch_index", max_genus_index),
        pl.Series("max_genus_label", max_genus_label),
    )

    info = {
        "target_index": target_index,
        "target_col": target_col,
        "target_scientific_name": scientific_name,
        "genus_indices": genus_indices,
        "genus_cols": genus_cols,
        "perch_labels": perch_labels,
    }
    return logits, info


def top_mean_logits(logits: pl.DataFrame, perch_labels: pl.DataFrame, top_n: int = 30) -> pl.DataFrame:
    """Compute top Perch taxa by mean logit for these windows."""
    logit_cols = [c for c in logits.columns if c.startswith("logit_")]
    means = np.asarray(logits.select(logit_cols).mean().row(0), dtype=np.float32)
    order = np.argsort(means)[::-1][:top_n]
    return pl.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "perch_index": order.astype(np.int64),
            "perch_label": [perch_labels["perch_label"][int(i)] for i in order],
            "mean_logit": means[order],
            "mean_prob": sigmoid(means[order]),
        }
    )


def read_audio_window(filename: str, start_sec: float, end_sec: float) -> np.ndarray:
    """Read a mono 32 kHz audio window from train_audio with librosa."""
    path = TRAIN_AUDIO_DIR / filename
    audio, _ = librosa.load(
        path,
        sr=MEL_SAMPLE_RATE,
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


def window_mel_spectrum(filename: str, start_sec: float, end_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized mel-band energy for one audio window."""
    audio = read_audio_window(filename, start_sec, end_sec)
    if audio.size < MEL_N_FFT:
        raise ValueError(f"Too little audio in {filename} {start_sec}-{end_sec}")
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=MEL_SAMPLE_RATE,
        n_fft=MEL_N_FFT,
        hop_length=MEL_HOP_LENGTH,
        n_mels=MEL_N_MELS,
        fmin=MEL_FMIN,
        fmax=MEL_FMAX,
        power=2.0,
    )
    mel_energy = np.maximum(mel.mean(axis=1).astype(np.float64), 1e-18)
    mel_energy = mel_energy / mel_energy.sum()
    mel_freqs = librosa.mel_frequencies(n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX)
    return mel_freqs, mel_energy


def add_spectral_features(logits: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Attach per-window mel spectral features and return the mel-energy matrix."""
    spectra: list[np.ndarray] = []
    freqs_ref: np.ndarray | None = None
    rows = []

    for row in logits.iter_rows(named=True):
        freqs, spectrum = window_mel_spectrum(row["filename"], float(row["start_sec"]), float(row["end_sec"]))
        if freqs_ref is None:
            freqs_ref = freqs
        spectra.append(spectrum)

        band_values = {}
        for lo, hi in BANDS_HZ:
            mask = (freqs_ref >= lo) & (freqs_ref < hi)
            band_values[f"band_{lo}_{hi}_mel_energy_frac"] = float(spectrum[mask].sum())
        peak_hz = float(freqs_ref[np.argmax(spectrum)])
        centroid_hz = float(np.sum(freqs_ref * spectrum) / np.sum(spectrum))
        rows.append({"mel_peak_hz": peak_hz, "mel_centroid_hz": centroid_hz, **band_values})

    assert freqs_ref is not None
    return logits.with_columns(pl.DataFrame(rows)), freqs_ref, np.vstack(spectra)


def band_summary(windows: pl.DataFrame) -> pl.DataFrame:
    """Summarize spectral energy by band for high/low genus-logit windows."""
    high_threshold = float(windows["max_genus_logit"].quantile(0.75))
    low_threshold = float(windows["max_genus_logit"].quantile(0.25))
    rows = []
    for lo, hi in BANDS_HZ:
        col = f"band_{lo}_{hi}_mel_energy_frac"
        high = windows.filter(pl.col("max_genus_logit") >= high_threshold)[col].mean()
        low = windows.filter(pl.col("max_genus_logit") <= low_threshold)[col].mean()
        all_mean = windows[col].mean()
        rows.append(
            {
                "band_hz": f"{lo}-{hi}",
                "lo_hz": lo,
                "hi_hz": hi,
                "mean_energy_frac_all": all_mean,
                "mean_energy_frac_high_genus_logit": high,
                "mean_energy_frac_low_genus_logit": low,
                "high_minus_low": high - low,
                "high_div_low": high / low if low and low > 0 else None,
            }
        )
    return pl.DataFrame(rows).sort("high_minus_low", descending=True)


def recommended_band(summary: pl.DataFrame) -> tuple[int, int]:
    """Pick a compact recommended band from the top adjacent informative bands."""
    useful = summary.filter(
        (pl.col("lo_hz") >= 250)
        & (pl.col("hi_hz") <= 3000)
        & (pl.col("high_minus_low") > 0)
    ).sort("high_minus_low", descending=True)
    if useful.is_empty():
        return 250, 3000
    top = useful.head(3).sort("lo_hz")
    return int(top["lo_hz"].min()), int(top["hi_hz"].max())


def plot_logit_summary(top_logits: pl.DataFrame, target_index: int, output_path: Path) -> None:
    """Plot top mean Perch logits."""
    df = top_logits.sort("mean_logit")
    colors = ["#d62728" if int(i) == target_index else "#1f77b4" for i in df["perch_index"].to_list()]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    ax.barh(df["perch_label"].to_list(), df["mean_logit"].to_list(), color=colors, alpha=0.85)
    ax.set_xlabel("mean Perch logit across train-audio windows")
    ax.set_title("Top Perch taxa by mean logit")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_spectrum_comparison(
    windows: pl.DataFrame,
    freqs: np.ndarray,
    spectra: np.ndarray,
    band: tuple[int, int],
    output_path: Path,
) -> None:
    """Plot mean mel spectra for high/low max-genus-logit windows."""
    scores = windows["max_genus_logit"].to_numpy()
    high_mask = scores >= np.quantile(scores, 0.75)
    low_mask = scores <= np.quantile(scores, 0.25)
    all_mean = spectra.mean(axis=0)
    high_mean = spectra[high_mask].mean(axis=0)
    low_mean = spectra[low_mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=180)
    ax.plot(freqs, 10 * np.log10(all_mean + 1e-18), label="all windows", color="black", alpha=0.7)
    ax.plot(freqs, 10 * np.log10(high_mean + 1e-18), label="top quartile max Leptodactylus logit", color="#d62728")
    ax.plot(freqs, 10 * np.log10(low_mean + 1e-18), label="bottom quartile max Leptodactylus logit", color="#1f77b4")
    ax.axvspan(band[0], band[1], color="#ff9896", alpha=0.22, label=f"candidate band {band[0]}-{band[1]} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized mean mel energy (dB)")
    ax.set_title("Marbled White-lipped Frog train-audio mel spectra vs Perch genus-logit support")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_window_scores(windows: pl.DataFrame, output_path: Path) -> None:
    """Plot direct target and max genus logits for every window."""
    df = windows.sort("max_genus_logit", descending=True).with_columns(
        (
            pl.col("filename")
            + " ["
            + pl.col("start_sec").cast(pl.Int64).cast(pl.Utf8)
            + "-"
            + pl.col("end_sec").cast(pl.Int64).cast(pl.Utf8)
            + "s]"
        ).alias("window")
    )
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    y = np.arange(df.height)
    ax.scatter(df["max_genus_logit"], y, label="max Leptodactylus logit", color="#d62728", s=34)
    ax.scatter(df["target_logit"], y, label="direct L. elenae logit", color="#1f77b4", s=24)
    ax.set_yticks(y)
    ax.set_yticklabels(df["window"].to_list(), fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Perch logit")
    ax.set_title("Window-level Perch evidence")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_markdown(
    label: str,
    taxon: dict[str, str],
    windows: pl.DataFrame,
    top_logits: pl.DataFrame,
    bands: pl.DataFrame,
    band: tuple[int, int],
    output_dir: Path,
) -> str:
    """Render notebook summary."""
    top_windows = windows.sort("max_genus_logit", descending=True).select(
        "filename",
        "start_sec",
        "end_sec",
        "target_logit",
        "target_prob",
        "max_genus_logit",
        "max_genus_prob",
        "max_genus_label",
        "mel_peak_hz",
        "mel_centroid_hz",
    ).head(12)

    target_rank = top_logits.filter(pl.col("perch_label") == taxon["scientific_name"])
    if target_rank.is_empty():
        target_rank_text = "not in top list"
    else:
        r = target_rank.row(0, named=True)
        target_rank_text = f"rank {r['rank']} by mean logit in top table"

    return f"""# Notebook: {taxon['common_name']} Frequency Band + Perch Logit Analysis

## Header

| Field | Value |
|---|---|
| Date | 2026-05-10 |
| BirdCLEF label | `{label}` |
| Scientific name | {taxon['scientific_name']} |
| Common name | {taxon['common_name']} |
| Class | {taxon['class_name']} |
| Train-audio files | {windows['filename'].n_unique()} |
| Train-audio 5s windows | {windows.height} |
| Output directory | `{output_dir}` |

## At-a-glance takeaway

Perch's **direct** `{taxon['scientific_name']}` logit is weak on these windows, but Perch strongly fires on nearby **Leptodactylus** taxa, especially the top mean-logit congeners shown below. For relevance ranking, max Leptodactylus-genus logit is more useful than the direct species logit.

Candidate data-supported frequency band: **{band[0]}-{band[1]} Hz**.

This band is selected from frequency bands whose normalized mel energy is enriched in windows with high max-Leptodactylus Perch logits. Treat this as a working band for inspection/filtering, not a ground-truth call box.

## Plots

- `top_mean_perch_logits.png` — top Perch taxa by mean logit across this species' train-audio windows.
- `window_perch_scores.png` — direct target logit vs max Leptodactylus-genus logit by window.
- `spectrum_high_vs_low_genus_logit.png` — average mel spectra for high- vs low-genus-logit windows, with candidate band shaded.

## Top mean Perch logits

```text
{top_logits.head(20).write_csv(separator='|').strip()}
```

Direct target species logit status: {target_rank_text}.

## Frequency band enrichment summary

```text
{bands.write_csv(separator='|').strip()}
```

## Highest max-Leptodactylus-logit windows

```text
{top_windows.write_csv(separator='|').strip()}
```

## Interpretation notes

- The dataset gives only file-level labels for `train_audio`; window relevance is inferred, not known.
- Direct Perch `{taxon['scientific_name']}` scores are not reliable here, but genus-level Perch evidence is strong.
- The candidate band should be used as a prior for spectrogram inspection and pseudo-label ranking.
- Manual checking is still needed before treating any window/frequency region as ground truth.
"""


def run(
    output_dir: Path | None = None,
    *,
    label: str = DEFAULT_LABEL,
    genus: str = DEFAULT_GENUS,
) -> Path:
    """Run analysis."""
    taxon = load_taxon(label)
    output_dir = output_dir or artifact_dir(f"species_frequency_perch_{label}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = load_train_files(label)
    logits_for_scores, info = load_logits_for_files(files, label, genus)

    # Reload all logits only for the small selected file set to identify top Perch taxa.
    all_logits = (
        pl.scan_parquet(PERCH_TRAIN_AUDIO_LOGITS)
        .join(files.select("filename").lazy(), on="filename", how="inner")
        .collect()
    )
    top_logits = top_mean_logits(all_logits, info["perch_labels"])

    windows, freqs, spectra = add_spectral_features(logits_for_scores)
    bands = band_summary(windows)
    band = recommended_band(bands)

    files.write_csv(output_dir / "train_files.csv")
    windows.write_csv(output_dir / "window_scores_and_spectral_features.csv")
    top_logits.write_csv(output_dir / "top_mean_perch_logits.csv")
    bands.write_csv(output_dir / "band_enrichment.csv")

    plot_logit_summary(top_logits, int(info["target_index"]), output_dir / "top_mean_perch_logits.png")
    plot_window_scores(windows, output_dir / "window_perch_scores.png")
    plot_spectrum_comparison(windows, freqs, spectra, band, output_dir / "spectrum_high_vs_low_genus_logit.png")

    (output_dir / "summary.md").write_text(
        render_markdown(label, taxon, windows, top_logits, bands, band, output_dir)
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--genus", default=DEFAULT_GENUS)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = run(args.output_dir, label=args.label, genus=args.genus)
    print(f"Wrote species frequency/Perch analysis to {output_dir}")


if __name__ == "__main__":
    main()
