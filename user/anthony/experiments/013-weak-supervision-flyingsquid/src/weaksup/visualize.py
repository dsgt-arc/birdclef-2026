"""Soft label spectrogram visualizations for focal and soundscape splits.

Generates a multi-row figure where each row is a 10-second mel spectrogram
with semi-transparent green overlays on tokens where the fused soft label
exceeds a threshold. Helps validate fusion quality visually.

CLI:
    weaksup-visualize --split focal \\
        --fused-dir ~/scratch/birdclef-2026/outputs/exp013/fused \\
        --data-dir ~/scratch/birdclef-2026/data/birdclef-2025 \\
        --output ~/scratch/birdclef-2026/outputs/exp013/eval/focal/soft_label_spectrograms.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from rich.console import Console

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

DEFAULT_FUSED_DIR = Path("~/scratch/birdclef-2026/outputs/exp013/fused").expanduser()
DEFAULT_DATA_DIR = Path("~/scratch/birdclef-2026/data/birdclef-2025").expanduser()


def load_audio(path: Path, sr: int = 32000, max_duration: float = 10.0) -> np.ndarray:
    """Load first max_duration seconds of audio as mono float32."""
    import soundfile as sf

    info = sf.info(str(path))
    max_frames = int(max_duration * info.samplerate)
    data, file_sr = sf.read(str(path), frames=max_frames, dtype="float32")

    # Mono downmix
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if file_sr != sr:
        import librosa

        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)

    return data


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 320,
    n_mels: int = 128,
    fmin: float = 50.0,
    fmax: float = 14000.0,
) -> np.ndarray:
    """Compute log-mel spectrogram. Returns dB-scaled (n_mels, T) array."""
    import librosa

    S = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    return librosa.power_to_db(S, ref=np.max)


def sample_recordings(
    soft_df: pl.DataFrame,
    file_to_species: dict[str, str] | None,
    species_cols: list[str],
    n_candidates: int,
    n_rows: int,
    threshold: float,
    seed: int,
) -> list[dict]:
    """Select visually interesting recordings with diverse species.

    For focal: uses primary_label from train.csv.
    For soundscape: uses argmax across species columns per token.

    Returns list of {file, species, tokens: [(start_time, soft_label), ...]}.
    """
    rng = np.random.RandomState(seed)

    # Filter to tokens in first 10 seconds
    df10 = soft_df.filter(pl.col("start_time") < 10.0)

    # Per-recording: determine dominant species and count active tokens
    recordings: dict[str, dict] = {}
    for row in df10.iter_rows(named=True):
        fkey = row["file"]
        if fkey not in recordings:
            # Determine species for this recording
            if file_to_species is not None:
                species = file_to_species.get(fkey)
                if species is None:
                    continue
            else:
                # Soundscape: argmax across species columns
                vals = np.array([row[s] for s in species_cols])
                species = species_cols[int(np.argmax(vals))]

            recordings[fkey] = {
                "file": fkey,
                "species": species,
                "tokens": [],
            }

        rec = recordings[fkey]
        species = rec["species"]
        soft_val = row.get(species, 0.0)
        rec["tokens"].append((row["start_time"], soft_val))

    # Filter: 2-8 active tokens in first 10s (visually interesting)
    candidates = []
    for rec in recordings.values():
        active = sum(1 for _, v in rec["tokens"] if v > threshold)
        if 2 <= active <= 8:
            candidates.append(rec)

    if not candidates:
        # Relax: any recording with at least 1 active token
        for rec in recordings.values():
            active = sum(1 for _, v in rec["tokens"] if v > threshold)
            if active >= 1:
                candidates.append(rec)

    if not candidates:
        console.print("[yellow]Warning: no recordings found with active tokens above threshold[/yellow]")
        # Fall back to first n_rows recordings
        candidates = list(recordings.values())[:n_rows]
        return candidates

    # Shuffle and diversify across species (max 2 per species)
    rng.shuffle(candidates)
    selected: list[dict] = []
    species_count: dict[str, int] = {}

    for rec in candidates:
        sp = rec["species"]
        if species_count.get(sp, 0) >= 2:
            continue
        selected.append(rec)
        species_count[sp] = species_count.get(sp, 0) + 1
        if len(selected) >= n_candidates:
            break

    # Final random selection down to n_rows
    if len(selected) > n_rows:
        indices = rng.choice(len(selected), size=n_rows, replace=False)
        selected = [selected[i] for i in sorted(indices)]

    return selected[:n_rows]


def render_figure(
    recordings: list[dict],
    audio_dir: Path,
    n_rows: int,
    threshold: float,
    split: str,
    sr: int = 32000,
    hop_length: int = 320,
):
    """Render multi-row spectrogram figure with soft label overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = min(n_rows, len(recordings))
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, rec in zip(axes, recordings[:n_rows]):
        fkey = rec["file"]
        species = rec["species"]
        tokens = rec["tokens"]

        # Resolve audio path
        audio_path = audio_dir / fkey
        if not audio_path.exists():
            ax.set_title(f"{species} | {fkey} [NOT FOUND]")
            ax.set_facecolor("lightgray")
            continue

        # Load and compute spectrogram
        waveform = load_audio(audio_path, sr=sr)
        S_db = compute_mel_spectrogram(waveform, sr=sr, hop_length=hop_length)
        duration = len(waveform) / sr

        # Display spectrogram
        ax.imshow(
            S_db,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[0, duration, 0, 14],
        )

        # Overlay green spans for active tokens
        for t_start, soft_val in tokens:
            if soft_val > threshold:
                # Alpha proportional to confidence
                alpha = 0.15 + 0.35 * (soft_val - threshold) / max(1 - threshold, 1e-6)
                alpha = min(alpha, 0.5)
                ax.axvspan(t_start, t_start + 1.0, color="lime", alpha=alpha)

        ax.set_title(f"{species} | {fkey}", fontsize=10)
        ax.set_ylabel("Freq (kHz)", fontsize=8)
        ax.set_xlim(0, min(duration, 10.0))

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    fig.suptitle(f"Soft Label Overlay — {split}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


@app.command()
def visualize(
    split: Annotated[
        str, typer.Option(help="Data split: focal or soundscape")
    ] = "focal",
    fused_dir: Annotated[
        Path, typer.Option(help="Dir with focal/ and soundscape/ subdirs containing soft_labels.parquet")
    ] = DEFAULT_FUSED_DIR,
    data_dir: Annotated[
        Path, typer.Option(help="BirdCLEF data dir (train.csv, train_audio/, train_soundscapes/)")
    ] = DEFAULT_DATA_DIR,
    output: Annotated[
        Path | None, typer.Option(help="Output PNG path (default: eval/<split>/soft_label_spectrograms.png)")
    ] = None,
    n_rows: Annotated[int, typer.Option(help="Number of spectrogram rows")] = 5,
    n_candidates: Annotated[int, typer.Option(help="Candidate pool size before final selection")] = 20,
    threshold: Annotated[float, typer.Option(help="Soft label threshold for highlighting")] = 0.3,
    seed: Annotated[int, typer.Option(help="Random seed for sampling")] = 42,
) -> None:
    """Generate spectrogram visualizations with soft label overlays."""
    if split not in ("focal", "soundscape"):
        raise typer.BadParameter(f"split must be 'focal' or 'soundscape', got {split!r}")

    # Resolve paths
    soft_labels_path = fused_dir / split / "soft_labels.parquet"
    if not soft_labels_path.exists():
        console.print(f"[red]Not found: {soft_labels_path}[/red]")
        raise typer.Exit(1)

    if output is None:
        output = fused_dir.parent / "eval" / split / "soft_label_spectrograms.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    is_focal = split == "focal"
    audio_dir = data_dir / ("train_audio" if is_focal else "train_soundscapes")

    console.rule(f"[bold]Soft Label Visualization — {split}")
    console.print(f"  Fused labels: {soft_labels_path}")
    console.print(f"  Audio dir:    {audio_dir}")
    console.print(f"  Output:       {output}")

    # Load fused soft labels
    console.print("Loading soft labels...")
    soft_df = pl.read_parquet(soft_labels_path)
    species_cols = [c for c in soft_df.columns if c not in ("file", "start_time")]
    console.print(f"  {soft_df.shape[0]} tokens, {len(species_cols)} species")

    # Load train.csv for focal (species mapping); None for soundscape
    file_to_species = None
    if is_focal:
        import csv

        file_to_species = {}
        with open(data_dir / "train.csv") as f:
            for row in csv.DictReader(f):
                file_to_species[row["filename"]] = row["primary_label"]
        console.print(f"  {len(file_to_species)} recordings in train.csv")

    # Sample recordings
    console.print(f"Sampling {n_rows} recordings (candidates={n_candidates}, threshold={threshold})...")
    recordings = sample_recordings(
        soft_df, file_to_species, species_cols,
        n_candidates=n_candidates, n_rows=n_rows,
        threshold=threshold, seed=seed,
    )
    console.print(f"  Selected {len(recordings)} recordings:")
    for rec in recordings:
        active = sum(1 for _, v in rec["tokens"] if v > threshold)
        console.print(f"    {rec['species']:20s} | {rec['file']} ({active} active tokens)")

    # Render figure
    console.print("Rendering spectrograms...")
    fig = render_figure(
        recordings, audio_dir,
        n_rows=n_rows, threshold=threshold, split=split,
    )

    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    console.print(f"[green]Saved: {output}[/green]")

    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    app()
