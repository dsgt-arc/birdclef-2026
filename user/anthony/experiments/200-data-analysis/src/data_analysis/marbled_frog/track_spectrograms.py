"""Full-track, full-frequency spectrograms for Marbled Frog train_audio.

One PNG per recording in `train_audio/<label>/`, showing the entire track on
a linear time axis vs log frequency from 50 Hz to Nyquist (16 kHz at the
shared 32 kHz sample rate). Designed for eyeballing **where** in time the
high-frequency activity lives — e.g. is the 4-6 kHz energy a sustained
insect chorus, or transient peaks aligned with the frog call?

Two reference bands are shaded for context:
- cyan: 1-3 kHz Marbled Frog call prior (from frequency_perch).
- red:  4-6 kHz Insecta-dominant band (from soundscapes spectrum reference).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from data_analysis.common.paths import (
    TRAIN_AUDIO_DIR,
    artifact_dir,
    fresh_subdir,
)
from data_analysis.common.spectrogram import (
    MEL_HOP_LENGTH,
    MEL_N_FFT,
    MEL_SAMPLE_RATE,
)
from data_analysis.marbled_frog.frequency_perch import DEFAULT_LABEL

CALL_BAND = (1000.0, 3000.0)
INSECT_BAND = (4000.0, 6000.0)
DISPLAY_FMIN = 50.0


def list_label_files(label: str) -> list[Path]:
    """Return sorted list of `.ogg` files for a label under train_audio."""
    label_dir = TRAIN_AUDIO_DIR / label
    return sorted(p for p in label_dir.glob("*.ogg") if p.is_file())


def stft_power_db(audio: np.ndarray) -> np.ndarray:
    """STFT magnitude in dB, peak-normalized for display."""
    spec = librosa.stft(audio, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH, window="hann")
    power = np.abs(spec) ** 2
    return librosa.power_to_db(power, ref=np.max)


def plot_track_spectrogram(audio_path: Path, output_path: Path) -> dict[str, float]:
    """Render a full-track log-frequency spectrogram and return summary stats."""
    audio, _ = librosa.load(audio_path, sr=MEL_SAMPLE_RATE, mono=True)
    duration = audio.size / MEL_SAMPLE_RATE
    db = stft_power_db(audio)
    nyquist = MEL_SAMPLE_RATE / 2

    fig, ax = plt.subplots(figsize=(max(8, duration * 0.45), 5.0), dpi=180)
    img = librosa.display.specshow(
        db,
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        x_axis="time",
        y_axis="log",
        ax=ax,
        cmap="magma",
        vmin=-80,
        vmax=0,
    )
    ax.set_ylim(DISPLAY_FMIN, nyquist)
    ax.axhspan(*CALL_BAND, color="cyan", alpha=0.10, zorder=2)
    ax.axhspan(*INSECT_BAND, color="red", alpha=0.07, zorder=2)
    for hz in (1000, 2000, 3000, 4000, 6000, 8000):
        ax.axhline(hz, color="white", linewidth=0.4, alpha=0.18, zorder=3)
    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
    ax.set_title(
        f"{audio_path.parent.name}/{audio_path.name}  ({duration:.1f}s)\n"
        f"cyan = 1-3 kHz call prior, red = 4-6 kHz insect band"
    )
    ax.set_ylabel("Frequency (Hz, log)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "filename": f"{audio_path.parent.name}/{audio_path.name}",
        "duration_s": float(duration),
        "rms_db": float(20.0 * np.log10(np.sqrt(np.mean(audio**2) + 1e-12))),
        "plot": output_path.name,
    }


def render_summary(rows: list[dict[str, float]], output_dir: Path, label: str) -> str:
    """Render a Markdown index of generated track spectrograms."""
    table = pl.DataFrame(rows).sort("filename").select(
        "filename", "duration_s", "rms_db", "plot"
    )
    return f"""# Marbled Frog Full-Track Spectrograms

Per-recording full-frequency (50 Hz to 16 kHz, log y-axis) spectrograms for
every file in `train_audio/{label}/`. Use these to see **where** in time the
high-frequency activity sits relative to the 1-3 kHz frog call.

- cyan band: 1-3 kHz Marbled Frog call prior.
- red band:  4-6 kHz Insecta-dominant band (see `soundscapes/spectra/`).

```text
{table.write_csv(separator='|').strip()}
```

Output directory: `{output_dir}`
"""


def run(output_dir: Path | None = None, *, label: str = DEFAULT_LABEL) -> Path:
    """Generate full-track spectrograms for every file under `train_audio/<label>/`."""
    output_dir = output_dir or artifact_dir("marbled_frog/track_spectrograms")
    plots_dir = fresh_subdir(output_dir, "plots")

    files = list_label_files(label)
    if not files:
        raise FileNotFoundError(f"No .ogg files under {TRAIN_AUDIO_DIR / label}")

    rows: list[dict[str, float]] = []
    for audio_path in files:
        out = plots_dir / f"{audio_path.stem}.png"
        rows.append(plot_track_spectrogram(audio_path, out))
        print(f"wrote {out.name}: {rows[-1]['duration_s']:.1f}s")

    (output_dir / "summary.md").write_text(render_summary(rows, output_dir, label))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to artifacts/marbled_frog/track_spectrograms/.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=DEFAULT_LABEL,
        help="BirdCLEF primary_label whose train_audio files to render.",
    )
    args = parser.parse_args()
    output_dir = run(args.output_dir, label=args.label)
    print(f"Wrote track spectrograms to {output_dir}")


if __name__ == "__main__":
    main()
