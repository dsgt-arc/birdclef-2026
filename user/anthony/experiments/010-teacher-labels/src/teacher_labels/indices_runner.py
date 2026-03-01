"""Acoustic indices runner: per-window spectral features for weak supervision LFs.

Computes 5 acoustic indices per 3-second window (1-second step) to match
BirdNET's temporal resolution for direct join in experiment 013.

Indices:
  - NDSI: biophony vs anthropophony ratio (scikit-maad)
  - Spectral contrast: harmonic peak prominence (librosa)
  - Spectral flatness: tonality / Wiener entropy (librosa)
  - Bioacoustic index: energy in 2-15 kHz (scikit-maad)
  - ROI detected: any activity in this window (scikit-maad)

Output structure:
  indices/train_audio/parts/part_NNNN.parquet
  indices/train_soundscapes/parts/part_NNNN.parquet
"""

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import typer

from teacher_labels.utils import (
    ensure_output_dir,
    list_audio_files,
    load_config,
    partition_files,
)

app = typer.Typer()

# STFT parameters (shared across librosa and maad)
N_FFT = 2048
HOP_LENGTH = 480  # ~10ms at 48kHz


def _compute_roi_mask(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute a binary activity mask from ROI detection on the full signal.

    Returns a boolean array of length ceil(len(y) / HOP_LENGTH) indicating
    frames with detected activity (energy > 6dB above median background).
    """
    from maad.sound import spectrogram as maad_spectrogram

    Sxx, tn, fn, ext = maad_spectrogram(y, sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH)
    # Remove background via median and threshold at 6dB
    Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-10))
    bg = np.median(Sxx_db, axis=1, keepdims=True)
    Sxx_clean = Sxx_db - bg

    # Binary mask: any frequency bin above threshold
    activity = np.any(Sxx_clean > 6.0, axis=0)
    return activity, tn


def _process_file(
    filepath: Path,
    audio_dir: Path,
    sr: int,
    window_sec: float,
    window_step: float,
) -> list[dict]:
    """Process a single audio file and return rows of acoustic indices.

    Args:
        filepath: Absolute path to audio file.
        audio_dir: Root audio directory (for computing relative paths).
        sr: Target sample rate.
        window_sec: Window duration in seconds.
        window_step: Window step in seconds.

    Returns:
        List of dicts, one per window.
    """
    from maad.features import soundscape_index, bioacoustics_index
    from maad.sound import spectrogram as maad_spectrogram

    # Load audio
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    duration = len(y) / sr

    if duration < window_sec:
        return []

    # Relative path: "species_code/filename.ogg" or "filename.ogg"
    try:
        rel_path = filepath.relative_to(audio_dir).as_posix()
    except ValueError:
        rel_path = filepath.name

    # Full-file librosa features (per-frame arrays)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    sc = librosa.feature.spectral_contrast(S=S, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    # sc shape: (7 bands, T) — take mean across bands per frame
    sc_mean = np.mean(sc, axis=0)  # (T,)

    sf = librosa.feature.spectral_flatness(S=S, hop_length=HOP_LENGTH)
    sf_squeeze = sf.squeeze()  # (T,)

    # Full-file ROI detection
    roi_activity, roi_tn = _compute_roi_mask(y, sr)

    # Window parameters in samples and frames
    window_samples = int(window_sec * sr)
    frames_per_sec = sr / HOP_LENGTH

    rows = []
    t = 0.0
    while t + window_sec <= duration + 1e-6:  # small epsilon for float rounding
        sample_start = int(t * sr)
        sample_end = min(sample_start + window_samples, len(y))
        y_win = y[sample_start:sample_end]

        if len(y_win) < window_samples // 2:
            t += window_step
            continue

        # --- Per-window maad indices (require whole-spectrogram scalars) ---
        # Compute one spectrogram, derive both PSD and amplitude
        try:
            Sxx_psd, tn_w, fn_w, _ = maad_spectrogram(
                y_win, sr, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH, mode="psd"
            )
            ndsi_val, _, _, _ = soundscape_index(Sxx_psd, fn_w)
            Sxx_amp = np.sqrt(Sxx_psd)
            bi_val = bioacoustics_index(Sxx_amp, fn_w)
        except Exception:
            ndsi_val = np.nan
            bi_val = np.nan

        # --- Aggregate librosa per-frame values for this window ---
        frame_start = int(t * frames_per_sec)
        frame_end = int((t + window_sec) * frames_per_sec)
        frame_end = min(frame_end, len(sc_mean))

        if frame_start < len(sc_mean) and frame_start < frame_end:
            sc_win = float(np.mean(sc_mean[frame_start:frame_end]))
            sf_win = float(np.mean(sf_squeeze[frame_start:frame_end]))
        else:
            sc_win = np.nan
            sf_win = np.nan

        # --- ROI activity for this window ---
        roi_frame_start = int(t * frames_per_sec)
        roi_frame_end = int((t + window_sec) * frames_per_sec)
        roi_frame_end = min(roi_frame_end, len(roi_activity))

        if roi_frame_start < len(roi_activity) and roi_frame_start < roi_frame_end:
            roi_det = bool(np.any(roi_activity[roi_frame_start:roi_frame_end]))
        else:
            roi_det = False

        rows.append({
            "file": rel_path,
            "window_start": round(t, 1),
            "ndsi": ndsi_val,
            "spectral_contrast": sc_win,
            "spectral_flatness": sf_win,
            "bioacoustic_index": bi_val,
            "roi_detected": roi_det,
        })

        t += window_step

    return rows


@app.command()
def run(
    config: str = "configs/teacher.yaml",
    source: str = typer.Argument(..., help="'focal' or 'soundscape'"),
    partition_id: int = typer.Argument(..., help="Partition ID (0-indexed)"),
):
    """Compute acoustic indices on a partition of audio files."""
    cfg = load_config(config)
    source_key = "focal_dir" if source == "focal" else "soundscape_dir"
    audio_dir = Path(cfg["data"][source_key]).expanduser()
    num_partitions = cfg["partitions"][source]
    indices_cfg = cfg["models"]["indices"]
    target_sr = indices_cfg["target_sr"]
    window_sec = indices_cfg["window_sec"]
    window_step = indices_cfg["window_step"]

    source_label = "train_audio" if source == "focal" else "train_soundscapes"
    output_dir = Path(cfg["output"]["root"]).expanduser() / "indices" / source_label / "parts"
    output_path = output_dir / f"part_{partition_id:04d}.parquet"
    timing_path = output_dir / "timing" / f"part_{partition_id:04d}.json"

    if output_path.exists():
        print(f"Output already exists: {output_path}, skipping.")
        return

    files = list_audio_files(audio_dir, source)
    partition = partition_files(files, num_partitions, partition_id)

    if not partition:
        print(f"No files for partition {partition_id}. Skipping.")
        return

    num_workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), len(partition))

    print(f"Indices {source} partition {partition_id}/{num_partitions}")
    print(f"  Total files: {len(files)}, this partition: {len(partition)}")
    print(f"  Window: {window_sec}s, step: {window_step}s, sr: {target_sr}")
    print(f"  Workers: {num_workers}")

    start = time.time()
    all_rows = []
    errors = 0
    done = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_file, fpath, audio_dir, target_sr, window_sec, window_step): fpath
            for fpath in partition
        }
        for future in as_completed(futures):
            fpath = futures[future]
            done += 1
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception:
                errors += 1
                traceback.print_exc()
                print(f"  ERROR processing {fpath.name}, skipping.")

            if done % 20 == 0:
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (len(partition) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(partition)}] {len(all_rows)} rows, "
                      f"{rate:.1f} files/s, ~{remaining:.0f}s left")

    elapsed = time.time() - start

    # Save output
    ensure_output_dir(output_dir)
    ensure_output_dir(timing_path.parent)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(output_path, index=False)
        print(f"  Saved {len(df)} rows to {output_path}")
    else:
        # Write empty parquet with correct schema
        df = pd.DataFrame(columns=[
            "file", "window_start", "ndsi", "spectral_contrast",
            "spectral_flatness", "bioacoustic_index", "roi_detected",
        ])
        df.to_parquet(output_path, index=False)
        print(f"  No rows produced (all files empty/errored).")

    with open(timing_path, "w") as f:
        json.dump({
            "partition_id": partition_id,
            "num_files": len(partition),
            "num_rows": len(all_rows),
            "errors": errors,
            "elapsed_s": round(elapsed, 2),
        }, f, indent=2)

    print(f"  Done in {elapsed:.1f}s — {len(partition)} files, {errors} errors")


def main():
    app()


if __name__ == "__main__":
    main()
