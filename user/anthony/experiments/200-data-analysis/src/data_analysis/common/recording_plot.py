"""Two-panel recording diagnostic plot, shared across motif pipelines.

Lifted from ``marbled_frog.motif_recording_diagnostics`` so the manual and auto
pipelines render the same layout: full-recording spectrogram on top with
template + envelope event boxes, normalized envelope trace below sharing x.

The manual pipeline re-exports this for backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path

import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import polars as pl

from data_analysis.common.spectrogram import (
    DISPLAY_BAND,
    MEL_HOP_LENGTH,
    MEL_SAMPLE_RATE,
    WIDEBAND,
)


def _by(
    frame: pl.DataFrame, col: str, value: object, *, sort: str | None = None
) -> pl.DataFrame:
    """Equality-filter ``frame`` on ``col``, tolerating an empty/schemaless frame.

    Empty diagnostic inputs (a recording with no envelope peaks, a species with
    no detections) arrive as a 0-column polars frame, where ``pl.col(col)``
    raises ``ColumnNotFoundError``. Return the (empty) frame untouched in that
    case so the caller's ``iter_rows`` loops simply no-op.
    """
    if col not in frame.columns:
        return frame
    out = frame.filter(pl.col(col) == value)
    return out.sort(sort) if sort and sort in out.columns else out


def plot_recording_diagnostic(
    filename: str,
    db_band: np.ndarray,
    z: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    envelope: np.ndarray,
    peak_rows: list[dict],
    events: pl.DataFrame,
    envelope_events: pl.DataFrame,
    output_path: Path,
    *,
    band: tuple[float, float],
    display_ylim: tuple[float, float] = DISPLAY_BAND,
    title_extra: str = "",
) -> None:
    """Plot spectrogram, consolidated template events, and band-envelope peaks.

    Two-panel layout: full-recording spectrogram on top (template event boxes
    colored by cluster, envelope event boxes cyan/red by template-match status,
    vertical lines at envelope peak times), normalized envelope below sharing
    the x-axis.
    """
    cmap = plt.cm.tab10
    width = max(12.0, min(24.0, 0.42 * float(times[-1]) + 6.0))
    fig, (ax_spec, ax_env) = plt.subplots(
        2,
        1,
        figsize=(width, 8.2),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    librosa.display.specshow(
        db_band,
        x_coords=times,
        y_coords=freqs,
        x_axis="time",
        y_axis="linear",
        sr=MEL_SAMPLE_RATE,
        hop_length=MEL_HOP_LENGTH,
        ax=ax_spec,
        cmap="magma",
    )
    ax_spec.axhspan(
        band[0], band[1], color="white", alpha=0.08,
        label=f"diagnostic band {band[0]:.0f}-{band[1]:.0f} Hz",
    )
    env_df = _by(envelope_events, "filename", filename, sort="peak_time_sec")
    for row in env_df.iter_rows(named=True):
        matched = any(
            abs(float(row["peak_time_sec"]) - float(p["envelope_peak_time_sec"])) < 1e-6
            and p["matched_template_event"]
            for p in peak_rows
        )
        ax_spec.add_patch(
            Rectangle(
                (row["start_sec"], row["fmin_hz"]),
                row["end_sec"] - row["start_sec"],
                row["fmax_hz"] - row["fmin_hz"],
                fill=False,
                edgecolor="cyan" if matched else "red",
                linewidth=1.0,
                linestyle=":" if matched else "-",
                alpha=0.85,
            )
        )
    df = _by(events, "filename", filename, sort="peak_time_sec")
    for row in df.iter_rows(named=True):
        cluster = int(row["best_motif_cluster"])
        color = cmap(cluster % 10)
        ax_spec.add_patch(
            Rectangle(
                (row["start_sec"], row["fmin_hz"]),
                row["end_sec"] - row["start_sec"],
                row["fmax_hz"] - row["fmin_hz"],
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                alpha=0.95,
            )
        )
    for peak in peak_rows:
        color = "cyan" if peak["matched_template_event"] else "red"
        ax_spec.axvline(peak["envelope_peak_time_sec"], color=color, linewidth=0.8, alpha=0.65)
    ax_spec.set_ylim(display_ylim)
    base_title = (
        f"Recording diagnostic: {Path(filename).stem}\n"
        "cyan/red boxes=band-envelope motif events (matched/unmatched to template); colored boxes=template events"
    )
    ax_spec.set_title(base_title + (f"\n{title_extra}" if title_extra else ""))
    ax_spec.legend(loc="upper right", fontsize=8)

    env = envelope.astype(float)
    env_norm = (env - np.nanmin(env)) / (np.nanmax(env) - np.nanmin(env) + 1e-9)
    ax_env.plot(times, env_norm, color="black", linewidth=1.1)
    for peak in peak_rows:
        color = "cyan" if peak["matched_template_event"] else "red"
        ax_env.axvline(peak["envelope_peak_time_sec"], color=color, linewidth=0.8, alpha=0.7)
    for row in df.iter_rows(named=True):
        ax_env.axvline(row["peak_time_sec"], color=cmap(int(row["best_motif_cluster"]) % 10), linewidth=1.0, alpha=0.65)
    ax_env.set_ylim(-0.02, 1.05)
    ax_env.set_ylabel("norm envelope")
    ax_env.set_xlabel("Time (s)")
    ax_env.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_motif_activation_diagnostic(
    motif_cluster: int,
    support: dict,
    threshold: float,
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    score_tracks: pl.DataFrame,
    events: pl.DataFrame,
    output_path: Path,
    *,
    member_scores: np.ndarray | None = None,
    background_scores: np.ndarray | None = None,
    dprime: float = float("nan"),
    auc: float = float("nan"),
    title_extra: str = "",
) -> None:
    """Per-motif-family activation diagnostic across all focal recordings.

    Top panel: EM-member vs background score histograms with the detection
    threshold and the d'/AUC separation (a label-free template-quality
    screen). Then each recording is a two-panel block: spectrogram zoomed to
    the family's template band with a box around every detected activation
    (`events`), and the dense template score track below with the threshold
    and detected peaks marked. Recordings where the family never fires are
    still shown (flat track) so misses are visible.
    """
    member_scores = (
        np.asarray([] if member_scores is None else member_scores, dtype=float)
    )
    background_scores = (
        np.asarray([] if background_scores is None else background_scores, dtype=float)
    )
    st = _by(score_tracks, "motif_cluster", motif_cluster)
    ev = _by(events, "motif_cluster", motif_cluster)
    files = sorted(cache.keys())
    n = len(files)

    fmin = float(support["template_fmin_hz"])
    fmax = float(support["template_fmax_hz"])
    pad = max(80.0, 0.30 * (fmax - fmin))
    ylim = (max(WIDEBAND[0], fmin - pad), min(WIDEBAND[1], fmax + pad))

    # Shared score y-range across recordings (comparable panels): span the
    # track's range and the threshold, with a small margin.
    if st.height:
        s_lo = min(float(st["score"].min()), threshold)
        s_hi = max(float(st["score"].max()), threshold)
    else:
        s_lo, s_hi = threshold - 0.1, threshold + 0.1
    s_margin = 0.05 * (s_hi - s_lo + 1e-6)
    score_ylim = (s_lo - s_margin, s_hi + s_margin)

    max_dur = max((float(cache[f][3][-1]) for f in files if cache[f][3].size), default=10.0)
    width = max(12.0, min(24.0, 0.42 * max_dur + 6.0))
    fig, axes = plt.subplots(
        2 * n + 1, 1, figsize=(width, 3.1 * n + 2.6), dpi=170,
        gridspec_kw={"height_ratios": [2.2] + [3.0, 1.4] * n},
    )

    # Top: label-free template-quality screen — member vs background score
    # separation. Low d'/AUC => template does not separate its own picks from
    # noise, so its activations are unreliable regardless of recall.
    ax_hist = axes[0]
    if member_scores.size and background_scores.size:
        lo = float(min(member_scores.min(), background_scores.min()))
        hi = float(max(member_scores.max(), background_scores.max()))
        bins = np.linspace(lo, hi, 60)
        ax_hist.hist(background_scores, bins=bins, density=True, color="grey",
                     alpha=0.55, label=f"background (n={background_scores.size})")
        ax_hist.hist(member_scores, bins=bins, density=True, color="seagreen",
                     alpha=0.7, label=f"EM members (n={member_scores.size})")
    ax_hist.axvline(threshold, color="red", linestyle="--", linewidth=1.1,
                    label=f"threshold {threshold:.2f}")
    ax_hist.set_xlabel("template match score", fontsize=8)
    ax_hist.set_ylabel("density", fontsize=8)
    ax_hist.legend(fontsize=7, loc="upper right")
    sep = "  ".join(
        s for s in (
            "" if np.isnan(dprime) else f"d'={dprime:.2f}",
            "" if np.isnan(auc) else f"AUC={auc:.3f}",
        ) if s
    )
    ax_hist.set_title(
        f"Motif activation diagnostic — family {motif_cluster:04d} "
        f"({fmin:.0f}-{fmax:.0f} Hz, thr={threshold:.2f})   {sep}\n"
        "member vs background separation = label-free template-quality screen; "
        "green box = detected activation"
        + (f"\n{title_extra}" if title_extra else ""),
        fontsize=10,
    )

    n_det_total = 0
    for i, filename in enumerate(files):
        ax_spec, ax_score = axes[1 + 2 * i], axes[2 + 2 * i]
        db_band, _z, freqs, times = cache[filename]
        librosa.display.specshow(
            db_band, x_coords=times, y_coords=freqs, x_axis="time",
            y_axis="linear", sr=MEL_SAMPLE_RATE, hop_length=MEL_HOP_LENGTH,
            ax=ax_spec, cmap="magma",
        )
        ax_spec.axhspan(fmin, fmax, color="white", alpha=0.08)
        det = _by(ev, "filename", filename, sort="peak_time_sec")
        n_det_total += det.height
        for row in det.iter_rows(named=True):
            ax_spec.add_patch(
                Rectangle(
                    (row["start_sec"], row["fmin_hz"]),
                    row["end_sec"] - row["start_sec"],
                    row["fmax_hz"] - row["fmin_hz"],
                    fill=False, edgecolor="lime", linewidth=1.4, alpha=0.95,
                )
            )
            ax_spec.text(
                row["peak_time_sec"], row["fmax_hz"], f"{row['score']:.2f}",
                color="lime", fontsize=7, ha="center", va="bottom",
            )
        ax_spec.set_ylim(ylim)
        ax_spec.set_ylabel(f"{Path(filename).stem}\nHz", fontsize=8)

        trk = _by(st, "filename", filename, sort="time_sec")
        if trk.height:
            ax_score.fill_between(
                trk["time_sec"], score_ylim[0], trk["score"],
                color="steelblue", alpha=0.35, linewidth=0,
            )
            ax_score.plot(trk["time_sec"], trk["score"], color="navy", linewidth=0.8)
        ax_score.axhline(
            threshold, color="red", linestyle="--", linewidth=1.1, alpha=0.9,
        )
        if det.height:
            ax_score.scatter(
                det["peak_time_sec"], det["score"], color="lime",
                edgecolor="black", s=26, zorder=5,
            )
        ax_score.set_xlim(0.0, float(times[-1]) if times.size else 1.0)
        ax_spec.set_xlim(ax_score.get_xlim())
        ax_score.set_ylim(score_ylim)
        ax_score.set_ylabel("score", fontsize=8)
        ax_score.grid(alpha=0.25)
        if i == n - 1:
            ax_score.set_xlabel("Time (s)")

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
