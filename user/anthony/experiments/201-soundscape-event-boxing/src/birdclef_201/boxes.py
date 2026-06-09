"""Time-frequency box proposal + per-sonotype discriminative bands + attribution.

One pass over the labeled windows computes, per window, a PCEN spectrogram from
which we derive both (a) candidate time-frequency boxes (maad connected
components) and (b) a per-bin spectral summary. The summaries feed the
present-vs-absent **discriminative band** per insect sonotype; the bands feed a
tentative box->label **attribution**. All three are weak/auto outputs to seed
the Label Studio curation loop — not ground truth.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass

import matplotlib

warnings.filterwarnings("ignore", category=FutureWarning, module="maad")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from maad import rois, util  # noqa: E402
from scipy.ndimage import binary_closing  # noqa: E402

from birdclef_201.band import (  # noqa: E402
    BandDiscoveryConfig,
    consensus_spectrum,
    discover_bands,
    per_recording_mean_spectrum,
)
from birdclef_201.geom import interval_iou  # noqa: E402
from birdclef_201.labels import (  # noqa: E402
    insect_sonotypes,
    load_windows,
)
from birdclef_201.paths import artifact_dir  # noqa: E402
from birdclef_201.spectrogram import (  # noqa: E402
    MEL_N_FFT,
    read_soundscape_window,
    wideband_pcen_spectrogram,
)

logger = logging.getLogger(__name__)

MIN_PRESENT = 8  # sonotypes below this have unreliable discriminative bands


@dataclass(frozen=True)
class BoxConfig:
    seed_percentile: float = 0.99
    grow_percentile: float = 0.90
    closing_freq_bins: int = 1
    closing_time_bins: int = 3
    min_area_pixels: int = 6
    min_duration_sec: float = 0.05
    max_duration_sec: float = 6.0
    min_bandwidth_hz: float = 50.0
    max_bandwidth_hz: float = 4000.0


def _hysteresis_thresholds(score: np.ndarray, cfg: BoxConfig) -> tuple[float, float]:
    seed = float(np.quantile(score, cfg.seed_percentile))
    grow = float(np.quantile(score, cfg.grow_percentile))
    if seed <= grow:
        seed = grow + 1e-3
    return seed, grow


def propose_boxes(
    pcen: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    cfg: BoxConfig = BoxConfig(),
    band: tuple[float, float] | None = None,
) -> list[dict]:
    """Hysteresis + connected-component time-frequency boxes from a PCEN patch.

    Distilled from exp 200's ``mining._rois_from_mask`` /
    ``_hysteresis_thresholds`` / ``_propose_band_rois`` into one function (no
    AutoConfig, no band list). ``band`` optionally restricts to a frequency
    sub-range before thresholding. Returns clip-local-time box dicts.
    """
    if band is not None:
        m = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(m):
            return []
        pcen, freqs = pcen[m], freqs[m]
    if pcen.size == 0 or times.size == 0:
        return []
    seed, grow = _hysteresis_thresholds(pcen, cfg)
    im_bin = rois.create_mask(pcen, mode_bin="absolute", bin_h=seed, bin_l=grow)
    if cfg.closing_freq_bins > 1 or cfg.closing_time_bins > 1:
        im_bin = binary_closing(
            im_bin.astype(bool),
            structure=np.ones((cfg.closing_freq_bins, cfg.closing_time_bins), dtype=bool),
        )
    im_rois, rois_df = rois.select_rois(im_bin.astype(bool), min_roi=cfg.min_area_pixels)
    if rois_df.empty:
        return []
    rois_df = util.format_features(rois_df, times, freqs)
    # cell sizes — boxes span cell *edges* (±½ cell) so duration/bandwidth are
    # the true extent, not the (n-1)*step underestimate from bin centers.
    dt = float(times[1] - times[0]) if times.size > 1 else 0.0
    df_hz = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0

    out: list[dict] = []
    for roi_row in rois_df.itertuples(index=False):
        label_id = int(getattr(roi_row, "labelID"))
        rr, cc = np.where(im_rois == label_id)
        if rr.size < cfg.min_area_pixels:
            continue
        f_lo, f_hi = int(rr.min()), int(rr.max())
        t_lo, t_hi = int(cc.min()), int(cc.max())
        start = max(0.0, float(times[t_lo]) - dt / 2)
        end = float(times[t_hi]) + dt / 2
        fmin = max(0.0, float(freqs[f_lo]) - df_hz / 2)
        fmax = float(freqs[f_hi]) + df_hz / 2
        dur, bw = end - start, fmax - fmin
        if not (cfg.min_duration_sec <= dur <= cfg.max_duration_sec):
            continue
        if not (cfg.min_bandwidth_hz <= bw <= cfg.max_bandwidth_hz):
            continue
        vals = pcen[rr, cc]
        pk = int(np.argmax(vals))
        bbox_area = (f_hi - f_lo + 1) * (t_hi - t_lo + 1)
        out.append(
            {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": dur,
                "fmin_hz": fmin,
                "fmax_hz": fmax,
                "bandwidth_hz": bw,
                "peak_time_sec": float(times[cc[pk]]),
                "peak_freq_hz": float(freqs[rr[pk]]),
                "peak_pcen": float(vals.max()),
                "mean_pcen": float(vals.mean()),
                "area_pixels": int(rr.size),
                "fill_ratio": float(rr.size / bbox_area),
            }
        )
    return out


def analyze_labeled_windows(
    wide: pl.DataFrame, *, cfg: BoxConfig = BoxConfig(), limit: int = 0
) -> tuple[pl.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Single pass over labeled windows -> (boxes_df, {window_id: (spectrum, freqs)}).

    For each window: read audio, PCEN, derive candidate boxes (recording-time)
    and a per-bin spectral summary. ``limit`` caps windows (0 = all) for smoke tests.
    """
    rows: list[dict] = []
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    records = (wide.head(limit) if limit else wide).to_dicts()
    for i, w in enumerate(records):
        audio, clip_start, _ = read_soundscape_window(w["filename"], w["start_sec"], w["end_sec"])
        if audio.size < MEL_N_FFT:
            continue
        _, pcen, freqs, times = wideband_pcen_spectrogram(audio)
        spectra[w["window_id"]] = (per_recording_mean_spectrum(pcen), freqs)
        for b in propose_boxes(pcen, freqs, times, cfg=cfg):
            rows.append(
                {
                    "filename": w["filename"],
                    "window_id": w["window_id"],
                    "site": w["site"],
                    "start_sec": clip_start + b["start_sec"],
                    "end_sec": clip_start + b["end_sec"],
                    "fmin_hz": b["fmin_hz"],
                    "fmax_hz": b["fmax_hz"],
                    "peak_time_sec": clip_start + b["peak_time_sec"],
                    "peak_freq_hz": b["peak_freq_hz"],
                    "peak_pcen": b["peak_pcen"],
                    "mean_pcen": b["mean_pcen"],
                    "area_pixels": b["area_pixels"],
                    "fill_ratio": b["fill_ratio"],
                }
            )
        if (i + 1) % 100 == 0:
            logger.info("  analyzed %d/%d windows", i + 1, len(records))
    boxes = pl.DataFrame(rows) if rows else pl.DataFrame()
    return boxes, spectra


def discriminative_band(
    label: str,
    wide: pl.DataFrame,
    spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    min_present: int = MIN_PRESENT,
    cfg: BandDiscoveryConfig | None = None,
) -> dict:
    """Present-vs-absent consensus-spectrum contrast -> a sonotype's band(s).

    Absent windows are matched to the **same files** that contain the label
    elsewhere (falling back to same-site), so the contrast subtracts that site's
    stationary background rather than capturing it. Gated at ``min_present``.
    """
    cfg = cfg or BandDiscoveryConfig(max_bands=2)
    has = pl.col("labels").list.contains(label)
    present = wide.filter(has)
    present_ids = [w for w in present["window_id"].to_list() if w in spectra]
    n_present = len(present_ids)
    if n_present < min_present:
        return {"label": label, "bands": [], "n_present": n_present, "n_absent": 0,
                "status": "insufficient"}

    files_with = set(present["filename"].to_list())
    absent = wide.filter(pl.col("filename").is_in(files_with) & ~has)
    if absent.height < min_present:  # fall back to same-site negatives
        sites_with = set(present["site"].to_list())
        absent = wide.filter(pl.col("site").is_in(sites_with) & ~has)
    absent_ids = [w for w in absent["window_id"].to_list() if w in spectra]

    pres_specs = [spectra[w][0] for w in present_ids]
    freqs = spectra[present_ids[0]][1]
    pres_con = consensus_spectrum(pres_specs)
    abs_con = consensus_spectrum([spectra[w][0] for w in absent_ids]) if absent_ids else np.zeros_like(pres_con)
    contrast = pres_con - abs_con
    bands = discover_bands(contrast, freqs, config=cfg)
    return {
        "label": label,
        "bands": bands,
        "freqs": freqs,
        "present_spectrum": pres_con,
        "contrast": contrast,
        "n_present": n_present,
        "n_absent": len(absent_ids),
        "status": "ok",
    }


def attribute_boxes(
    boxes: pl.DataFrame,
    wide: pl.DataFrame,
    label_bands: dict[str, list[tuple[float, float]]],
    insect_set: set[str],
    *,
    tau: float = 0.1,
) -> pl.DataFrame:
    """Tentatively assign each box to one of its window's labels by band overlap.

    For each box, score each present label by the max frequency IoU between the
    box and that label's discriminative band(s); assign the argmax (insects break
    ties) if it clears ``tau``, else ``unattributed``. ``ambiguous`` marks boxes
    whose top-2 label scores are within 20% — the human-priority items.
    """
    if boxes.is_empty():
        return boxes
    win_labels = {w["window_id"]: list(w["labels"]) for w in wide.to_dicts()}
    out: list[dict] = []
    for b in boxes.to_dicts():
        labels = win_labels.get(b["window_id"], [])
        scored: list[tuple[float, bool, str]] = []
        for lab in labels:
            bands = label_bands.get(lab, [])
            ov = max((interval_iou(b["fmin_hz"], b["fmax_hz"], lo, hi) for lo, hi in bands), default=0.0)
            scored.append((ov, lab in insect_set, lab))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top = scored[0] if scored else (0.0, False, None)
        attributed = top[2] if top[0] >= tau else "unattributed"
        runner = scored[1][0] if len(scored) > 1 else 0.0
        ambiguous = runner > 0 and (top[0] - runner) < 0.2 * max(top[0], 1e-9)
        out.append(
            {
                **b,
                "attributed_label": attributed,
                "attr_score": round(top[0], 4),
                "is_insect": attributed in insect_set,
                "ambiguous": bool(ambiguous),
            }
        )
    return pl.DataFrame(out)


def _plot_contrast(band_info: dict, out_png) -> None:
    freqs = band_info["freqs"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(freqs, band_info["present_spectrum"], color="#888", lw=1, label="present consensus")
    ax.plot(freqs, band_info["contrast"], color="#e63946", lw=1.4, label="present - absent")
    ax.axhline(0, color="k", lw=0.5)
    for lo, hi in band_info["bands"]:
        ax.axvspan(lo, hi, color="#2a9d8f", alpha=0.25)
    ax.set_title(
        f"{band_info['label']}  (n_present={band_info['n_present']}, "
        f"n_absent={band_info['n_absent']})  bands={band_info['bands']}",
        fontsize=8,
    )
    ax.set_xlabel("Hz", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def _run(limit: int) -> None:
    long, wide = load_windows()
    son = insect_sonotypes()
    son_set = set(son)
    out = artifact_dir("boxes")
    bands_dir = artifact_dir("boxes/bands")

    logger.info("analyzing %d labeled windows ...", wide.height if not limit else limit)
    boxes, spectra = analyze_labeled_windows(wide, limit=limit)
    boxes.write_parquet(out / "labeled_boxes.parquet")
    logger.info("proposed %d boxes over %d windows", boxes.height, len(spectra))

    band_rows = []
    label_bands: dict[str, list[tuple[float, float]]] = {}
    for label in son:
        info = discriminative_band(label, wide, spectra)
        label_bands[label] = info["bands"]
        band_rows.append(
            {
                "label": label,
                "status": info["status"],
                "n_present": info["n_present"],
                "n_absent": info["n_absent"],
                "n_bands": len(info["bands"]),
                "bands_hz": str(info["bands"]),
            }
        )
        if info["status"] == "ok":
            _plot_contrast(info, bands_dir / f"{label}.png")
    pl.DataFrame(band_rows).write_parquet(out / "discriminative_bands.parquet")
    pl.DataFrame(band_rows).write_csv(out / "discriminative_bands.csv")

    attributed = attribute_boxes(boxes, wide, label_bands, son_set)
    if not attributed.is_empty():
        attributed.write_parquet(out / "attributed_boxes.parquet")
        summary = (
            attributed.group_by("attributed_label").len().sort("len", descending=True)
        )
        logger.info("attribution: %s", summary.head(12).to_dicts())
    logger.info("wrote box artifacts under %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="cap windows (0 = all; smoke test)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _run(args.limit)


if __name__ == "__main__":
    main()
