"""Geometry: interval IoU + exact spectrogram-pixel <-> (time, freq) mapping.

This is the single source of truth the Label Studio exporter and parser both
call, so the round-trip is lossless. Spectrogram PNGs are written with
``plt.imsave(png, np.flipud(norm(pcen)), cmap=...)`` so 1 image pixel == 1
(freq_bin, time_frame) cell and ``SpecExtent`` is trivial. Label Studio
``RectangleLabels`` use PERCENT coords with origin TOP-LEFT and y increasing
downward; frequency increases UPWARD in the spectrogram, hence the y-flip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def interval_iou(a0: float, a1: float, b0: float, b1: float) -> float:
    """IoU of two 1-D intervals (time or frequency). Ported from exp 200."""
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return float(inter / union) if union > 0 else 0.0


@dataclass(frozen=True)
class SpecExtent:
    """Physical extent of a rendered PCEN PNG (no axes/margins, origin lower).

    ``t0,t1`` = clip-local seconds of column 0 .. last column (``times[0]``,
    ``times[-1]``); ``f0,f1`` = Hz of row 0 .. last row (``freqs[0]``,
    ``freqs[-1]``). ``width_px``/``height_px`` = PNG dims (== ``times.size`` /
    ``freqs.size`` for a 1px-per-cell ``imsave``).
    """

    t0: float
    t1: float
    f0: float
    f1: float
    width_px: int
    height_px: int

    def as_dict(self) -> dict:
        return {
            "t0": self.t0, "t1": self.t1, "f0": self.f0, "f1": self.f1,
            "width_px": self.width_px, "height_px": self.height_px,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpecExtent":
        return cls(
            t0=float(d["t0"]), t1=float(d["t1"]), f0=float(d["f0"]), f1=float(d["f1"]),
            width_px=int(d["width_px"]), height_px=int(d["height_px"]),
        )


def box_to_ls_percent(
    t0: float, t1: float, f0: float, f1: float, ext: SpecExtent
) -> tuple[float, float, float, float]:
    """(time, freq) box -> Label Studio (x, y, width, height) percent, y-flipped."""
    dt = ext.t1 - ext.t0 or 1.0
    df = ext.f1 - ext.f0 or 1.0
    x = 100.0 * (t0 - ext.t0) / dt
    w = 100.0 * (t1 - t0) / dt
    y = 100.0 * (ext.f1 - f1) / df  # top edge = HIGH freq
    h = 100.0 * (f1 - f0) / df
    return x, y, w, h


def ls_percent_to_box(
    x: float, y: float, w: float, h: float, ext: SpecExtent
) -> tuple[float, float, float, float]:
    """Inverse of :func:`box_to_ls_percent`: percent -> (t0, t1, f0, f1)."""
    dt = ext.t1 - ext.t0
    df = ext.f1 - ext.f0
    t0 = ext.t0 + (x / 100.0) * dt
    t1 = t0 + (w / 100.0) * dt
    f1 = ext.f1 - (y / 100.0) * df
    f0 = f1 - (h / 100.0) * df
    return t0, t1, f0, f1


def norm01(a: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0,1] for image rendering (robust 1/99 percentiles)."""
    lo, hi = np.percentile(a, [1, 99]) if a.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
