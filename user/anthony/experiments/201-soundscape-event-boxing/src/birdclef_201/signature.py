"""Per-sonotype insect signatures (v1): a frequency band + in-band spectral shape.

A stationary insect is well described by *where* it sits (the discriminative
band) and the *shape* of its in-band power spectrum. We deliberately keep the
signature compact (no FM templates, no EM) — that is the whole reason insects are
the easy first target. Signatures are built from the persisted window spectra,
optionally restricted to a window subset (e.g. S22 only) for the cross-site test.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

from birdclef_201.boxes import discriminative_band, load_window_spectra
from birdclef_201.labels import insect_sonotypes
from birdclef_201.paths import artifact_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsectSignature:
    label: str
    f_lo: float
    f_hi: float
    band_spectrum: np.ndarray  # L2-normalized in-band shape
    n_support: int

    def band_mask(self, freqs: np.ndarray) -> np.ndarray:
        return (freqs >= self.f_lo) & (freqs <= self.f_hi)


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def build_signature(label: str, info: dict, freqs: np.ndarray) -> InsectSignature | None:
    """From a ``discriminative_band`` result, take the dominant band + its
    present-consensus in-band shape (L2-normalized)."""
    if info["status"] != "ok" or not info["bands"]:
        return None
    f_lo, f_hi = info["bands"][0]
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return None
    shape = _l2(np.clip(info["present_spectrum"][mask], 0.0, None))
    return InsectSignature(label=label, f_lo=f_lo, f_hi=f_hi, band_spectrum=shape, n_support=info["n_present"])


def build_signatures(
    *, window_subset: set[str] | None = None
) -> dict[str, InsectSignature]:
    """Build a signature per usable insect sonotype.

    ``window_subset`` restricts both the present and absent pools (pass S22
    window-ids to build site-specific signatures for the cross-site test).
    """
    spectra, wide, freqs = load_window_spectra()
    if window_subset is not None:
        spectra = {w: v for w, v in spectra.items() if w in window_subset}
        wide = wide.filter(pl.col("window_id").is_in(list(window_subset)))
    sigs: dict[str, InsectSignature] = {}
    for label in insect_sonotypes():
        info = discriminative_band(label, wide, spectra)
        sig = build_signature(label, info, freqs)
        if sig is not None:
            sigs[label] = sig
    return sigs


def save_signatures(sigs: dict[str, InsectSignature], name: str = "signatures") -> None:
    out = artifact_dir("signatures")
    rows, npz = [], {}
    for s in sigs.values():
        rows.append({"label": s.label, "f_lo": s.f_lo, "f_hi": s.f_hi, "n_support": s.n_support})
        npz[s.label] = s.band_spectrum
        npz[f"{s.label}::band"] = np.array([s.f_lo, s.f_hi])
    pl.DataFrame(rows).write_csv(out / f"{name}.csv")
    np.savez(out / f"{name}.npz", **npz)
    logger.info("saved %d signatures -> %s", len(sigs), out / f"{name}.csv")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sigs = build_signatures()
    save_signatures(sigs)
    print(pl.read_csv(artifact_dir("signatures") / "signatures.csv"))


if __name__ == "__main__":
    main()
