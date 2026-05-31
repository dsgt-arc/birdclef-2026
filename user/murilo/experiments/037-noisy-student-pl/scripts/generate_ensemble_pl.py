"""Exp 037 — generate soft pseudo-labels from the 0.929 ensemble teacher.

Pure transform over two ALREADY-CACHED teacher outputs (no model inference):
  - Perch (TF, 0.925-quality) 234-class probs:
      shared/inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet
  - SED v5j-4fold raw logits (4 folds):
      shared/inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes/fold{0..3}/logits.parquet

Pipeline (the 0.929 ensemble composition, prob-space):
    sed_prob   = mean_folds(sigmoid(sed_logits))
    blend      = w_perch * perch_prob + (1 - w_perch) * sed_prob          # w_perch=0.5
    pl         = transform(blend)                                          # T (temperature) and/or gamma (power)

**Inverts every exp-033 failure knob:** no sharpening (T>=1, gamma<=1 only), NO filtering
(no chunk-discard, no per-class trim — they HURT -0.027 in exp 033), strong ensemble teacher
(not own-SED). Softening direction is uncertain a priori (Perch probe is diffuse, mean~0.47;
SED is peaked) → we generate a small grid {raw, t2, g05} and LB-arbitrate.

Output schema (consumed by PLDS): `filename, chunk_idx, <234 species cols ∈ [0,1]>`, one row
per 5s window over all 10,658 soundscapes (the datamodule's `exclude_labeled_sc=true` drops the
66 labeled files at train time, so we keep all rows here for reuse).

Usage (CPU, ~minutes):
    python scripts/generate_ensemble_pl.py --variant raw          # T=1.0, gamma=1.0 (faithful blend)
    python scripts/generate_ensemble_pl.py --variant t2           # T=2.0 (temperature-soften)
    python scripts/generate_ensemble_pl.py --variant g05          # gamma=0.5 (power-soften, deep-research)
    python scripts/generate_ensemble_pl.py --temperature 1.5 --gamma 1.0 --out-name v037-ens-t15
    python scripts/generate_ensemble_pl.py --variant raw --limit 5 --out-name v037-smoke   # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Paths + constants
# ----------------------------------------------------------------------
SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
PERCH_PROBS = (
    SHARED / "inference/birdclef-2026/perch-v2-tf/train_soundscapes/perch_234_probs.parquet"
)
SED_DIR = SHARED / "inference/birdclef-2026/hgnet-b0-sed-v5j-4fold/train_soundscapes"
SED_N_FOLDS = 4
PL_OUT_ROOT = SHARED / "pseudo-labels/birdclef-2026"

META_COLS = {"filename", "chunk_idx", "start_sec", "end_sec", "source", "site", "fold", "hour_utc"}
KEY = ["filename", "chunk_idx"]
EPS = 1e-7

# variant -> (temperature, gamma, default out dir name)
VARIANTS = {
    "raw": (1.0, 1.0, "v037-ens-raw"),
    "t2": (2.0, 1.0, "v037-ens-t2"),
    "g05": (1.0, 0.5, "v037-ens-g05"),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def temperature_scale(p: np.ndarray, T: float) -> np.ndarray:
    """sigmoid(logit(p) / T). T>1 pulls toward 0.5 (soften); T=1 identity. T<1 is BANNED (sharpen)."""
    if T == 1.0:
        return p
    return sigmoid(to_logit(p) / T)


def power_transform(p: np.ndarray, gamma: float) -> np.ndarray:
    """p ** gamma. gamma<1 lifts low probs (soften); gamma=1 identity. gamma>1 is BANNED (sharpen)."""
    if gamma == 1.0:
        return p
    return np.power(np.clip(p, 0.0, 1.0), gamma)


def _species_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def load_teacher_blend(w_perch: float, limit: int | None) -> tuple[pd.DataFrame, list[str]]:
    """Return (key_df_with_blend_probs, species_cols). Aligned on (filename, chunk_idx)."""
    # --- Perch probs (already in [0,1]) ---
    perch = pd.read_parquet(PERCH_PROBS)
    perch["chunk_idx"] = perch["chunk_idx"].astype(int)
    species = _species_cols(perch)
    assert len(species) == 234, f"expected 234 perch species cols, got {len(species)}"
    perch = perch.set_index(KEY).sort_index()

    if limit is not None:
        keep_files = sorted(perch.index.get_level_values("filename").unique())[:limit]
        perch = perch[perch.index.get_level_values("filename").isin(keep_files)]
    index = perch.index
    perch_P = perch[species].to_numpy(np.float32)

    # --- SED v5j-4fold: sigmoid each fold's logits, average (prob-space) ---
    sed_sum = np.zeros_like(perch_P)
    for fold in range(SED_N_FOLDS):
        f = SED_DIR / f"fold{fold}" / "logits.parquet"
        if not f.exists():
            raise FileNotFoundError(f"missing SED fold cache: {f}")
        sed = pd.read_parquet(f)
        sed["chunk_idx"] = sed["chunk_idx"].astype(int)
        sed_species = _species_cols(sed)
        assert set(sed_species) == set(species), f"fold{fold} species mismatch vs perch"
        sed = sed.set_index(KEY).reindex(index)  # align to perch key order
        assert not sed[species].isna().any().any(), f"fold{fold} missing windows after align"
        sed_sum += sigmoid(sed[species].to_numpy(np.float32))
    sed_P = sed_sum / SED_N_FOLDS

    blend = w_perch * perch_P + (1.0 - w_perch) * sed_P  # prob-space ensemble (matches 0.929)

    out = pd.DataFrame(blend, columns=species, index=index).reset_index()
    return out, species


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variant",
        choices=sorted(VARIANTS),
        default=None,
        help="Shorthand for (temperature, gamma, out-name). Overridden by explicit flags.",
    )
    ap.add_argument(
        "--temperature", type=float, default=None, help="T>=1 (soften). T<1 banned (sharpen)."
    )
    ap.add_argument(
        "--gamma", type=float, default=None, help="gamma<=1 (soften). gamma>1 banned (sharpen)."
    )
    ap.add_argument(
        "--w-perch", type=float, default=0.5, help="Perch weight in the prob-space blend."
    )
    ap.add_argument(
        "--out-name", type=str, default=None, help="Output dir name under pseudo-labels root."
    )
    ap.add_argument("--out-root", type=Path, default=PL_OUT_ROOT)
    ap.add_argument("--limit", type=int, default=None, help="First N files (smoke test).")
    args = ap.parse_args()

    # Resolve T, gamma, out-name from variant + explicit overrides
    if args.variant:
        T_def, g_def, name_def = VARIANTS[args.variant]
    else:
        T_def, g_def, name_def = 1.0, 1.0, "v037-ens-custom"
    T = args.temperature if args.temperature is not None else T_def
    gamma = args.gamma if args.gamma is not None else g_def
    out_name = args.out_name or name_def

    # Guardrails: this experiment NEVER sharpens (exp 033's documented failure mode).
    if T < 1.0:
        ap.error(f"temperature {T} < 1.0 would SHARPEN (exp 033 failure mode). Use T>=1.")
    if gamma > 1.0:
        ap.error(f"gamma {gamma} > 1.0 would SHARPEN (exp 033 failure mode). Use gamma<=1.")

    print(
        f"Teacher blend: w_perch={args.w_perch} | transform: T={T}, gamma={gamma} | out={out_name}"
    )
    blend_df, species = load_teacher_blend(args.w_perch, args.limit)

    P = blend_df[species].to_numpy(np.float32)
    print(
        f"  blend: rows={len(P):,} files={blend_df['filename'].nunique():,} "
        f"mean={P.mean():.4f} median={np.median(P):.4f} frac>0.5={(P > 0.5).mean():.4f}"
    )

    P = power_transform(temperature_scale(P, T), gamma).astype(np.float32)
    blend_df[species] = P
    print(
        f"  after transform: mean={P.mean():.4f} median={np.median(P):.4f} "
        f"frac>0.5={(P > 0.5).mean():.4f} max={P.max():.4f}"
    )

    assert P.min() >= 0.0 and P.max() <= 1.0, "PL probs out of [0,1]"
    assert list(blend_df.columns[:2]) == KEY, "schema must start with (filename, chunk_idx)"
    assert len(species) == 234

    out_dir = args.out_root / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ensemble_pseudo_labels.parquet"
    blend_df[KEY + species].to_parquet(out_path, index=False)
    print(f"Wrote {len(blend_df):,} rows x {2 + len(species)} cols -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
