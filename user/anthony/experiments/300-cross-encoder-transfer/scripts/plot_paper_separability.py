"""Paper figure: encoder embedding-space separability (exp 300, CEUR working note).

A 2x3 grid of PaCMAP projections — one per deployed encoder — of FOCAL training clips
for a curated 8-species subset, colored by species. The figure's only job is to *show*
the separability gap that the kNN table quantifies: bird specialists (Perch, BirdNET)
carve clean per-species islands; AudioSet transformers blur them; SSAMBA collapses to mush.

Design (v2 paper-ready — slim filmstrip):
  - substrate: focal clips, mean-pooled to one vector per recording (single-species)
  - 8 species spanning taxa, incl. 4 birds for the hard within-class case
  - color = species (8 distinct); uniform small circles (s~3, fine point-cloud)
  - preprocessing per exp-200 recipe: StandardScaler -> PCA(<=50) -> PaCMAP
  - robust per-panel axis limits (percentile clip) so fine dots fill each panel
  - faint NxN reference gridlines per panel (decorative; PaCMAP axes carry no scale)
  - minimal text: encoder name only in-panel, no tick labels, one shared legend, no kNN nums
  - 1x6 strip at final print size ~5.79 x 1.9 in = 14.70 x ~4.8 cm (full \textwidth x ~1/5 height)
  - output: vector PDF (for \includegraphics) + 600-DPI PNG fallback + coords parquet

Standalone: polars, scikit-learn, pacmap, matplotlib, numpy.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import polars as pl
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCR = Path.home() / "scratch/birdclef-2026/features"

# Encoders in LB / kNN-descending order (degrades left->right, top->bottom).
ENCODERS = [
    ("perch", "Perch", SCR / "exp106/focal"),
    ("birdnet", "BirdNET", SCR / "exp300-deploy/birdnet/focal"),
    ("ast", "AST", SCR / "exp300-deploy/ast/focal"),
    ("eat", "EAT", SCR / "exp300-deploy/eat/focal"),
    ("beats", "BEATs", SCR / "exp300-deploy/beats/focal"),
    ("ssamba_hf", "SSAMBA", SCR / "exp300-deploy/ssamba_hf/focal"),
]

# Curated subset: (species_code, short_name, class, color). 4 birds first (within-class
# hard case), then one each insect/amphibian + two mammals (cross-taxa contrast).
SUBSET = [
    ("banana", "Bananaquit", "Aves", "#1f77b4"),
    ("rubthr1", "Thrush", "Aves", "#ff7f0e"),
    ("fepowl", "Pygmy Owl", "Aves", "#2ca02c"),
    ("soulap1", "Lapwing", "Aves", "#d62728"),
    ("244024", "Cicada", "Insecta", "#9467bd"),
    ("22973", "Grass Frog", "Amphibia", "#8c564b"),
    ("47144", "Dog", "Mammalia", "#e377c2"),
    ("41970", "Jaguar", "Mammalia", "#17becf"),
]
CLASS_MARKER = {"Aves": "o", "Insecta": "s", "Amphibia": "^", "Mammalia": "D"}
CODES = [s[0] for s in SUBSET]
COLOR = {s[0]: s[3] for s in SUBSET}
MARKER = {s[0]: CLASS_MARKER[s[2]] for s in SUBSET}


def good_shards(d: Path):
    out = []
    for s in sorted(glob.glob(str(d / "*.parquet"))):
        try:
            pl.scan_parquet(s).select(pl.len()).collect()
            out.append(s)
        except Exception:
            print(f"  skip corrupt shard {Path(s).name}")
    return out


def load_subset(d: Path, cap: int, seed: int):
    """Mean-pool each recording's windows -> one vector; cap recordings/species."""
    df = (
        pl.scan_parquet(good_shards(d))
        .filter(pl.col("primary_label").is_in(CODES))
        .select(["filename", "primary_label", "embedding"])
        .collect()
    )
    X = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    files = df["filename"].to_numpy()
    order = np.argsort(files, kind="stable")
    files, X = files[order], X[order]
    uniq, idx, counts = np.unique(files, return_index=True, return_counts=True)
    pooled = np.add.reduceat(X, idx, axis=0) / counts[:, None]
    sp_map = dict(zip(df["filename"].to_list(), df["primary_label"].to_list()))
    species = np.array([sp_map[f] for f in uniq])

    rng = np.random.default_rng(seed)
    keep = []
    for c in CODES:
        ix = np.where(species == c)[0]
        if len(ix) > cap:
            ix = rng.choice(ix, cap, replace=False)
        keep.extend(ix.tolist())
    keep = np.array(sorted(keep))
    return uniq[keep], pooled[keep], species[keep]


def project(X: np.ndarray, seed: int) -> np.ndarray:
    """exp-200 recipe: standardize -> PCA(<=50) -> PaCMAP."""
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    nc = min(50, Xs.shape[0] - 1, Xs.shape[1])
    if nc > 2:
        Xs = PCA(n_components=nc, random_state=seed).fit_transform(Xs).astype(np.float32)
    nn = min(10, max(2, Xs.shape[0] - 1))
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=nn, MN_ratio=0.5, FP_ratio=2.0,
                            random_state=seed, verbose=False)
    return reducer.fit_transform(Xs, init="pca")


def set_rcparams():
    mpl.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--cap", type=int, default=60, help="max recordings per species")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=float, default=1.9, help="figure height in inches")
    ap.add_argument("--marker-size", type=float, default=3.0)
    ap.add_argument("--layout", choices=["strip", "grid"], default="strip",
                    help="strip = 1x6 filmstrip (slim); grid = 2x3")
    ap.add_argument("--grid-divisions", type=int, default=4, help="NxN faint reference grid")
    ap.add_argument("--pct", type=float, default=1.5,
                    help="robust axis-limit percentile (clip outlier whitespace)")
    args = ap.parse_args()

    set_rcparams()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nrow, ncol = (1, 6) if args.layout == "strip" else (2, 3)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.79, args.height))
    axes = axes.ravel()
    coord_frames = []

    for ax, (enc, disp, d) in zip(axes, ENCODERS):
        print(f"[{enc}] {d}", flush=True)
        files, X, species = load_subset(d, args.cap, args.seed)
        counts = {c: int((species == c).sum()) for c in CODES}
        print("  per-species n:", counts, flush=True)
        Y = project(X, args.seed)

        for code, _short, _cls, _col in SUBSET:
            m = species == code
            if not m.any():
                continue
            ax.scatter(Y[m, 0], Y[m, 1], s=args.marker_size, c=COLOR[code],
                       marker="o", alpha=0.75, linewidths=0, edgecolors="none")
        # robust axis limits: clip outlier-driven whitespace so the fine cloud fills the panel
        xlo, xhi = np.percentile(Y[:, 0], [args.pct, 100 - args.pct])
        ylo, yhi = np.percentile(Y[:, 1], [args.pct, 100 - args.pct])
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        # faint NxN reference grid (decorative — PaCMAP axes carry no scale); labels hidden
        nd = args.grid_divisions
        ax.set_xticks(np.linspace(xlo, xhi, nd + 1))
        ax.set_yticks(np.linspace(ylo, yhi, nd + 1))
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.grid(True, color="0.85", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.text(0.04, 0.95, disp, transform=ax.transAxes, fontsize=6, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.8))

        coord_frames.append(pl.DataFrame({
            "filename": files, "primary_label": species,
            "pacmap_x": Y[:, 0], "pacmap_y": Y[:, 1], "encoder": enc}))

    # one shared legend: colored dots (circles) + short species names
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=col,
                      markeredgecolor="none", markersize=4, label=short)
               for code, short, cls, col in SUBSET]
    leg_ncol = 8 if args.layout == "strip" else 4
    fig.legend(handles=handles, loc="lower center", ncol=leg_ncol, fontsize=6,
               frameon=False, handletextpad=0.2, columnspacing=0.9,
               bbox_to_anchor=(0.5, -0.02))

    if args.layout == "strip":
        fig.subplots_adjust(left=0.006, right=0.994, top=0.97, bottom=0.17, wspace=0.06)
    else:
        fig.subplots_adjust(left=0.006, right=0.994, top=0.985, bottom=0.135,
                            wspace=0.04, hspace=0.06)

    pdf = out_dir / "paper_separability.pdf"
    png = out_dir / "paper_separability.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    pl.concat(coord_frames).write_parquet(out_dir / "paper_separability_coords.parquet")
    w, h = fig.get_size_inches()
    print(f"\nwrote {pdf}\nwrote {png}  ({w*2.54:.2f} x {h*2.54:.2f} cm, "
          f"png ~{int(w*600)}px wide @600dpi)")


if __name__ == "__main__":
    main()
