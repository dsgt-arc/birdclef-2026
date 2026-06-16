"""PaCMAP 2D projection of the 6 deployed encoders on the labeled soundscapes.

For each encoder we read its DEPLOYED-variant frozen-embedding cache (one pooled
vector per 5 s labeled window), standardize, and run PaCMAP independently (dims
differ per model: Perch 1536, BirdNET 1024, AST/EAT/BEATs 768, SSAMBA 384). Points
are colored by taxonomic class (taxonomy.csv `class_name`) joined on primary_label.

Outputs (artifacts/):
  pacmap_embeddings_grid.png   2x3 grid, panels ordered by LB public, class-colored
  pacmap_coords.parquet        per-window 2D coords for re-plotting without recompute

Standalone: needs polars, scikit-learn, pacmap, matplotlib, numpy only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import polars as pl
from sklearn.preprocessing import StandardScaler

# (encoder, deployed variant, LB public) — order = LB-descending panel order.
DEPLOYED = [
    ("perch", "frozen", 0.8656),
    ("birdnet", "frozen", 0.8143),
    ("eat", "frozen-cls-ft", 0.7796),
    ("ast", "frozen-cls", 0.7780),
    ("beats", "frozen", 0.7771),
    ("ssamba_hf", "frozen", 0.7308),
]
DISPLAY = {
    "perch": "Perch", "birdnet": "BirdNET", "eat": "EAT",
    "ast": "AST", "beats": "BEATs", "ssamba_hf": "SSAMBA",
}
POOL = {  # human label for the pooling actually deployed
    "frozen": "mean-patch", "frozen-cls": "CLS", "frozen-cls-ft": "CLS (ft)",
}
# fixed class -> color so panels are comparable
CLASS_COLORS = {
    "Amphibia": "#1f77b4", "Insecta": "#2ca02c", "Mammalia": "#d62728",
    "Aves": "#ff7f0e", "Reptilia": "#9467bd",
}
CLASS_ORDER = ["Amphibia", "Insecta", "Mammalia", "Aves", "Reptilia"]


def load(emb_base: Path, tax: pl.DataFrame, enc: str, var: str):
    p = emb_base / f"encoder={enc}/variant={var}/source=soundscape-labeled/data.parquet"
    df = pl.read_parquet(p).with_columns(pl.col("primary_label").cast(pl.Utf8))
    df = df.join(tax, on="primary_label", how="left")
    X = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    classes = df["class_name"].fill_null("Unknown").to_list()
    return df, X, classes


def project(X: np.ndarray, seed: int) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    reducer = pacmap.PaCMAP(n_components=2, random_state=seed, verbose=False)
    return reducer.fit_transform(Xs, init="pca")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-base", default=str(Path.home() / "scratch/birdclef-2026/features/exp300/embeddings"))
    ap.add_argument("--taxonomy", default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    emb_base = Path(args.emb_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tax = (
        pl.read_csv(args.taxonomy)
        .with_columns(pl.col("primary_label").cast(pl.Utf8))
        .select(["primary_label", "class_name"])
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    coord_frames = []

    for ax, (enc, var, lb) in zip(axes, DEPLOYED):
        print(f"[{enc}/{var}] reading + projecting ...", flush=True)
        df, X, classes = load(emb_base, tax, enc, var)
        Y = project(X, args.seed)
        classes = np.array(classes)

        for cls in CLASS_ORDER:
            m = classes == cls
            if not m.any():
                continue
            ax.scatter(
                Y[m, 0], Y[m, 1], s=14, alpha=0.7,
                c=CLASS_COLORS[cls], label=f"{cls} ({int(m.sum())})",
                edgecolors="none",
            )
        ax.set_title(f"{DISPLAY[enc]}  ({POOL[var]}, {X.shape[1]}-d)  ·  LB {lb:.3f}", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=1.3)

        coord_frames.append(
            df.select(["filename", "window_idx", "primary_label", "class_name"]).with_columns(
                pl.Series("pacmap_x", Y[:, 0]),
                pl.Series("pacmap_y", Y[:, 1]),
                pl.lit(enc).alias("encoder"),
                pl.lit(var).alias("variant"),
            )
        )

    fig.suptitle(
        "PaCMAP of frozen encoder embeddings on 739 labeled soundscape windows "
        "(deployed variant per model, colored by taxonomic class)",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = out_dir / "pacmap_embeddings_grid.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    print(f"wrote {grid_path}")

    coords = pl.concat(coord_frames)
    coords_path = out_dir / "pacmap_coords.parquet"
    coords.write_parquet(coords_path)
    print(f"wrote {coords_path}  ({coords.shape[0]} rows)")


if __name__ == "__main__":
    main()
