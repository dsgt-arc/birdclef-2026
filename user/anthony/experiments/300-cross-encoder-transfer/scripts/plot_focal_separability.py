"""PaCMAP + kNN species-separability of the 6 deployed encoders on FOCAL training clips.

Focal recordings are ~single-species, so this is a far cleaner separability probe than
the frog-dominated labeled soundscapes. Per encoder we read the deployed-variant focal
embedding cache, **mean-pool each recording's windows** to one clip-level vector, cap
recordings/species for balance, standardize, then:
  - PaCMAP 2D, colored by taxonomic class (taxonomy.csv class_name)
  - leave-one-out kNN species accuracy (k) macro-averaged over species  [the headline #]
  - leave-one-out kNN class accuracy (coarse)

Outputs (artifacts/):
  focal_pacmap_grid.png        2x3 grid, panels ordered by LB public, class-colored
  focal_pacmap_coords.parquet  per-recording 2D coords + labels
  focal_separability.json      kNN species/class accuracy per encoder

Standalone: polars, scikit-learn, pacmap, matplotlib, numpy.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import polars as pl
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SCR = Path.home() / "scratch/birdclef-2026/features"
# (encoder, focal dir, deployed-pooling label, LB public)
FOCAL = [
    ("perch", SCR / "exp106/focal", "mean-patch", 0.8656),
    ("birdnet", SCR / "exp300-deploy/birdnet/focal", "mean-patch", 0.8143),
    ("eat", SCR / "exp300-deploy/eat/focal", "CLS (ft)", 0.7796),
    ("ast", SCR / "exp300-deploy/ast/focal", "CLS (10s)", 0.7780),
    ("beats", SCR / "exp300-deploy/beats/focal", "mean-patch", 0.7771),
    ("ssamba_hf", SCR / "exp300-deploy/ssamba_hf/focal", "mean-patch", 0.7308),
]
DISPLAY = {"perch": "Perch", "birdnet": "BirdNET", "eat": "EAT",
           "ast": "AST", "beats": "BEATs", "ssamba_hf": "SSAMBA"}
CLASS_COLORS = {"Aves": "#ff7f0e", "Amphibia": "#1f77b4", "Insecta": "#2ca02c",
                "Mammalia": "#d62728", "Reptilia": "#9467bd"}
CLASS_ORDER = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]


def good_shards(d: Path):
    out = []
    for s in sorted(glob.glob(str(d / "*.parquet"))):
        try:
            pl.scan_parquet(s).select(pl.len()).collect()
            out.append(s)
        except Exception as e:  # corrupt/truncated shard
            print(f"  skip corrupt shard {Path(s).name}: {e}")
    return out


def sample_recordings(lf, cap: int, seed: int):
    """Pick up to `cap` recordings per species (seeded) — returns filename set + meta."""
    meta = (
        lf.select(["filename", "primary_label"]).unique().collect()
    )
    rng = np.random.default_rng(seed)
    keep = []
    for sp, sub in meta.group_by("primary_label"):
        files = sub["filename"].to_list()
        if len(files) > cap:
            files = list(rng.choice(files, size=cap, replace=False))
        keep.extend(files)
    return set(keep)


def load_pooled(d: Path, cap: int, seed: int):
    shards = good_shards(d)
    lf = pl.scan_parquet(shards)
    keep = sample_recordings(lf, cap, seed)
    df = (
        lf.filter(pl.col("filename").is_in(list(keep)))
        .select(["filename", "primary_label", "embedding"])
        .collect()
    )
    # mean-pool windows -> one vector per recording, preserving species
    X = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    files = df["filename"].to_numpy()
    order = np.argsort(files, kind="stable")
    files, X = files[order], X[order]
    uniq, idx, counts = np.unique(files, return_index=True, return_counts=True)
    pooled = np.add.reduceat(X, idx, axis=0) / counts[:, None]
    sp_map = dict(zip(df["filename"].to_list(), df["primary_label"].to_list()))
    species = np.array([sp_map[f] for f in uniq])
    return uniq, pooled, species


def knn_loo(X: np.ndarray, labels: np.ndarray, k: int):
    """Leave-one-out kNN accuracy, macro-averaged over labels."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    ind = ind[:, 1:]  # drop self
    neigh = labels[ind]
    # majority vote per row
    preds = []
    for row in neigh:
        vals, cnts = np.unique(row, return_counts=True)
        preds.append(vals[cnts.argmax()])
    preds = np.array(preds)
    correct = preds == labels
    # macro over labels
    accs = [correct[labels == c].mean() for c in np.unique(labels)]
    return float(np.mean(accs)), float(correct.mean())  # macro, micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--cap", type=int, default=60, help="max recordings per species")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tax = (pl.read_csv(args.taxonomy).with_columns(pl.col("primary_label").cast(pl.Utf8))
           .select(["primary_label", "class_name"]))
    cls_of = dict(zip(tax["primary_label"].to_list(), tax["class_name"].to_list()))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()
    coord_frames, metrics = [], {}

    for ax, (enc, d, pool, lb) in zip(axes, FOCAL):
        print(f"[{enc}] pooling + projecting from {d} ...", flush=True)
        files, X, species = load_pooled(d, args.cap, args.seed)
        classes = np.array([cls_of.get(s, "Unknown") for s in species])
        Xs = StandardScaler().fit_transform(X)

        sp_macro, sp_micro = knn_loo(Xs, species, args.k)
        cl_macro, cl_micro = knn_loo(Xs, classes, args.k)
        metrics[enc] = {"n_recordings": int(len(files)), "n_species": int(len(np.unique(species))),
                        "knn_species_macro": sp_macro, "knn_species_micro": sp_micro,
                        "knn_class_macro": cl_macro, "knn_class_micro": cl_micro, "lb_public": lb}
        print(f"  n={len(files)} species_kNN(macro)={sp_macro:.3f} class_kNN(macro)={cl_macro:.3f}")

        Y = pacmap.PaCMAP(n_components=2, random_state=args.seed, verbose=False).fit_transform(Xs, init="pca")
        for cls in CLASS_ORDER:
            m = classes == cls
            if not m.any():
                continue
            ax.scatter(Y[m, 0], Y[m, 1], s=6, alpha=0.5, c=CLASS_COLORS[cls],
                       label=f"{cls} ({int(m.sum())})", edgecolors="none")
        ax.set_title(f"{DISPLAY[enc]} ({pool}, {X.shape[1]}-d) · LB {lb:.3f} · "
                     f"kNN sp {sp_macro:.2f}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=7, framealpha=0.85, markerscale=2)

        coord_frames.append(pl.DataFrame({
            "filename": files, "primary_label": species, "class_name": classes,
            "pacmap_x": Y[:, 0], "pacmap_y": Y[:, 1], "encoder": enc}))

    fig.suptitle("PaCMAP of deployed encoder embeddings on FOCAL clips "
                 f"(mean-pooled per recording, ≤{args.cap}/species, colored by class; "
                 "kNN sp = leave-one-out species macro-acc)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    gp = out_dir / "focal_pacmap_grid.png"
    fig.savefig(gp, dpi=150, bbox_inches="tight")
    print(f"wrote {gp}")

    pl.concat(coord_frames).write_parquet(out_dir / "focal_pacmap_coords.parquet")
    (out_dir / "focal_separability.json").write_text(json.dumps(metrics, indent=2))
    print("wrote focal_pacmap_coords.parquet + focal_separability.json")
    print("\n=== kNN species macro-acc (separability ranking) ===")
    for enc, m in sorted(metrics.items(), key=lambda kv: -kv[1]["knn_species_macro"]):
        print(f"  {DISPLAY[enc]:8s} {m['knn_species_macro']:.3f}  (LB {m['lb_public']:.3f}, n={m['n_recordings']})")


if __name__ == "__main__":
    main()
