"""Quantize WavTokenizer codebook to super-tokens via k-means.

Clusters the 4096-entry codebook embeddings into K super-tokens,
creating a reduced alphabet for approximate k-mer matching.
"""

import json
from pathlib import Path

import numpy as np
import typer
import yaml
from sklearn.cluster import KMeans

app = typer.Typer()


@app.command()
def main(config: str = "configs/exp104.yaml"):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg["output"]["root"])
    vocab_sizes = cfg["retrieval"]["supertoken_vocab_sizes"]

    embeddings_path = output_root / "codebook_embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Run tokenize first: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    print(f"Codebook embeddings: {embeddings.shape}")  # (4096, D)

    for K in vocab_sizes:
        print(f"\nClustering to K={K} super-tokens...")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(embeddings)

        # Save mapping: code_id -> super_token_id
        mapping = {int(i): int(label) for i, label in enumerate(kmeans.labels_)}
        mapping_path = output_root / f"supertoken_map_{K}.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f)

        # Save cluster centers for analysis
        centers_path = output_root / f"supertoken_centers_{K}.npy"
        np.save(centers_path, kmeans.cluster_centers_)

        # Print cluster size distribution
        sizes = np.bincount(kmeans.labels_)
        print(f"  Inertia: {kmeans.inertia_:.1f}")
        print(f"  Cluster sizes: min={sizes.min()}, max={sizes.max()}, "
              f"mean={sizes.mean():.1f}, median={np.median(sizes):.1f}")
        print(f"  Saved: {mapping_path}")


if __name__ == "__main__":
    app()
