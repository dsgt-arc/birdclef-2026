"""Phase 3: Search soundscape windows for focal-derived species motifs.

Extracts the most frequent k-mers per species from focal recordings,
then searches soundscape token sequences for matches. Scores against
ground-truth labels from the 739 labeled windows.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

app = typer.Typer()


def _to_supertokens(tokens: np.ndarray, mapping: dict) -> tuple:
    return tuple(mapping[str(int(t))] for t in tokens)


def _extract_kmers(seq: tuple, k: int) -> list:
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def _extract_species_motifs(subset, tokens_dir, mapping, k, top_n=50):
    """Extract top-N most frequent k-mers per species as motifs."""
    motifs = {}
    for sp_id, info in subset.items():
        counter = Counter()
        sp_dir = tokens_dir / sp_id
        for filename in info["files"]:
            stem = Path(filename).stem
            npy_path = sp_dir / f"{stem}.npy"
            if not npy_path.exists():
                continue
            tokens = np.load(npy_path)
            st = _to_supertokens(tokens, mapping)
            kmers = _extract_kmers(st, k)
            counter.update(kmers)

        # Take top-N most frequent k-mers
        motifs[sp_id] = set(kmer for kmer, _ in counter.most_common(top_n))

    return motifs


def _parse_time_to_sec(time_str: str) -> float:
    """Parse HH:MM:SS to seconds."""
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


@app.command()
def main(
    config: str = "configs/exp104.yaml",
    supertoken_k: int = 128,
    kmer_k: int = 5,
    top_motifs: int = 50,
):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg["output"]["root"])
    tokens_dir = output_root / "tokens"
    sc_tokens_dir = output_root / "soundscape_tokens"

    subset_path = output_root / "subset.json"
    with open(subset_path) as f:
        subset = json.load(f)

    mapping_path = output_root / f"supertoken_map_{supertoken_k}.json"
    with open(mapping_path) as f:
        mapping = json.load(f)

    # Load soundscape labels
    labels_df = pd.read_csv(cfg["data"]["soundscape_labels_csv"])
    # Parse start/end to seconds
    labels_df["start_sec"] = labels_df["start"].apply(_parse_time_to_sec)
    labels_df["end_sec"] = labels_df["end"].apply(_parse_time_to_sec)

    # Species in our subset
    subset_species = set(subset.keys())

    # Extract motifs from focal recordings
    print(f"Extracting motifs: K={supertoken_k}, k={kmer_k}, top_motifs={top_motifs}")
    motifs = _extract_species_motifs(subset, tokens_dir, mapping, kmer_k, top_motifs)
    for sp_id, sp_motifs in motifs.items():
        print(f"  {sp_id} ({subset[sp_id]['common_name']}): {len(sp_motifs)} motifs")

    # WavTokenizer frame rate: 24000 / 320 = 75 frames/sec
    frame_rate = 75  # tokens per second

    results = []
    filenames = labels_df["filename"].unique()

    for fname in filenames:
        stem = Path(fname).stem
        npy_path = sc_tokens_dir / f"{stem}.npy"
        if not npy_path.exists():
            continue

        tokens = np.load(npy_path)
        st = _to_supertokens(tokens, mapping)

        # Get windows for this file
        file_labels = labels_df[labels_df["filename"] == fname]

        for _, row in file_labels.iterrows():
            start_sec = row["start_sec"]
            end_sec = row["end_sec"]

            # Extract tokens for this window
            start_frame = int(start_sec * frame_rate)
            end_frame = int(end_sec * frame_rate)
            window_st = st[start_frame:end_frame]
            window_kmers = set(_extract_kmers(window_st, kmer_k))

            if not window_kmers:
                continue

            # Parse ground truth species
            gt_species = set(str(s).strip() for s in str(row["primary_label"]).split(";"))

            # Score each species
            for sp_id, sp_motifs in motifs.items():
                if not sp_motifs:
                    continue
                matches = len(window_kmers & sp_motifs)
                score = matches / len(sp_motifs)

                results.append({
                    "filename": fname,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "species_id": sp_id,
                    "common_name": subset[sp_id]["common_name"],
                    "score": score,
                    "n_matches": matches,
                    "n_motifs": len(sp_motifs),
                    "gt_present": sp_id in gt_species,
                })

    if not results:
        print("\nNo results! Check soundscape tokens exist.")
        return

    df = pd.DataFrame(results)
    output_path = output_root / "soundscape_results.parquet"
    df.to_parquet(output_path, index=False)

    # Analysis
    print(f"\n=== SOUNDSCAPE SEARCH RESULTS ===")
    print(f"Windows scored: {df[['filename', 'start_sec']].drop_duplicates().shape[0]}")
    print(f"Species evaluated: {df['species_id'].nunique()}")

    # Per-species: how well do motif scores separate present vs absent?
    for sp_id in df["species_id"].unique():
        sp_df = df[df["species_id"] == sp_id]
        present = sp_df[sp_df["gt_present"]]
        absent = sp_df[~sp_df["gt_present"]]

        if len(present) > 0 and len(absent) > 0:
            print(f"\n  {sp_id} ({subset[sp_id]['common_name']}):")
            print(f"    Present (n={len(present)}): score mean={present['score'].mean():.4f}")
            print(f"    Absent  (n={len(absent)}):  score mean={absent['score'].mean():.4f}")
            sep = present["score"].mean() - absent["score"].mean()
            print(f"    Separation: {sep:.4f}")

    # Overall: simple threshold-based recall
    for threshold in [0.01, 0.02, 0.05, 0.10]:
        predicted_present = df[df["score"] >= threshold]
        tp = predicted_present[predicted_present["gt_present"]].shape[0]
        fp = predicted_present[~predicted_present["gt_present"]].shape[0]
        total_positive = df[df["gt_present"]].shape[0]
        recall = tp / total_positive if total_positive > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"\n  Threshold {threshold}: precision={precision:.3f}, recall={recall:.3f} "
              f"(TP={tp}, FP={fp}, total_pos={total_positive})")

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    app()
