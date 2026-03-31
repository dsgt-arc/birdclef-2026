"""Phase 2: k-mer inverted index and leave-one-out species retrieval.

Maps token sequences to super-tokens, builds k-mer index, and evaluates
whether same-species recordings can be retrieved via sequence matching.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml

app = typer.Typer()


def _load_tokens(tokens_dir: Path, subset: dict) -> dict:
    """Load all token sequences as {(species_id, filename): np.array}."""
    sequences = {}
    for sp_id, info in subset.items():
        sp_dir = tokens_dir / sp_id
        for filename in info["files"]:
            stem = Path(filename).stem
            npy_path = sp_dir / f"{stem}.npy"
            if npy_path.exists():
                sequences[(sp_id, filename)] = np.load(npy_path)
    return sequences


def _to_supertokens(tokens: np.ndarray, mapping: dict) -> tuple:
    """Map raw token IDs to super-token IDs. Returns tuple for hashing."""
    return tuple(mapping[str(int(t))] for t in tokens)


def _extract_kmers(seq: tuple, k: int) -> list:
    """Extract all k-mers from a super-token sequence."""
    if len(seq) < k:
        return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]


def _build_index(all_kmers: dict) -> dict:
    """Build inverted index: kmer -> list of (species_id, filename)."""
    index = defaultdict(list)
    for (sp_id, fname), kmers in all_kmers.items():
        for kmer in set(kmers):  # deduplicate per recording
            index[kmer].append((sp_id, fname))
    return dict(index)


def _leave_one_out_retrieval(
    sequences: dict, mapping: dict, k: int, top_n: int = 10
) -> list:
    """Leave-one-out retrieval: for each recording, query the rest."""
    # Pre-compute super-token sequences and k-mers
    all_supertokens = {}
    all_kmers = {}
    for key, tokens in sequences.items():
        st = _to_supertokens(tokens, mapping)
        all_supertokens[key] = st
        all_kmers[key] = _extract_kmers(st, k)

    # Build full index
    full_index = _build_index(all_kmers)

    results = []
    keys = list(sequences.keys())

    for query_key in keys:
        query_sp, query_fname = query_key
        query_kmers = set(all_kmers[query_key])

        if not query_kmers:
            continue

        # Count shared k-mers with each other recording
        scores = defaultdict(int)
        for kmer in query_kmers:
            if kmer in full_index:
                for hit_key in full_index[kmer]:
                    if hit_key != query_key:
                        scores[hit_key] += 1

        if not scores:
            results.append({
                "query_species": query_sp,
                "query_file": query_fname,
                "precision_at_10": 0.0,
                "n_candidates": 0,
            })
            continue

        # Rank by score, take top-N
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        n_same_species = sum(1 for (sp, _), _ in ranked if sp == query_sp)

        results.append({
            "query_species": query_sp,
            "query_file": query_fname,
            "precision_at_10": n_same_species / min(top_n, len(ranked)),
            "n_candidates": len(scores),
            "top1_species": ranked[0][0][0] if ranked else None,
            "top1_score": ranked[0][1] if ranked else 0,
        })

    return results


@app.command()
def main(config: str = "configs/exp104.yaml"):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    output_root = Path(cfg["output"]["root"])
    vocab_sizes = cfg["retrieval"]["supertoken_vocab_sizes"]
    kmer_lengths = cfg["retrieval"]["kmer_lengths"]

    subset_path = output_root / "subset.json"
    with open(subset_path) as f:
        subset = json.load(f)

    tokens_dir = output_root / "tokens"
    sequences = _load_tokens(tokens_dir, subset)
    print(f"Loaded {len(sequences)} token sequences")

    all_results = []

    for K in vocab_sizes:
        mapping_path = output_root / f"supertoken_map_{K}.json"
        if not mapping_path.exists():
            print(f"SKIP K={K}: mapping not found (run codebook first)")
            continue

        with open(mapping_path) as f:
            mapping = json.load(f)

        for k in kmer_lengths:
            print(f"\nK={K}, k={k}:")
            results = _leave_one_out_retrieval(sequences, mapping, k)

            if not results:
                print("  No results")
                continue

            mean_p10 = np.mean([r["precision_at_10"] for r in results])
            median_p10 = np.median([r["precision_at_10"] for r in results])

            print(f"  Precision@10: mean={mean_p10:.3f}, median={median_p10:.3f}")

            # Per-species breakdown
            by_species = defaultdict(list)
            for r in results:
                by_species[r["query_species"]].append(r["precision_at_10"])
            for sp, vals in sorted(by_species.items()):
                sp_name = subset[sp]["common_name"]
                print(f"    {sp} ({sp_name}): P@10={np.mean(vals):.3f} (n={len(vals)})")

            for r in results:
                r["supertoken_K"] = K
                r["kmer_k"] = k
            all_results.extend(results)

    if all_results:
        df = pd.DataFrame(all_results)
        output_path = output_root / "retrieval_results.parquet"
        df.to_parquet(output_path, index=False)
        print(f"\nSaved: {output_path}")

        # Best config
        summary = df.groupby(["supertoken_K", "kmer_k"])["precision_at_10"].mean()
        best = summary.idxmax()
        print(f"\nBest config: K={best[0]}, k={best[1]} -> P@10={summary[best]:.3f}")

        gate_pass = summary[best] > 0.80
        print(f"\n{'PASS' if gate_pass else 'FAIL'}: "
              f"{'Proceed to Phase 3' if gate_pass else 'Tokens not discriminative enough'}")


if __name__ == "__main__":
    app()
