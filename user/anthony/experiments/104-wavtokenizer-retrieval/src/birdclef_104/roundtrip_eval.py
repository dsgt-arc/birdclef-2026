"""Phase 1: Round-trip quality evaluation using Perch v2 (ONNX).

Runs Perch v2 on original and WavTokenizer-reconstructed audio.
Compares embeddings (cosine similarity) and logits (top-k agreement)
to assess whether the codec preserves species-discriminative features.

Uses: https://huggingface.co/justinchuby/Perch-onnx
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import typer
import yaml
from huggingface_hub import hf_hub_download

app = typer.Typer()

PERCH_SR = 32000
PERCH_WINDOW = 5.0  # seconds
PERCH_SAMPLES = int(PERCH_SR * PERCH_WINDOW)  # 160000


def _load_perch(cache_dir: Path = None):
    """Download and load Perch v2 ONNX model."""
    import onnxruntime as ort

    model_path = hf_hub_download(
        repo_id="justinchuby/Perch-onnx",
        filename="perch_v2_no_dft.onnx",  # broader onnxruntime compatibility
    )
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def _load_audio_32k(path: str, max_samples: int = None) -> np.ndarray:
    """Load audio, resample to 32kHz mono, return float32 array."""
    import torchaudio
    import torch

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio_t = torch.from_numpy(audio).float()
    if sr != PERCH_SR:
        audio_t = torchaudio.functional.resample(audio_t, orig_freq=sr, new_freq=PERCH_SR)
    audio_np = audio_t.numpy()
    if max_samples and len(audio_np) > max_samples:
        audio_np = audio_np[:max_samples]
    return audio_np


def _run_perch_chunks(session, audio: np.ndarray) -> dict:
    """Run Perch on 5s chunks, return pooled embedding and max logits.

    Returns dict with:
      - embedding: (1536,) mean-pooled across chunks
      - logits: (14795,) max-pooled across chunks
      - n_chunks: number of 5s windows
    """
    # Pad to at least one full window
    if len(audio) < PERCH_SAMPLES:
        audio = np.pad(audio, (0, PERCH_SAMPLES - len(audio)))

    # Chunk into 5s non-overlapping windows
    n_chunks = len(audio) // PERCH_SAMPLES
    chunks = audio[:n_chunks * PERCH_SAMPLES].reshape(n_chunks, PERCH_SAMPLES)

    all_embeddings = []
    all_logits = []

    for i in range(0, n_chunks, 8):  # batch up to 8 chunks
        batch = chunks[i:i+8].astype(np.float32)
        outputs = session.run(None, {"inputs": batch})
        all_embeddings.append(outputs[0])  # (B, 1536)
        all_logits.append(outputs[3])      # (B, 14795)

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, 1536)
    logits = np.concatenate(all_logits, axis=0)           # (N, 14795)

    return {
        "embedding": embeddings.mean(axis=0),    # (1536,)
        "logits": logits.max(axis=0),            # (14795,)
        "n_chunks": n_chunks,
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _topk_agreement(logits_a: np.ndarray, logits_b: np.ndarray, k: int = 10) -> float:
    """Fraction of top-k species that agree between two logit vectors."""
    topk_a = set(np.argsort(logits_a)[-k:])
    topk_b = set(np.argsort(logits_b)[-k:])
    return len(topk_a & topk_b) / k


@app.command()
def main(config: str = "configs/exp104.yaml"):
    with open(config) as f:
        cfg = yaml.safe_load(f)

    focal_dir = Path(cfg["data"]["focal_dir"])
    output_root = Path(cfg["output"]["root"])
    recon_dir = output_root / "reconstructed"

    subset_path = output_root / "subset.json"
    if not subset_path.exists():
        raise FileNotFoundError(f"Run select_subset first: {subset_path}")
    with open(subset_path) as f:
        subset = json.load(f)

    if not recon_dir.exists():
        raise FileNotFoundError(f"Run tokenize first: {recon_dir}")

    # Load Perch v2 ONNX
    print("Loading Perch v2 (ONNX)...")
    session = _load_perch()
    print("  Model loaded")

    results = []
    for sp_id, info in subset.items():
        print(f"\n{sp_id}: {info['common_name']} ({info['scientific_name']})")

        for filename in info["files"]:
            original_path = focal_dir / filename
            stem = Path(filename).stem
            reconstructed_path = recon_dir / sp_id / f"{stem}.wav"

            if not original_path.exists() or not reconstructed_path.exists():
                continue

            try:
                # Load and run Perch on original (resample to 32kHz)
                audio_orig = _load_audio_32k(str(original_path))
                perch_orig = _run_perch_chunks(session, audio_orig)

                # Load and run Perch on reconstructed (24kHz -> 32kHz)
                audio_recon = _load_audio_32k(str(reconstructed_path))
                perch_recon = _run_perch_chunks(session, audio_recon)

                # Compare embeddings
                cos_sim = _cosine_similarity(perch_orig["embedding"], perch_recon["embedding"])

                # Compare logits
                top5_agree = _topk_agreement(perch_orig["logits"], perch_recon["logits"], k=5)
                top10_agree = _topk_agreement(perch_orig["logits"], perch_recon["logits"], k=10)

                # Top-1 species agreement
                top1_orig = int(np.argmax(perch_orig["logits"]))
                top1_recon = int(np.argmax(perch_recon["logits"]))
                top1_match = top1_orig == top1_recon

                results.append({
                    "species_id": sp_id,
                    "scientific_name": info["scientific_name"],
                    "common_name": info["common_name"],
                    "class_name": info["class_name"],
                    "filename": filename,
                    "cosine_similarity": cos_sim,
                    "top1_match": top1_match,
                    "top5_agreement": top5_agree,
                    "top10_agreement": top10_agree,
                    "top1_orig_idx": top1_orig,
                    "top1_recon_idx": top1_recon,
                    "n_chunks_orig": perch_orig["n_chunks"],
                    "n_chunks_recon": perch_recon["n_chunks"],
                })

                print(f"  {stem}: cos={cos_sim:.4f} top1={'match' if top1_match else 'DIFF'} "
                      f"top5={top5_agree:.0%} top10={top10_agree:.0%}")

            except Exception as e:
                print(f"  ERROR {filename}: {e}")
                continue

    if not results:
        print("\nNo results! Check that tokenize has been run.")
        return

    df = pd.DataFrame(results)

    # Save results
    output_path = output_root / "roundtrip_scores.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Summary
    print("\n=== ROUND-TRIP QUALITY SUMMARY (Perch v2) ===")
    print(f"Files evaluated: {len(df)}")
    print(f"Embedding cosine similarity: mean={df['cosine_similarity'].mean():.4f}, "
          f"median={df['cosine_similarity'].median():.4f}, "
          f"min={df['cosine_similarity'].min():.4f}")
    print(f"Top-1 species match: {df['top1_match'].mean():.1%}")
    print(f"Top-5 agreement: {df['top5_agreement'].mean():.1%}")
    print(f"Top-10 agreement: {df['top10_agreement'].mean():.1%}")

    print("\nPer-species summary:")
    species_summary = df.groupby(["species_id", "common_name", "class_name"]).agg(
        mean_cos_sim=("cosine_similarity", "mean"),
        top1_match_rate=("top1_match", "mean"),
        top5_agree=("top5_agreement", "mean"),
        n_files=("filename", "count"),
    ).reset_index()
    print(species_summary.to_string(index=False))

    # Gate decision
    mean_cos = df["cosine_similarity"].mean()
    top1_rate = df["top1_match"].mean()

    # Pass if embeddings are highly similar (>0.90 cosine) or top-1 species mostly agrees (>80%)
    gate_pass = mean_cos > 0.90 or top1_rate > 0.80
    print(f"\nGate: cos_sim={mean_cos:.4f} (>0.90?), top1_match={top1_rate:.1%} (>80%?)")
    print(f"{'PASS' if gate_pass else 'FAIL'}: "
          f"{'Proceed to Phase 2' if gate_pass else 'Codec loses too much — stop here'}")


if __name__ == "__main__":
    app()
