"""Phase 0: Feasibility gate for prompted SED.

Tests whether the SSAMBA backbone encodes species-discriminative information
in its hidden states, and where that information lives (mean pool, CLS,
first/last patch token).

Compares 4 checkpoints:
  - AudioSet pretrained (generic audio)
  - Pretrain-full (from-scratch SSL on BirdCLEF)
  - Pretrain-warm (AudioSet init + SSL)
  - Finetuned (supervised, upper bound)

Outputs results JSON + PaCMAP visualizations to output_dir.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from nnAudio.features import MelSpectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from ssamba_hf.config import SSAMBAConfig
from ssamba_hf.convert import convert_ssamba_weights
from ssamba_hf.model import SSAMBAModel

from birdclef_009.data.audio import load_audio_chunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 32000
CHUNK_DURATION = 5.0  # seconds
NUM_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)  # 160000
N_MELS = 128
TARGET_FRAMES = 500  # 5s at hop_length=320
HOP_LENGTH = 320  # sr * chunk_duration / target_frames = 32000*5/500
N_FFT = 1024
WIN_LENGTH = 1024
FMIN = 50
FMAX = 14000

NUM_SPECIES = 20
CLIPS_PER_SPECIES = 10
BATCH_SIZE = 32

POOLING_METHODS = ["mean_pool", "cls", "first_patch", "last_patch"]


# ── Data Sampling ────────────────────────────────────────────────────────────


def _find_peak_energy_offset(audio_path, target_sr, num_samples):
    """Find the start offset of the highest-energy 5s window in a file.

    Scans the full file in non-overlapping 5s hops and returns the offset
    of the window with the highest RMS energy.
    """
    info = sf.info(audio_path)
    total_samples = int(info.duration * target_sr)
    if total_samples <= num_samples:
        return 0

    # Read the full file once (cheaper than many partial reads)
    audio, file_sr = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample if needed
    if file_sr != target_sr:
        audio_t = torch.from_numpy(audio).float()
        audio_t = torchaudio.functional.resample(audio_t, file_sr, target_sr)
        audio = audio_t.numpy()

    # Slide in non-overlapping windows, find peak energy
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(audio) - num_samples + 1, num_samples):
        chunk = audio[start : start + num_samples]
        energy = float(np.mean(chunk**2))
        if energy > best_energy:
            best_energy = energy
            best_start = start

    return best_start


def sample_clips(metadata_csv, audio_dir, num_species, clips_per_species, seed=42):
    """Sample highest-energy 5s clips from top species.

    For each recording, selects the 5s window with peak RMS energy
    to maximize the chance of capturing actual vocalizations.

    Returns:
        waveforms: Tensor (N, num_samples)
        labels: list of species labels (len N)
    """
    rng = random.Random(seed)
    df = pd.read_csv(metadata_csv)

    # Select top species by recording count
    species_counts = df["primary_label"].value_counts()
    top_species = species_counts.head(num_species).index.tolist()
    logger.info(
        f"Selected {len(top_species)} species "
        f"(min {species_counts[top_species[-1]]} recordings)"
    )

    waveforms = []
    labels = []

    for species in sorted(top_species):
        species_df = df[df["primary_label"] == species]
        files = species_df["filename"].tolist()
        rng.shuffle(files)

        collected = 0
        for fname in files:
            if collected >= clips_per_species:
                break
            audio_path = os.path.join(audio_dir, fname)
            if not os.path.exists(audio_path):
                continue
            try:
                start = _find_peak_energy_offset(audio_path, SAMPLE_RATE, NUM_SAMPLES)
                wav = load_audio_chunk(
                    audio_path, SAMPLE_RATE, start_sample=start, num_samples=NUM_SAMPLES
                )
                # Pad if shorter
                if wav.shape[0] < NUM_SAMPLES:
                    wav = F.pad(wav, (0, NUM_SAMPLES - wav.shape[0]))
                waveforms.append(wav)
                labels.append(species)
                collected += 1
            except Exception as e:
                logger.warning(f"Error loading {fname}: {e}")
                continue

        if collected < clips_per_species:
            logger.warning(
                f"Only got {collected}/{clips_per_species} clips for {species}"
            )

    waveforms = torch.stack(waveforms)
    logger.info(f"Loaded {len(waveforms)} clips from {len(set(labels))} species")
    return waveforms, labels


# ── Model Loading ────────────────────────────────────────────────────────────


def make_config():
    """Create SSAMBA Tiny config for 5s clips with 128x1 patches."""
    return SSAMBAConfig.tiny(
        img_size=(N_MELS, TARGET_FRAMES),
        patch_size=(128, 1),
    )


def load_backbone_audioset(ckpt_path, config, device="cpu"):
    """Load AudioSet pretrained backbone (Mamba weights only).

    The AudioSet checkpoint uses 16x16 patches, so patch_embed and pos_embed
    are incompatible with our 128x1 config. Only Mamba layer weights and
    norm_f are transferred.
    """
    model = SSAMBAModel(config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    converted = convert_ssamba_weights(checkpoint)

    # Only load weights with matching shapes
    model_state = model.state_dict()
    compatible = {}
    for k, v in converted.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v

    model.load_state_dict(compatible, strict=False)
    loaded_params = sum(v.numel() for v in compatible.values())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"AudioSet: loaded {len(compatible)}/{len(model_state)} params "
        f"({loaded_params:,}/{total_params:,} = {100*loaded_params/total_params:.1f}%)"
    )
    return model.eval()


def load_backbone_lightning(ckpt_path, config, device="cpu"):
    """Load backbone from Lightning checkpoint (SSL or finetune).

    Extracts model.backbone.* keys and handles pos_embed center-cropping
    via birdclef_009's utility.
    """
    from birdclef_009.utils.pretrained import _load_weights_into_backbone

    model = SSAMBAModel(config).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Extract backbone weights
    prefix = "model.backbone."
    backbone_weights = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }
    if not backbone_weights:
        raise ValueError(
            f"No backbone weights with prefix '{prefix}' in {ckpt_path}. "
            f"Keys: {list(state_dict.keys())[:10]}"
        )

    stats = _load_weights_into_backbone(model, backbone_weights)
    logger.info(
        f"Lightning: loaded {stats['loaded_params']:,}/{stats['total_params']:,} "
        f"= {stats['pct']:.1f}%"
    )
    return model.eval()


# ── Feature Extraction ───────────────────────────────────────────────────────


@torch.no_grad()
def extract_representations(backbone, mel_batch, device="cuda"):
    """Extract 4 pooling representations from mel spectrograms.

    Args:
        backbone: SSAMBAModel in eval mode
        mel_batch: Tensor (N, 1, 128, 500)
        device: compute device

    Returns:
        dict mapping pooling method name -> (N, D) numpy array
    """
    backbone = backbone.to(device)
    N = mel_batch.shape[0]
    D = backbone.config.embed_dim

    # Pre-allocate
    all_reps = {method: np.zeros((N, D), dtype=np.float32) for method in POOLING_METHODS}

    for start in range(0, N, BATCH_SIZE):
        end = min(start + BATCH_SIZE, N)
        batch = mel_batch[start:end].to(device)

        # (B, T+1, D) — includes CLS token
        tokens = backbone.forward_features(batch, return_all_tokens=True)
        tokens = tokens.float().cpu().numpy()

        # CLS is at position 0 (use_middle_cls_token=False)
        cls_pos = 0
        B, T_plus_1, _ = tokens.shape

        # Mean pool: average of patch tokens (exclude CLS)
        patch_tokens = np.concatenate(
            [tokens[:, :cls_pos, :], tokens[:, cls_pos + 1 :, :]], axis=1
        )
        all_reps["mean_pool"][start:end] = patch_tokens.mean(axis=1)

        # CLS token
        all_reps["cls"][start:end] = tokens[:, cls_pos, :]

        # First patch token (position after CLS when CLS is at 0)
        first_patch_idx = 1 if cls_pos == 0 else 0
        all_reps["first_patch"][start:end] = tokens[:, first_patch_idx, :]

        # Last patch token
        all_reps["last_patch"][start:end] = tokens[:, -1, :]

    return all_reps


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_representations(embeddings, labels):
    """Compute metrics for a set of embeddings.

    Args:
        embeddings: (N, D) numpy array
        labels: list of N string labels

    Returns:
        dict with silhouette, linear_probe_acc, cosine_within, cosine_between
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # L2 normalize for cosine-based metrics
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    emb_normed = embeddings / norms

    # Silhouette score (cosine metric)
    sil = silhouette_score(emb_normed, y, metric="cosine")

    # Linear probe: logistic regression with 5-fold CV
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    cv_scores = cross_val_score(clf, emb_normed, y, cv=5, scoring="accuracy")
    probe_acc = cv_scores.mean()
    probe_std = cv_scores.std()

    # Pairwise cosine similarity (within-class vs between-class)
    sim_matrix = emb_normed @ emb_normed.T
    within_sims = []
    between_sims = []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            if y[i] == y[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    cosine_within = float(np.mean(within_sims)) if within_sims else 0.0
    cosine_between = float(np.mean(between_sims)) if between_sims else 0.0

    return {
        "silhouette": round(float(sil), 4),
        "linear_probe_acc": round(float(probe_acc), 4),
        "linear_probe_std": round(float(probe_std), 4),
        "cosine_within": round(cosine_within, 4),
        "cosine_between": round(cosine_between, 4),
        "n_classes": n_classes,
        "n_samples": len(y),
    }


# ── Visualization ────────────────────────────────────────────────────────────


def plot_tsne(embeddings_dict, labels, checkpoint_name, output_path):
    """Create t-SNE visualization with 4 subplots (one per pooling method).

    Args:
        embeddings_dict: {method: (N, D) array}
        labels: list of N string labels
        checkpoint_name: name for the plot title
        output_path: path to save PNG
    """
    from sklearn.manifold import TSNE
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"t-SNE — {checkpoint_name}", fontsize=16, y=0.98)

    cmap = plt.colormaps.get_cmap("nipy_spectral").resampled(n_classes)

    for ax, method in zip(axes.flat, POOLING_METHODS):
        emb = embeddings_dict[method]

        tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric="cosine")
        coords = tsne.fit_transform(emb)

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=y,
            cmap=cmap,
            s=15,
            alpha=0.7,
            edgecolors="none",
        )
        ax.set_title(method, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved PaCMAP plot to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 0 feasibility gate")
    parser.add_argument(
        "--metadata-csv",
        default=os.path.expanduser(
            "~/scratch/birdclef-2026/data/birdclef-2025/train.csv"
        ),
    )
    parser.add_argument(
        "--audio-dir",
        default=os.path.expanduser(
            "~/scratch/birdclef-2026/data/birdclef-2025/train_audio/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser(
            "~/scratch/birdclef-2026/outputs/exp011/phase0"
        ),
    )
    parser.add_argument("--num-species", type=int, default=NUM_SPECIES)
    parser.add_argument("--clips-per-species", type=int, default=CLIPS_PER_SPECIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip t-SNE plots (faster)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Checkpoint registry ──────────────────────────────────────────────
    exp009 = Path(__file__).resolve().parents[2] / "009-birdclef-pipeline"

    checkpoints = {
        "audioset": {
            "path": os.path.expanduser(
                "~/scratch/birdclef-2026/checkpoints/ssamba/ssamba_tiny_400.pth"
            ),
            "type": "audioset",
        },
        "pretrain-full": {
            "path": str(exp009 / "logs/pretrain-full/checkpoints/last.ckpt"),
            "type": "ssl",
        },
        "pretrain-warm": {
            "path": str(exp009 / "logs/pretrain-warm/checkpoints/last.ckpt"),
            "type": "ssl",
        },
        "finetuned": {
            "path": str(
                exp009
                / "logs/finetune/checkpoints/finetune-epoch=11-val/cmap=0.4252.ckpt"
            ),
            "type": "finetune",
        },
    }

    # Validate checkpoint paths
    for name, info in checkpoints.items():
        if not os.path.exists(info["path"]):
            logger.warning(f"Checkpoint not found: {name} at {info['path']}")

    # ── Sample data ──────────────────────────────────────────────────────
    logger.info("Sampling clips...")
    waveforms, labels = sample_clips(
        args.metadata_csv,
        args.audio_dir,
        args.num_species,
        args.clips_per_species,
        seed=args.seed,
    )

    # ── Compute mel spectrograms ─────────────────────────────────────────
    logger.info("Computing mel spectrograms...")
    mel_transform = MelSpectrogram(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        trainable_mel=False,
        trainable_STFT=False,
    ).to(device)

    # Process in batches to avoid OOM
    mel_list = []
    for start in range(0, len(waveforms), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(waveforms))
        batch = waveforms[start:end].to(device)
        with torch.no_grad():
            mel = mel_transform(batch)  # (B, n_mels, frames)
        # Add channel dim: (B, 1, n_mels, frames)
        mel = mel.unsqueeze(1).cpu()
        mel_list.append(mel)

    mel_batch = torch.cat(mel_list, dim=0)
    logger.info(f"Mel batch shape: {mel_batch.shape}")

    # Verify frame count
    actual_frames = mel_batch.shape[-1]
    if actual_frames != TARGET_FRAMES:
        logger.warning(
            f"Got {actual_frames} frames, expected {TARGET_FRAMES}. "
            f"Adjusting config."
        )

    # ── Process each checkpoint ──────────────────────────────────────────
    config = SSAMBAConfig.tiny(
        img_size=(N_MELS, actual_frames),
        patch_size=(128, 1),
    )
    logger.info(
        f"Model config: img_size={config.img_size}, patch_size={config.patch_size}, "
        f"num_patches={config.num_patches}, embed_dim={config.embed_dim}"
    )

    results = {}

    for name, info in checkpoints.items():
        ckpt_path = info["path"]
        if not os.path.exists(ckpt_path):
            logger.warning(f"Skipping {name}: checkpoint not found")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing checkpoint: {name}")
        logger.info(f"  Path: {ckpt_path}")

        # Load backbone
        if info["type"] == "audioset":
            backbone = load_backbone_audioset(ckpt_path, config, device="cpu")
        else:
            backbone = load_backbone_lightning(ckpt_path, config, device="cpu")

        # Extract representations
        logger.info("Extracting representations...")
        reps = extract_representations(backbone, mel_batch, device=device)

        # Evaluate each pooling method
        results[name] = {}
        for method in POOLING_METHODS:
            logger.info(f"  Evaluating {method}...")
            metrics = evaluate_representations(reps[method], labels)
            results[name][method] = metrics
            logger.info(
                f"    silhouette={metrics['silhouette']:.4f}, "
                f"probe_acc={metrics['linear_probe_acc']:.4f} "
                f"(+/-{metrics['linear_probe_std']:.4f}), "
                f"cos_within={metrics['cosine_within']:.4f}, "
                f"cos_between={metrics['cosine_between']:.4f}"
            )

        # PaCMAP visualization
        if not args.skip_plots:
            plot_path = os.path.join(args.output_dir, f"tsne_{name}.png")
            plot_tsne(reps, labels, name, plot_path)

        # Free GPU memory
        del backbone
        torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # ── Print summary table ──────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    header = f"{'Checkpoint':<16} {'Pooling':<14} {'Silhouette':>10} {'Probe Acc':>10} {'Cos-W':>8} {'Cos-B':>8}"
    logger.info(header)
    logger.info("-" * 80)

    for ckpt_name, methods in results.items():
        for method, metrics in methods.items():
            logger.info(
                f"{ckpt_name:<16} {method:<14} "
                f"{metrics['silhouette']:>10.4f} "
                f"{metrics['linear_probe_acc']:>10.4f} "
                f"{metrics['cosine_within']:>8.4f} "
                f"{metrics['cosine_between']:>8.4f}"
            )

    # ── Decision gate ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("DECISION GATE")
    logger.info("=" * 80)

    for ckpt_name, methods in results.items():
        mp = methods.get("mean_pool", {})
        sil = mp.get("silhouette", 0)
        acc = mp.get("linear_probe_acc", 0)
        lp = methods.get("last_patch", {})
        last_acc = lp.get("linear_probe_acc", 0)

        status = "FAIL"
        if sil > 0.1 and acc > 0.30:
            status = "PASS"
        if last_acc > 0.30:
            status = "STRONG PASS (last-token works)"

        logger.info(
            f"  {ckpt_name}: {status} "
            f"(mean_pool sil={sil:.3f}, acc={acc:.3f}; last_patch acc={last_acc:.3f})"
        )

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
