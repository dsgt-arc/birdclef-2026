"""Zero-training prototypical baseline for EfficientNet-B1.

Tests whether the frozen backbone has metric structure for species discrimination
by computing prototypes from training embeddings and evaluating cosine similarity.

Usage:
    python scripts/zero_training_baseline.py [--exp012-ckpt PATH]
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nnAudio.features import MelSpectrogram
from sklearn.metrics import average_precision_score, silhouette_score
from torch.utils.data import DataLoader
from transformers import EfficientNetModel

from birdclef_009.data.audio import collate_finetune
from birdclef_009.data.finetune import BirdCLEFTrainDataset, build_species_list

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config (matches exp 012 finetune.yaml) ---
AUDIO_DIR = "~/scratch/birdclef-2026/data/birdclef-2025/train_audio/"
METADATA_CSV = "~/scratch/birdclef-2026/data/birdclef-2025/train.csv"
MODEL_NAME = "DBD-research-group/EfficientNet-B1-BirdSet-XCL"
SAMPLE_RATE = 32000
CHUNK_DURATION = 5.0
N_MELS = 256
N_FFT = 2048
HOP_LENGTH = 256
TARGET_FRAMES = 625
NORM_MEAN = -4.268
NORM_STD = 4.569
BATCH_SIZE = 128
NUM_WORKERS = 8
VAL_FRACTION = 0.1
SEED = 42
COSINE_GAP_PER_SPECIES = 5  # clips per species for stratified cosine gap sampling


def build_mel_layer():
    return MelSpectrogram(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=50,
        fmax=14000,
        trainable_mel=False,
        trainable_STFT=False,
    )


def compute_mel(mel_layer, waveforms, device):
    """Compute normalized mel spectrogram (B, 1, n_mels, T)."""
    mel = mel_layer(waveforms.to(device))
    mel = torch.log(mel.clamp(min=1e-10))
    mel = (mel - NORM_MEAN) / (NORM_STD * 2)
    T = mel.shape[2]
    if T > TARGET_FRAMES:
        mel = mel[:, :, :TARGET_FRAMES]
    elif T < TARGET_FRAMES:
        mel = F.pad(mel, (0, TARGET_FRAMES - T), value=0)
    return mel.unsqueeze(1)


@torch.no_grad()
def extract_features(backbone, mel_layer, dataloader, device):
    """Extract frame-level features for all clips.

    Returns:
        features: list of (1280, num_frames) L2-normed tensors (CPU)
        features_raw: list of (1280, num_frames) unnormalized tensors (CPU)
        labels: (N, num_classes) tensor
    """
    all_features = []
    all_features_raw = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        waveforms = batch["waveform"].to(device)
        labels = batch["clip_labels"]

        mel = compute_mel(mel_layer, waveforms, device)
        outputs = backbone(pixel_values=mel)
        # last_hidden_state: (B, 1280, H, W) where H=freq, W=time
        feat = outputs.last_hidden_state  # (B, 1280, H, W)
        # Frequency pool: mean over H dimension
        feat = feat.mean(dim=2)  # (B, 1280, W)

        # Store raw (for Strategy C) and L2-normed (for A/B)
        all_features_raw.extend(feat.cpu().unbind(0))
        feat_normed = F.normalize(feat, dim=1)  # (B, 1280, W)
        all_features.extend(feat_normed.cpu().unbind(0))
        all_labels.append(labels)

        if (batch_idx + 1) % 50 == 0:
            logger.info(f"  Extracted {(batch_idx + 1) * dataloader.batch_size} clips")

    labels = torch.cat(all_labels, dim=0)
    logger.info(f"  Total: {len(all_features)} clips extracted")
    return all_features, all_features_raw, labels


def compute_prototypes(features, labels, species_list):
    """Compute per-species prototypes from training features.

    Args:
        features: list of (1280, W) tensors
        labels: (N, C) multi-hot tensor

    Returns:
        prototypes: (C, 1280) L2-normalized prototype matrix
        counts: (C,) number of clips per species
    """
    num_classes = len(species_list)
    embed_dim = features[0].shape[0]

    # Accumulate all frame embeddings per species
    proto_sums = torch.zeros(num_classes, embed_dim)
    counts = torch.zeros(num_classes)

    for feat, label in zip(features, labels):
        # feat: (1280, W), label: (C,)
        clip_embed = feat.mean(dim=1)  # (1280,) mean over frames
        active_species = label.nonzero(as_tuple=True)[0]
        for s in active_species:
            proto_sums[s] += clip_embed
            counts[s] += 1

    # L2-normalize prototypes
    prototypes = F.normalize(proto_sums, dim=1)

    n_empty = (counts == 0).sum().item()
    if n_empty > 0:
        logger.warning(f"{n_empty} species have no training clips — prototype is zero")

    logger.info(
        f"Prototypes: {num_classes} species, "
        f"min clips={counts[counts > 0].min().item():.0f}, "
        f"max clips={counts.max().item():.0f}, "
        f"median={counts[counts > 0].median().item():.0f}"
    )
    return prototypes, counts


def cosine_gap_analysis(features, labels, per_species=COSINE_GAP_PER_SPECIES):
    """Compute within-species vs between-species cosine similarity gap.

    Uses stratified sampling (K clips per species) to ensure enough within-species
    pairs for stable estimates.
    """
    rng = np.random.RandomState(SEED)
    all_primary = labels.argmax(dim=1).numpy()  # (N,)

    # Stratified sample: up to K clips per species
    indices = []
    for sp in np.unique(all_primary):
        sp_indices = np.where(all_primary == sp)[0]
        k = min(per_species, len(sp_indices))
        indices.extend(rng.choice(sp_indices, k, replace=False).tolist())
    indices = np.array(indices)
    n = len(indices)

    # Get clip-level embeddings
    embeds = torch.stack([features[i].mean(dim=1) for i in indices])  # (n, 1280)
    embeds = F.normalize(embeds, dim=1)
    primary_labels = torch.from_numpy(all_primary[indices])

    # Pairwise cosine similarity
    sim = embeds @ embeds.T  # (n, n)

    within_sims = []
    between_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            s = sim[i, j].item()
            if primary_labels[i] == primary_labels[j]:
                within_sims.append(s)
            else:
                between_sims.append(s)

    within_mean = np.mean(within_sims) if within_sims else 0.0
    between_mean = np.mean(between_sims) if between_sims else 0.0
    gap = within_mean - between_mean

    # Silhouette score
    sil = float("nan")
    unique_labels = np.unique(primary_labels.numpy())
    if len(unique_labels) > 1:
        try:
            sil = silhouette_score(embeds.numpy(), primary_labels.numpy(), metric="cosine")
        except Exception as e:
            logger.warning(f"Silhouette score failed: {e}")

    logger.info(f"Cosine gap analysis ({n} clips):")
    logger.info(f"  Within-species mean cosine:  {within_mean:.4f} ({len(within_sims)} pairs)")
    logger.info(f"  Between-species mean cosine: {between_mean:.4f} ({len(between_sims)} pairs)")
    logger.info(f"  Gap (within - between):      {gap:.4f}")
    logger.info(f"  Silhouette score (cosine):   {sil:.4f}")

    return {"within_mean": within_mean, "between_mean": between_mean, "gap": gap, "silhouette": sil}


def evaluate_clip_level(val_features, val_labels, prototypes):
    """Strategy A: clip-level cosine similarity → cmAP."""
    # Mean-pool frames per clip
    clip_embeds = torch.stack([f.mean(dim=1) for f in val_features])  # (N, 1280)
    clip_embeds = F.normalize(clip_embeds, dim=1)

    # Cosine similarity to prototypes
    scores = clip_embeds @ prototypes.T  # (N, C)

    # cmAP (class-mean average precision)
    labels_np = val_labels.numpy()
    scores_np = scores.numpy()

    # Filter to classes with at least one positive
    active = labels_np.sum(axis=0) > 0
    if active.sum() == 0:
        return 0.0

    cmap = average_precision_score(labels_np[:, active], scores_np[:, active], average="macro")
    logger.info(f"Strategy A (clip-level cosine): cmAP = {cmap:.4f} ({active.sum()} active classes)")
    return cmap


def evaluate_frame_level_max(val_features, val_labels, prototypes):
    """Strategy B: per-frame cosine similarity, max over frames → cmAP."""
    scores_list = []
    for feat in val_features:
        # feat: (1280, W), prototypes: (C, 1280)
        frame_sim = prototypes @ feat  # (C, W) — already L2-normed
        clip_score = frame_sim.max(dim=1).values  # (C,) — max over frames
        scores_list.append(clip_score)

    scores = torch.stack(scores_list)  # (N, C)
    labels_np = val_labels.numpy()
    scores_np = scores.numpy()

    active = labels_np.sum(axis=0) > 0
    if active.sum() == 0:
        return 0.0

    cmap = average_precision_score(labels_np[:, active], scores_np[:, active], average="macro")
    logger.info(f"Strategy B (frame-level max):   cmAP = {cmap:.4f} ({active.sum()} active classes)")
    return cmap


def evaluate_linear_probe_framewise(val_features_raw, val_labels, ckpt_path, num_classes):
    """Strategy C: exp 012 linear head applied per-frame, max → cmAP."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Extract linear head weights from Lightning checkpoint
    # Keys are like "model.head.1.weight", "model.head.1.bias" (after Dropout)
    head_weight = None
    head_bias = None
    for k, v in state_dict.items():
        if "head" in k and "weight" in k and v.shape[0] == num_classes:
            head_weight = v
        if "head" in k and "bias" in k and v.shape[0] == num_classes:
            head_bias = v

    if head_weight is None:
        logger.warning(f"Could not find linear head in {ckpt_path}")
        return None

    logger.info(f"Loaded linear head: {head_weight.shape} from {ckpt_path}")

    scores_list = []
    for feat in val_features_raw:
        # feat: (1280, W) — unnormalized features (matching exp 012 training)
        # Apply linear head per frame: (C, 1280) @ (1280, W) + (C, 1) → (C, W)
        frame_logits = head_weight @ feat  # (C, W)
        if head_bias is not None:
            frame_logits += head_bias.unsqueeze(1)
        clip_score = frame_logits.max(dim=1).values  # (C,) — max over frames
        scores_list.append(clip_score)

    scores = torch.stack(scores_list)  # (N, C)
    labels_np = val_labels.numpy()
    scores_np = scores.numpy()

    active = labels_np.sum(axis=0) > 0
    if active.sum() == 0:
        return 0.0

    cmap = average_precision_score(labels_np[:, active], scores_np[:, active], average="macro")
    logger.info(f"Strategy C (linear framewise):  cmAP = {cmap:.4f} ({active.sum()} active classes)")
    return cmap


def main():
    parser = argparse.ArgumentParser(description="Zero-training prototypical baseline")
    parser.add_argument("--exp012-ckpt", type=str, default=None, help="Exp 012 checkpoint for strategy C")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 1. Build species list and data module
    species_list = build_species_list(str(Path(METADATA_CSV).expanduser()))
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes}")

    # 2. Load backbone
    logger.info(f"Loading backbone: {MODEL_NAME}")
    backbone = EfficientNetModel.from_pretrained(MODEL_NAME)
    backbone.eval()
    backbone.to(device)

    mel_layer = build_mel_layer().to(device)

    # 3. Build datasets (same split as exp 012)
    df = pd.read_csv(str(Path(METADATA_CSV).expanduser()))
    rng = np.random.RandomState(SEED)
    val_indices = []
    train_indices = []
    for _species, group in df.groupby("primary_label"):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * VAL_FRACTION))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    train_df = df.loc[train_indices].reset_index(drop=True)
    val_df = df.loc[val_indices].reset_index(drop=True)
    logger.info(f"Train: {len(train_df)} recordings, Val: {len(val_df)} recordings")

    # num_tokens doesn't matter for feature extraction (we don't use padding mask)
    # but we need it for the dataset constructor
    num_tokens = TARGET_FRAMES  # placeholder
    train_dataset = BirdCLEFTrainDataset(
        audio_dir=str(Path(AUDIO_DIR).expanduser()),
        metadata_df=train_df,
        species_list=species_list,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
        num_tokens=num_tokens,
    )
    val_dataset = BirdCLEFTrainDataset(
        audio_dir=str(Path(AUDIO_DIR).expanduser()),
        metadata_df=val_df,
        species_list=species_list,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
        num_tokens=num_tokens,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_finetune,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_finetune,
        pin_memory=True,
    )

    # 4. Extract features
    logger.info("Extracting train features...")
    train_features, _train_raw, train_labels = extract_features(backbone, mel_layer, train_loader, device)
    logger.info("Extracting val features...")
    val_features, val_features_raw, val_labels = extract_features(backbone, mel_layer, val_loader, device)

    # 5. Cosine gap analysis
    logger.info("=" * 60)
    gap_results = cosine_gap_analysis(train_features, train_labels)

    # 6. Compute prototypes from training data
    logger.info("=" * 60)
    prototypes, counts = compute_prototypes(train_features, train_labels, species_list)

    # 7. Evaluate strategies
    logger.info("=" * 60)
    cmap_a = evaluate_clip_level(val_features, val_labels, prototypes)
    cmap_b = evaluate_frame_level_max(val_features, val_labels, prototypes)

    cmap_c = None
    if args.exp012_ckpt and Path(args.exp012_ckpt).exists():
        cmap_c = evaluate_linear_probe_framewise(val_features_raw, val_labels, args.exp012_ckpt, num_classes)

    # 8. Summary & decision gates
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Cosine gap:                    {gap_results['gap']:.4f}")
    logger.info(f"Silhouette score:              {gap_results['silhouette']:.4f}")
    logger.info(f"Strategy A (clip cosine) cmAP: {cmap_a:.4f}")
    logger.info(f"Strategy B (frame max)   cmAP: {cmap_b:.4f}")
    if cmap_c is not None:
        logger.info(f"Strategy C (linear frame) cmAP: {cmap_c:.4f}")

    logger.info("")
    logger.info("DECISION GATES:")
    if cmap_a < 0.10 and cmap_b < 0.10:
        logger.info("  !! A and B both < 0.10 → ABORT: no metric structure")
    elif cmap_b > cmap_a + 0.03:
        logger.info("  >> B > A by >0.03 → PROCEED: framewise max adds value")
    elif abs(cmap_b - cmap_a) <= 0.03:
        logger.info("  ~~ B ≈ A → REFRAME: no SED benefit from framewise approach")
    if gap_results["gap"] < 0.02:
        logger.info("  !! Cosine gap < 0.02 → ABORT: no metric structure")


if __name__ == "__main__":
    main()
