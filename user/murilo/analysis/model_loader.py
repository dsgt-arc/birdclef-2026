"""Config-driven model loading for any SED model.

Replicates the construction pattern from experiments/019-sed-ablations/train.py
but decoupled from the training pipeline. Works for any SED model config.

Usage:
    from analysis.model_loader import load_model

    model, mel_transform, species_list = load_model(
        config_path="experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml",
        checkpoint_path="/path/to/best-epochepoch=19.ckpt",
        device="cpu",
    )
"""

import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path so we can import birdclef modules
_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from birdclef.data.soundscape import build_full_species_list  # noqa: E402
from birdclef.encoders.efficientnet import EfficientNetEncoder  # noqa: E402
from birdclef.heads.sed_head import AttBlockV2, AttHead, LSEHead  # noqa: E402
from birdclef.lightning.sed_finetune import SEDFinetuneModule  # noqa: E402
from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram  # noqa: E402

logger = logging.getLogger(__name__)

HEAD_REGISTRY = {
    "lse_head": LSEHead,
    "att_head": AttHead,
    "att_block_v2": AttBlockV2,
}


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_mel_transform(config: dict) -> torch.nn.Module:
    """Construct the mel spectrogram transform from config.

    Dispatches on config["mel"]["type"]: "nnaudio" (default) or "ttahara".
    """
    sample_rate = config["data"]["sample_rate"]
    window_duration = config["data"]["window_duration"]
    mel_cfg = config.get("mel", {})
    mel_type = mel_cfg.get("type", "nnaudio")

    if mel_type == "ttahara":
        from birdclef.preprocessing.ttahara_mel import TtaharaMelTransform

        in_chans = config["encoder"].get("in_chans", 1)
        return TtaharaMelTransform(
            sample_rate=sample_rate,
            n_fft=mel_cfg.get("n_fft", 2048),
            win_length=mel_cfg.get("win_length", 626),
            hop_length=mel_cfg.get("hop_length", 313),
            f_min=mel_cfg.get("f_min", 20),
            n_mels=mel_cfg.get("n_mels", 256),
            top_db=mel_cfg.get("top_db", 80.0),
            target_shape=tuple(mel_cfg.get("target_shape", [256, 256])),
            out_channels=in_chans,
        )

    # Default: NNAudio mel spectrogram
    hop_length = mel_cfg.get("hop_length")
    if hop_length is None:
        frame_shift_ms = mel_cfg.get("frame_shift_ms", 10)
        hop_length = int(frame_shift_ms / 1000.0 * sample_rate)
    target_frames = int(window_duration * sample_rate / hop_length)

    return NNAudioMelSpectrogram(
        sample_rate=sample_rate,
        n_mels=mel_cfg.get("n_mels", 128),
        target_frames=target_frames,
        n_fft=mel_cfg.get("n_fft"),
        hop_length=hop_length,
        fmax=mel_cfg.get("fmax"),
        norm_type=mel_cfg.get("norm_type", "audioset"),
        top_db=mel_cfg.get("top_db", 80.0),
    )


def build_model(config: dict, num_classes: int) -> SEDFinetuneModule:
    """Construct encoder + head + Lightning module from config.

    Uses pretrained=False since weights come from checkpoint.
    """
    # Encoder
    encoder = EfficientNetEncoder(
        model_name=config["encoder"]["model_name"],
        pretrained=False,
        drop_path_rate=config["encoder"].get("drop_path_rate", 0.0),
        in_chans=config["encoder"].get("in_chans", 1),
        projection_dim=config["encoder"].get("projection_dim"),
    )
    embed_dim = encoder.embed_dim

    # SED head
    head_type = config["sed_head"].get("type", "att_head")
    head_cls = HEAD_REGISTRY.get(head_type, AttHead)
    head_kwargs = {
        "in_channels": embed_dim,
        "num_classes": num_classes,
        "hidden_dim": config["sed_head"]["hidden_dim"],
        "dropout": config["sed_head"]["dropout"],
        "gem_p": config["sed_head"]["gem_p"],
    }
    if head_type == "lse_head":
        head_kwargs["lse_temperature"] = config["sed_head"].get("lse_temperature", 1.0)
        head_kwargs["freq_pool"] = config["sed_head"].get("freq_pool", "gem")
        head_kwargs["n_chunks"] = config["sed_head"].get("n_chunks", 1)
    sed_head = head_cls(**head_kwargs)

    # Lightning module — only inference-relevant params matter
    training_cfg = config.get("training", {})
    return SEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        lr=training_cfg.get("lr", 1e-4),
        head_lr=training_cfg.get("head_lr", 1e-3),
        blend_clip=training_cfg.get("blend_clip", 1.0),
        blend_frame=training_cfg.get("blend_frame", 0.0),
        loss_type=training_cfg.get("loss_type", "bce"),
    )


def load_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
    data_dir: str | Path | None = None,
) -> tuple[SEDFinetuneModule, torch.nn.Module, list[str]]:
    """Load a trained SED model from config + checkpoint.

    Args:
        config_path: Path to YAML config file.
        checkpoint_path: Path to PyTorch Lightning checkpoint.
        device: Target device ("cpu" or "cuda").
        data_dir: Override for config["data"]["data_dir"]. If None, uses config value.

    Returns:
        (model, mel_transform, species_list)
        - model: SEDFinetuneModule in eval mode on device
        - mel_transform: Mel spectrogram transform on device
        - species_list: 234 species names from taxonomy.csv
    """
    config = load_config(config_path)

    # Species list
    if data_dir is None:
        data_dir = Path(config["data"]["data_dir"]).expanduser()
    else:
        data_dir = Path(data_dir)
    taxonomy_csv = data_dir / "taxonomy.csv"
    species_list = build_full_species_list(taxonomy_csv)
    num_classes = len(species_list)

    # Build model architecture
    model = build_model(config, num_classes)

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out metric keys — those are expected to be missing
        real_missing = [
            k
            for k in missing
            if not any(m in k for m in ("_cmap", "_auroc", "_f1_", "_top1", "_top5", "_val_sc"))
        ]
        if real_missing:
            logger.warning(f"Missing keys in checkpoint: {real_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected}")

    model.eval()
    model.to(device)

    # Mel transform
    mel_transform = build_mel_transform(config)
    mel_transform.to(device)

    logger.info(
        f"Loaded model from {checkpoint_path} "
        f"({sum(p.numel() for p in model.parameters()):,} params, "
        f"{num_classes} classes, device={device})"
    )

    return model, mel_transform, species_list
