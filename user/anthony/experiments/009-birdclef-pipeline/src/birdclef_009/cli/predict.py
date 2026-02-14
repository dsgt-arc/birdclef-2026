"""CLI entry point for inference and submission generation."""

import argparse
import logging

import torch
import yaml
from nnAudio.features import MelSpectrogram
from ssamba_hf.config import SSAMBAConfig

from ..data.finetune import build_species_list
from ..inference.predict import predict_directory
from ..inference.submission import build_submission
from ..models.finetune import SSAMBAClassifier

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF Inference + Submission")
    parser.add_argument("--config", type=str, required=True, help="Path to finetune config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to finetuned Lightning checkpoint")
    parser.add_argument("--soundscape-dir", type=str, required=True, help="Path to soundscape audio files")
    parser.add_argument("--sample-submission", type=str, required=True, help="Path to sample_submission.csv")
    parser.add_argument("--output", type=str, default="predictions/submission.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu or cuda)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # Build species list
    species_list = build_species_list(data_cfg["metadata_csv"])
    num_classes = len(species_list)

    # Build model
    mel_spec = MelSpectrogram(
        sr=data_cfg["sample_rate"],
        n_fft=1024,
        win_length=1024,
        hop_length=int(data_cfg["sample_rate"] * data_cfg["chunk_duration"] / data_cfg["target_frames"]),
        n_mels=data_cfg["n_mels"],
        fmin=50,
        fmax=14000,
        trainable_mel=False,
        trainable_STFT=False,
    )

    config = SSAMBAConfig(
        img_size=tuple(model_cfg["img_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
    )

    model = SSAMBAClassifier(
        config=config,
        num_classes=num_classes,
        mel_spec_layer=mel_spec,
        target_frames=data_cfg["target_frames"],
        head_dropout=model_cfg.get("head_dropout", 0.3),
    )

    # Load trained weights from Lightning checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    # Strip 'model.' prefix from Lightning state dict
    model_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value
        else:
            model_state[key] = value
    model.load_state_dict(model_state)
    model.to(args.device)
    model.eval()

    logger.info(f"Model loaded from {args.checkpoint}")

    # Run inference
    predictions = predict_directory(
        model=model,
        soundscape_dir=args.soundscape_dir,
        species_list=species_list,
        sample_rate=data_cfg["sample_rate"],
        chunk_duration=data_cfg["chunk_duration"],
        device=args.device,
    )

    # Build submission
    build_submission(predictions, args.sample_submission, args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
