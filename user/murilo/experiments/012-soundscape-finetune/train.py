#!/usr/bin/env python3
"""Training script for Experiment 012: Labeled Soundscape Fine-Tuning.

Loads exp 011 checkpoints, expands the classification head from 206 → 234 species,
and fine-tunes on mixed focal + soundscape data.

Supports all three encoder architectures via config dispatch:
  - eat:      EAT-base (AudioSet pretrained), LoRA on attention layers
  - convnext: ConvNeXt-Base (ImageNet pretrained), LoRA on MLP layers
  - protoclr: ProtoCLR/CvT-13 (Xeno-Canto pretrained), LoRA on attention layers

Usage:
    python train.py --config config/eat_soundscape.yaml
    python train.py --config config/convnext_soundscape.yaml
    python train.py --config config/protoclr_soundscape.yaml
    python train.py --config config/eat_soundscape.yaml --fast-dev-run
"""

import argparse
import json
import logging
import math
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from user-level .env (must be before W&B imports)
_user_dir = Path(__file__).parent.parent.parent
load_dotenv(_user_dir / ".env")

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger  # noqa: E402

# Add src to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from birdclef.data import build_species_list  # noqa: E402
from birdclef.data.mixed import MixedDataModule  # noqa: E402
from birdclef.data.soundscape import build_full_species_list  # noqa: E402
from birdclef.lightning.lora_finetune import LoRAFinetuneModule  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "012-soundscape-finetune"


# ============================================================================
# Encoder + transform + LoRA setup (same as exp 011)
# ============================================================================


def build_encoder(config: dict):
    """Build encoder from config, dispatching on encoder type."""
    encoder_type = config["encoder"]["type"]

    if encoder_type == "eat":
        from birdclef.encoders.eat import EATEncoder

        return EATEncoder(
            model_name=config["encoder"].get("model_name", "worstchan/EAT-base_epoch30_pretrain"),
            target_length=config["encoder"].get("target_length", 512),
            precomputed_features=True,
        )
    elif encoder_type == "convnext":
        from birdclef.encoders.convnext import ConvNeXtEncoder

        return ConvNeXtEncoder(
            model_name=config["encoder"]["model_name"],
            pretrained=config["encoder"].get("pretrained", True),
        )
    elif encoder_type == "protoclr":
        from birdclef.encoders.protoclr import ProtoCLREncoder

        return ProtoCLREncoder(
            model_name=config["encoder"]["model_name"],
            checkpoint=config["encoder"].get("checkpoint", "protoclr.pth"),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def build_audio_transform(config: dict):
    """Build the audio transform for the dataloader, dispatching on encoder type."""
    encoder_type = config["encoder"]["type"]

    if encoder_type == "eat":
        from birdclef.encoders.eat import waveform_to_fbank

        return partial(
            waveform_to_fbank,
            target_length=config["encoder"].get("target_length", 512),
        )
    elif encoder_type == "convnext":
        from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram

        return NNAudioMelSpectrogram(
            sample_rate=config["data"]["sample_rate"],
        )
    elif encoder_type == "protoclr":
        from birdclef.preprocessing.protoclr_mel import ProtoCLRMelSpectrogram

        return ProtoCLRMelSpectrogram(
            sample_rate=config["data"]["sample_rate"],
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def apply_lora_to_encoder(encoder, config: dict):
    """Apply PEFT LoRA adapters to the encoder, dispatching on encoder type."""
    lora_cfg = config["lora"]
    encoder_type = config["encoder"]["type"]

    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    if encoder_type == "eat":
        encoder.model = get_peft_model(encoder.model, lora_config)
        encoder.model.print_trainable_parameters()
    elif encoder_type == "convnext":
        encoder.backbone = get_peft_model(encoder.backbone, lora_config)
        encoder.backbone.print_trainable_parameters()
    elif encoder_type == "protoclr":
        encoder.model = get_peft_model(encoder.model, lora_config)
        encoder.model.print_trainable_parameters()
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder


# ============================================================================
# Head expansion: 206 → 234 species
# ============================================================================


def remap_head_weights(
    state_dict: dict,
    species_206: list[str],
    species_234: list[str],
    hidden_dim: int,
    embed_dim: int,
) -> dict:
    """Remap classification head weights from 206 to 234 classes.

    For each of the 206 old species, finds its new index in the 234-sorted list
    and copies the corresponding weight row. The 28 new species rows are
    initialized with kaiming_uniform (matching nn.Linear default).

    Args:
        state_dict: Full state dict from exp 011 checkpoint
        species_206: Sorted species list from exp 011 (206 species)
        species_234: Sorted species list from taxonomy.csv (234 species)
        hidden_dim: Hidden dimension of MLP head (0 = linear head)
        embed_dim: Encoder embedding dimension

    Returns:
        Modified state_dict with expanded head weights
    """
    idx_234 = {s: i for i, s in enumerate(species_234)}
    remap = []
    for old_idx, s in enumerate(species_206):
        if s in idx_234:
            remap.append((old_idx, idx_234[s]))
        else:
            logger.warning(f"Species '{s}' from 206-list not found in 234-list")

    logger.info(f"Head expansion: {len(remap)}/{len(species_206)} species remapped")

    # Determine which state_dict keys hold the final linear layer
    if hidden_dim > 0:
        # MLP head: Sequential(Linear, ReLU, Dropout, Linear)
        # head.0 = Linear(embed_dim, hidden_dim) — unchanged
        # head.3 = Linear(hidden_dim, num_classes) — needs expansion
        weight_key = "head.3.weight"
        bias_key = "head.3.bias"
        fan_in = hidden_dim
    else:
        # Linear head: Linear(embed_dim, num_classes)
        weight_key = "head.weight"
        bias_key = "head.bias"
        fan_in = embed_dim

    old_w = state_dict[weight_key]  # (206, fan_in)
    old_b = state_dict[bias_key]  # (206,)

    # Initialize new weight/bias with kaiming_uniform (nn.Linear default)
    new_w = torch.empty(234, fan_in)
    new_b = torch.empty(234)
    nn.init.kaiming_uniform_(new_w, a=math.sqrt(5))
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(new_b, -bound, bound)

    # Copy old weights to new positions
    for old_idx, new_idx in remap:
        new_w[new_idx] = old_w[old_idx]
        new_b[new_idx] = old_b[old_idx]

    state_dict[weight_key] = new_w
    state_dict[bias_key] = new_b

    # Log weight norms for verification
    old_norms = [new_w[new_idx].norm().item() for _, new_idx in remap]
    all_indices = set(range(234))
    new_indices = all_indices - {new_idx for _, new_idx in remap}
    new_norms = [new_w[i].norm().item() for i in sorted(new_indices)]

    logger.info(
        f"Head weights — copied species mean norm: {sum(old_norms)/len(old_norms):.4f}, "
        f"new species mean norm: {sum(new_norms)/len(new_norms):.4f}"
    )

    return state_dict


# ============================================================================
# Main
# ============================================================================


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune LoRA models on labeled soundscapes (206 → 234 species)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., config/eat_soundscape.yaml)",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help=f"PACE project allocation name (default: {DEFAULT_PROJECT_NAME})",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run quick validation with 5 batches",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers from config",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1)",
    )
    args = parser.parse_args()

    # Derive project directory from project name
    project_dir = Path(f"/storage/project/{args.project_name}/mgustineli3")

    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    run_name = config["run_name"]
    encoder_type = config["encoder"]["type"]

    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Encoder type: {encoder_type}")

    # Override num_workers if specified
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
        logger.info(f"Overriding num_workers to {args.num_workers}")

    # ========================================================================
    # Resolve data paths and species lists
    # ========================================================================

    data_dir = Path(config["data"]["data_dir"]).expanduser()
    taxonomy_csv = data_dir / "taxonomy.csv"
    focal_csv = data_dir / "train.csv"

    # Build species lists
    species_234 = build_full_species_list(taxonomy_csv)
    species_206 = build_species_list(focal_csv)
    num_classes = len(species_234)
    logger.info(f"Full species list: {num_classes} classes")
    logger.info(f"Focal species list: {len(species_206)} classes")

    # ========================================================================
    # Build encoder + LoRA
    # ========================================================================

    audio_transform = build_audio_transform(config)
    logger.info(f"Audio transform: {type(audio_transform).__name__}")

    logger.info(f"Initializing {encoder_type.upper()} encoder...")
    encoder = build_encoder(config)

    lora_cfg = config["lora"]
    logger.info(
        f"LoRA: rank={lora_cfg['rank']}, alpha={lora_cfg['alpha']}, "
        f"dropout={lora_cfg['dropout']}, targets={lora_cfg['target_modules']}"
    )
    encoder = apply_lora_to_encoder(encoder, config)

    # ========================================================================
    # Build Lightning module (234 classes)
    # ========================================================================

    hidden_dim = config["training"].get("hidden_dim", 0)
    embed_dim = config["encoder"].get("embed_dim", 768)

    logger.info("Initializing LoRAFinetuneModule with 234 classes...")
    pl_module = LoRAFinetuneModule(
        encoder=encoder,
        num_classes=num_classes,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=config["training"].get("dropout", 0.1),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        max_epochs=config["training"]["max_epochs"],
        warmup_epochs=config["training"]["warmup_epochs"],
        threshold=config["training"]["threshold"],
    )

    # ========================================================================
    # Load exp 011 checkpoint and expand head
    # ========================================================================

    checkpoint_path = config["checkpoint"]["resume_from"]
    logger.info(f"Loading exp 011 checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Remap head weights from 206 → 234
    state_dict = remap_head_weights(state_dict, species_206, species_234, hidden_dim, embed_dim)

    # Load with strict=False (torchmetrics state shapes differ 206 vs 234)
    missing, unexpected = pl_module.load_state_dict(state_dict, strict=False)

    # Filter out expected mismatches (metrics state)
    real_missing = [
        k
        for k in missing
        if not any(m in k for m in ["cmap", "auroc", "f1_macro", "f1_micro", "top1", "top5"])
    ]
    if real_missing:
        logger.warning(f"Missing keys (non-metric): {real_missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:10]}")

    logger.info("Checkpoint loaded and head expanded successfully")

    # ========================================================================
    # Initialize data module
    # ========================================================================

    logger.info("Initializing MixedDataModule...")
    data_module = MixedDataModule(
        data_dir=str(data_dir),
        taxonomy_csv=str(taxonomy_csv),
        sample_rate=config["data"]["sample_rate"],
        chunk_duration=config["data"]["chunk_duration"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        val_fraction=config["data"].get("val_fraction", 0.1),
        test_fraction=config["data"].get("test_fraction", 0.1),
        soundscape_val_files=config["data"].get("soundscape_val_files", 8),
        soundscape_test_files=config["data"].get("soundscape_test_files", 8),
        seed=config["data"].get("seed", 42),
        transform=audio_transform,
    )

    # ========================================================================
    # Callbacks
    # ========================================================================

    callbacks = []

    # Model checkpoint — save to project shared storage
    checkpoint_dir = project_dir / "experiments" / EXPERIMENT_NAME / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor=config["callbacks"]["model_checkpoint"]["monitor"],
        mode=config["callbacks"]["model_checkpoint"]["mode"],
        save_top_k=config["callbacks"]["model_checkpoint"]["save_top_k"],
        save_last=config["callbacks"]["model_checkpoint"]["save_last"],
        filename=config["callbacks"]["model_checkpoint"]["filename"],
        dirpath=checkpoint_dir,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    # Early stopping
    early_stopping = EarlyStopping(
        monitor=config["callbacks"]["early_stopping"]["monitor"],
        mode=config["callbacks"]["early_stopping"]["mode"],
        patience=config["callbacks"]["early_stopping"]["patience"],
        min_delta=config["callbacks"]["early_stopping"]["min_delta"],
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # GPU stats monitor
    device_stats = DeviceStatsMonitor()
    callbacks.append(device_stats)

    # ========================================================================
    # Loggers
    # ========================================================================

    loggers = []
    experiment_name = f"exp012-{run_name}"

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=Path(__file__).parent / "outputs",
        name="tensorboard",
        version=experiment_name,
    )
    loggers.append(tb_logger)

    if args.fast_dev_run:
        logger.info("Fast dev run — skipping W&B logging")
    else:
        logger.info("Initializing W&B logger...")
        wandb_logger = WandbLogger(
            entity=config["logging"]["entity"],
            project=config["logging"]["project"],
            name=experiment_name,
            save_dir=Path(__file__).parent / config["logging"]["save_dir"],
            log_model=config["logging"]["log_model"],
        )
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(
            {
                "finetune_mode": "lora_soundscape",
                "num_classes": num_classes,
                "num_focal_species": len(species_206),
                "checkpoint_from": checkpoint_path,
            }
        )
        loggers.append(wandb_logger)

    # ========================================================================
    # Trainer
    # ========================================================================

    logger.info(f"Using {args.devices} GPU(s)")
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=config["training"]["precision"],
        gradient_clip_val=config["training"]["grad_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        fast_dev_run=5 if args.fast_dev_run else False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ========================================================================
    # Train
    # ========================================================================

    logger.info("Starting training...")
    trainer.fit(pl_module, data_module)

    # ========================================================================
    # Evaluate on test set
    # ========================================================================

    test_results = None
    if not args.fast_dev_run and checkpoint_callback.best_model_path:
        logger.info(
            f"Evaluating best checkpoint on test set: {checkpoint_callback.best_model_path}"
        )
        test_results = trainer.test(
            pl_module, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
        )
    elif args.fast_dev_run:
        logger.info("Fast dev run — running test step for pipeline validation")
        test_results = trainer.test(pl_module, datamodule=data_module)

    # ========================================================================
    # Save results
    # ========================================================================

    if not args.fast_dev_run:
        results_dir = Path(__file__).parent / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{run_name}.json"

        raw_results = {
            "config_path": str(config_path),
            "run_name": run_name,
            "finetune_mode": "lora_soundscape",
            "num_classes": num_classes,
            "num_focal_species": len(species_206),
            "checkpoint_from": checkpoint_path,
            "best_val_score": (
                float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None
                else None
            ),
            "best_checkpoint_path": str(checkpoint_callback.best_model_path),
            "stopped_epoch": trainer.current_epoch,
            "encoder_type": encoder_type,
            "test_metrics": test_results[0] if test_results else None,
            "lora": {
                "rank": lora_cfg["rank"],
                "alpha": lora_cfg["alpha"],
                "dropout": lora_cfg["dropout"],
                "target_modules": lora_cfg["target_modules"],
            },
        }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
