#!/usr/bin/env python3
"""Training script for SED experiments (shared by exp 017 and 019).

Supports EfficientNet-B0, HGNetV2-B0, and other timm backbones with
AttHead, AttBlockV2, or LSEHead. Configurable via YAML configs.

Usage:
    python train.py --config config/effnet_b0_sed_v5a.yaml
    python train.py --config config/effnet_b0_sed_v5a.yaml --fast-dev-run
    python train.py --config config/effnet_b0_sed_v5a.yaml --gpu-check
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from user-level .env (must be before W&B imports)
_user_dir = Path(__file__).parent.parent.parent
load_dotenv(_user_dir / ".env")

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
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

from birdclef.data.sed import SEDDataModule  # noqa: E402
from birdclef.data.soundscape import build_full_species_list  # noqa: E402
from birdclef.encoders.efficientnet import EfficientNetEncoder  # noqa: E402
from birdclef.heads.sed_head import AttBlockV2, AttHead, LSEHead  # noqa: E402
from birdclef.lightning.sed_finetune import SEDFinetuneModule  # noqa: E402
from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "017-mel-overhaul"


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Mel overhaul training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--project-name", type=str, default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--gpu-check", type=int, nargs="?", const=1, default=0)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Fold ID for cross-validation (0-3). Uses fold_assignments.json.",
    )
    args = parser.parse_args()

    if args.fast_dev_run and args.gpu_check > 0:
        parser.error("--fast-dev-run and --gpu-check are mutually exclusive")

    gpu_check_mode = args.gpu_check > 0
    project_dir = Path(f"/storage/project/{args.project_name}/mgustineli3")

    # Load config
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    run_name = config["run_name"]

    logger.info(f"Config: {config_path}")
    logger.info(f"Run name: {run_name}")

    if gpu_check_mode:
        config["training"]["max_epochs"] = args.gpu_check
        logger.info(f"GPU check mode: {args.gpu_check} epoch(s)")

    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers

    # ========================================================================
    # Species list
    # ========================================================================

    data_dir = Path(config["data"]["data_dir"]).expanduser()
    taxonomy_csv = data_dir / "taxonomy.csv"
    species_list = build_full_species_list(taxonomy_csv)
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes} classes")

    # ========================================================================
    # Build encoder + SED head
    # ========================================================================

    logger.info("Building EfficientNet encoder...")
    encoder = EfficientNetEncoder(
        model_name=config["encoder"]["model_name"],
        pretrained=config["encoder"].get("pretrained", True),
        drop_path_rate=config["encoder"].get("drop_path_rate", 0.0),
        in_chans=config["encoder"].get("in_chans", 1),
        projection_dim=config["encoder"].get("projection_dim"),
    )

    # Use encoder's actual embed_dim (not config) — features_only=True may differ
    embed_dim = encoder.embed_dim
    logger.info(f"Encoder embed_dim: {embed_dim}")

    head_type = config["sed_head"].get("type", "att_head")
    logger.info(f"Building SED head: {head_type}...")
    head_cls = {"att_head": AttHead, "att_block_v2": AttBlockV2, "lse_head": LSEHead}.get(
        head_type, AttHead
    )
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
    sed_head = head_cls(**head_kwargs)

    # ========================================================================
    # Build mel spectrogram transform
    # ========================================================================

    sample_rate = config["data"]["sample_rate"]
    window_duration = config["data"]["window_duration"]
    mel_cfg = config.get("mel", {})
    mel_type = mel_cfg.get("type", "nnaudio")

    if mel_type == "ttahara":
        from birdclef.preprocessing.ttahara_mel import TtaharaMelTransform

        in_chans = config["encoder"].get("in_chans", 1)
        audio_transform = TtaharaMelTransform(
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
        logger.info(
            f"Mel transform (ttahara): sr={sample_rate}, n_fft={mel_cfg.get('n_fft', 2048)}, "
            f"win_length={mel_cfg.get('win_length', 626)}, hop={mel_cfg.get('hop_length', 313)}, "
            f"n_mels={mel_cfg.get('n_mels', 256)}, target_shape={mel_cfg.get('target_shape', [256, 256])}"
        )
    else:
        # Default: NNAudio mel spectrogram
        hop_length = mel_cfg.get("hop_length")
        if hop_length is None:
            frame_shift_ms = mel_cfg.get("frame_shift_ms", 10)
            hop_length = int(frame_shift_ms / 1000.0 * sample_rate)
        target_frames = int(window_duration * sample_rate / hop_length)

        norm_type = mel_cfg.get("norm_type", "audioset")
        audio_transform = NNAudioMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=mel_cfg.get("n_mels", 128),
            target_frames=target_frames,
            n_fft=mel_cfg.get("n_fft"),
            hop_length=hop_length,
            fmax=mel_cfg.get("fmax"),
            norm_type=norm_type,
            top_db=mel_cfg.get("top_db", 80.0),
        )
        logger.info(
            f"Mel transform: sr={sample_rate}, n_fft={mel_cfg.get('n_fft', 'auto')}, "
            f"n_mels={mel_cfg.get('n_mels', 128)}, hop={hop_length}, "
            f"target_frames={target_frames}, norm={norm_type}"
        )

    # ========================================================================
    # Build Lightning module
    # ========================================================================

    training_cfg = config["training"]
    waveform_mixup = training_cfg.get("raw_waveform_mixup", False)
    gpu_mel = mel_cfg.get("gpu", True)  # Default: compute mel on GPU

    # GPU mel transform: always used unless explicitly disabled.
    # With n_fft=4096, CPU-side mel in DataLoader workers is a bottleneck.
    gpu_mel_transform = None
    if gpu_mel or waveform_mixup:
        if mel_type == "ttahara":
            from birdclef.preprocessing.ttahara_mel import TtaharaMelTransform

            gpu_mel_transform = TtaharaMelTransform(
                sample_rate=sample_rate,
                n_fft=mel_cfg.get("n_fft", 2048),
                win_length=mel_cfg.get("win_length", 626),
                hop_length=mel_cfg.get("hop_length", 313),
                f_min=mel_cfg.get("f_min", 20),
                n_mels=mel_cfg.get("n_mels", 256),
                top_db=mel_cfg.get("top_db", 80.0),
                target_shape=tuple(mel_cfg.get("target_shape", [256, 256])),
                out_channels=config["encoder"].get("in_chans", 1),
            )
        else:
            gpu_mel_transform = NNAudioMelSpectrogram(
                sample_rate=sample_rate,
                n_mels=mel_cfg.get("n_mels", 128),
                target_frames=target_frames,
                n_fft=mel_cfg.get("n_fft"),
                hop_length=hop_length,
                fmax=mel_cfg.get("fmax"),
                norm_type=norm_type,
                top_db=mel_cfg.get("top_db", 80.0),
            )
        if waveform_mixup:
            logger.info("GPU mel + waveform mixup: mixup before mel on GPU")
        else:
            logger.info("GPU mel + spectrogram mixup: mel on GPU, then mixup")

    pl_module = SEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        lr=training_cfg["lr"],
        head_lr=training_cfg["head_lr"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=training_cfg["max_epochs"],
        warmup_epochs=training_cfg["warmup_epochs"],
        threshold=training_cfg["threshold"],
        clipwise_weight=training_cfg["clipwise_weight"],
        framewise_weight=training_cfg["framewise_weight"],
        blend_clip=training_cfg["blend_clip"],
        blend_frame=training_cfg["blend_frame"],
        mixup_prob=training_cfg.get("mixup_prob", 0.5),
        mixup_alpha=training_cfg.get("mixup_alpha", 0.5),
        mixup_warmup_epochs=training_cfg.get("mixup_warmup_epochs", 0),
        mixup_theta=training_cfg.get("mixup_theta", 1.0),
        mixup_per_sample=training_cfg.get("mixup_per_sample", False),
        loss_type=training_cfg.get("loss_type", "soft_ce"),
        label_smoothing=training_cfg.get("label_smoothing", 0.0),
        asl_gamma_neg=training_cfg.get("asl_gamma_neg", 4.0),
        asl_gamma_pos=training_cfg.get("asl_gamma_pos", 0.0),
        asl_clip=training_cfg.get("asl_clip", 0.05),
        mel_transform=gpu_mel_transform,
        waveform_mixup=waveform_mixup,
        scheduler_type=training_cfg.get("scheduler_type", "cosine"),
        onecycle_div_factor=training_cfg.get("onecycle_div_factor", 25.0),
        onecycle_final_div_factor=training_cfg.get("onecycle_final_div_factor", 4.0),
    )

    # ========================================================================
    # Build data module
    # ========================================================================

    # When mel runs on GPU, all datasets return raw waveforms (no CPU mel).
    # The lightning module computes mel on GPU in training_step, validation_step, test_step.
    train_transform = None if gpu_mel_transform is not None else audio_transform
    eval_transform = None if gpu_mel_transform is not None else audio_transform

    # Fold-based CV: override test settings, load fold assignments
    fold_id = args.fold
    fold_assignments_path = None
    if fold_id is not None:
        fold_assignments_path = Path(__file__).parent / "fold_assignments.json"
        if not fold_assignments_path.exists():
            raise FileNotFoundError(
                f"fold_assignments.json not found at {fold_assignments_path}. "
                "Run: python generate_folds.py"
            )
        config["data"]["test_fraction"] = 0.0
        config["data"]["soundscape_test_files"] = 0
        logger.info(f"Fold mode: fold_id={fold_id}, assignments={fold_assignments_path}")

    data_module = SEDDataModule(
        data_dir=str(data_dir),
        taxonomy_csv=str(taxonomy_csv),
        sample_rate=sample_rate,
        window_duration=window_duration,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        val_fraction=config["data"].get("val_fraction", 0.1),
        test_fraction=config["data"].get("test_fraction", 0.1),
        soundscape_val_files=config["data"].get("soundscape_val_files", 8),
        soundscape_test_files=config["data"].get("soundscape_test_files", 8),
        seed=config["data"].get("seed", 42),
        transform=train_transform,
        eval_transform=eval_transform,
        fold_id=fold_id,
        fold_assignments_path=str(fold_assignments_path) if fold_assignments_path else None,
    )

    # ========================================================================
    # Callbacks
    # ========================================================================

    callbacks = []
    checkpoint_callback = None
    gpu_monitor = None
    fulldata = config["data"].get("val_fraction", 0.1) == 0.0

    if gpu_check_mode:
        from birdclef.callbacks.gpu_monitor import GPUUtilizationMonitor

        gpu_monitor = GPUUtilizationMonitor(log_every_n_steps=10)
        callbacks.append(gpu_monitor)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        logger.info("GPU check mode: skipping checkpoints, early stopping, and device stats")
    else:
        if fold_id is not None:
            checkpoint_dir = (
                project_dir
                / "experiments"
                / EXPERIMENT_NAME
                / "checkpoints"
                / run_name
                / f"fold{fold_id}"
            )
        else:
            checkpoint_dir = (
                project_dir / "experiments" / EXPERIMENT_NAME / "checkpoints" / run_name
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if fulldata:
            # Fulldata mode: no val set, no early stopping.
            # Save checkpoints every N epochs + last. No "best" monitoring.
            every_n = config["callbacks"]["model_checkpoint"].get("every_n_epochs", 5)
            checkpoint_callback = ModelCheckpoint(
                every_n_epochs=every_n,
                save_top_k=-1,  # keep all periodic checkpoints
                save_last=True,
                filename="epoch{epoch:02d}",
                dirpath=checkpoint_dir,
                verbose=True,
            )
            callbacks.append(checkpoint_callback)
            logger.info(f"Fulldata mode: saving every {every_n} epochs + last (no early stopping)")
            logger.info(f"Checkpoints: {checkpoint_dir}")
        else:
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
            logger.info(f"Checkpoints: {checkpoint_dir}")

            early_stopping = EarlyStopping(
                monitor=config["callbacks"]["early_stopping"]["monitor"],
                mode=config["callbacks"]["early_stopping"]["mode"],
                patience=config["callbacks"]["early_stopping"]["patience"],
                min_delta=config["callbacks"]["early_stopping"]["min_delta"],
                verbose=True,
            )
            callbacks.append(early_stopping)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        device_stats = DeviceStatsMonitor()
        callbacks.append(device_stats)

    # ========================================================================
    # Loggers
    # ========================================================================

    loggers = []
    experiment_name = f"exp017-{run_name}"
    if fold_id is not None:
        experiment_name = f"{experiment_name}-fold{fold_id}"

    tb_logger = TensorBoardLogger(
        save_dir=Path(__file__).parent / "outputs",
        name="tensorboard",
        version=experiment_name,
    )
    loggers.append(tb_logger)

    if args.fast_dev_run or gpu_check_mode:
        logger.info(f"{'GPU check' if gpu_check_mode else 'Fast dev run'} — skipping W&B logging")
    else:
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
                "architecture": "sed",
                "num_classes": num_classes,
                "encoder_name": config["encoder"]["model_name"],
            }
        )
        loggers.append(wandb_logger)

    # ========================================================================
    # Trainer
    # ========================================================================

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
        limit_val_batches=0 if fulldata else 1.0,
        num_sanity_val_steps=0 if fulldata else 2,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ========================================================================
    # Train
    # ========================================================================

    logger.info("Starting training...")
    trainer.fit(pl_module, data_module)

    # GPU check: print summary and exit
    if gpu_check_mode:
        import os

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / f"{run_name}_{job_id}.txt"
        gpu_monitor.print_summary(save_path=summary_file)
        logger.info("GPU check complete!")
        return

    # ========================================================================
    # Test
    # ========================================================================

    test_results = None
    has_test = data_module.test_dataset is not None
    if not has_test:
        logger.info("No test set (test_fraction=0) — skipping test phase")
    elif not args.fast_dev_run and checkpoint_callback.best_model_path:
        logger.info(f"Testing: {checkpoint_callback.best_model_path}")
        test_results = trainer.test(
            pl_module, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path
        )
    elif args.fast_dev_run:
        logger.info("Fast dev run — running test step")
        test_results = trainer.test(pl_module, datamodule=data_module)

    # ========================================================================
    # Save results
    # ========================================================================

    if not args.fast_dev_run:
        results_dir = Path(__file__).parent / "outputs" / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        if fold_id is not None:
            results_file = results_dir / f"{run_name}-fold{fold_id}.json"
        else:
            results_file = results_dir / f"{run_name}.json"

        # Capture val metrics from trainer's logged values
        val_metrics = {}
        for key in [
            "val/auroc",
            "val/cmap",
            "val/f1_macro",
            "val/f1_micro",
            "val/top1_acc",
            "val/top5_acc",
            "val/loss",
        ]:
            val = trainer.callback_metrics.get(key)
            if val is not None:
                val_metrics[key] = float(val)

        raw_results = {
            "config_path": str(config_path),
            "run_name": run_name,
            "architecture": "sed",
            "num_classes": num_classes,
            "encoder": config["encoder"]["model_name"],
            "best_val_score": (
                float(checkpoint_callback.best_model_score)
                if hasattr(checkpoint_callback, "best_model_score")
                and checkpoint_callback.best_model_score is not None
                else None
            ),
            "best_checkpoint_path": str(
                checkpoint_callback.best_model_path
                if checkpoint_callback.best_model_path
                else checkpoint_callback.last_model_path or ""
            ),
            "stopped_epoch": trainer.current_epoch,
            "test_metrics": test_results[0] if test_results else None,
            "val_metrics": val_metrics if val_metrics else None,
        }
        with open(results_file, "w") as f:
            json.dump(raw_results, f, indent=2)
        logger.info(f"Results: {results_file}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
