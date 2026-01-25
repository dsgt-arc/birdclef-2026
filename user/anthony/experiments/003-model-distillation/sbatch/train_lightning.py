#!/usr/bin/env python
"""Distillation training script using PyTorch Lightning.

Standalone script for training student models with comprehensive monitoring.

Usage:
    python train_lightning.py --student convnext-small --source-cache /path/to/cache

    # With profiling
    python train_lightning.py --student convnext-small --source-cache /path/to/cache --profile

    # Custom hyperparameters
    python train_lightning.py --student convnext-small --source-cache /path/to/cache \
        --batch-size 128 --lr 2e-4 --max-epochs 30
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from birdclef.callbacks import DataPerfMonitor, SystemIOMonitor
from birdclef.lightning import DistillationDataModule, DistillationLitModule
from birdclef.students import count_parameters, get_student_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distillation training with PyTorch Lightning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--student",
        type=str,
        default="convnext-small",
        help="Student model name (convnext-tiny, convnext-small, convnext-base, etc.)",
    )

    # Data arguments
    parser.add_argument(
        "--source-cache",
        type=str,
        required=True,
        help="Directory containing safetensors teacher cache files",
    )
    parser.add_argument(
        "--local-cache",
        type=str,
        default=None,
        help="Local NVMe cache directory (default: $TMPDIR/teacher_cache)",
    )
    parser.add_argument(
        "--no-copy-to-local",
        action="store_true",
        help="Disable copying shards to local NVMe cache",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size per iteration",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Maximum training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of shards for validation",
    )

    # DataLoader arguments
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches to prefetch per worker",
    )
    parser.add_argument(
        "--shards-per-worker",
        type=int,
        default=3,
        help="Maximum shards to keep in memory per worker",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="distillation",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=50,
        help="Log metrics every N steps",
    )

    # Profiling and debugging
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler (generates trace files)",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
        help="Directory for profiler output (default: output_dir/profiler)",
    )
    parser.add_argument(
        "--profiler",
        type=str,
        choices=["simple", "advanced", "pytorch"],
        default="simple",
        help="Profiler type: simple (timing), advanced (cProfile), pytorch (CUDA traces)",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run quick sanity check (1 batch train + val)",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="Limit training batches per epoch (useful for profiling)",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=None,
        help="Limit validation batches per epoch (useful for profiling)",
    )

    # Hardware
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, gpu, cpu)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Training precision (32, 16-mixed, bf16-mixed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # Set up paths
    source_cache = Path(args.source_cache)
    if not source_cache.exists():
        raise ValueError(f"Source cache directory does not exist: {source_cache}")

    local_cache = args.local_cache
    if local_cache is None:
        tmpdir = os.environ.get("TMPDIR", "/tmp")
        local_cache = Path(tmpdir) / "teacher_cache"
    else:
        local_cache = Path(local_cache)

    output_dir = args.output_dir
    if output_dir is None:
        scratch_dir = Path.home() / "scratch" / "birdclef-2026"
        output_dir = scratch_dir / "outputs" / "exp003-distillation-lightning"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DISTILLATION TRAINING - PyTorch Lightning")
    print("=" * 60)
    print(f"Student model: {args.student}")
    print(f"Source cache: {source_cache}")
    print(f"Local cache: {local_cache}")
    print(f"Output dir: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Precision: {args.precision}")
    print("=" * 60)

    # Create student model
    print(f"\nCreating student model: {args.student}")
    student = get_student_model(args.student)
    n_params = count_parameters(student)
    print(f"  Parameters: {n_params:,}")

    # Create data module
    print("\nSetting up data module...")
    datamodule = DistillationDataModule(
        source_cache_dir=source_cache,
        local_cache_dir=local_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shards_per_worker=args.shards_per_worker,
        copy_to_local=not args.no_copy_to_local,
        val_split=args.val_split,
        seed=args.seed,
    )

    # Create Lightning module
    print("Creating Lightning module...")
    model = DistillationLitModule(
        student=student,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
    )

    # Set up callbacks
    callbacks = [
        # Model checkpointing
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename=f"{args.student}-{{epoch:02d}}-{{val/loss:.4f}}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        # Early stopping
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=args.patience,
            verbose=True,
        ),
        # Learning rate monitoring
        LearningRateMonitor(logging_interval="epoch"),
        # GPU stats
        DeviceStatsMonitor(cpu_stats=False),
        # Custom data performance monitoring
        DataPerfMonitor(log_every_n_batches=args.log_every_n_steps),
        # System I/O monitoring
        SystemIOMonitor(log_every_n_batches=args.log_every_n_steps),
    ]

    # Set up logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=args.experiment_name,
        version=args.student,
    )

    # Set up profiler if requested
    profiler = None
    if args.profile:
        profile_dir = args.profile_dir
        if profile_dir is None:
            profile_dir = output_dir / "profiler"
        else:
            profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)

        if args.profiler == "simple":
            from pytorch_lightning.profilers import SimpleProfiler

            profiler = SimpleProfiler(
                dirpath=profile_dir,
                filename=f"fit-{args.student}_profile",
            )
            print(f"\nSimpleProfiler enabled - output: {profile_dir}")

        elif args.profiler == "advanced":
            from pytorch_lightning.profilers import AdvancedProfiler

            profiler = AdvancedProfiler(
                dirpath=profile_dir,
                filename=f"fit-{args.student}_profile",
            )
            print(f"\nAdvancedProfiler enabled - output: {profile_dir}")

        else:  # pytorch
            from pytorch_lightning.profilers import PyTorchProfiler

            profiler = PyTorchProfiler(
                dirpath=profile_dir,
                filename=f"{args.student}_profile",
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=2,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            print(f"\nPyTorchProfiler enabled - output: {profile_dir}")

    # Create trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=logger,
        profiler=profiler,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=True,
        fast_dev_run=args.fast_dev_run,
        gradient_clip_val=1.0,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Student: {args.student}")
    print(f"Parameters: {n_params:,}")

    # Checkpoint info (not available in fast_dev_run mode)
    if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
        print(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        if trainer.checkpoint_callback.best_model_score is not None:
            print(f"Best val/loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    else:
        print("Best checkpoint: (disabled in fast_dev_run mode)")

    print(f"TensorBoard logs: {logger.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
