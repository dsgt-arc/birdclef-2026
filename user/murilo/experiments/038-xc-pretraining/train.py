#!/usr/bin/env python3
"""Exp 038 Stage 1 — broad Neotropical XC supervised pretraining (EffNetV2-S backbone).

Pretrains a multi-label bird classifier over the curated BirdSet-Neotropical vocabulary under the
SAME regime as the downstream SED stack — EffNetV2-S (ImageNet-21k) → Tucker mel →
`AttHeadLogitSpace` — so the backbone transfers when Stage 2 re-heads it to the 234 targets and
runs the KD-SED recipe. Reuses `DistilledSEDFinetuneModule` UNCHANGED (`num_classes=2683`, KD off);
the only new data plumbing is `XCPretrainDataModule`.

This is intentionally a stripped-down clone of exp 037's `train.py`: no KD, no pseudo-labels, no
folds, no warm-start/weight-transfer, Tucker mel only. The artifact is the backbone — checkpoints
are kept with `save_top_k=-1` so Stage 2 can pick the XC-val macro-AUC peak.

Configs (thin diffs over `configs/_common.yaml`):
    configs/pretrain_cellA_dualbce.yaml   # primary: dual-BCE + LS 0.05 + full aug
    configs/pretrain_cellB_focalbce.yaml  # insurance: focal-BCE
    configs/pretrain_cellC_nomixup.yaml   # insurance: no mixup

Usage:
    python train.py --config configs/pretrain_cellA_dualbce.yaml
    python train.py --config configs/pretrain_cellA_dualbce.yaml --fast-dev-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# user-level .env (W&B / HF tokens) — load before W&B imports
_user_dir = Path(__file__).parent.parent.parent
load_dotenv(_user_dir / ".env")

import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger  # noqa: E402

# Add <repo>/user/murilo/src to path (birdclef is imported from there, not pip-installed editable)
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from birdclef.data.xc_pretrain_dataset import XCPretrainDataModule  # noqa: E402
from birdclef.encoders.efficientnet import EfficientNetEncoder  # noqa: E402
from birdclef.heads.sed_head import (  # noqa: E402
    AttBlockV2,
    AttHead,
    AttHeadLogitSpace,
    AttHeadMultiContext,
    LSEHead,
)
from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule  # noqa: E402
from birdclef.preprocessing.spec_augment import SpecAugment  # noqa: E402
from birdclef.preprocessing.tucker_mel import TuckerMelSpectrogram  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "038-xc-pretraining"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge — override wins. Returns a NEW dict (inputs unmutated)."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Path) -> dict:
    """Load a cell YAML and deep-merge it over `_common.yaml` (same dir) if present."""
    with open(config_path) as f:
        override = yaml.safe_load(f) or {}
    common_p = config_path.parent / "_common.yaml"
    if common_p.exists() and config_path.name != "_common.yaml":
        with open(common_p) as f:
            base = yaml.safe_load(f) or {}
        logger.info(f"Loaded config: {config_path.name} (inherits _common.yaml)")
        return _deep_merge(base, override)
    return override


def build_encoder(cfg: dict) -> EfficientNetEncoder:
    enc = EfficientNetEncoder(
        model_name=cfg["model_name"],
        pretrained=cfg.get("pretrained", True),
        drop_path_rate=cfg.get("drop_path_rate", 0.0),
        in_chans=cfg.get("in_chans", 1),
        use_post_head=cfg.get("use_post_head", True),
    )
    logger.info(f"Encoder: {cfg['model_name']} (embed_dim={enc.embed_dim})")
    return enc


def build_head(cfg: dict, in_channels: int, num_classes: int):
    head_type = cfg.get("type", "att_head_logit_space")
    head_cls = {
        "att_head": AttHead,
        "att_head_logit_space": AttHeadLogitSpace,
        "att_head_multi_context": AttHeadMultiContext,
        "att_block_v2": AttBlockV2,
        "lse_head": LSEHead,
    }.get(head_type, AttHeadLogitSpace)
    kwargs = {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "hidden_dim": cfg.get("hidden_dim", 512),
        "dropout": cfg.get("dropout", 0.5),
        "gem_p": cfg.get("gem_p", 3.0),
    }
    logger.info(f"SED head: {head_type} (in={in_channels}, classes={num_classes})")
    return head_cls(**kwargs)


def build_mel(cfg: dict, sample_rate: int) -> TuckerMelSpectrogram:
    mel = TuckerMelSpectrogram(
        sample_rate=sample_rate,
        n_fft=cfg.get("n_fft", 2048),
        hop_length=cfg.get("hop_length", 512),
        n_mels=cfg.get("n_mels", 256),
        fmin=cfg.get("fmin", 20),
        fmax=cfg.get("fmax", 16000),
        top_db=cfg.get("top_db", 80.0),
        power=cfg.get("power", 2.0),
    )
    logger.info(f"Mel: TuckerMelSpectrogram ({mel.extra_repr()})")
    return mel


def main():
    parser = argparse.ArgumentParser(description="Exp 038 XC supervised pretraining")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--project-name", type=str, default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override data.batch_size (e.g. smaller on 24 GB rtx6000 vs 80 GB a100).",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=None,
        help="Override training.accumulate_grad_batches (keep effective batch ~128 on smaller GPUs).",
    )
    parser.add_argument(
        "--gpu-check",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Run N epochs (default 1) with GPU-utilization monitoring; no W&B, no checkpoints. "
        "Prints a throughput/util summary + real epoch wall-time (→ 25-ep ETA) and flags whether "
        "data loading is the bottleneck. Use before committing the full grid. Mutually exclusive "
        "with --fast-dev-run.",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    run_name = config["run_name"]

    seed = config["data"].get("seed", 42)
    pl.seed_everything(seed, workers=True)
    # Enable TF32 fp32 matmuls (free speedup on A100/rtx6000 tensor cores; we train in 16-mixed
    # anyway, so this only affects residual fp32 ops). PyTorch flags this when unset.
    torch.set_float32_matmul_precision("high")
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.accumulate_grad_batches is not None:
        config["training"]["accumulate_grad_batches"] = args.accumulate_grad_batches

    gpu_check_mode = args.gpu_check > 0
    if args.fast_dev_run and gpu_check_mode:
        parser.error("--fast-dev-run and --gpu-check are mutually exclusive")
    if gpu_check_mode:
        config["training"]["max_epochs"] = args.gpu_check
        if config["training"].get("warmup_epochs", 0) >= args.gpu_check:
            config["training"]["warmup_epochs"] = 0  # keep cosine T_max > 0 for short schedules
        logger.info(
            f"GPU-check mode: {args.gpu_check} epoch(s) — GPU monitor on, no W&B/checkpoints"
        )
    logger.info(f"Config: {config_path} | run: {run_name} | seed: {seed}")

    data_cfg = config["data"]
    train_cfg = config["training"]
    sample_rate = data_cfg["sample_rate"]
    window_samples = int(data_cfg["window_duration"] * sample_rate)

    # --- species vocabulary (defines num_classes + the head width) ---
    species_list_path = Path(data_cfg["species_list_path"])
    species_list = json.loads(species_list_path.read_text())
    num_classes = len(species_list)
    logger.info(f"Pretrain vocabulary: {num_classes} species ({species_list_path})")

    # --- model: EffNetV2-S → AttHeadLogitSpace, Tucker mel, SpecAugment ---
    encoder = build_encoder(config["encoder"])
    sed_head = build_head(config.get("sed_head", {}), encoder.embed_dim, num_classes)
    mel_transform = build_mel(config.get("mel", {}), sample_rate)

    sa_cfg = config.get("spec_augment", {}) or {}
    spec_augment = None
    if sa_cfg.get("enabled", True):
        spec_augment = SpecAugment(
            freq_mask_param=sa_cfg.get("freq_mask_param", 10),
            time_mask_param=sa_cfg.get("time_mask_param", 10),
            num_freq_masks=sa_cfg.get("num_freq_masks", 1),
            num_time_masks=sa_cfg.get("num_time_masks", 2),
        )
        logger.info(f"SpecAugment: {spec_augment.extra_repr()}")

    pl_module = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        mel_transform=mel_transform,
        spec_augment=spec_augment,
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        max_epochs=train_cfg["max_epochs"],
        warmup_epochs=train_cfg.get("warmup_epochs", 2),
        warmup_start_factor=train_cfg.get("warmup_start_factor", 1.0 / 25),
        eta_min=train_cfg.get("eta_min", 1e-6),
        clipwise_weight=train_cfg.get("clipwise_weight", 0.5),
        framewise_weight=train_cfg.get("framewise_weight", 0.5),
        blend_clip=train_cfg.get("blend_clip", 0.5),
        blend_frame=train_cfg.get("blend_frame", 0.5),
        per_sample_mel_norm=train_cfg.get("per_sample_mel_norm", True),
        use_perch_distill=False,  # pretraining is pure supervised classification (no KD)
        loss_type=train_cfg.get("loss_type", "bce"),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
        focal_alpha=train_cfg.get("focal_alpha", 0.25),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
    )

    # --- data ---
    mixup_cfg = config.get("mixup", {}) or {}
    data_module = XCPretrainDataModule(
        manifest_path=data_cfg["manifest_path"],
        species_list_path=species_list_path,
        audio_dir=data_cfg["audio_dir"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 8),
        window_samples=window_samples,
        val_fraction=data_cfg.get("val_fraction", 0.02),
        aug=data_cfg.get("aug", True),
        mixup=mixup_cfg.get("enabled", True),
        mixup_prob=mixup_cfg.get("prob", 0.5),
        mixup_alpha=mixup_cfg.get("alpha", 0.4),
        aug_prob=mixup_cfg.get("aug_prob", 0.5),
        sample_rate=sample_rate,
        seed=seed,
        require_audio=data_cfg.get("require_audio", True),
    )

    # --- output dirs + provenance ---
    project_dir = Path(f"/storage/project/{args.project_name}/mgustineli3")
    run_dir = project_dir / "experiments" / EXPERIMENT_NAME / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # Persist the exact species→idx so Stage 2 can map the pretrain head if ever needed.
    (run_dir / "species_to_idx.json").write_text(
        json.dumps({sp: i for i, sp in enumerate(species_list)})
    )

    # --- callbacks ---
    if gpu_check_mode:
        from birdclef.callbacks.gpu_monitor import GPUUtilizationMonitor

        gpu_monitor = GPUUtilizationMonitor(log_every_n_steps=10)
        checkpoint_callback = None
        callbacks = [gpu_monitor, LearningRateMonitor(logging_interval="step")]
    else:
        gpu_monitor = None
        ckpt_cfg = config.get("callbacks", {}).get("model_checkpoint", {})
        checkpoint_callback = ModelCheckpoint(
            monitor=ckpt_cfg.get("monitor", "val/auroc"),
            mode=ckpt_cfg.get("mode", "max"),
            save_top_k=ckpt_cfg.get("save_top_k", -1),
            save_last=ckpt_cfg.get("save_last", True),
            # Name by epoch only — embedding {val/auroc} in the filename is fragile (the "/" in the
            # metric key breaks the format string). save_top_k=-1 keeps all; best is tracked via
            # best_model_path / best_model_score and written to pretrain_results.json.
            filename=ckpt_cfg.get("filename", "epoch{epoch:02d}"),
            auto_insert_metric_name=False,
            dirpath=run_dir,
            verbose=True,
        )
        callbacks = [
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            DeviceStatsMonitor(),
        ]
        logger.info(
            f"Checkpoints → {run_dir} (monitor={checkpoint_callback.monitor}, save_top_k=-1)"
        )

    # --- loggers ---
    experiment_name = f"exp038-{run_name}"
    loggers = [
        TensorBoardLogger(
            save_dir=Path(__file__).parent / "outputs", name="tensorboard", version=experiment_name
        )
    ]
    if not args.fast_dev_run and not gpu_check_mode:
        log_cfg = config.get("logging", {})
        wandb_logger = WandbLogger(
            entity=log_cfg.get("entity", "dsgt-arc"),
            project=log_cfg.get("project", "birdclef-2026-exp038"),
            name=experiment_name,
            save_dir=str(Path(__file__).parent / log_cfg.get("save_dir", "outputs/wandb")),
            log_model=log_cfg.get("log_model", False),
        )
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(
            {"num_classes": num_classes, "encoder_name": config["encoder"]["model_name"]}
        )
        loggers.append(wandb_logger)
    else:
        logger.info(f"{'GPU-check' if gpu_check_mode else 'Fast dev run'} — skipping W&B")

    # --- trainer ---
    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        precision=train_cfg.get("precision", "16-mixed"),
        gradient_clip_val=train_cfg.get("grad_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.get("logging", {}).get("log_every_n_steps", 50),
        fast_dev_run=5 if args.fast_dev_run else False,
        num_sanity_val_steps=2,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=train_cfg.get("deterministic", False),
    )

    logger.info("Starting pretraining...")
    trainer.fit(pl_module, data_module)

    if gpu_check_mode:
        import os

        job_id = os.environ.get("SLURM_JOB_ID", "local")
        summary_dir = Path(__file__).parent / "outputs" / "gpu_check"
        summary_dir.mkdir(parents=True, exist_ok=True)
        gpu_monitor.print_summary(save_path=summary_dir / f"{run_name}_{job_id}.txt")
        logger.info("GPU-check complete!")
        return

    if not args.fast_dev_run:
        results = {
            "config_path": str(config_path),
            "run_name": run_name,
            "encoder": config["encoder"]["model_name"],
            "num_classes": num_classes,
            "loss_type": train_cfg.get("loss_type", "bce"),
            "best_val_auroc": (
                float(checkpoint_callback.best_model_score)
                if checkpoint_callback.best_model_score is not None
                else None
            ),
            "best_checkpoint_path": checkpoint_callback.best_model_path or "",
            "last_checkpoint_path": checkpoint_callback.last_model_path or "",
            "stopped_epoch": trainer.current_epoch,
        }
        results_file = run_dir / "pretrain_results.json"
        results_file.write_text(json.dumps(results, indent=2))
        logger.info(f"Results → {results_file}")

    logger.info("Pretraining complete!")


if __name__ == "__main__":
    main()
