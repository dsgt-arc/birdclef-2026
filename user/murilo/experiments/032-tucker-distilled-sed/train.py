#!/usr/bin/env python3
"""Training script for exp 032: Full Tucker distilled-SED replication.

Faithful port of tuckerarrants/bc2026-distilled-sed:
- EffNet-B0 NS-JFT backbone (tf_efficientnet_b0.ns_jft_in1k)
- Logit-space AttHead with GeMFreqPool (per
  analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md)
- Dual-supervision loss (0.5·BCE(clip) + 0.5·BCE(frame_max))
- Perch v2 embedding-space KD (MSE, ALPHA_DISTILL=1.0)
- Waveform mixup with MIXUP_HARD (max labels), Beta(0.4, 0.4)
- Focal × labeled-SS mixup
- SpecAugment (1 freq×10 + 2 time×10)
- AdamW + 2 ep linear warmup + 23 ep cosine to eta_min=1e-6
- MixSamp 90/10 focal/SS batch composition
- 5-fold (StratifiedKFold focal + GroupKFold SS)

Configs:
- configs/distilled_sed_control.yaml — KD off (back-compat receipt for the new
  pipeline; isolates KD lever from recipe changes)
- configs/distilled_sed_kd.yaml      — KD on (the test)

Both submitted as paired SLURM jobs with their own slugs; read deltas
vs paired control, NOT vs historical 0.906 (cross-run hardware variance
~0.007 LB per exp 031 drift investigation).

Usage:
    python train.py --config configs/distilled_sed_kd.yaml
    # fast-dev-run gradient-flow check:
    python train.py --config configs/distilled_sed_kd.yaml --fast-dev-run
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

from birdclef.data.distilled_sed_datamodule import (  # noqa: E402
    DistilledSEDDataModule,
    build_sc_cache_meta,
)
from birdclef.data.soundscape import build_full_species_list  # noqa: E402
from birdclef.encoders.efficientnet import EfficientNetEncoder  # noqa: E402
from birdclef.heads.sed_head import (  # noqa: E402
    AttBlockV2,
    AttHead,
    AttHeadLogitSpace,
    AttHeadMultiContext,
    LSEHead,
)
from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule  # noqa: E402
from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram  # noqa: E402
from birdclef.preprocessing.spec_augment import SpecAugment  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

DEFAULT_PROJECT_NAME = "ps-clef2026_birdclef-0"
EXPERIMENT_NAME = "032-tucker-distilled-sed"

# `repo_root` above is actually <repo>/user/murilo (3 parents from train.py).
# The actual repo root (where vendor/ lives) is one level higher.
ACTUAL_REPO_ROOT = repo_root.parent.parent


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def transfer_pretrained_weights(
    pl_module: pl.LightningModule,
    ckpt_path: str,
    source_species_list: list[str],
    target_species_list: list[str],
) -> None:
    """Transfer weights from a pretrained checkpoint to pl_module.

    Behaviour:
    - For every parameter in the checkpoint state_dict that matches the
      shape of the corresponding parameter in pl_module, copy directly.
    - For classifier-style parameters (whose first dimension equals
      ``len(source_species_list)``), do a partial copy: for each species
      in ``source ∩ target``, copy that row from the source index to the
      target index. Other rows keep their existing initialization.
    - Parameters absent in pl_module's state_dict are skipped silently.

    Logs a summary of how many parameters were copied directly, partially,
    or skipped, plus the names of any partial-copy keys (typically
    ``model.head.cls_fc.3.weight`` and ``model.head.cls_fc.3.bias``).
    """
    src_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_state = src_ckpt.get("state_dict", src_ckpt)
    dst_state = pl_module.state_dict()

    src_idx = {s: i for i, s in enumerate(source_species_list)}
    overlap = [s for s in target_species_list if s in src_idx]
    n_src = len(source_species_list)

    n_full = 0
    n_partial = 0
    n_skipped = 0
    partial_keys: list[str] = []

    for k, src_v in src_state.items():
        if k not in dst_state:
            n_skipped += 1
            continue
        dst_v = dst_state[k]
        if src_v.shape == dst_v.shape:
            dst_state[k] = src_v
            n_full += 1
        elif src_v.dim() >= 1 and src_v.shape[0] == n_src:
            new = dst_v.clone()
            for sp in overlap:
                s = src_idx[sp]
                t = target_species_list.index(sp)
                new[t] = src_v[s]
            dst_state[k] = new
            n_partial += 1
            partial_keys.append(k)
        else:
            n_skipped += 1

    pl_module.load_state_dict(dst_state, strict=True)

    logger.info(
        f"Pretrained weight transfer from {ckpt_path}: "
        f"{n_full} full copy, {n_partial} partial copy "
        f"({len(overlap)} of {len(target_species_list)} target species mapped from source), "
        f"{n_skipped} skipped"
    )
    if partial_keys:
        logger.info(f"  Partial-transfer keys: {partial_keys}")


def main():
    parser = argparse.ArgumentParser(description="Model training fixes")
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

    # Reproducibility: seed all RNGs before anything else
    seed = config["data"].get("seed", 42)
    pl.seed_everything(seed, workers=True)
    logger.info(f"Seed: {seed} (pl.seed_everything)")

    logger.info(f"Config: {config_path}")
    logger.info(f"Run name: {run_name}")

    if gpu_check_mode:
        config["training"]["max_epochs"] = args.gpu_check
        # Cap warmup to avoid OneCycleLR pct_start > 1.0
        warmup = config["training"].get("warmup_epochs", 0)
        if warmup >= args.gpu_check:
            config["training"]["warmup_epochs"] = 0
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
        use_post_head=config["encoder"].get("use_post_head", False),
    )

    # Use encoder's actual embed_dim (not config) — features_only=True may differ
    embed_dim = encoder.embed_dim
    logger.info(f"Encoder embed_dim: {embed_dim}")

    head_type = config["sed_head"].get("type", "att_head")
    logger.info(f"Building SED head: {head_type}...")
    head_cls = {
        "att_head": AttHead,
        "att_head_logit_space": AttHeadLogitSpace,
        "att_head_multi_context": AttHeadMultiContext,
        "att_block_v2": AttBlockV2,
        "lse_head": LSEHead,
    }.get(head_type, AttHead)
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
    elif head_type == "att_head_multi_context":
        head_kwargs["train_period"] = config["sed_head"].get("train_period", 15.0)
        head_kwargs["infer_period"] = config["sed_head"].get("infer_period", 5.0)
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
        # With max trick, mel is computed per 5s segment, not the full window
        max_trick_factor = config["training"].get("max_trick_factor", 1)
        if config["training"].get("max_trick", False) and max_trick_factor > 1:
            segment_duration = window_duration / max_trick_factor
            target_frames = int(segment_duration * sample_rate / hop_length)
            logger.info(
                f"Max trick: mel target_frames={target_frames} ({segment_duration}s segments)"
            )
        else:
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
        elif mel_type == "tucker":
            # Tucker-faithful mel: torchaudio MelSpectrogram + AmplitudeToDB(80),
            # NO internal normalization (per-sample z-score handled by the
            # module's per_sample_mel_norm step). Mirrors Tucker's
            # `MelSpecTransform` from bc2026-distilled-sed.ipynb cells 10/14.
            # See Phase D § D.2 diagnostic in
            # experiments/032-tucker-distilled-sed/scripts/diag_mel_pipeline.py
            # for the side-by-side that motivated this dispatch path.
            from birdclef.preprocessing.tucker_mel import TuckerMelSpectrogram

            gpu_mel_transform = TuckerMelSpectrogram(
                sample_rate=sample_rate,
                n_fft=mel_cfg.get("n_fft", 2048),
                hop_length=mel_cfg.get("hop_length", 512),
                n_mels=mel_cfg.get("n_mels", 256),
                fmin=mel_cfg.get("fmin", 20),
                fmax=mel_cfg.get("fmax", 16000),
                top_db=mel_cfg.get("top_db", 80.0),
                power=mel_cfg.get("power", 2.0),
            )
            logger.info(
                "GPU mel: TuckerMelSpectrogram (torchaudio MelSpec + "
                "AmplitudeToDB(80), no internal norm; per-sample z-score via module)"
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

    # SpecAugment (task 0.5): Tucker's notebook cell 10 applies
    # 1 freq mask (param=10) + 2 time masks (param=10) after per-sample
    # mel normalization, training only. Config block `spec_augment`
    # under top-level config controls it. Default: enabled with Tucker's
    # exact params.
    sa_cfg = config.get("spec_augment", {}) or {}
    if sa_cfg.get("enabled", True):
        spec_augment_module = SpecAugment(
            freq_mask_param=sa_cfg.get("freq_mask_param", 10),
            time_mask_param=sa_cfg.get("time_mask_param", 10),
            num_freq_masks=sa_cfg.get("num_freq_masks", 1),
            num_time_masks=sa_cfg.get("num_time_masks", 2),
        )
        logger.info(f"SpecAugment enabled: {spec_augment_module.extra_repr()}")
    else:
        spec_augment_module = None
        logger.info("SpecAugment disabled (config.spec_augment.enabled=false)")

    pl_module = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        mel_transform=gpu_mel_transform,
        spec_augment=spec_augment_module,
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=training_cfg["max_epochs"],
        warmup_epochs=training_cfg["warmup_epochs"],
        warmup_start_factor=training_cfg.get("warmup_start_factor", 1.0 / 25),
        eta_min=training_cfg.get("eta_min", 1e-6),
        clipwise_weight=training_cfg.get("clipwise_weight", 0.5),
        framewise_weight=training_cfg.get("framewise_weight", 0.5),
        blend_clip=training_cfg.get("blend_clip", 0.5),
        blend_frame=training_cfg.get("blend_frame", 0.5),
        threshold=training_cfg.get("threshold", 0.5),
        per_sample_mel_norm=training_cfg.get("per_sample_mel_norm", True),
        # --- Perch v2 KD branch (exp 031 wiring, exp 032 recipe) ---
        use_perch_distill=training_cfg.get("use_perch_distill", False),
        perch_onnx_path=(
            str(ACTUAL_REPO_ROOT / training_cfg["perch_onnx_path"])
            if training_cfg.get("use_perch_distill", False)
            else None
        ),
        perch_embed_dim=training_cfg.get("perch_embed_dim", 1536),
        alpha_distill=training_cfg.get("alpha_distill", 1.0),
    )

    # ========================================================================
    # Optional: load pretrained weights with classifier transfer
    # ========================================================================

    init_cfg = config.get("init", {}) or {}
    pretrain_ckpt = init_cfg.get("from_checkpoint", "")
    if pretrain_ckpt:
        src_taxonomy_csv = init_cfg.get("source_taxonomy_csv")
        if not src_taxonomy_csv:
            raise ValueError(
                "init.from_checkpoint set but init.source_taxonomy_csv is missing. "
                "Specify the taxonomy CSV used for pretraining so we know which species "
                "indices to copy."
            )
        src_species_list = build_full_species_list(Path(src_taxonomy_csv))
        logger.info(f"Loading pretrained weights from {pretrain_ckpt}")
        logger.info(f"  Source taxonomy ({len(src_species_list)} species): {src_taxonomy_csv}")
        logger.info(f"  Target taxonomy ({len(species_list)} species): {taxonomy_csv}")
        transfer_pretrained_weights(pl_module, pretrain_ckpt, src_species_list, species_list)
    else:
        logger.info(
            "No init.from_checkpoint set — using ImageNet init for backbone "
            "(equivalent to v5j baseline)."
        )

    # ========================================================================
    # Build data module
    # ========================================================================

    # DistilledSEDDataModule always returns raw waveforms — the LightningModule
    # computes mel on GPU via mel_transform. The legacy audio_transform handling
    # is unused here. (Retained `audio_transform` build above for backward-
    # compatible config keys; reference suppressed via `_ = audio_transform`.)
    _ = audio_transform

    # Fold-based CV: --fold N overrides `data.fold_k` and adds /fold{N} to
    # the checkpoint dir + W&B run name. DistilledSEDDataModule builds folds
    # inline via StratifiedKFold(seed=42) + GroupKFold, so no external
    # fold_assignments.json is needed (the file requirement was vestigial
    # copy-paste from exp 019/021's split-file pattern).
    fold_id = args.fold
    if fold_id is not None:
        if not 0 <= fold_id < 5:
            raise ValueError(f"--fold must be in [0, 4], got {fold_id}")
        logger.info(f"Fold mode: fold_id={fold_id} (inline-built via DistilledSEDDataModule)")

    # ------------------------------------------------------------------
    # DistilledSEDDataModule (task 0.8): load focal + SS data, build folds.
    # ------------------------------------------------------------------
    import pandas as pd

    focal_df = pd.read_csv(data_dir / "train.csv")
    sc_labels_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    labeled_sc_files = sorted(sc_labels_df["filename"].unique().tolist())
    sc_df = build_sc_cache_meta(labeled_sc_files)
    species_to_idx = {s: i for i, s in enumerate(species_list)}

    window_samples = int(window_duration * sample_rate)
    data_cfg = config["data"]
    # Tucker's mixup defaults are configured under config["mixup"] for clarity;
    # all task-0.4 FocalDS kwargs flow through here.
    mixup_cfg = config.get("mixup", {}) or {}
    focal_ds_kwargs = {
        "mixup_prob": mixup_cfg.get("focal_focal_prob", 0.5),
        "mixup_alpha": mixup_cfg.get("alpha", 0.4),
        "mixup_hard": mixup_cfg.get("hard", True),
        "focal_sc_mixup_prob": mixup_cfg.get("focal_sc_prob", 0.5),
        "focal_sc_mixup_alpha": mixup_cfg.get("focal_sc_alpha", 0.4),
        "aug_prob": mixup_cfg.get("aug_prob", 0.5),
    }
    # Audio dirs: focal lives at data_dir/train_audio, SS at data_dir/train_soundscapes.
    focal_audio_dir = data_dir / "train_audio"
    sc_audio_dir = data_dir / "train_soundscapes"
    from birdclef.data.distilled_sed_dataset import _default_audio_loader as _audio_loader

    def _focal_loader(path, n_samples, start=0, sr=sample_rate):
        return _audio_loader(focal_audio_dir / path, n_samples, start, sr)

    def _sc_loader(path, n_samples, start=0, sr=sample_rate):
        return _audio_loader(sc_audio_dir / path, n_samples, start, sr)

    data_module = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        fold_k=fold_id if fold_id is not None else 0,
        n_folds=data_cfg.get("n_folds", 5),
        focal_seed=data_cfg.get("seed", 42),
        min_sample=data_cfg.get("min_sample", 20),
        window_samples=window_samples,
        sample_rate=sample_rate,
        batch_size=data_cfg["batch_size"],
        focal_share=data_cfg.get("focal_share", 0.9),
        sc_share=data_cfg.get("sc_share", 0.1),
        steps_per_epoch=data_cfg.get("steps_per_epoch"),
        num_workers=data_cfg.get("num_workers", 0),
        focal_audio_loader=_focal_loader,
        sc_audio_loader=_sc_loader,
        focal_ds_kwargs=focal_ds_kwargs,
        seed_epoch_shift=data_cfg.get("seed_epoch_shift", True),
        base_seed=data_cfg.get("base_seed", 42),
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
            every_n = config["callbacks"]["model_checkpoint"].get("every_n_epochs", 5)
            checkpoint_callback = ModelCheckpoint(
                every_n_epochs=every_n,
                save_top_k=-1,
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
    experiment_name = f"exp032-{run_name}"
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
        deterministic=True,
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

    # DistilledSEDDataModule does not expose a test_dataloader — Phase 1
    # uses the val fold (held-out SS) as its LB-style metric, and there's no
    # separate test set in the Tucker recipe. Skip the test stage entirely.
    test_results = None
    logger.info("Skipping test phase (DistilledSEDDataModule has no test split)")

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
