"""Exp 034 train entrypoint — XCPretrain fine-tune on BirdCLEF 2026.

Reuses exp 032's `DistilledSEDDataModule` for the 2026 data plumbing (focal +
labeled-SS, 5-fold split seed=42, raw-waveform batches — see results.md § B.8.3)
and wires in `XCPretrainWaveClassifier` + `XCPretrainFinetuneModule`.

Phase B.9.2 gate: `--fast-dev-run` runs 1 train + 1 val batch end-to-end
(no NaN/Inf, exit 0). Phases D/E layer in frozen-backbone, the pretrained ckpt,
W&B logging, checkpointing, and YAML configs.

Usage:
    # B.9.2 fast-dev-run smoke (random init, CPU-friendly)
    python train.py --fast-dev-run --batch-size 4 --num-workers 0 --accelerator cpu

    # Phase D Gate 2 (frozen backbone + pretrained, head-only, 5 epochs)
    python train.py --fold 0 --freeze-backbone --load-pretrained --max-epochs 5

    # Phase E (frozen-backbone fine-tune)
    python train.py --fold 0 --freeze-backbone --load-pretrained --max-epochs 25
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent.parent / "src"))  # user/murilo/src (belt-and-suspenders)
sys.path.insert(0, str(EXP_DIR / "scripts"))  # for load_sydorskyi_backbone

from birdclef.data.distilled_sed_datamodule import (  # noqa: E402
    DistilledSEDDataModule,
    build_sc_cache_meta,
)
from birdclef.data.distilled_sed_dataset import _default_audio_loader  # noqa: E402
from birdclef.data.soundscape import build_full_species_list  # noqa: E402
from birdclef.lightning.xcpretrain_finetune import XCPretrainFinetuneModule  # noqa: E402
from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("xcpretrain-train")

DEFAULT_DATA_DIR = "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026"
SAMPLE_RATE = 32000
WINDOW_DURATION = 5.0

# Sydorskyi SpecAugment config (selected_ebs.py:151-164).
SPEC_AUGMENT_CONFIG = {
    "freq_mask": {"mask_max_length": 10, "mask_max_masks": 3, "p": 0.3, "inplace": True},
    "time_mask": {"mask_max_length": 20, "mask_max_masks": 3, "p": 0.3, "inplace": True},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default="tf_efficientnetv2_s_in21k")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument(
        "--load-pretrained",
        action="store_true",
        help="load the Sydorskyi pretrained backbone into model.backbone (strict=True)",
    )
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--no-spec-augment", action="store_true")
    p.add_argument("--run-name", default="xcpretrain")
    p.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="gradient_clip_val (0 = none, matching Sydorskyi). Set e.g. 1.0 for full-FT stability.",
    )
    p.add_argument(
        "--save-ckpt",
        action="store_true",
        help="save checkpoints (Phase E onward). Gate 2 omits this (reads val/auroc only).",
    )
    p.add_argument(
        "--ckpt-root",
        default="/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/034-pretrained-backbone/checkpoints",
        help="checkpoint root on PROJECT storage (never home — 20GB quota).",
    )
    p.add_argument(
        "--wandb", action="store_true", help="also log to W&B (needs WANDB_API_KEY in env)."
    )
    args = p.parse_args()

    pl.seed_everything(1243, workers=True)  # Sydorskyi seed
    data_dir = Path(args.data_dir).expanduser()

    # Species / class list.
    species_list = build_full_species_list(data_dir / "taxonomy.csv")
    num_classes = len(species_list)
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    logger.info(f"Species: {num_classes} classes")

    # Data (reuse exp 032 plumbing).
    focal_df = pd.read_csv(data_dir / "train.csv")
    sc_labels_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    labeled_sc_files = sorted(sc_labels_df["filename"].unique().tolist())
    sc_df = build_sc_cache_meta(labeled_sc_files)
    window_samples = int(WINDOW_DURATION * SAMPLE_RATE)
    focal_audio_dir = data_dir / "train_audio"
    sc_audio_dir = data_dir / "train_soundscapes"

    def _focal_loader(path, n_samples, start=0, sr=SAMPLE_RATE):
        return _default_audio_loader(focal_audio_dir / path, n_samples, start, sr)

    def _sc_loader(path, n_samples, start=0, sr=SAMPLE_RATE):
        return _default_audio_loader(sc_audio_dir / path, n_samples, start, sr)

    data_module = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        fold_k=args.fold,
        n_folds=5,
        focal_seed=42,
        min_sample=20,
        window_samples=window_samples,
        sample_rate=SAMPLE_RATE,
        batch_size=args.batch_size,
        focal_share=0.9,
        sc_share=0.1,
        num_workers=args.num_workers,
        focal_audio_loader=_focal_loader,
        sc_audio_loader=_sc_loader,
    )

    # Model.
    head_config = {"p": 0.5, "num_class": num_classes, "train_period": 5.0, "infer_period": 5.0}
    spec_aug = None if args.no_spec_augment else SPEC_AUGMENT_CONFIG
    model = XCPretrainWaveClassifier(
        backbone=args.backbone,
        pretrained=False,
        exportable=True,
        head_config=head_config,
        spec_augment_config=spec_aug,
    )
    if args.load_pretrained:
        from load_sydorskyi_backbone import load_sydorskyi_backbone

        loaded = load_sydorskyi_backbone(args.backbone, verbose=True)
        model.backbone.load_state_dict(loaded.state_dict(), strict=True)
        logger.info("Loaded Sydorskyi pretrained backbone into model.backbone (strict=True)")

    module = XCPretrainFinetuneModule(
        model=model,
        num_classes=num_classes,
        lr=args.lr,
        max_epochs=args.max_epochs,
        freeze_backbone=args.freeze_backbone,
    )

    run_tag = f"{args.run_name}-fold{args.fold}"
    out_dir = EXP_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    loggers: list = [] if args.fast_dev_run else [CSVLogger(save_dir=str(out_dir), name=run_tag)]
    if args.wandb and not args.fast_dev_run:
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(
            WandbLogger(project="birdclef-2026-exp034", name=run_tag, save_dir=str(out_dir))
        )

    callbacks: list = []
    if args.save_ckpt and not args.fast_dev_run:
        ckpt_dir = Path(args.ckpt_root) / run_tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="{epoch}-{step}",
                monitor="val/auroc",
                mode="max",
                save_top_k=3,  # Sydorskyi n_checkpoints_to_save=3
                save_last=True,  # genuine last.ckpt — selected for inference (Sydorskyi pattern)
            )
        )
        logger.info(
            f"Checkpoints → {ckpt_dir} (save_top_k=3 + save_last; inference uses last.ckpt)"
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        precision="32-true",  # Sydorskyi: no AMP
        gradient_clip_val=(args.grad_clip or None),  # Sydorskyi: none (trainer_params={})
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
        logger=(loggers if loggers else False),
        callbacks=callbacks,
    )
    trainer.fit(module, data_module)

    # Persist the OOF metric (Gate 2 reads val/auroc on the fold-0 holdout SS pool).
    if not args.fast_dev_run:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                pass
        result = {
            "run_name": args.run_name,
            "fold": args.fold,
            "backbone": args.backbone,
            "freeze_backbone": args.freeze_backbone,
            "load_pretrained": args.load_pretrained,
            "max_epochs": args.max_epochs,
            "val_auroc": metrics.get("val/auroc"),
            "val_loss": metrics.get("val/loss"),
            "all_metrics": metrics,
        }
        result_path = out_dir / f"{run_tag}.json"
        result_path.write_text(json.dumps(result, indent=2))
        logger.info(f"OOF val/auroc = {result['val_auroc']}  →  {result_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
