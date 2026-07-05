"""Exp 034 train entrypoint — XCPretrain fine-tune on BirdCLEF 2026 (config-driven).

Matches the exp 032/033 convention: a YAML config (configs/*.yaml, inheriting
configs/_common.yaml) holds the recipe; this script takes `--config` + `--fold`.
Reuses exp 032's `DistilledSEDDataModule` for the 2026 data plumbing (see
results.md § B.8.3) and wires in `XCPretrainWaveClassifier` +
`XCPretrainFinetuneModule`.

Usage:
    # Phase D Gate 2 (sanity — frozen, 5 ep, no W&B/ckpt):
    python train.py --config configs/gate2-effv2s.yaml --fold 0
    # Phase E (frozen-backbone fine-tune, 25 ep, W&B + ckpt):
    python train.py --config configs/xcpretrain-effv2s-frozen.yaml --fold 0
    # fast-dev-run smoke (CPU):
    python train.py --config configs/gate2-effv2s.yaml --fast-dev-run --num-workers 0 --accelerator cpu

Logging (matches exp 032): TensorBoardLogger → outputs/tensorboard (always);
WandbLogger (entity=dsgt-arc) when logging.use_wandb.
Checkpoints → <project>/experiments/034-pretrained-backbone/checkpoints/<run>/fold<N>/.
Metrics → outputs/test_results/<run>-fold<N>.json (consumed by save_metrics.py).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

EXP_DIR = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parents[
    3
]  # .../birdclef-2026 (repo root; for repo-relative assets, e.g. Perch ONNX)
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

EXPERIMENT_NAME = "034-pretrained-backbone"
PROJECT_DIR = Path("/storage/project/ps-clef2026_birdclef-0/mgustineli3")

# Sydorskyi SpecAugment config (selected_ebs.py:151-164) — applied inside the model.
SPEC_AUGMENT_CONFIG = {
    "freq_mask": {"mask_max_length": 10, "mask_max_masks": 3, "p": 0.3, "inplace": True},
    "time_mask": {"mask_max_length": 20, "mask_max_masks": 3, "p": 0.3, "inplace": True},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (leaf values win)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    """Load a YAML config, resolving a single-level `inherit:` against _common.yaml."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.pop("inherit", None)
    if inherit:
        base = load_config(path.parent / inherit)
        cfg = _deep_merge(base, cfg)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, required=True, help="path to a configs/*.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--fast-dev-run", action="store_true")
    p.add_argument("--num-workers", type=int, default=None, help="override config data.num_workers")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", default="auto")
    args = p.parse_args()

    config_path = (
        (EXP_DIR / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    )
    cfg = load_config(config_path)
    run_name = cfg["run_name"]
    dcfg, mcfg, tcfg, ccfg, lcfg = (
        cfg["data"],
        cfg["model"],
        cfg["training"],
        cfg["callbacks"],
        cfg["logging"],
    )
    num_workers = args.num_workers if args.num_workers is not None else dcfg["num_workers"]
    experiment_name = f"exp034-{run_name}-fold{args.fold}"

    pl.seed_everything(tcfg["seed"], workers=True)
    logger.info(f"Config: {config_path.name}  Run: {run_name}  Fold: {args.fold}")

    # --- Species / class list -------------------------------------------------
    data_dir = Path(dcfg["data_dir"]).expanduser()
    sample_rate = dcfg["sample_rate"]
    species_list = build_full_species_list(data_dir / "taxonomy.csv")
    num_classes = len(species_list)
    species_to_idx = {s: i for i, s in enumerate(species_list)}
    assert num_classes == dcfg["num_classes"], f"{num_classes} != config {dcfg['num_classes']}"
    logger.info(f"Species: {num_classes} classes")

    # --- Data (reuse exp 032 plumbing) ---------------------------------------
    focal_df = pd.read_csv(data_dir / "train.csv")
    sc_labels_df = pd.read_csv(data_dir / "train_soundscapes_labels.csv")
    labeled_sc_files = sorted(sc_labels_df["filename"].unique().tolist())
    sc_df = build_sc_cache_meta(labeled_sc_files)
    window_samples = int(dcfg["window_seconds"] * sample_rate)
    focal_audio_dir = data_dir / "train_audio"
    sc_audio_dir = data_dir / "train_soundscapes"

    def _focal_loader(path, n_samples, start=0, sr=sample_rate):
        return _default_audio_loader(focal_audio_dir / path, n_samples, start, sr)

    def _sc_loader(path, n_samples, start=0, sr=sample_rate):
        return _default_audio_loader(sc_audio_dir / path, n_samples, start, sr)

    data_module = DistilledSEDDataModule(
        focal_df=focal_df,
        sc_df=sc_df,
        sc_labels_df=sc_labels_df,
        species_to_idx=species_to_idx,
        num_classes=num_classes,
        fold_k=args.fold,
        n_folds=dcfg["n_folds"],
        focal_seed=dcfg["seed"],
        min_sample=dcfg["min_sample"],
        window_samples=window_samples,
        sample_rate=sample_rate,
        batch_size=dcfg["batch_size"],
        focal_share=dcfg["focal_share"],
        sc_share=dcfg["sc_share"],
        num_workers=num_workers,
        focal_audio_loader=_focal_loader,
        sc_audio_loader=_sc_loader,
    )

    # --- Model ----------------------------------------------------------------
    head_config = {"num_class": num_classes, **mcfg["head"]}
    spec_aug = SPEC_AUGMENT_CONFIG if mcfg["spec_augment"] else None
    model = XCPretrainWaveClassifier(
        backbone=mcfg["backbone"],
        pretrained=False,
        exportable=True,
        head_config=head_config,
        spec_augment_config=spec_aug,
    )
    if mcfg["load_pretrained"]:
        from load_sydorskyi_backbone import load_sydorskyi_backbone

        loaded = load_sydorskyi_backbone(mcfg["backbone"], verbose=True)
        model.backbone.load_state_dict(loaded.state_dict(), strict=True)
        logger.info("Loaded Sydorskyi pretrained backbone into model.backbone (strict=True)")

    # Perch v2 KD (Phase H) — resolve the repo-relative ONNX path; fail loud if absent.
    kd_on = tcfg.get("use_perch_distill", False)
    perch_path = None
    if kd_on:
        perch_path = REPO_DIR / tcfg["perch_onnx_path"]
        if not perch_path.exists():
            raise FileNotFoundError(f"Perch v2 ONNX teacher not found: {perch_path}")
        logger.info(f"KD enabled — Perch teacher: {perch_path}")

    module = XCPretrainFinetuneModule(
        model=model,
        num_classes=num_classes,
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        max_epochs=tcfg["max_epochs"],
        eta_min=tcfg["eta_min"],
        freeze_backbone=mcfg["freeze_backbone"],
        label_smoothing=tcfg["label_smoothing"],
        use_random_filtering=tcfg["random_filtering"],
        rf_min_db=tcfg["rf_min_db"],
        focal_alpha=tcfg["focal_alpha"],
        focal_gamma=tcfg["focal_gamma"],
        use_perch_distill=kd_on,
        perch_onnx_path=(str(perch_path) if kd_on else None),
        perch_embed_dim=tcfg.get("perch_embed_dim", 1536),
        alpha_distill=tcfg.get("alpha_distill", 1.0),
        kd_detach_cls_features=tcfg.get("kd_detach_cls_features", True),
    )

    # --- Loggers (TensorBoard always; W&B for full runs) ----------------------
    out_dir = EXP_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    loggers: list = []
    if not args.fast_dev_run:
        loggers.append(
            TensorBoardLogger(save_dir=str(out_dir), name="tensorboard", version=experiment_name)
        )
        if lcfg.get("use_wandb"):
            from pytorch_lightning.loggers import WandbLogger

            wandb_logger = WandbLogger(
                entity=lcfg["entity"],
                project=lcfg["project"],
                name=experiment_name,
                save_dir=str(EXP_DIR / lcfg["save_dir"]),
                log_model=lcfg.get("log_model", False),
            )
            wandb_logger.experiment.config.update(cfg)
            wandb_logger.experiment.config.update(
                {"fold": args.fold, "num_classes": num_classes, "backbone": mcfg["backbone"]}
            )
            loggers.append(wandb_logger)

    # --- Callbacks ------------------------------------------------------------
    callbacks: list = []
    if ccfg.get("save_checkpoint") and not args.fast_dev_run:
        ckpt_dir = (
            PROJECT_DIR
            / "experiments"
            / EXPERIMENT_NAME
            / "checkpoints"
            / run_name
            / f"fold{args.fold}"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        mc = ccfg["model_checkpoint"]
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename=mc["filename"],
                monitor=mc["monitor"],
                mode=mc["mode"],
                save_top_k=mc["save_top_k"],
                save_last=mc["save_last"],
                verbose=True,
            )
        )
        logger.info(
            f"Checkpoints → {ckpt_dir} (save_top_k={mc['save_top_k']} + save_last; use last.ckpt)"
        )
    if loggers:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    # Force the tqdm progress bar (file-friendly) over the auto-selected RichProgressBar,
    # which repaints multi-line ANSI blocks every step and floods SLURM .log files.
    # refresh_rate=50 → a handful of clean progress lines per epoch (matches exp 032/033).
    callbacks.append(TQDMProgressBar(refresh_rate=50))

    # --- Trainer --------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=tcfg["max_epochs"],
        devices=args.devices,
        accelerator=args.accelerator,
        precision=tcfg["precision"],  # Sydorskyi: 32-true (no AMP)
        gradient_clip_val=(tcfg["grad_clip_val"] or None),  # Sydorskyi: none
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=lcfg.get("log_every_n_steps", 50),
        logger=(loggers if loggers else False),
        callbacks=callbacks,
    )
    trainer.fit(module, data_module)

    # --- Persist OOF metrics (consumed by save_metrics.py) --------------------
    if not args.fast_dev_run:
        metrics = {}
        for k, v in trainer.callback_metrics.items():
            try:
                metrics[k] = float(v)
            except (TypeError, ValueError):
                pass
        result = {
            "run_name": run_name,
            "fold": args.fold,
            "config": config_path.name,
            "backbone": mcfg["backbone"],
            "freeze_backbone": mcfg["freeze_backbone"],
            "load_pretrained": mcfg["load_pretrained"],
            "max_epochs": tcfg["max_epochs"],
            "use_perch_distill": kd_on,
            "kd_detach_cls_features": tcfg.get("kd_detach_cls_features", True) if kd_on else None,
            "alpha_distill": tcfg.get("alpha_distill", 1.0) if kd_on else None,
            "val_auroc": metrics.get("val/auroc"),
            "val_loss": metrics.get("val/loss"),
            "all_metrics": metrics,
        }
        results_dir = out_dir / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        result_path = results_dir / f"{run_name}-fold{args.fold}.json"
        result_path.write_text(json.dumps(result, indent=2))
        logger.info(f"OOF val/auroc = {result['val_auroc']}  →  {result_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
