"""CLI entry point for exp 300.

    birdclef-300 extract --config configs/exp300.yaml --encoder eat --variant frozen
    birdclef-300 probe   --config configs/exp300.yaml --variant frozen --workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml

from birdclef_300.registry import REGISTRY


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(prog="birdclef-300")
    sub = parser.add_subparsers(dest="command")

    p_ext = sub.add_parser("extract", help="Extract one encoder's labeled-window embeddings")
    p_ext.add_argument("--config", required=True)
    p_ext.add_argument("--encoder", required=True, choices=sorted(REGISTRY))
    p_ext.add_argument("--variant", default="frozen",
                       help="Cache tag, e.g. frozen / lora-r8 / dapt-ep10 (used by exp 301)")
    p_ext.add_argument("--checkpoint", default=None,
                       help="Override checkpoint / fine-tuned model (torch backends)")
    p_ext.add_argument("--adapter", default=None,
                       help="PEFT LoRA adapter dir to merge in (exp 301 re-extraction)")
    p_ext.add_argument("--state-dict", default=None,
                       help="DAPT-updated state_dict to load (exp 301 re-extraction)")
    p_ext.add_argument("--output", default=None, help="Override output parquet path")
    p_ext.add_argument("--pooling", default=None,
                       help="Override token pooling (mean_patch|cls|mean_all) for torch encoders; "
                            "pair with a distinct --variant tag (e.g. frozen-cls)")
    p_ext.add_argument("--limit", type=int, default=None, help="Cap #windows (smoke test)")
    p_ext.add_argument("--patch", action="store_true",
                       help="Cache the per-window patch grid (patches.parquet) for the "
                            "prototypical probe's frame-level readout, not the pooled vector "
                            "(torch encoders only; leaves data.parquet untouched)")

    p_probe = sub.add_parser("probe", help="Run the ridge LOSO probe (Luigi)")
    p_probe.add_argument("--config", default=None, help="Config (for output.base_dir); optional")
    p_probe.add_argument("--encoders", default="beats,ast,eat,ssamba_hf,perch,birdnet")
    p_probe.add_argument("--variant", default="frozen")
    p_probe.add_argument("--features-dir", default=None,
                        help="Override cache dir (default: config output.base_dir)")
    p_probe.add_argument("--artifacts-dir", default=None)
    p_probe.add_argument("--no-standardize", action="store_true")
    p_probe.add_argument("--readout", default="ridge", choices=["ridge", "proto"],
                        help="ridge = linear LOSO probe (default); proto = prototypical "
                             "LOSO (clip-cosine + frame-max if patches cached) — the "
                             "MAE-appropriate readout for masked-SSL features")
    p_probe.add_argument("--workers", type=int, default=4)

    p_fit = sub.add_parser("fit-head", help="Fit a deployable ridge head on all training data")
    p_fit.add_argument("--features-dir", default=None,
                       help="Training cache (default: deploy.fit_head DEFAULT_FEATURES_DIR, exp106)")
    p_fit.add_argument("--mapping", default="artifacts/perch_species_mapping.json",
                       help="perch_species_mapping.json (234-species roster)")
    p_fit.add_argument("--sources", default="focal,soundscape",
                       help="comma-separated subset of {focal,soundscape}")
    p_fit.add_argument("--lambda", dest="lam", type=float, default=None,
                       help="ridge λ (default: deploy.fit_head DEFAULT_LAMBDA)")
    p_fit.add_argument("--encoder", default="perch")
    p_fit.add_argument("--out", default="artifacts/head.npz")

    args = parser.parse_args(argv)

    if args.command == "extract":
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # --pooling overrides the encoder's token pooling (torch backends read
        # enc_cfg["pooling"]); used for the CLS sensitivity variant.
        if args.pooling:
            cfg.setdefault("encoders", {}).setdefault(args.encoder, {})["pooling"] = args.pooling
        from birdclef_300.extract import extract_labeled

        sys.exit(extract_labeled(
            cfg, encoder=args.encoder, variant=args.variant,
            checkpoint=args.checkpoint, adapter=args.adapter, state_dict=args.state_dict,
            output_path=args.output, limit=args.limit, patch=args.patch,
        ))

    elif args.command == "probe":
        import multiprocessing
        import os

        import luigi

        # spawn avoids the polars/rayon fork deadlock (see exp 106 cli).
        multiprocessing.set_start_method("spawn", force=True)
        from birdclef_300.probe.base import ARTIFACTS_DIR, FEATURES_DIR
        from birdclef_300.probe.tasks import AllProbes

        features_dir = args.features_dir
        if features_dir is None and args.config:
            with open(args.config) as f:
                features_dir = yaml.safe_load(f)["output"]["base_dir"]
        if features_dir is None:
            features_dir = FEATURES_DIR

        luigi.build(
            [AllProbes(
                encoders=args.encoders,
                variant=args.variant,
                readout=args.readout,
                standardize=not args.no_standardize,
                features_dir=os.path.expanduser(features_dir),
                artifacts_dir=args.artifacts_dir or ARTIFACTS_DIR,
            )],
            workers=args.workers,
            local_scheduler=True,
            no_lock=True,
        )

    elif args.command == "fit-head":
        from birdclef_300.deploy.fit_head import (
            DEFAULT_FEATURES_DIR, DEFAULT_LAMBDA, fit_head, save_head,
            validate_on_soundscape,
        )

        features_dir = args.features_dir or DEFAULT_FEATURES_DIR
        head = fit_head(
            features_dir=features_dir,
            mapping_path=args.mapping,
            sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()),
            lam=args.lam if args.lam is not None else DEFAULT_LAMBDA,
            encoder=args.encoder,
        )
        save_head(head, args.out)
        validate_on_soundscape(head, features_dir)

    else:
        parser.print_help()
        sys.exit(1)
