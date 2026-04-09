"""CLI entry point for exp 106."""

from __future__ import annotations

import argparse
import logging
import sys

import yaml


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(prog="birdclef-106")
    sub = parser.add_subparsers(dest="command")

    # extract subcommand
    p_ext = sub.add_parser("extract", help="Extract Perch features to parquet")
    p_ext.add_argument("--config", required=True, help="YAML config path")
    p_ext.add_argument(
        "--source",
        required=True,
        choices=["soundscape-labeled", "soundscape-unlabeled", "focal"],
    )
    p_ext.add_argument("--output", required=True, help="Output parquet path")
    p_ext.add_argument("--shard-index", type=int, default=0)
    p_ext.add_argument("--num-shards", type=int, default=1)

    # analyze subcommand
    p_analyze = sub.add_parser("analyze", help="Run analysis pipeline (Luigi tasks)")
    p_analyze.add_argument("--phase", type=int, default=0, choices=[0, 1, 2, 3],
                           help="Phase to run (0=all, 1=static, 2=+models, 3=+UKF)")
    p_analyze.add_argument("--workers", type=int, default=4,
                           help="Luigi worker count (task parallelism)")
    p_analyze.add_argument("--features-dir",
                           default="~/scratch/birdclef-2026/features/exp106")
    p_analyze.add_argument("--artifacts-dir", default="artifacts")

    args = parser.parse_args(argv)

    if args.command == "extract":
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        from birdclef_106.extract import (
            extract_focal,
            extract_soundscape_labeled,
            extract_soundscape_unlabeled,
        )

        if args.source == "soundscape-labeled":
            rc = extract_soundscape_labeled(cfg, args.output)
        elif args.source == "soundscape-unlabeled":
            rc = extract_soundscape_unlabeled(
                cfg, args.output,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        elif args.source == "focal":
            rc = extract_focal(
                cfg, args.output,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        sys.exit(rc)
    elif args.command == "analyze":
        import multiprocessing
        import os
        import luigi
        # "spawn" starts fresh interpreter per worker — avoids polars/rayon
        # fork deadlock where child inherits parent's stale thread pool locks.
        multiprocessing.set_start_method("spawn", force=True)
        from birdclef_106.tasks import AllAnalysis
        luigi.build(
            [AllAnalysis(
                features_dir=os.path.expanduser(args.features_dir),
                artifacts_dir=args.artifacts_dir,
                phase=args.phase,
            )],
            workers=args.workers,
            local_scheduler=True,
            no_lock=True,
        )
        # Note: Luigi default retry_count=1 (2 attempts). Override via
        # [worker] retry_count=0 in luigi.cfg if retries waste time on bugs.
    else:
        parser.print_help()
        sys.exit(1)
