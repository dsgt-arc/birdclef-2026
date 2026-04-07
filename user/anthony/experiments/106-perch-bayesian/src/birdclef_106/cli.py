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
    else:
        parser.print_help()
        sys.exit(1)
