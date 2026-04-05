"""CLI entry point for experiment 105.

Subcommands:
    extract  — run Perch v2 ONNX on BirdCLEF audio, write pooled-embedding cache
    train    — train a head on a cached shard against the exp 102 soundscape split
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="birdclef-105")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract Perch embeddings / frame features")
    p_extract.add_argument("--config", required=True, help="Path to exp105.yaml")
    p_extract.add_argument(
        "--source",
        choices=["soundscape", "focal"],
        default="soundscape",
        help="Which source to extract (default: soundscape)",
    )
    p_extract.add_argument(
        "--limit",
        type=int,
        default=None,
        help="(soundscape only) process only the first N audio files (pilot mode)",
    )
    p_extract.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="(focal only) which shard to process (0-indexed)",
    )
    p_extract.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="(focal only) total number of shards the focal set is split into",
    )

    p_train = sub.add_parser("train", help="Train a head on cached embeddings")
    p_train.add_argument("--config", required=True, help="Path to exp105.yaml")
    p_train.add_argument(
        "--head",
        choices=["linear", "mlp"],
        default="linear",
        help="Head architecture to train",
    )
    p_train.add_argument(
        "--run-name",
        default=None,
        help="Optional run name for logs/checkpoints",
    )

    args = parser.parse_args(argv)

    if args.command == "extract":
        from birdclef_105 import extract

        return extract.run(
            config_path=args.config,
            limit=args.limit,
            source=args.source,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    elif args.command == "train":
        from birdclef_105 import training

        return training.run(
            config_path=args.config, head=args.head, run_name=args.run_name
        )
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
