#!/usr/bin/env python3
"""Stage 1: bootstrap exp107 cache metadata for the SLURM dependency chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from birdclef_107.cache_pipeline import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES_DIR,
    bootstrap_metadata_path,
    build_bootstrap_metadata,
    list_work_units,
    load_bootstrap_metadata,
    resolve_paths,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap exp107 cache metadata")
    p.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--artifacts-dir", type=str, default=DEFAULT_ARTIFACTS_DIR)
    p.add_argument("--cache-force-rebuild-base", action="store_true")
    p.add_argument("--print-metadata", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features_dir = str(Path(args.features_dir).expanduser())
    paths = resolve_paths(args.cache_dir)

    metadata = None
    if not args.cache_force_rebuild_base:
        metadata = load_bootstrap_metadata(paths, features_dir)

    if metadata is None:
        print("bootstrap: building metadata", flush=True)
        metadata = build_bootstrap_metadata(
            features_dir=features_dir,
            artifacts_dir=args.artifacts_dir,
            paths=paths,
        )
    else:
        print("bootstrap: metadata already fresh", flush=True)

    units = list_work_units(features_dir)
    print(
        f"bootstrap: work_units={len(units)} "
        f"(labeled=1 focal={sum(u.kind == 'focal' for u in units)} "
        f"unsup={sum(u.kind == 'unsup' for u in units)})",
        flush=True,
    )
    print(f"bootstrap: wrote {bootstrap_metadata_path(paths)}", flush=True)

    if args.print_metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
