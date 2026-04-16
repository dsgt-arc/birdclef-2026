#!/usr/bin/env python3
"""Stage 2: build one exp107 cache work unit for the SLURM shard array."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from birdclef_107.cache_pipeline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES_DIR,
    build_unit,
    list_work_units,
    load_bootstrap_metadata,
    resolve_paths,
    unit_complete,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build one exp107 cache shard work unit")
    p.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--task-id", type=int, default=None,
                   help="0-indexed work-unit id. Defaults to SLURM_ARRAY_TASK_ID.")
    p.add_argument(
        "--include-perch-features",
        "--include-perch-logits",
        dest="include_perch_features",
        action="store_true",
    )
    p.add_argument("--cache-force-rebuild-base", action="store_true")
    p.add_argument("--print-plan", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task_id = args.task_id
    if task_id is None:
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if slurm_task_id is None:
            raise SystemExit("--task-id is required when SLURM_ARRAY_TASK_ID is unset")
        task_id = int(slurm_task_id)

    features_dir = str(Path(args.features_dir).expanduser())
    paths = resolve_paths(args.cache_dir)
    units = list_work_units(features_dir)

    if args.print_plan:
        for i, unit in enumerate(units):
            print(f"{i:02d}: {unit.tag}")
        return

    if task_id < 0 or task_id >= len(units):
        raise SystemExit(f"task-id {task_id} outside work range 0..{len(units)-1}")

    metadata = load_bootstrap_metadata(paths, features_dir)
    if metadata is None:
        raise SystemExit("bootstrap metadata missing/stale; run stage 1 first")

    unit = units[task_id]
    print(f"shard[{task_id}]: unit={unit.tag}", flush=True)
    if not args.cache_force_rebuild_base and unit_complete(
        features_dir=features_dir,
        paths=paths,
        unit=unit,
        include_perch=bool(args.include_perch_features),
    ):
        print(f"shard[{task_id}]: skip fresh {unit.tag}", flush=True)
        return

    build_unit(
        features_dir=features_dir,
        paths=paths,
        metadata=metadata,
        unit=unit,
        include_perch=bool(args.include_perch_features),
    )
    print(f"shard[{task_id}]: wrote {unit.tag}", flush=True)


if __name__ == "__main__":
    main()
