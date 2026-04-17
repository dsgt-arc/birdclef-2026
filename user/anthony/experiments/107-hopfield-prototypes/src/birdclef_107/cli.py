"""Package CLI for experiment 107."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from birdclef_107.cache_pipeline import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES_DIR,
    all_units_complete,
    build_bootstrap_metadata,
    build_optional_indexes,
    build_unit,
    finalize_manifest,
    list_work_units,
    load_bootstrap_metadata,
    manifest_complete,
    resolve_paths,
    unit_complete,
    verify_cache,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experiment 107 utilities")
    sub = p.add_subparsers(dest="command", required=True)

    p_cache = sub.add_parser("cache", help="Cache build/verify helpers")
    cache_sub = p_cache.add_subparsers(dest="cache_command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
        sp.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)

    p_plan = cache_sub.add_parser("plan", help="Print ordered work units")
    add_common(p_plan)
    p_plan.add_argument("--count-only", action="store_true")

    p_boot = cache_sub.add_parser("bootstrap", help="Build bootstrap metadata/work plan")
    add_common(p_boot)
    p_boot.add_argument("--artifacts-dir", type=str, default=DEFAULT_ARTIFACTS_DIR)
    p_boot.add_argument("--cache-force-rebuild-base", action="store_true")
    p_boot.add_argument("--print-metadata", action="store_true")

    p_shard = cache_sub.add_parser("shard", help="Build one labeled/focal/unsup work unit")
    add_common(p_shard)
    p_shard.add_argument("--task-id", type=int, default=None,
                         help="0-indexed work-unit id. Defaults to SLURM_ARRAY_TASK_ID.")
    p_shard.add_argument(
        "--include-perch-features",
        "--include-perch-logits",
        dest="include_perch_features",
        action="store_true",
    )
    p_shard.add_argument("--cache-force-rebuild-base", action="store_true")
    p_shard.add_argument("--print-plan", action="store_true")

    p_fin = cache_sub.add_parser("finalize", help="Write final manifest + optional idx files")
    add_common(p_fin)
    p_fin.add_argument(
        "--include-perch-features",
        "--include-perch-logits",
        dest="include_perch_features",
        action="store_true",
    )
    p_fin.add_argument("--focal-cache-size", type=int, default=0)
    p_fin.add_argument("--unsup-cache-size", type=int, default=0)
    p_fin.add_argument("--seed", type=int, default=42)
    p_fin.add_argument("--cache-force-rebuild", action="store_true")
    p_fin.add_argument("--cache-force-rebuild-base", action="store_true")
    p_fin.add_argument("--print-manifest", action="store_true")

    p_verify = cache_sub.add_parser("verify", help="Verify finalized cache outputs")
    add_common(p_verify)
    p_verify.add_argument(
        "--include-perch-features",
        "--include-perch-logits",
        dest="include_perch_features",
        action="store_true",
    )
    p_verify.add_argument("--focal-cache-size", type=int, default=0)
    p_verify.add_argument("--unsup-cache-size", type=int, default=0)
    p_verify.add_argument("--seed", type=int, default=42)
    p_verify.add_argument("--sample-rows", type=int, default=8)
    p_verify.add_argument("--print-manifest", action="store_true")

    return p


def _cache_plan(args: argparse.Namespace) -> int:
    units = list_work_units(args.features_dir)
    if args.count_only:
        print(len(units))
        return 0
    for i, unit in enumerate(units):
        print(f"{i:02d}: {unit.tag}")
    return 0


def _cache_bootstrap(args: argparse.Namespace) -> int:
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
    if args.print_metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return 0


def _cache_shard(args: argparse.Namespace) -> int:
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
        return 0
    if task_id < 0 or task_id >= len(units):
        raise SystemExit(f"task-id {task_id} outside work range 0..{len(units)-1}")

    metadata = load_bootstrap_metadata(paths, features_dir)
    if metadata is None:
        raise SystemExit("bootstrap metadata missing/stale; run cache bootstrap first")

    unit = units[task_id]
    print(f"shard[{task_id}]: unit={unit.tag}", flush=True)
    if not args.cache_force_rebuild_base and unit_complete(
        features_dir=features_dir,
        paths=paths,
        unit=unit,
        include_perch=bool(args.include_perch_features),
    ):
        print(f"shard[{task_id}]: skip fresh {unit.tag}", flush=True)
        return 0

    build_unit(
        features_dir=features_dir,
        paths=paths,
        metadata=metadata,
        unit=unit,
        include_perch=bool(args.include_perch_features),
    )
    print(f"shard[{task_id}]: wrote {unit.tag}", flush=True)
    return 0


def _cache_finalize(args: argparse.Namespace) -> int:
    features_dir = str(Path(args.features_dir).expanduser())
    paths = resolve_paths(args.cache_dir)
    metadata = load_bootstrap_metadata(paths, features_dir)
    if metadata is None:
        raise SystemExit("bootstrap metadata missing/stale; run cache bootstrap first")

    units = list_work_units(features_dir)
    if not all_units_complete(
        features_dir=features_dir,
        paths=paths,
        units=units,
        include_perch=bool(args.include_perch_features),
    ):
        raise SystemExit("not all shard outputs are fresh; cache shard stage did not complete successfully")

    if args.cache_force_rebuild_base or not manifest_complete(
        features_dir=features_dir,
        paths=paths,
        include_perch=bool(args.include_perch_features),
    ):
        manifest = finalize_manifest(
            features_dir=features_dir,
            paths=paths,
            metadata=metadata,
        )
        print(f"finalize: wrote {paths['manifest']}", flush=True)
    else:
        manifest = json.loads(paths["manifest"].read_text())
        print("finalize: manifest already fresh", flush=True)

    written = build_optional_indexes(
        features_dir=features_dir,
        paths=paths,
        focal_cache_size=int(args.focal_cache_size),
        unsup_cache_size=int(args.unsup_cache_size),
        seed=int(args.seed),
        force_rebuild=bool(args.cache_force_rebuild),
    )
    for path in written:
        print(f"finalize: wrote {path}", flush=True)
    print(
        "finalize: summary "
        f"labeled={manifest['labeled']['n_rows']} "
        f"focal={sum(int(e['n_rows']) for e in manifest['focal_shards'])} "
        f"unsup={sum(int(e['n_rows']) for e in manifest['unsup_shards'])}",
        flush=True,
    )
    if args.print_manifest:
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


def _cache_verify(args: argparse.Namespace) -> int:
    manifest = verify_cache(
        features_dir=args.features_dir,
        cache_dir=args.cache_dir,
        include_perch=bool(args.include_perch_features),
        focal_cache_size=int(args.focal_cache_size),
        unsup_cache_size=int(args.unsup_cache_size),
        seed=int(args.seed),
        sample_rows=int(args.sample_rows),
    )
    print("verify: bootstrap metadata OK", flush=True)
    print(f"verify: work plan OK ({1 + len(manifest['focal_shards']) + len(manifest['unsup_shards'])} units)", flush=True)
    print("verify: shard freshness OK", flush=True)
    print("verify: manifest freshness OK", flush=True)
    print(
        "verify: manifest counts "
        f"labeled={manifest['labeled']['n_rows']} "
        f"focal={sum(int(e['n_rows']) for e in manifest['focal_shards'])} "
        f"unsup={sum(int(e['n_rows']) for e in manifest['unsup_shards'])}",
        flush=True,
    )
    if int(args.focal_cache_size) > 0:
        print(f"verify: focal idx OK (focal-n{int(args.focal_cache_size)}-seed{int(args.seed)}.npy)", flush=True)
    if int(args.unsup_cache_size) > 0:
        print(f"verify: unsup idx OK (unsup-n{int(args.unsup_cache_size)}-seed{int(args.seed)}.npy)", flush=True)
    if args.include_perch_features:
        print("verify: compact Perch sidecars OK", flush=True)
    if args.print_manifest:
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    print("verify: cache OK", flush=True)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "cache":
        if args.cache_command == "plan":
            return _cache_plan(args)
        if args.cache_command == "bootstrap":
            return _cache_bootstrap(args)
        if args.cache_command == "shard":
            return _cache_shard(args)
        if args.cache_command == "finalize":
            return _cache_finalize(args)
        if args.cache_command == "verify":
            return _cache_verify(args)
    raise SystemExit(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
