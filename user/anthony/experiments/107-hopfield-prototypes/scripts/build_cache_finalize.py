#!/usr/bin/env python3
"""Stage 3: finalize exp107 cache manifest and optional subset indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from birdclef_107.cache_pipeline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES_DIR,
    all_units_complete,
    build_optional_indexes,
    finalize_manifest,
    list_work_units,
    load_bootstrap_metadata,
    manifest_complete,
    resolve_paths,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finalize exp107 cache manifest/indexes")
    p.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--include-perch-features",
        "--include-perch-logits",
        dest="include_perch_features",
        action="store_true",
    )
    p.add_argument("--focal-cache-size", type=int, default=0)
    p.add_argument("--unsup-cache-size", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-force-rebuild", action="store_true")
    p.add_argument("--cache-force-rebuild-base", action="store_true")
    p.add_argument("--print-manifest", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features_dir = str(Path(args.features_dir).expanduser())
    paths = resolve_paths(args.cache_dir)
    metadata = load_bootstrap_metadata(paths, features_dir)
    if metadata is None:
        raise SystemExit("bootstrap metadata missing/stale; run stage 1 first")

    units = list_work_units(features_dir)
    if not all_units_complete(
        features_dir=features_dir,
        paths=paths,
        units=units,
        include_perch=bool(args.include_perch_features),
    ):
        raise SystemExit("not all shard outputs are fresh; stage 2 did not complete successfully")

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


if __name__ == "__main__":
    main()
