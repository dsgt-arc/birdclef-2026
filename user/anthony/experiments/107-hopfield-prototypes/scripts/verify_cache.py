#!/usr/bin/env python3
"""Verify a finalized exp107 cache build.

Checks structural invariants without rebuilding anything:
- bootstrap metadata exists and is fresh
- work plan exists and matches source shard enumeration
- all expected shard outputs are fresh
- final manifest exists, is fresh, and matches actual shard totals
- optional Perch sidecars exist when requested
- optional subset index files exist when requested
- sample rows can be materialized from the cached shards
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from birdclef_107.cache import _load_manifest, _materialize_from_shards
from birdclef_107.cache_pipeline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURES_DIR,
    all_units_complete,
    list_work_units,
    load_bootstrap_metadata,
    load_work_plan,
    manifest_complete,
    output_paths_for_unit,
    resolve_paths,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify exp107 cache outputs")
    p.add_argument("--features-dir", type=str, default=DEFAULT_FEATURES_DIR)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--include-perch-features",
        "--include-perch-logits",
        dest="include_perch_features",
        action="store_true",
        help="Require compact Perch sidecars to be present and sample-loadable.",
    )
    p.add_argument(
        "--focal-cache-size",
        type=int,
        default=0,
        help="If >0, require idx-v1/focal-n{N}-seed{S}.npy to exist.",
    )
    p.add_argument(
        "--unsup-cache-size",
        type=int,
        default=0,
        help="If >0, require idx-v1/unsup-n{N}-seed{S}.npy to exist.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sample-rows",
        type=int,
        default=8,
        help="Rows to sample-materialize from focal/unsup shards for sanity.",
    )
    p.add_argument("--print-manifest", action="store_true")
    return p.parse_args()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> None:
    args = parse_args()
    features_dir = str(Path(args.features_dir).expanduser())
    paths = resolve_paths(args.cache_dir)
    include_perch = bool(args.include_perch_features)

    metadata = load_bootstrap_metadata(paths, features_dir)
    require(metadata is not None, "verify: bootstrap metadata missing or stale")
    print("verify: bootstrap metadata OK", flush=True)

    work_plan = load_work_plan(paths)
    expected_units = list_work_units(features_dir)
    require(work_plan is not None, "verify: work_plan.json missing or invalid")
    require(work_plan == expected_units, "verify: work plan does not match source shard enumeration")
    print(f"verify: work plan OK ({len(work_plan)} units)", flush=True)

    require(
        all_units_complete(
            features_dir=features_dir,
            paths=paths,
            units=expected_units,
            include_perch=include_perch,
        ),
        "verify: one or more shard outputs are missing/stale",
    )
    print("verify: shard freshness OK", flush=True)

    manifest = _load_manifest(paths, features_dir)
    require(manifest_complete(features_dir=features_dir, paths=paths, include_perch=include_perch),
            "verify: base_manifest.json missing or stale")
    print("verify: manifest freshness OK", flush=True)

    require(manifest["species_list"] == metadata["species_list"], "verify: species_list mismatch metadata vs manifest")
    require(manifest["site_to_idx"] == metadata["site_to_idx"], "verify: site_to_idx mismatch metadata vs manifest")
    require(manifest["recording_to_idx"] == metadata["recording_to_idx"], "verify: recording_to_idx mismatch metadata vs manifest")

    focal_total = sum(int(e["n_rows"]) for e in manifest["focal_shards"])
    unsup_total = sum(int(e["n_rows"]) for e in manifest["unsup_shards"])
    require(int(manifest["labeled"]["n_rows"]) > 0, "verify: labeled row count is zero")
    require(focal_total > 0, "verify: focal shard total is zero")
    require(unsup_total > 0, "verify: unsup shard total is zero")
    print(
        f"verify: manifest counts labeled={manifest['labeled']['n_rows']} focal={focal_total} unsup={unsup_total}",
        flush=True,
    )

    for unit in expected_units:
        outs = output_paths_for_unit(paths, unit, include_perch=include_perch)
        require(outs["core"].exists(), f"verify: missing core output for {unit.tag}")
        if include_perch:
            require(outs["perch"].exists(), f"verify: missing Perch sidecar for {unit.tag}")

    if int(args.focal_cache_size) > 0:
        idx = paths["idx"] / f"focal-n{int(args.focal_cache_size)}-seed{int(args.seed)}.npy"
        require(idx.exists(), f"verify: missing focal idx file {idx.name}")
        print(f"verify: focal idx OK ({idx.name})", flush=True)
    if int(args.unsup_cache_size) > 0:
        idx = paths["idx"] / f"unsup-n{int(args.unsup_cache_size)}-seed{int(args.seed)}.npy"
        require(idx.exists(), f"verify: missing unsup idx file {idx.name}")
        print(f"verify: unsup idx OK ({idx.name})", flush=True)

    sample_rows = max(1, int(args.sample_rows))

    focal_sample_n = min(sample_rows, focal_total)
    focal_fields = _materialize_from_shards(
        paths["focal"],
        manifest["focal_shards"],
        np.arange(focal_sample_n, dtype=np.int64),
        ["embeddings", "labels"],
    )
    require(focal_fields["embeddings"].shape[0] == focal_sample_n, "verify: focal sample embeddings wrong shape")
    require(focal_fields["labels"].shape[0] == focal_sample_n, "verify: focal sample labels wrong shape")

    unsup_sample_n = min(sample_rows, unsup_total)
    unsup_fields = _materialize_from_shards(
        paths["unsup"],
        manifest["unsup_shards"],
        np.arange(unsup_sample_n, dtype=np.int64),
        ["embeddings", "site_indices", "recording_indices"],
    )
    require(unsup_fields["embeddings"].shape[0] == unsup_sample_n, "verify: unsup sample embeddings wrong shape")
    require(unsup_fields["site_indices"].shape[0] == unsup_sample_n, "verify: unsup site_indices wrong shape")
    require(unsup_fields["recording_indices"].shape[0] == unsup_sample_n, "verify: unsup recording_indices wrong shape")

    if include_perch:
        labeled_sidecar = np.load(paths["labeled_perch"], allow_pickle=False)
        try:
            mapped = np.asarray(labeled_sidecar["mapped_probs"])
            topk_idx = np.asarray(labeled_sidecar["topk_idx"])
            topk_val = np.asarray(labeled_sidecar["topk_val"])
        finally:
            labeled_sidecar.close()
        require(mapped.shape[0] == int(manifest["labeled"]["n_rows"]), "verify: labeled mapped_probs row count mismatch")
        require(mapped.shape[1] == len(manifest["species_list"]), "verify: labeled mapped_probs width mismatch")
        require(topk_idx.shape == topk_val.shape, "verify: labeled topk idx/val shape mismatch")

        focal_perch = _materialize_from_shards(
            paths["focal_perch"],
            manifest["focal_shards"],
            np.arange(focal_sample_n, dtype=np.int64),
            ["mapped_probs", "topk_idx", "topk_val"],
        )
        require(focal_perch["mapped_probs"].shape == (focal_sample_n, len(manifest["species_list"])),
                "verify: focal mapped_probs wrong shape")
        require(focal_perch["topk_idx"].shape == focal_perch["topk_val"].shape,
                "verify: focal topk idx/val shape mismatch")

        unsup_perch = _materialize_from_shards(
            paths["unsup_perch"],
            manifest["unsup_shards"],
            np.arange(unsup_sample_n, dtype=np.int64),
            ["mapped_probs", "topk_idx", "topk_val"],
        )
        require(unsup_perch["mapped_probs"].shape == (unsup_sample_n, len(manifest["species_list"])),
                "verify: unsup mapped_probs wrong shape")
        require(unsup_perch["topk_idx"].shape == unsup_perch["topk_val"].shape,
                "verify: unsup topk idx/val shape mismatch")
        print("verify: compact Perch sidecars OK", flush=True)

    if args.print_manifest:
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    print("verify: cache OK", flush=True)


if __name__ == "__main__":
    main()
