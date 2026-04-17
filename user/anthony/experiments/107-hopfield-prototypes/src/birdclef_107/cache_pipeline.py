"""Shared helpers for staged exp107 cache builds.

This module contains the pure build logic used by the packaged CLI and
SLURM cache pipeline:

1. bootstrap metadata
2. shard-array workers
3. manifest/index finalization
4. final verification

The stages are intentionally coordination-free: orchestration happens in
CLI/sbatch wrappers, while this module defines the reusable build and
freshness semantics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from birdclef_106.data_loading import (
    build_binary_label_matrix,
    build_species_list,
    embeddings_to_numpy,
    extract_site,
    list_sites,
    logits_to_numpy,
)
from birdclef_107.cache import (
    SCHEMA_VERSION,
    _atomic_write_json,
    _atomic_write_npy,
    _atomic_write_npz,
    _load_manifest,
    _materialize_from_shards,
    _mtime_ns,
    _paths,
    _source_focal_shards,
    _source_labeled_path,
    _source_unsup_shards,
)
from birdclef_107.data import stratified_unsup_subset, weighted_focal_subset
from birdclef_107.perch_features import project_perch_sidecar

DEFAULT_FEATURES_DIR = "~/scratch/birdclef-2026/features/exp106"
DEFAULT_CACHE_DIR = "~/scratch/birdclef-2026/cache/exp107"
DEFAULT_ARTIFACTS_DIR = str(
    Path(__file__).resolve().parents[3] / "106-perch-bayesian" / "artifacts"
)


@dataclass(frozen=True)
class WorkUnit:
    kind: str
    shard_name: str | None = None

    @property
    def tag(self) -> str:
        return self.kind if self.shard_name is None else f"{self.kind}-{self.shard_name}"


def resolve_paths(cache_dir: str | Path) -> dict[str, Path]:
    return _paths(Path(cache_dir).expanduser())


def bootstrap_metadata_path(paths: dict[str, Path]) -> Path:
    return paths["base"] / "bootstrap_metadata.json"


def work_plan_path(paths: dict[str, Path]) -> Path:
    return paths["base"] / "work_plan.json"


def list_work_units(features_dir: str | Path) -> list[WorkUnit]:
    features_dir = str(Path(features_dir).expanduser())
    units = [WorkUnit("labeled")]
    units.extend(WorkUnit("focal", p.stem) for p in _source_focal_shards(features_dir))
    units.extend(WorkUnit("unsup", p.stem) for p in _source_unsup_shards(features_dir))
    return units


def write_work_plan(paths: dict[str, Path], units: list[WorkUnit]) -> None:
    payload = {
        "n_units": len(units),
        "units": [asdict(u) for u in units],
        "built_at": int(time.time()),
    }
    _atomic_write_json(work_plan_path(paths), payload)


def load_work_plan(paths: dict[str, Path]) -> list[WorkUnit] | None:
    path = work_plan_path(paths)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    try:
        return [WorkUnit(**u) for u in payload["units"]]
    except Exception:
        return None


def load_bootstrap_metadata(paths: dict[str, Path], features_dir: str | Path) -> dict | None:
    path = bootstrap_metadata_path(paths)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    features_dir = str(Path(features_dir).expanduser())
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    if payload.get("features_dir") != features_dir:
        return None

    labeled = _source_labeled_path(features_dir)
    if payload.get("labeled_parquet_mtime_ns") != _mtime_ns(labeled):
        return None

    focal_sig = [{"name": p.stem, "mtime_ns": _mtime_ns(p)} for p in _source_focal_shards(features_dir)]
    unsup_sig = [{"name": p.stem, "mtime_ns": _mtime_ns(p)} for p in _source_unsup_shards(features_dir)]
    if payload.get("focal_sources") != focal_sig:
        return None
    if payload.get("unsup_sources") != unsup_sig:
        return None
    return payload


def build_bootstrap_metadata(
    *,
    features_dir: str | Path,
    artifacts_dir: str | Path,
    paths: dict[str, Path],
) -> dict:
    features_dir = str(Path(features_dir).expanduser())
    artifacts_dir = str(Path(artifacts_dir).expanduser())
    paths["base"].mkdir(parents=True, exist_ok=True)

    species_list = build_species_list(artifacts_dir)
    labeled_path = _source_labeled_path(features_dir)
    labeled_df = pl.read_parquet(labeled_path, columns=["filename"])
    labeled_sites = {extract_site(str(fn)) for fn in labeled_df["filename"].to_list()}
    all_sites = sorted(set(list_sites(features_dir)) | labeled_sites)
    site_to_idx = {s: i for i, s in enumerate(all_sites)}

    recording_to_idx: dict[str, int] = {}
    unsup_sources = []
    for src in _source_unsup_shards(features_dir):
        unsup_sources.append({"name": src.stem, "mtime_ns": _mtime_ns(src)})
        df = pl.read_parquet(src, columns=["filename"])
        for recording in df["filename"].to_list():
            recording = str(recording)
            if recording not in recording_to_idx:
                recording_to_idx[recording] = len(recording_to_idx)

    focal_sources = [
        {"name": src.stem, "mtime_ns": _mtime_ns(src)}
        for src in _source_focal_shards(features_dir)
    ]

    payload = {
        "schema_version": SCHEMA_VERSION,
        "features_dir": features_dir,
        "species_list": species_list,
        "site_to_idx": site_to_idx,
        "recording_to_idx": recording_to_idx,
        "labeled_parquet_mtime_ns": _mtime_ns(labeled_path),
        "focal_sources": focal_sources,
        "unsup_sources": unsup_sources,
        "built_at": int(time.time()),
    }
    _atomic_write_json(bootstrap_metadata_path(paths), payload)
    write_work_plan(paths, list_work_units(features_dir))
    return payload


def source_path_for_unit(features_dir: str | Path, unit: WorkUnit) -> Path:
    features_path = Path(features_dir).expanduser()
    if unit.kind == "labeled":
        return _source_labeled_path(str(features_path))
    if unit.kind == "focal":
        assert unit.shard_name is not None
        return features_path / "focal" / f"{unit.shard_name}.parquet"
    if unit.kind == "unsup":
        assert unit.shard_name is not None
        return features_path / "soundscape_unlabeled" / f"{unit.shard_name}.parquet"
    raise ValueError(f"unknown unit kind: {unit.kind}")


def output_paths_for_unit(paths: dict[str, Path], unit: WorkUnit, include_perch: bool) -> dict[str, Path]:
    if unit.kind == "labeled":
        out = {"core": paths["labeled"]}
        if include_perch:
            out["perch"] = paths["labeled_perch"]
        return out
    if unit.kind == "focal":
        assert unit.shard_name is not None
        out = {"core": paths["focal"] / f"{unit.shard_name}.npz"}
        if include_perch:
            out["perch"] = paths["focal_perch"] / f"{unit.shard_name}.npz"
        return out
    if unit.kind == "unsup":
        assert unit.shard_name is not None
        out = {"core": paths["unsup"] / f"{unit.shard_name}.npz"}
        if include_perch:
            out["perch"] = paths["unsup_perch"] / f"{unit.shard_name}.npz"
        return out
    raise ValueError(f"unknown unit kind: {unit.kind}")


def _compact_perch_sidecar_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            return all(k in z.files for k in ("mapped_probs", "topk_idx", "topk_val"))
    except Exception:
        return False


def unit_complete(
    *,
    features_dir: str | Path,
    paths: dict[str, Path],
    unit: WorkUnit,
    include_perch: bool,
) -> bool:
    src = source_path_for_unit(features_dir, unit)
    src_mtime = _mtime_ns(src)
    outs = output_paths_for_unit(paths, unit, include_perch)
    core = outs["core"]
    if not core.exists() or core.stat().st_mtime_ns < src_mtime:
        return False
    if include_perch:
        perch = outs["perch"]
        if (
            not perch.exists()
            or perch.stat().st_mtime_ns < src_mtime
            or not _compact_perch_sidecar_ok(perch)
        ):
            return False
    return True


def build_unit(
    *,
    features_dir: str | Path,
    paths: dict[str, Path],
    metadata: dict,
    unit: WorkUnit,
    include_perch: bool,
) -> None:
    src = source_path_for_unit(features_dir, unit)

    if unit.kind == "labeled":
        df = pl.read_parquet(src, columns=["embedding", "ground_truth", "filename"])
        embeddings = embeddings_to_numpy(df)
        labels = build_binary_label_matrix(df, metadata["species_list"])
        filenames = [str(s) for s in df["filename"].to_list()]
        sites = np.asarray([extract_site(f) for f in filenames], dtype="U")
        _atomic_write_npz(
            paths["labeled"],
            embeddings=embeddings,
            labels=labels,
            sites=sites,
            filenames=np.asarray(filenames, dtype="U"),
        )
        if include_perch:
            perch_df = pl.read_parquet(src, columns=["perch_logits"])
            _atomic_write_npz(
                paths["labeled_perch"],
                **project_perch_sidecar(logits_to_numpy(perch_df), metadata["species_list"]),
            )
        return

    if unit.kind == "focal":
        assert unit.shard_name is not None
        df = pl.read_parquet(src, columns=["embedding", "ground_truth"])
        _atomic_write_npz(
            paths["focal"] / f"{unit.shard_name}.npz",
            embeddings=embeddings_to_numpy(df),
            labels=build_binary_label_matrix(df, metadata["species_list"]),
        )
        if include_perch:
            perch_df = pl.read_parquet(src, columns=["perch_logits"])
            _atomic_write_npz(
                paths["focal_perch"] / f"{unit.shard_name}.npz",
                **project_perch_sidecar(logits_to_numpy(perch_df), metadata["species_list"]),
            )
        return

    if unit.kind == "unsup":
        assert unit.shard_name is not None
        df = pl.read_parquet(src, columns=["embedding", "filename"])
        embeddings = embeddings_to_numpy(df)
        fnames = [str(s) for s in df["filename"].to_list()]
        site_idx = np.asarray(
            [metadata["site_to_idx"].get(extract_site(f), 0) for f in fnames],
            dtype=np.int64,
        )
        rec_idx = np.asarray(
            [metadata["recording_to_idx"][f] for f in fnames],
            dtype=np.int64,
        )
        _atomic_write_npz(
            paths["unsup"] / f"{unit.shard_name}.npz",
            embeddings=embeddings,
            site_indices=site_idx,
            recording_indices=rec_idx,
        )
        if include_perch:
            perch_df = pl.read_parquet(src, columns=["perch_logits"])
            _atomic_write_npz(
                paths["unsup_perch"] / f"{unit.shard_name}.npz",
                **project_perch_sidecar(logits_to_numpy(perch_df), metadata["species_list"]),
            )
        return

    raise ValueError(f"unknown unit kind: {unit.kind}")


def all_units_complete(
    *,
    features_dir: str | Path,
    paths: dict[str, Path],
    units: list[WorkUnit],
    include_perch: bool,
) -> bool:
    return all(
        unit_complete(
            features_dir=features_dir,
            paths=paths,
            unit=unit,
            include_perch=include_perch,
        )
        for unit in units
    )


def manifest_complete(
    *,
    features_dir: str | Path,
    paths: dict[str, Path],
    include_perch: bool,
) -> bool:
    manifest_path = paths["manifest"]
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False
    if manifest.get("features_dir") != str(Path(features_dir).expanduser()):
        return False
    if include_perch:
        if not manifest.get("labeled_has_perch", False):
            return False
        if not all(bool(e.get("has_perch", False)) for e in manifest.get("focal_shards", [])):
            return False
        if not all(bool(e.get("has_perch", False)) for e in manifest.get("unsup_shards", [])):
            return False
    latest_src = max(
        [_mtime_ns(_source_labeled_path(str(Path(features_dir).expanduser())))]
        + [_mtime_ns(p) for p in _source_focal_shards(str(Path(features_dir).expanduser()))]
        + [_mtime_ns(p) for p in _source_unsup_shards(str(Path(features_dir).expanduser()))]
    )
    return manifest_path.stat().st_mtime_ns >= latest_src


def finalize_manifest(
    *,
    features_dir: str | Path,
    paths: dict[str, Path],
    metadata: dict,
) -> dict:
    with np.load(paths["labeled"], allow_pickle=False) as z:
        labeled_n = int(len(z["embeddings"]))

    focal_entries = []
    cum_offset = 0
    for src in _source_focal_shards(str(Path(features_dir).expanduser())):
        core = paths["focal"] / f"{src.stem}.npz"
        perch = paths["focal_perch"] / f"{src.stem}.npz"
        with np.load(core, allow_pickle=False) as z:
            n_rows = int(len(z["embeddings"]))
        focal_entries.append({
            "name": src.stem,
            "parquet_mtime_ns": _mtime_ns(src),
            "n_rows": n_rows,
            "cum_offset": cum_offset,
            "has_perch": perch.exists(),
        })
        cum_offset += n_rows

    unsup_entries = []
    cum_offset = 0
    for src in _source_unsup_shards(str(Path(features_dir).expanduser())):
        core = paths["unsup"] / f"{src.stem}.npz"
        perch = paths["unsup_perch"] / f"{src.stem}.npz"
        with np.load(core, allow_pickle=False) as z:
            n_rows = int(len(z["embeddings"]))
        unsup_entries.append({
            "name": src.stem,
            "parquet_mtime_ns": _mtime_ns(src),
            "n_rows": n_rows,
            "cum_offset": cum_offset,
            "has_perch": perch.exists(),
        })
        cum_offset += n_rows

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "features_dir": str(Path(features_dir).expanduser()),
        "species_list": metadata["species_list"],
        "site_to_idx": metadata["site_to_idx"],
        "recording_to_idx": metadata["recording_to_idx"],
        "labeled": {
            "parquet_mtime_ns": metadata["labeled_parquet_mtime_ns"],
            "n_rows": labeled_n,
        },
        "labeled_has_perch": paths["labeled_perch"].exists(),
        "focal_shards": focal_entries,
        "unsup_shards": unsup_entries,
        "built_at": int(time.time()),
    }
    _atomic_write_json(paths["manifest"], manifest)
    return manifest


def build_optional_indexes(
    *,
    features_dir: str | Path,
    paths: dict[str, Path],
    focal_cache_size: int,
    unsup_cache_size: int,
    seed: int,
    force_rebuild: bool,
) -> list[Path]:
    manifest = _load_manifest(paths, str(Path(features_dir).expanduser()))
    manifest_path = paths["manifest"]
    written: list[Path] = []

    if int(focal_cache_size) > 0:
        idx_path = paths["idx"] / f"focal-n{int(focal_cache_size)}-seed{int(seed)}.npy"
        need = (
            force_rebuild
            or not idx_path.exists()
            or idx_path.stat().st_mtime_ns < manifest_path.stat().st_mtime_ns
        )
        if need:
            labels = _materialize_from_shards(
                paths["focal"], manifest["focal_shards"], None, ["labels"]
            )["labels"]
            idx = weighted_focal_subset(labels, int(focal_cache_size), int(seed))
            _atomic_write_npy(idx_path, idx)
            written.append(idx_path)

    if int(unsup_cache_size) > 0:
        idx_path = paths["idx"] / f"unsup-n{int(unsup_cache_size)}-seed{int(seed)}.npy"
        need = (
            force_rebuild
            or not idx_path.exists()
            or idx_path.stat().st_mtime_ns < manifest_path.stat().st_mtime_ns
        )
        if need:
            site_indices = _materialize_from_shards(
                paths["unsup"], manifest["unsup_shards"], None, ["site_indices"]
            )["site_indices"]
            idx = stratified_unsup_subset(site_indices, int(unsup_cache_size), int(seed))
            _atomic_write_npy(idx_path, idx)
            written.append(idx_path)

    return written


def verify_cache(
    *,
    features_dir: str | Path,
    cache_dir: str | Path,
    include_perch: bool,
    focal_cache_size: int = 0,
    unsup_cache_size: int = 0,
    seed: int = 42,
    sample_rows: int = 8,
) -> dict:
    """Validate a finalized cache build and return the manifest."""
    features_dir = str(Path(features_dir).expanduser())
    paths = resolve_paths(cache_dir)

    metadata = load_bootstrap_metadata(paths, features_dir)
    if metadata is None:
        raise ValueError("bootstrap metadata missing or stale")

    work_plan = load_work_plan(paths)
    expected_units = list_work_units(features_dir)
    if work_plan is None:
        raise ValueError("work_plan.json missing or invalid")
    if work_plan != expected_units:
        raise ValueError("work plan does not match source shard enumeration")

    if not all_units_complete(
        features_dir=features_dir,
        paths=paths,
        units=expected_units,
        include_perch=include_perch,
    ):
        raise ValueError("one or more shard outputs are missing/stale")

    manifest = _load_manifest(paths, features_dir)
    if not manifest_complete(features_dir=features_dir, paths=paths, include_perch=include_perch):
        raise ValueError("base_manifest.json missing or stale")

    if manifest["species_list"] != metadata["species_list"]:
        raise ValueError("species_list mismatch metadata vs manifest")
    if manifest["site_to_idx"] != metadata["site_to_idx"]:
        raise ValueError("site_to_idx mismatch metadata vs manifest")
    if manifest["recording_to_idx"] != metadata["recording_to_idx"]:
        raise ValueError("recording_to_idx mismatch metadata vs manifest")

    focal_total = sum(int(e["n_rows"]) for e in manifest["focal_shards"])
    unsup_total = sum(int(e["n_rows"]) for e in manifest["unsup_shards"])
    if int(manifest["labeled"]["n_rows"]) <= 0 or focal_total <= 0 or unsup_total <= 0:
        raise ValueError("manifest row counts are invalid")

    for unit in expected_units:
        outs = output_paths_for_unit(paths, unit, include_perch=include_perch)
        if not outs["core"].exists():
            raise ValueError(f"missing core output for {unit.tag}")
        if include_perch and not outs["perch"].exists():
            raise ValueError(f"missing Perch sidecar for {unit.tag}")

    if int(focal_cache_size) > 0:
        idx = paths["idx"] / f"focal-n{int(focal_cache_size)}-seed{int(seed)}.npy"
        if not idx.exists():
            raise ValueError(f"missing focal idx file {idx.name}")
    if int(unsup_cache_size) > 0:
        idx = paths["idx"] / f"unsup-n{int(unsup_cache_size)}-seed{int(seed)}.npy"
        if not idx.exists():
            raise ValueError(f"missing unsup idx file {idx.name}")

    sample_rows = max(1, int(sample_rows))
    focal_sample_n = min(sample_rows, focal_total)
    focal_fields = _materialize_from_shards(
        paths["focal"],
        manifest["focal_shards"],
        np.arange(focal_sample_n, dtype=np.int64),
        ["embeddings", "labels"],
    )
    if focal_fields["embeddings"].shape[0] != focal_sample_n or focal_fields["labels"].shape[0] != focal_sample_n:
        raise ValueError("focal sample materialization failed")

    unsup_sample_n = min(sample_rows, unsup_total)
    unsup_fields = _materialize_from_shards(
        paths["unsup"],
        manifest["unsup_shards"],
        np.arange(unsup_sample_n, dtype=np.int64),
        ["embeddings", "site_indices", "recording_indices"],
    )
    if (
        unsup_fields["embeddings"].shape[0] != unsup_sample_n
        or unsup_fields["site_indices"].shape[0] != unsup_sample_n
        or unsup_fields["recording_indices"].shape[0] != unsup_sample_n
    ):
        raise ValueError("unsup sample materialization failed")

    if include_perch:
        with np.load(paths["labeled_perch"], allow_pickle=False) as z:
            mapped = np.asarray(z["mapped_probs"])
            topk_idx = np.asarray(z["topk_idx"])
            topk_val = np.asarray(z["topk_val"])
        if mapped.shape[0] != int(manifest["labeled"]["n_rows"]):
            raise ValueError("labeled mapped_probs row count mismatch")
        if mapped.shape[1] != len(manifest["species_list"]):
            raise ValueError("labeled mapped_probs width mismatch")
        if topk_idx.shape != topk_val.shape:
            raise ValueError("labeled topk idx/val shape mismatch")

        focal_perch = _materialize_from_shards(
            paths["focal_perch"],
            manifest["focal_shards"],
            np.arange(focal_sample_n, dtype=np.int64),
            ["mapped_probs", "topk_idx", "topk_val"],
        )
        if focal_perch["mapped_probs"].shape != (focal_sample_n, len(manifest["species_list"])):
            raise ValueError("focal mapped_probs wrong shape")
        if focal_perch["topk_idx"].shape != focal_perch["topk_val"].shape:
            raise ValueError("focal topk idx/val shape mismatch")

        unsup_perch = _materialize_from_shards(
            paths["unsup_perch"],
            manifest["unsup_shards"],
            np.arange(unsup_sample_n, dtype=np.int64),
            ["mapped_probs", "topk_idx", "topk_val"],
        )
        if unsup_perch["mapped_probs"].shape != (unsup_sample_n, len(manifest["species_list"])):
            raise ValueError("unsup mapped_probs wrong shape")
        if unsup_perch["topk_idx"].shape != unsup_perch["topk_val"].shape:
            raise ValueError("unsup topk idx/val shape mismatch")

    return manifest


__all__ = [
    "DEFAULT_ARTIFACTS_DIR",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_FEATURES_DIR",
    "WorkUnit",
    "all_units_complete",
    "bootstrap_metadata_path",
    "build_bootstrap_metadata",
    "build_optional_indexes",
    "build_unit",
    "finalize_manifest",
    "list_work_units",
    "load_bootstrap_metadata",
    "load_work_plan",
    "manifest_complete",
    "output_paths_for_unit",
    "resolve_paths",
    "source_path_for_unit",
    "unit_complete",
    "verify_cache",
    "work_plan_path",
    "write_work_plan",
]
