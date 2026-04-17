"""Persistent sharded cache for exp107 training data.

See ``docs/cache-usage.md`` for the user-facing doc and rationale.

Layout (under ``cache_root``)::

    base-v1/
    ├── base_manifest.json         # schema_version, species_list, site_to_idx,
    │                              # recording_to_idx, per-shard fingerprints,
    │                              # labeled_has_perch, built_at
    ├── labeled.npz                # embeddings, labels, sites, filenames
    ├── labeled_perch.npz          # OPTIONAL mapped+dense Perch sidecar
    ├── focal/shard-NN.npz         # embeddings, labels (per source shard)
    ├── focal_perch/shard-NN.npz   # OPTIONAL mapped+dense Perch sidecar
    ├── unsup/shard-NN.npz         # embeddings, site_indices, recording_indices
    └── unsup_perch/shard-NN.npz   # OPTIONAL mapped+dense Perch sidecar
    idx-v1/
    ├── focal-n{N}-seed{S}.npy
    └── unsup-n{N}-seed{S}.npy

Core entry point is :func:`load_training_data_cached`. All other names are
internal but unprefixed for easier testing.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from birdclef_106.data_loading import (
    build_binary_label_matrix,
    embeddings_to_numpy,
    extract_site,
    list_sites,
    load_labeled,
    logits_to_numpy,
)

from birdclef_107.data import (
    FEATURES_DIR,
    FocalData,
    LabeledSoundscapeData,
    UnlabeledSoundscapeData,
    get_species_list,
    stratified_unsup_subset,
    weighted_focal_subset,
)
from birdclef_107.perch_features import project_perch_sidecar

SCHEMA_VERSION = 2
BASE_DIRNAME = "base-v1"
IDX_DIRNAME = "idx-v1"


# ---------------------------------------------------------------------------
# Paths & locking
# ---------------------------------------------------------------------------


def _paths(cache_root: Path) -> dict[str, Path]:
    base = cache_root / BASE_DIRNAME
    return {
        "base": base,
        "manifest": base / "base_manifest.json",
        "labeled": base / "labeled.npz",
        "labeled_perch": base / "labeled_perch.npz",
        "focal": base / "focal",
        "focal_perch": base / "focal_perch",
        "unsup": base / "unsup",
        "unsup_perch": base / "unsup_perch",
        "idx": cache_root / IDX_DIRNAME,
        "lock": cache_root / ".manifest.lock",
    }


@contextlib.contextmanager
def _manifest_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "w")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    # np.savez appends ".npz" if missing, so the temp must already end in .npz
    # or the final rename target would be wrong.
    assert path.suffix == ".npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.partial-{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    assert path.suffix == ".npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}.partial-{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.partial-{os.getpid()}"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Source parquet enumeration
# ---------------------------------------------------------------------------


def _source_focal_shards(features_dir: str) -> list[Path]:
    d = Path(features_dir).expanduser() / "focal"
    return sorted(d.glob("shard-*.parquet"))


def _source_unsup_shards(features_dir: str) -> list[Path]:
    d = Path(features_dir).expanduser() / "soundscape_unlabeled"
    return sorted(d.glob("shard-*.parquet"))


def _source_labeled_path(features_dir: str) -> Path:
    return Path(features_dir).expanduser() / "soundscape_labeled.parquet"


def _mtime_ns(p: Path) -> int:
    return p.stat().st_mtime_ns


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def _empty_manifest(features_dir: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "features_dir": str(Path(features_dir).expanduser()),
        "species_list": [],
        "site_to_idx": {},
        "recording_to_idx": {},
        "labeled": {"parquet_mtime_ns": 0, "n_rows": 0},
        "labeled_has_perch": False,
        "focal_shards": [],
        "unsup_shards": [],
        "built_at": None,
    }


def _load_manifest(paths: dict[str, Path], features_dir: str) -> dict:
    if not paths["manifest"].exists():
        return _empty_manifest(features_dir)
    try:
        m = json.loads(paths["manifest"].read_text())
    except json.JSONDecodeError:
        return _empty_manifest(features_dir)
    if m.get("schema_version") != SCHEMA_VERSION:
        return _empty_manifest(features_dir)
    if m.get("features_dir") != str(Path(features_dir).expanduser()):
        return _empty_manifest(features_dir)
    return m


# ---------------------------------------------------------------------------
# Cold / incremental build
# ---------------------------------------------------------------------------


def _build_labeled(
    features_dir: str,
    paths: dict[str, Path],
    manifest: dict,
    *,
    include_perch: bool,
) -> dict:
    labeled_path = _source_labeled_path(features_dir)
    labeled_mtime = _mtime_ns(labeled_path)

    core_stale = (
        not paths["labeled"].exists()
        or manifest["labeled"]["parquet_mtime_ns"] != labeled_mtime
    )
    perch_stale = include_perch and (
        not paths["labeled_perch"].exists()
        or not manifest.get("labeled_has_perch", False)
        or manifest["labeled"]["parquet_mtime_ns"] != labeled_mtime
    )

    if not core_stale and not perch_stale:
        return manifest

    df = load_labeled(features_dir)
    species_list = manifest["species_list"] or get_species_list()
    site_to_idx = manifest["site_to_idx"]

    embeddings = embeddings_to_numpy(df)
    labels = build_binary_label_matrix(df, species_list)
    filenames = df["filename"].to_list()
    sites = np.array([extract_site(f) for f in filenames])

    unlabeled_sites = set(list_sites(features_dir))
    labeled_sites = set(sites.tolist())
    all_sites = sorted(unlabeled_sites | labeled_sites)
    site_to_idx = {s: i for i, s in enumerate(all_sites)}

    if core_stale:
        _atomic_write_npz(
            paths["labeled"],
            embeddings=embeddings,
            labels=labels,
            sites=sites.astype("U"),
            filenames=np.asarray(filenames, dtype="U"),
        )

    if perch_stale:
        perch = project_perch_sidecar(logits_to_numpy(df), species_list)
        _atomic_write_npz(paths["labeled_perch"], **perch)

    manifest["species_list"] = species_list
    manifest["site_to_idx"] = site_to_idx
    manifest["labeled"] = {
        "parquet_mtime_ns": labeled_mtime,
        "n_rows": int(len(embeddings)),
    }
    if include_perch:
        manifest["labeled_has_perch"] = True
    elif not paths["labeled_perch"].exists():
        manifest["labeled_has_perch"] = False
    return manifest


def _shard_entry(entries: list[dict], name: str) -> dict | None:
    for e in entries:
        if e["name"] == name:
            return e
    return None


def _build_focal_shards(
    features_dir: str,
    paths: dict[str, Path],
    manifest: dict,
    *,
    include_perch: bool,
) -> dict:
    source_shards = _source_focal_shards(features_dir)
    if not source_shards:
        raise FileNotFoundError(
            f"No focal shards found under {features_dir}/focal/"
        )

    species_list = manifest["species_list"] or get_species_list()
    entries: list[dict] = []
    cum_offset = 0

    for src in source_shards:
        name = src.stem  # "shard-NN"
        dest_core = paths["focal"] / f"{name}.npz"
        dest_perch = paths["focal_perch"] / f"{name}.npz"
        src_mtime = _mtime_ns(src)
        prev = _shard_entry(manifest.get("focal_shards", []), name)

        core_stale = (
            not dest_core.exists()
            or prev is None
            or prev["parquet_mtime_ns"] != src_mtime
        )
        perch_stale = include_perch and (
            not dest_perch.exists()
            or prev is None
            or not prev.get("has_perch", False)
            or prev["parquet_mtime_ns"] != src_mtime
        )

        if core_stale or perch_stale:
            df = pl.read_parquet(src)
            if core_stale:
                emb = embeddings_to_numpy(df)
                lab = build_binary_label_matrix(df, species_list)
                _atomic_write_npz(dest_core, embeddings=emb, labels=lab)
                n_rows = int(len(emb))
            else:
                n_rows = int(prev["n_rows"])
            if perch_stale:
                perch = project_perch_sidecar(logits_to_numpy(df), species_list)
                _atomic_write_npz(dest_perch, **perch)
        else:
            n_rows = int(prev["n_rows"])

        has_perch = (
            dest_perch.exists()
            if include_perch or (prev and prev.get("has_perch", False))
            else False
        )
        entries.append({
            "name": name,
            "parquet_mtime_ns": src_mtime,
            "n_rows": n_rows,
            "cum_offset": cum_offset,
            "has_perch": bool(has_perch),
        })
        cum_offset += n_rows

    manifest["species_list"] = species_list
    manifest["focal_shards"] = entries
    return manifest


def _build_unsup_shards(
    features_dir: str,
    paths: dict[str, Path],
    manifest: dict,
    *,
    include_perch: bool,
) -> dict:
    source_shards = _source_unsup_shards(features_dir)
    if not source_shards:
        raise FileNotFoundError(
            f"No unsup shards found under {features_dir}/soundscape_unlabeled/"
        )

    # site_to_idx must come from the labeled-SC union (labeled already built);
    # fall back to just unlabeled sites if labeled pass hasn't run yet.
    site_to_idx = manifest["site_to_idx"]
    if not site_to_idx:
        all_sites = sorted(list_sites(features_dir))
        site_to_idx = {s: i for i, s in enumerate(all_sites)}
        manifest["site_to_idx"] = site_to_idx

    # recording_to_idx is global across shards. Seed with any existing mapping
    # so incremental rebuilds keep recording ids stable for untouched shards.
    recording_to_idx: dict[str, int] = dict(manifest.get("recording_to_idx", {}))

    entries: list[dict] = []
    cum_offset = 0

    for src in source_shards:
        name = src.stem
        dest_core = paths["unsup"] / f"{name}.npz"
        dest_perch = paths["unsup_perch"] / f"{name}.npz"
        src_mtime = _mtime_ns(src)
        prev = _shard_entry(manifest.get("unsup_shards", []), name)

        core_stale = (
            not dest_core.exists()
            or prev is None
            or prev["parquet_mtime_ns"] != src_mtime
        )
        perch_stale = include_perch and (
            not dest_perch.exists()
            or prev is None
            or not prev.get("has_perch", False)
            or prev["parquet_mtime_ns"] != src_mtime
        )

        if core_stale or perch_stale:
            df = pl.read_parquet(src)
            fnames = df["filename"].to_list()

            # Extend global recording_to_idx with any new names.
            for r in fnames:
                if r not in recording_to_idx:
                    recording_to_idx[r] = len(recording_to_idx)

            if core_stale:
                emb = embeddings_to_numpy(df)
                sites = [extract_site(f) for f in fnames]
                site_idx = np.asarray(
                    [site_to_idx.get(s, 0) for s in sites], dtype=np.int64
                )
                rec_idx = np.asarray(
                    [recording_to_idx[r] for r in fnames], dtype=np.int64
                )
                _atomic_write_npz(
                    dest_core,
                    embeddings=emb,
                    site_indices=site_idx,
                    recording_indices=rec_idx,
                )
                n_rows = int(len(emb))
            else:
                n_rows = int(prev["n_rows"])

            if perch_stale:
                perch = project_perch_sidecar(logits_to_numpy(df), manifest["species_list"])
                _atomic_write_npz(dest_perch, **perch)
        else:
            n_rows = int(prev["n_rows"])

        has_perch = (
            dest_perch.exists()
            if include_perch or (prev and prev.get("has_perch", False))
            else False
        )
        entries.append({
            "name": name,
            "parquet_mtime_ns": src_mtime,
            "n_rows": n_rows,
            "cum_offset": cum_offset,
            "has_perch": bool(has_perch),
        })
        cum_offset += n_rows

    manifest["recording_to_idx"] = recording_to_idx
    manifest["unsup_shards"] = entries
    return manifest


def _build_or_update_base(
    features_dir: str,
    paths: dict[str, Path],
    *,
    include_perch: bool,
    force: bool,
) -> dict:
    paths["base"].mkdir(parents=True, exist_ok=True)

    with _manifest_lock(paths["lock"]):
        manifest = _load_manifest(paths, features_dir)

        if force:
            manifest = _empty_manifest(features_dir)

        # Order matters: labeled populates species_list + site_to_idx, which
        # unsup shards reuse. Focal doesn't depend on site_to_idx.
        manifest = _build_labeled(
            features_dir, paths, manifest, include_perch=include_perch,
        )
        manifest = _build_focal_shards(
            features_dir, paths, manifest, include_perch=include_perch,
        )
        manifest = _build_unsup_shards(
            features_dir, paths, manifest, include_perch=include_perch,
        )
        manifest["built_at"] = int(time.time())
        _atomic_write_json(paths["manifest"], manifest)

    return manifest


# ---------------------------------------------------------------------------
# Index files
# ---------------------------------------------------------------------------


def _idx_path(paths: dict[str, Path], kind: str, n: int, seed: int) -> Path:
    return paths["idx"] / f"{kind}-n{n}-seed{seed}.npy"


def _get_or_build_idx(
    path: Path,
    compute: Callable[[], np.ndarray],
    *,
    force: bool,
) -> np.ndarray:
    if path.exists() and not force:
        return np.load(path)
    idx = compute()
    _atomic_write_npy(path, idx)
    return idx


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _cum_offsets(entries: list[dict]) -> np.ndarray:
    return np.asarray([e["cum_offset"] for e in entries], dtype=np.int64)


def _group_by_shard(
    global_idx: np.ndarray, entries: list[dict]
) -> list[tuple[int, np.ndarray]]:
    """Partition sorted global indices into (shard_id, local_idx) groups."""
    offsets = _cum_offsets(entries)
    # np.searchsorted with side='right' places each global idx in the right shard.
    shard_ids = np.searchsorted(offsets, global_idx, side="right") - 1
    groups: list[tuple[int, np.ndarray]] = []
    # Stable split: indices are sorted, shard_ids non-decreasing.
    uniq, start = np.unique(shard_ids, return_index=True)
    for i, sid in enumerate(uniq):
        end = start[i + 1] if i + 1 < len(start) else len(global_idx)
        local = global_idx[start[i]:end] - offsets[sid]
        groups.append((int(sid), local.astype(np.int64)))
    return groups


def _materialize_from_shards(
    shard_dir: Path,
    entries: list[dict],
    global_idx: np.ndarray | None,
    field_names: list[str],
) -> dict[str, np.ndarray]:
    """Load specified fields for the given global indices from sharded NPZs.

    If ``global_idx`` is None, returns the full concatenation across shards.
    """
    if global_idx is None:
        parts: dict[str, list[np.ndarray]] = {f: [] for f in field_names}
        for e in entries:
            with np.load(shard_dir / f"{e['name']}.npz", allow_pickle=False) as z:
                for f in field_names:
                    parts[f].append(np.asarray(z[f]))
        return {f: np.concatenate(parts[f], axis=0) for f in field_names}

    groups = _group_by_shard(global_idx, entries)
    parts: dict[str, list[np.ndarray]] = {f: [] for f in field_names}
    for sid, local in groups:
        name = entries[sid]["name"]
        with np.load(shard_dir / f"{name}.npz", allow_pickle=False) as z:
            for f in field_names:
                parts[f].append(np.asarray(z[f][local]))
    return {f: np.concatenate(parts[f], axis=0) for f in field_names}


def _load_labeled(
    paths: dict[str, Path],
    manifest: dict,
    *,
    include_perch: bool,
) -> LabeledSoundscapeData:
    with np.load(paths["labeled"], allow_pickle=False) as z:
        embeddings = np.asarray(z["embeddings"])
        labels = np.asarray(z["labels"])
        sites = np.asarray(z["sites"])
        filenames = [str(s) for s in z["filenames"].tolist()]
    perch_mapped = perch_topk_idx = perch_topk_val = None
    if include_perch:
        with np.load(paths["labeled_perch"], allow_pickle=False) as z:
            perch_mapped = np.asarray(z["mapped_probs"])
            perch_topk_idx = np.asarray(z["topk_idx"])
            perch_topk_val = np.asarray(z["topk_val"])
    return LabeledSoundscapeData(
        embeddings=embeddings,
        labels=labels,
        sites=sites,
        filenames=filenames,
        species_list=list(manifest["species_list"]),
        site_to_idx={str(k): int(v) for k, v in manifest["site_to_idx"].items()},
        perch_mapped=perch_mapped,
        perch_topk_idx=perch_topk_idx,
        perch_topk_val=perch_topk_val,
    )


def _materialize_focal(
    paths: dict[str, Path],
    manifest: dict,
    idx: np.ndarray | None,
    *,
    include_perch: bool,
) -> FocalData:
    entries = manifest["focal_shards"]
    fields = _materialize_from_shards(
        paths["focal"], entries, idx, ["embeddings", "labels"]
    )
    perch_mapped = perch_topk_idx = perch_topk_val = None
    if include_perch:
        perch_fields = _materialize_from_shards(
            paths["focal_perch"], entries, idx, ["mapped_probs", "topk_idx", "topk_val"]
        )
        perch_mapped = perch_fields["mapped_probs"]
        perch_topk_idx = perch_fields["topk_idx"]
        perch_topk_val = perch_fields["topk_val"]
    return FocalData(
        embeddings=fields["embeddings"],
        labels=fields["labels"],
        species_list=list(manifest["species_list"]),
        perch_mapped=perch_mapped,
        perch_topk_idx=perch_topk_idx,
        perch_topk_val=perch_topk_val,
    )


def _materialize_unsup(
    paths: dict[str, Path],
    manifest: dict,
    idx: np.ndarray | None,
    *,
    include_perch: bool,
) -> UnlabeledSoundscapeData:
    entries = manifest["unsup_shards"]
    fields = _materialize_from_shards(
        paths["unsup"],
        entries,
        idx,
        ["embeddings", "site_indices", "recording_indices"],
    )
    perch_mapped = perch_topk_idx = perch_topk_val = None
    if include_perch:
        perch_fields = _materialize_from_shards(
            paths["unsup_perch"], entries, idx, ["mapped_probs", "topk_idx", "topk_val"]
        )
        perch_mapped = perch_fields["mapped_probs"]
        perch_topk_idx = perch_fields["topk_idx"]
        perch_topk_val = perch_fields["topk_val"]
    return UnlabeledSoundscapeData(
        embeddings=fields["embeddings"],
        site_indices=fields["site_indices"],
        recording_indices=fields["recording_indices"],
        site_to_idx={str(k): int(v) for k, v in manifest["site_to_idx"].items()},
        recording_to_idx={
            str(k): int(v) for k, v in manifest["recording_to_idx"].items()
        },
        perch_mapped=perch_mapped,
        perch_topk_idx=perch_topk_idx,
        perch_topk_val=perch_topk_val,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_training_data_cached(
    features_dir: str | Path = FEATURES_DIR,
    cache_root: str | Path = "~/scratch/birdclef-2026/cache/exp107",
    focal_size: int = 0,
    unsup_size: int = 0,
    seed: int = 42,
    *,
    include_perch_logits: bool = False,
    force_rebuild_base: bool = False,
    force_rebuild_index: bool = False,
    focal_subset_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
    unsup_subset_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
) -> tuple[LabeledSoundscapeData, FocalData, UnlabeledSoundscapeData]:
    """Load labeled + focal + unsup training data via the persistent cache.

    Args:
        features_dir: Source parquet root (e.g. the exp106 cache).
        cache_root: Persistent cache root on scratch. Contains ``base-v1/``
            and ``idx-v1/`` after the first build.
        focal_size: If >0, fancy-index a focal subset of this many rows; 0
            uses all focal rows.
        unsup_size: Same for unlabeled soundscapes.
        seed: Seed for the subsample sampler (separate from training seed).
        include_perch_logits: If True, hydrate the optional compact Perch
            sidecar fields (dense mapped 234-d probs + sparse top-k teacher
            activations) on each dataclass. Triggers sidecar build on first use.
        force_rebuild_base: Ignore existing base shards and rebuild from
            parquet. Useful if a source parquet changed without its mtime
            advancing, or to recover from corruption.
        force_rebuild_index: Recompute the index ``.npy`` files even if
            they exist. Leaves the base shards alone.
        focal_subset_fn, unsup_subset_fn: Injection points for the
            subsample policy (defaults to the ones defined in
            ``scripts/train.py``). Passed as arguments so ``cache.py``
            doesn't import a script module.

    Returns:
        ``(LabeledSoundscapeData, FocalData, UnlabeledSoundscapeData)`` —
        the public dataclasses the rest of exp107 consumes.
    """
    features_dir = str(Path(features_dir).expanduser())
    cache_root = Path(cache_root).expanduser()
    paths = _paths(cache_root)

    manifest = _build_or_update_base(
        features_dir,
        paths,
        include_perch=include_perch_logits,
        force=force_rebuild_base,
    )

    labeled = _load_labeled(paths, manifest, include_perch=include_perch_logits)

    focal_idx: np.ndarray | None = None
    if focal_size > 0:
        focal_idx = _get_or_build_focal_idx(
            paths, manifest, focal_size, seed,
            force=force_rebuild_index,
            subset_fn=focal_subset_fn,
        )
    unsup_idx: np.ndarray | None = None
    if unsup_size > 0:
        unsup_idx = _get_or_build_unsup_idx(
            paths, manifest, unsup_size, seed,
            force=force_rebuild_index,
            subset_fn=unsup_subset_fn,
        )

    focal = _materialize_focal(
        paths, manifest, focal_idx, include_perch=include_perch_logits
    )
    unsup = _materialize_unsup(
        paths, manifest, unsup_idx, include_perch=include_perch_logits
    )
    return labeled, focal, unsup


def _get_or_build_focal_idx(
    paths: dict[str, Path],
    manifest: dict,
    n: int,
    seed: int,
    *,
    force: bool,
    subset_fn: Callable[[np.ndarray, int, int], np.ndarray] | None,
) -> np.ndarray:
    if subset_fn is None:
        subset_fn = weighted_focal_subset

    path = _idx_path(paths, "focal", n, seed)

    def compute() -> np.ndarray:
        entries = manifest["focal_shards"]
        labels = _materialize_from_shards(
            paths["focal"], entries, None, ["labels"]
        )["labels"]
        return subset_fn(labels, n, seed)

    return _get_or_build_idx(path, compute, force=force)


def _get_or_build_unsup_idx(
    paths: dict[str, Path],
    manifest: dict,
    n: int,
    seed: int,
    *,
    force: bool,
    subset_fn: Callable[[np.ndarray, int, int], np.ndarray] | None,
) -> np.ndarray:
    if subset_fn is None:
        subset_fn = stratified_unsup_subset

    path = _idx_path(paths, "unsup", n, seed)

    def compute() -> np.ndarray:
        entries = manifest["unsup_shards"]
        site_indices = _materialize_from_shards(
            paths["unsup"], entries, None, ["site_indices"]
        )["site_indices"]
        return subset_fn(site_indices, n, seed)

    return _get_or_build_idx(path, compute, force=force)


# Re-export dataclass field names for test introspection.
__all__ = [
    "load_training_data_cached",
    "SCHEMA_VERSION",
    "BASE_DIRNAME",
    "IDX_DIRNAME",
]


def _sanity_check_dataclasses() -> None:
    """Assert the dataclasses still carry the expected fields (tests rely on this)."""
    expected = {
        LabeledSoundscapeData: {
            "embeddings", "labels", "sites", "filenames", "species_list",
            "site_to_idx", "perch_mapped", "perch_topk_idx", "perch_topk_val",
        },
        FocalData: {"embeddings", "labels", "species_list", "perch_mapped", "perch_topk_idx", "perch_topk_val"},
        UnlabeledSoundscapeData: {
            "embeddings", "site_indices", "recording_indices",
            "site_to_idx", "recording_to_idx", "perch_mapped", "perch_topk_idx", "perch_topk_val",
        },
    }
    for cls, fset in expected.items():
        actual = {f.name for f in dataclass_fields(cls)}
        missing = fset - actual
        assert not missing, f"{cls.__name__} missing fields: {missing}"


_sanity_check_dataclasses()
