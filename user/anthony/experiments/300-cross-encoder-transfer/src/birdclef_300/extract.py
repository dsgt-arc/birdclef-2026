"""Encoder-agnostic extraction of the labeled-soundscape windows to the unified cache.

Three backends, one output schema (see data_loading.SCHEMA_COLUMNS):

  * torch       — drive a shared birdclef.encoders wrapper over the 749 windows.
  * perch_cache — re-key exp 106's Perch embeddings into the unified schema.
  * birdnet_bmz — delegate to extract_birdnet (separate TF venv).

All three iterate the SAME labels CSV in the SAME order, so the per-(filename,
start_sec) rows align across encoders. The ridge probe builds its own labels from
each parquet, so per-encoder self-consistency is all that matters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from birdclef_300 import audio as audio_mod
from birdclef_300.data_loading import labeled_parquet_path, patches_parquet_path

logger = logging.getLogger("birdclef_300.extract")


def _time_to_sec(t: str) -> int:
    parts = str(t).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def labeled_window_rows(cfg: dict, limit: int | None = None) -> list[dict]:
    """Build the ordered list of labeled-window metadata from the labels CSV.

    Returns dicts with: filename, start_sec, window_idx, split, primary_label,
    ground_truth. Shared by the torch and birdnet extraction paths.
    """
    data = cfg["data"]
    labels_df = pl.read_csv(
        data["soundscape_labels_csv"], infer_schema_length=10000
    ).unique(subset=["filename", "start"]).sort(["filename", "start"])

    # Reuse the exp 102/105/106 seed=42 half split for train/val tagging.
    all_files = sorted(labels_df["filename"].unique().to_list())
    rng = np.random.RandomState(data["split_seed"])
    rng.shuffle(all_files)
    train_files = set(all_files[: len(all_files) // 2])

    rows: list[dict] = []
    for row in labels_df.iter_rows(named=True):
        start_sec = _time_to_sec(row["start"])
        gt = str(row["primary_label"]).strip()
        rows.append(dict(
            filename=row["filename"],
            start_sec=float(start_sec),
            window_idx=int(start_sec // 5),
            split="train" if row["filename"] in train_files else "val",
            primary_label=gt.split(";")[0] if gt else "",
            ground_truth=gt,
        ))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _write_unified_parquet(
    output_path: str | Path,
    meta_rows: list[dict],
    emb_list: list[np.ndarray],
    encoder: str,
    variant: str,
    source: str = "soundscape-labeled",
) -> None:
    """Write metadata + embedding column to a unified-schema parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not meta_rows:
        logger.warning("No rows to write for %s/%s", encoder, variant)
        return

    emb_matrix = np.stack(emb_list).astype(np.float32)
    embed_dim = int(emb_matrix.shape[1])

    meta_df = pl.DataFrame([
        {**m, "source": source, "encoder": encoder,
         "variant": variant, "embed_dim": embed_dim}
        for m in meta_rows
    ])
    vec_df = pl.DataFrame({"embedding": emb_matrix.tolist()}).cast(
        {"embedding": pl.List(pl.Float32)}
    )
    df = pl.concat([meta_df, vec_df], how="horizontal")
    df.write_parquet(output_path)
    logger.info(
        "Written %s (%d rows, dim=%d, %.1f MB)",
        output_path, len(meta_rows), embed_dim, output_path.stat().st_size / 1e6,
    )


def _write_patch_parquet(
    output_path: str | Path,
    meta_rows: list[dict],
    patch_list: list[np.ndarray],
    encoder: str,
    variant: str,
) -> None:
    """Write metadata + per-window patch grids (List(List(Float32))) to a parquet.

    Each row's `patch_embedding` is (num_patches, embed_dim). Used by the
    prototypical probe's frame-level (Strategy B) max-similarity readout.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not meta_rows:
        logger.warning("No patch rows to write for %s/%s", encoder, variant)
        return
    embed_dim = int(patch_list[0].shape[1])
    n_patches = int(patch_list[0].shape[0])
    meta_df = pl.DataFrame([
        {**m, "source": "soundscape-labeled", "encoder": encoder,
         "variant": variant, "embed_dim": embed_dim}
        for m in meta_rows
    ])
    vec_df = pl.DataFrame(
        {"patch_embedding": [p.astype(np.float32).tolist() for p in patch_list]}
    ).cast({"patch_embedding": pl.List(pl.List(pl.Float32))})
    df = pl.concat([meta_df, vec_df], how="horizontal")
    df.write_parquet(output_path)
    logger.info("Written %s (%d rows, %d patches x %d dim, %.1f MB)",
                output_path, len(meta_rows), n_patches, embed_dim,
                output_path.stat().st_size / 1e6)


def _apply_overlay(model, adapter: str | None, state_dict: str | None, device: str) -> None:
    """Apply a 301 adaptation overlay onto the encoder's inner `.model` in place.

    All four torch wrappers expose `.model`, so this is encoder-agnostic:
      adapter    — a PEFT LoRA adapter dir; merged into the base for inference.
      state_dict — a full DAPT-updated state_dict (continued-SSL weights).
    """
    import torch

    if adapter:
        from peft import PeftModel

        logger.info("Applying LoRA adapter: %s", adapter)
        model.model = PeftModel.from_pretrained(model.model, adapter).merge_and_unload()
    elif state_dict:
        logger.info("Loading DAPT state_dict: %s", state_dict)
        model.model.load_state_dict(torch.load(state_dict, map_location=device))
    model.model.eval()


def extract_torch_labeled(
    cfg: dict,
    encoder: str,
    variant: str = "frozen",
    checkpoint: str | None = None,
    adapter: str | None = None,
    state_dict: str | None = None,
    output_path: str | Path | None = None,
    limit: int | None = None,
    patch: bool = False,
) -> int:
    """Run a torch-backend encoder over the labeled windows -> unified parquet.

    `patch=True` caches the per-window patch grid (`patches.parquet`) instead of the
    pooled vector, for the prototypical probe's frame-level readout. The pooled
    `data.parquet` is left untouched (run the two extractions separately) so the
    frozen ranking exp 300 reports is never perturbed.
    """
    import torch

    from birdclef_300.registry import build_torch_encoder

    enc_cfg = cfg["encoders"][encoder]
    device = enc_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU for %s", encoder)
        device = "cpu"
    batch_size = int(enc_cfg.get("batch_size", 16))
    sdir = Path(cfg["data"]["soundscape_dir"])
    if output_path is None:
        path_fn = patches_parquet_path if patch else labeled_parquet_path
        output_path = path_fn(cfg["output"]["base_dir"], encoder, variant)

    logger.info("Building encoder %s (variant=%s, device=%s, patch=%s)",
                encoder, variant, device, patch)
    model = build_torch_encoder(encoder, enc_cfg, device, checkpoint)
    _apply_overlay(model, adapter, state_dict, device)
    if patch and not hasattr(model, "extract_patch_features"):
        raise ValueError(f"encoder '{encoder}' has no extract_patch_features (patch mode)")

    rows = labeled_window_rows(cfg, limit=limit)
    logger.info("Extracting %d labeled windows with %s", len(rows), encoder)

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []
    n_skipped = 0
    t0 = time.time()

    def flush() -> None:
        if not buf_wavs:
            return
        batch = np.stack(buf_wavs).astype(np.float32)            # (B, N)
        wav = torch.from_numpy(batch).unsqueeze(1)               # (B, 1, N)
        if patch:
            feats = model.extract_patch_features(wav).detach().cpu().numpy()  # (B, P, D)
        else:
            feats = model.extract_features(wav).detach().cpu().numpy()        # (B, D)
        for k, meta in enumerate(buf_meta):
            meta_rows.append(meta)
            emb_list.append(feats[k].astype(np.float32))
        buf_wavs.clear()
        buf_meta.clear()

    # Rows are sorted by (filename, start), so decode each file once at 16 kHz
    # and slice the in-memory waveform per window rather than re-opening and
    # resampling a 5 s clip per row.
    cur_file: str | None = None
    cur_wav: np.ndarray | None = None
    for row in rows:
        path = sdir / row["filename"]
        if row["filename"] != cur_file:
            cur_file = row["filename"]
            try:
                cur_wav = audio_mod.load_audio_16k(str(path)) if path.exists() else None
            except Exception as e:  # noqa: BLE001 — log and skip unreadable files
                logger.warning("Load failed %s: %s", row["filename"], e)
                cur_wav = None
        if cur_wav is None:
            n_skipped += 1
            continue
        start = int(row["start_sec"] * audio_mod.ENCODER_SR)
        buf_wavs.append(audio_mod.fit_window(cur_wav[start:start + audio_mod.ENCODER_WINDOW_SAMPLES]))
        buf_meta.append({k: row[k] for k in
                         ("filename", "window_idx", "start_sec", "split", "primary_label", "ground_truth")})
        if len(buf_wavs) >= batch_size:
            flush()
    flush()

    logger.info("%s: %d windows in %.1fs (skipped %d)",
                encoder, len(meta_rows), time.time() - t0, n_skipped)
    if patch:
        _write_patch_parquet(output_path, meta_rows, emb_list, encoder, variant)
    else:
        _write_unified_parquet(output_path, meta_rows, emb_list, encoder, variant)
    return 0


def extract_torch_focal(
    cfg: dict,
    encoder: str,
    variant: str = "frozen",
    checkpoint: str | None = None,
    output_path: str | Path | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
) -> int:
    """Run a torch-backend encoder over the FOCAL recordings -> unified parquet shard.

    The deploy head (`deploy.fit_head`) is fit on a focal+soundscape cache laid out as
    `<dir>/focal/shard-*.parquet` + `<dir>/soundscape_labeled.parquet`. exp 300 only
    extracted the 739 soundscape-labeled windows per encoder; this fills the focal side
    so any roster encoder can be deployed like Perch. Mirrors exp 106's `extract_focal`
    (shard by row index, per-file decode, non-overlapping 5 s windows) but at the 16 kHz
    geometry the torch encoders expect.
    """
    import torch

    from birdclef_300.registry import build_torch_encoder

    enc_cfg = cfg["encoders"][encoder]
    device = enc_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU for %s", encoder)
        device = "cpu"
    batch_size = int(enc_cfg.get("batch_size", 16))
    audio_dir = Path(cfg["focal"]["audio_dir"])
    if output_path is None:
        raise ValueError("extract_torch_focal requires an explicit output_path (shard parquet)")

    logger.info("Building encoder %s (variant=%s, device=%s) for focal extraction",
                encoder, variant, device)
    model = build_torch_encoder(encoder, enc_cfg, device, checkpoint)
    model.model.eval()

    train_df = pl.read_csv(
        cfg["data"]["train_csv"], infer_schema_length=10000
    ).sort("filename")
    train_df = train_df.with_row_index("_idx").filter(
        pl.col("_idx") % num_shards == shard_index
    ).drop("_idx")
    if limit is not None:
        train_df = train_df.head(limit)
    logger.info("Focal shard %d/%d: %d recordings", shard_index, num_shards, len(train_df))

    meta_rows: list[dict] = []
    emb_list: list[np.ndarray] = []
    buf_wavs: list[np.ndarray] = []
    buf_meta: list[dict] = []
    n_failed = 0
    t0 = time.time()

    def flush() -> None:
        if not buf_wavs:
            return
        batch = np.stack(buf_wavs).astype(np.float32)            # (B, N)
        wav = torch.from_numpy(batch).unsqueeze(1)               # (B, 1, N)
        feats = model.extract_features(wav).detach().cpu().numpy()  # (B, D)
        for k, meta in enumerate(buf_meta):
            meta_rows.append(meta)
            emb_list.append(feats[k].astype(np.float32))
        buf_wavs.clear()
        buf_meta.clear()

    for fi, rec in enumerate(train_df.iter_rows(named=True)):
        if fi % 1000 == 0 and fi > 0:
            logger.info("Progress: %d/%d files, %d windows", fi, len(train_df), len(meta_rows))
        path = audio_dir / rec["filename"]
        try:
            wav = audio_mod.load_audio_16k(str(path))
        except Exception as e:  # noqa: BLE001 — log and skip unreadable files
            logger.warning("Load failed %s: %s", rec["filename"], e)
            n_failed += 1
            continue

        # ground_truth mirrors exp 106's focal cache exactly: `primary;<secondary_str>`.
        # train.csv `secondary_labels` is a list-literal STRING (e.g. "['trokin']"),
        # which build_binary_label_matrix (split on ';') can't match — so the deployed
        # Perch head trained on PRIMARY labels in effect. We reproduce that regime
        # verbatim so the AST head is an apples-to-apples mirror, not a richer-label fork.
        primary = str(rec["primary_label"])
        secondary = str(rec.get("secondary_labels", ""))
        gt = primary
        if secondary and secondary not in ("", "None", "nan", "null", "[]"):
            gt = f"{primary};{secondary}"

        for wi, win in enumerate(audio_mod.window_audio(wav)):
            buf_wavs.append(win)
            buf_meta.append(dict(
                filename=rec["filename"], window_idx=wi, start_sec=float(wi * 5.0),
                split="", primary_label=primary, ground_truth=gt,
            ))
            if len(buf_wavs) >= batch_size:
                flush()
    flush()

    logger.info("focal shard %d (%s): %d windows from %d recordings in %.1fs (%d failed)",
                shard_index, encoder, len(meta_rows), len(train_df), time.time() - t0, n_failed)
    _write_unified_parquet(output_path, meta_rows, emb_list, encoder, variant, source="focal")
    return 0


def ingest_perch_labeled(
    cfg: dict,
    variant: str = "frozen",
    output_path: str | Path | None = None,
    limit: int | None = None,
) -> int:
    """Re-key exp 106's Perch embeddings into the unified schema (no recompute)."""
    enc_cfg = cfg["encoders"]["perch"]
    src = Path(enc_cfg["source_parquet"]).expanduser()
    output_path = output_path or labeled_parquet_path(cfg["output"]["base_dir"], "perch", variant)
    logger.info("Ingesting Perch embeddings from %s", src)

    df = pl.read_parquet(src)
    if limit is not None:
        df = df.head(limit)
    rows = df.select(
        ["filename", "window_idx", "start_sec", "split", "primary_label", "ground_truth"]
    ).to_dicts()
    emb_list = [np.asarray(e, dtype=np.float32) for e in df["embedding"].to_list()]
    _write_unified_parquet(output_path, rows, emb_list, "perch", variant)
    return 0


def extract_labeled(
    cfg: dict,
    encoder: str,
    variant: str = "frozen",
    checkpoint: str | None = None,
    adapter: str | None = None,
    state_dict: str | None = None,
    output_path: str | Path | None = None,
    limit: int | None = None,
    patch: bool = False,
) -> int:
    """Dispatch extraction by the encoder's backend (see registry)."""
    from birdclef_300.registry import BIRDNET_BMZ, PERCH_CACHE, TORCH, get_spec

    backend = get_spec(encoder).backend
    if backend == TORCH:
        return extract_torch_labeled(
            cfg, encoder, variant, checkpoint, adapter, state_dict, output_path, limit, patch
        )
    if patch:
        raise ValueError(f"patch mode is torch-only; '{encoder}' is backend '{backend}'")
    if backend == PERCH_CACHE:
        return ingest_perch_labeled(cfg, variant, output_path, limit)
    if backend == BIRDNET_BMZ:
        from birdclef_300.extract_birdnet import extract_birdnet_labeled

        return extract_birdnet_labeled(cfg, variant, output_path, limit)
    raise ValueError(f"Unknown backend '{backend}' for encoder '{encoder}'")
