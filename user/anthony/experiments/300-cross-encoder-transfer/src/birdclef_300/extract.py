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
from birdclef_300.data_loading import labeled_parquet_path

logger = logging.getLogger("birdclef_300.extract")

WINDOW_SAMPLES = audio_mod.ENCODER_WINDOW_SAMPLES  # 80000


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
        {**m, "source": "soundscape-labeled", "encoder": encoder,
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
) -> int:
    """Run a torch-backend encoder over the labeled windows -> unified parquet."""
    import torch

    from birdclef_300.registry import build_torch_encoder

    enc_cfg = cfg["encoders"][encoder]
    device = enc_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to CPU for %s", encoder)
        device = "cpu"
    batch_size = int(enc_cfg.get("batch_size", 16))
    sdir = Path(cfg["data"]["soundscape_dir"])
    output_path = output_path or labeled_parquet_path(cfg["output"]["base_dir"], encoder, variant)

    logger.info("Building encoder %s (variant=%s, device=%s)", encoder, variant, device)
    model = build_torch_encoder(encoder, enc_cfg, device, checkpoint)
    _apply_overlay(model, adapter, state_dict, device)

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
        feats = model.extract_features(wav).detach().cpu().numpy()
        for k, meta in enumerate(buf_meta):
            meta_rows.append(meta)
            emb_list.append(feats[k].astype(np.float32))
        buf_wavs.clear()
        buf_meta.clear()

    for row in rows:
        path = sdir / row["filename"]
        if not path.exists():
            n_skipped += 1
            continue
        try:
            wav = audio_mod.load_audio_16k(str(path), start_sec=row["start_sec"], duration_sec=5.0)
        except Exception as e:  # noqa: BLE001 — log and skip unreadable windows
            logger.warning("Load failed %s @ %.0f: %s", row["filename"], row["start_sec"], e)
            n_skipped += 1
            continue
        buf_wavs.append(audio_mod.fit_window(wav))
        buf_meta.append({k: row[k] for k in
                         ("filename", "window_idx", "start_sec", "split", "primary_label", "ground_truth")})
        if len(buf_wavs) >= batch_size:
            flush()
    flush()

    logger.info("%s: %d windows in %.1fs (skipped %d)",
                encoder, len(meta_rows), time.time() - t0, n_skipped)
    _write_unified_parquet(output_path, meta_rows, emb_list, encoder, variant)
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
) -> int:
    """Dispatch extraction by the encoder's backend (see registry)."""
    from birdclef_300.registry import BIRDNET_BMZ, PERCH_CACHE, TORCH, get_spec

    backend = get_spec(encoder).backend
    if backend == TORCH:
        return extract_torch_labeled(
            cfg, encoder, variant, checkpoint, adapter, state_dict, output_path, limit
        )
    if backend == PERCH_CACHE:
        return ingest_perch_labeled(cfg, variant, output_path, limit)
    if backend == BIRDNET_BMZ:
        from birdclef_300.extract_birdnet import extract_birdnet_labeled

        return extract_birdnet_labeled(cfg, variant, output_path, limit)
    raise ValueError(f"Unknown backend '{backend}' for encoder '{encoder}'")
