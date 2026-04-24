#!/usr/bin/env python3
"""Bulk export of attribution figures for top-K highest/lowest loss samples.

Programmatic version of model_explorer.ipynb. For each selected sample, saves:
- 4-panel PNG (mel, gradient saliency, top-10 species, per-frame analysis)
- Competitor comparison PNG (GT vs top-1 or top-2, see rule below)
- 5-second audio crop as WAV

Plus summary batch figures for top-5 highest and top-5 lowest loss (raw mode).

Two modes:

**Raw mode** (default): ranks samples by loss across the whole val set (or a
`--source` subset) and outputs under `figures/<model>/<source>/highest_loss/`
and `lowest_loss/`. This is the original behavior.

**Filtered mode**: activated by any `--filter-category`, `--filter-species`, or
`--filter-saliency-informative` flag. Joins with the triage parquet, applies
filters, dedupes on `(filename, offset_sec)`, outputs a flat dir under
`figures/<model>/<slug>/` where `slug = <category>[_<species>][_saliency]`.
No `highest_loss`/`lowest_loss` split — we're inspecting a curated set.

Competitor comparison rule:
- highest_loss bucket (and filtered mode): always save (top-1 when top-1 != GT,
  top-2 when top-1 == GT).
- lowest_loss  bucket: save only when top-1 != GT.

Usage:
    # Raw top-20 highest/lowest on all val samples:
    python export_figures.py \\
        --losses data/losses_v5j.parquet \\
        --config ../experiments/019-sed-ablations/config/hgnet_b0_sed_v5j.yaml \\
        --checkpoint /path/to/best-epochepoch=19.ckpt \\
        --top-k 20 --source all

    # Filtered: all roahaw mixed_uncertainty samples:
    python export_figures.py \\
        --losses data/losses_v5j.parquet \\
        --config ... --checkpoint ... \\
        --filter-category mixed_uncertainty \\
        --filter-species roahaw \\
        --top-k -1  # -1 or 0 means "all matched samples"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

_analysis_dir = Path(__file__).parent
sys.path.insert(0, str(_analysis_dir))

from attribution import compute_attribution, load_audio  # noqa: E402
from model_loader import load_config, load_model  # noqa: E402
from plotting import plot_batch_grid, plot_competitor_comparison, plot_four_panel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _infer_model_name(losses_path: Path, override: str | None) -> str:
    if override:
        return override
    stem = losses_path.stem
    return stem.removeprefix("losses_") if stem.startswith("losses_") else stem


def _rank_filename(rank: int, loss: float, row_id: int, suffix: str = "") -> str:
    return f"rank{rank:02d}_loss{loss:.4f}_row{row_id}{suffix}"


def _infer_triage_path(losses_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    stem = losses_path.stem.replace("losses_", "triage_", 1)
    return losses_path.parent / f"{stem}.parquet"


def apply_triage_filters(
    losses_df: pd.DataFrame,
    triage_path: Path,
    categories: list[str] | None,
    species: str | None,
    saliency_only: bool,
) -> tuple[pd.DataFrame, str]:
    """Join with triage, apply category/species/saliency filters, dedupe.

    Returns:
        (filtered_df, slug). The slug is `<cat1+cat2...>[_<species>][_saliency]`.
    """
    if not triage_path.exists():
        raise SystemExit(f"Triage parquet not found: {triage_path}. Run triage.py first.")
    triage = pd.read_parquet(triage_path)
    merge_cols = ["idx", "triage_category", "saliency_informative"]
    merged = losses_df.merge(triage[merge_cols], on="idx", how="left")

    if categories:
        merged = merged[merged["triage_category"].isin(categories)]
    if species:
        merged = merged[merged["primary_label"] == species]
    if saliency_only:
        merged = merged[merged["saliency_informative"].astype(bool)]

    before = len(merged)
    merged = merged.drop_duplicates(subset=["filename", "offset_sec"]).reset_index(drop=True)
    after = len(merged)
    if before != after:
        logger.info(f"Deduped on (filename, offset_sec): {before} -> {after}")

    slug_parts: list[str] = []
    if categories:
        slug_parts.append("+".join(sorted(categories)))
    if species:
        slug_parts.append(species)
    if saliency_only:
        slug_parts.append("saliency")
    slug = "_".join(slug_parts)
    return merged, slug


def _compute_and_save_sample(
    row: pd.Series,
    rank: int,
    bucket: str,
    out_dir: Path,
    model,
    mel_transform,
    species_list: list[str],
    lse_temp: float,
    sample_rate: int,
    window_duration: float,
    device: str,
) -> dict:
    """Process one sample: compute attribution, save 4-panel + competitor + WAV.

    Returns a dict with per-sample metadata for the manifest.
    """
    row_id = int(row["idx"])
    loss = float(row["loss"])
    primary = row["primary_label"]
    top1 = row["top1_species"]
    source = row["source"]
    offset = float(row.get("offset_sec", 0.0) or 0.0)

    sample_meta = {
        "idx": row_id,
        "primary_label": primary,
        "source": source,
        "loss": loss,
        "gt_rank": int(row["gt_rank"]),
        "top1_species": top1,
    }

    # Load audio once; reuse for GT + competitor attribution.
    waveform = load_audio(
        row["filename"],
        sample_rate=sample_rate,
        duration=window_duration,
        offset=offset,
    )

    base = _rank_filename(rank, loss, row_id)

    # Save WAV crop alongside the figures.
    wav_path = out_dir / f"{base}.wav"
    sf.write(wav_path, waveform.numpy(), sample_rate, subtype="PCM_16")

    # Skip samples whose GT species is not in the model's output list (rare edge).
    if primary not in species_list:
        logger.warning(f"Skipping row {row_id}: primary label {primary!r} not in species list")
        return {
            "rank": rank,
            "row_id": row_id,
            "loss": loss,
            "primary_label": primary,
            "top1_species": top1,
            "four_panel": None,
            "competitor": None,
            "audio": wav_path.name,
            "note": "primary_label not in species list",
        }

    gt_idx = species_list.index(primary)
    gt_result = compute_attribution(
        model=model,
        mel_transform=mel_transform,
        waveform=waveform,
        species_idx=gt_idx,
        species_name=primary,
        lse_temperature=lse_temp,
        device=device,
    )

    four_panel_path = out_dir / f"{base}.png"
    plot_four_panel(gt_result, species_list, sample_meta, four_panel_path)

    # Competitor comparison rule.
    top1_is_gt = top1 == primary
    save_competitor = False
    competitor_role = None
    competitor_species = None

    if bucket == "highest_loss":
        save_competitor = True
        if top1_is_gt:
            competitor_role = "top2"
            # top-2 = second highest prob species.
            top_order = np.argsort(gt_result.clipwise_probs)[::-1]
            top2_idx = next(i for i in top_order if species_list[i] != primary)
            competitor_species = species_list[int(top2_idx)]
        else:
            competitor_role = "top1"
            competitor_species = top1
    else:  # lowest_loss
        if not top1_is_gt:
            save_competitor = True
            competitor_role = "top1"
            competitor_species = top1

    competitor_filename = None
    if save_competitor and competitor_species in species_list:
        comp_idx = species_list.index(competitor_species)
        comp_result = compute_attribution(
            model=model,
            mel_transform=mel_transform,
            waveform=waveform,
            species_idx=comp_idx,
            species_name=competitor_species,
            lse_temperature=lse_temp,
            device=device,
        )
        competitor_path = out_dir / f"{base}_competitor.png"
        plot_competitor_comparison(
            gt_result, comp_result, sample_meta, competitor_path, competitor_role
        )
        competitor_filename = competitor_path.name

    return {
        "rank": rank,
        "row_id": row_id,
        "loss": loss,
        "primary_label": primary,
        "source": source,
        "gt_rank": int(row["gt_rank"]),
        "top1_species": top1,
        "top1_prob": float(row["top1_prob"]),
        "gt_prob": float(row["gt_prob"]),
        "four_panel": four_panel_path.name,
        "competitor": competitor_filename,
        "competitor_role": competitor_role if competitor_filename else None,
        "competitor_species": competitor_species if competitor_filename else None,
        "audio": wav_path.name,
        "offset_sec": offset,
        "filename": row["filename"],
    }


def _collect_batch_results(
    batch_df: pd.DataFrame,
    model,
    mel_transform,
    species_list: list[str],
    lse_temp: float,
    sample_rate: int,
    window_duration: float,
    device: str,
) -> tuple[list, list[dict]]:
    """Compute attributions for rows in order; return parallel result + meta lists."""
    results, metas = [], []
    for _, row in batch_df.iterrows():
        primary = row["primary_label"]
        if primary not in species_list:
            continue
        offset = float(row.get("offset_sec", 0.0) or 0.0)
        waveform = load_audio(
            row["filename"],
            sample_rate=sample_rate,
            duration=window_duration,
            offset=offset,
        )
        sp_idx = species_list.index(primary)
        res = compute_attribution(
            model=model,
            mel_transform=mel_transform,
            waveform=waveform,
            species_idx=sp_idx,
            species_name=primary,
            lse_temperature=lse_temp,
            device=device,
        )
        results.append(res)
        metas.append(
            {
                "idx": int(row["idx"]),
                "primary_label": primary,
                "source": row["source"],
                "loss": float(row["loss"]),
                "gt_rank": int(row["gt_rank"]),
                "top1_species": row["top1_species"],
            }
        )
    return results, metas


def export_figures(
    losses_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    top_k: int,
    batch_k: int,
    source_filter: str,
    model_name: str | None,
    device: str,
    filter_categories: list[str] | None = None,
    filter_species: str | None = None,
    filter_saliency_informative: bool = False,
    triage_path_override: Path | None = None,
) -> dict:
    config = load_config(config_path)
    sample_rate = config["data"]["sample_rate"]
    window_duration = config["data"]["window_duration"]
    lse_temp = config.get("sed_head", {}).get("lse_temperature", 1.0)

    df = pd.read_parquet(losses_path)
    logger.info(f"Loaded {len(df)} samples from {losses_path}")

    model_name = _infer_model_name(losses_path, model_name)
    filtered_mode = bool(filter_categories or filter_species or filter_saliency_informative)

    if filtered_mode:
        triage_path = _infer_triage_path(losses_path, triage_path_override)
        logger.info(f"Filtered mode. Triage parquet: {triage_path}")
        df, filter_slug = apply_triage_filters(
            df, triage_path, filter_categories, filter_species, filter_saliency_informative
        )
        logger.info(f"After triage filter: {len(df)} samples matched filter slug='{filter_slug}'")
        if source_filter != "all":
            df = df[df["source"] == source_filter].reset_index(drop=True)
            logger.info(f"After --source={source_filter}: {len(df)} samples")
        if len(df) == 0:
            raise SystemExit("No samples matched the filter.")

        run_dir = output_dir / model_name / filter_slug
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output dir: {run_dir}")

        logger.info("Loading model")
        model, mel_transform, species_list = load_model(config_path, checkpoint_path, device=device)

        # top_k <= 0 means "all matched samples".
        effective_k = len(df) if top_k <= 0 else min(top_k, len(df))
        selected = df.nlargest(effective_k, "loss").reset_index(drop=True)
        logger.info(f"Ranking by loss desc; exporting {effective_k} of {len(df)} matched samples")

        manifest_ranks = []
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            logger.info(
                f"[{rank}/{effective_k}] row={row['idx']} species={row['primary_label']} "
                f"source={row['source']} loss={row['loss']:.4f} "
                f"top1={row['top1_species']}({row['top1_prob']:.3f})"
            )
            entry = _compute_and_save_sample(
                row,
                rank,
                "highest_loss",  # competitor rule: always save top-1 or top-2
                run_dir,
                model,
                mel_transform,
                species_list,
                lse_temp,
                sample_rate,
                window_duration,
                device,
            )
            manifest_ranks.append(entry)

        # Optional summary batch — first batch_k after loss-desc ranking.
        if batch_k > 0 and len(selected) > 0:
            summary_dir = run_dir / "summary"
            summary_dir.mkdir(exist_ok=True)
            batch_results, batch_metas = _collect_batch_results(
                selected.head(batch_k),
                model,
                mel_transform,
                species_list,
                lse_temp,
                sample_rate,
                window_duration,
                device,
            )
            if batch_results:
                plot_batch_grid(
                    batch_results,
                    batch_metas,
                    f"Filter '{filter_slug}': top-{batch_k} by loss",
                    summary_dir / f"top{batch_k}_batch.png",
                )

        manifest = {
            "mode": "filtered",
            "model_name": model_name,
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "losses_path": str(losses_path),
            "triage_path": str(triage_path),
            "filter": {
                "categories": filter_categories,
                "species": filter_species,
                "saliency_informative": filter_saliency_informative,
                "source_filter": source_filter,
                "dedupe_key": "filename,offset_sec",
            },
            "filter_slug": filter_slug,
            "top_k_requested": top_k,
            "effective_k": effective_k,
            "batch_k": batch_k,
            "device": device,
            "sample_rate": sample_rate,
            "window_duration": window_duration,
            "lse_temperature": lse_temp,
            "n_samples_matched": int(len(df)),
            "n_species": len(species_list),
            "generated_at": datetime.now(UTC).isoformat(),
            "ranks": manifest_ranks,
        }
        manifest_path = run_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Wrote manifest to {manifest_path}")
        return manifest

    # --- Raw mode (existing behavior) ---
    if source_filter != "all":
        df = df[df["source"] == source_filter].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} samples with source={source_filter}")
        if len(df) == 0:
            raise SystemExit(f"No samples with source={source_filter}")

    run_dir = output_dir / model_name / source_filter
    highest_dir = run_dir / "highest_loss"
    lowest_dir = run_dir / "lowest_loss"
    summary_dir = run_dir / "summary"
    for d in (highest_dir, lowest_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {run_dir}")

    logger.info("Loading model")
    model, mel_transform, species_list = load_model(config_path, checkpoint_path, device=device)

    highest_df = df.nlargest(top_k, "loss").reset_index(drop=True)
    lowest_df = df.nsmallest(top_k, "loss").reset_index(drop=True)

    # Per-sample exports.
    manifest_ranks = {"highest_loss": [], "lowest_loss": []}
    for rank, (_, row) in enumerate(highest_df.iterrows(), start=1):
        logger.info(
            f"[highest {rank}/{top_k}] row={row['idx']} species={row['primary_label']} "
            f"loss={row['loss']:.4f}"
        )
        entry = _compute_and_save_sample(
            row,
            rank,
            "highest_loss",
            highest_dir,
            model,
            mel_transform,
            species_list,
            lse_temp,
            sample_rate,
            window_duration,
            device,
        )
        manifest_ranks["highest_loss"].append(entry)

    for rank, (_, row) in enumerate(lowest_df.iterrows(), start=1):
        logger.info(
            f"[lowest {rank}/{top_k}] row={row['idx']} species={row['primary_label']} "
            f"loss={row['loss']:.4f}"
        )
        entry = _compute_and_save_sample(
            row,
            rank,
            "lowest_loss",
            lowest_dir,
            model,
            mel_transform,
            species_list,
            lse_temp,
            sample_rate,
            window_duration,
            device,
        )
        manifest_ranks["lowest_loss"].append(entry)

    # Summary batch figures — reuse the first batch_k entries from each bucket.
    logger.info(f"Generating summary batch figures (batch_k={batch_k})")
    batch_hi_df = highest_df.head(batch_k)
    batch_lo_df = lowest_df.head(batch_k)
    hi_results, hi_metas = _collect_batch_results(
        batch_hi_df,
        model,
        mel_transform,
        species_list,
        lse_temp,
        sample_rate,
        window_duration,
        device,
    )
    lo_results, lo_metas = _collect_batch_results(
        batch_lo_df,
        model,
        mel_transform,
        species_list,
        lse_temp,
        sample_rate,
        window_duration,
        device,
    )
    if hi_results:
        plot_batch_grid(
            hi_results,
            hi_metas,
            f"Top-{batch_k} Highest-Loss Samples: Mel + Gradient Saliency",
            summary_dir / f"top{batch_k}_highest_batch.png",
        )
    if lo_results:
        plot_batch_grid(
            lo_results,
            lo_metas,
            f"Top-{batch_k} Lowest-Loss Samples: Mel + Gradient Saliency",
            summary_dir / f"top{batch_k}_lowest_batch.png",
        )

    manifest = {
        "mode": "raw",
        "model_name": model_name,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "losses_path": str(losses_path),
        "source_filter": source_filter,
        "top_k": top_k,
        "batch_k": batch_k,
        "device": device,
        "sample_rate": sample_rate,
        "window_duration": window_duration,
        "lse_temperature": lse_temp,
        "n_samples_total": int(len(df)),
        "n_species": len(species_list),
        "generated_at": datetime.now(UTC).isoformat(),
        "ranks": manifest_ranks,
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--losses", type=Path, required=True, help="Precomputed losses parquet")
    parser.add_argument(
        "--logits",
        type=Path,
        default=None,
        help="Optional: precomputed logits .npy (unused currently; reserved)",
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint .ckpt path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Root output directory (default: figures/)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Per-sample figures per bucket (raw mode) or cap on filtered exports "
        "(-1 or 0 = all matched samples in filtered mode)",
    )
    parser.add_argument("--batch-k", type=int, default=5, help="Samples in summary batch figures")
    parser.add_argument(
        "--source",
        choices=["all", "focal", "soundscape"],
        default="all",
        help="Filter samples by source",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name (default: inferred from losses filename)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    # Triage-aware filters (activate filtered mode)
    parser.add_argument(
        "--filter-category",
        nargs="+",
        default=None,
        help="Filter by triage_category (one or more). Activates filtered mode. "
        "Examples: mixed_uncertainty, confident_wrong, gt_invisible",
    )
    parser.add_argument(
        "--filter-species",
        default=None,
        help="Filter by primary_label (single species code). Activates filtered mode.",
    )
    parser.add_argument(
        "--filter-saliency-informative",
        action="store_true",
        help="Filter to samples where gt_prob in [0.05, 0.9]. Activates filtered mode.",
    )
    parser.add_argument(
        "--triage",
        type=Path,
        default=None,
        help="Triage parquet path. Defaults to inferring triage_<model>.parquet next "
        "to the losses parquet.",
    )

    args = parser.parse_args()

    export_figures(
        losses_path=args.losses,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        top_k=args.top_k,
        batch_k=args.batch_k,
        source_filter=args.source,
        model_name=args.model_name,
        device=args.device,
        filter_categories=args.filter_category,
        filter_species=args.filter_species,
        filter_saliency_informative=args.filter_saliency_informative,
        triage_path_override=args.triage,
    )


if __name__ == "__main__":
    main()
