"""Hopfield null-site-anchored residual (exp 107) Kaggle inference.

Loads the `softshrink_null_site_anchored_residual` checkpoint from exp 107
(best single-seed LOSO: 0.7801 pooled AUROC at S18) and runs per-window
predictions.

Forward pass:
    1. Parse site id from each test soundscape filename stem
       (BC2026_Test_<file_id>_<site>_<date>_<time>). OOD sites → site_idx=None.
    2. Run Perch v2 ONNX to get (embedding, label_logits) per 5s window.
    3. Project label_logits through `PerchSidecarProjector` to compact
       (mapped_probs, topk_idx, topk_val).
    4. Invoke `PrototypeRetrievalModel.forward(emb, site_idx, ...)` → sigmoid
       probabilities of shape (12, 234).

Outputs a Kaggle-format submission.csv.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from birdclef_107.model import PrototypeRetrievalModel

from birdclef_kaggle.audio import WINDOWS_PER_CLIP, load_clip
from birdclef_kaggle.ood import resolve_site_indices
from birdclef_kaggle.perch_features import PerchSidecarProjector
from birdclef_kaggle.perch_session import PerchSession
from birdclef_kaggle.row_id import parse as parse_row_id
from birdclef_kaggle.species import build_species_list
from birdclef_kaggle.submission import build_submission


@dataclass
class LoadedModel:
    model: PrototypeRetrievalModel
    args: dict
    val_auroc: float
    fold_site: str


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> LoadedModel:
    """Rebuild `PrototypeRetrievalModel` from a checkpoint and load weights."""
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    args = ckpt["args"] if isinstance(ckpt["args"], dict) else vars(ckpt["args"])

    model = PrototypeRetrievalModel(
        n_species=234,
        n_sites=23,
        K_species=int(args.get("k_species", 8)),
        K_site=int(args.get("k_site", 8)),
        d_input=1536,
        d_hidden=int(args.get("d_hidden", 1024)),
        d_proj=int(args.get("d_proj", 512)),
        cross_attention_heads=int(args.get("cross_attention_heads", 4)),
        proj_dropout=float(args.get("proj_dropout", 0.0)),
        attn_dropout=float(args.get("attn_dropout", 0.0)),
        use_perch_features=bool(args.get("include_perch_features", False)),
        centered_site_residual=bool(args.get("centered_site_residual", False)),
        null_site_anchored_residual=bool(args.get("null_site_anchored_residual", False)),
        softshrink_null_site_anchored_residual=bool(
            args.get("softshrink_null_site_anchored_residual", False)
        ),
        shared_null_site_anchor=bool(args.get("shared_null_site_anchor", False)),
        shared_null_site_shrink=bool(args.get("shared_null_site_shrink", False)),
        shared_site_blend=bool(args.get("shared_site_blend", False)),
        perch_presence_bias=bool(args.get("perch_presence_bias", False)),
        site_species_logit_bias_rank=int(args.get("site_species_logit_bias_rank", 0)),
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        raise RuntimeError(f"missing keys in checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")
    model.to(device).eval()
    return LoadedModel(
        model=model,
        args=args,
        val_auroc=float(ckpt.get("val_auroc", float("nan"))),
        fold_site=str(ckpt.get("fold_site", "")),
    )


def _iter_audio_files(audio_dir: Path) -> Iterable[Path]:
    for ext in ("*.ogg", "*.wav", "*.flac"):
        yield from sorted(audio_dir.glob(ext))


def _site_from_stem(stem: str) -> str | None:
    """Return the S## token from a BC2026 test filename stem, or None."""
    parts = stem.split("_")
    for p in parts:
        if len(p) == 3 and p[0] == "S" and p[1:].isdigit():
            return p
    return None


@torch.inference_mode()
def predict(
    audio_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    sample_submission_path: Path | None = None,
    perch_model_path: Path | None = None,
    perch_batch_size: int = 16,
    device: str = "cpu",
) -> pd.DataFrame:
    audio_dir = Path(audio_dir)
    files = list(_iter_audio_files(audio_dir))
    if not files:
        raise FileNotFoundError(f"No audio files under {audio_dir}")

    species = build_species_list()
    loaded = load_checkpoint(checkpoint_path, device=device)
    model = loaded.model
    print(
        f"loaded ckpt: fold={loaded.fold_site} val_auroc={loaded.val_auroc:.4f} "
        f"use_perch_features={bool(loaded.args.get('include_perch_features'))}",
        flush=True,
    )
    projector = PerchSidecarProjector()

    perch = PerchSession(model_path=perch_model_path)
    needs_logits = bool(loaded.args.get("include_perch_features", False))
    perch_outputs = ["embedding", "label"] if needs_logits else ["embedding"]

    all_row_ids: list[str] = []
    all_scores: list[np.ndarray] = []

    for idx, audio_path in enumerate(files, start=1):
        clip = load_clip(audio_path)  # (12, 160000) float32
        stem = audio_path.stem
        site = _site_from_stem(stem)
        sites_per_window = [site or ""] * WINDOWS_PER_CLIP

        emb_batches = []
        logit_batches = []
        for start in range(0, clip.windows.shape[0], perch_batch_size):
            batch = clip.windows[start : start + perch_batch_size]
            out = perch.run(batch, outputs=perch_outputs)
            emb_batches.append(out["embedding"])
            if needs_logits:
                logit_batches.append(out["label"])
        embeddings = np.concatenate(emb_batches, axis=0)  # (12, 1536)

        if needs_logits:
            logits = np.concatenate(logit_batches, axis=0)  # (12, 14795)
            projected = projector.project(logits)
            perch_mapped = torch.from_numpy(projected["mapped_probs"]).to(device)
            perch_topk_idx = torch.from_numpy(projected["topk_idx"]).to(device)
            perch_topk_val = torch.from_numpy(projected["topk_val"]).to(device)
        else:
            perch_mapped = None
            perch_topk_idx = None
            perch_topk_val = None

        emb = torch.from_numpy(embeddings).to(device)

        site_idx_arr, known = resolve_site_indices(sites_per_window)
        if known.all():
            site_idx_t = torch.from_numpy(site_idx_arr).long().to(device)
            probs = model(
                emb,
                site_idx=site_idx_t,
                perch_mapped=perch_mapped,
                perch_topk_idx=perch_topk_idx,
                perch_topk_val=perch_topk_val,
            )
        else:
            probs = model(
                emb,
                site_idx=None,
                perch_mapped=perch_mapped,
                perch_topk_idx=perch_topk_idx,
                perch_topk_val=perch_topk_val,
            )

        all_scores.append(probs.detach().cpu().numpy().astype(np.float32))
        all_row_ids.extend(f"{stem}_{end_s}" for end_s in clip.end_seconds)

        if idx % 25 == 0:
            print(f"  processed {idx} / {len(files)} files (site={site})", flush=True)

    predictions = np.concatenate(all_scores, axis=0)  # (N*12, 234)
    if predictions.shape[0] != len(all_row_ids):
        raise RuntimeError(
            f"Row/score count mismatch: rows={len(all_row_ids)} scores={predictions.shape[0]}"
        )
    if predictions.shape[0] != len(files) * WINDOWS_PER_CLIP:
        raise RuntimeError(
            f"Expected {len(files) * WINDOWS_PER_CLIP} rows, got {predictions.shape[0]}"
        )

    return build_submission(
        row_ids=all_row_ids,
        predictions=predictions,
        species=species,
        sample_submission_path=sample_submission_path,
        output_path=output_path,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-submission", type=Path, default=None)
    p.add_argument("--perch-model", type=Path, default=None, help="Optional local Perch ONNX path")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    df = predict(
        audio_dir=args.audio_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        sample_submission_path=args.sample_submission,
        perch_model_path=args.perch_model,
        perch_batch_size=args.batch_size,
        device=args.device,
    )
    print(f"submission shape: {df.shape}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
