#!/usr/bin/env python3
"""Kaggle kernel: BirdCLEF 2026 Hopfield null-site-anchored residual (exp 107).

Checkpoint: `softshrink_null_site_anchored_residual` seed 42 fold S18
(pooled LOSO 0.7801 on labeled SC). Runs offline on Kaggle CPU.

Pipeline per 60s test soundscape:
  1. load_clip -> (12, 160000) float32 @ 32 kHz
  2. Perch v2 ONNX -> (12, 1536) embedding + (12, 14795) label logits
  3. PerchSidecarProjector -> (mapped_probs (12,234), topk_idx (12,64), topk_val (12,64))
  4. Parse site from filename stem. OOD sites -> site_idx=None (shared-species path).
  5. PrototypeRetrievalModel forward -> (12, 234) probabilities
  6. Emit 12 rows per clip with row_id = f"{stem}_{end_s}" for end_s in 5..60

Inputs (Kaggle datasets):
  - /kaggle/input/birdclef-2026-deps/                 wheels for offline pip install
  - /kaggle/input/birdclef-2026-weights-null-anchor/  perch_v2_no_dft.onnx, best.pt,
                                                      perch_species_mapping.json,
                                                      birdclef_kaggle/*.py (vendored),
                                                      birdclef_107_model.py

Competition data (mounted read-only):
  - /kaggle/input/birdclef-2026/test_soundscapes/*.ogg
  - /kaggle/input/birdclef-2026/sample_submission.csv

Output:
  - /kaggle/working/submission.csv
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

KAGGLE_USER = "acmiyaguchi"


def _find_input(slug: str) -> str:
    for c in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}"):
        if os.path.exists(c):
            return c
    return f"/kaggle/input/{slug}"


DEPS_DIR = _find_input("birdclef-2026-deps")
WEIGHTS_DIR = _find_input("birdclef-2026-weights-null-anchor")
COMPETITION_DIR = (
    "/kaggle/input/competitions/birdclef-2026"
    if os.path.exists("/kaggle/input/competitions/birdclef-2026")
    else "/kaggle/input/birdclef-2026"
)

if os.path.exists(DEPS_DIR):
    print(f"Installing offline deps from {DEPS_DIR}", flush=True)
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--no-index", "--find-links", DEPS_DIR,
            "onnxruntime", "soundfile", "numpy", "pandas",
        ]
    )

sys.path.insert(0, WEIGHTS_DIR)


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import vendored exp 107 model module under a synthesized `birdclef_107.model` name
# so `PrototypeRetrievalModel.__module__` stays stable if any torch checkpoint
# pickling relied on it (no effect on state_dict load, but harmless).
birdclef_107_pkg = type(sys)("birdclef_107")
sys.modules["birdclef_107"] = birdclef_107_pkg
_model_module = _load_module_from_path("birdclef_107.model", Path(WEIGHTS_DIR) / "birdclef_107_model.py")
birdclef_107_pkg.model = _model_module

import numpy as np  # noqa: E402
import torch  # noqa: E402

from birdclef_107.model import PrototypeRetrievalModel  # noqa: E402

from birdclef_kaggle.audio import WINDOWS_PER_CLIP, load_clip  # noqa: E402
from birdclef_kaggle.ood import resolve_site_indices  # noqa: E402
from birdclef_kaggle.perch_features import PerchSidecarProjector  # noqa: E402
from birdclef_kaggle.perch_session import PerchSession  # noqa: E402
from birdclef_kaggle.species import build_species_list  # noqa: E402
from birdclef_kaggle.submission import build_submission  # noqa: E402


def _site_from_stem(stem: str) -> str | None:
    for p in stem.split("_"):
        if len(p) == 3 and p[0] == "S" and p[1:].isdigit():
            return p
    return None


def _load_checkpoint(path: Path, device: str = "cpu"):
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
        raise RuntimeError(f"missing keys: {missing}")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected}")
    model.to(device).eval()
    return model, args, ckpt


def main() -> None:
    perch_path = Path(WEIGHTS_DIR) / "perch_v2_no_dft.onnx"
    ckpt_path = Path(WEIGHTS_DIR) / "best.pt"
    sample_submission = Path(COMPETITION_DIR) / "sample_submission.csv"
    test_dir = Path(COMPETITION_DIR) / "test_soundscapes"
    out_path = Path("/kaggle/working/submission.csv")

    print(f"Perch ONNX: {perch_path}", flush=True)
    print(f"Checkpoint: {ckpt_path}", flush=True)
    print(f"Test dir: {test_dir}", flush=True)

    device = "cpu"
    species = build_species_list()
    model, args, ckpt = _load_checkpoint(ckpt_path, device=device)
    print(
        f"loaded ckpt: fold={ckpt.get('fold_site')} val_auroc={ckpt.get('val_auroc')} "
        f"use_perch_features={bool(args.get('include_perch_features'))}",
        flush=True,
    )

    projector = PerchSidecarProjector()
    perch = PerchSession(model_path=perch_path)
    needs_logits = bool(args.get("include_perch_features", False))
    perch_outputs = ["embedding", "label"] if needs_logits else ["embedding"]

    audio_files = sorted(test_dir.glob("*.ogg"))
    print(f"Found {len(audio_files)} test soundscapes", flush=True)

    row_ids: list[str] = []
    scores_chunks: list[np.ndarray] = []

    with torch.inference_mode():
        for i, audio_path in enumerate(audio_files):
            clip = load_clip(audio_path)
            stem = audio_path.stem
            site = _site_from_stem(stem)

            out = perch.run(clip.windows, outputs=perch_outputs)
            embeddings = out["embedding"]

            if needs_logits:
                logits = out["label"]
                projected = projector.project(logits)
                perch_mapped = torch.from_numpy(projected["mapped_probs"])
                perch_topk_idx = torch.from_numpy(projected["topk_idx"])
                perch_topk_val = torch.from_numpy(projected["topk_val"])
            else:
                perch_mapped = perch_topk_idx = perch_topk_val = None

            emb = torch.from_numpy(embeddings)

            sites_per_window = [site or ""] * WINDOWS_PER_CLIP
            site_idx_arr, known = resolve_site_indices(sites_per_window)
            if known.all():
                site_idx_t = torch.from_numpy(site_idx_arr).long()
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

            scores_chunks.append(probs.detach().cpu().numpy().astype(np.float32))
            row_ids.extend(f"{stem}_{end_s}" for end_s in clip.end_seconds)

            if (i + 1) % 100 == 0:
                print(f"  processed {i + 1} / {len(audio_files)}", flush=True)

    predictions = (
        np.concatenate(scores_chunks, axis=0)
        if scores_chunks
        else np.zeros((0, 234), dtype=np.float32)
    )

    df = build_submission(
        row_ids=row_ids,
        predictions=predictions,
        species=species,
        sample_submission_path=sample_submission if sample_submission.exists() else None,
        output_path=out_path,
    )
    print(f"Wrote submission: {df.shape} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
