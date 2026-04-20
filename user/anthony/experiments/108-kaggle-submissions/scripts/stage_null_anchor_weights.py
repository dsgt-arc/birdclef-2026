#!/usr/bin/env python3
"""Stage the `birdclef-2026-weights-null-anchor` Kaggle dataset directory.

Bundles the softshrink null-site-anchored residual hopfield checkpoint
(exp 107, seed 42, fold S18 — best single-seed pooled LOSO 0.7801).

Contents:
  - perch_v2_no_dft.onnx              (downloaded via huggingface_hub)
  - best.pt                           (exp 107 S18 checkpoint)
  - perch_species_mapping.json        (exp 106 artifact)
  - birdclef_kaggle/*.py              (vendored shared module)
  - birdclef_107_model.py             (vendored exp 107 model module)
  - dataset-metadata.json             (for `kaggle datasets create`)

Usage:
    python scripts/stage_null_anchor_weights.py --output-dir artifacts/dist/weights-null-anchor
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

EXP108_DIR = Path(__file__).resolve().parent.parent
EXP106_ARTIFACTS = EXP108_DIR.parent / "106-perch-bayesian" / "artifacts"
EXP107_SRC = EXP108_DIR.parent / "107-hopfield-prototypes" / "src" / "birdclef_107"
BIRDCLEF_KAGGLE_SRC = EXP108_DIR / "src" / "birdclef_kaggle"

PERCH_REPO = "justinchuby/Perch-onnx"
PERCH_FILE = "perch_v2_no_dft.onnx"

DEFAULT_CHECKPOINT = Path(
    "/storage/home/hcoda1/8/amiyaguchi3/scratch/birdclef-2026/checkpoints/"
    "exp107-ar-softshrink-null-site-anchored-residual-seed42-sbatch/S18/best.pt"
)


def stage(
    output_dir: Path,
    kaggle_user: str,
    checkpoint_path: Path,
) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Staging to {output_dir}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    dest_ckpt = output_dir / "best.pt"
    shutil.copy2(checkpoint_path, dest_ckpt)
    print(f"  copied best.pt ({dest_ckpt.stat().st_size / 1e6:.1f} MB)")

    species_src = EXP106_ARTIFACTS / "perch_species_mapping.json"
    if not species_src.exists():
        raise FileNotFoundError(species_src)
    shutil.copy2(species_src, output_dir / species_src.name)
    print(f"  copied {species_src.name}")

    print(f"  downloading {PERCH_REPO}/{PERCH_FILE}...")
    perch_path = Path(hf_hub_download(repo_id=PERCH_REPO, filename=PERCH_FILE))
    dest_perch = output_dir / PERCH_FILE
    shutil.copy2(perch_path, dest_perch)
    print(f"  copied Perch ONNX ({dest_perch.stat().st_size / 1e6:.1f} MB)")

    vendored = output_dir / "birdclef_kaggle"
    if vendored.exists():
        shutil.rmtree(vendored)
    shutil.copytree(
        BIRDCLEF_KAGGLE_SRC,
        vendored,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"  vendored birdclef_kaggle ({len(list(vendored.glob('*.py')))} modules)")

    # Vendor just model.py as a standalone module under a unique name so the
    # Kaggle kernel can `import birdclef_107_model` without the rest of the
    # exp 107 package tree (which pulls polars/duckdb for cache helpers).
    model_src = EXP107_SRC / "model.py"
    if not model_src.exists():
        raise FileNotFoundError(model_src)
    shutil.copy2(model_src, output_dir / "birdclef_107_model.py")
    print(f"  vendored birdclef_107.model as birdclef_107_model.py")

    meta = {
        "title": "birdclef-2026-weights-null-anchor",
        "id": f"{kaggle_user}/birdclef-2026-weights-null-anchor",
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = output_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path.name}")

    print("\nContents:")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(output_dir)
            size_mb = p.stat().st_size / 1e6
            print(f"  {rel} ({size_mb:.2f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=Path,
        default=EXP108_DIR / "artifacts" / "dist" / "weights-null-anchor",
    )
    p.add_argument("--kaggle-user", default="acmiyaguchi")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = p.parse_args()
    stage(args.output_dir, args.kaggle_user, args.checkpoint)


if __name__ == "__main__":
    main()
