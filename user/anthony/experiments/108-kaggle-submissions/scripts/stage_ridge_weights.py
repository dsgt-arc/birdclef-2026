#!/usr/bin/env python3
"""Stage the `birdclef-2026-weights-ridge` Kaggle dataset directory.

Bundles:
  - perch_v2_no_dft.onnx              (downloaded via huggingface_hub)
  - ridge_model.npz                   (exp 106 artifact)
  - perch_species_mapping.json        (exp 106 artifact)
  - birdclef_kaggle/*.py              (vendored shared module)
  - dataset-metadata.json             (for `kaggle datasets create`)

Usage:
    python scripts/stage_ridge_weights.py --output-dir artifacts/dist/weights-ridge

After staging, push with:
    kaggle datasets create -p artifacts/dist/weights-ridge
    kaggle datasets version -p artifacts/dist/weights-ridge -m "update"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

EXP108_DIR = Path(__file__).resolve().parent.parent
EXP106_ARTIFACTS = EXP108_DIR.parent / "106-perch-bayesian" / "artifacts"
BIRDCLEF_KAGGLE_SRC = EXP108_DIR / "src" / "birdclef_kaggle"

PERCH_REPO = "justinchuby/Perch-onnx"
PERCH_FILE = "perch_v2_no_dft.onnx"


def stage(output_dir: Path, kaggle_user: str) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Staging to {output_dir}")

    ridge_src = EXP106_ARTIFACTS / "ridge_model.npz"
    species_src = EXP106_ARTIFACTS / "perch_species_mapping.json"
    for src in (ridge_src, species_src):
        if not src.exists():
            raise FileNotFoundError(src)
        dest = output_dir / src.name
        shutil.copy2(src, dest)
        print(f"  copied {src.name} ({dest.stat().st_size / 1e6:.1f} MB)")

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

    meta = {
        "title": "birdclef-2026-weights-ridge",
        "id": f"{kaggle_user}/birdclef-2026-weights-ridge",
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
    p.add_argument("--output-dir", type=Path, default=EXP108_DIR / "artifacts" / "dist" / "weights-ridge")
    p.add_argument("--kaggle-user", default="acmiyaguchi")
    args = p.parse_args()
    stage(args.output_dir, args.kaggle_user)


if __name__ == "__main__":
    main()
