#!/usr/bin/env python3
"""Stage the `birdclef-2026-deps` Kaggle dataset directory.

Downloads CPU wheels (onnxruntime, soundfile, numpy, pandas) into a
directory so the Kaggle kernel can `pip install --no-index --find-links`
offline.

Usage:
    python scripts/stage_deps.py --output-dir artifacts/dist/deps

After staging:
    kaggle datasets create -p artifacts/dist/deps
    kaggle datasets version -p artifacts/dist/deps -m "update"
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EXP108_DIR = Path(__file__).resolve().parent.parent

# Minimal runtime deps for ridge submission. Pinned loose; kaggle pip resolves.
REQUIREMENTS = [
    "onnxruntime",
    "soundfile",
    "numpy",
    "pandas",
]

# Kaggle kernels run CPython x86_64; pin the interpreter to what the live
# Kaggle image ships. As of 2026-04 the default image is 3.12. Modern
# onnxruntime wheels ship with `manylinux_2_28` tags, so we pass both that
# and the older `manylinux2014` target to keep resolution flexible.
PLATFORM_ARGS = [
    "--platform", "manylinux_2_28_x86_64",
    "--platform", "manylinux2014_x86_64",
    "--python-version", "3.12",
    "--implementation", "cp",
    "--only-binary", ":all:",
]


def stage(output_dir: Path, kaggle_user: str) -> None:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Staging wheels to {output_dir}")

    cmd = [
        "pip", "download",
        "--no-deps", *PLATFORM_ARGS,
        "-d", str(output_dir),
        *REQUIREMENTS,
    ]
    print(f"  running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Second pass: resolve transitive deps for the ones that have them.
    cmd_with_deps = [
        "pip", "download", *PLATFORM_ARGS,
        "-d", str(output_dir),
        *REQUIREMENTS,
    ]
    print(f"  running: {' '.join(cmd_with_deps)}")
    subprocess.check_call(cmd_with_deps)

    meta = {
        "title": "birdclef-2026-deps",
        "id": f"{kaggle_user}/birdclef-2026-deps",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (output_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    wheels = sorted(output_dir.glob("*.whl"))
    print(f"\n{len(wheels)} wheels staged:")
    for w in wheels:
        print(f"  {w.name} ({w.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=EXP108_DIR / "artifacts" / "dist" / "deps")
    p.add_argument("--kaggle-user", default="acmiyaguchi")
    args = p.parse_args()
    stage(args.output_dir, args.kaggle_user)


if __name__ == "__main__":
    main()
