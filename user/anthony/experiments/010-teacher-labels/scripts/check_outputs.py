#!/usr/bin/env python3
"""Verify completeness of teacher label outputs.

Checks that all expected partitions produced output files.
"""

import sys
from pathlib import Path

import yaml


def check_model_partitions(label_dir: Path, model: str, source_label: str, num_parts: int) -> list[int]:
    """Check which partitions are missing for a model/source combo."""
    if model == "bambird":
        output_dir = label_dir / model / source_label
        expected = {output_dir / f"part_{i:04d}.parquet" for i in range(num_parts)}
    else:
        output_dir = label_dir / model / source_label / "parts"
        # Check both embed and predict
        expected = set()
        for subdir in ["embed", "predict"]:
            for i in range(num_parts):
                expected.add(output_dir / subdir / f"part_{i:04d}.parquet")

    missing = [p for p in sorted(expected) if not p.exists()]
    return missing


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/teacher.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    label_dir = Path(cfg["output"]["root"]).expanduser()
    focal_parts = cfg["partitions"]["focal"]
    soundscape_parts = cfg["partitions"]["soundscape"]

    all_ok = True
    for model in ["bambird", "birdnet", "perch"]:
        for source_label, num_parts in [("train_audio", focal_parts), ("train_soundscapes", soundscape_parts)]:
            missing = check_model_partitions(label_dir, model, source_label, num_parts)
            status = "OK" if not missing else f"MISSING {len(missing)}"
            print(f"  {model}/{source_label}: {status}")
            if missing:
                all_ok = False
                for p in missing[:5]:
                    print(f"    {p}")
                if len(missing) > 5:
                    print(f"    ... and {len(missing) - 5} more")

    if all_ok:
        print("\nAll outputs complete!")
    else:
        print("\nSome outputs are missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
