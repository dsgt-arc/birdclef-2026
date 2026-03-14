"""CLI entry point for EDA pipeline."""

import argparse

import luigi

from .tasks import AllEDA, DATA_DIR, ARTIFACTS_DIR


def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 EDA pipeline")
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Path to BirdCLEF 2026 competition data",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=ARTIFACTS_DIR,
        help="Output directory for derived datasets",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Luigi worker count (task parallelism)",
    )
    parser.add_argument(
        "--n-audio-workers",
        type=int,
        default=8,
        help="Multiprocessing pool size for audio duration measurement",
    )
    parser.add_argument(
        "--birdclef-2025-csv",
        default="~/scratch/birdclef-2026/data/birdclef-2025/train.csv",
        help="Path to BirdCLEF 2025 train.csv for species overlap analysis",
    )
    args = parser.parse_args()

    luigi.build(
        [
            AllEDA(
                data_dir=args.data_dir,
                artifacts_dir=args.artifacts_dir,
                n_workers=args.n_audio_workers,
                birdclef_2025_csv=args.birdclef_2025_csv,
            ),
        ],
        workers=args.workers,
        local_scheduler=True,
    )
