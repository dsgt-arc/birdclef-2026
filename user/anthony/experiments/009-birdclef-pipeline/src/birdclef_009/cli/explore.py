"""CLI entry point for BirdCLEF data exploration."""

import argparse
import json
import logging

from ..data.explore import explore_dataset


def main():
    parser = argparse.ArgumentParser(description="Explore BirdCLEF dataset")
    parser.add_argument("data_dir", type=str, help="Path to BirdCLEF dataset root")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for CSV outputs (default: data_dir)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save summary results to JSON file")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Thread pool size for parallel file scanning (default: 16)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    results = explore_dataset(
        args.data_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output_json}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
