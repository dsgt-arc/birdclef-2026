#!/usr/bin/env python3
"""Extract TensorBoard metrics and query with DuckDB.

Usage:
    tb_metrics.py <log_dir> [--tags TAG,...] [--last N] [--query SQL] [--export FILE]

Examples:
    # Show latest values for all tags
    tb_metrics.py logs/focal/lightning_logs/

    # Show last 5 values of specific tags
    tb_metrics.py logs/ --tags val/cmap,val/auroc --last 5

    # Custom DuckDB query
    tb_metrics.py logs/ --query "SELECT tag, max(value) as best FROM tb GROUP BY tag"

    # Multi-run comparison (glob pattern)
    tb_metrics.py logs/*/lightning_logs/ --tags val/cmap

    # Export to parquet
    tb_metrics.py logs/ --export metrics.parquet
"""

import argparse
import glob
import os
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("duckdb not installed. Install with: uv pip install duckdb", file=sys.stderr)
    sys.exit(1)

try:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
except ImportError:
    print(
        "tensorboard not installed. Install with: uv pip install tensorboard",
        file=sys.stderr,
    )
    sys.exit(1)


def find_event_files(log_dir: str) -> list[tuple[str, str]]:
    """Find TB event files, returning (run_name, event_dir) pairs."""
    log_path = Path(log_dir)
    results = []

    # Direct event files in directory
    events = list(log_path.glob("events.out.tfevents.*"))
    if events:
        results.append((log_path.name, str(log_path)))
        return results

    # Search subdirectories (lightning_logs/version_*/  or direct version dirs)
    for events_file in sorted(log_path.rglob("events.out.tfevents.*")):
        event_dir = str(events_file.parent)
        # Build a run name from the full path, keeping meaningful parts
        # e.g. logs/focal/lightning_logs/version_123 -> "focal"
        rel = events_file.parent.relative_to(log_path)
        parts = list(rel.parts)
        # Strip "lightning_logs" and "version_*" noise
        parts = [
            p
            for p in parts
            if p != "lightning_logs" and not p.startswith("version_")
        ]
        if not parts:
            # If all parts stripped, use the parent of lightning_logs
            # e.g. log_dir = logs/focal/lightning_logs -> "focal"
            # Walk up from log_path to find a meaningful name
            candidate = log_path
            while candidate.name in ("lightning_logs",) or candidate.name.startswith(
                "version_"
            ):
                candidate = candidate.parent
            parts = [candidate.name]
        run_name = "/".join(parts) if parts else log_path.name
        results.append((run_name, event_dir))

    return results


def load_events(event_dir: str, tags_filter: list[str] | None = None) -> list[dict]:
    """Load scalar events from a TB event directory."""
    ea = EventAccumulator(event_dir)
    ea.Reload()

    available = ea.Tags().get("scalars", [])
    if tags_filter:
        tags = [t for t in available if t in tags_filter]
    else:
        tags = available

    rows = []
    for tag in tags:
        for event in ea.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="TensorBoard → DuckDB metrics")
    parser.add_argument(
        "log_dirs", nargs="+", help="TB log directories (supports glob patterns)"
    )
    parser.add_argument("--tags", help="Comma-separated tag filter")
    parser.add_argument(
        "--last", type=int, default=0, help="Show last N values per tag"
    )
    parser.add_argument("--query", help="Custom DuckDB SQL query (table name: tb)")
    parser.add_argument("--export", help="Export to parquet file")
    parser.add_argument(
        "--summary", action="store_true", default=True, help="Show summary (default)"
    )
    args = parser.parse_args()

    tags_filter = args.tags.split(",") if args.tags else None

    # Expand glob patterns
    expanded_dirs = []
    for pattern in args.log_dirs:
        matches = glob.glob(pattern)
        if matches:
            expanded_dirs.extend(matches)
        else:
            expanded_dirs.append(pattern)

    # Collect all events
    all_rows = []
    for log_dir in expanded_dirs:
        if not os.path.isdir(log_dir):
            print(f"Warning: {log_dir} is not a directory, skipping", file=sys.stderr)
            continue
        for run_name, event_dir in find_event_files(log_dir):
            rows = load_events(event_dir, tags_filter)
            for r in rows:
                r["run"] = run_name
            all_rows.extend(rows)

    if not all_rows:
        print("No events found.", file=sys.stderr)
        sys.exit(1)

    # Load into DuckDB
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE tb (
            run VARCHAR,
            tag VARCHAR,
            step BIGINT,
            value DOUBLE,
            wall_time DOUBLE
        )
    """
    )

    con.executemany(
        "INSERT INTO tb VALUES (?, ?, ?, ?, ?)",
        [(r["run"], r["tag"], r["step"], r["value"], r["wall_time"]) for r in all_rows],
    )

    # Export if requested
    if args.export:
        con.execute(f"COPY tb TO '{args.export}' (FORMAT PARQUET)")
        print(f"Exported {len(all_rows)} rows to {args.export}")

    # Custom query
    if args.query:
        result = con.execute(args.query).fetchdf()
        print(result.to_string(index=False))
        return

    # Default: summary view
    if args.last > 0:
        result = con.execute(
            f"""
            SELECT run, tag, step, round(value, 6) as value
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY run, tag ORDER BY step DESC) as rn
                FROM tb
            )
            WHERE rn <= {args.last}
            ORDER BY run, tag, step
        """
        ).fetchdf()
        print(result.to_string(index=False))
    else:
        # Show latest value per run/tag
        result = con.execute(
            """
            SELECT
                run,
                tag,
                LAST(step ORDER BY step) as latest_step,
                round(LAST(value ORDER BY step), 6) as latest,
                round(MIN(value), 6) as min,
                round(MAX(value), 6) as max,
                COUNT(*) as n_points
            FROM tb
            GROUP BY run, tag
            ORDER BY run, tag
        """
        ).fetchdf()
        print(result.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
