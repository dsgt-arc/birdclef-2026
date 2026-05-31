"""Exp 038 Stage 0 — inspect the BirdSet HF repo file layout (datasets 4.x can't run its script).

`datasets` 4.x removed remote dataset-script support, so `load_dataset("DBD-research-group/BirdSet",
"XCL", trust_remote_code=True)` no longer works. We instead read BirdSet's parquet directly via
`huggingface_hub` + pyarrow column-pruned reads (pull metadata columns, skip the audio-bytes column).

This one-shot inspector discovers the layout so we can finalize `load_birdset_metadata`:
  1) list repo files (sizes) → which dirs hold XCL data + any metadata-only file
  2) for one XCL parquet shard: schema (columns/types) + num_rows via HfFileSystem (footer read, no
     full download)
  3) column-pruned read of ~5 rows of the non-audio columns → confirm it works + show sample values
  4) total XCL parquet shard count + bytes → size estimate

Run (PACE, internet; set a token first to avoid throttling):
    export HF_TOKEN=<your_hf_token>     # from https://huggingface.co/settings/tokens (read scope)
    python scripts/inspect_birdset_repo.py
"""

from __future__ import annotations

import sys

REPO = "DBD-research-group/BirdSet"


def main() -> int:
    from huggingface_hub import HfApi, HfFileSystem

    api = HfApi()
    print(f"=== repo_info({REPO}, dataset) ===")
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    sib = info.siblings
    print(f"total files: {len(sib)}")

    parquet = [(s.rfilename, s.size or 0) for s in sib if s.rfilename.endswith(".parquet")]
    xcl = [(p, sz) for p, sz in parquet if "XCL" in p or "xcl" in p]
    meta_like = [
        s.rfilename
        for s in sib
        if any(k in s.rfilename.lower() for k in ("meta", "metadata"))
        and not s.rfilename.endswith(".parquet")
    ]

    print(f"\n=== parquet files: {len(parquet)} total, {len(xcl)} XCL-tagged ===")
    for p, sz in (xcl or parquet)[:40]:
        print(f"  {sz/1e6:9.1f} MB  {p}")
    if len(xcl or parquet) > 40:
        print(f"  ... (+{len(xcl or parquet) - 40} more)")
    print(f"\nXCL parquet total: {sum(sz for _, sz in xcl)/1e9:.1f} GB across {len(xcl)} shards")
    print(f"metadata-only-looking files: {meta_like}")

    # top-level dirs
    dirs = sorted({s.rfilename.split("/")[0] for s in sib if "/" in s.rfilename})
    print(f"\ntop-level dirs: {dirs}")

    shards = [p for p, _ in (xcl or parquet)]
    if not shards:
        print("\nNo parquet shards found — inspect the dir listing above for the data format.")
        return 0

    shard = shards[0]
    print(f"\n=== schema of one shard via HfFileSystem (footer only): {shard} ===")
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    full = f"datasets/{REPO}/{shard}"
    with fs.open(full, "rb") as f:
        pf = pq.ParquetFile(f)
        print(f"num_rows in this shard: {pf.metadata.num_rows}")
        print("columns (name: type):")
        for field in pf.schema_arrow:
            print(f"  {field.name}: {field.type}")
        # column-pruned read of metadata cols (skip audio bytes)
        meta_cols = [
            c for c in pf.schema_arrow.names if c not in ("audio",) and "audio" not in c.lower()
        ]
        print(f"\nmetadata columns (audio excluded): {meta_cols}")
        head = (
            pf.read_row_group(0, columns=[c for c in meta_cols if c in pf.schema_arrow.names])
            .slice(0, 5)
            .to_pandas()
        )
        import pandas as pd

        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("\n=== sample 5 rows (metadata cols only) ===")
            print(head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
