"""Exp 038 Stage 0b — stream-extract the curated BirdSet-XCL audio (NO GPU; big I/O).

We can't selectively pull individual files (filepath = bare XC id, no shard index; tar.gz isn't
random-access) and the full 480 GB won't fit beside the extract on a 540 GB disk. So we STREAM:
for each `XCL/XCL_shard_*.tar.gz` (revision `data`) → download → extract only the basenames in our
curated manifest → **delete the tarball** → next. Peak disk ≈ one shard (~5 GB) + the kept audio
(~150 GB at cap-500). Resumable (per-shard done-set + skip-if-dest-exists). Output layout:
`<out>/audio/<ebird_code>/<XCid>.ogg` (mirrors the comp's train_audio/<label>/<file>).

*** CLAUDE.md: NEVER fill home (20 GB quota). Downloads go to PROJECT storage; set HF_HOME too. ***

Run (PACE; smoke 2 shards first):
    export HF_TOKEN=<token>
    export HF_HOME=/storage/project/ps-clef2026_birdclef-0/mgustineli3/cache/huggingface
    python scripts/download_birdset_audio.py --limit-shards 2     # verify pipeline + a few extracts
    python scripts/download_birdset_audio.py                      # full 98 shards (resumable)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import pandas as pd

REPO = "DBD-research-group/BirdSet"
REVISION = "data"
SHARD_GLOB_PREFIX = "XCL/XCL_shard_"


def extract_wanted_from_tar(
    tar_path: Path, wanted: dict[str, str], out_audio_dir: Path
) -> list[str]:
    """Extract members whose BASENAME is in `wanted` (basename -> ebird_code) to
    out_audio_dir/<code>/<basename>. Skips already-present dests (resumable). Returns extracted
    basenames. Matches by basename so it handles both bare `XC*.ogg` and `data/.../XC*.ogg` members."""
    extracted: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            code = wanted.get(base)
            if code is None:
                continue
            dest = out_audio_dir / code / base
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                continue
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp, "wb") as out:
                shutil.copyfileobj(src, out)
            tmp.rename(dest)  # atomic-ish
            extracted.append(base)
    return extracted


def list_xcl_shards() -> list[str]:
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION)
    shards = sorted(f for f in files if f.startswith(SHARD_GLOB_PREFIX) and f.endswith(".tar.gz"))
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/storage/project/ps-clef2026_birdclef-0/shared/pretrain/birdset-neotropical"),
    )
    ap.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Where transient tarballs download+delete (default: <out-dir>/_shards). On SLURM set "
        "to $TMPDIR for fast local NVMe + auto-clean (keeps project-disk peak to just the audio).",
    )
    ap.add_argument(
        "--limit-shards", type=int, default=None, help="Process only first N shards (smoke)."
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="List shards + wanted count; download nothing."
    )
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    manifest = pd.read_parquet(args.out_dir / "manifest.parquet")
    wanted = {
        os.path.basename(str(fp)): str(code)
        for fp, code in zip(manifest["filepath"], manifest["ebird_code"], strict=False)
    }
    print(
        f"curated manifest: {len(manifest):,} rows, {len(wanted):,} unique basenames, "
        f"{manifest['ebird_code'].nunique()} species"
    )

    shards = list_xcl_shards()
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    print(f"XCL shards to process: {len(shards)} (e.g. {shards[:2]})")

    audio_dir = args.out_dir / "audio"
    shard_dir = args.shard_dir if args.shard_dir else args.out_dir / "_shards"
    done_path = args.out_dir / "_done_shards.json"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()

    if args.dry_run:
        print(
            f"[DRY-RUN] {len(shards)} shards, {len(wanted):,} wanted files; done so far: {len(done)}"
        )
        return 0

    total_extracted = 0
    for i, shard in enumerate(shards, 1):
        if shard in done:
            print(f"[{i}/{len(shards)}] {shard} — already done, skip")
            continue
        print(f"[{i}/{len(shards)}] downloading {shard} ...", flush=True)
        local = hf_hub_download(
            repo_id=REPO,
            filename=shard,
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(shard_dir),
        )
        got = extract_wanted_from_tar(Path(local), wanted, audio_dir)
        total_extracted += len(got)
        os.remove(local)  # free the ~5 GB tarball immediately (peak-disk control)
        done.add(shard)
        done_path.write_text(json.dumps(sorted(done)))
        print(f"    extracted {len(got):,} (running total {total_extracted:,}/{len(wanted):,})")

    have = {
        p.name for code_dir in audio_dir.iterdir() if code_dir.is_dir() for p in code_dir.iterdir()
    }
    missing = sorted(set(wanted) - have)
    print(
        f"\nDONE. extracted-this-run={total_extracted:,}; on-disk={len(have):,}/{len(wanted):,}; "
        f"missing={len(missing):,} (e.g. {missing[:5]})"
    )
    (args.out_dir / "missing_files.json").write_text(json.dumps(missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
