"""Luigi sharded OGG -> MP3 re-encoding pipeline.

Ports `scripts/encode_mp3.sh` to a sharded Luigi pipeline for resumable bulk
encoding (>~1k files). For small batches (tens to low hundreds), the bash
script is still appropriate.

Sharding: hash(filename) % n_shards. Site distribution in train_soundscapes is
very uneven (S22=3383, S03=5), so by-site sharding produces wildly imbalanced
shards. Hash sharding gives even load; site is recorded in the manifest so
downstream stages can still slice by site cheaply.

Encoding parameters mirror the bash script: libmp3lame, CBR 128 kbps mono @
32 kHz. ffmpeg is invoked via `nix shell nixpkgs#ffmpeg -c` if not on PATH.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import luigi
import polars as pl

from data_analysis.common.paths import (
    RAVEN_PRO_MP3_DIR,
    TRAIN_AUDIO_DIR,
    TRAIN_SOUNDSCAPES_DIR,
)

logger = logging.getLogger(__name__)

SOURCES = {
    "train_soundscapes": TRAIN_SOUNDSCAPES_DIR,
    "train_audio": TRAIN_AUDIO_DIR,
}

SITE_RE = re.compile(r"_S(\d+)_")


def _ffmpeg_cmd() -> list[str]:
    if shutil.which("ffmpeg"):
        return ["ffmpeg"]
    return ["nix", "shell", "nixpkgs#ffmpeg", "-c", "ffmpeg"]


def _shard_of(rel_path: str, n_shards: int) -> int:
    digest = hashlib.sha1(rel_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % n_shards


def _site_of(name: str) -> str | None:
    m = SITE_RE.search(name)
    return f"S{m.group(1)}" if m else None


def _list_source_files(source: str) -> list[Path]:
    src_dir = SOURCES[source]
    return sorted(src_dir.rglob("*.ogg"))


@dataclass(frozen=True)
class _EncodeJob:
    src: str
    dst: str
    rel: str
    site: str | None
    bitrate: str


def _encode_one(job: _EncodeJob) -> dict:
    """Run ffmpeg on one file. Idempotent: skip if dst exists and is non-empty."""
    dst_path = Path(job.dst)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return {
            "input_path": job.src,
            "output_path": job.dst,
            "rel_path": job.rel,
            "site": job.site,
            "dst_bitrate": job.bitrate,
            "status": "skipped",
            "error": None,
        }

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_ffmpeg_cmd(),
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", job.src,
        "-ac", "1", "-ar", "32000",
        "-c:a", "libmp3lame", "-b:a", job.bitrate,
        "-y", job.dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        status = "encoded"
        error: str | None = None
    except subprocess.CalledProcessError as e:
        status = "failed"
        error = (e.stderr or "").strip()[:500]

    return {
        "input_path": job.src,
        "output_path": job.dst,
        "rel_path": job.rel,
        "site": job.site,
        "dst_bitrate": job.bitrate,
        "status": status,
        "error": error,
    }


class EncodeShardTask(luigi.Task):
    """Encode one shard of OGG files to MP3, write per-shard manifest."""

    source = luigi.ChoiceParameter(choices=list(SOURCES.keys()))
    shard_id = luigi.IntParameter()
    n_shards = luigi.IntParameter(default=16)
    bitrate = luigi.Parameter(default="128k")
    out_dir = luigi.Parameter(default=str(RAVEN_PRO_MP3_DIR))
    workers = luigi.IntParameter(default=8)

    def output(self):
        manifest_dir = Path(self.out_dir) / "_manifests" / self.source
        manifest_dir.mkdir(parents=True, exist_ok=True)
        name = f"shard_{self.shard_id:03d}_of_{self.n_shards:03d}.parquet"
        return luigi.LocalTarget(str(manifest_dir / name))

    def run(self):
        src_dir = SOURCES[self.source]
        out_root = Path(self.out_dir)
        all_files = _list_source_files(self.source)
        shard_files = [
            p for p in all_files
            if _shard_of(str(p.relative_to(src_dir)), self.n_shards) == self.shard_id
        ]

        jobs: list[_EncodeJob] = []
        for src in shard_files:
            rel = str(src.relative_to(src_dir))
            # Mirror source layout: out_dir/<source>/<rel>.mp3 for both
            # flat (train_soundscapes) and nested (train_audio/<species>/...) sources.
            dst = out_root / self.source / Path(rel).with_suffix(".mp3")
            jobs.append(_EncodeJob(
                src=str(src),
                dst=str(dst),
                rel=rel,
                site=_site_of(src.name),
                bitrate=str(self.bitrate),
            ))

        logger.info(
            "EncodeShardTask source=%s shard=%d/%d files=%d workers=%d",
            self.source, self.shard_id, self.n_shards, len(jobs), self.workers,
        )

        results: list[dict] = []
        if not jobs:
            self._write_manifest(results)
            return

        with ProcessPoolExecutor(max_workers=int(self.workers)) as pool:
            futures = [pool.submit(_encode_one, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % 50 == 0 or i == len(jobs):
                    enc = sum(1 for r in results if r["status"] == "encoded")
                    skp = sum(1 for r in results if r["status"] == "skipped")
                    fld = sum(1 for r in results if r["status"] == "failed")
                    logger.info(
                        "  shard=%d progress %d/%d enc=%d skp=%d fail=%d",
                        self.shard_id, i, len(jobs), enc, skp, fld,
                    )

        self._write_manifest(results)

    def _write_manifest(self, results: list[dict]) -> None:
        schema = {
            "input_path": pl.Utf8,
            "output_path": pl.Utf8,
            "rel_path": pl.Utf8,
            "site": pl.Utf8,
            "dst_bitrate": pl.Utf8,
            "status": pl.Utf8,
            "error": pl.Utf8,
        }
        df = pl.DataFrame(results, schema=schema)
        out = Path(self.output().path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        n_failed = int(df.filter(pl.col("status") == "failed").height)
        if n_failed:
            logger.warning("shard %d: %d failed encodes (see manifest)", self.shard_id, n_failed)


class EncodeAllTask(luigi.WrapperTask):
    """Wrapper: encode all shards, then write a master manifest."""

    source = luigi.ChoiceParameter(choices=list(SOURCES.keys()))
    n_shards = luigi.IntParameter(default=16)
    bitrate = luigi.Parameter(default="128k")
    out_dir = luigi.Parameter(default=str(RAVEN_PRO_MP3_DIR))
    workers = luigi.IntParameter(default=8)

    def requires(self):
        return [
            EncodeShardTask(
                source=self.source,
                shard_id=i,
                n_shards=self.n_shards,
                bitrate=self.bitrate,
                out_dir=self.out_dir,
                workers=self.workers,
            )
            for i in range(int(self.n_shards))
        ]


class MergeManifestsTask(luigi.Task):
    """Concatenate per-shard manifests into a single master manifest."""

    source = luigi.ChoiceParameter(choices=list(SOURCES.keys()))
    n_shards = luigi.IntParameter(default=16)
    bitrate = luigi.Parameter(default="128k")
    out_dir = luigi.Parameter(default=str(RAVEN_PRO_MP3_DIR))
    workers = luigi.IntParameter(default=8)

    def requires(self):
        return EncodeAllTask(
            source=self.source,
            n_shards=self.n_shards,
            bitrate=self.bitrate,
            out_dir=self.out_dir,
            workers=self.workers,
        )

    def output(self):
        manifest_dir = Path(self.out_dir) / "_manifests" / self.source
        return luigi.LocalTarget(str(manifest_dir / "master_manifest.parquet"))

    def run(self):
        shards = [t.output().path for t in self.requires().requires()]
        df = pl.concat([pl.read_parquet(p) for p in shards])
        out = Path(self.output().path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out)
        counts = (
            df.group_by("status").len().sort("status").to_dicts()
        )
        logger.info("master manifest: %d rows %s", df.height, counts)


def main() -> None:
    """CLI entry point. Defaults to encoding all train_soundscapes."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=list(SOURCES.keys()), default="train_soundscapes")
    parser.add_argument("--n-shards", type=int, default=16)
    parser.add_argument("--bitrate", default="128k")
    parser.add_argument("--out-dir", default=str(RAVEN_PRO_MP3_DIR))
    parser.add_argument("--workers", type=int, default=8,
                        help="ffmpeg processes per shard")
    parser.add_argument("--luigi-workers", type=int, default=1,
                        help="number of shards to run concurrently")
    parser.add_argument("--shard", type=int, default=None,
                        help="run only this shard id (smoke test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.shard is not None:
        task = EncodeShardTask(
            source=args.source,
            shard_id=args.shard,
            n_shards=args.n_shards,
            bitrate=args.bitrate,
            out_dir=args.out_dir,
            workers=args.workers,
        )
    else:
        task = MergeManifestsTask(
            source=args.source,
            n_shards=args.n_shards,
            bitrate=args.bitrate,
            out_dir=args.out_dir,
            workers=args.workers,
        )

    luigi.build(
        [task],
        local_scheduler=True,
        workers=args.luigi_workers,
        log_level="INFO",
    )


if __name__ == "__main__":
    main()
