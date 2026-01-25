"""Teacher output caching for distillation training.

Caches BEATs teacher embeddings and token IDs to safetensors shards
for fast loading during distillation training.

Cache format (v2 - optimized):
    - teacher_embeds: (N, seq_len, 768) fp16 - contextualized embeddings
    - token_ids: (N, seq_len) int16 - discrete token IDs for static embedding lookup

Static embeddings are NOT cached - they are reconstructed at load time from
token_ids using a lookup table (~24MB). This reduces cache size by 50%.

Usage:
    from birdclef.caching import TeacherCacher

    cacher = TeacherCacher(
        encoder_path="path/to/BEATs_iter3.pt",
        tokenizer_path="path/to/Tokenizers_iter3.pt",
    )
    cacher.process_shard(tar_path, output_path)
"""

from __future__ import annotations

import io
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Add vendor path for BEATs imports
VENDOR_PATH = Path(__file__).parents[4] / "vendor" / "unilm" / "beats"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))


class TeacherCacher:
    """Caches BEATs teacher outputs for distillation training.

    Args:
        encoder_path: Path to BEATs_iter3.pt checkpoint
        tokenizer_path: Path to Tokenizers_iter3.pt checkpoint
        device: Device for inference (default: cuda if available)
        target_length: Target audio length in samples (default: 160000 = 10s at 16kHz)
        batch_size: Batch size for inference (default: 32)
    """

    def __init__(
        self,
        encoder_path: str | Path,
        tokenizer_path: str | Path,
        device: str | None = None,
        target_length: int = 160000,
        batch_size: int = 32,
    ):
        from birdclef.distillation import load_teacher_and_tokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_length = target_length
        self.batch_size = batch_size

        print("Loading teacher and tokenizer...")
        self.teacher, self.tokenizer = load_teacher_and_tokenizer(
            encoder_path, tokenizer_path
        )
        self.teacher = self.teacher.to(self.device)
        self.tokenizer = self.tokenizer.to(self.device)
        print(f"  Device: {self.device}")

    def load_audio_from_tar(
        self, tar: tarfile.TarFile, member_name: str
    ) -> torch.Tensor:
        """Load and preprocess audio from an open tar file.

        Args:
            tar: Open tarfile object
            member_name: Name of the audio file within the tar

        Returns:
            Preprocessed waveform tensor (target_length,)
        """
        f = tar.extractfile(member_name)
        audio_data, sr = sf.read(io.BytesIO(f.read()))

        waveform = torch.from_numpy(audio_data.astype(np.float32))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=-1)

        # Pad or trim to target length
        if len(waveform) < self.target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.target_length - len(waveform))
            )
        else:
            waveform = waveform[: self.target_length]

        return waveform

    def process_shard(self, tar_path: Path) -> dict[str, torch.Tensor]:
        """Process all samples in a tar shard.

        Args:
            tar_path: Path to the tar file containing audio

        Returns:
            Dictionary with 'teacher_embeds' and 'token_ids' tensors
        """
        shard_teacher_embeds = []
        shard_token_ids = []

        with tarfile.open(tar_path, "r") as tar:
            audio_members = [
                m
                for m in tar.getmembers()
                if m.name.endswith((".wav", ".flac", ".mp3"))
            ]

            for i in range(0, len(audio_members), self.batch_size):
                batch_members = audio_members[i : i + self.batch_size]

                waveforms = torch.stack(
                    [self.load_audio_from_tar(tar, m.name) for m in batch_members]
                )
                waveforms = waveforms.to(self.device)

                with torch.no_grad():
                    # Get teacher embeddings
                    teacher_embeds, _ = self.teacher.extract_features(
                        waveforms, padding_mask=None
                    )

                    # Get token IDs
                    token_ids = self.tokenizer.extract_labels(waveforms)
                    if token_ids.dim() == 1:
                        seq_len = token_ids.shape[0] // waveforms.shape[0]
                        token_ids = token_ids.view(waveforms.shape[0], seq_len)

                # Store as FP16/int16 to save space
                shard_teacher_embeds.append(teacher_embeds.cpu().half())
                shard_token_ids.append(token_ids.cpu().short())

        return {
            "teacher_embeds": torch.cat(shard_teacher_embeds, dim=0),
            "token_ids": torch.cat(shard_token_ids, dim=0),
        }

    def cache_shards(
        self,
        tar_files: list[Path],
        output_dir: Path,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> dict[str, int]:
        """Cache multiple tar shards to safetensors files.

        Args:
            tar_files: List of tar file paths to process
            output_dir: Directory to write safetensors files
            progress_callback: Optional callback(shard_name, n_samples) for progress

        Returns:
            Statistics dictionary with processed/skipped counts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_samples = 0
        processed = 0
        skipped = 0

        for tar_path in tqdm(tar_files, desc="Caching shards"):
            shard_name = tar_path.stem
            output_path = output_dir / f"{shard_name}.safetensors"
            temp_path = output_dir / f"{shard_name}.safetensors.tmp"

            # Skip if already exists (idempotent)
            if output_path.exists():
                skipped += 1
                continue

            # Clean up any leftover temp file
            if temp_path.exists():
                temp_path.unlink()

            try:
                tensors = self.process_shard(tar_path)
                n_samples = tensors["teacher_embeds"].shape[0]
                total_samples += n_samples

                # Atomic write: save to temp, then rename
                save_file(tensors, str(temp_path))
                temp_path.rename(output_path)
                processed += 1

                if progress_callback:
                    progress_callback(shard_name, n_samples)
                else:
                    tqdm.write(f"  {shard_name}: {n_samples} samples")

            except Exception as e:
                tqdm.write(f"  ERROR {shard_name}: {e}")
                if temp_path.exists():
                    temp_path.unlink()

        return {
            "processed": processed,
            "skipped": skipped,
            "total_samples": total_samples,
        }


def cache_teacher_outputs(
    audioset_dir: Path,
    cache_dir: Path,
    encoder_path: Path,
    tokenizer_path: Path,
    shard_range: tuple[int, int] | None = None,
    device: str | None = None,
) -> dict[str, int]:
    """High-level function to cache teacher outputs.

    Args:
        audioset_dir: Directory containing train/*.tar files
        cache_dir: Output directory for safetensors cache
        encoder_path: Path to BEATs encoder checkpoint
        tokenizer_path: Path to BEATs tokenizer checkpoint
        shard_range: Optional (start, end) tuple to process subset of shards
        device: Device for inference

    Returns:
        Statistics dictionary
    """
    # Find tar files
    tar_files = sorted((audioset_dir / "train").glob("*.tar"))
    print(f"Found {len(tar_files)} tar shards")

    if shard_range:
        start, end = shard_range
        tar_files = tar_files[start : end + 1]
        print(f"Processing shards {start} to {end} ({len(tar_files)} shards)")

    # Initialize cacher
    cacher = TeacherCacher(
        encoder_path=encoder_path,
        tokenizer_path=tokenizer_path,
        device=device,
    )

    # Process shards
    start_time = time.time()
    stats = cacher.cache_shards(tar_files, cache_dir)
    elapsed = time.time() - start_time

    # Report results
    print("\n" + "=" * 60)
    print("CACHING COMPLETE")
    print("=" * 60)
    print(f"  Processed: {stats['processed']} shards ({stats['total_samples']} samples)")
    print(f"  Skipped:   {stats['skipped']} shards (already cached)")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if stats["total_samples"] > 0:
        print(f"  Speed:     {elapsed/stats['total_samples']*1000:.1f}ms per sample")

    # Report cache size
    cache_files = list(cache_dir.glob("*.safetensors"))
    if cache_files:
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"\nCache directory: {cache_dir}")
        print(f"  Files: {len(cache_files)}")
        print(f"  Total size: {total_size / (1024**3):.2f} GB")

    return stats


def verify_cache_format(cache_dir: Path) -> dict[str, any]:
    """Verify cache format and report statistics.

    Args:
        cache_dir: Directory containing safetensors cache files

    Returns:
        Dictionary with cache statistics
    """
    from safetensors import safe_open

    cache_files = sorted(cache_dir.glob("*.safetensors"))
    if not cache_files:
        raise ValueError(f"No safetensors files found in {cache_dir}")

    # Check first shard
    sample_file = cache_files[0]
    with safe_open(str(sample_file), framework="pt") as f:
        keys = list(f.keys())
        shapes = {k: f.get_slice(k).get_shape() for k in keys}
        dtypes = {}
        for k in keys:
            tensor = f.get_tensor(k)
            dtypes[k] = str(tensor.dtype)

    # Determine format version
    has_static_embeds = "static_embeds" in keys
    format_version = "v1" if has_static_embeds else "v2"

    # Calculate sizes
    total_size = sum(f.stat().st_size for f in cache_files)
    n_shards = len(cache_files)

    # Count total samples
    total_samples = 0
    for cf in cache_files:
        with safe_open(str(cf), framework="pt") as f:
            total_samples += f.get_slice("teacher_embeds").get_shape()[0]

    per_sample_bytes = total_size / total_samples if total_samples > 0 else 0

    stats = {
        "format_version": format_version,
        "n_shards": n_shards,
        "total_samples": total_samples,
        "total_size_gb": total_size / (1024**3),
        "per_sample_kb": per_sample_bytes / 1024,
        "keys": keys,
        "shapes": shapes,
        "dtypes": dtypes,
        "has_static_embeds": has_static_embeds,
    }

    print(f"Cache format: {format_version}")
    print(f"  Shards: {n_shards}")
    print(f"  Samples: {total_samples:,}")
    print(f"  Size: {stats['total_size_gb']:.2f} GB")
    print(f"  Per sample: {stats['per_sample_kb']:.2f} KB")
    print(f"  Keys: {keys}")
    for k, shape in shapes.items():
        print(f"    {k}: {shape} ({dtypes[k]})")

    if has_static_embeds:
        print("\nNote: This cache uses v1 format with redundant static_embeds.")
        print("Consider regenerating with v2 format for 50% space savings.")

    return stats
