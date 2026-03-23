"""Inference dataset for extracting embeddings and predictions from audio files.

Chunks arbitrary audio files into non-overlapping segments for batch inference.
Unlike BirdCLEFDataset (which requires metadata with labels), this dataset
takes a flat list of file descriptors and returns (transformed_audio, metadata).
"""

import logging

import soundfile as sf
import torch
from torch.utils.data import Dataset

from birdclef.data.dataset import _load_audio_chunk

logger = logging.getLogger(__name__)


class InferenceChunkDataset(Dataset):
    """Dataset that chunks audio files for inference.

    Takes a list of file descriptors, probes each file's duration,
    splits into non-overlapping chunks, and returns (mel_spectrogram, metadata).

    Args:
        file_list: List of dicts with keys: path (Path), filename (str), source (str)
        sample_rate: Target sample rate
        chunk_duration: Chunk duration in seconds
        transform: Mel spectrogram transform
    """

    def __init__(
        self,
        file_list: list[dict],
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,
        transform=None,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.transform = transform

        # Build chunk index
        self.chunks = []
        skipped = 0
        for file_info in file_list:
            audio_path = file_info["path"]
            try:
                info = sf.info(str(audio_path))
                total_samples = int(info.frames * sample_rate / info.samplerate)
            except Exception:
                skipped += 1
                continue

            chunk_idx = 0
            pos = 0
            while pos < total_samples:
                real_samples = min(self.chunk_samples, total_samples - pos)
                start_sec = pos / sample_rate
                end_sec = (pos + real_samples) / sample_rate
                self.chunks.append(
                    {
                        "audio_path": str(audio_path),
                        "filename": file_info["filename"],
                        "source": file_info["source"],
                        "start_sample": pos,
                        "real_samples": real_samples,
                        "chunk_idx": chunk_idx,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    }
                )
                pos += self.chunk_samples
                chunk_idx += 1

        if skipped > 0:
            logger.warning(f"Skipped {skipped} files (unreadable)")
        logger.info(
            f"InferenceChunkDataset: {len(self.chunks)} chunks "
            f"from {len(file_list) - skipped} files"
        )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        waveform = _load_audio_chunk(
            chunk["audio_path"],
            self.sample_rate,
            start_sample=chunk["start_sample"],
            num_samples=chunk["real_samples"],
        )

        # Pad to chunk_samples if needed
        if waveform.shape[0] < self.chunk_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_samples - waveform.shape[0])
            )

        # Apply mel transform
        if self.transform is not None:
            waveform = self.transform(waveform)

        metadata = {
            "filename": chunk["filename"],
            "source": chunk["source"],
            "chunk_idx": chunk["chunk_idx"],
            "start_sec": chunk["start_sec"],
            "end_sec": chunk["end_sec"],
        }

        return waveform, metadata


def inference_collate_fn(batch):
    """Collate function that stacks waveforms and collects metadata as lists."""
    waveforms = torch.stack([item[0] for item in batch])
    metadata_keys = batch[0][1].keys()
    metadata = {key: [item[1][key] for item in batch] for key in metadata_keys}
    return waveforms, metadata
