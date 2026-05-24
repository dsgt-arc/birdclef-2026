"""SpecAugment for mel spectrograms — verbatim port of Tucker's notebook cell 10.

Tucker's implementation:
    FREQ_MASK_PARAM = 10
    TIME_MASK_PARAM = 10
    NUM_FREQ_MASKS  = 1
    NUM_TIME_MASKS  = 2

    class SpecAugment(nn.Module):
        def __init__(self):
            super().__init__()
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_PARAM)

        def forward(self, mel):
            for _ in range(NUM_FREQ_MASKS):
                mel = self.freq_mask(mel)
            for _ in range(NUM_TIME_MASKS):
                mel = self.time_mask(mel)
            return mel

Each call samples one frequency mask of width ~ Uniform[0, freq_mask_param)
and zeroes that band, then samples NUM_TIME_MASKS time masks each of width
~ Uniform[0, time_mask_param) and zeroes those columns. Tucker applies this
AFTER per-sample mel normalization, training only (validation step skips it).
Both behaviors are enforced by the caller (`DistilledSEDFinetuneModule._compute_mel`).

Reference:
- analysis/notes/2026-05-09-tucker-distilled-sed-code-audit.md (§ Tucker's mel pipeline)
- analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md (§ 2.4 augmentations)
- vendor/kaggle/notebooks/birdclef-2026/bc2026-distilled-sed/bc2026-distilled-sed.ipynb (cell 10)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class SpecAugment(nn.Module):
    """SpecAugment: time + frequency masking on mel spectrograms.

    Wraps `torchaudio.transforms.FrequencyMasking` and `TimeMasking` and applies
    them in sequence. Tucker uses 1 freq mask + 2 time masks with mask widths
    sampled from `Uniform[0, 10)`.

    Args:
        freq_mask_param: Max frequency-mask width (default: 10, Tucker's).
        time_mask_param: Max time-mask width (default: 10, Tucker's).
        num_freq_masks: How many independent freq masks to apply (default: 1).
        num_time_masks: How many independent time masks to apply (default: 2).

    Input shape: (B, C, freq, time) — typical mel-spectrogram layout from
    `torchaudio.transforms.MelSpectrogram` + `AmplitudeToDB`.
    Output shape: identical.

    Notes:
        - Both `FrequencyMasking` and `TimeMasking` from torchaudio mask in-place
          with the value 0 by default. Since this runs AFTER per-sample
          normalization in our pipeline, "0" represents the per-sample mean —
          consistent with Tucker's order in cell 14.
        - The torchaudio modules do NOT honor `self.training`; they always mask.
          Eval-mode skipping is the caller's responsibility (the LightningModule
          guards this via `if training and self.spec_augment is not None`).
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 10,
        num_freq_masks: int = 1,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply freq masks then time masks in sequence."""
        for _ in range(self.num_freq_masks):
            mel = self.freq_mask(mel)
        for _ in range(self.num_time_masks):
            mel = self.time_mask(mel)
        return mel

    def extra_repr(self) -> str:
        return (
            f"freq_mask_param={self.freq_mask_param}, "
            f"time_mask_param={self.time_mask_param}, "
            f"num_freq_masks={self.num_freq_masks}, "
            f"num_time_masks={self.num_time_masks}"
        )
