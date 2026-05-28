"""Sydorskyi 2025 SpecAugment (freq/time masking) — direct port.

Source: vendor/sydorskyi-2025/code_base/augmentations/spec_augment.py @ 053594f

Applied INSIDE the model forward (XCPretrainWaveClassifier), training-mode only,
on the mel spectrogram (B, 1, n_mels, time). Wraps torchaudio
FrequencyMasking / TimeMasking with per-sample stochastic application:

    for each sample i in the batch:
        with prob p:
            n = randint(1, mask_max_masks+1)   # 1..mask_max_masks masks
            apply n independent torchaudio masks of max width mask_max_length

Verified config (docs/sydorskyi-config-verified.md § SpecAugment, selected_ebs.py:151-164):
    freq_mask: mask_max_length=10, mask_max_masks=3, p=0.3, inplace=True
    time_mask: mask_max_length=20, mask_max_masks=3, p=0.3, inplace=True

Confidence: VERIFIED (Phase A.2). 2026-05-28.
"""

import numpy as np
import torch.nn as nn
from torch import Tensor
from torchaudio.transforms import FrequencyMasking, TimeMasking


class CustomMasking(nn.Module):
    """Per-sample stochastic masking wrapper (base class).

    Args:
        mask_max_length: torchaudio mask_param (max mask width in bins/frames).
        mask_max_masks: max number of independent masks applied per sample (>=1).
        p: per-sample Bernoulli probability of applying any masking.
        inplace: if True, mutate input tensor; else operate on a clone.
    """

    def __init__(
        self,
        mask_max_length: int,
        mask_max_masks: int,
        p: float = 1.0,
        inplace: bool = True,
    ):
        super().__init__()
        assert isinstance(mask_max_masks, int) and mask_max_masks > 0
        self.mask_max_masks = mask_max_masks
        self.mask_max_length = mask_max_length
        self.mask_module: nn.Module | None = None  # set by subclass
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            output = x.clone()
        for i in range(x.shape[0]):
            if np.random.binomial(n=1, p=self.p):
                n_applies = np.random.randint(low=1, high=self.mask_max_masks + 1)
                for _ in range(n_applies):
                    if self.inplace:
                        x[i : i + 1] = self.mask_module(x[i : i + 1])
                    else:
                        output[i : i + 1] = self.mask_module(output[i : i + 1])
        return x if self.inplace else output


class CustomTimeMasking(CustomMasking):
    """Per-sample stochastic time masking (torchaudio TimeMasking)."""

    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p: float = 1.0, inplace: bool = True
    ):
        super().__init__(mask_max_length, mask_max_masks, p=p, inplace=inplace)
        self.mask_module = TimeMasking(time_mask_param=mask_max_length)


class CustomFreqMasking(CustomMasking):
    """Per-sample stochastic frequency masking (torchaudio FrequencyMasking)."""

    def __init__(
        self, mask_max_length: int, mask_max_masks: int, p: float = 1.0, inplace: bool = True
    ):
        super().__init__(mask_max_length, mask_max_masks, p=p, inplace=inplace)
        self.mask_module = FrequencyMasking(freq_mask_param=mask_max_length)
