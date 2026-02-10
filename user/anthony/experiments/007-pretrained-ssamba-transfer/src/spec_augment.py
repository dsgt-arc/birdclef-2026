"""SpecAugment: frequency and time masking for mel-spectrograms.

Applied during training only to improve generalization.
Reference: Park et al. "SpecAugment" (2019).
"""

import torch


class SpecAugment:
    """Apply frequency and time masking to mel-spectrograms.

    Args:
        freq_mask_param: Maximum width of frequency masks (in bins)
        time_mask_param: Maximum width of time masks (in frames)
        num_freq_masks: Number of frequency masks to apply
        num_time_masks: Number of time masks to apply
    """

    def __init__(
        self,
        freq_mask_param: int = 20,
        time_mask_param: int = 50,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, mel_specs: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment masking.

        Args:
            mel_specs: (B, 1, n_mels, frames) tensor

        Returns:
            Masked tensor (same shape), modified in-place
        """
        mel_specs = mel_specs.clone()
        _, _, n_mels, frames = mel_specs.shape

        for _ in range(self.num_freq_masks):
            f = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(1, n_mels - f), (1,)).item()
            mel_specs[:, :, f0 : f0 + f, :] = 0

        for _ in range(self.num_time_masks):
            t = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(1, frames - t), (1,)).item()
            mel_specs[:, :, :, t0 : t0 + t] = 0

        return mel_specs
