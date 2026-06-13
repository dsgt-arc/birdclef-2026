"""Gradient-enabled pooled forward for the shared encoder wrappers.

`AudioEncoder.extract_features` is decorated `@torch.no_grad()`, which blocks the
gradient flow LoRA/DAPT need. This helper mirrors each wrapper's pooling body
(reusing the wrapper's own mel/fbank front-end) but keeps autograd on. Kept in one
place so LoRA and DAPT share identical preprocessing with extraction.
"""

from __future__ import annotations

import torch


def pooled_features(wrapper, encoder: str, wav: torch.Tensor) -> torch.Tensor:
    """(B, 1, N) 16 kHz waveform -> (B, embed_dim) pooled features, WITH grad.

    Reuses `wrapper._waveform_to_*` so adaptation and exp-300 extraction agree on
    preprocessing; only the no-grad guard differs.
    """
    device = wrapper.device
    x = wav.squeeze(1)  # (B, N)

    if encoder == "eat":
        mel = wrapper._waveform_to_mel(x).to(device)
        feats = wrapper.model.extract_features(mel)
        return feats[:, 1:].mean(dim=1)

    if encoder == "ssamba_hf":
        fbank = wrapper._waveform_to_fbank(x).to(device)
        return wrapper.model.get_embeddings(fbank)

    if encoder == "beats":
        feats, _ = wrapper.model.extract_features(x.to(device))
        return feats.mean(dim=1)

    raise ValueError(f"No grad-forward defined for encoder '{encoder}' (adaptation roster: beats/eat/ssamba_hf)")
