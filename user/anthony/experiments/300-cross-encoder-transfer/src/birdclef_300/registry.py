"""Encoder registry — the single source of truth for the 6-encoder roster.

Each entry declares a `backend` that selects the extraction path in extract.py:

    torch        — a shared `birdclef.encoders.AudioEncoder` wrapper whose
                   `extract_features((B, 1, N)@16kHz) -> (B, embed_dim)` is driven
                   by the encoder-agnostic batch loop. Builders lazy-import the
                   wrapper so the BirdNET/Perch venvs never need torch+transformers.
    perch_cache  — ingest the 1536-d Perch embeddings already extracted by exp 106.
    birdnet_bmz  — BirdNET embeddings via bioacoustics-model-zoo (separate TF venv).

301 builds adapted encoders and re-drives the same `torch` builders with an
overridden checkpoint, so keep builder signatures stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

TORCH = "torch"
PERCH_CACHE = "perch_cache"
BIRDNET_BMZ = "birdnet_bmz"


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    backend: str
    # builder(enc_cfg: dict, device: str, checkpoint: str | None) -> AudioEncoder
    builder: Callable | None = None


def _expand(p: str | None) -> str | None:
    return str(Path(p).expanduser()) if p else None


def _build_beats(enc_cfg: dict, device: str, checkpoint: str | None):
    from birdclef.encoders import BEATsEncoder

    ckpt = _expand(checkpoint or enc_cfg.get("checkpoint"))
    if ckpt is None:
        raise ValueError("beats requires a checkpoint path (config `checkpoint` or --checkpoint)")
    return BEATsEncoder(checkpoint_path=ckpt, device=device)


def _build_ast(enc_cfg: dict, device: str, checkpoint: str | None):
    from birdclef.encoders import ASTEncoder

    # AST is a frozen HF model; `checkpoint` (a fine-tuned dir) overrides model_name.
    model_name = checkpoint or enc_cfg.get("model_name", "MIT/ast-finetuned-audioset-10-10-0.4593")
    return ASTEncoder(model_name=model_name, device=device)


def _build_eat(enc_cfg: dict, device: str, checkpoint: str | None):
    from birdclef.encoders import EATEncoder

    model_name = checkpoint or enc_cfg.get("model_name", "worstchan/EAT-base_epoch30_pretrain")
    return EATEncoder(
        model_name=model_name,
        target_length=int(enc_cfg.get("target_length", 512)),
        device=device,
    )


def _build_ssamba_hf(enc_cfg: dict, device: str, checkpoint: str | None):
    from birdclef.encoders import SSAMBAHFEncoder

    ckpt = _expand(checkpoint or enc_cfg.get("checkpoint"))
    if ckpt is None:
        raise ValueError("ssamba_hf requires a checkpoint path (config `checkpoint` or --checkpoint)")
    return SSAMBAHFEncoder(
        checkpoint_path=ckpt,
        model_size=enc_cfg.get("model_size", "small"),
        input_tdim=int(enc_cfg.get("input_tdim", 512)),
        device=device,
    )


REGISTRY: dict[str, EncoderSpec] = {
    "beats": EncoderSpec("beats", TORCH, _build_beats),
    "ast": EncoderSpec("ast", TORCH, _build_ast),
    "eat": EncoderSpec("eat", TORCH, _build_eat),
    "ssamba_hf": EncoderSpec("ssamba_hf", TORCH, _build_ssamba_hf),
    "perch": EncoderSpec("perch", PERCH_CACHE),
    "birdnet": EncoderSpec("birdnet", BIRDNET_BMZ),
}


def get_spec(name: str) -> EncoderSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown encoder '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def build_torch_encoder(name: str, enc_cfg: dict, device: str, checkpoint: str | None = None):
    """Instantiate a torch-backend encoder wrapper."""
    spec = get_spec(name)
    if spec.backend != TORCH or spec.builder is None:
        raise ValueError(f"'{name}' is not a torch-backend encoder")
    return spec.builder(enc_cfg, device, checkpoint)
