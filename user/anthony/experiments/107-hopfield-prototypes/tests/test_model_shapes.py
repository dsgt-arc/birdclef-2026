"""Shape and output-range sanity for PrototypeRetrievalModel."""

from __future__ import annotations

import torch

from birdclef_107.model import PrototypeRetrievalModel


def test_default_model_param_count_is_about_5m() -> None:
    """v2 default: ~3.94M (v1 backbone + heads) + ~1.05M (cross-attn MHA)
    - 513 (v1 scalar gate, removed) ≈ 4.99M trainable params."""
    model = PrototypeRetrievalModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 4_800_000 <= n_params <= 5_100_000, (
        f"expected ~5.0M trainable params, got {n_params:,}"
    )


def test_forward_with_site_returns_batch_species_shape() -> None:
    """With a site_idx, forward routes through cross-attention."""
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))
    with torch.no_grad():
        out = model(emb, site_idx)
    assert out.shape == (8, 234)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_forward_without_site_returns_batch_species_shape() -> None:
    """With site_idx=None, forward uses the shared species bank (focal
    path, also the OOD-site Kaggle fallback)."""
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    with torch.no_grad():
        out = model(emb, site_idx=None)
    assert out.shape == (8, 234)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_unsupervised_site_contrastive_returns_scalar() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))
    loss = model.unsupervised_site_contrastive(emb, site_idx)
    assert loss.shape == ()
    assert torch.isfinite(loss).all()


def test_unsupervised_cross_attn_contrastive_returns_scalar() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))
    loss = model.unsupervised_cross_attn_contrastive(emb, site_idx)
    assert loss.shape == ()
    assert torch.isfinite(loss).all()
