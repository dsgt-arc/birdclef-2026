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


def test_dropout_is_deterministic_in_eval_and_stochastic_in_train() -> None:
    """v4 regularization: proj_dropout + attn_dropout must be disabled by
    .eval() and active in .train(). Catches a common bug where dropout is
    accidentally always-on or always-off after a model refactor."""
    torch.manual_seed(0)
    model = PrototypeRetrievalModel(proj_dropout=0.5, attn_dropout=0.5)
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.eval()
    with torch.no_grad():
        out1 = model(emb, site_idx)
        out2 = model(emb, site_idx)
    assert torch.allclose(out1, out2), "eval mode should be deterministic"

    model.train()
    # Stochastic path — attn + proj dropout at 0.5 guarantees divergence
    # between two forward calls on the same input (probability of identity
    # at this width is vanishing).
    torch.manual_seed(1)
    out3 = model(emb, site_idx)
    torch.manual_seed(2)
    out4 = model(emb, site_idx)
    assert not torch.allclose(out3, out4), "train mode should be stochastic"
