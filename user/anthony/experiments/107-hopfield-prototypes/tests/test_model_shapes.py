"""Phase A test 1 — shape and output-range sanity for PrototypeRetrievalModel."""

from __future__ import annotations

import torch

from birdclef_107.model import PrototypeRetrievalModel


def test_default_model_param_count_is_about_3_9m() -> None:
    model = PrototypeRetrievalModel()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Actual count at default shapes is 3,939,799. Check we're in the right
    # ballpark so accidental layer-shape drift still trips the test.
    # (design.md claims ~3.4M; the real total is ~3.9M — the doc estimate
    # was optimistic. Keeping the assertion pinned to reality.)
    assert 3_800_000 <= n_params <= 4_100_000, (
        f"expected ~3.9M trainable params, got {n_params:,}"
    )


def test_forward_with_site_returns_batch_species_shape() -> None:
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
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    with torch.no_grad():
        out = model(emb, site_idx=None)
    assert out.shape == (8, 234)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_unsupervised_site_energy_returns_scalar() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))
    loss = model.unsupervised_site_energy(emb, site_idx)
    assert loss.shape == ()
    assert torch.isfinite(loss).all()
