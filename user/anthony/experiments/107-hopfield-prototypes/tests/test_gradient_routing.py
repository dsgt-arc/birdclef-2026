"""Phase A test 2 — L_site_unsup must not leak gradient into the projection.

This is the load-bearing invariant from exp 106's failure analysis: if the
projection is updated by the unsupervised site loss, site structure and
domain shift entangle in the projected space and the species bank
collapses onto site-identifying features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from birdclef_107.model import PrototypeRetrievalModel


def _proj_grad_sum(model: PrototypeRetrievalModel) -> float:
    total = 0.0
    for p in model.proj.parameters():
        if p.grad is not None:
            total += float(p.grad.abs().sum().item())
    return total


def test_unsup_site_energy_does_not_update_projection() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.zero_grad(set_to_none=True)
    loss = model.unsupervised_site_energy(emb, site_idx)
    loss.backward()

    assert _proj_grad_sum(model) == 0.0, (
        "projection parameters received gradient from L_site_unsup "
        "(expected zero — projection is detached)"
    )

    # Site bank, site key projection, and site log-beta must all move.
    assert model.site_memory.grad is not None
    assert float(model.site_memory.grad.abs().sum().item()) > 0.0

    assert model.W_K_site.weight.grad is not None
    assert float(model.W_K_site.weight.grad.abs().sum().item()) > 0.0

    assert model.log_beta_site.grad is not None
    assert float(model.log_beta_site.grad.abs().sum().item()) > 0.0


def test_labeled_bce_does_update_projection() -> None:
    """Negative control: the labeled path must flow gradient into proj.*"""
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))
    labels = torch.zeros(8, 234)
    labels[:, 0] = 1.0

    model.zero_grad(set_to_none=True)
    out = model(emb, site_idx)
    F.binary_cross_entropy(out, labels).backward()

    assert _proj_grad_sum(model) > 0.0, (
        "projection parameters received zero gradient from labeled BCE "
        "(expected non-zero — labeled path trains the projection)"
    )
