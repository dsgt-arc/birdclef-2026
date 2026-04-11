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


def _backbone_grad_sum(model: PrototypeRetrievalModel) -> float:
    """Sum of |grad| across the shared projection backbone.

    The submodule boundary is the detach boundary: every parameter under
    ``model.projection`` is expected to stay at zero gradient when
    ``L_site_unsup`` is the sole loss.
    """
    total = 0.0
    for p in model.projection.parameters():
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

    assert _backbone_grad_sum(model) == 0.0, (
        "projection backbone parameters received gradient from L_site_unsup "
        "(expected zero — projection is detached)"
    )

    # Site bank, site key projection, and site log-beta must all move.
    site_head = model.site_head
    assert site_head.M.grad is not None
    assert float(site_head.M.grad.abs().sum().item()) > 0.0

    assert site_head.W_K.weight.grad is not None
    assert float(site_head.W_K.weight.grad.abs().sum().item()) > 0.0

    assert site_head.log_beta.grad is not None
    assert float(site_head.log_beta.grad.abs().sum().item()) > 0.0


def test_labeled_bce_does_update_projection() -> None:
    """Negative control: the labeled path must flow gradient into the backbone."""
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))
    labels = torch.zeros(8, 234)
    labels[:, 0] = 1.0

    model.zero_grad(set_to_none=True)
    out = model(emb, site_idx)
    F.binary_cross_entropy(out, labels).backward()

    assert _backbone_grad_sum(model) > 0.0, (
        "projection backbone received zero gradient from labeled BCE "
        "(expected non-zero — labeled path trains the shared backbone)"
    )
