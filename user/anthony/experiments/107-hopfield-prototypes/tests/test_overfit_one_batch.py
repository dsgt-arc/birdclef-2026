"""Phase A test 4 — overfit a tiny model on one labeled batch.

Drives the labeled BCE to near-zero on a single 16-window fake batch. Any
wiring error in the Platt head, the Hopfield logsumexp, or the einsum
contraction manifests as a plateau above the threshold.
"""

from __future__ import annotations

import torch
from torch.optim import AdamW

from birdclef_107.losses import compute_loss
from birdclef_107.model import PrototypeRetrievalModel


def test_overfits_one_labeled_batch() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel(
        n_species=5,
        n_sites=2,
        K_species=2,
        K_site=2,
        d_input=32,
        d_hidden=16,
        d_proj=8,
    )
    model.train()

    emb = torch.randn(16, 32)
    site_idx = torch.randint(0, 2, (16,))
    labels = torch.zeros(16, 5)
    labels[:, 0] = 1.0

    batch = {
        "strong": {"emb": emb, "site_idx": site_idx, "labels": labels},
        "weak_focal": None,
        "unsup": None,
    }

    # Family index is unused when lambda_fam=0, but compute_loss still needs
    # a valid tensor argument if we ever flip the flag.
    family_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    optimizer = AdamW(model.parameters(), lr=1e-2)
    final_loss = float("inf")
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        total, components = compute_loss(
            model,
            batch,
            family_idx,
            lambda_site=0.0,
            lambda_fam=0.0,
        )
        total.backward()
        optimizer.step()
        final_loss = components["labeled"]

    assert final_loss < 1e-2, (
        f"labeled BCE did not converge below 1e-2 in 200 steps "
        f"(final={final_loss:.4f})"
    )
