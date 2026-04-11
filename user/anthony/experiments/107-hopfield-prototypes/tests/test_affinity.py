"""Phase A test 3 — affinity_matrix einsum must match a brute-force reference.

We fold every (k_s, k_l) pair into a single logsumexp per (s, l) cell. The
reference implementation uses four nested Python loops so the einsum
contraction has no place to hide a bug.
"""

from __future__ import annotations

import torch

from birdclef_107.model import PrototypeRetrievalModel, affinity_matrix


def _reference_affinity(model: PrototypeRetrievalModel) -> torch.Tensor:
    beta = float(model.log_beta_species.exp().item())
    M_s = model.W_K_species(model.M_species)  # (s ks d)
    M_l = model.W_K_site(model.site_memory)  # (l kl d)
    s, ks, d = M_s.shape
    l, kl, _ = M_l.shape
    logits = torch.empty(s, l, ks * kl, dtype=M_s.dtype, device=M_s.device)
    for si in range(s):
        for li in range(l):
            idx = 0
            for ki in range(ks):
                for lj in range(kl):
                    logits[si, li, idx] = torch.dot(M_s[si, ki], M_l[li, lj])
                    idx += 1
    return torch.logsumexp(beta * logits, dim=-1) / beta  # (s l)


def test_affinity_matrix_matches_brute_force() -> None:
    torch.manual_seed(42)
    model = PrototypeRetrievalModel(
        n_species=3,
        n_sites=2,
        K_species=2,
        K_site=2,
        d_input=8,
        d_hidden=6,
        d_proj=4,
    )
    model.eval()

    with torch.no_grad():
        got = affinity_matrix(model)
        expected = _reference_affinity(model)

    assert got.shape == (3, 2)
    max_abs_err = float((got - expected).abs().max().item())
    assert max_abs_err < 1e-5, (
        f"affinity_matrix einsum disagrees with brute-force reference: "
        f"max abs err = {max_abs_err}"
    )


def test_default_affinity_shape() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    model.eval()
    with torch.no_grad():
        a = affinity_matrix(model)
    assert a.shape == (234, 23)
    assert torch.isfinite(a).all()
