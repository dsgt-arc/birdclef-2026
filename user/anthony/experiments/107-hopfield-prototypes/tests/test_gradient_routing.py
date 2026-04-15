"""Gradient-routing invariants for v3 unsupervised losses.

Four invariants — see design.md §Gradient routing.

1. **Projection frozen** under both unsupervised losses. (v1 invariant —
   prevents site-clustering loss from re-encoding mic/ambient features
   in the shared backbone, the exp 106 failure mode.)

2. **Species retrieval operator frozen** under L_xattn_contrastive.
   (v2 invariant — `M_species`, species `W_K`, species `log_β` are
   detached inside the cross-attn unsupervised path so the species
   bank stays BCE-shaped.)

3. **M_site frozen under L_xattn_contrastive** (NEW v3 invariant). v3
   gives `L_site_contrastive` exclusive shaping authority over
   `M_site` — the cross-attn unsup term detaches `M_site` so the two
   unsup terms don't fight over the same parameter. One objective per
   parameter.

4. **Cross-attn weights** are the exclusive trainable target of
   `L_xattn_contrastive`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from birdclef_107.model import PrototypeRetrievalModel


def _param_grad_sum(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.abs().sum().item())
    return total


# ---------------------------------------------------------------------------
# Invariant 1: projection frozen under both unsupervised losses
# ---------------------------------------------------------------------------


def test_unsup_site_contrastive_does_not_update_projection() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.zero_grad(set_to_none=True)
    loss = model.unsupervised_site_contrastive(emb, site_idx)
    loss.backward()

    assert _param_grad_sum(model.projection.parameters()) == 0.0, (
        "projection received gradient from L_site_contrastive — "
        "the query_from_detached invariant leaks."
    )

    # Site-side params are the exclusive trainable target.
    site_head = model.site_head
    assert float(site_head.M.grad.abs().sum().item()) > 0.0
    assert float(site_head.W_K.weight.grad.abs().sum().item()) > 0.0
    assert float(site_head.log_beta.grad.abs().sum().item()) > 0.0


def test_unsup_cross_attn_contrastive_does_not_update_projection() -> None:
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.zero_grad(set_to_none=True)
    loss = model.unsupervised_cross_attn_contrastive(emb, site_idx)
    loss.backward()

    assert _param_grad_sum(model.projection.parameters()) == 0.0, (
        "projection received gradient from L_xattn_contrastive — "
        "the query_from_detached invariant leaks."
    )


# ---------------------------------------------------------------------------
# Invariant 2: species retrieval operator frozen under L_xattn_contrastive
# ---------------------------------------------------------------------------


def test_unsup_cross_attn_contrastive_does_not_update_species_retrieval_ops() -> None:
    """``M_species``, ``species_head.W_K.weight``, ``species_head.log_beta``
    are explicitly ``.detach()``-ed inside the xattn contrastive loss.

    This is the v2 analog of the v1 "projection not trained by site
    loss" invariant, applied to the species-facing params.
    """
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.zero_grad(set_to_none=True)
    loss = model.unsupervised_cross_attn_contrastive(emb, site_idx)
    loss.backward()

    sp = model.species_head
    assert sp.M.grad is None or float(sp.M.grad.abs().sum().item()) == 0.0, (
        "M_species received unsupervised gradient (expected zero — detached)"
    )
    assert (
        sp.W_K.weight.grad is None
        or float(sp.W_K.weight.grad.abs().sum().item()) == 0.0
    ), "species W_K received unsupervised gradient (expected zero — detached)"
    assert (
        sp.log_beta.grad is None
        or float(sp.log_beta.grad.abs().sum().item()) == 0.0
    ), "species log_beta received unsupervised gradient (expected zero — detached)"


# ---------------------------------------------------------------------------
# Invariant 3: M_site frozen under L_xattn_contrastive (v3 NEW)
# ---------------------------------------------------------------------------


def test_unsup_cross_attn_contrastive_does_not_update_M_site() -> None:
    """v3-only invariant: the xattn contrastive loss detaches M_site
    inside, so L_site_contrastive has exclusive shaping authority over
    M_site. Test even with non-zero out_proj (post step-1 unlock) to
    confirm the detach holds throughout training.
    """
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    # Pre-populate out_proj so the K/V gradient path through M_site
    # would normally be active (i.e., past the zero-init unlock).
    with torch.no_grad():
        model.cross_attn.attn.out_proj.weight.normal_(std=0.1)

    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.zero_grad(set_to_none=True)
    loss = model.unsupervised_cross_attn_contrastive(emb, site_idx)
    loss.backward()

    assert (
        model.site_head.M.grad is None
        or float(model.site_head.M.grad.abs().sum().item()) == 0.0
    ), (
        "M_site received gradient from L_xattn_contrastive (expected zero — "
        "v3 detaches M_site inside the xattn loss; L_site_contrastive is the "
        "exclusive shaping authority)."
    )


# ---------------------------------------------------------------------------
# Invariant 4: cross-attn weights ARE trained by L_xattn_contrastive
# ---------------------------------------------------------------------------


def test_unsup_cross_attn_contrastive_trains_out_proj_at_init() -> None:
    """Zero-init out_proj means out_proj receives gradient from step 1.
    ``d_loss/d_out_proj.weight = d_loss/d_attended · (softmax · V)`` is
    nonzero even when out_proj.weight = 0 (the term doesn't involve
    out_proj itself).
    """
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.zero_grad(set_to_none=True)
    loss = model.unsupervised_cross_attn_contrastive(emb, site_idx)
    loss.backward()

    out_proj_w = model.cross_attn.attn.out_proj.weight
    assert out_proj_w.grad is not None
    assert float(out_proj_w.grad.abs().sum().item()) > 0.0, (
        "cross_attn.out_proj.weight did not receive contrastive gradient "
        "at init — attention is deadlocked."
    )


def test_unsup_cross_attn_contrastive_trains_in_proj_after_unlock() -> None:
    """After ``out_proj.weight`` is non-zero (simulating step 2+),
    gradient propagates into ``in_proj_weight`` (Q/K/V projections).
    M_site stays detached so does NOT receive gradient — that's
    invariant 3 above; this test only checks in_proj.
    """
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    with torch.no_grad():
        model.cross_attn.attn.out_proj.weight.normal_(std=0.1)

    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))

    model.zero_grad(set_to_none=True)
    loss = model.unsupervised_cross_attn_contrastive(emb, site_idx)
    loss.backward()

    ca = model.cross_attn.attn
    assert ca.in_proj_weight.grad is not None
    assert float(ca.in_proj_weight.grad.abs().sum().item()) > 0.0, (
        "cross_attn.in_proj_weight did not receive contrastive gradient "
        "with non-zero out_proj"
    )


# ---------------------------------------------------------------------------
# v3 sanity: contrastive losses are nonzero at init (not the v2 saturation bug)
# ---------------------------------------------------------------------------


def test_site_contrastive_nonzero_at_init() -> None:
    """At random init all 23 sites look ≈ equal under the per-window
    aggregate, so InfoNCE ≈ log(L) ≈ 3.14. The v2 ``softplus(-agg)``
    saturated to 0 here — this test guards against regressing to that.
    """
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(64, 1536)
    site_idx = torch.randint(0, 23, (64,))

    with torch.no_grad():
        loss = model.unsupervised_site_contrastive(emb, site_idx)

    expected = torch.log(torch.tensor(23.0)).item()
    # Loose bracket: random init similarities give some site-spread, so
    # the actual loss may be 1.5–3.5. Crucially, it must not be ≈0.
    assert 0.5 < float(loss) < expected + 1.0, (
        f"L_site_contrastive at init = {float(loss):.3f}; "
        f"expected ≈ log(23) ≈ {expected:.3f}. "
        "If ≈0, the saturation bug from v2 has regressed."
    )


def test_xattn_contrastive_nonzero_at_init() -> None:
    """Same idea for the cross-attn contrastive: at zero-init out_proj,
    refined[l] is identical across l, so score_per_site is constant
    over l, so loss = log(L) - score_self = log(L) ≈ 3.14 (constant
    score gather equals the constant value).
    """
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(64, 1536)
    site_idx = torch.randint(0, 23, (64,))

    with torch.no_grad():
        loss = model.unsupervised_cross_attn_contrastive(emb, site_idx)

    expected = torch.log(torch.tensor(23.0)).item()
    assert abs(float(loss) - expected) < 0.01, (
        f"L_xattn_contrastive at zero-init out_proj = {float(loss):.3f}; "
        f"expected ≈ log(23) ≈ {expected:.3f} (refined identical across "
        "sites → score uniform → loss = log(L) exactly)."
    )


# ---------------------------------------------------------------------------
# Negative control: labeled BCE still shapes the backbone
# ---------------------------------------------------------------------------


def test_labeled_bce_does_update_projection() -> None:
    """Sanity check: the labeled path flows gradient into the backbone."""
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()
    emb = torch.randn(8, 1536)
    site_idx = torch.randint(0, 23, (8,))
    labels = torch.zeros(8, 234)
    labels[:, 0] = 1.0

    model.zero_grad(set_to_none=True)
    out = model(emb, site_idx)
    F.binary_cross_entropy(out, labels).backward()

    assert _param_grad_sum(model.projection.parameters()) > 0.0, (
        "projection received zero gradient from labeled BCE "
        "(expected non-zero — labeled path trains the shared backbone)"
    )
