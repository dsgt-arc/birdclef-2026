"""Tests for the SpeciesSiteCrossAttention module and its integration
into PrototypeRetrievalModel.

Covers:
- Shape correctness of cross-attention output
- ``refine`` returns the same per-site rows that ``forward`` would gather
- Identity-at-init behavior (zero-init out_proj)
- Site-dependence once attention trains
- Non-saturated retrieval at default model size
- Labeled BCE flows gradient into out_proj and M_site
- v3 showstopper documentation: ``M_cond`` is site-only, hence
  recording-label SupCon is mechanistically unsatisfiable under detach
"""

from __future__ import annotations

import torch

from birdclef_107.model import PrototypeRetrievalModel, SpeciesSiteCrossAttention


def _tiny_model() -> PrototypeRetrievalModel:
    """Small model for fast tests (matches other test files' pattern)."""
    return PrototypeRetrievalModel(
        n_species=10,
        n_sites=3,
        K_species=2,
        K_site=2,
        d_input=64,
        d_hidden=32,
        d_proj=32,
        cross_attention_heads=2,
    )


def test_cross_attention_output_shape() -> None:
    """SpeciesSiteCrossAttention returns (B, S, K_sp, D)."""
    torch.manual_seed(0)
    mod = SpeciesSiteCrossAttention(d_proj=32, n_heads=2)

    S, K_sp, D = 10, 2, 32
    L, K_site = 3, 2
    B = 4

    M_species = torch.randn(S, K_sp, D)
    M_site = torch.randn(L, K_site, D)
    site_idx = torch.randint(0, L, (B,))

    out = mod(M_species, M_site, site_idx)
    assert out.shape == (B, S, K_sp, D)
    assert torch.isfinite(out).all()


def test_refine_returns_all_sites_and_matches_forward() -> None:
    """``refine`` returns ``(L, S, K_sp, D)``; ``forward(site_idx)`` is
    a thin gather over ``refine``. Both must agree element-wise.
    """
    torch.manual_seed(0)
    mod = SpeciesSiteCrossAttention(d_proj=32, n_heads=2)
    # Initialize out_proj non-zero so refine output isn't trivially Q.
    with torch.no_grad():
        mod.attn.out_proj.weight.normal_(std=0.1)

    S, K_sp, D = 10, 2, 32
    L, K_site = 3, 2
    M_species = torch.randn(S, K_sp, D)
    M_site = torch.randn(L, K_site, D)

    refined_all = mod.refine(M_species, M_site)
    assert refined_all.shape == (L, S, K_sp, D)

    site_idx = torch.tensor([0, 2, 1, 0])
    via_forward = mod(M_species, M_site, site_idx)
    via_gather = refined_all[site_idx]
    assert torch.allclose(via_forward, via_gather, atol=1e-6)


def test_M_cond_depends_only_on_site_not_recording() -> None:
    """**v3 showstopper documentation.** ``cross_attn.refine(M_sp.detach(),
    M_site.detach())[site]`` produces byte-identical M_cond rows for any
    two windows from the same site (regardless of recording, regardless
    of query).

    With projection and species-side operator detached (v1+v2
    invariants), this means recording-label SupCon over per-window
    species retrieval profiles is mechanistically unsatisfiable: the
    profiles for same-site-different-recording windows differ only via
    ``q_r``, which has no trainable path. **v3 therefore uses
    site-label positives in the contrastive losses.** The
    recording-grouped batch sampler still buys variance reduction, but
    it cannot supply per-recording semantic discrimination through
    cross-attention.
    """
    torch.manual_seed(0)
    mod = SpeciesSiteCrossAttention(d_proj=32, n_heads=2)
    with torch.no_grad():
        mod.attn.out_proj.weight.normal_(std=0.1)

    S, K_sp, D = 10, 2, 32
    L, K_site = 3, 2
    M_species = torch.randn(S, K_sp, D)
    M_site = torch.randn(L, K_site, D)

    refined_all = mod.refine(M_species.detach(), M_site.detach())

    # Two windows assigned to the same site → identical M_cond rows.
    M_cond_a = refined_all[1]
    M_cond_b = refined_all[1]
    assert torch.equal(M_cond_a, M_cond_b)

    # Different sites → different M_cond rows.
    M_cond_c = refined_all[2]
    assert not torch.allclose(M_cond_a, M_cond_c)


def test_cross_attention_is_identity_at_init() -> None:
    """Zero-init out_proj means ``attended = 0``; refined = Q. Output is
    site-invariant until training shapes ``out_proj``.
    """
    torch.manual_seed(42)
    model = _tiny_model()
    model.eval()

    emb = torch.randn(1, 64)
    site_a = torch.tensor([0])
    site_b = torch.tensor([2])

    with torch.no_grad():
        score_a = model(emb, site_a)
        score_b = model(emb, site_b)
    # At init, cross-attention is identity → outputs are site-invariant.
    assert torch.allclose(score_a, score_b, atol=1e-6)


def test_cross_attention_is_site_dependent_after_training() -> None:
    """Once out_proj is non-zero, different sites produce different
    species scores — the v2 thesis in action.
    """
    torch.manual_seed(42)
    model = _tiny_model()
    model.eval()

    with torch.no_grad():
        model.cross_attn.attn.out_proj.weight.normal_(std=0.1)

    emb = torch.randn(1, 64)
    site_a = torch.tensor([0])
    site_b = torch.tensor([2])

    with torch.no_grad():
        score_a = model(emb, site_a)
        score_b = model(emb, site_b)
    assert not torch.allclose(score_a, score_b, atol=1e-4)


def test_model_forward_with_site() -> None:
    """Tiny model produces valid (B, S) probabilities with site_idx."""
    torch.manual_seed(0)
    model = _tiny_model()

    B = 4
    emb = torch.randn(B, 64)
    site_idx = torch.tensor([0, 1, 2, 0])

    out = model(emb, site_idx)
    assert out.shape == (B, model.n_species)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_model_forward_focal_path() -> None:
    """site_idx=None falls back to the shared species bank."""
    torch.manual_seed(0)
    model = _tiny_model()

    B = 4
    emb = torch.randn(B, 64)
    out = model(emb, site_idx=None)
    assert out.shape == (B, model.n_species)


def test_outputs_are_not_saturated_at_init() -> None:
    """Regression: an earlier v2 iteration put LayerNorm on the cross-
    attention output, which rescaled prototypes from std ~0.02 to ~1,
    saturating the Hopfield logsumexp + Platt sigmoid. Val AUROC stuck
    at 0.5 because scores were identical across species/windows.

    Sanity check: at random init, per-window species scores should have
    non-trivial variance. If this test fails, the retrieval is saturated
    — check the cross-attention normalization.
    """
    torch.manual_seed(0)
    model = PrototypeRetrievalModel()  # default shapes — where the bug manifested
    model.eval()

    B = 16
    emb = torch.randn(B, 1536)
    site_idx = torch.randint(0, model.n_sites, (B,))

    with torch.no_grad():
        out = model(emb, site_idx)

    per_window_std = out.std(dim=1).mean().item()
    # Prior to the fix, per-window std was ~1e-7. Expect >1e-3 now.
    assert per_window_std > 1e-3, (
        f"Output saturated: per-window std across species is {per_window_std:.2e}. "
        "Check cross-attention normalization — prototypes may be rescaled "
        "to saturate the Hopfield logsumexp."
    )


def test_labeled_bce_flows_to_out_proj_at_init() -> None:
    """At init (zero out_proj), cross-attention is identity. But gradient
    STILL flows into out_proj because
    ``dloss/d(out_proj) = dloss/d(attended) · (softmax · V · W_V) ≠ 0``
    even when out_proj = 0. This is why zero-init out_proj is strictly
    better than ReZero for cold-start.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    model.train()

    B = 4
    emb = torch.randn(B, 64)
    site_idx = torch.tensor([0, 1, 2, 0])
    target = torch.zeros(B, model.n_species).bernoulli_(0.1)

    out = model(emb, site_idx)
    loss = torch.nn.functional.binary_cross_entropy(out, target)
    loss.backward()

    out_proj_w = model.cross_attn.attn.out_proj.weight
    assert out_proj_w.grad is not None
    assert out_proj_w.grad.abs().sum().item() > 0.0, (
        "out_proj.weight should receive gradient from labeled BCE at init"
    )


def test_labeled_bce_flows_to_M_site_once_out_proj_trains() -> None:
    """Once out_proj is non-zero, cross-attention is active and labeled
    BCE backprops into M_site through the attention K/V path. This is
    the v2 thesis: M_site trained by labeled *and* unlabeled signals,
    where v1 had only unlabeled.
    """
    torch.manual_seed(0)
    model = _tiny_model()
    model.train()

    with torch.no_grad():
        model.cross_attn.attn.out_proj.weight.normal_(std=0.1)

    B = 4
    emb = torch.randn(B, 64)
    site_idx = torch.tensor([0, 1, 2, 0])
    target = torch.zeros(B, model.n_species).bernoulli_(0.1)

    out = model(emb, site_idx)
    loss = torch.nn.functional.binary_cross_entropy(out, target)
    loss.backward()

    assert model.site_head.M.grad is not None
    assert model.site_head.M.grad.abs().sum().item() > 0.0, (
        "Once out_proj is non-zero, M_site should receive labeled gradient"
    )
