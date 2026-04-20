"""Four-component training loss for experiment 107 (v3).

``L_total = L_labeled + λ_site · L_site + λ_xattn · L_xattn + λ_fam · L_family``

Four terms, each with a clear trainable target:

- **``L_labeled``**: strong SC + weak focal BCE. Shapes projection,
  species bank, Platt, cross-attention (via strong SC cross-attn path).
- **``L_site``** (``unsupervised_site_contrastive``): InfoNCE over 23
  sites on direct M_site retrieval. Replaces v2's saturating
  ``softplus(-agg)``. Exclusively shapes ``site_head.{M, W_K, log_β}``.
- **``L_xattn``** (``unsupervised_cross_attn_contrastive``): site-level
  InfoNCE on cross-attention-refined species retrieval profiles.
  Replaces v2's collapsing ``softplus(-logsumexp_s)``. With ``M_site``
  detached inside, exclusively shapes ``cross_attn.*``.
- **``L_family``**: hierarchical-Bayes partial pooling on ``M_species``.

See ``design.md`` §Loss for the derivation and gradient-routing
invariants, and the v3 concept note for why recording-level positives
are mechanistically unsatisfiable under the detach policy (hence
site-level positives for the contrastive terms).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F

from birdclef_107.model import PrototypeRetrievalModel


def compute_loss(
    model: PrototypeRetrievalModel,
    batch: Mapping[str, Mapping[str, Any]],
    family_idx: torch.Tensor,
    lambda_site: float = 0.01,
    lambda_xattn: float = 0.01,
    lambda_fam: float = 1e-2,
    lambda_presence: float = 0.0,
    label_smoothing: float = 0.0,
    include_strong: bool = True,
    strong_site_dropout: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the four-term total loss and return per-component diagnostics.

    Args:
        model: a ``PrototypeRetrievalModel`` in train mode.
        batch: a dict with three keys ``"strong"``, ``"weak_focal"``,
            ``"unsup"``. Each sub-dict contains:
              * ``"strong"``: ``emb`` (B, d_input), ``site_idx`` (B,),
                ``labels`` (B, n_species).
              * ``"weak_focal"``: ``emb``, ``labels``.
              * ``"unsup"``: ``emb``, ``site_idx``.
            Any sub-dict may be ``None`` to disable that stream.
        family_idx: (n_species,) long tensor mapping species → contiguous
            family index.
        lambda_site: weight on the direct-M_site InfoNCE term.
        lambda_xattn: weight on the cross-attention-refined InfoNCE term.
        lambda_fam: weight on the family shrinkage term.
        label_smoothing: BCE label smoothing ε applied to strong + focal
            targets as ``y * (1-ε) + 0.5 * ε``. Default 0.0 (disabled).
        lambda_presence: weight on the Perch-guided weak-focal presence /
            no-call auxiliary. Target = max mapped Perch support over the
            recording's weak labels.
        include_strong: if False, the strong SC BCE term is skipped
            regardless of whether ``batch["strong"]`` is present. Used by
            the training loop to rate-limit strong SC (v4 strong_stride).
        strong_site_dropout: probability of routing a strong labeled
            soundscape window through the site-free species path
            (``site_idx=None``) instead of the site-conditioned path.
            Acts only on the strong stream; focal and unsup are unchanged.

    Returns:
        ``(total_loss, components)`` where ``components`` logs each
        sub-term as a Python float: ``labeled``, ``site``, ``site_xattn``
        (cross-attention contrastive), ``family``. Component dict keys
        preserved from v2 for TensorBoard continuity.
    """
    device = next(model.parameters()).device
    zero = torch.zeros((), device=device)

    components: dict[str, float] = {
        "labeled": 0.0,
        "site": 0.0,
        "site_xattn": 0.0,
        "family": 0.0,
        "presence": 0.0,
    }
    total = zero

    strong = batch.get("strong") if include_strong else None
    weak_focal = batch.get("weak_focal")

    def _smooth(y: torch.Tensor) -> torch.Tensor:
        if label_smoothing == 0.0:
            return y
        return y * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if strong is not None or weak_focal is not None:
        losses = []
        if strong is not None:
            if strong_site_dropout > 0.0:
                drop_mask = (
                    torch.rand(strong["site_idx"].shape[0], device=device)
                    < strong_site_dropout
                )
                sp_strong = torch.empty_like(strong["labels"], dtype=torch.float32)
                keep_mask = ~drop_mask
                if keep_mask.any():
                    keep_kwargs = {}
                    for key in ("perch_mapped", "perch_topk_idx", "perch_topk_val"):
                        val = strong.get(key)
                        if val is not None:
                            keep_kwargs[key] = val[keep_mask]
                    sp_strong[keep_mask] = model(
                        strong["emb"][keep_mask],
                        strong["site_idx"][keep_mask],
                        **keep_kwargs,
                    )
                if drop_mask.any():
                    drop_kwargs = {}
                    for key in ("perch_mapped", "perch_topk_idx", "perch_topk_val"):
                        val = strong.get(key)
                        if val is not None:
                            drop_kwargs[key] = val[drop_mask]
                    sp_strong[drop_mask] = model(
                        strong["emb"][drop_mask], site_idx=None, **drop_kwargs
                    )
            else:
                sp_strong = model(
                    strong["emb"],
                    strong["site_idx"],
                    perch_mapped=strong.get("perch_mapped"),
                    perch_topk_idx=strong.get("perch_topk_idx"),
                    perch_topk_val=strong.get("perch_topk_val"),
                )
            losses.append(
                F.binary_cross_entropy(sp_strong, _smooth(strong["labels"]))
            )
        if weak_focal is not None:
            sp_weak = model(
                weak_focal["emb"],
                site_idx=None,
                perch_mapped=weak_focal.get("perch_mapped"),
                perch_topk_idx=weak_focal.get("perch_topk_idx"),
                perch_topk_val=weak_focal.get("perch_topk_val"),
            )
            losses.append(
                F.binary_cross_entropy(sp_weak, _smooth(weak_focal["labels"]))
            )
        L_labeled = sum(losses)
        total = total + L_labeled
        components["labeled"] = float(L_labeled.detach().item())

        if (
            weak_focal is not None
            and lambda_presence != 0.0
            and getattr(model, "perch_presence_bias", False)
            and weak_focal.get("perch_mapped") is not None
        ):
            presence_pred = model.presence_probability(
                weak_focal["emb"],
                perch_mapped=weak_focal.get("perch_mapped"),
                perch_topk_idx=weak_focal.get("perch_topk_idx"),
                perch_topk_val=weak_focal.get("perch_topk_val"),
            )
            presence_target = (
                weak_focal["perch_mapped"] * weak_focal["labels"]
            ).amax(dim=1, keepdim=True).detach()
            L_presence = F.binary_cross_entropy(presence_pred, presence_target)
            total = total + lambda_presence * L_presence
            components["presence"] = float(L_presence.detach().item())

    unsup = batch.get("unsup")
    if unsup is not None:
        if lambda_site != 0.0:
            L_site = model.unsupervised_site_contrastive(
                unsup["emb"], unsup["site_idx"]
            )
            total = total + lambda_site * L_site
            components["site"] = float(L_site.detach().item())
        if lambda_xattn != 0.0:
            L_xattn = model.unsupervised_cross_attn_contrastive(
                unsup["emb"], unsup["site_idx"]
            )
            total = total + lambda_xattn * L_xattn
            components["site_xattn"] = float(L_xattn.detach().item())

    if lambda_fam != 0.0:
        L_family = model.family_shrinkage_loss(family_idx)
        total = total + lambda_fam * L_family
        components["family"] = float(L_family.detach().item())

    return total, components
