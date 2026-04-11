"""Three-term training loss for experiment 107.

`L_total = L_labeled + lambda_site * L_site_unsup + lambda_fam * L_family`

See `design.md` §Loss for the derivation and `design.md` §Training for the
locked defaults of ``lambda_site`` and ``lambda_fam``.
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
    lambda_site: float = 0.1,
    lambda_fam: float = 1e-2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the three-term total loss and return per-component diagnostics.

    Args:
        model: a ``PrototypeRetrievalModel`` in train mode.
        batch: a dict with three keys ``"strong"``, ``"weak_focal"``,
            ``"unsup"``. Each sub-dict contains:
              * ``"strong"``: ``emb`` (B, d_input), ``site_idx`` (B,),
                ``labels`` (B, n_species).
              * ``"weak_focal"``: ``emb``, ``labels``.
              * ``"unsup"``: ``emb``, ``site_idx``.
            Any sub-dict may be ``None`` to disable that stream in a given
            step (useful for smoke tests that exercise only the labeled path).
        family_idx: (n_species,) long tensor mapping species → contiguous
            family index.
        lambda_site: weight on the unsupervised Hopfield energy term.
        lambda_fam: weight on the family shrinkage term.

    Returns:
        (total_loss, components) where ``components`` is a dict with keys
        ``labeled``, ``site``, ``family`` (each a Python float for logging).
    """
    device = next(model.parameters()).device
    zero = torch.zeros((), device=device)

    components: dict[str, float] = {"labeled": 0.0, "site": 0.0, "family": 0.0}
    total = zero

    strong = batch.get("strong")
    weak_focal = batch.get("weak_focal")
    if strong is not None or weak_focal is not None:
        losses = []
        if strong is not None:
            sp_strong = model(strong["emb"], strong["site_idx"])
            losses.append(F.binary_cross_entropy(sp_strong, strong["labels"]))
        if weak_focal is not None:
            sp_weak = model(weak_focal["emb"], site_idx=None)
            losses.append(F.binary_cross_entropy(sp_weak, weak_focal["labels"]))
        L_labeled = sum(losses)
        total = total + L_labeled
        components["labeled"] = float(L_labeled.detach().item())

    unsup = batch.get("unsup")
    if unsup is not None and lambda_site != 0.0:
        L_site = model.unsupervised_site_energy(unsup["emb"], unsup["site_idx"])
        total = total + lambda_site * L_site
        components["site"] = float(L_site.detach().item())

    if lambda_fam != 0.0:
        L_family = model.family_shrinkage_loss(family_idx)
        total = total + lambda_fam * L_family
        components["family"] = float(L_family.detach().item())

    return total, components
