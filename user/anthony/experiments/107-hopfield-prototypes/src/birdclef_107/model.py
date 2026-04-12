"""Prototype retrieval model, factored into four submodules.

Design:

- ``PerchProjection``: shared 1536 → 512 backbone + query head ``W_Q``.
  Every retrieval path routes through this module. Labeled BCE shapes it;
  unsupervised losses call ``query_from_detached`` to freeze it.
- ``HopfieldSpeciesHead``: flat species memory bank with per-species Platt
  calibration. Owns the ``M_species`` parameter and therefore owns
  ``family_shrinkage_loss``.
- ``HopfieldSiteHead``: flat site memory bank. Exposes two paths —
  ``forward`` (per-window sigmoid typicality used in gated fusion) and
  ``retrieval_energy`` (scalar loss used by the unsupervised training
  term).
- ``PrototypeRetrievalModel``: composes the three above with a per-window
  fusion gate. The shipped Kaggle checkpoint only needs the projection,
  species head, and gate — the site head can simply be unloaded for the
  OOD-site fallback.

See ``design.md`` §Model for the architectural rationale and
``docs/star-schema-dimensional-analysis.md`` for the Hopfield energy
derivation and the soft-EM gradient identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange


class PerchProjection(nn.Module):
    """Shared 1536 → 512 backbone + query head.

    The submodule boundary *is* the detach boundary for unsupervised
    losses: every parameter under ``model.projection`` is frozen when
    ``L_site_unsup`` is active. Make new shared-backbone parameters live
    here, not on sibling modules, so the detach guarantee stays
    declarative.
    """

    def __init__(self, d_input: int, d_hidden: int, d_proj: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
            nn.LayerNorm(d_proj),
        )
        self.W_Q = nn.Linear(d_proj, d_proj, bias=False)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project embeddings and return ``(q_base, q_query)``.

        ``q_base`` is used by the fusion gate (it needs the unprojected
        representation, not the query-space one). ``q_query`` flows into
        the species and site retrieval heads.
        """
        q = self.proj(emb)
        return q, self.W_Q(q)

    def query_from_detached(self, emb: torch.Tensor) -> torch.Tensor:
        """Stop-grad query projection for unsupervised losses.

        Runs the full backbone pass inside ``torch.no_grad()`` so neither
        ``self.proj`` nor ``self.W_Q`` receives gradient from the caller.
        The returned tensor has ``requires_grad=False``; downstream head
        parameters still receive gradient because they are new
        ``nn.Parameter`` leaves introduced after this point.

        This is a strictly stronger guarantee than detaching only
        ``self.proj(emb)`` and then running it through ``self.W_Q`` — the
        latter still routes gradient into ``W_Q.weight`` via the forward
        matmul, so ``W_Q`` would be shaped by site-clustering loss.
        Design.md §Model describes this as "projection is detached" and
        claims the unsupervised loss trains only ``site_memory``,
        ``W_K_site``, and ``log_beta_site``; making W_Q's gradient
        literally zero is what makes that claim accurate.
        """
        with torch.no_grad():
            return self.W_Q(self.proj(emb))


class HopfieldSpeciesHead(nn.Module):
    """Flat species memory bank with modern-Hopfield logsumexp retrieval.

    Per-species Platt calibration is built in: ``sigmoid(a_s * agg + b_s)``
    makes multi-label BCE gradients cleaner and lets the downstream gated
    fusion treat species scores as calibrated probabilities for the
    Bayesian ``species × site`` composition.

    Also owns ``family_shrinkage_loss`` since that loss operates on
    ``self.M`` directly.
    """

    def __init__(self, n_species: int, K: int, d_proj: int) -> None:
        super().__init__()
        self.n_species = n_species
        self.K = K
        self.W_K = nn.Linear(d_proj, d_proj, bias=False)
        self.M = nn.Parameter(torch.zeros(n_species, K, d_proj))
        self.log_beta = nn.Parameter(torch.zeros(()))
        # Per-species Platt (identity init). Load-bearing: the only
        # per-species differentiator for zero-data orphans that otherwise
        # collide with their data-rich family siblings.
        self.cal_scale = nn.Parameter(torch.ones(n_species))
        self.cal_bias = nn.Parameter(torch.zeros(n_species))
        nn.init.normal_(self.M, std=0.02)

    def forward(self, q_r: torch.Tensor) -> torch.Tensor:
        """Return calibrated species probabilities of shape ``(B, n_species)``."""
        beta = self.log_beta.exp()
        M = self.W_K(self.M)  # (s k d)
        sim = einsum(q_r, M, "b d, s k d -> b s k")
        agg = torch.logsumexp(beta * sim, dim=-1) / beta  # (b s)
        return torch.sigmoid(self.cal_scale * agg + self.cal_bias)

    def family_shrinkage_loss(self, family_idx: torch.Tensor) -> torch.Tensor:
        """Hierarchical-Bayes partial pooling on ``self.M``.

        At each step, compute the family centroid as a stop-grad uniform
        mean over siblings and pull each species's prototypes toward its
        centroid.

        Args:
            family_idx: ``(n_species,)`` long tensor mapping each species
                to a contiguous family index in ``[0, n_families)``.

        Returns:
            Scalar squared-Euclidean shrinkage loss to the stop-grad
            family mean.
        """
        M = self.M  # (s k d)
        n_families = int(family_idx.max().item()) + 1
        sums = torch.zeros(
            n_families,
            *M.shape[1:],
            device=M.device,
            dtype=M.dtype,
        )
        counts = torch.zeros(n_families, device=M.device, dtype=M.dtype)
        sums.index_add_(0, family_idx, M)
        counts.index_add_(
            0, family_idx, torch.ones_like(family_idx, dtype=M.dtype)
        )
        family_mean = sums / counts.clamp_min(1).view(-1, 1, 1)  # (f k d)

        # stop-grad target: the shrinkage centroid is a fixed point of
        # view for M at each step.
        target = family_mean[family_idx].detach()  # (s k d)
        return ((M - target) ** 2).mean()


class HopfieldSiteHead(nn.Module):
    """Flat site memory bank with two paths.

    - ``forward(q_r, site_idx)``: per-window sigmoid typicality in
      ``[0, 1]``, consumed by the gated fusion.
    - ``retrieval_energy(q_r, site_idx)``: scalar loss ``= mean(-lse /
      beta)``, consumed by the unsupervised training term.

    Both paths share the same key projection and memory bank — they
    differ only in what the caller does with the aggregated logit.
    """

    def __init__(self, n_sites: int, K: int, d_proj: int) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.K = K
        self.W_K = nn.Linear(d_proj, d_proj, bias=False)
        self.M = nn.Parameter(torch.zeros(n_sites, K, d_proj))
        self.log_beta = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.M, std=0.02)

    def _aggregate(
        self, q_r: torch.Tensor, site_idx: torch.Tensor
    ) -> torch.Tensor:
        """Common path: returns ``(B,)`` logsumexp retrieval aggregate."""
        beta = self.log_beta.exp()
        M = self.W_K(self.M[site_idx])  # (b k d)
        sim = einsum(q_r, M, "b d, b k d -> b k")
        return torch.logsumexp(beta * sim, dim=-1) / beta  # (b,)

    def forward(
        self, q_r: torch.Tensor, site_idx: torch.Tensor
    ) -> torch.Tensor:
        """Per-window site typicality scalar in ``[0, 1]``."""
        return torch.sigmoid(self._aggregate(q_r, site_idx))

    def retrieval_energy(
        self, q_r: torch.Tensor, site_idx: torch.Tensor
    ) -> torch.Tensor:
        """Bounded ``L_site_unsup`` — log-sigmoid of retrieval aggregate.

        Equivalent to ``BCE(sigmoid(agg), 1)`` — every window *should*
        match its site, so the target is all-ones.  This preserves the
        gradient direction (pulls each window toward some prototype in
        its site's bank) while bounding the loss to ``[0, ∞)`` with a
        natural saturation near 0 once the site head is confident.
        The original ``-mean(agg)`` formulation was unbounded below and
        diverged to −∞ during training, dominating ``L_labeled``.

        Does not itself detach the projection — that's the caller's
        responsibility (use ``PerchProjection.query_from_detached``).
        """
        agg = self._aggregate(q_r, site_idx)  # (B,)
        # -log(sigmoid(x)) = softplus(-x), numerically stable
        return torch.nn.functional.softplus(-agg).mean()


class PrototypeRetrievalModel(nn.Module):
    """Composition: projection + species head + site head + fusion gate.

    Total ~3.94M parameters at default shapes. Forward pass delegates
    everything to the sub-modules and only owns the gate + fusion logic.

    Family hierarchy is NOT part of the forward pass — it lives entirely
    in ``family_shrinkage_loss``, which is delegated to the species head.
    The shipped Kaggle checkpoint contains no family-level parameters.
    """

    def __init__(
        self,
        n_species: int = 234,
        n_sites: int = 23,
        K_species: int = 8,
        K_site: int = 8,
        d_input: int = 1536,
        d_hidden: int = 1024,
        d_proj: int = 512,
    ) -> None:
        super().__init__()
        self.n_species = n_species
        self.n_sites = n_sites
        self.K_species = K_species
        self.K_site = K_site
        self.d_proj = d_proj

        self.projection = PerchProjection(d_input, d_hidden, d_proj)
        self.species_head = HopfieldSpeciesHead(n_species, K_species, d_proj)
        self.site_head = HopfieldSiteHead(n_sites, K_site, d_proj)
        self.gate = nn.Linear(d_proj, 1)

    def forward(
        self,
        emb: torch.Tensor,
        site_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return calibrated multi-label probabilities of shape
        ``(B, n_species)``.

        Args:
            emb: ``(B, d_input)`` Perch embeddings.
            site_idx: ``(B,)`` long tensor of site indices, or ``None`` for
                focal windows (site-less → species-only path).
        """
        q, q_r = self.projection(emb)
        sp_score = self.species_head(q_r)

        if site_idx is None:
            return sp_score

        site_score = self.site_head(q_r, site_idx)  # (b,)

        # Gated fusion. Both ``g`` and ``site_score_col`` carry a trailing
        # singleton so they broadcast against ``sp_score`` of shape (b, s).
        g = torch.sigmoid(self.gate(q))  # (b, 1)
        site_score_col = rearrange(site_score, "b -> b 1")  # (b, 1)
        return g * sp_score + (1 - g) * sp_score * site_score_col

    def unsupervised_site_energy(
        self,
        emb: torch.Tensor,
        site_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Hopfield energy minimization on unlabeled soundscape windows.

        Routes the detach through ``PerchProjection.query_from_detached``,
        so the shared backbone (``self.projection``) receives zero
        gradient; only the site head moves.
        """
        q_r = self.projection.query_from_detached(emb)
        return self.site_head.retrieval_energy(q_r, site_idx)

    def family_shrinkage_loss(self, family_idx: torch.Tensor) -> torch.Tensor:
        """Delegates to the species head — that's where ``M_species`` lives."""
        return self.species_head.family_shrinkage_loss(family_idx)


@torch.no_grad()
def affinity_matrix(model: PrototypeRetrievalModel) -> torch.Tensor:
    """Compute the eval-time species × site affinity matrix.

    Folds every ``(k_s, k_l)`` prototype pair into a single logsumexp per
    ``(s, l)`` cell. Used for interpretation / reporting — not part of
    the training forward pass.

    Returns:
        ``(n_species, n_sites)`` tensor of retrieval-energy affinities.
    """
    sp = model.species_head
    si = model.site_head
    beta = sp.log_beta.exp()
    M_s = sp.W_K(sp.M)  # (s ks d)
    M_l = si.W_K(si.M)  # (l kl d)
    pair = einsum(M_s, M_l, "s ks d, l kl d -> s l ks kl")
    flat = rearrange(pair, "s l ks kl -> s l (ks kl)")
    return torch.logsumexp(beta * flat, dim=-1) / beta  # (s l)
