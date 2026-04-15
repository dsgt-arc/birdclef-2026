"""Prototype retrieval model (v3): cross-attention + contrastive unsup.

The four sub-modules own their own parameters; the top-level
``PrototypeRetrievalModel`` composes them.

- ``PerchProjection``: shared 1536 → 512 backbone + query head ``W_Q``.
  Every retrieval path routes through this module. Labeled BCE shapes
  it; unsupervised losses call ``query_from_detached`` to freeze it.
- ``HopfieldSpeciesHead``: flat species memory bank ``M_species`` with
  per-species Platt calibration. Accepts either shared prototypes
  ``(S, K, D)`` (focal / inference fallback) or per-window
  site-conditioned prototypes ``(B, S, K, D)`` (strong SC / unsup SC).
  Owns ``family_shrinkage_loss``.
- ``HopfieldSiteHead``: flat site memory bank ``M_site`` with its key
  projection and inverse temperature. No longer queried directly at
  inference — feeds cross-attention as K/V. Shaped exclusively by
  ``L_site_contrastive`` (InfoNCE over sites).
- ``SpeciesSiteCrossAttention``: per-window refinement of species
  prototypes by the current window's site bank. Zero-init out_proj so
  the module starts at identity but trains from step 1. Exposes
  ``refine(M_species, M_site) -> (L, S, K_sp, D)`` for all sites at
  once (used by the contrastive xattn loss) and a thin ``forward``
  that gathers per window.

**v3 unsupervised losses (``PrototypeRetrievalModel``):**

- ``unsupervised_site_contrastive``: InfoNCE over 23 sites on direct
  M_site retrieval. Replaces v2's ``softplus(-agg)`` which saturated
  at init. Exclusively shapes ``site_head.{M, W_K, log_β}``.
- ``unsupervised_cross_attn_contrastive``: site-level InfoNCE on
  cross-attention-refined species retrieval profiles. Replaces v2's
  ``softplus(-logsumexp_s)`` which collapsed to "match anything". With
  ``M_site`` detached inside, exclusively shapes ``cross_attn.*``.

See ``design.md`` §Model and §"Mathematical specification" for the
contrastive formulas and gradient-routing invariants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from birdclef_106.data_loading import load_species_mapping
from birdclef_107.data import ARTIFACTS_DIR, get_species_list


class PerchProjection(nn.Module):
    """Shared 1536 → 512 backbone + query head.

    The submodule boundary *is* the detach boundary for unsupervised
    losses: every parameter under ``model.projection`` is frozen when
    the unsupervised loss is active. Make new shared-backbone parameters
    live here, not on sibling modules, so the detach guarantee stays
    declarative.
    """

    def __init__(
        self, d_input: int, d_hidden: int, d_proj: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_proj),
            nn.LayerNorm(d_proj),
        )
        self.W_Q = nn.Linear(d_proj, d_proj, bias=False)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Project an embedding to its retrieval query ``q_r``.

        ``q_r`` flows into both retrieval paths (species and site). There
        is no longer a separate ``q_base`` — the v1 scalar fusion gate
        that consumed it has been removed.
        """
        return self.W_Q(self.proj(emb))

    def query_from_detached(self, emb: torch.Tensor) -> torch.Tensor:
        """Stop-grad query projection for unsupervised losses.

        Runs the full backbone pass inside ``torch.no_grad()`` so neither
        ``self.proj`` nor ``self.W_Q`` receives gradient from the caller.
        The returned tensor has ``requires_grad=False``; downstream head
        parameters still receive gradient because they are new
        ``nn.Parameter`` leaves introduced after this point.

        This is strictly stronger than detaching only ``self.proj(emb)``
        and then running it through ``self.W_Q`` — the latter still
        routes gradient into ``W_Q.weight`` via the forward matmul, so
        ``W_Q`` would be shaped by site-clustering loss. ``torch.no_grad()``
        over the whole pass closes that leak.
        """
        with torch.no_grad():
            return self.W_Q(self.proj(emb))


class HopfieldSpeciesHead(nn.Module):
    """Flat species memory bank with modern-Hopfield logsumexp retrieval.

    Per-species Platt calibration is built in: ``sigmoid(a_s * agg + b_s)``
    makes multi-label BCE gradients cleaner and is the only per-species
    differentiator for zero-data orphans that otherwise collide with
    their data-rich family siblings.

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
        # per-species differentiator for zero-data orphans.
        self.cal_scale = nn.Parameter(torch.ones(n_species))
        self.cal_bias = nn.Parameter(torch.zeros(n_species))
        nn.init.normal_(self.M, std=0.02)

    def forward(self, q_r: torch.Tensor) -> torch.Tensor:
        """Return calibrated species probabilities of shape ``(B, n_species)``
        using the shared ``M_species`` (no site conditioning)."""
        return self.forward_with_prototypes(q_r, self.M)

    def forward_with_prototypes(
        self,
        q_r: torch.Tensor,
        M_species: torch.Tensor,
    ) -> torch.Tensor:
        """Same retrieval, but with caller-supplied species prototypes.

        Args:
            q_r: ``(B, D)`` retrieval queries.
            M_species: either ``(S, K, D)`` (shared across batch — matches
                ``self.forward``) or ``(B, S, K, D)`` (per-window,
                site-conditioned — produced by ``SpeciesSiteCrossAttention``).

        Returns ``(B, n_species)`` calibrated probabilities.
        """
        beta = self.log_beta.exp()
        M = self.W_K(M_species)
        if M.dim() == 3:
            sim = einsum(q_r, M, "b d, s k d -> b s k")
        else:
            sim = einsum(q_r, M, "b d, b s k d -> b s k")
        agg = torch.logsumexp(beta * sim, dim=-1) / beta  # (b s)
        return torch.sigmoid(self.cal_scale * agg + self.cal_bias)

    def family_shrinkage_loss(self, family_idx: torch.Tensor) -> torch.Tensor:
        """Hierarchical-Bayes partial pooling on ``self.M``.

        Compute the family centroid as a stop-grad uniform mean over
        siblings and pull each species's prototypes toward its centroid.

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
            n_families, *M.shape[1:], device=M.device, dtype=M.dtype
        )
        counts = torch.zeros(n_families, device=M.device, dtype=M.dtype)
        sums.index_add_(0, family_idx, M)
        counts.index_add_(
            0, family_idx, torch.ones_like(family_idx, dtype=M.dtype)
        )
        family_mean = sums / counts.clamp_min(1).view(-1, 1, 1)  # (f k d)
        target = family_mean[family_idx].detach()  # (s k d)
        return ((M - target) ** 2).mean()


class HopfieldSiteHead(nn.Module):
    """Flat site memory bank.

    In v3 the site bank is no longer queried directly at inference — the
    forward pass uses site-conditioned species prototypes via
    ``SpeciesSiteCrossAttention``, which reads ``M_site`` as its attention
    keys/values. This head exists only to hold ``M_site``, its key
    projection, and inverse temperature. The unsupervised shaping of
    ``M_site`` lives in ``PrototypeRetrievalModel.unsupervised_site_contrastive``
    (InfoNCE over sites) — v2's ``softplus(-agg)`` formulation saturated
    at random init and was removed.
    """

    def __init__(self, n_sites: int, K: int, d_proj: int) -> None:
        super().__init__()
        self.n_sites = n_sites
        self.K = K
        self.W_K = nn.Linear(d_proj, d_proj, bias=False)
        self.M = nn.Parameter(torch.zeros(n_sites, K, d_proj))
        self.log_beta = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.M, std=0.02)


class SpeciesSiteCrossAttention(nn.Module):
    """Per-window site-conditioning of species prototypes.

    For each (species, site) pair, the species prototype is updated by a
    site-dependent additive offset computed via multi-head cross-attention:

        Q = M_species : (S, K_sp, D)   — shared across batch
        K, V = M_site : (L, K_site, D) — site prototype banks
        M_refined[l, s, k] = M_species[s, k] + MHA(M_species[s, k], M_site[l])
        M_cond[b, s, k]    = M_refined[site_idx[b], s, k]

    Attention is computed once across all L sites (not per batch element),
    so memory is ``L × S × K_sp × D`` ≈ 88 MB at default shapes. The
    site-indexed lookup then distributes the refined prototypes per
    window.

    **Zero-init out_proj.** ``attended = softmax(…) · V · W_V · 0 = 0``
    at init, so ``refined = Q`` (cross-attention is numerically identity).
    Unlike ReZero (scalar gate ``α · attended`` with ``α = 0``), this init
    **still routes gradient into out_proj**: ``dloss/d(out_proj) =
    dloss/d(attended) · (softmax · V · W_V) ≠ 0`` even when
    ``out_proj = 0``. So attention weights train from step 1.

    ReZero was tried first and failed the cold-start test here: α stuck
    at -0.009 through 1200 steps because ``dloss/d(attended) = α = 0`` —
    attended couldn't train, and α couldn't grow without meaningful
    attended. Zero-init out_proj breaks the deadlock.
    """

    def __init__(
        self, d_proj: int, n_heads: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_proj % n_heads != 0:
            raise ValueError(
                f"d_proj={d_proj} must be divisible by n_heads={n_heads}"
            )
        self.attn = nn.MultiheadAttention(
            d_proj, n_heads, batch_first=True, dropout=dropout
        )
        self.update_gate = nn.Linear(d_proj, 1)
        nn.init.zeros_(self.update_gate.weight)
        nn.init.zeros_(self.update_gate.bias)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def refine(
        self,
        M_species: torch.Tensor,  # (S, K_sp, D)
        M_site: torch.Tensor,  # (L, K_site, D)
    ) -> torch.Tensor:
        """Site-conditioned species prototypes for ALL sites, ``(L, S, K_sp, D)``.

        Shared by ``forward`` (gathers per-window via site_idx) and by
        ``unsupervised_cross_attn_contrastive`` (computes retrieval
        profiles at every site for InfoNCE).
        """
        S, K_sp, D = M_species.shape
        L = M_site.shape[0]
        Q = M_species.flatten(0, 1).unsqueeze(0).expand(L, -1, -1).contiguous()
        attended, _ = self.attn(Q, M_site, M_site)  # (L, S*K_sp, D)
        gate = torch.sigmoid(self.update_gate(Q))
        refined = Q + gate * attended  # identity at init: attended=0
        return refined.reshape(L, S, K_sp, D)

    def forward(
        self,
        M_species: torch.Tensor,  # (S, K_sp, D)
        M_site: torch.Tensor,  # (L, K_site, D)
        site_idx: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        """Return site-conditioned species prototypes ``(B, S, K_sp, D)``
        gathered by ``site_idx``. Thin wrapper over ``refine()``."""
        return self.refine(M_species, M_site)[site_idx]


class FixedMappedPerchLogitAdapter(nn.Module):
    """Zero-init side branch from fixed Perch logits into query space."""

    def __init__(
        self,
        d_proj: int,
        species_list: list[str] | None = None,
        artifacts_dir: str = ARTIFACTS_DIR,
    ) -> None:
        super().__init__()
        species_list = species_list or get_species_list(artifacts_dir)
        mapping = load_species_mapping(artifacts_dir)
        direct_map = {str(k): int(v) for k, v in mapping["direct_map"].items()}
        genus_proxy_map = {
            str(k): [int(i) for i in v]
            for k, v in mapping["genus_proxy_map"].items()
        }

        direct_idx = torch.full((len(species_list),), -1, dtype=torch.long)
        genus_species: list[int] = []
        genus_indices: list[torch.Tensor] = []
        mapped_mask = torch.zeros(len(species_list), dtype=torch.float32)
        for s_idx, species in enumerate(species_list):
            if species in direct_map:
                direct_idx[s_idx] = direct_map[species]
                mapped_mask[s_idx] = 1.0
            elif species in genus_proxy_map:
                genus_species.append(s_idx)
                genus_indices.append(
                    torch.tensor(genus_proxy_map[species], dtype=torch.long)
                )
                mapped_mask[s_idx] = 1.0

        self.register_buffer("direct_idx", direct_idx)
        self.register_buffer("mapped_mask", mapped_mask)
        self.genus_species = genus_species
        self.genus_indices = genus_indices
        self.proj = nn.Linear(len(species_list), d_proj, bias=False)
        nn.init.zeros_(self.proj.weight)

    def map_logits(self, perch_logits: torch.Tensor) -> torch.Tensor:
        perch_probs = torch.sigmoid(perch_logits)
        mapped = perch_probs.new_zeros((perch_probs.shape[0], self.direct_idx.shape[0]))
        direct_mask = self.direct_idx >= 0
        if direct_mask.any():
            mapped[:, direct_mask] = perch_probs[:, self.direct_idx[direct_mask]]
        for s_idx, perch_idx in zip(self.genus_species, self.genus_indices):
            mapped[:, s_idx] = perch_probs.index_select(
                1, perch_idx.to(perch_probs.device)
            ).amax(dim=1)
        return mapped * self.mapped_mask.to(mapped.device)

    def forward(self, perch_logits: torch.Tensor) -> torch.Tensor:
        return self.proj(self.map_logits(perch_logits))


class PrototypeRetrievalModel(nn.Module):
    """Composition: projection + species head + site head + cross-attention.

    All windows with a ``site_idx`` route species retrieval through
    ``SpeciesSiteCrossAttention``; focal windows (``site_idx=None``) fall
    back to the species-only path. The v1 scalar fusion gate was
    ablated out and does not exist — empirically it hurt by -0.045
    AUROC on S22.

    The site head no longer enters the forward pass directly; it exists
    only to hold ``M_site`` (read by cross-attention as K/V). Its
    parameters are shaped exclusively by ``unsupervised_site_contrastive``
    (InfoNCE over 23 sites) — the v1 ``retrieval_energy`` method was
    removed in v3 because its ``softplus(-agg)`` formulation saturated
    at random init.
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
        cross_attention_heads: int = 4,
        proj_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_perch_logits: bool = False,
    ) -> None:
        super().__init__()
        self.n_species = n_species
        self.n_sites = n_sites
        self.K_species = K_species
        self.K_site = K_site
        self.d_proj = d_proj

        self.projection = PerchProjection(
            d_input, d_hidden, d_proj, dropout=proj_dropout
        )
        self.species_head = HopfieldSpeciesHead(n_species, K_species, d_proj)
        self.site_head = HopfieldSiteHead(n_sites, K_site, d_proj)
        self.cross_attn = SpeciesSiteCrossAttention(
            d_proj=d_proj,
            n_heads=cross_attention_heads,
            dropout=attn_dropout,
        )
        self.perch_adapter = (
            FixedMappedPerchLogitAdapter(d_proj=d_proj, species_list=list(get_species_list()))
            if use_perch_logits
            else None
        )

    def forward(
        self,
        emb: torch.Tensor,
        site_idx: torch.Tensor | None = None,
        perch_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return calibrated multi-label probabilities of shape
        ``(B, n_species)``.

        Args:
            emb: ``(B, d_input)`` Perch embeddings.
            site_idx: ``(B,)`` long tensor of site indices, or ``None`` for
                focal recordings / OOD-site Kaggle fallback. With a site
                index the species bank is cross-attention-refined per
                window; without it, the shared species bank is used.
        """
        q_r = self.projection(emb)
        if self.perch_adapter is not None and perch_logits is not None:
            q_r = q_r + self.perch_adapter(perch_logits)

        sp_shared = self.species_head(q_r)
        if site_idx is None:
            return sp_shared

        M_cond = self.cross_attn(
            self.species_head.M, self.site_head.M, site_idx
        )  # (B, S, K, D)
        sp_cond = self.species_head.forward_with_prototypes(q_r, M_cond)
        return 0.8 * sp_cond + 0.2 * sp_shared

    def unsupervised_site_contrastive(
        self,
        emb: torch.Tensor,
        site_idx: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE over all 23 sites on the direct ``M_site`` retrieval.

        For each unlabeled SC window with known ``site_idx``, compute the
        per-window-per-site Hopfield aggregate
        ``agg[b, l] = lse_k(β · q_r[b] · W_K · M_site[l, k]) / β``
        and require the own-site aggregate to be *larger* than every
        other site's:

            L = mean( logsumexp_l(agg) - agg_self ) ≥ 0

        This replaces v2's ``softplus(-agg_self)`` bound which saturated
        at init because random prototypes already gave ``agg_self > 0``
        and the softplus derivative went to zero. InfoNCE stays
        gradient-rich: at random init all sites are ≈ equal, so the
        loss is ≈ log(L=23) ≈ 3.14 and the gradient actively pushes
        ``M_site[self]`` toward the query while pushing ``M_site[wrong]``
        away. The retrieval equation itself — ``lse_k(β · q · W_K · M)`` —
        is unchanged; only the objective over it is contrastive.

        **Gradient routing**:

        - ``projection`` — frozen via ``query_from_detached``.
        - ``site_head.{M, W_K, log_beta}`` — all trained. This is v3's
          exclusive shaping force on ``M_site``; the cross-attn
          contrastive term detaches ``M_site`` so the two unsup terms
          don't fight over the same parameter.
        - everything else (species head, cross-attn) — untouched.
        """
        q_r = self.projection.query_from_detached(emb)
        beta = self.site_head.log_beta.exp()
        # Compute retrieval aggregate for every (window, site) pair at once.
        M_keys = self.site_head.W_K(self.site_head.M)  # (L, K, D)
        sim = einsum(q_r, M_keys, "b d, l k d -> b l k")
        agg_all = torch.logsumexp(beta * sim, dim=-1) / beta  # (B, L)
        logZ = torch.logsumexp(agg_all, dim=1)  # (B,)
        agg_self = agg_all.gather(1, site_idx.unsqueeze(1)).squeeze(1)  # (B,)
        return (logZ - agg_self).mean()

    def unsupervised_cross_attn_contrastive(
        self,
        emb: torch.Tensor,
        site_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Site-InfoNCE on cross-attention-refined species retrieval.

        For each unlabeled SC window, compute the peakedness of the
        species retrieval profile under *every* site's cross-attention
        refinement, and require the own-site peakedness to exceed every
        wrong site's:

            refined[l, s, k] = cross_attn.refine(M_species, M_site)[l, s, k]
            per_sp[b, l, s]  = lse_k(β_sp · q_r[b] · W_K_sp · refined[l,s,k]) / β_sp
            score[b, l]      = lse_s(per_sp[b, l, :])         # (B, L)
            L = mean( logsumexp_l(score) - score_self )

        Replaces v2's ``softplus(-lse_s per_species)`` which collapsed
        to a "match-anything" fixed point — making one prototype stick
        for every window. The InfoNCE formulation explicitly punishes
        site-invariant profiles: if ``score[b, :]`` is constant across
        sites, the loss is ``log(L) - score_self = log(L)`` with zero
        gradient toward that collapse.

        **Gradient routing** (v3):

        - ``projection``, ``species_head.{M, W_K, log_β}`` — all
          detached (v1 + v2 invariants preserved).
        - ``site_head.M`` — **also detached** inside this loss. v3 gives
          ``L_site_contrastive`` exclusive shaping authority over
          ``M_site``; this loss trains *only* the cross-attention
          weights. One-objective-per-parameter = cleaner dynamics.
        - ``cross_attn.*`` — exclusive trainable target.

        Could recording-idx be used as the positive label instead of
        site-idx? No — ``cross_attn.refine(M_sp_det, M_site_det)`` is
        keyed only on the (detached) banks, so two windows from the
        same site (different recordings) produce byte-identical
        ``refined`` rows. Their retrieval profiles differ only via
        ``q_r``, which has no trainable path. Recording-label SupCon
        is mechanistically unsatisfiable under the detach policy. The
        recording-grouped batch sampler still adds value as variance
        reduction on the site-level positives.
        """
        q_r = self.projection.query_from_detached(emb)
        # Detach everything species-side (v1/v2 invariants).
        M_sp_frozen = self.species_head.M.detach()
        W_K_sp_frozen = self.species_head.W_K.weight.detach()
        beta_sp_frozen = self.species_head.log_beta.detach().exp()
        # Detach M_site — v3 isolates M_site to L_site_contrastive.
        M_site_frozen = self.site_head.M.detach()

        refined_all = self.cross_attn.refine(
            M_sp_frozen, M_site_frozen
        )  # (L, S, K, D)
        M_key_all = F.linear(refined_all, W_K_sp_frozen)  # (L, S, K, D)
        sim = einsum(q_r, M_key_all, "b d, l s k d -> b l s k")
        per_sp = (
            torch.logsumexp(beta_sp_frozen * sim, dim=-1) / beta_sp_frozen
        )  # (B, L, S)
        score_per_site = torch.logsumexp(per_sp, dim=-1)  # (B, L)
        logZ = torch.logsumexp(score_per_site, dim=1)  # (B,)
        score_self = score_per_site.gather(
            1, site_idx.unsqueeze(1)
        ).squeeze(1)
        return (logZ - score_self).mean()

    def family_shrinkage_loss(self, family_idx: torch.Tensor) -> torch.Tensor:
        """Delegates to the species head — that's where ``M_species`` lives."""
        return self.species_head.family_shrinkage_loss(family_idx)


@torch.no_grad()
def affinity_matrix(model: PrototypeRetrievalModel) -> torch.Tensor:
    """Compute the static species × site affinity matrix for interpretation.

    Folds every ``(k_s, k_l)`` prototype pair into a single logsumexp per
    ``(s, l)`` cell using the *pre-attention* banks. This is the v1
    affinity that appears in the ecological reporting deliverable; the
    v2 inference-time score additionally applies the learned
    cross-attention offset per window.

    Returns ``(n_species, n_sites)`` tensor.
    """
    sp = model.species_head
    si = model.site_head
    beta = sp.log_beta.exp()
    M_s = sp.W_K(sp.M)  # (s ks d)
    M_l = si.W_K(si.M)  # (l kl d)
    pair = einsum(M_s, M_l, "s ks d, l kl d -> s l ks kl")
    flat = rearrange(pair, "s l ks kl -> s l (ks kl)")
    return torch.logsumexp(beta * flat, dim=-1) / beta  # (s l)
