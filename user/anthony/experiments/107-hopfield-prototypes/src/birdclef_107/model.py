"""PrototypeRetrievalModel — flat species + site memory banks over a learned
512-d projection of Perch v2 embeddings.

Modern-Hopfield logsumexp retrieval with learnable per-head inverse
temperatures, per-species Platt calibration, and a per-window gated fusion
between species-only and species × site scores.

Family hierarchy is NOT part of the forward pass — it lives entirely in the
`L_family` training loss as a stop-grad shrinkage regularizer on
`M_species`. At inference, the model contains no family-level parameters.

See `design.md` §Model for the architectural rationale and
`docs/star-schema-dimensional-analysis.md` for the Hopfield energy
derivation and the soft-EM gradient identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange


class PrototypeRetrievalModel(nn.Module):
    """Flat species and site memory banks with modern-Hopfield logsumexp
    retrieval over a learned projection of Perch v2 embeddings.

    Total ~3.4M parameters at default shapes.
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

        self.proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
            nn.LayerNorm(d_proj),
        )

        self.W_Q = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_species = nn.Linear(d_proj, d_proj, bias=False)
        self.W_K_site = nn.Linear(d_proj, d_proj, bias=False)

        # Flat species memory (234 species, K=8 slots each). Decisions are
        # made from this bank alone. The family hierarchy is implemented via
        # the `L_family` loss at training time — no family-level parameters
        # live in the model itself.
        self.M_species = nn.Parameter(torch.zeros(n_species, K_species, d_proj))

        # Flat site memory (23 Pantanal sites S01..S23).
        self.site_memory = nn.Parameter(torch.zeros(n_sites, K_site, d_proj))

        # Learnable Hopfield inverse temperatures (one per head).
        self.log_beta_species = nn.Parameter(torch.zeros(()))
        self.log_beta_site = nn.Parameter(torch.zeros(()))

        # Per-species Platt calibration. Identity init. Load-bearing: the
        # only per-species differentiator for zero-data orphans, which
        # otherwise collide with their data-rich family siblings.
        self.cal_scale = nn.Parameter(torch.ones(n_species))
        self.cal_bias = nn.Parameter(torch.zeros(n_species))

        # Per-window gate between species-only and species × site.
        self.gate = nn.Linear(d_proj, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.M_species, std=0.02)
        nn.init.normal_(self.site_memory, std=0.02)

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        return self.proj(emb)

    def forward(
        self,
        emb: torch.Tensor,
        site_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute calibrated multi-label probabilities.

        Args:
            emb: (B, d_input) Perch embeddings.
            site_idx: (B,) long tensor of site indices, or ``None`` for focal
                windows (site-less → species-only path).

        Returns:
            (B, n_species) calibrated multi-label probabilities in [0, 1].
        """
        q = self.proj(emb)
        q_r = self.W_Q(q)

        # Species retrieval (flat — no family-level concatenation).
        beta_s = self.log_beta_species.exp()
        M_s = self.W_K_species(self.M_species)  # (s k d)
        sp_sim = einsum(q_r, M_s, "b d, s k d -> b s k")
        sp_agg = torch.logsumexp(beta_s * sp_sim, dim=-1) / beta_s  # (b s)
        sp_score = torch.sigmoid(self.cal_scale * sp_agg + self.cal_bias)

        if site_idx is None:
            return sp_score

        # Site retrieval → per-window typicality scalar.
        beta_l = self.log_beta_site.exp()
        M_l = self.W_K_site(self.site_memory[site_idx])  # (b k d)
        site_sim = einsum(q_r, M_l, "b d, b k d -> b k")
        site_agg = torch.logsumexp(beta_l * site_sim, dim=-1) / beta_l  # (b,)
        site_score = torch.sigmoid(site_agg)  # (b,)

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

        The projection is DETACHED — this loss trains only ``site_memory``,
        ``W_K_site``, and ``log_beta_site``. It does not update the
        projection, which prevents the site bank from pulling the projection
        toward site-identifying artifacts (exp 106's 'site entangled with
        domain shift' failure mode).

        Returns:
            Scalar loss: mean negative log-sum-exp retrieval energy. Pulls
            each unlabeled SC window toward SOME prototype in its known
            site's memory bank via soft-EM clustering.
        """
        q_det = self.proj(emb).detach()
        q_r = self.W_Q(q_det)
        beta_l = self.log_beta_site.exp()
        M_l = self.W_K_site(self.site_memory[site_idx])  # (b k d)
        sim = einsum(q_r, M_l, "b d, b k d -> b k")
        energy = -(torch.logsumexp(beta_l * sim, dim=-1) / beta_l)  # (b,)
        return energy.mean()

    def family_shrinkage_loss(self, family_idx: torch.Tensor) -> torch.Tensor:
        """Hierarchical-Bayes partial pooling on ``M_species``.

        At each step, compute the family centroid as a stop-grad uniform
        mean over siblings and pull each species's prototypes toward its
        centroid.

        Args:
            family_idx: (n_species,) long tensor mapping each species to a
                contiguous family index in ``[0, n_families)``.

        Returns:
            Scalar squared-Euclidean shrinkage loss to the stop-grad family
            mean.
        """
        M_species = self.M_species  # (s k d)
        n_families = int(family_idx.max().item()) + 1
        sums = torch.zeros(
            n_families,
            *M_species.shape[1:],
            device=M_species.device,
            dtype=M_species.dtype,
        )
        counts = torch.zeros(
            n_families, device=M_species.device, dtype=M_species.dtype
        )
        sums.index_add_(0, family_idx, M_species)
        counts.index_add_(
            0, family_idx, torch.ones_like(family_idx, dtype=M_species.dtype)
        )
        M_family_mean = sums / counts.clamp_min(1).view(-1, 1, 1)  # (f k d)

        # stop-grad target: the shrinkage centroid is a fixed point of view
        # for M_species at each step.
        target = M_family_mean[family_idx].detach()  # (s k d)
        return ((M_species - target) ** 2).mean()


@torch.no_grad()
def affinity_matrix(model: PrototypeRetrievalModel) -> torch.Tensor:
    """Compute the eval-time species × site affinity matrix.

    Folds the species and site memory banks into a single logsumexp over
    all ``(k_s, k_l)`` prototype pairs per ``(s, l)`` cell. Used for
    interpretation / reporting — not part of the training forward pass.

    Returns:
        (n_species, n_sites) tensor of retrieval-energy affinities.
    """
    beta = model.log_beta_species.exp()
    M_s = model.W_K_species(model.M_species)  # (s ks d)
    M_l = model.W_K_site(model.site_memory)  # (l kl d)
    pair = einsum(M_s, M_l, "s ks d, l kl d -> s l ks kl")
    flat = rearrange(pair, "s l ks kl -> s l (ks kl)")
    return torch.logsumexp(beta * flat, dim=-1) / beta  # (s l)
