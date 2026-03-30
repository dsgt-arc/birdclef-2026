"""ProtoSSM v4 and ResidualSSM model definitions.

Cross-ref: ref_yuriy_protossm_v5.py lines 1359-1579 (ProtoSSM), 2362-2442 (ResidualSSM).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SelectiveSSM — Mamba-style state space model
# Cross-ref: ref_yuriy_protossm_v5.py lines 1359-1405
# ---------------------------------------------------------------------------


class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective state space model.

    Input-dependent (selective) discretization of continuous-time SSM.
    For T=12 bioacoustic windows, the sequential scan is efficient on CPU.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, D) -> (B, T, D)."""
        B_size, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)

        h = torch.zeros(B_size, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :]
            dA = torch.exp(A[None, :, :] * dt_t[:, :, None])
            dB = dt_t[:, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            y_t = (h * C[:, t, None, :]).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


# ---------------------------------------------------------------------------
# TemporalCrossAttention — Non-local temporal pattern capture
# Cross-ref: ref_yuriy_protossm_v5.py lines 1408-1436
# ---------------------------------------------------------------------------


class TemporalCrossAttention(nn.Module):
    """Multi-head self-attention between temporal windows.

    Captures non-local patterns (e.g., dawn chorus onset, counter-singing)
    that sequential SSM may miss. Uses pre-norm residual blocks.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, D) -> (B, T, D)."""
        # Self-attention with pre-norm
        residual = x
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# ProtoSSMv2 — Full prototypical state space model
# Cross-ref: ref_yuriy_protossm_v5.py lines 1439-1579
# ---------------------------------------------------------------------------


class ProtoSSMv2(nn.Module):
    """Prototypical State Space Model v4 with cross-attention and metadata.

    Architecture:
      1. Linear(1536 → d_model) + LayerNorm + GELU + Dropout
      2. Learnable positional encoding (1, 12, d_model)
      3. Site/hour metadata embeddings → projected to d_model
      4. Bidirectional SSM layers × n_ssm_layers
      5. Optional cross-attention layer
      6. Cosine similarity with learnable prototypes × temperature + bias
      7. Gated fusion with Perch logits: α*sim + (1-α)*perch
      8. Optional taxonomic auxiliary head
    """

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 128,
        d_state: int = 16,
        n_ssm_layers: int = 2,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.15,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_windows = n_windows

        # 1. Feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Learnable positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # 3. Metadata embeddings
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # 4. Bidirectional SSM layers
        self.ssm_fwd = nn.ModuleList()
        self.ssm_bwd = nn.ModuleList()
        self.ssm_merge = nn.ModuleList()
        self.ssm_norm = nn.ModuleList()
        for _ in range(n_ssm_layers):
            self.ssm_fwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_bwd.append(SelectiveSSM(d_model, d_state))
            self.ssm_merge.append(nn.Linear(2 * d_model, d_model))
            self.ssm_norm.append(nn.LayerNorm(d_model))
        self.ssm_drop = nn.Dropout(dropout)

        # 4b. Cross-attention after SSM
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = TemporalCrossAttention(
                d_model, n_heads=cross_attn_heads, dropout=dropout
            )

        # 5. Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))

        # 6. Per-class calibration bias
        self.class_bias = nn.Parameter(torch.zeros(n_classes))

        # 7. Per-class gated fusion with Perch logits
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

        # 8. Taxonomic auxiliary head (initialized later if needed)
        self.n_families = 0
        self.family_head = None

    def init_prototypes_from_data(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Initialize prototypes as class-mean projected embeddings."""
        with torch.no_grad():
            h = self.input_proj(embeddings)
            for c in range(self.n_classes):
                mask = labels[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def init_family_head(self, n_families: int, class_to_family: list[int]) -> None:
        """Initialize taxonomic auxiliary head."""
        self.n_families = n_families
        self.family_head = nn.Linear(self.d_model, n_families)
        self.register_buffer("class_to_family", torch.tensor(class_to_family, dtype=torch.long))

    def forward(
        self,
        emb: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Forward pass.

        Args:
            emb: (B, T, 1536) Perch embeddings
            perch_logits: (B, T, 234) Perch mapped logits
            site_ids: (B,) site indices
            hours: (B,) hour values (0-23)

        Returns:
            species_logits: (B, T, n_classes)
            family_logits: (B, n_families) or None
            h_temporal: (B, T, d_model) final SSM output
        """
        B, T, _ = emb.shape

        # Project embeddings
        h = self.input_proj(emb)
        h = h + self.pos_enc[:, :T, :]

        # Add metadata embeddings
        if site_ids is not None and hours is not None:
            s_emb = self.site_emb(site_ids)
            h_emb = self.hour_emb(hours)
            meta = self.meta_proj(torch.cat([s_emb, h_emb], dim=-1))
            h = h + meta[:, None, :]

        # Bidirectional SSM
        for fwd, bwd, merge, norm in zip(
            self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm, strict=False
        ):
            residual = h
            h_f = fwd(h)
            h_b = bwd(h.flip(1)).flip(1)
            h = merge(torch.cat([h_f, h_b], dim=-1))
            h = self.ssm_drop(h)
            h = norm(h + residual)

        # Cross-attention for non-local temporal patterns
        if self.use_cross_attn:
            h = self.cross_attn(h)

        h_temporal = h

        # Prototypical cosine similarity + class bias
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        temp = F.softplus(self.proto_temp)
        sim = torch.matmul(h_norm, p_norm.T) * temp + self.class_bias[None, None, :]

        # Gated fusion with Perch logits
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            species_logits = alpha * sim + (1 - alpha) * perch_logits
        else:
            species_logits = sim

        # Taxonomic auxiliary prediction
        family_logits = None
        if self.family_head is not None:
            h_pool = h.mean(dim=1)
            family_logits = self.family_head(h_pool)

        return species_logits, family_logits, h_temporal

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# ResidualSSM — Second-pass correction model
# Cross-ref: ref_yuriy_protossm_v5.py lines 2362-2442
# ---------------------------------------------------------------------------


class ResidualSSM(nn.Module):
    """Lightweight SSM for second-pass correction on first-pass errors.

    Takes embeddings + first-pass scores, predicts additive corrections.
    Output head is zero-initialized so corrections start small.
    """

    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 64,
        d_state: int = 8,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Project embeddings + first-pass scores
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Metadata
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)

        # Single bidirectional SSM layer (lightweight)
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)

        # Output: per-class correction (additive)
        self.output_head = nn.Linear(d_model, n_classes)

        # Initialize output near zero (corrections start small)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        emb: torch.Tensor,
        first_pass_scores: torch.Tensor,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            emb: (B, T, 1536)
            first_pass_scores: (B, T, 234) logits from first pass
            site_ids: (B,)
            hours: (B,)

        Returns:
            correction: (B, T, n_classes) additive correction
        """
        B, T, _ = emb.shape

        # Concatenate embeddings with first-pass scores
        x = torch.cat([emb, first_pass_scores], dim=-1)
        h = self.input_proj(x)

        # Add metadata
        if site_ids is not None and hours is not None:
            site_e = self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1))
            hour_e = self.hour_emb(hours.clamp(0, 23))
            meta = self.meta_proj(torch.cat([site_e, hour_e], dim=-1))
            h = h + meta.unsqueeze(1)

        h = h + self.pos_enc[:, :T, :]

        # Bidirectional SSM
        residual = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.ssm_merge(torch.cat([h_f, h_b], dim=-1))
        h = self.ssm_drop(h)
        h = self.ssm_norm(h + residual)

        # Output correction
        correction = self.output_head(h)
        return correction

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
