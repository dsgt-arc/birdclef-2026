"""Sydorskyi 2025 SED-attention head — direct port (simplified).

Source: vendor/sydorskyi-2025/code_base/models/blocks.py:202-365 @ 053594f

Architecture:
    backbone features (B, C, H_freq, W_time)
        → GeMFreq pool over freq axis → (B, C, 1, W_time)
        → squeeze + permute → (B, W_time, C)
        → Dropout(p/2) + Linear(C → 512) + ReLU + Dropout(p) → (B, W_time, 512)
        → permute → (B, 512, W_time)
        → Conv1d(512, num_classes, kernel=1) for ATTENTION → tanh → time_att
        → Conv1d(512, num_classes, kernel=1) for VALUE  → feat_v (logit-scale)
        → softmax(time_att) over time
        → clipwise_logits_long = sum_t(feat_v[t] * softmax(time_att[t]))

Key differences from our existing `AttHead` in sed_head.py:
- Aggregation in LOGIT space (matches Tucker/Nikita/Sydorskyi — see
  feedback_atthead_aggregation_domain memory). Our existing AttHead aggregates in
  probability space (sigmoid before sum), which we suspect drove the AttHead
  regressions in exps 020/021.
- Hidden dim default = 512 (matches Sydorskyi). Existing AttHead defaults vary.
- p=0.5 master arg → dropout(0.25) before linear, dropout(0.5) after ReLU.

Simplifications vs blocks.py:
- Dropped `rnn_config` (None in our selected configs)
- Dropped `deep_supervision_steps` (None in our selected configs)
- Dropped `infer_period < train_period` short-window crop path
  (our config has train_period == infer_period == 5.0)
- Dropped `output_type` overrides — we always return the dict form

Confidence: VERIFIED (Phase A.2 — every parameter line-cited in
docs/sydorskyi-config-verified.md). 2026-05-28.
"""

import torch
import torch.nn as nn
from torch import Tensor

from birdclef.preprocessing.yoink_mel import GeMFreq


class YoinkAttHead(nn.Module):
    """SED attention head, Sydorskyi 2025 style.

    Args:
        in_chans: backbone output channels (256 for EffNetV2-S features_only,
            2304 for NFNet-L0). See docs/sydorskyi-config-verified.md.
        hidden_chans: dense projection dim. Default 512 (Sydorskyi-verified).
        p: master dropout rate. p/2 applied before linear; p applied after ReLU.
            Default 0.5 (Sydorskyi config).
        num_class: classifier output dim. Default 234 (BirdCLEF 2026).
            Note: Sydorskyi's default was 397 (BirdCLEF 2025).
        exportable: pass-through to GeMFreq for ONNX export compatibility.
    """

    def __init__(
        self,
        in_chans: int,
        hidden_chans: int = 512,
        p: float = 0.5,
        num_class: int = 234,
        exportable: bool = False,
        # Kept for config-API compatibility with WaveCNNAttenClasifier — unused.
        # Our configs have train_period == infer_period == 5.0, so the short-window
        # path in the original AttHead never fires.
        train_period: float = 5.0,
        infer_period: float = 5.0,
    ):
        super().__init__()
        assert train_period == infer_period, (
            f"YoinkAttHead simplification: requires train_period == infer_period "
            f"(got {train_period} != {infer_period}). For Sydorskyi configs with "
            f"train_period > infer_period, port the short-window crop path from "
            f"vendor/sydorskyi-2025/code_base/models/blocks.py:330-365."
        )

        self.train_period = train_period
        self.infer_period = infer_period

        self.pooling = GeMFreq(exportable=exportable)
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),  # 0.25
            nn.Linear(in_chans, hidden_chans),
            nn.ReLU(),
            nn.Dropout(p),  # 0.5
        )
        self.attention = nn.Conv1d(
            in_channels=hidden_chans,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels=hidden_chans,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat: Tensor) -> dict[str, Tensor]:
        """Forward pass — returns dict with both logit-space and prob-space aggregations.

        Args:
            feat: backbone feature map, shape (B, C, H_freq, W_time)

        Returns:
            dict with keys:
                clipwise_logits_long: (B, num_class) — LOGIT-space attention-weighted sum.
                    THIS is the tensor used as input to FocalLossBCE (training).
                clipwise_pred_long: (B, num_class) — PROBABILITY-space aggregation.
                    Computed as sum_t(sigmoid(feat_v[t]) * softmax(time_att[t])).
                    Used for metric tracking, NOT for loss.
                framewise_logits_long: (B, W_time, num_class) — pre-aggregation framewise logits.
        """
        # Pool over freq axis: (B, C, H_freq, W_time) → (B, C, 1, W_time)
        feat = self.pooling(feat)
        # Squeeze freq dim + permute to (B, W_time, C)
        feat = feat.squeeze(-2).permute(0, 2, 1)
        # Dense block: (B, W_time, C) → (B, W_time, hidden) → permute → (B, hidden, W_time)
        feat = self.dense_layers(feat).permute(0, 2, 1)

        # Two parallel Conv1d heads
        time_att = torch.tanh(self.attention(feat))  # (B, num_class, W_time)
        feat_v = self.fix_scale(feat)  # (B, num_class, W_time)

        # Soft-attention-weighted aggregations
        softmaxed_att = torch.softmax(time_att, dim=-1)  # (B, num_class, W_time)

        # Probability-space aggregation (for metric tracking — NOT loss)
        clipwise_pred_long = torch.sum(torch.sigmoid(feat_v) * softmaxed_att, dim=-1)

        # LOGIT-space aggregation (THIS is the loss input)
        clipwise_logits_long = torch.sum(feat_v * softmaxed_att, dim=-1)

        return {
            "clipwise_pred_long": clipwise_pred_long,
            "clipwise_logits_long": clipwise_logits_long,
            "framewise_logits_long": feat_v.permute(0, 2, 1),  # (B, W_time, num_class)
        }
