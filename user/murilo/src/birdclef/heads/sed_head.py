"""Sound Event Detection (SED) heads with attention-based temporal pooling.

Three head variants:

AttHead (v1/v4a, exps 016-021 Phase C):
  backbone features (B, C, F, T)
    → GeM pool frequency axis → (B, C, T)
    → dense projection + dropout → (B, hidden, T)
    → framewise logits via Conv1d → (B, num_classes, T)
    → attention weights via Conv1d + softmax → (B, num_classes, T)
    → clipwise = attention-weighted sum of raw logits → (B, num_classes)
  Note: differs from competition-standard (Tattaka/BirdCLEF 2023 5th) in two ways:
    1. No tanh before softmax (ours: softmax(x), Tattaka: softmax(tanh(x)))
    2. Logit-space clipwise (ours: sum(att*logits), Tattaka: sum(att*sigmoid(logits)))
  See evgeniimaslov2/birdclef-5th-place-code for reference implementation.

AttBlockV2 (v4c, exp 017):
  Same GeM + projection, but:
    → attention weights via Conv1d + tanh(x) * softmax(x) (PANNs-style)
    → framewise logits via Conv1d + sigmoid (bounded [0,1])
    → clipwise = attention-weighted sum of sigmoid framewise
  PANNs design (Kong et al. 2020). Note: PANNs uses tanh(x)*softmax(x),
  while Tattaka uses softmax(tanh(x)) — these are different operations.
  Returns logits (inverse-sigmoid) for compatibility with dual CE loss.

LSEHead (v5d+, exps 019-021):
  backbone features (B, C, F, T)
    → freq pool (GeM or mean) → (B, C, T)
    → Linear + ReLU + Dropout + Linear → (B, T, num_classes)
    → LSE temporal pooling → (B, num_classes)
  Adapted from ttahara's HGNetV2-B0 recipe. Currently our best-performing
  head on LB (v5j: 0.906 LB with HGNetV2-B0).

AttHeadMultiContext (v7+, exp 021 Phase D):
  Competition-standard attention head from BirdCLEF 2023 5th place (Tattaka).
  Key differences from AttHead:
    → attention: softmax(tanh(x)) — tanh bounds inputs before softmax
    → clipwise: sum(sigmoid(logits) * att_weights) — probability-space aggregation
    → train_period/infer_period: trains on long windows (e.g. 15s), infers on
      center crop (e.g. 5s). This IS the multi-context mechanism — the backbone
      sees full temporal context but predictions are for the center segment only.
    → pre-dropout before linear (Dropout(p/2))
  Ref: evgeniimaslov2/birdclef-5th-place-code, hengck23's 0.896 CNN-only approach.

Training uses dual supervision on clipwise and framewise-max predictions.
Inference blends clipwise and framewise-max (configurable ratio).
"""

import math

import torch
import torch.nn as nn


class GeM(nn.Module):
    """Generalized Mean Pooling on the frequency dimension.

    Collapses the second-to-last dimension (frequency) with a learnable power
    parameter p. At p=1 this is average pooling; as p→∞ it approaches max pooling.

    Args:
        p: Initial power parameter (default: 3.0).
        eps: Small constant for numerical stability.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool frequency dimension: (B, C, F, T) → (B, C, T)."""
        x_clamped = x.clamp(min=self.eps)
        return x_clamped.pow(self.p).mean(dim=-2).pow(1.0 / self.p)


class AttHead(nn.Module):
    """SED attention head with GeM frequency pooling and dual output.

    Produces two sets of logits for dual supervision:
    - clipwise: attention-weighted temporal aggregation (captures sustained calls)
    - framewise: per-frame predictions (framewise-max captures brief vocalizations)

    Args:
        in_channels: Number of input channels from backbone (e.g., 1280 for EfficientNet-B0).
        num_classes: Number of output classes (e.g., 234).
        hidden_dim: Hidden projection dimension (default: 512).
        dropout: Dropout rate for projection layer (default: 0.5).
        gem_p: Initial GeM power parameter (default: 3.0).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.gem = GeM(p=gem_p)
        self.project = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.frame_logits_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.attention_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass from backbone features to dual logits.

        Args:
            features: (B, C, F, T) feature map from backbone.

        Returns:
            clipwise_logits: (B, num_classes) attention-weighted temporal sum.
            framewise_logits: (B, num_classes, T) per-frame predictions.
        """
        # GeM pool frequency: (B, C, F, T) → (B, C, T)
        x = self.gem(features)

        # Dense projection: (B, C, T) → permute → Linear → permute → (B, hidden, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.project(x)  # (B, T, hidden)
        x = x.permute(0, 2, 1)  # (B, hidden, T)

        # Per-frame logits and attention weights
        frame_logits = self.frame_logits_conv(x)  # (B, num_classes, T)
        att_weights = torch.softmax(self.attention_conv(x), dim=-1)  # (B, num_classes, T)

        # Clipwise: attention-weighted sum of frame logits over time
        clipwise_logits = (att_weights * frame_logits).sum(dim=-1)  # (B, num_classes)

        return clipwise_logits, frame_logits


class AttHeadMultiContext(nn.Module):
    """Competition-standard attention head with multi-context train/infer cropping.

    Adapted from BirdCLEF 2023 5th place (Tattaka/evgeniimaslov2). Key design:
    - Trains on long windows (e.g. 15s) for temporal context
    - Infers on center crop (e.g. 5s) for evaluation alignment
    - Attention uses softmax(tanh(x)) for bounded inputs
    - Clipwise aggregates in probability space: sum(sigmoid(logits) * att_weights)

    The train_period/infer_period mechanism is the multi-context approach described
    by hengck23 (0.896 LB CNN-only) and Tom (6th place, 4x5sec). The backbone
    sees full temporal context but predictions are for the center segment only.

    Args:
        in_channels: Number of input channels from backbone.
        num_classes: Number of output classes.
        hidden_dim: Hidden projection dimension (default: 512).
        dropout: Dropout rate for post-linear layer (default: 0.5).
        gem_p: Initial GeM power parameter (default: 3.0).
        train_period: Training window duration in seconds (default: 15.0).
        infer_period: Inference window duration in seconds (default: 5.0).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        gem_p: float = 3.0,
        train_period: float = 15.0,
        infer_period: float = 5.0,
    ):
        super().__init__()
        assert (
            train_period >= infer_period
        ), f"train_period ({train_period}) must be >= infer_period ({infer_period})"
        self.train_period = train_period
        self.infer_period = infer_period
        self.gem = GeM(p=gem_p)
        self.project = nn.Sequential(
            nn.Dropout(dropout / 2),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.frame_logits_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.attention_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with multi-context temporal cropping.

        During training: processes full temporal extent (e.g. 15s).
        During inference: crops to center infer_period (e.g. 5s).

        Args:
            features: (B, C, F, T) feature map from backbone.

        Returns:
            clipwise_logits: (B, num_classes) — logits (inverse-sigmoid of
                probability-space aggregation) for loss compatibility.
            framewise_logits: (B, num_classes, T) — per-frame raw logits.
                T matches full temporal extent during training, or center crop
                during inference.
        """
        # GeM pool frequency: (B, C, F, T) → (B, C, T)
        x = self.gem(features)

        # Dense projection: (B, C, T) → (B, hidden, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.project(x)  # (B, T, hidden)
        x = x.permute(0, 2, 1)  # (B, hidden, T)

        # Per-frame logits and attention (full temporal extent)
        frame_logits = self.frame_logits_conv(x)  # (B, num_classes, T)
        att_raw = torch.tanh(self.attention_conv(x))  # (B, num_classes, T)

        # Crop to center infer_period during inference
        if not self.training and self.train_period > self.infer_period:
            feat_time = x.size(-1)
            ratio = self.infer_period / self.train_period
            start = int(feat_time / 2 - feat_time * ratio / 2)
            end = int(start + feat_time * ratio)
            frame_logits = frame_logits[:, :, start:end]
            att_raw = att_raw[:, :, start:end]

        # Attention-weighted sum in probability space
        att_weights = torch.softmax(att_raw, dim=-1)  # (B, num_classes, T)
        frame_probs = torch.sigmoid(frame_logits)  # (B, num_classes, T)
        clipwise_probs = (att_weights * frame_probs).sum(dim=-1)  # (B, num_classes)

        # Convert to logits for loss compatibility (BCE/CE expect logits)
        clipwise_logits = torch.logit(clipwise_probs.clamp(1e-7, 1.0 - 1e-7))

        return clipwise_logits, frame_logits


class AttBlockV2(nn.Module):
    """SED attention head with tanh+softmax attention and sigmoid classification.

    Competition-standard design from PANNs (Kong et al. 2020), used by Tom's
    experiments and multiple BirdCLEF top solutions. Key differences from AttHead:
    - Attention: tanh(value) * softmax(weight) — bounded attention values
    - Classification: sigmoid(framewise) — bounded [0,1] probabilities per frame
    - Clipwise: attention-weighted sum of sigmoid probabilities

    Returns logits (inverse-sigmoid of clipwise, raw framewise) so the existing
    dual CE loss in SEDFinetuneModule works without changes.

    Args:
        in_channels: Number of input channels from backbone.
        num_classes: Number of output classes.
        hidden_dim: Hidden projection dimension (default: 512).
        dropout: Dropout rate for projection layer (default: 0.5).
        gem_p: Initial GeM power parameter (default: 3.0).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        gem_p: float = 3.0,
    ):
        super().__init__()
        self.gem = GeM(p=gem_p)
        self.project = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.frame_logits_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)
        self.att_conv = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with tanh+softmax attention.

        Args:
            features: (B, C, F, T) feature map from backbone.

        Returns:
            clipwise_logits: (B, num_classes) — logit-space for CE loss compatibility.
            framewise_logits: (B, num_classes, T) — raw logits before sigmoid.
        """
        # GeM pool frequency: (B, C, F, T) → (B, C, T)
        x = self.gem(features)

        # Dense projection
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.project(x)  # (B, T, hidden)
        x = x.permute(0, 2, 1)  # (B, hidden, T)

        # Framewise logits and attention
        frame_logits = self.frame_logits_conv(x)  # (B, num_classes, T)
        att_raw = self.att_conv(x)  # (B, num_classes, T)

        # AttBlockV2: tanh(value) * softmax(weight)
        att_weights = torch.tanh(att_raw) * torch.softmax(att_raw, dim=-1)

        # Clipwise: attention-weighted sum of sigmoid(frame_logits)
        frame_probs = torch.sigmoid(frame_logits)  # (B, num_classes, T)
        clipwise_probs = (att_weights * frame_probs).sum(dim=-1)  # (B, num_classes)

        # Convert clipwise probs back to logits for CE loss compatibility
        clipwise_logits = torch.logit(clipwise_probs.clamp(1e-7, 1.0 - 1e-7))

        return clipwise_logits, frame_logits


class LSEPooling(nn.Module):
    """Log-Sum-Exp temporal pooling — smooth approximation to max.

    T→0: hard max, T→∞: mean. T=1.0 is a balanced default.
    Cross-ref: ttahara-hgnetv2-b0-training.py lines 633-646.
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, C)."""
        return self.temperature * (
            torch.logsumexp(x / self.temperature, dim=1) - math.log(x.shape[1])
        )


class LSEHead(nn.Module):
    """SED head with Log-Sum-Exp temporal pooling.

    Architecture: freq_pool(freq) → permute → Linear+ReLU+Dropout+Linear → LSEPool(time).
    Returns (clipwise_logits, framewise_logits) tuple matching AttHead contract.

    Args:
        freq_pool: Frequency pooling method. "gem" (default, learnable power-mean)
            or "mean" (simple average, matching ttahara's implementation).

    Cross-ref: ttahara-hgnetv2-b0-training.py lines 648-701.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        gem_p: float = 3.0,
        lse_temperature: float = 1.0,
        freq_pool: str = "gem",
    ):
        super().__init__()
        self.freq_pool_type = freq_pool
        if freq_pool == "gem":
            self.gem = GeM(p=gem_p)
        self.cls_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.lse_pool = LSEPooling(temperature=lse_temperature)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            features: (B, C, F, T) from backbone

        Returns:
            clipwise_logits: (B, num_classes)
            framewise_logits: (B, num_classes, T) — for dual supervision
        """
        if self.freq_pool_type == "mean":
            h = features.mean(dim=2)  # (B, C, F, T) -> (B, C, T)
        else:
            h = self.gem(features)  # (B, C, F, T) -> (B, C, T)
        h = h.permute(0, 2, 1)  # (B, T, C)
        framewise_logits = self.cls_fc(h)  # (B, T, num_classes)
        clipwise_logits = self.lse_pool(framewise_logits)  # (B, num_classes)
        return clipwise_logits, framewise_logits.permute(0, 2, 1)  # (B, num_classes, T)
