"""BirdCLEF 2025 1st Place SED Architecture — Nikita Babich.

Reference code extracted from:
    kaggle kernels pull nikitababich/birdclef2025-1st-place-inference

This is the INFERENCE version of the model. Key observation: the AttHead
has its attention mechanism COMMENTED OUT for inference — only framewise
logits are returned. Training uses both clipwise (attention-weighted) and
framewise max with 50/50 loss weighting.

Critical details:
    1. Inference uses ONLY framewise_logit → sigmoid → overlap reconstruction → max
    2. Attention is a training-time regularizer, disabled at inference
    3. 3-channel input: mel spectrogram stacked 3× to match ImageNet pretrained stem
    4. Per-sample mel normalization (standardize + min-max to [0,1])
    5. Overlap reconstruction with delta blending (temporal TTA without extra passes)
    6. GeMFreq pools frequency only, preserving temporal structure
    7. Pre-dropout (p/2) before Linear in dense layers

See also:
    - docs/references/kaggle/birdclef-2025/birdclef2025-1st-place-inference.md
    - user/murilo/notes/2026-04-05-sed-gap-analysis.md
    - user/murilo/experiments/021-sed-training/atthead-comparison.md
"""

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def gem_freq(x, p=3, eps=1e-6):
    """Generalized Mean pooling on frequency axis only."""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)


# ---------------------------------------------------------------------------
# SED Head — INFERENCE VERSION (attention commented out)
# ---------------------------------------------------------------------------


class AttHead(nn.Module):
    """Attention-based SED head.

    IMPORTANT: In the inference notebook, the attention mechanism is completely
    commented out. Only framewise_logit is returned. The attention (clipwise)
    output is used during TRAINING for dual supervision, then discarded at
    inference time.

    Training loss (from forward() method):
        loss = 0.5 * criterion(logit(clipwise_prob), labels)
             + 0.5 * criterion(framewise_logit.max(dim=2)[0], labels)

    Inference prediction (from get_head_preds()):
        framewise_logit.sigmoid() → overlap reconstruction → max per segment
    """

    def __init__(self, in_chans, p=0.5, num_class=397, hidden_dim=512):
        super().__init__()

        self.pooling = GeMFreq()
        self.dense_layers = nn.Sequential(
            nn.Dropout(p / 2),  # <-- pre-dropout (missing from our AttHead)
            nn.Linear(in_chans, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(  # attention branch (used in training only)
            in_channels=hidden_dim,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.fix_scale = nn.Conv1d(  # frame logits branch
            in_channels=hidden_dim,
            out_channels=num_class,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, feat):
        feat = self.pooling(feat).squeeze(-2).permute(0, 2, 1)  # (B, T, C)
        feat = self.dense_layers(feat).permute(0, 2, 1)  # (B, hidden, T)

        # --- ATTENTION IS COMMENTED OUT FOR INFERENCE ---
        # time_att_logit = torch.tanh(self.attention(feat))

        framewise_logit = self.fix_scale(feat)
        # framewise_prob = torch.sigmoid(framewise_logit)

        # clipwise_prob = torch.sum(
        #     segmentwise_prob * torch.softmax(time_att_logit, dim=-1),
        #     dim=-1,
        # )

        output = {
            # "clipwise_prob": clipwise_prob,        # disabled at inference
            # "time_att_logit": time_att_logit,      # disabled at inference
            "framewise_logit": framewise_logit  # ONLY output used
        }
        return output


# ---------------------------------------------------------------------------
# Mel spectrogram with per-sample normalization
# ---------------------------------------------------------------------------


class NormalizeMelSpec(nn.Module):
    """Per-sample mel spectrogram normalization.

    "default" mode: standardize (subtract mean, divide std) then min-max
    scale to [0, 1]. Each spectrogram is normalized independently.

    This differs from our approach (global dataset mean/std).
    """

    def __init__(self, norm_type="default", eps=1e-6, constant=80):
        super().__init__()
        self.eps = eps
        self.norm_type = norm_type
        self.constant = constant

    def forward(self, X):
        if self.norm_type == "default":
            mean = X.mean((1, 2), keepdim=True)
            std = X.std((1, 2), keepdim=True)

            Xstd = (X - mean) / (std + self.eps)
            norm_max = torch.amax(Xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(Xstd, dim=(1, 2), keepdim=True)
            return (Xstd - norm_min) / (norm_max - norm_min + self.eps)

        elif self.norm_type == "top_db":
            X = (X + 80) / 80
            return X

        elif self.norm_type == "constant":
            X = X / self.constant
            return X


class SpecFeatureExtractor(nn.Module):
    """Mel spectrogram extraction with optional normalization and resize."""

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length=None,
        sample_rate=200,
        f_max=20,
        f_min=0.5,
        n_mels=128,
        top_db=120,
        normalized=False,
        sample_mel_normalize=None,
        output_size=(256, 256),
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential()
        self.feature_extractor.append(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                normalized=normalized,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                f_max=f_max,
                n_mels=n_mels,
                f_min=f_min,
            )
        )
        self.feature_extractor.append(T.AmplitudeToDB(top_db=top_db))

        if sample_mel_normalize is not None:
            self.feature_extractor.append(
                NormalizeMelSpec(norm_type=sample_mel_normalize)
            )

        if output_size is not None:
            self.resize = nn.UpsamplingBilinear2d(size=output_size)
        else:
            self.resize = None

    def forward(self, x: torch.Tensor, without_norm=False) -> torch.Tensor:
        if without_norm:
            img = self.feature_extractor[:-1](x)
        else:
            img = self.feature_extractor(x)
        if self.resize is not None:
            img = self.resize(img.unsqueeze(1)).squeeze(1)
        return img


# ---------------------------------------------------------------------------
# Full SED Model
# ---------------------------------------------------------------------------


class CLEFClassifierSED(nn.Module):
    """Full SED classifier: mel spec → backbone → AttHead → predictions.

    Key design choices:
        - 3-channel input: mel stacked 3× to match ImageNet stem
        - features_only=True: backbone used as feature extractor
        - backbone_dim from last feature level
        - Inference via get_head_preds() with overlap reconstruction
    """

    def __init__(self, config, disable_spectr_generator=False):
        super().__init__()

        spectrogram_config = config["spectrogram"]
        backbone_config = config["backbone"]
        head_config = config["head"]

        self.mel_spectr_generator = SpecFeatureExtractor(**spectrogram_config)

        self.backbone = timm.create_model(
            backbone_config["backbone_name"],
            pretrained=backbone_config["pretrained"],
            features_only=True,
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]

        self.head = AttHead(
            in_chans=backbone_dim,
            p=head_config["dropout"],
            num_class=head_config["num_classes"],
        )
        self.infer_duration = head_config["infer_duration"]
        self.duration = head_config["duration"]
        self.inference_type = config["inference_type"]
        self.multilabel_to_train_labels = config.get(
            "multilabel_to_train_labels", "one2one"
        )

    def get_head_preds(self, input, tta_delta=2, sigmoid=True):
        """Inference-time prediction with overlap reconstruction.

        Takes backbone features (already extracted), runs head, then
        reconstructs per-5s predictions from overlapping windows using
        the overlap_average_max_delta strategy:

        1. Place each window's framewise predictions on a global timeline
        2. Average where windows overlap
        3. Crop to aligned region
        4. Max within each 5s segment
        5. Delta blending: 0.5*center + 0.25*left_shifted + 0.25*right_shifted

        The delta blending is a form of temporal TTA without extra forward passes.
        """
        head_output = self.head(input)
        feat_time = input.shape[-1]
        framewise_prob = head_output["framewise_logit"].sigmoid().numpy()
        num_segments, num_labels, frames_per_segment = framewise_prob.shape

        step = int(feat_time * (self.infer_duration / self.duration))
        pad_frames = (frames_per_segment - step) // 2
        extra_pad = 1 if (frames_per_segment - step) % 2 != 0 else 0
        framewise_ss_len = frames_per_segment + step * (num_segments - 1)

        framewise_ss_preds = np.zeros(
            (framewise_ss_len, num_segments, num_labels), dtype=np.float16
        )
        framewise_ss_preds_mask = np.zeros((framewise_ss_len, 1), dtype=np.float16)

        for segment_ind in range(num_segments):
            start = segment_ind * step
            end = start + frames_per_segment
            framewise_ss_preds[start:end, segment_ind] += framewise_prob[segment_ind].T
            framewise_ss_preds_mask[start:end] += 1

        if self.inference_type in [
            "overlap_average_max",
            "overlap_average_max_delta",
        ]:
            framewise_ss_preds = framewise_ss_preds.sum(1)
            framewise_ss_preds = framewise_ss_preds / framewise_ss_preds_mask
        elif self.inference_type == "overlap_max":
            framewise_ss_preds = framewise_ss_preds.max(1)

        framewise_ss_preds = framewise_ss_preds[pad_frames : -pad_frames - extra_pad]
        segmentwise_preds = framewise_ss_preds.reshape(
            num_segments, step, num_labels
        ).max(1)

        if self.inference_type == "overlap_average_max_delta":
            segmentwise_preds *= 0.5
            for segment_ind in range(num_segments):
                # Left-shifted delta
                if segment_ind == 0:
                    left = framewise_ss_preds[
                        segment_ind * step : (segment_ind + 1) * step
                    ].max(0)
                else:
                    left = framewise_ss_preds[
                        segment_ind * step - tta_delta : (segment_ind + 1) * step
                        - tta_delta
                    ].max(0)

                # Right-shifted delta
                if segment_ind == (num_segments - 1):
                    right = framewise_ss_preds[
                        segment_ind * step : (segment_ind + 1) * step
                    ].max(0)
                else:
                    right = framewise_ss_preds[
                        segment_ind * step + tta_delta : (segment_ind + 1) * step
                        + tta_delta
                    ].max(0)

                segmentwise_preds[segment_ind] += left * 0.25
                segmentwise_preds[segment_ind] += right * 0.25

        return segmentwise_preds

    def forward(self, input, labels=None, train=False):
        """Training forward pass.

        Note: training code references clipwise_prob and segmentwise_logit
        which implies the TRAINING version of AttHead returns those outputs
        (attention mechanism enabled). The INFERENCE version (above) has
        attention commented out.
        """
        if not train:
            return {"preds": self.inference(input)}

        x = self.mel_spectr_generator(input)

        # 3-CHANNEL INPUT: stack mono mel 3× for ImageNet pretrained stem
        x = torch.stack([x, x, x], 1)

        x = self.backbone(x)[-1]
        head_output = self.head(x)

        if train and labels is not None:
            # DUAL SUPERVISION: 50% clipwise + 50% framewise max
            loss = 0.5 * self.critarion(
                torch.logit(head_output["clipwise_prob"]), labels
            ) + 0.5 * self.critarion(head_output["segmentwise_logit"].max(2)[0], labels)
        else:
            loss = None

        output = {
            "preds": torch.logit(head_output["clipwise_prob"]),
            "loss": loss,
            "labels": labels,
        }
        return output
