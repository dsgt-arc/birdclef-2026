"""Sydorskyi 2025 wave classifier orchestrator — direct port (simplified).

Source: vendor/sydorskyi-2025/code_base/models/wave_clasifier.py:37-298 @ 053594f
        (note typo in source: "clasifier" not "classifier")

Composes:
    Sydorskyi mel chain (TraceableMelspec + ChannelAgnosticAmplitudeToDB + NormalizeMelSpec)
      + timm features_only backbone (EffNetV2-S or NFNet-L0, in_chans=1)
      + SpecAugment (CustomFreqMasking + CustomTimeMasking, training only — TODO Phase B sub-commit)
      + YoinkAttHead (512 hidden, GeMFreq, dropout 0.25/0.5)

Pipeline (forward):
    waveform (B, T_audio)
      → mel_extractor → (B, n_mels, T_frame)
      → unsqueeze channel → (B, 1, n_mels, T_frame)
      → (training-only) spec_augment
      → backbone(features_only=True)[-1] → (B, C, H_freq, W_time)
      → head → dict {clipwise_logits_long, clipwise_pred_long, framewise_logits_long}

Simplifications vs wave_clasifier.py:
- Drop multi-mel-spec list support (`_n_specs > 1` branch)
- Drop transformer_backbone branch (unused with timm CNN backbones)
- Drop quantizable / CQT / LEAF spec extractors
- Drop atten_smoothing_config / deep_supervision_steps
- Drop central_crop_input, selected_indices, permute_backbone_emb
- Drop spec_resize, no_head_inference_mode, use_sigmoid
- Drop first_conv_stride_overwrite
- Use train-time SpecAugment ONLY (no power_aug / lower_high / white_noise / bandpass options
  — those aren't in our selected configs)

The simplifications collapse 260 lines into ~80; keeps the exact branches our
selected_ebs.py / selected_eca.py configs hit.

Confidence: VERIFIED (Phase A.2). 2026-05-28.
"""

from typing import Any

import timm
import torch.nn as nn
from torch import Tensor

from birdclef.heads.yoink_sed_head import YoinkAttHead
from birdclef.preprocessing.yoink_mel import build_sydorskyi_mel_extractor


class YoinkWaveClassifier(nn.Module):
    """Sydorskyi 2025 WaveCNNAttenClasifier, simplified to our selected-config path.

    Args:
        backbone: timm model name. "tf_efficientnetv2_s_in21k" or "eca_nfnet_l0".
        mel_spec_params: kwargs for TraceableMelspec. Defaults to Sydorskyi config
            (sample_rate=32000, n_mels=128, n_fft=2048, hop_length=512, f_min=20,
             normalized=True).
        head_config: kwargs for YoinkAttHead. At minimum {"p": 0.5, "num_class": 234,
            "train_period": 5.0, "infer_period": 5.0}.
        pretrained: pass-through to timm.create_model. Default False (we load Sydorskyi
            ckpt manually via load_sydorskyi_backbone).
        exportable: pass-through to TraceableMelspec + GeMFreq + NormalizeMelSpec.
            Set True to match Sydorskyi's exportable=True training mode (the path
            the pretrained ckpts were trained against).
        top_db: amplitude-to-dB cutoff. Default 80 (Sydorskyi).
        amin: amplitude-to-dB floor. Default 1e-10 (Sydorskyi).
        spec_augment_config: optional dict of {"freq_mask": {...}, "time_mask": {...}}
            kwargs forwarded to CustomFreqMasking / CustomTimeMasking. Applied only
            in training mode. Default None (no spec aug — Phase B parity testing).
            For training, pass:
                {"freq_mask": {"mask_max_length": 10, "mask_max_masks": 3, "p": 0.3, "inplace": True},
                 "time_mask": {"mask_max_length": 20, "mask_max_masks": 3, "p": 0.3, "inplace": True}}

    Pretrained ckpt loading is NOT done in __init__ — call
    `birdclef.scripts.load_sydorskyi_backbone` separately after instantiation
    (matches Sydorskyi's pretrain_config plumbing).
    """

    def __init__(
        self,
        backbone: str,
        mel_spec_params: dict[str, Any] | None = None,
        head_config: dict[str, Any] | None = None,
        pretrained: bool = False,
        exportable: bool = True,
        top_db: float = 80.0,
        amin: float = 1e-10,
        spec_augment_config: dict[str, Any] | None = None,
    ):
        super().__init__()

        # 1. Mel extractor (TraceableMelspec → ChannelAgnosticAmplitudeToDB → NormalizeMelSpec)
        self.logmelspec_extractor = build_sydorskyi_mel_extractor(
            mel_params=mel_spec_params,
            top_db=top_db,
            amin=amin,
            exportable=exportable,
        )

        # 2. SpecAugment (optional, training-only) — TODO Phase B sub-commit
        # For now we defer the CustomFreqMasking / CustomTimeMasking import (those
        # modules will land alongside the dataset + loss + sampler ports). Forward
        # raises clearly if config is provided but module not implemented.
        if spec_augment_config is not None:
            raise NotImplementedError(
                "YoinkWaveClassifier: spec_augment_config support pending Phase B sub-commit "
                "(needs birdclef.preprocessing.yoink_spec_augment). For Gate 1 parity testing "
                "and Gate 2 frozen-backbone sanity, leave spec_augment_config=None."
            )
        self.spec_augment = None

        # 3. Backbone (timm features_only mode — strips conv_head/bn2/classifier
        #    automatically; matches Sydorskyi's stripped pretrained ckpt key set)
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            exportable=exportable,
            in_chans=1,
        )

        # 4. Head — read backbone output channels from the timm feature_info API
        backbone_channels = self.backbone.feature_info.channels()
        if head_config is None:
            head_config = {"p": 0.5, "num_class": 234, "train_period": 5.0, "infer_period": 5.0}
        self.head = YoinkAttHead(
            in_chans=backbone_channels[-1],
            exportable=exportable,
            **head_config,
        )

    def extract_spec(self, x: Tensor) -> Tensor:
        """Compute the mel spectrogram from a raw waveform (B, T_audio) → (B, 1, n_mels, T_frame).

        Exposed as a separate method so callers can pre-compute and cache mels if needed,
        or for parity testing.
        """
        spec = self.logmelspec_extractor(x)  # (B, n_mels, T_frame)
        return spec[:, None]  # add channel dim → (B, 1, n_mels, T_frame)

    def forward(
        self,
        x: Tensor,
        spec: Tensor | None = None,
        return_spec_feature: bool = False,
        return_cnn_emb: bool = False,
    ) -> dict[str, Tensor]:
        """Run the full pipeline.

        Args:
            x: raw waveform (B, T_audio). Ignored if spec is provided.
            spec: pre-computed mel (B, 1, n_mels, T_frame). If None, computed from x.
            return_spec_feature: short-circuit return after mel extraction (debug/parity).
            return_cnn_emb: short-circuit return after backbone (debug/parity).

        Returns:
            Head output dict (see YoinkAttHead.forward), OR the intermediate tensor
            if a short-circuit flag is set.
        """
        if spec is None:
            spec = self.extract_spec(x)

        if self.spec_augment is not None and self.training:
            spec = self.spec_augment(spec)

        if return_spec_feature:
            return spec

        # Backbone features_only mode returns a LIST of feature maps per stage;
        # we want the last stage's output.
        emb = self.backbone(spec)[-1]  # (B, C, H_freq, W_time)

        if return_cnn_emb:
            return emb

        return self.head(emb)
