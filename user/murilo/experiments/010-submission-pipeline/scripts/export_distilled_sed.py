#!/usr/bin/env python3
"""Export Tucker distilled-SED model (exp 032) to ONNX.

Faithful to Tucker Arrants's `bc2026-distilled-sed` recipe:
- EffNet-B0 NS-JFT backbone (use_post_head=True → 1280-d feature map)
- AttHeadLogitSpace head (LOGIT-space clipwise aggregation; no sigmoid roundtrip)
- ONNX output: `0.5 * clip_logits + 0.5 * frame_max_logits` in LOGIT space
  (single output; pre-blended via the export wrapper)

Smoothing (5-tap Gaussian across 12 windows-per-file) + sigmoid at end
happen submission-side per `analysis/notes/2026-05-23-tucker-dataset-and-inference-audit.md` § 7.3,
NOT inside the ONNX. The kernel routes via `mel_config["type"] == "tucker_mel"`.

The distillation head (`distill_head.proj.*` for KD checkpoints) is NOT
part of the inference forward; we build the module with
`use_perch_distill=False` and load with `strict=False`, ignoring those keys.

Usage:
    python scripts/export_distilled_sed.py \\
        --checkpoint /path/to/last.ckpt \\
        --config user/murilo/experiments/032-tucker-distilled-sed/configs/distilled_sed_control.yaml \\
        --train-csv /storage/.../taxonomy.csv \\
        --output-dir artifacts/weights/distilled-sed-control/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)

USER_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "src"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--train-csv",
        type=str,
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv",
        help="Taxonomy CSV for 234-species list.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts/weights/distilled-sed-control/"
    )
    parser.add_argument(
        "--blend-clip",
        type=float,
        default=0.5,
        help="Logit-space blend weight for clipwise (Tucker default 0.5).",
    )
    parser.add_argument(
        "--blend-frame",
        type=float,
        default=0.5,
        help="Logit-space blend weight for frame_max (Tucker default 0.5).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sys.path.insert(0, str(USER_SRC_DIR))

    from birdclef.data.soundscape import build_full_species_list
    from birdclef.encoders.efficientnet import EfficientNetEncoder
    from birdclef.heads.sed_head import AttHeadLogitSpace
    from birdclef.lightning.distilled_sed_finetune import DistilledSEDFinetuneModule
    from birdclef.preprocessing.nnaudio_mel import NNAudioMelSpectrogram
    from birdclef.preprocessing.tucker_mel import TuckerMelSpectrogram

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    species_list = build_full_species_list(args.train_csv)
    num_classes = len(species_list)
    logger.info(f"Species: {num_classes}")

    # Encoder — Tucker mode (use_post_head=True → 1280-d post-conv_head feature map)
    # drop_path_rate=0.0 for export (DropPath is identity in eval; cleaner ONNX trace)
    in_chans = cfg["encoder"].get("in_chans", 1)
    encoder = EfficientNetEncoder(
        model_name=cfg["encoder"]["model_name"],
        pretrained=False,  # checkpoint provides all weights
        drop_path_rate=0.0,
        in_chans=in_chans,
        use_post_head=cfg["encoder"].get("use_post_head", True),
    )
    embed_dim = encoder.embed_dim
    logger.info(
        f"Encoder: {cfg['encoder']['model_name']} (use_post_head=True, embed_dim={embed_dim})"
    )

    # Head — AttHeadLogitSpace (Tucker)
    head_cfg = cfg["sed_head"]
    head_type = head_cfg.get("type", "att_head_logit_space")
    if head_type != "att_head_logit_space":
        raise ValueError(
            f"export_distilled_sed expects head_type='att_head_logit_space', got {head_type!r}"
        )
    sed_head = AttHeadLogitSpace(
        in_channels=embed_dim,
        num_classes=num_classes,
        hidden_dim=head_cfg["hidden_dim"],
        dropout=head_cfg["dropout"],
        gem_p=head_cfg.get("gem_p", 3.0),
    )
    logger.info(
        f"SED head: AttHeadLogitSpace (hidden={head_cfg['hidden_dim']}, "
        f"dropout={head_cfg['dropout']}, gem_p={head_cfg.get('gem_p', 3.0)})"
    )

    # Build module with use_perch_distill=False so the distill_head doesn't exist;
    # KD checkpoints' distill_head.* keys will land in `unexpected` under strict=False.
    # DistilledSEDFinetuneModule requires non-None mel_transform; we build one
    # that MATCHES the training-time mel transform so its state-dict keys load
    # cleanly. The mel transform is NOT used at export (kernel computes mel
    # CPU-side via compute_mel_tucker).
    training_cfg = cfg["training"]
    mel_cfg_train = cfg["mel"]
    sample_rate_train = cfg["data"]["sample_rate"]
    window_duration_train = cfg["data"]["window_duration"]
    mel_type_train = mel_cfg_train.get("type", "nnaudio")
    if mel_type_train == "tucker":
        # Post-Phase-D path: TuckerMelSpectrogram (torchaudio MelSpec + AmplitudeToDB).
        stub_mel_transform = TuckerMelSpectrogram(
            sample_rate=sample_rate_train,
            n_fft=mel_cfg_train.get("n_fft", 2048),
            hop_length=mel_cfg_train.get("hop_length", 512),
            n_mels=mel_cfg_train.get("n_mels", 256),
            fmin=mel_cfg_train.get("fmin", 20),
            fmax=mel_cfg_train.get("fmax", 16000),
            top_db=mel_cfg_train.get("top_db", 80.0),
            power=mel_cfg_train.get("power", 2.0),
        )
        logger.info("Mel transform (stub for ckpt load): TuckerMelSpectrogram")
    else:
        # Legacy path: NNAudioMelSpectrogram. Phase 1 checkpoints used this.
        target_frames_train = int(
            window_duration_train * sample_rate_train / mel_cfg_train["hop_length"]
        )
        stub_mel_transform = NNAudioMelSpectrogram(
            sample_rate=sample_rate_train,
            n_mels=mel_cfg_train["n_mels"],
            target_frames=target_frames_train,
            n_fft=mel_cfg_train.get("n_fft"),
            hop_length=mel_cfg_train["hop_length"],
            fmax=mel_cfg_train.get("fmax"),
            norm_type=mel_cfg_train.get("norm_type", "audioset"),
            top_db=mel_cfg_train.get("top_db", 80.0),
        )
        logger.info(
            f"Mel transform (stub for ckpt load): NNAudioMelSpectrogram"
            f"(norm_type={mel_cfg_train.get('norm_type', 'audioset')!r})"
        )
    module = DistilledSEDFinetuneModule(
        encoder=encoder,
        sed_head=sed_head,
        num_classes=num_classes,
        mel_transform=stub_mel_transform,
        spec_augment=None,
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
        max_epochs=training_cfg["max_epochs"],
        warmup_epochs=training_cfg["warmup_epochs"],
        warmup_start_factor=training_cfg.get("warmup_start_factor", 1.0 / 25),
        eta_min=training_cfg.get("eta_min", 1e-6),
        clipwise_weight=training_cfg.get("clipwise_weight", 0.5),
        framewise_weight=training_cfg.get("framewise_weight", 0.5),
        blend_clip=training_cfg.get("blend_clip", 0.5),
        blend_frame=training_cfg.get("blend_frame", 0.5),
        threshold=training_cfg.get("threshold", 0.5),
        per_sample_mel_norm=training_cfg.get("per_sample_mel_norm", True),
        use_perch_distill=False,
        perch_onnx_path=None,
        perch_embed_dim=training_cfg.get("perch_embed_dim", 1536),
        alpha_distill=training_cfg.get("alpha_distill", 1.0),
    )

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", state)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    # Tolerate distill_head.* + torchmetric state buffers; raise on anything else.
    unexpected_non_distill = [k for k in unexpected if not k.startswith("distill_head.")]
    real_missing = [
        k
        for k in missing
        if not any(m in k for m in ["cmap", "auroc", "f1_macro", "f1_micro", "top1", "top5"])
    ]
    if real_missing or unexpected_non_distill:
        raise RuntimeError(
            f"Schema drift loading {args.checkpoint}: "
            f"missing={real_missing}, unexpected_non_distill={unexpected_non_distill}"
        )
    if unexpected:
        n_distill = sum(1 for k in unexpected if k.startswith("distill_head."))
        logger.info(f"Ignored {n_distill} distill_head.* keys (KD branch, not used at inference)")
    logger.info("Checkpoint loaded")

    # Wrapper: pre-blend clip + frame_max in LOGIT space → single output
    class TuckerSEDWrapper(nn.Module):
        """ONNX-friendly wrapper that returns blended logits (LOGIT space).

        Per audit § 7.3 the per-window blend `blend_clip·clip + blend_frame·frame_max`
        must happen in logit space; smoothing across windows + final sigmoid stay
        submission-side (in `predict_soundscape_tucker`). Baking the blend into the
        wrapper means the ONNX has exactly one output and the kernel just stacks
        per-window logits before smoothing.
        """

        def __init__(self, encoder, sed_head, blend_clip: float, blend_frame: float):
            super().__init__()
            self.encoder = encoder
            self.sed_head = sed_head
            self.blend_clip = blend_clip
            self.blend_frame = blend_frame

        def forward(self, mel: torch.Tensor) -> torch.Tensor:
            # mel: (B, 1, n_mels, T)
            features = self.encoder.forward_features(mel)  # (B, 1280, F, T)
            clip_logits, frame_logits = self.sed_head(features)
            frame_max_logits = frame_logits.max(dim=-1).values  # (B, num_classes)
            return self.blend_clip * clip_logits + self.blend_frame * frame_max_logits

    wrapper = TuckerSEDWrapper(
        module.encoder,
        module.sed_head,
        blend_clip=args.blend_clip,
        blend_frame=args.blend_frame,
    )
    wrapper.eval()

    # Dummy input matching Tucker's mel params (audit § 7.1)
    mel_cfg = cfg["mel"]
    sample_rate = cfg["data"]["sample_rate"]
    window_duration = cfg["data"]["window_duration"]
    n_mels = mel_cfg["n_mels"]
    hop_length = mel_cfg["hop_length"]
    # Tucker's mel: T = window_samples // hop + 1 = 160000 // 512 + 1 = 313
    target_frames = int(window_duration * sample_rate / hop_length) + 1

    dummy_mel = torch.randn(1, in_chans, n_mels, target_frames)
    with torch.no_grad():
        ref_out = wrapper(dummy_mel)
    logger.info(
        f"Test output shape: {ref_out.shape}, range: [{ref_out.min():.4f}, {ref_out.max():.4f}]"
    )

    # Export ONNX
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    logger.info(f"Exporting ONNX to {onnx_path}")
    # Mark mel/time as dynamic axes per audit § 7.2 (avoids the SEDExportWrapper bug
    # where Tucker hardcoded INF_N_MELS=128 in the dummy_mel; we keep n_mels=256
    # consistent with training AND mark time dynamic so the ONNX accepts whatever
    # frame count the kernel produces).
    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        str(onnx_path),
        opset_version=17,
        input_names=["mel"],
        output_names=["blended_logits"],
        dynamic_axes={
            "mel": {0: "batch", 2: "n_mels", 3: "time"},
            "blended_logits": {0: "batch"},
        },
        dynamo=False,
    )
    logger.info(f"ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Verify ONNX matches PyTorch
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"mel": dummy_mel.numpy()})[0]
    max_diff = float(np.max(np.abs(ref_out.numpy() - onnx_out)))
    logger.info(f"ONNX verification: max diff = {max_diff:.2e}")
    assert max_diff < 5e-2, f"ONNX output mismatch too large: {max_diff}"

    # Save species list
    species_path = output_dir / "species_list.json"
    with open(species_path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(f"Species list saved: {species_path}")

    # Save mel config matching Tucker's training pipeline (audit § 7.1)
    # The kernel uses `type: "tucker_mel"` to route to `compute_mel_tucker` and
    # `output_type: "tucker_logits"` to route to `predict_soundscape_tucker`.
    mel_config = {
        "type": "tucker_mel",
        "output_type": "tucker_logits",
        "architecture": "tucker_sed",
        "sr": sample_rate,
        "n_fft": mel_cfg["n_fft"],
        "hop_length": hop_length,
        "n_mels": n_mels,
        "fmin": mel_cfg.get("fmin", 20),
        "fmax": mel_cfg.get("fmax", 16000),
        "top_db": mel_cfg.get("top_db", 80.0),
        "power": mel_cfg.get("power", 2.0),
        "target_frames": target_frames,
        "window_duration": window_duration,
        "inference_window": window_duration,
        "in_chans": in_chans,
        "blend_clip": args.blend_clip,
        "blend_frame": args.blend_frame,
        # 5-tap Gaussian smoothing across 12 windows-per-file in LOGIT space (audit § 7.3)
        "gaussian_smooth_logit_space": True,
        "gaussian_kernel": [0.1, 0.2, 0.4, 0.2, 0.1],
    }
    mel_config_path = output_dir / "mel_config.json"
    with open(mel_config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    logger.info(f"Mel config saved: {mel_config_path}")

    logger.info("Distilled SED export complete!")


if __name__ == "__main__":
    main()
