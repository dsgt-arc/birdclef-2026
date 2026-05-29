#!/usr/bin/env python3
"""Export an exp 034 XCPretrain model (Sydorskyi backbone + SED head) to ONNX.

Unlike the other SED export scripts (which export backbone+head taking a *mel*
input and recompute mel in the kernel), this exports the **FULL** model —
raw audio → nnAudio mel → backbone → head → `clipwise_pred_long` — so the Kaggle
kernel feeds **raw audio** windows and does NO mel reimplementation. Sydorskyi's
`TraceableMelspec` uses nnAudio STFT precisely because it's ONNX-traceable, so the
whole pipeline exports cleanly (verified: max|onnx-eager| ~1e-7). This eliminates
the mel-mismatch risk entirely.

Usage:
    python scripts/export_xcpretrain.py \
        --checkpoint .../checkpoints/xcpretrain-effv2s-frozen/fold0/last.ckpt \
        --config .../experiments/034-pretrained-backbone/configs/xcpretrain-effv2s-frozen.yaml \
        --output-dir artifacts/weights/xcpretrain-effv2s-frozen/
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


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def load_config(path: Path) -> dict:
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    inherit = cfg.pop("inherit", None)
    if inherit:
        cfg = _deep_merge(load_config(path.parent / inherit), cfg)
    return cfg


class XCPretrainONNXWrapper(nn.Module):
    """raw audio (B, T) → clipwise_pred_long probabilities (B, num_class).

    Mel (nnAudio STFT + torchaudio MelScale + amplitude-to-dB + standardize/min-max)
    is INSIDE the graph — no kernel-side mel. Output is the clip-only prob-space
    aggregation (matches training/val scoring; NO clip+frame blend).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spec = self.model.extract_spec(audio)  # (B, 1, n_mels, T_frame)
        emb = self.model.backbone(spec)[-1]  # (B, C, H, W)
        return self.model.head(emb)["clipwise_pred_long"]  # (B, num_class) probs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="exp 034 Lightning last.ckpt")
    p.add_argument("--config", required=True, help="exp 034 configs/*.yaml (with inherit)")
    p.add_argument(
        "--train-csv",
        default="/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/taxonomy.csv",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.path.insert(0, str(USER_SRC_DIR))

    from birdclef.data.soundscape import build_full_species_list
    from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier

    cfg = load_config(Path(args.config))
    dcfg, mcfg = cfg["data"], cfg["model"]
    sample_rate = dcfg["sample_rate"]
    window_samples = int(dcfg["window_seconds"] * sample_rate)

    species_list = build_full_species_list(Path(args.train_csv))
    num_classes = len(species_list)
    assert num_classes == dcfg["num_classes"], f"{num_classes} != {dcfg['num_classes']}"
    logger.info(f"Species: {num_classes}; window {window_samples} samples @ {sample_rate} Hz")

    # Build the model (mel chain + backbone + head); spec_augment OFF for inference
    # (it's training-only + stateless, so it contributes no state_dict keys either way).
    mel_params = {
        "sample_rate": sample_rate,
        "n_mels": dcfg.get("n_mels", 128),
        "f_min": 20,
        "n_fft": 2048,
        "hop_length": 512,
        "normalized": True,
    }
    head_config = {"num_class": num_classes, **mcfg["head"]}
    model = XCPretrainWaveClassifier(
        backbone=mcfg["backbone"],
        pretrained=False,
        exportable=True,
        head_config=head_config,
        spec_augment_config=None,
        mel_spec_params=mel_params,
    )

    # Load the trained weights: extract the `model.*` subset from the Lightning ckpt.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model_sd = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    # GeMFreq.p / conv weights etc. must all load; only allow benign misses.
    if missing or unexpected:
        raise RuntimeError(
            f"State load mismatch:\n  missing={missing[:10]}\n  unexpected={unexpected[:10]}"
        )
    logger.info(f"Loaded {len(model_sd)} model tensors from {Path(args.checkpoint).name}")

    wrapper = XCPretrainONNXWrapper(model).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    dummy = torch.randn(1, window_samples)
    with torch.no_grad():
        ref = wrapper(dummy)
    logger.info(f"Eager output: {tuple(ref.shape)} (expect (1, {num_classes}))")

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["audio"],
        output_names=["probs"],
        dynamic_axes={"audio": {0: "batch", 1: "time"}, "probs": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    logger.info(f"ONNX saved: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Parity check on a DIFFERENT random batch (exercises dynamic batch axis).
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    a2 = torch.randn(3, window_samples)
    with torch.no_grad():
        r2 = wrapper(a2).numpy()
    o2 = sess.run(None, {"audio": a2.numpy()})[0]
    max_diff = float(np.abs(o2 - r2).max())
    logger.info(f"ONNX↔eager parity: max|diff| = {max_diff:.3e} (batch=3)")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX parity FAILED: max diff {max_diff:.3e} > 1e-4")

    # species_list.json (reference) + species_mapping.json (what the kernel loads).
    # Emit the mapping DIRECTLY here (234-identity) so the export is self-sufficient —
    # the kernel reads idx_to_name_2026; relying on a separate `cp` of the canonical
    # mapping is fragile (a stale leftover shipped the 41-overlap map once). Our model
    # is 234-class on the 2026 taxonomy, so idx i ↔ species_list[i] is identity.
    (output_dir / "species_list.json").write_text(json.dumps(species_list, indent=2))
    idx_to_name_2026 = {str(i): name for i, name in enumerate(species_list)}
    species_mapping = {
        "model_species": species_list,
        "birdclef2026_species": species_list,
        "idx_to_name_2026": idx_to_name_2026,
        "idx_2025_to_name_2026": idx_to_name_2026,  # backward-compat key for the kernel
    }
    (output_dir / "species_mapping.json").write_text(json.dumps(species_mapping, indent=2))
    logger.info(
        f"species_mapping.json: {len(idx_to_name_2026)} idx_to_name_2026 entries (234-identity)"
    )

    # mel_config.json — signals the kernel to feed RAW AUDIO (mel is in the ONNX).
    mel_config = {
        "type": "xcpretrain_audio_onnx",
        "sr": sample_rate,
        "window_samples": window_samples,
        "window_duration": dcfg["window_seconds"],
        "output_type": "clipwise_probs",  # ONNX already returns probabilities (no sigmoid)
        "num_classes": num_classes,
        "run_name": cfg["run_name"],
        "backbone": mcfg["backbone"],
    }
    (output_dir / "mel_config.json").write_text(json.dumps(mel_config, indent=2))
    logger.info(f"Saved species_list.json + mel_config.json → {output_dir}")
    logger.info("Export complete ✅")


if __name__ == "__main__":
    main()
