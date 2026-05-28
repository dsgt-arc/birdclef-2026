"""Phase C Gate 1 — feature-parity test (LOCAL, vs vendored Sydorskyi source).

Compares our reimplemented forward path (`birdclef.models.XCPretrainWaveClassifier`)
against Sydorskyi's own `WaveCNNAttenClasifier` (vendored at
`vendor/sydorskyi-2025/`) on N fixed audios, at two stages:

    Stage 1 — mel spectrogram          (forward(..., return_spec_feature=True))
    Stage 2 — backbone CNN embedding    (forward(..., return_cnn_emb=True))

Why local-vs-vendored (not the published Kaggle inference notebook):
    We have Sydorskyi's full source as a git submodule. Running his *own*
    `WaveCNNAttenClasifier` is a stronger, faster, fully-reproducible oracle for
    "is our PORT faithful?" than re-deriving reference tensors on Kaggle. Library
    version skew (a separate, lower-risk concern that fine-tuning absorbs) is NOT
    tested here — that is what Gate 2 (frozen-backbone OOF AUC) covers.

Design guarantees that isolate the variable under test:
  * BOTH classifiers receive the *same* waveform tensor → audio-loading backend
    differences cannot affect the comparison.
  * BOTH backbones are loaded from the *same* cleaned pretrained `state_dict`
    (via `load_sydorskyi_backbone`) → any Stage-2 divergence is attributable
    purely to the mel front-end, not the conv weights.
  * Both run in `eval()` (no dropout, no spec-augment).

Expectation: our `xcpretrain_mel.py` is a near-verbatim port of vendored `blocks.py`,
and both resolve `nnAudio ... STFT` to the *same* class, so outputs should be
near-bit-identical (cosine ~1.0, max-abs-diff ~1e-6).

Gate (proposal Phase C): per-audio backbone-embedding cosine >= 0.999 AND
mean L2 < 1e-4 on every audio. A clean PASS validates the entire B.1-B.5 forward
path before any LB burn.

Usage:
    ~/clef/birdclef-2026/.venv/bin/python \
        user/murilo/experiments/034-pretrained-backbone/scripts/gate1_parity.py \
        --backbone tf_efficientnetv2_s_in21k --n-audios 5
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch

# --- path wiring -----------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[4]  # scripts/ → 034.. → experiments → murilo → user → repo
VENDOR = REPO / "vendor" / "sydorskyi-2025"

sys.path.insert(0, str(VENDOR))  # for `code_base` (vendored oracle)
sys.path.insert(0, str(SCRIPT_DIR))  # for `load_sydorskyi_backbone`

from code_base.models.wave_clasifier import WaveCNNAttenClasifier  # noqa: E402
from load_sydorskyi_backbone import load_sydorskyi_backbone  # noqa: E402

from birdclef.models.xcpretrain_wave_classifier import XCPretrainWaveClassifier  # noqa: E402

# --- Sydorskyi selected_ebs.py config (verbatim) --------------------------
MEL_PARAMS = {
    "sample_rate": 32000,
    "n_mels": 128,
    "f_min": 20,
    "n_fft": 2048,
    "hop_length": 512,
    "normalized": True,
}
# num_class is irrelevant for backbone-emb parity (head not invoked), but kept
# at Sydorskyi's 2025 value for a faithful instantiation.
HEAD_CONFIG = {"p": 0.5, "num_class": 206, "train_period": 5.0, "infer_period": 5.0}

DEFAULT_AUDIO_DIR = Path(
    "/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes"
)

# Gate thresholds (proposal Phase C exit criterion).
COSINE_GATE = 0.999
L2_GATE = 1e-4


def load_audio(path: Path, sr: int = 32000, seconds: float = 5.0) -> torch.Tensor:
    """Load a fixed deterministic `seconds`-window from `path`, resampled to `sr`.

    Returns (1, T) float32 tensor. The SAME tensor is fed to both classifiers,
    so the audio backend is not part of the parity comparison.
    """
    import librosa

    wav, _ = librosa.load(str(path), sr=sr, mono=True)
    n = int(sr * seconds)
    if len(wav) >= n:
        wav = wav[:n]
    else:
        wav = np.pad(wav, (0, n - len(wav)))
    return torch.from_numpy(wav).float().unsqueeze(0)


def pick_audios(audio_dir: Path, n: int) -> list[Path]:
    """Deterministically pick the first `n` audio files (sorted) from `audio_dir`."""
    files: list[str] = []
    for ext in ("*.ogg", "*.wav", "*.flac", "*.mp3"):
        files.extend(glob(str(audio_dir / ext)))
    files = sorted(files)
    if len(files) < n:
        raise FileNotFoundError(f"Found only {len(files)} audio files in {audio_dir}; need {n}.")
    return [Path(f) for f in files[:n]]


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """Cosine / RMS-L2 / max-abs-diff between two tensors (computed in float64)."""
    af = a.detach().flatten().double()
    bf = b.detach().flatten().double()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    l2 = (af - bf).pow(2).mean().sqrt().item()
    maxabs = (af - bf).abs().max().item()
    return {"cosine": cos, "rms_l2": l2, "max_abs_diff": maxabs}


def build_models(backbone: str) -> tuple[XCPretrainWaveClassifier, WaveCNNAttenClasifier]:
    """Instantiate our + vendored classifiers and load the SAME pretrained backbone."""
    ours = XCPretrainWaveClassifier(
        backbone=backbone,
        mel_spec_params=MEL_PARAMS,
        head_config=HEAD_CONFIG,
        pretrained=False,
        exportable=True,
        top_db=80.0,
        amin=1e-10,
        spec_augment_config=None,
    )
    vendored = WaveCNNAttenClasifier(
        backbone=backbone,
        device="cpu",
        mel_spec_paramms=MEL_PARAMS,
        head_config=HEAD_CONFIG,
        top_db=80.0,
        amin=1e-10,
        fixed_amplitude_to_db=True,
        exportable=True,
        pretrained=False,
        spec_augment_config=None,
    )

    # Load ONE cleaned pretrained state_dict into BOTH backbones (identical weights).
    loaded = load_sydorskyi_backbone(backbone, verbose=False)
    sd = loaded.state_dict()
    ours.backbone.load_state_dict(sd, strict=True)
    vendored.backbone.load_state_dict(sd, strict=True)

    ours.eval()
    vendored.eval()
    return ours, vendored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="tf_efficientnetv2_s_in21k")
    parser.add_argument("--n-audios", type=int, default=5)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR.parent / "outputs" / "gate1_parity.json",
        help="Where to write the JSON result record.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"== Gate 1 parity: {args.backbone} (local vs vendored Sydorskyi source) ==")
    print(f"  vendor: {VENDOR}")
    audios = pick_audios(args.audio_dir, args.n_audios)
    print(f"  audios ({len(audios)}) from {args.audio_dir}:")
    for a in audios:
        print(f"    - {a.name}")

    ours, vendored = build_models(args.backbone)
    print("  models built; identical pretrained backbone loaded into both (strict=True)")

    per_audio = []
    emb_cosines = []
    emb_l2s = []
    for path in audios:
        x = load_audio(path, sr=MEL_PARAMS["sample_rate"], seconds=args.seconds)
        with torch.no_grad():
            mel_o = ours(x, return_spec_feature=True)
            mel_v = vendored(x, return_spec_feature=True)
            emb_o = ours(x, return_cnn_emb=True)
            emb_v = vendored(x, return_cnn_emb=True)
        mel_cmp = compare(mel_o, mel_v)
        emb_cmp = compare(emb_o, emb_v)
        emb_cosines.append(emb_cmp["cosine"])
        emb_l2s.append(emb_cmp["rms_l2"])
        per_audio.append(
            {
                "file": path.name,
                "mel_shape": list(mel_o.shape),
                "emb_shape": list(emb_o.shape),
                "mel": mel_cmp,
                "emb": emb_cmp,
            }
        )
        print(
            f"  {path.name:40s} | mel cos={mel_cmp['cosine']:.8f} maxabs={mel_cmp['max_abs_diff']:.2e}"
            f" | emb cos={emb_cmp['cosine']:.8f} l2={emb_cmp['rms_l2']:.2e} maxabs={emb_cmp['max_abs_diff']:.2e}"
        )

    min_emb_cos = min(emb_cosines)
    max_emb_l2 = max(emb_l2s)
    passed = (min_emb_cos >= COSINE_GATE) and (max_emb_l2 < L2_GATE)

    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backbone": args.backbone,
        "n_audios": len(audios),
        "seconds": args.seconds,
        "mel_params": MEL_PARAMS,
        "gate": {"cosine_min": COSINE_GATE, "l2_max": L2_GATE},
        "result": {
            "min_emb_cosine": min_emb_cos,
            "max_emb_rms_l2": max_emb_l2,
            "passed": passed,
        },
        "per_audio": per_audio,
        "audio_dir": str(args.audio_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))

    print("\n== Summary ==")
    print(f"  min backbone-emb cosine : {min_emb_cos:.8f}  (gate >= {COSINE_GATE})")
    print(f"  max backbone-emb RMS L2 : {max_emb_l2:.2e}  (gate < {L2_GATE})")
    print(f"  result written to       : {args.out}")
    print(f"\n  GATE 1: {'PASS ✅' if passed else 'FAIL ❌'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
