"""BirdCLEF 2026 — exp 300 frozen-encoder ridge-head baseline (backbone ablation).

Minimal, backbone-agnostic inference: encode each 5 s window to an embedding, apply a
fixed ridge head (standardize -> W -> clip) to score all 234 species, write the CSV.
No PCA, probes, priors, or temporal smoothing — the clean baseline the exp 300 yardstick
fits. This is an ABLATION harness: the SAME pipeline runs Perch / BEATs / BirdNET; only
the encoder (and its `head.npz`) changes. The encoder is selected by `head["encoder"]`,
so one script serves all three kernels.

Per encoder the transferable core is identical (`emb -> head -> 234`). Perch additionally
has native logits, used ONLY to fill species the head can't cover — but focal+soundscape
covers all 234, so that path is inert and absent for BEATs/BirdNET.

Offline Kaggle layout (internet OFF). Dataset/model sources per encoder:
  - competition  birdclef-2026                       (test_soundscapes/, sample_submission.csv)
  - dataset      birdclef-2026-exp300-<enc>-head     (head.npz [+ perch_species_mapping.json])
  - Perch:   perch-v2-onnx        (perch_v2_no_dft.onnx — SAME graph as the training cache)
  - BEATs:   birdclef-2026-beats  (BEATs_iter3_plus_AS2M.pt + vendor/unilm/beats code)
  - BirdNET: birdclef-2026-birdnet + deps wheels (bioacoustics-model-zoo, ai-edge-litert)
  - AST:     birdclef-2026-exp300-ast-onnx (ast_cls[.int8].onnx + preprocessor_config.json;
             ONNX transformer to fit the 90-min CPU wall, onnxruntime from birdclef-2025-deps)
The encoder forward is the SAME code that produced the training embeddings, so the head
transfers with no train/inference drift.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

KAGGLE_USER = "acmiyaguchi"

WINDOW_SEC = 5.0
PERCH_SR = 32000
ENCODER_SR_16K = 16000
PERCH_LOGIT_OFFSET = 1  # Perch `label` drops idx 0 (background); see deploy/species_map.py


# --------------------------------------------------------------------------- deps
def _ensure_deps(modules: list[str], *deps_slugs: str) -> None:
    """Offline-install `modules` from a deps dataset if the Kaggle base image lacks
    them (internet is OFF). Mirrors Murilo 018's bootstrap; no-op for present modules."""
    import importlib.util
    import subprocess

    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return
    cands = []
    for slug in deps_slugs:
        cands += [f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}"]
    cands.append(os.environ.get("DEPS_DIR", ""))
    for cand in cands:
        if cand and os.path.isdir(cand):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-index",
                 f"--find-links={cand}", *missing],
                check=False,
            )
            return
    print(f"WARNING: deps dir not found; {missing} may be unavailable")


# --------------------------------------------------------------------------- IO
def _find_input(*slugs: str) -> Path | None:
    """First existing /kaggle/input mount among the given dataset slugs."""
    for slug in slugs:
        for cand in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}"):
            if os.path.exists(cand):
                return Path(cand)
    return None


def find_competition_dir() -> Path:
    for cand in ("/kaggle/input/competitions/birdclef-2026", "/kaggle/input/birdclef-2026"):
        if os.path.exists(cand):
            return Path(cand)
    return Path(os.environ.get("COMPETITION_DIR", "."))


# ------------------------------------------------------------------------ audio
def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load mono float32 at `target_sr` (resample only if needed)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


def load_audio_torch(path: str, target_sr: int) -> np.ndarray:
    """Like `load_audio` but resamples with torchaudio (sinc) — MUST match the training
    extractor's `birdclef_300.audio.load_audio_16k`. librosa vs torchaudio resamplers
    diverge enough to drift BEATs embeddings (~0.94 cosine), so the BEATs forward uses
    this loader. (BirdNET/bmz load the path themselves, so they are unaffected.)"""
    import torch
    import torchaudio.functional as taF

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = taF.resample(torch.from_numpy(audio).float(), orig_freq=sr, new_freq=target_sr).numpy()
    return audio.astype(np.float32, copy=False)


def window_audio(audio: np.ndarray, window_samples: int) -> np.ndarray:
    """Non-overlapping fixed windows, tail zero-padded; (n_windows, window_samples)."""
    n = max(1, int(np.ceil(len(audio) / window_samples)))
    padded = n * window_samples
    if len(audio) < padded:
        audio = np.pad(audio, (0, padded - len(audio)))
    return audio[:padded].reshape(n, window_samples).astype(np.float32, copy=False)


# --------------------------------------------------------------------- encoders
# Each encoder owns its own preprocessing and returns ONE embedding row per 5 s
# window (so row_ids align across encoders). `encode_file` -> (emb (n,D), logits|None);
# logits are Perch-only (for the inert fallback).
class PerchEncoder:
    """Perch v2 via onnxruntime — identical to exp 106's PerchSession (32 kHz, 5 s)."""

    requires = ["onnxruntime", "librosa"]

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        onnx_path = next(iter(model_dir.glob("*.onnx")))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.window_samples = int(PERCH_SR * WINDOW_SEC)  # 160000

    def encode_file(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        windows = window_audio(load_audio(path, PERCH_SR), self.window_samples)
        emb, logits = self.session.run(
            ["embedding", "label"], {self.input_name: windows.astype(np.float32)}
        )
        return emb.astype(np.float32), logits.astype(np.float32)


class BeatsEncoder:
    """BEATs via torch CPU — the SAME forward as birdclef.encoders.BEATsEncoder
    (16 kHz, 5 s, mean-pool over the patch sequence). Zero train/inference drift."""

    requires = ["librosa"]  # torch ships in the Kaggle base image

    def __init__(self, model_dir: Path, batch_size: int = 16):
        import torch

        # Vendored unilm/beats code ships beside the checkpoint in the dataset.
        vendor = model_dir / "beats"
        if not vendor.exists():
            vendor = model_dir
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))
        from BEATs import BEATs, BEATsConfig

        ckpt_path = next(iter(model_dir.glob("*.pt")))
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.torch = torch
        self.batch_size = batch_size
        self.window_samples = int(ENCODER_SR_16K * WINDOW_SEC)  # 80000

    def encode_file(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        windows = window_audio(load_audio_torch(path, ENCODER_SR_16K), self.window_samples)
        out = []
        with self.torch.no_grad():
            for i in range(0, len(windows), self.batch_size):
                wav = self.torch.from_numpy(windows[i:i + self.batch_size]).float()  # (B, 80000)
                feats, _ = self.model.extract_features(wav)        # (B, seq, D)
                out.append(feats.mean(dim=1).cpu().numpy())        # (B, D)
        return np.concatenate(out).astype(np.float32), None


AST_WINDOW_SEC = 10.0  # AST runs on 10 s windows (see AstEncoder); the submission grid stays 5 s


class AstEncoder:
    """AST cls embedding via ONNX, on 10 s windows — two stacked tricks to fit the 90-min
    CPU wall. (1) ONNX: the transformer (ASTModel + cls pooling) is exported
    (scripts/export_ast_onnx.py) and run via onnxruntime (INT8), like Perch. (2) 10 s
    geometry: AST's mel ALWAYS pads to a fixed 1024-frame (~10.24 s) tensor, so a 5 s
    window wastes half the forward on silence. Feeding 10 s windows (6 per 60 s clip
    instead of 12) HALVES the forward count at identical per-window cost. Eager 5 s was
    ~1.72 win/s -> timeout; INT8 + 10 s is ~5x cheaper end-to-end.

    The submission grid is still 5 s (row_id `{clip}_{end}`, end in {5,10,...,60}), so each
    10 s window's embedding is repeated for its two 5 s halves — both halves share the
    coarser 10 s score. head_ast.npz is refit on 10 s-window embeddings (the distribution
    shifts vs 5 s+silence), and the ONNX graph is unchanged (1024-frame input either way).
    `model_dir` holds <name>.onnx + preprocessor_config.json (no ASTModel weights needed)."""

    requires = ["onnxruntime", "librosa"]  # torch/torchaudio/transformers in Kaggle base

    def __init__(self, model_dir: Path):
        import onnxruntime as ort
        from transformers import ASTFeatureExtractor

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(str(model_dir))
        onnx_files = sorted(Path(model_dir).glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx under {model_dir}")
        # prefer the INT8 graph if both are shipped (validated to cache parity upstream)
        onnx_path = next((p for p in onnx_files if "int8" in p.name.lower()), onnx_files[0])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.window_samples = int(ENCODER_SR_16K * AST_WINDOW_SEC)  # 160000 (10 s)
        self.repeat = int(round(AST_WINDOW_SEC / WINDOW_SEC))       # 2 sub-segments per window

    def encode_file(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        windows = window_audio(load_audio_torch(path, ENCODER_SR_16K), self.window_samples)
        mels = np.stack([
            self.feature_extractor(w, sampling_rate=ENCODER_SR_16K, return_tensors="np")
            ["input_values"][0]
            for w in windows
        ]).astype(np.float32)  # (n_10s_windows, 1024, 128)
        emb = self.session.run(None, {self.input_name: mels})[0]  # (n_10s_windows, 768), cls
        # Expand each 10 s window to its `repeat` consecutive 5 s submission rows so main()
        # emits the full 5 s grid (end_time 5,10,...) from half as many forwards.
        emb = np.repeat(emb, self.repeat, axis=0)  # (repeat * n_10s_windows, 768)
        return emb.astype(np.float32), None


EAT_NORM_MEAN = -4.268   # birdclef.encoders.EATEncoder defaults — must match the cache
EAT_NORM_STD = 4.569
EAT_TARGET_FRAMES = 512  # 5 s at 10 ms hop


class EatEncoder:
    """EAT cls embedding via ONNX (worstchan/EAT-base_epoch30_finetune_AS2M). The
    transformer (`extract_features` + CLS token at position 0) is exported
    (scripts/export_eat_onnx.py) and run via onnxruntime — this sidesteps the
    `trust_remote_code` the eager model needs, which the offline kernel can't fetch.
    The mel is torchaudio kaldi-fbank replicating birdclef.encoders.EATEncoder.
    _waveform_to_mel EXACTLY (5 s/512-frame, num_mel_bins=128, hanning, frame_shift=10,
    (mel - (-4.268)) / (4.569*2)), so the in-kernel forward reproduces the frozen-cls-ft
    cache (export parity cosine 1.0). ~257 tokens/window -> ~8.5 win/s on 4 vCPU, far
    under the 90-min wall (FP32 — no INT8 needed, unlike AST). 5 s submission grid, one
    forward per 5 s row. `model_dir` holds eat_cls.onnx (no EAT weights / remote code)."""

    requires = ["onnxruntime"]  # torch/torchaudio in Kaggle base

    def __init__(self, model_dir: Path, batch_size: int = 16):
        import onnxruntime as ort
        import torch

        self.torch = torch
        onnx_files = sorted(Path(model_dir).glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx under {model_dir}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_files[0]), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.window_samples = int(ENCODER_SR_16K * WINDOW_SEC)  # 80000 (5 s)
        self.batch_size = batch_size

    def _mel(self, windows: np.ndarray) -> np.ndarray:
        """(n, 80000) waveforms -> (n, 1, 512, 128) mels, matching EATEncoder exactly."""
        import torchaudio

        mels = []
        for w in windows:
            wav = self.torch.from_numpy(w).float()
            wav = wav - wav.mean()
            mel = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0), htk_compat=True, sample_frequency=ENCODER_SR_16K,
                use_energy=False, window_type="hanning", num_mel_bins=128,
                dither=0.0, frame_shift=10,
            )  # (time, 128)
            n = mel.shape[0]
            if n < EAT_TARGET_FRAMES:
                mel = self.torch.nn.functional.pad(mel, (0, 0, 0, EAT_TARGET_FRAMES - n))
            elif n > EAT_TARGET_FRAMES:
                mel = mel[:EAT_TARGET_FRAMES, :]
            mels.append(mel)
        mel = self.torch.stack(mels, dim=0)               # (n, 512, 128)
        mel = (mel - EAT_NORM_MEAN) / (EAT_NORM_STD * 2)
        return mel.unsqueeze(1).numpy().astype(np.float32)  # (n, 1, 512, 128)

    def encode_file(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        windows = window_audio(load_audio_torch(path, ENCODER_SR_16K), self.window_samples)
        mels = self._mel(windows)
        out = []
        for i in range(0, len(mels), self.batch_size):
            out.append(self.session.run(None, {self.input_name: mels[i:i + self.batch_size]})[0])
        return np.concatenate(out).astype(np.float32), None  # (n, 768), cls


SSAMBA_NORM_MEAN = -4.2677393  # birdclef.encoders.SSAMBAHFEncoder AudioSet defaults
SSAMBA_NORM_STD = 4.5689974
SSAMBA_TARGET_FRAMES = 1024    # model config.img_size time dim (8x64=512 patches + CLS)
SSAMBA_WINDOW_SEC = 10.0       # 10 s geometry: fills the 1024-frame mel, halves forwards
SSAMBA_BATCH = 16              # batch-16 is the CPU win/s sweet spot (1.6 win/s @ 4 vCPU)
SSAMBA_STRIDE = 2              # process every 2nd 10 s window (20 s hop): full coverage at
                              # 1.6 win/s -> ~62 min fwd TIMED OUT on the hidden set, so we
                              # halve the forward count (3/clip) -> ~31 min fwd (~45 min real),
                              # tiling each 10 s embedding across its 20 s block.


class SsambaEncoder:
    """SSAMBA-HF mean-patch embedding (SSL ssamba_small_400, 384-dim) run EAGER on CPU.

    SSAMBA's deploy blocker was the Mamba selective-scan. The unrolled-ONNX path is
    non-viable (513-token x 24-layer x 2-dir -> ~25K-node 203 MB graph; ORT session-init
    alone times out). Instead we run the exp004 HF port's PURE-PYTORCH scan directly: on
    CPU `_should_use_cuda_kernels` auto-falls-back (and we force it off), so no mamba-ssm /
    onnxruntime is needed — just torch + torchaudio (both in the Kaggle base). At 4 vCPU,
    batch-16 gives ~1.6 win/s; the model always pads the mel to 1024 frames (~10 s), so a
    5 s window wastes half on silence -> we run 10 s windows. Full coverage (6/clip, ~6K
    forwards, ~62 min fwd) TIMED OUT on the hidden set (the fused-scan mamba-onnx op was only
    ~15% faster AND returned all-NaN, so it was abandoned). Fix: 20 s hop (SSAMBA_STRIDE=2)
    -> forward only every 2nd 10 s window (3/clip, ~31 min fwd, ~45 min real), tiling each
    embedding across its 20 s block, then `repeat(emb,2)` to fill the 5 s submission grid. The
    head is REFIT on the 10 s-geometry cache. The mel is torchaudio kaldi-fbank replicating
    SSAMBAHFEncoder._waveform_to_fbank EXACTLY (num_mel_bins=128, hanning, frame_shift=10,
    (mel - (-4.2677393)) / (4.5689974*2), padded to 1024). `model_dir` holds the ssamba_hf
    package + ssamba_small_400.pth."""

    requires: list[str] = []  # torch + torchaudio in Kaggle base; NO mamba-ssm / onnxruntime

    def __init__(self, model_dir: Path, batch_size: int = SSAMBA_BATCH):
        import torch

        self.torch = torch
        torch.set_num_threads(os.cpu_count() or 4)
        model_dir = Path(model_dir)
        # The staged dir holds the `ssamba_hf` package + the .pth; add it to the path so the
        # pure-PyTorch model architecture is importable offline.
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from ssamba_hf import load_from_ssamba_checkpoint

        ckpts = sorted(model_dir.glob("*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"no .pth under {model_dir}")
        self.model, self.config = load_from_ssamba_checkpoint(
            str(ckpts[0]), device="cpu", verbose=False
        )
        self.model.eval()
        for m in self.model.modules():  # force the pure-PyTorch scan (no mamba-ssm on Kaggle)
            if hasattr(m, "_use_cuda_kernels"):
                m._use_cuda_kernels = False
        self.fdim, self.tdim = self.config.img_size  # (128 freq, 1024 time)
        self.window_samples = int(ENCODER_SR_16K * SSAMBA_WINDOW_SEC)  # 160000 (10 s)
        self.repeat = int(round(SSAMBA_WINDOW_SEC / WINDOW_SEC))       # 2 sub-segments / window
        self.stride = SSAMBA_STRIDE                                    # process every Nth window
        self.batch_size = batch_size

    def _mel(self, windows: np.ndarray):
        """(n, 160000) waveforms -> (n, 1024, 128) mels, matching SSAMBAHFEncoder exactly."""
        import torchaudio

        mels = []
        for w in windows:
            wav = self.torch.from_numpy(w).float()
            wav = wav - wav.mean()
            mel = torchaudio.compliance.kaldi.fbank(
                wav.unsqueeze(0), htk_compat=True, sample_frequency=ENCODER_SR_16K,
                use_energy=False, window_type="hanning", num_mel_bins=self.fdim,
                dither=0.0, frame_shift=10,
            )  # (time, 128)
            n = mel.shape[0]
            if n < self.tdim:
                mel = self.torch.nn.functional.pad(mel, (0, 0, 0, self.tdim - n))
            elif n > self.tdim:
                mel = mel[:self.tdim, :]
            mels.append(mel)
        mel = self.torch.stack(mels, dim=0)               # (n, 1024, 128)
        mel = (mel - SSAMBA_NORM_MEAN) / (SSAMBA_NORM_STD * 2)
        return mel  # torch (n, 1024, 128) = (B, T, F), what model.forward expects

    def encode_file(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        windows = window_audio(load_audio_torch(path, ENCODER_SR_16K), self.window_samples)
        n = len(windows)                                           # 6 for a 60 s clip
        # 20 s hop: only the every-`stride`-th 10 s window (indices 0,2,4 for n=6) is computed,
        # then its embedding is tiled across the windows it covers. We mel + forward ONLY those
        # selected windows — halving BOTH the mel front-end and the Mamba-scan cost. Full
        # coverage (6/clip mel + fwd, ~65 min fwd + mel/IO overhead) overshot the 90-min wall;
        # 3/clip (~32 min fwd) clears it. The forward math is byte-identical to full coverage
        # (already consistency-verified, cos=1.000) — only temporal sampling is coarser.
        sel = list(range(0, n, self.stride))                       # processed window indices
        sel_mels = self._mel(windows[sel])                         # (len(sel), 1024, 128)
        sel_emb = []
        with self.torch.no_grad():
            for i in range(0, len(sel_mels), self.batch_size):
                emb = self.model.forward(sel_mels[i:i + self.batch_size])  # (b, 384) mean_patch
                sel_emb.append(emb.cpu().numpy())
        sel_emb = np.concatenate(sel_emb).astype(np.float32)       # (len(sel), 384)
        # Window j is represented by the processed window at floor(j/stride) (the 10 s slice
        # at the start of its 20 s block).
        per_window = sel_emb[[j // self.stride for j in range(n)]]  # (n, 384)
        # Expand each 10 s window to its `repeat` consecutive 5 s submission rows.
        return np.repeat(per_window, self.repeat, axis=0).astype(np.float32), None  # (repeat*n, 384)


class BirdnetEncoder:
    """BirdNET via bioacoustics-model-zoo (TFLite) — the SAME 3 s -> 5 s pooling as
    extract_birdnet.extract_birdnet_focal. CPU; needs bmz + litert staged offline."""

    requires = ["librosa"]  # bmz/litert imported lazily below

    def __init__(self, model_dir: Path):
        import bioacoustics_model_zoo as bmz

        # Internet is OFF: pass the staged .tflite + labels as LOCAL paths. bmz only
        # downloads when checkpoint_url/label_url start with "http"; a local path is
        # used verbatim (no network, no cache dir). Same weights as training -> the
        # embed() math is identical, so the consistency check still holds.
        def _find(*pats: str) -> str:
            for pat in pats:
                hits = sorted(Path(model_dir).rglob(pat))
                if hits:
                    return str(hits[0])
            raise FileNotFoundError(f"{pats} not under {model_dir}")

        ckpt = _find("*Model_FP16.tflite", "*.tflite")
        labels = _find("*Labels*.txt", "*.txt")
        self.model = bmz.BirdNET(checkpoint_url=ckpt, label_url=labels)

    def encode_file(self, path: str) -> tuple[np.ndarray, np.ndarray | None]:
        embed_df = self.model.embed([path], return_preds=False).reset_index()
        start_col = "start_time" if "start_time" in embed_df.columns else embed_df.columns[1]
        feat_cols = [c for c in embed_df.columns if isinstance(c, int) or str(c).isdigit()]
        if not feat_cols:
            feat_cols = [c for c in embed_df.columns if c not in ("file", start_col, "end_time")]
        starts = embed_df[start_col].to_numpy(dtype=np.float32)
        feats = embed_df[feat_cols].to_numpy(dtype=np.float32)
        order = np.argsort(starts)
        starts, feats = starts[order], feats[order]
        n_windows = int(starts.max() // WINDOW_SEC) + 1 if len(starts) else 1
        rows = []
        for wi in range(n_windows):
            t = wi * WINDOW_SEC
            sel = (starts >= t) & (starts < t + WINDOW_SEC)
            if not sel.any():
                sel = np.zeros(len(starts), dtype=bool)
                if len(starts):
                    sel[int(np.argmin(np.abs(starts - t)))] = True
            rows.append(feats[sel].mean(axis=0) if sel.any() else np.zeros(feats.shape[1]))
        return np.stack(rows).astype(np.float32), None


def build_encoder(encoder: str):
    """Locate the encoder's model dataset and construct it (deps installed first)."""
    if encoder == "perch":
        _ensure_deps(PerchEncoder.requires, "birdclef-2025-deps")
        d = _find_input("perch-v2-onnx", "birdclef-2026-perch-onnx") \
            or _find_input("birdclef-2026-exp300-perch-head")
        env = os.environ.get("PERCH_ONNX")
        if env and os.path.exists(env):
            d = Path(env).parent
        if d is None:
            raise FileNotFoundError("Perch ONNX dataset not found")
        return PerchEncoder(Path(d))
    if encoder == "beats":
        _ensure_deps(BeatsEncoder.requires, "birdclef-2025-deps")
        d = _find_input("birdclef-2026-beats") or Path(os.environ.get("BEATS_DIR", ""))
        if d is None or not Path(d).exists():
            raise FileNotFoundError("BEATs model dataset not found")
        return BeatsEncoder(Path(d))
    if encoder == "ast":
        _ensure_deps(AstEncoder.requires, "birdclef-2025-deps")
        d = _find_input("birdclef-2026-exp300-ast-onnx") or Path(os.environ.get("AST_ONNX_DIR", ""))
        if d is None or not Path(d).exists():
            raise FileNotFoundError("AST ONNX dataset not found")
        return AstEncoder(Path(d))
    if encoder == "eat":
        _ensure_deps(EatEncoder.requires, "birdclef-2025-deps")
        d = _find_input("birdclef-2026-exp300-eat-onnx") or Path(os.environ.get("EAT_ONNX_DIR", ""))
        if d is None or not Path(d).exists():
            raise FileNotFoundError("EAT ONNX dataset not found")
        return EatEncoder(Path(d))
    if encoder == "ssamba_hf":
        # Eager pure-PyTorch (no ONNX): the dataset holds the ssamba_hf package + the .pth.
        d = _find_input("birdclef-2026-exp300-ssamba-weights") or Path(os.environ.get("SSAMBA_WEIGHTS_DIR", ""))
        if d is None or not Path(d).exists():
            raise FileNotFoundError("SSAMBA weights dataset not found")
        return SsambaEncoder(Path(d))
    if encoder == "birdnet":
        _ensure_deps(["bioacoustics_model_zoo", "ai_edge_litert", *BirdnetEncoder.requires],
                     "birdclef-2026-birdnet-deps", "birdclef-2025-deps")
        d = _find_input("birdclef-2026-birdnet") or Path(os.environ.get("BIRDNET_DIR", ""))
        return BirdnetEncoder(Path(d) if d else Path("."))
    raise ValueError(f"unknown encoder '{encoder}'")


# ------------------------------------------------------------------------- head
def load_head(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "W": z["W"].astype(np.float64),
        "mean": z["mean"].astype(np.float64),
        "std": z["std"].astype(np.float64),
        "species_list": [str(s) for s in z["species_list"]],
        "covered": z["covered"].astype(bool),
        "encoder": str(z["encoder"]) if "encoder" in z.files else "perch",
    }


def apply_head(head: dict, emb: np.ndarray) -> np.ndarray:
    """(emb - mean) / std @ W, clipped to [0, 1] — matches deploy.fit_head.apply_head."""
    Xs = (emb.astype(np.float64) - head["mean"]) / head["std"]
    return np.clip(Xs @ head["W"], 0.0, 1.0)


def perch_fallback_scores(logits: np.ndarray, mapping: dict, species_list: list[str]) -> np.ndarray:
    """Map Perch label logits -> (N, 234) sigmoid, column-aligned (Perch-only, inert
    when all 234 species are covered). Mirrors deploy.species_map.perch_mapped_scores."""
    direct = mapping["direct_map"]
    genus = mapping.get("genus_proxy_map", {})
    n, p = logits.shape
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    out = np.zeros((n, len(species_list)), dtype=np.float64)
    for i, label in enumerate(species_list):
        if label in direct:
            j = direct[label] - PERCH_LOGIT_OFFSET
            if 0 <= j < p:
                out[:, i] = probs[:, j]
        elif label in genus:
            idxs = [k - PERCH_LOGIT_OFFSET for k in genus[label] if 0 <= k - PERCH_LOGIT_OFFSET < p]
            if idxs:
                out[:, i] = probs[:, idxs].max(axis=1)
    return out


# -------------------------------------------------------------------------- run
def main() -> None:
    t0 = time.time()
    import json

    competition_dir = find_competition_dir()
    test_dir = competition_dir / "test_soundscapes"
    sample_sub_path = competition_dir / "sample_submission.csv"
    output_path = Path("/kaggle/working/submission.csv")
    if not test_dir.exists():
        test_dir = Path(os.environ.get("TEST_SOUNDSCAPES_DIR", "test_soundscapes"))
    if not sample_sub_path.exists():
        sample_sub_path = Path(os.environ.get("SAMPLE_SUBMISSION", "sample_submission.csv"))
    if not output_path.parent.exists():
        output_path = Path("predictions/submission.csv")

    head_dir = _find_input(
        "birdclef-2026-exp300-head",      # legacy single-encoder (Perch) dataset
        "birdclef-2026-exp300-perch-head",
        "birdclef-2026-exp300-beats-head",
        "birdclef-2026-exp300-birdnet-head",
        "birdclef-2026-exp300-ast-head",
        "birdclef-2026-exp300-eat-head",
        "birdclef-2026-exp300-ssamba-head",
    ) or Path(os.environ.get("HEAD_DIR", "artifacts"))
    # head file is `head.npz` (beats/birdnet staging) or `head_<enc>.npz` (AST dataset).
    head_npz = Path(head_dir) / "head.npz"
    if not head_npz.exists():
        cands = sorted(Path(head_dir).glob("*.npz"))
        if cands:
            head_npz = cands[0]
    head = load_head(head_npz)
    species_list = head["species_list"]
    covered = head["covered"]
    uncovered = ~covered
    encoder_name = head["encoder"]
    print(f"encoder={encoder_name}  head {head['W'].shape}  "
          f"{covered.sum()}/{len(species_list)} covered  {uncovered.sum()} uncovered")

    mapping = None
    mp = Path(head_dir) / "perch_species_mapping.json"
    if encoder_name == "perch" and mp.exists():
        with open(mp) as f:
            mapping = json.load(f)

    encoder = build_encoder(encoder_name)
    print(f"built {encoder_name} encoder ({time.time() - t0:.1f}s)")

    audio_files = sorted(test_dir.glob("*.ogg")) + sorted(test_dir.glob("*.wav"))
    print(f"{len(audio_files)} test soundscapes")

    rows = []
    for fi, audio_path in enumerate(audio_files):
        emb, logits = encoder.encode_file(str(audio_path))
        probs = apply_head(head, emb)  # (n, 234)
        if logits is not None and mapping is not None and uncovered.any():
            fb = perch_fallback_scores(logits, mapping, species_list)
            probs[:, uncovered] = fb[:, uncovered]

        soundscape_id = audio_path.stem
        for w in range(probs.shape[0]):
            end_time = int((w + 1) * WINDOW_SEC)
            row = {"row_id": f"{soundscape_id}_{end_time}"}
            for si, sp in enumerate(species_list):
                row[sp] = float(probs[w, si])
            rows.append(row)
        if (fi + 1) % 20 == 0 or fi == len(audio_files) - 1:
            print(f"  [{fi + 1}/{len(audio_files)}] {audio_path.name}: {probs.shape[0]} windows")

    # Align to sample_submission (column order + full row coverage).
    if sample_sub_path.exists():
        sample_sub = pd.read_csv(sample_sub_path)
        expected = list(sample_sub.columns)
        species_cols = [c for c in expected if c != "row_id"]
        if rows:
            pred_df = pd.DataFrame(rows)
            submission = sample_sub[["row_id"]].merge(pred_df, on="row_id", how="left")
            for col in species_cols:
                submission[col] = submission.get(col, 0.0)
                submission[col] = submission[col].fillna(0.0)
        else:
            submission = sample_sub.copy()
            for col in species_cols:
                submission[col] = 0.0
        submission = submission[expected]
    else:
        submission = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"\nwrote {output_path} ({len(submission)} rows) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
