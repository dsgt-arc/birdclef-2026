"""BirdCLEF 2026 — exp 300 frozen-encoder ridge-head baseline (Perch).

Minimal, backbone-agnostic inference: encode each 5 s window to an embedding, apply
a fixed ridge head (standardize -> W -> clip) to score all 234 species, and fill the
handful of species the head cannot cover from Perch's native logits. No PCA, probes,
priors, or temporal smoothing — this is the clean baseline the exp 300 yardstick fits.

Backbone swap: replace `encode_windows` (and drop the Perch fallback). Everything else
— head application, CSV building — is encoder-independent.

Offline Kaggle layout (internet OFF). Expected dataset/model sources:
  - competition  birdclef-2026                 (test_soundscapes/, sample_submission.csv)
  - dataset      birdclef-2026-exp300-head     (head.npz, perch_species_mapping.json)
  - dataset      perch-v2-onnx                 (perch_v2_no_dft.onnx)   [Perch backbone]
  - dataset      birdclef-deps                 (onnxruntime/librosa wheels, if needed)
The ONNX is the SAME graph that produced the training embeddings, so the head transfers
exactly (no SavedModel-vs-ONNX drift).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

KAGGLE_USER = "acmiyaguchi"


def _ensure_deps() -> None:
    """Offline-install onnxruntime/librosa from the deps dataset if the Kaggle base
    image lacks them (internet is OFF). Mirrors Murilo 018's bootstrap."""
    import importlib.util
    import subprocess
    import sys

    missing = [m for m in ("onnxruntime", "librosa") if importlib.util.find_spec(m) is None]
    if not missing:
        return
    for cand in (
        f"/kaggle/input/birdclef-2025-deps",
        f"/kaggle/input/datasets/{KAGGLE_USER}/birdclef-2025-deps",
        os.environ.get("DEPS_DIR", ""),
    ):
        if cand and os.path.isdir(cand):
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-index",
                 f"--find-links={cand}", *missing],
                check=False,
            )
            return
    print(f"WARNING: deps dir not found; {missing} may be unavailable")


PERCH_SR = 32000
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(PERCH_SR * WINDOW_SEC)  # 160000
PERCH_LOGIT_OFFSET = 1  # Perch `label` drops idx 0 (background); see deploy/species_map.py


# --------------------------------------------------------------------------- IO
def _find_input(*slugs: str) -> Path | None:
    """First existing /kaggle/input mount among the given dataset slugs."""
    for slug in slugs:
        for cand in (f"/kaggle/input/{slug}", f"/kaggle/input/datasets/{KAGGLE_USER}/{slug}"):
            if os.path.exists(cand):
                return Path(cand)
    return None


def find_competition_dir() -> Path:
    for cand in (
        "/kaggle/input/competitions/birdclef-2026",
        "/kaggle/input/birdclef-2026",
    ):
        if os.path.exists(cand):
            return Path(cand)
    return Path(os.environ.get("COMPETITION_DIR", "."))


def find_perch_onnx() -> Path | None:
    env = os.environ.get("PERCH_ONNX")  # local dry-run override
    if env and os.path.exists(env):
        return Path(env)
    d = _find_input("perch-v2-onnx", "birdclef-2026-perch-onnx")
    if d is None:
        d = _find_input("birdclef-2026-exp300-head")  # allow co-packaging
    if d is not None:
        hits = list(d.glob("*.onnx"))
        if hits:
            return hits[0]
    return None


# ------------------------------------------------------------------------ audio
def load_audio_32k(path: str) -> np.ndarray:
    """Load mono float32 at 32 kHz (resample only if needed)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != PERCH_SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=PERCH_SR)
    return audio.astype(np.float32, copy=False)


def window_audio(audio: np.ndarray) -> np.ndarray:
    """Non-overlapping 5 s windows, tail zero-padded; (n_windows, 160000)."""
    n = max(1, int(np.ceil(len(audio) / WINDOW_SAMPLES)))
    padded = n * WINDOW_SAMPLES
    if len(audio) < padded:
        audio = np.pad(audio, (0, padded - len(audio)))
    return audio[:padded].reshape(n, WINDOW_SAMPLES).astype(np.float32, copy=False)


# --------------------------------------------------------------------- encoder
class PerchEncoder:
    """Perch v2 via onnxruntime — identical to exp 106's PerchSession."""

    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 4
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def encode_windows(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, 160000) -> (embedding (B,1536), label logits (B,14795))."""
        emb, logits = self.session.run(
            ["embedding", "label"], {self.input_name: windows.astype(np.float32)}
        )
        return emb.astype(np.float32), logits.astype(np.float32)


# ------------------------------------------------------------------------- head
def load_head(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {
        "W": z["W"].astype(np.float64),
        "mean": z["mean"].astype(np.float64),
        "std": z["std"].astype(np.float64),
        "species_list": [str(s) for s in z["species_list"]],
        "covered": z["covered"].astype(bool),
    }


def apply_head(head: dict, emb: np.ndarray) -> np.ndarray:
    """(emb - mean) / std @ W, clipped to [0, 1] — matches deploy.fit_head.apply_head."""
    Xs = (emb.astype(np.float64) - head["mean"]) / head["std"]
    return np.clip(Xs @ head["W"], 0.0, 1.0)


def perch_fallback_scores(logits: np.ndarray, mapping: dict, species_list: list[str]) -> np.ndarray:
    """Map Perch label logits -> (N, 234) sigmoid, column-aligned. Mirrors
    deploy.species_map.perch_mapped_scores (direct idx, genus max, unmapped -> 0)."""
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
            idxs = [k - PERCH_LOGIT_OFFSET for k in genus[label]]
            idxs = [k for k in idxs if 0 <= k < p]
            if idxs:
                out[:, i] = probs[:, idxs].max(axis=1)
    return out


# -------------------------------------------------------------------------- run
def main() -> None:
    t0 = time.time()
    import json

    _ensure_deps()

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

    head_dir = _find_input("birdclef-2026-exp300-head") or Path(
        os.environ.get("HEAD_DIR", "artifacts")
    )
    head = load_head(Path(head_dir) / "head.npz")
    with open(Path(head_dir) / "perch_species_mapping.json") as f:
        mapping = json.load(f)
    species_list = head["species_list"]
    covered = head["covered"]
    uncovered = ~covered
    print(f"head: {head['W'].shape}, {covered.sum()}/{len(species_list)} covered, "
          f"{uncovered.sum()} via Perch fallback")

    onnx_path = find_perch_onnx()
    if onnx_path is None:
        raise FileNotFoundError("perch_v2_no_dft.onnx not found in any input dataset")
    encoder = PerchEncoder(onnx_path)
    print(f"perch onnx: {onnx_path} ({time.time() - t0:.1f}s)")

    audio_files = sorted(test_dir.glob("*.ogg")) + sorted(test_dir.glob("*.wav"))
    print(f"{len(audio_files)} test soundscapes")

    rows = []
    for fi, audio_path in enumerate(audio_files):
        windows = window_audio(load_audio_32k(str(audio_path)))
        emb, logits = encoder.encode_windows(windows)
        probs = apply_head(head, emb)  # (n, 234)
        if uncovered.any():
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
