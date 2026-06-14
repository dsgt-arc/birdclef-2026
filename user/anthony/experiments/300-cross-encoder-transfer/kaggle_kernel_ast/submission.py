"""BirdCLEF 2026 — exp 300 frozen-encoder ridge-head baseline (AST).

The AST sibling of `kaggle_kernel/submission.py` (Perch). Same backbone-agnostic
inference — encode each 5 s window to an embedding, apply a fixed ridge head
(standardize -> W -> clip) over all 234 species — but the encoder is AST run via
HuggingFace `transformers` in-kernel (no ONNX), reusing the EXACT `ASTEncoder` forward
(ASTFeatureExtractor + ASTModel, cls pooling) that built the training cache, so the head
transfers without preprocessing drift. The head covers 234/234 species (focal+soundscape),
so there is NO logit fallback — that whole branch is dropped vs the Perch kernel.

Offline Kaggle layout (internet OFF). Expected sources:
  - competition  birdclef-2026                  (test_soundscapes/, sample_submission.csv)
  - dataset      birdclef-2026-exp300-ast-head  (head_ast.npz)
  - dataset      ast-audioset-snapshot          (HF AST snapshot: config/preprocessor/weights)

Parity note: the kernel must reproduce the cache's AST forward. The cache was built with
`transformers<5` + torch 2.11 on PACE; the dry-run (sbatch/dryrun-kernel-ast.sbatch)
asserts a held-out window's kernel embedding matches the cached embedding (cosine ~ 1.0).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

KAGGLE_USER = "acmiyaguchi"

ENCODER_SR = 16000
WINDOW_SEC = 5.0
WINDOW_SAMPLES = int(ENCODER_SR * WINDOW_SEC)  # 80000


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


def find_ast_snapshot() -> Path:
    """Locate the local HF AST snapshot dir (config + preprocessor + weights)."""
    env = os.environ.get("AST_SNAPSHOT")  # local dry-run override
    if env and os.path.isdir(env):
        return Path(env)
    d = _find_input("ast-audioset-snapshot", "birdclef-2026-ast-snapshot")
    if d is None:
        raise FileNotFoundError("AST snapshot not found in any input dataset")
    # The snapshot may be nested one level (e.g. dataset/<model_name>/); find config.json.
    if (d / "config.json").exists():
        return d
    hits = list(d.rglob("config.json"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(f"no config.json under AST snapshot {d}")


# ------------------------------------------------------------------------ audio
def load_audio_16k(path: str) -> np.ndarray:
    """Load mono float32 at 16 kHz — identical resample path to birdclef_300.audio."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != ENCODER_SR:
        import torch
        import torchaudio.functional as taF

        audio = taF.resample(torch.from_numpy(audio).float(),
                             orig_freq=sr, new_freq=ENCODER_SR).numpy()
    return audio.astype(np.float32, copy=False)


def window_audio(audio: np.ndarray) -> np.ndarray:
    """Non-overlapping 5 s windows, tail zero-padded; (n_windows, 80000)."""
    n = max(1, int(np.ceil(len(audio) / WINDOW_SAMPLES)))
    padded = n * WINDOW_SAMPLES
    if len(audio) < padded:
        audio = np.pad(audio, (0, padded - len(audio)))
    return audio[:padded].reshape(n, WINDOW_SAMPLES).astype(np.float32, copy=False)


# --------------------------------------------------------------------- encoder
class ASTEncoder:
    """AST via HF transformers — mirrors birdclef.encoders.ASTEncoder (cls pooling).

    Vendored verbatim (no repo import in-kernel): per-window ASTFeatureExtractor mel +
    ASTModel forward, pooled as mean of the [CLS]+[DIST] tokens (hidden_states[:, :2]).
    """

    def __init__(self, snapshot_dir: Path):
        import torch
        from transformers import ASTFeatureExtractor, ASTModel

        self.torch = torch
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(str(snapshot_dir))
        self.model = ASTModel.from_pretrained(str(snapshot_dir))
        self.model.eval()

    def encode_windows(self, windows: np.ndarray) -> np.ndarray:
        """(B, 80000) -> (B, 768) cls-pooled embeddings."""
        torch = self.torch
        feats = []
        with torch.no_grad():
            for w in windows:
                inputs = self.feature_extractor(
                    w, sampling_rate=ENCODER_SR, return_tensors="pt"
                )
                out = self.model(inputs["input_values"])
                pooled = out.last_hidden_state[:, :2].mean(dim=1)  # cls = [CLS]+[DIST]
                feats.append(pooled.cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)


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


# -------------------------------------------------------------------------- run
def main() -> None:
    t0 = time.time()

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

    head_dir = _find_input("birdclef-2026-exp300-ast-head") or Path(
        os.environ.get("HEAD_DIR", "artifacts")
    )
    head = load_head(Path(head_dir) / "head_ast.npz")
    species_list = head["species_list"]
    n_uncovered = int((~head["covered"]).sum())
    print(f"head: {head['W'].shape}, {head['covered'].sum()}/{len(species_list)} covered "
          f"({n_uncovered} uncovered -> left at 0; AST has no logit fallback)")

    snapshot = find_ast_snapshot()
    encoder = ASTEncoder(snapshot)
    print(f"AST snapshot: {snapshot} ({time.time() - t0:.1f}s)")

    audio_files = sorted(test_dir.glob("*.ogg")) + sorted(test_dir.glob("*.wav"))
    print(f"{len(audio_files)} test soundscapes")

    rows = []
    for fi, audio_path in enumerate(audio_files):
        windows = window_audio(load_audio_16k(str(audio_path)))
        emb = encoder.encode_windows(windows)
        probs = apply_head(head, emb)  # (n, 234)

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
