"""Downstream gate for AST INT8 ONNX: does the quantized forward + the EXISTING
head_ast.npz still score the 739 labeled soundscapes as well as the eager head?

The strict per-window cosine gate (>0.999) fails for INT8 (0.993). But cosine is a
proxy; what actually matters is whether the deployed scores move. So we score the head
end-to-end on the 739 labeled soundscapes three ways and compare MACRO AUROC (the probe
metric, pooled per-species):

  - parquet-eager : head applied to the cached eager embeddings (the deployed number)
  - recomputed-fp32 : recompute each window's mel here, run FP32 ONNX (== eager, cos 1.0),
                      apply head. Cross-checks our windowing reproduces the extraction.
  - recomputed-int8 : same mels through the INT8 ONNX, apply head. The delta vs fp32 is
                      the *pure quantization* impact on the deployed metric.

If int8 macro AUROC ~= eager (within noise), INT8 is safe to ship with the existing head
and the 0.993 cosine is a red herring. Mels are computed once per window and shared
across fp32/int8 so the only difference is the graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from birdclef_300 import audio as audio_mod
from birdclef_300.data_loading import build_binary_label_matrix, embeddings_to_numpy
from birdclef_300.probe.ridge_probe import _per_column_scores

EXP = Path(__file__).resolve().parents[1]
SOUNDSCAPE_DIR = Path("/storage/project/ps-clef2026_birdclef-0/shared/data/birdclef-2026/train_soundscapes")
SR = audio_mod.ENCODER_SR
WIN = audio_mod.ENCODER_WINDOW_SAMPLES


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def macro_auroc(Y, pred):
    aurocs = [a for a, _ in _per_column_scores(Y, pred).values()]
    return float(np.mean(aurocs)) if aurocs else 0.0, len(aurocs)


def apply_head(head, X):
    Xs = (X.astype(np.float64) - head["mean"]) / head["std"]
    return np.clip(Xs @ head["W"], 0.0, 1.0)


def recompute_embeddings(df, fe, sessions, batch=16):
    """Mel per (filename,start_sec) window -> {name: embeddings} for each ORT session.
    Mirrors extract_torch_labeled: decode 16 kHz once per file, slice, fit_window."""
    mels, cur_file, cur_wav = [], None, None
    for fn, start_sec in zip(df["filename"], df["start_sec"]):
        if fn != cur_file:
            cur_file = fn
            p = SOUNDSCAPE_DIR / fn
            cur_wav = audio_mod.load_audio_16k(str(p)) if p.exists() else None
        if cur_wav is None:
            raise FileNotFoundError(f"missing soundscape audio: {fn}")
        s = int(start_sec * SR)
        wav = audio_mod.fit_window(cur_wav[s : s + WIN])
        mels.append(fe(wav, sampling_rate=SR, return_tensors="np")["input_values"][0])
    mels = np.stack(mels).astype(np.float32)  # (N, 1024, 128)
    out = {}
    for name, (sess, iname) in sessions.items():
        chunks = [sess.run(None, {iname: mels[i : i + batch]})[0] for i in range(0, len(mels), batch)]
        out[name] = np.concatenate(chunks).astype(np.float64)
    return out


def ort_session(path, threads):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", default=str(Path("~/scratch/birdclef-2026/features/exp300-ast").expanduser()))
    ap.add_argument("--onnx-dir", default=str(EXP / "artifacts" / "ast_onnx"))
    ap.add_argument("--head", default=str(EXP / "artifacts" / "head_ast.npz"))
    ap.add_argument("--snapshot", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(EXP / "artifacts" / "ast_int8_validation.json"))
    args = ap.parse_args()

    z = np.load(args.head, allow_pickle=True)
    head = {"W": z["W"].astype(np.float64), "mean": z["mean"].astype(np.float64),
            "std": z["std"].astype(np.float64), "species_list": [str(s) for s in z["species_list"]]}
    species_list = head["species_list"]

    df = pl.read_parquet(
        Path(args.features_dir) / "soundscape_labeled.parquet",
        columns=["filename", "start_sec", "ground_truth", "embedding"],
    )
    Y = build_binary_label_matrix(df, species_list).astype(np.float64)
    X_eager = embeddings_to_numpy(df).astype(np.float64)
    print(f"windows={len(df)} species_cols={len(species_list)} positives_cols={int((Y.sum(0)>0).sum())}")

    onnx_dir = Path(args.onnx_dir)
    fp32 = next(p for p in onnx_dir.glob("*.onnx") if "int8" not in p.name.lower())
    int8 = next(p for p in onnx_dir.glob("*.onnx") if "int8" in p.name.lower())

    from transformers import ASTFeatureExtractor

    fe = ASTFeatureExtractor.from_pretrained(args.snapshot)
    sessions = {"fp32": ort_session(fp32, args.threads), "int8": ort_session(int8, args.threads)}
    rec = recompute_embeddings(df, fe, sessions)

    # cross-check: our recomputed FP32 reproduces the cached eager embedding
    win_cos = np.mean([cos(X_eager[i], rec["fp32"][i]) for i in range(len(X_eager))])

    results = {"windows": len(df), "recompute_fp32_vs_cached_eager_meancos": round(float(win_cos), 6)}
    for name, X in [("cached_eager", X_eager), ("recomputed_fp32", rec["fp32"]), ("recomputed_int8", rec["int8"])]:
        m, n = macro_auroc(Y, apply_head(head, X))
        results[name] = {"macro_auroc": round(m, 6), "n_species": n}
        print(f"{name:18s} macro_auroc={m:.6f} ({n} species)")

    d = results["recomputed_int8"]["macro_auroc"] - results["recomputed_fp32"]["macro_auroc"]
    results["int8_minus_fp32_auroc"] = round(d, 6)
    print(f"INT8 - FP32 macro AUROC delta = {d:+.6f}  (recompute reproduces eager: cos={win_cos:.5f})")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
