"""Call vs no-call detection analysis.

Binary task: is anything vocalizing in this 5s window?
Uses max prediction across all species as the "call" score.

Usage:
    python call_nocall.py --output artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SHARED = Path("/storage/project/ps-clef2026_birdclef-0/shared")
DATA_DIR = SHARED / "data/birdclef-2026"
INFERENCE_DIR = SHARED / "inference/birdclef-2026"
EMBED_ROOT = Path("~/scratch/birdclef-2026/outputs/exp010-2026/labels").expanduser()
PROBE_DIR = Path("~/scratch/birdclef-2026/outputs/exp101/probes").expanduser()
EDA_ARTIFACTS = Path(__file__).resolve().parents[2] / "100-eda/artifacts"

NPZ_PATHS = {
    "protoclr": INFERENCE_DIR / "protoclr-2026-soundscape/labeled.npz",
    "convnext": INFERENCE_DIR / "convnext-2026-soundscape/labeled.npz",
    "eat": INFERENCE_DIR / "eat-2026-soundscape/labeled.npz",
}

NO_FOCAL_AMPHIBIA = {"1491113", "25073", "517063"}
SONOTYPE_PREFIX = "47158son"


def load_ground_truth(eda_artifacts):
    labels = pd.read_parquet(eda_artifacts / "soundscape_labels.parquet")
    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    species_list = sorted(taxonomy["primary_label"].astype(str).unique())

    windows = labels[["filename", "start_sec", "end_sec"]].drop_duplicates().copy()
    windows["site"] = windows["filename"].str.extract(r"_(S\d+)_")

    label_sets = (
        labels.groupby(["filename", "start_sec"])["primary_label"]
        .apply(set)
        .to_dict()
    )

    # All species binary columns
    sp_cols = {}
    for sp in species_list:
        sp_cols[sp] = [
            1 if sp in label_sets.get((f, s), set()) else 0
            for f, s in zip(windows["filename"], windows["start_sec"])
        ]
    windows = pd.concat([windows, pd.DataFrame(sp_cols, index=windows.index)], axis=1)

    # Separate evaluable vs all species
    evaluable = []
    all_sp = []
    for sp in species_list:
        if sp in windows.columns and windows[sp].sum() > 0:
            all_sp.append(sp)
        if sp.startswith(SONOTYPE_PREFIX) or sp in NO_FOCAL_AMPHIBIA:
            continue
        if sp in windows.columns and windows[sp].sum() >= 2:
            evaluable.append(sp)

    # Call/no-call labels
    # "any_call" = any species present (including sonotypes)
    # "evaluable_call" = any evaluable species present
    windows["n_species_all"] = windows[all_sp].sum(axis=1)
    windows["n_species_evaluable"] = windows[evaluable].sum(axis=1) if evaluable else 0
    windows["any_call"] = (windows["n_species_all"] > 0).astype(int)
    windows["evaluable_call"] = (windows["n_species_evaluable"] > 0).astype(int)

    return windows, species_list, evaluable


def load_murilo_preds(gt):
    preds = {}
    model_species = None
    for name, path in NPZ_PATHS.items():
        data = np.load(path, allow_pickle=True)
        p = data["predictions"]
        species = [str(s) for s in data["species"]]
        seg_keys = data["seg_keys"]
        filenames = [str(seg_keys[i, 0]) for i in range(len(seg_keys))]
        window_ends = [float(seg_keys[i, 1]) for i in range(len(seg_keys))]
        pred_cols = {f"pred_{sp}": p[:, i] for i, sp in enumerate(species)}
        pred_df = pd.DataFrame({
            "filename": filenames,
            "start_sec": [w - 5.0 for w in window_ends],
            **pred_cols,
        })
        merged = gt.merge(pred_df, on=["filename", "start_sec"], how="inner")
        preds[name] = merged[[f"pred_{sp}" for sp in species]].values
        model_species = species
    preds["ensemble"] = np.mean([preds[n] for n in ["protoclr", "convnext", "eat"]], axis=0)
    return preds, model_species


def load_probe_preds(gt, model):
    probe_path = PROBE_DIR / model / "species_probe.npz"
    if not probe_path.exists():
        return None, None
    probe = np.load(probe_path, allow_pickle=True)
    coef, intercept = probe["coef"], probe["intercept"]
    probe_species = [str(s) for s in probe["species_list"]]

    embed_dir = EMBED_ROOT / model / "train_soundscapes" / "parts" / "embed"
    parts = sorted(embed_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    meta = {"file", "start_time", "end_time"}
    embed_cols = [c for c in df.columns if c not in meta]
    embeddings = df[embed_cols].values.astype(np.float32)

    logits = embeddings @ coef.T + intercept
    probs = 1.0 / (1.0 + np.exp(-logits))

    ws = [float(int(t / 5.0) * 5.0) for t in df["start_time"]]
    agg_data = {"filename": [Path(f).name for f in df["file"]], "window_start": ws}
    for i in range(probs.shape[1]):
        agg_data[f"p_{i}"] = probs[:, i]
    agg_df = pd.DataFrame(agg_data)
    pcols = [f"p_{i}" for i in range(probs.shape[1])]
    agg = agg_df.groupby(["filename", "window_start"])[pcols].max().reset_index()
    agg = agg.rename(columns={"window_start": "start_sec"})

    merged = gt.merge(agg, on=["filename", "start_sec"], how="inner")
    return merged[pcols].values, probe_species


def confusion(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--eda-artifacts", type=Path, default=EDA_ARTIFACTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    gt, species_list, evaluable = load_ground_truth(args.eda_artifacts)

    n_any_call = int(gt["any_call"].sum())
    n_eval_call = int(gt["evaluable_call"].sum())
    print(f"  {len(gt)} windows")
    print(f"  any_call: {n_any_call} positive, {len(gt) - n_any_call} negative")
    print(f"  evaluable_call: {n_eval_call} positive, {len(gt) - n_eval_call} negative")

    print("\nLoading predictions...")
    murilo_preds, murilo_species = load_murilo_preds(gt)
    all_preds = dict(murilo_preds)
    all_species = {k: murilo_species for k in murilo_preds}

    for model in ["birdnet", "perch"]:
        p, sp = load_probe_preds(gt, model)
        if p is not None:
            all_preds[f"probe_{model}"] = p
            all_species[f"probe_{model}"] = sp

    # Compute call scores: max prediction across species, sum, and mean
    print("\n=== Call/No-Call Detection ===")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
    results = {}

    for call_def in ["any_call", "evaluable_call"]:
        y_true = gt[call_def].values
        print(f"\n--- Definition: {call_def} ({int(y_true.sum())} positive / {len(y_true) - int(y_true.sum())} negative) ---")

        for model_name, pred in all_preds.items():
            sp_list = all_species[model_name]

            # Different call score aggregations
            max_score = pred.max(axis=1)
            mean_score = pred.mean(axis=1)
            # Sum of top-5 predictions
            top5_score = np.sort(pred, axis=1)[:, -5:].sum(axis=1)
            # Number of species above 0.1
            n_above_01 = (pred > 0.1).sum(axis=1).astype(float)

            for agg_name, score in [("max", max_score), ("mean", mean_score),
                                     ("top5_sum", top5_score), ("n_above_0.1", n_above_01)]:
                # AUROC
                if y_true.sum() > 0 and y_true.sum() < len(y_true):
                    auroc = roc_auc_score(y_true, score)
                    ap = average_precision_score(y_true, score)
                else:
                    auroc = ap = float("nan")

                key = f"{model_name}_{call_def}_{agg_name}"
                results[key] = {"auroc": auroc, "ap": ap}

            # Print max score results (most natural)
            auroc = results[f"{model_name}_{call_def}_max"]["auroc"]
            ap = results[f"{model_name}_{call_def}_max"]["ap"]
            print(f"\n  {model_name} (max score):")
            print(f"    AUROC={auroc:.4f}, AP={ap:.4f}")

            # Confusion at thresholds
            for t in thresholds:
                tp, fp, fn, tn, prec, rec, f1 = confusion(y_true, max_score, t)
                print(f"    @{t:.2f}: TP={tp:>4} FP={fp:>4} FN={fn:>4} TN={tn:>4} "
                      f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

    # Summary table
    print("\n=== Summary: Call/No-Call AUROC (max score) ===")
    print(f"{'Model':<20} {'any_call':>10} {'eval_call':>12}")
    print("-" * 44)
    for model_name in all_preds:
        any_auroc = results.get(f"{model_name}_any_call_max", {}).get("auroc", float("nan"))
        eval_auroc = results.get(f"{model_name}_evaluable_call_max", {}).get("auroc", float("nan"))
        print(f"{model_name:<20} {any_auroc:>10.4f} {eval_auroc:>12.4f}")

    # Aggregation comparison
    print("\n=== Aggregation comparison (ensemble, evaluable_call) ===")
    print(f"{'Aggregation':<15} {'AUROC':>8} {'AP':>8}")
    print("-" * 33)
    for agg in ["max", "mean", "top5_sum", "n_above_0.1"]:
        r = results.get(f"ensemble_evaluable_call_{agg}", {})
        print(f"{agg:<15} {r.get('auroc', 0):>8.4f} {r.get('ap', 0):>8.4f}")

    with open(args.output / "call_nocall_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to {args.output / 'call_nocall_results.json'}")


if __name__ == "__main__":
    main()
