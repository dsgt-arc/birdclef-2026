"""Detailed per-species breakdown with TP/FP/FN/TN at multiple thresholds.

Produces a comprehensive CSV with per-species metrics for all models,
and a confusion summary for call vs no-call windows.

Usage:
    python detailed_breakdown.py --output artifacts/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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


def load_ground_truth(eda_artifacts: Path):
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
    sp_cols = {}
    for sp in species_list:
        sp_cols[sp] = [
            1 if sp in label_sets.get((f, s), set()) else 0
            for f, s in zip(windows["filename"], windows["start_sec"])
        ]
    sp_df = pd.DataFrame(sp_cols, index=windows.index)
    windows = pd.concat([windows, sp_df], axis=1)

    return windows, species_list


def load_murilo_preds(gt):
    """Load and align Murilo's npz predictions."""
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
        pc = [f"pred_{sp}" for sp in species]
        preds[name] = merged[pc].values
        model_species = species

    # Ensemble
    preds["ensemble"] = np.mean([preds[n] for n in ["protoclr", "convnext", "eat"]], axis=0)
    return preds, model_species


def load_probe_preds(gt, model, probe_dir, embed_root):
    """Load probe, apply to soundscape embeddings, align to GT."""
    probe_path = probe_dir / model / "species_probe.npz"
    if not probe_path.exists():
        return None, None

    probe = np.load(probe_path, allow_pickle=True)
    coef = probe["coef"]
    intercept = probe["intercept"]
    probe_species = [str(s) for s in probe["species_list"]]

    # Load embeddings
    embed_dir = embed_root / model / "train_soundscapes" / "parts" / "embed"
    parts = sorted(embed_dir.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)

    meta_cols = {"file", "start_time", "end_time"}
    embed_cols = [c for c in df.columns if c not in meta_cols]
    embeddings = df[embed_cols].values.astype(np.float32)

    # Apply probe
    logits = embeddings @ coef.T + intercept
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Aggregate to 5s windows (max)
    window_starts = [float(int(t / 5.0) * 5.0) for t in df["start_time"]]
    agg_df = pd.DataFrame({
        "filename": [Path(f).name for f in df["file"]],
        "window_start": window_starts,
    })
    for i in range(probs.shape[1]):
        agg_df[f"p_{i}"] = probs[:, i]
    pcols = [f"p_{i}" for i in range(probs.shape[1])]
    agg = agg_df.groupby(["filename", "window_start"])[pcols].max().reset_index()

    # Align
    pred_df = agg.rename(columns={"window_start": "start_sec"})
    for i, sp in enumerate(probe_species):
        pred_df = pred_df.rename(columns={f"p_{i}": f"pred_{sp}"})

    merged = gt.merge(pred_df, on=["filename", "start_sec"], how="inner")
    pc = [f"pred_{sp}" for sp in probe_species]
    return merged[pc].values, probe_species


def confusion_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, tn, precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--eda-artifacts", type=Path, default=EDA_ARTIFACTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    taxa_map = dict(zip(taxonomy["primary_label"], taxonomy.get("class_name", taxonomy.get("class", pd.Series()))))
    common_map = dict(zip(taxonomy["primary_label"], taxonomy.get("common_name", pd.Series(dtype=str))))

    print("Loading ground truth...")
    gt, species_list = load_ground_truth(args.eda_artifacts)

    print("Loading Murilo's predictions...")
    murilo_preds, murilo_species = load_murilo_preds(gt)
    murilo_sp_idx = {sp: i for i, sp in enumerate(murilo_species)}

    print("Loading probe predictions...")
    probe_preds = {}
    probe_species_lists = {}
    for model in ["birdnet", "perch"]:
        p, sp = load_probe_preds(gt, model, PROBE_DIR, EMBED_ROOT)
        if p is not None:
            probe_preds[f"probe_{model}"] = p
            probe_species_lists[f"probe_{model}"] = sp

    # All models
    all_preds = {**murilo_preds, **probe_preds}

    # Identify evaluable species
    evaluable = []
    for sp in species_list:
        if sp.startswith(SONOTYPE_PREFIX) or sp in NO_FOCAL_AMPHIBIA:
            continue
        if sp not in gt.columns:
            continue
        if gt[sp].sum() < 2:
            continue
        evaluable.append(sp)

    print(f"{len(evaluable)} evaluable species")

    # --- Per-species detailed breakdown ---
    thresholds = [0.3, 0.5, 0.7]
    rows = []

    for sp in evaluable:
        y_true = gt[sp].values
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        taxa = taxa_map.get(sp, "Unknown")
        common = common_map.get(sp, "")

        row = {
            "species": sp,
            "common_name": common,
            "taxa": taxa,
            "n_positive": n_pos,
            "n_negative": n_neg,
        }

        for model_name, pred in all_preds.items():
            if model_name in probe_species_lists:
                sp_list = probe_species_lists[model_name]
            else:
                sp_list = murilo_species
            sp_idx = {s: i for i, s in enumerate(sp_list)}

            if sp not in sp_idx:
                row[f"{model_name}_auroc"] = np.nan
                row[f"{model_name}_ap"] = np.nan
                for t in thresholds:
                    ts = str(t).replace(".", "")
                    for m in ["tp", "fp", "fn", "tn", "prec", "rec", "f1"]:
                        row[f"{model_name}_{m}@{ts}"] = np.nan
                continue

            y_score = pred[:, sp_idx[sp]]

            if n_pos == 0 or n_pos == len(y_true):
                auroc = np.nan
            else:
                auroc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score) if n_pos > 0 else np.nan

            row[f"{model_name}_auroc"] = auroc
            row[f"{model_name}_ap"] = ap

            for t in thresholds:
                tp, fp, fn, tn, prec, rec, f1 = confusion_at_threshold(y_true, y_score, t)
                ts = str(t).replace(".", "")
                row[f"{model_name}_tp@{ts}"] = tp
                row[f"{model_name}_fp@{ts}"] = fp
                row[f"{model_name}_fn@{ts}"] = fn
                row[f"{model_name}_tn@{ts}"] = tn
                row[f"{model_name}_prec@{ts}"] = prec
                row[f"{model_name}_rec@{ts}"] = rec
                row[f"{model_name}_f1@{ts}"] = f1

        rows.append(row)

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(args.output / "species_detailed.csv", index=False)

    # --- Print summary tables ---
    print("\n=== Per-species AUROC (ensemble, sorted) ===")
    print(f"{'Species':<20} {'Common':<30} {'Taxa':<12} {'n+':<5} {'AUROC':>7} {'AP':>7} | {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>5} {'Rec':>5} @ 0.5")
    print("-" * 115)
    sorted_df = detail_df.sort_values("ensemble_auroc", ascending=False)
    for _, r in sorted_df.iterrows():
        print(
            f"{r['species']:<20} {str(r.get('common_name',''))[:29]:<30} {r['taxa']:<12} "
            f"{int(r['n_positive']):<5} {r['ensemble_auroc']:>7.3f} {r['ensemble_ap']:>7.3f} | "
            f"{int(r.get('ensemble_tp@05',0)):>4} {int(r.get('ensemble_fp@05',0)):>4} "
            f"{int(r.get('ensemble_fn@05',0)):>4} {r.get('ensemble_prec@05',0):>5.2f} "
            f"{r.get('ensemble_rec@05',0):>5.2f}"
        )

    # --- Aggregate confusion stats ---
    print("\n=== Aggregate TP/FP/FN/TN across all evaluable species @ threshold ===")
    for model_name in ["ensemble", "protoclr", "eat", "convnext", "probe_birdnet", "probe_perch"]:
        print(f"\n  {model_name}:")
        for t in thresholds:
            ts = str(t).replace(".", "")
            tp_col = f"{model_name}_tp@{ts}"
            fp_col = f"{model_name}_fp@{ts}"
            fn_col = f"{model_name}_fn@{ts}"
            tn_col = f"{model_name}_tn@{ts}"
            if tp_col not in detail_df.columns:
                continue
            tp = int(detail_df[tp_col].sum())
            fp = int(detail_df[fp_col].sum())
            fn = int(detail_df[fn_col].sum())
            tn = int(detail_df[tn_col].sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"    @{t}: TP={tp:>5} FP={fp:>5} FN={fn:>5} TN={tn:>5} | Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    print(f"\nSaved to {args.output / 'species_detailed.csv'}")


if __name__ == "__main__":
    main()
