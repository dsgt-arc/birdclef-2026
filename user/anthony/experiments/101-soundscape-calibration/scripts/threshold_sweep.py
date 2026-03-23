"""Sweep thresholds on CV sites, evaluate on S22.

Tests global and per-taxa thresholds for all models (fine-tuned + probes).
LOSO CV on 8 train sites to select threshold, final eval on S22.

Usage:
    python threshold_sweep.py --output artifacts/
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
THRESHOLDS = np.arange(0.05, 0.95, 0.05).round(2).tolist()
TEST_SITE = "S22"


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
    sp_cols = {}
    for sp in species_list:
        sp_cols[sp] = [
            1 if sp in label_sets.get((f, s), set()) else 0
            for f, s in zip(windows["filename"], windows["start_sec"])
        ]
    windows = pd.concat([windows, pd.DataFrame(sp_cols, index=windows.index)], axis=1)
    return windows, species_list


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

    # Aggregate to 5s (max)
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


def get_evaluable(species_list, gt):
    evaluable = []
    for sp in species_list:
        if sp.startswith(SONOTYPE_PREFIX) or sp in NO_FOCAL_AMPHIBIA:
            continue
        if sp not in gt.columns or gt[sp].sum() < 2:
            continue
        evaluable.append(sp)
    return evaluable


def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def sweep_thresholds(gt, pred, species, evaluable, taxa_map):
    """Sweep thresholds, return per-threshold metrics."""
    sp_idx = {sp: i for i, sp in enumerate(species)}
    rows = []
    for t in THRESHOLDS:
        # Global
        all_tp = all_fp = all_fn = all_tn = 0
        taxa_stats = {}
        for sp in evaluable:
            if sp not in sp_idx or sp not in gt.columns:
                continue
            y_true = gt[sp].values
            y_score = pred[:, sp_idx[sp]]
            y_pred = (y_score >= t).astype(int)
            tp, fp, fn, tn, _, _, _ = confusion(y_true, y_pred)
            all_tp += tp; all_fp += fp; all_fn += fn; all_tn += tn

            taxa = taxa_map.get(sp, "Unknown")
            if taxa not in taxa_stats:
                taxa_stats[taxa] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            taxa_stats[taxa]["tp"] += tp
            taxa_stats[taxa]["fp"] += fp
            taxa_stats[taxa]["fn"] += fn
            taxa_stats[taxa]["tn"] += tn

        prec = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        rec = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        rows.append({
            "threshold": t, "scope": "global",
            "tp": all_tp, "fp": all_fp, "fn": all_fn, "tn": all_tn,
            "precision": prec, "recall": rec, "f1": f1,
        })

        for taxa, s in taxa_stats.items():
            tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            rows.append({
                "threshold": t, "scope": taxa,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec, "recall": rec, "f1": f1,
            })

    return pd.DataFrame(rows)


def loso_threshold_selection(gt, pred, species, evaluable, taxa_map):
    """LOSO CV on train sites to select best threshold, eval on S22."""
    test_mask = gt["site"] == TEST_SITE
    train_gt = gt[~test_mask].copy().reset_index(drop=True)
    test_gt = gt[test_mask].copy().reset_index(drop=True)
    train_pred = pred[~test_mask.values]
    test_pred = pred[test_mask.values]

    train_sites = sorted(train_gt["site"].unique())

    # LOSO CV: for each fold, sweep thresholds on held-out site
    fold_best = {"global": [], "per_taxa": {}}
    for held_out in train_sites:
        val_mask = train_gt["site"] == held_out
        val_gt = train_gt[val_mask].reset_index(drop=True)
        val_pred = train_pred[val_mask.values]

        sweep = sweep_thresholds(val_gt, val_pred, species, evaluable, taxa_map)

        # Best global threshold by F1
        global_sweep = sweep[sweep["scope"] == "global"]
        if len(global_sweep) > 0 and global_sweep["f1"].max() > 0:
            best_t = global_sweep.loc[global_sweep["f1"].idxmax(), "threshold"]
        else:
            best_t = 0.5
        fold_best["global"].append(best_t)

        # Best per-taxa threshold by F1
        for taxa in sweep["scope"].unique():
            if taxa == "global":
                continue
            taxa_sweep = sweep[sweep["scope"] == taxa]
            if len(taxa_sweep) > 0 and taxa_sweep["f1"].max() > 0:
                best_t = taxa_sweep.loc[taxa_sweep["f1"].idxmax(), "threshold"]
            else:
                best_t = 0.5
            fold_best.setdefault("per_taxa", {}).setdefault(taxa, []).append(best_t)

    # Aggregate: median of fold-selected thresholds
    global_threshold = float(np.median(fold_best["global"]))
    taxa_thresholds = {
        taxa: float(np.median(ts))
        for taxa, ts in fold_best.get("per_taxa", {}).items()
    }

    # Evaluate on S22 with selected thresholds
    sp_idx = {sp: i for i, sp in enumerate(species)}
    results = {"global": {}, "per_taxa": {}}

    # Global threshold on S22
    tp = fp = fn = tn = 0
    for sp in evaluable:
        if sp not in sp_idx or sp not in test_gt.columns:
            continue
        y_true = test_gt[sp].values
        y_pred = (test_pred[:, sp_idx[sp]] >= global_threshold).astype(int)
        t, f, n, g, _, _, _ = confusion(y_true, y_pred)
        tp += t; fp += f; fn += n; tn += g
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results["global"] = {
        "threshold": global_threshold, "cv_thresholds": fold_best["global"],
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
    }

    # Per-taxa threshold on S22
    tp = fp = fn = tn = 0
    for sp in evaluable:
        if sp not in sp_idx or sp not in test_gt.columns:
            continue
        taxa = taxa_map.get(sp, "Unknown")
        t_thresh = taxa_thresholds.get(taxa, global_threshold)
        y_true = test_gt[sp].values
        y_pred = (test_pred[:, sp_idx[sp]] >= t_thresh).astype(int)
        t, f, n, g, _, _, _ = confusion(y_true, y_pred)
        tp += t; fp += f; fn += n; tn += g
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results["per_taxa"] = {
        "thresholds": taxa_thresholds, "cv_folds": {k: v for k, v in fold_best.get("per_taxa", {}).items()},
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--eda-artifacts", type=Path, default=EDA_ARTIFACTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    taxonomy = pd.read_csv(DATA_DIR / "taxonomy.csv")
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    taxa_map = dict(zip(taxonomy["primary_label"],
                        taxonomy.get("class_name", taxonomy.get("class", pd.Series()))))

    print("Loading data...")
    gt, species_list = load_ground_truth(args.eda_artifacts)
    evaluable = get_evaluable(species_list, gt)
    print(f"  {len(gt)} windows, {len(evaluable)} evaluable species")

    # Load all models
    print("Loading predictions...")
    murilo_preds, murilo_species = load_murilo_preds(gt)

    all_preds = dict(murilo_preds)
    all_species = {"protoclr": murilo_species, "convnext": murilo_species,
                   "eat": murilo_species, "ensemble": murilo_species}

    for model in ["birdnet", "perch"]:
        p, sp = load_probe_preds(gt, model)
        if p is not None:
            all_preds[f"probe_{model}"] = p
            all_species[f"probe_{model}"] = sp

    # Full sweep on all 739 windows (descriptive)
    print("\n=== Full threshold sweep (739 windows) ===")
    for model_name in all_preds:
        sweep = sweep_thresholds(gt, all_preds[model_name],
                                 all_species[model_name], evaluable, taxa_map)
        sweep.to_csv(args.output / f"threshold_sweep_{model_name}.csv", index=False)

        global_sweep = sweep[sweep["scope"] == "global"]
        best_row = global_sweep.loc[global_sweep["f1"].idxmax()]
        print(f"\n  {model_name}: best global F1={best_row['f1']:.3f} @ t={best_row['threshold']:.2f}"
              f" (P={best_row['precision']:.3f} R={best_row['recall']:.3f})")

        for taxa in sorted(sweep["scope"].unique()):
            if taxa == "global":
                continue
            taxa_sweep = sweep[sweep["scope"] == taxa]
            if taxa_sweep["f1"].max() > 0:
                best = taxa_sweep.loc[taxa_sweep["f1"].idxmax()]
                print(f"    {taxa}: best F1={best['f1']:.3f} @ t={best['threshold']:.2f}")

    # LOSO CV threshold selection + S22 eval
    print("\n=== LOSO CV threshold selection → S22 test ===")
    all_loso = {}
    for model_name in all_preds:
        print(f"\n--- {model_name} ---")
        results = loso_threshold_selection(
            gt, all_preds[model_name], all_species[model_name],
            evaluable, taxa_map
        )
        all_loso[model_name] = results

        g = results["global"]
        print(f"  Global: t={g['threshold']:.2f} (CV picks: {g['cv_thresholds']})")
        print(f"    S22: P={g['precision']:.3f} R={g['recall']:.3f} F1={g['f1']:.3f}"
              f" (TP={g['tp']} FP={g['fp']} FN={g['fn']})")

        pt = results["per_taxa"]
        print(f"  Per-taxa: {pt['thresholds']}")
        print(f"    S22: P={pt['precision']:.3f} R={pt['recall']:.3f} F1={pt['f1']:.3f}"
              f" (TP={pt['tp']} FP={pt['fp']} FN={pt['fn']})")

    # Summary
    print("\n=== Summary (S22 test set) ===")
    print(f"{'Model':<20} {'Strategy':<12} {'Threshold':<12} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 65)
    for model_name in all_preds:
        r = all_loso[model_name]
        g = r["global"]
        print(f"{model_name:<20} {'global':<12} {g['threshold']:<12.2f} "
              f"{g['precision']:>6.3f} {g['recall']:>6.3f} {g['f1']:>6.3f}")
        pt = r["per_taxa"]
        t_str = ",".join(f"{k[:3]}={v:.2f}" for k, v in sorted(pt["thresholds"].items()))
        print(f"{'':<20} {'per-taxa':<12} {t_str:<12} "
              f"{pt['precision']:>6.3f} {pt['recall']:>6.3f} {pt['f1']:>6.3f}")

    with open(args.output / "threshold_loso_results.json", "w") as f:
        json.dump(all_loso, f, indent=2, default=str)

    print(f"\nArtifacts saved to {args.output}/")


if __name__ == "__main__":
    main()
