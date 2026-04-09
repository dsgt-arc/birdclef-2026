#!/usr/bin/env python3
"""Print a summary table of all analysis results from artifacts/*.json.

Usage: python scripts/summarize_results.py [--artifacts-dir artifacts]
"""

import json
import sys
from pathlib import Path


def main():
    artifacts_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts")

    print("=" * 70)
    print("Exp 106: Analysis Results Summary")
    print("=" * 70)

    # --- EDA ---
    print("\n## EDA / Baselines\n")

    zs = _load(artifacts_dir / "zero_shot_metrics.json")
    if zs:
        print(f"  Zero-shot (Perch readout):  AUROC={zs.get('macro_auroc', '?'):.4f}  "
              f"({zs.get('n_evaluated', '?')} species, {zs.get('n_unmapped', '?')} unmapped)")

    ds = _load(artifacts_dir / "distribution_shift.json")
    if ds:
        print(f"  Domain shift:              {ds.get('frac_significant', 0):.1%} dims shift significantly  "
              f"(median KS={ds.get('median_ks_statistic', 0):.3f})")

    fs = _load(artifacts_dir / "feature_selection.json")
    if fs:
        print(f"  Feature importance:         mean max AUROC={fs.get('mean_max_auroc', 0):.4f}  "
              f"({fs.get('n_species_evaluated', '?')} species)")

    # --- Models ---
    print("\n## Models\n")

    ridge = _load(artifacts_dir / "ridge_metrics.json")
    if ridge:
        for name in ["focal_emb_empirical_bayes", "focal_emb_identity",
                      "sc_emb_identity", "sc_logit_identity"]:
            r = ridge.get(name)
            if r and isinstance(r, dict):
                print(f"  Ridge ({name}):  AUROC={r.get('best_auroc', 0):.4f}  "
                      f"AP={r.get('best_ap', 0):.4f}  λ={r.get('best_lambda', '?')}")

    # --- Structure ---
    print("\n## Structure\n")

    sg = _load(artifacts_dir / "species_graph_summary.json")
    if sg:
        print(f"  Species graph:             {sg.get('n_with_data', '?')}/{sg.get('n_species', '?')} species")

    sr = _load(artifacts_dir / "site_residuals_summary.json")
    if sr:
        print(f"  Site residuals:            {sr.get('n_sites', '?')} sites")

    # --- Prototypes ---
    print("\n## Prototypes (NMF + k-means)\n")

    mp = _load(artifacts_dir / "multi_prototypes_summary.json")
    if mp:
        ev = mp.get("eval", {})
        print(f"  Species prototypes:        {mp.get('total_prototypes', '?')} total "
              f"(K={mp.get('n_prototypes_per_species', '?')} × {mp.get('n_species_with_data', '?')} species)")
        if ev.get("logit_multi_auroc"):
            print(f"    logit multi:  AUROC={ev['logit_multi_auroc']:.4f}  "
                  f"(single: {ev.get('logit_single_auroc', 0):.4f})")
        if ev.get("emb_multi_auroc"):
            print(f"    emb multi:    AUROC={ev['emb_multi_auroc']:.4f}  "
                  f"(single: {ev.get('emb_single_auroc', 0):.4f})")
    else:
        print("  Species prototypes:        NOT YET RUN")

    sp = _load(artifacts_dir / "site_prototypes_summary.json")
    if sp:
        print(f"  Site prototypes:           {sp.get('total_prototypes', '?')} total "
              f"(K={sp.get('n_prototypes_per_site', '?')} × {sp.get('n_sites', '?')} sites)")
    else:
        print("  Site prototypes:           NOT YET RUN")

    pr = _load(artifacts_dir / "prototype_ridge_metrics.json")
    if pr:
        print(f"\n  Prototype Ridge (LOSO CV):")
        for name in ["proto_species", "proto_site", "proto_combined",
                      "proto_plus_emb", "emb_only"]:
            r = pr.get(name)
            if r:
                print(f"    {name:20s}  AUROC={r.get('macro_auroc', 0):.4f}  "
                      f"({r.get('n_features', '?')} features)")
    else:
        print("  Prototype Ridge:           NOT YET RUN")

    # --- Archived (for reference) ---
    print("\n## Archived (broken/superseded)\n")

    pm = _load(artifacts_dir / "prototype_metrics.json")
    if pm:
        print(f"  Old prototype classifier:  AUROC={pm.get('macro_auroc', 0):.4f}  "
              f"({pm.get('n_evaluated', '?')} species) — SUPERSEDED")

    sl = _load(artifacts_dir / "sparse_logistic_metrics.json")
    if sl:
        print(f"  Sparse logistic:           AUROC={sl.get('macro_auroc', 0):.4f}  "
              f"({sl.get('n_evaluated', '?')} species) — BROKEN")

    print()


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()
