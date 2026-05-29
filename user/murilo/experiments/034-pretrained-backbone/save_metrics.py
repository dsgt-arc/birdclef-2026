"""Save structured training metrics to artifacts/ (mirrors exp 032 save_metrics.py).

Reads the per-run results written by train.py
(`outputs/test_results/<run>-fold<N>.json`) and writes a structured
`metrics.json` under `artifacts/training/<run>/fold<N>/`. Run by `_job.sh`
after training (skipped for fast-dev-run).

Usage:
    python save_metrics.py --config configs/xcpretrain-effv2s-frozen.yaml --fold 0
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to the configs/*.yaml used for training")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--artifacts-dir", default="artifacts/training")
    args = ap.parse_args()

    config_path = (
        (EXP_DIR / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    )
    run_name = load_config(config_path)["run_name"]

    results_file = EXP_DIR / "outputs" / "test_results" / f"{run_name}-fold{args.fold}.json"
    if not results_file.is_file():
        raise FileNotFoundError(
            f"No training results at {results_file} — did train.py finish (non-fast-dev)?"
        )
    raw = json.loads(results_file.read_text())

    out_dir = EXP_DIR / args.artifacts_dir / run_name / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment": "034-pretrained-backbone",
        "run_name": run_name,
        "fold": args.fold,
        "config": raw.get("config"),
        "backbone": raw.get("backbone"),
        "freeze_backbone": raw.get("freeze_backbone"),
        "load_pretrained": raw.get("load_pretrained"),
        "max_epochs": raw.get("max_epochs"),
        "val_auroc": raw.get("val_auroc"),
        "val_loss": raw.get("val_loss"),
        "all_metrics": raw.get("all_metrics", {}),
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    out_file = out_dir / "metrics.json"
    out_file.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics → {out_file}  (val_auroc={metrics['val_auroc']})")


if __name__ == "__main__":
    main()
