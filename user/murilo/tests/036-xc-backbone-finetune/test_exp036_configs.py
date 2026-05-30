"""Exp 036 Part 5 — config validation (load + inherit chain + key invariants).

Replicates train.py's single-level-recursive `inherit:` deep-merge so we don't import
the heavy train.py module. Verifies save_top_k=-1, a0==exp034-fullft on load-bearing
keys, the A-cell knobs, and the b1 → a2 → _common chain.
"""

from pathlib import Path

import yaml

CONFIGS = Path(
    "/storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026"
    "/user/murilo/experiments/036-xc-backbone-finetune/configs"
)
CELLS = [p for p in CONFIGS.glob("*.yaml") if p.name != "_common.yaml"]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    inherit = cfg.pop("inherit", None)
    if inherit:
        cfg = _deep_merge(load_config(path.parent / inherit), cfg)
    return cfg


def test_all_cells_load_with_required_keys():
    assert CELLS, "no cell configs found"
    for p in CELLS:
        cfg = load_config(p)
        assert "run_name" in cfg, p.name
        for k in ("data", "model", "training", "callbacks", "logging"):
            assert k in cfg, f"{p.name} missing {k}"


def test_save_top_k_minus_one_and_weights_only():
    for p in CELLS:
        mc = load_config(p)["callbacks"]["model_checkpoint"]
        assert mc["save_top_k"] == -1, p.name  # every epoch → SWA-able, no save_last quirk
        assert mc["save_weights_only"] is True, p.name


def test_a0_matches_exp034_fullft_on_loadbearing_keys():
    cfg = load_config(CONFIGS / "a0-fullft-repro.yaml")
    t = cfg["training"]
    assert t["lr"] == 1.0e-4 and t["max_epochs"] == 50 and t["grad_clip_val"] == 1.0
    assert t["optimizer_mode"] == "uniform"
    assert t["warmup_epochs"] == 0 and t["unfreeze_backbone_epoch"] == 0
    assert cfg["model"]["freeze_backbone"] is False
    assert cfg["data"]["use_class_balanced_sampler"] is False


def test_a1_lower_lr():
    assert load_config(CONFIGS / "a1-lower-lr.yaml")["training"]["lr"] == 5.0e-5


def test_a2_discriminative():
    t = load_config(CONFIGS / "a2-discriminative-lr.yaml")["training"]
    assert t["optimizer_mode"] == "discriminative"
    assert t["backbone_lr_mult"] == 0.01
    assert t["warmup_epochs"] == 2


def test_a3_gradual_unfreeze():
    t = load_config(CONFIGS / "a3-gradual-unfreeze.yaml")["training"]
    assert t["unfreeze_backbone_epoch"] == 5


def test_b1_inherits_a2_and_enables_class_balance():
    cfg = load_config(CONFIGS / "b1-class-balanced.yaml")
    assert cfg["data"]["use_class_balanced_sampler"] is True
    # chained inherit b1 → a2 → _common: a2's discriminative knobs come through
    assert cfg["training"]["optimizer_mode"] == "discriminative"
    assert cfg["training"]["backbone_lr_mult"] == 0.01
