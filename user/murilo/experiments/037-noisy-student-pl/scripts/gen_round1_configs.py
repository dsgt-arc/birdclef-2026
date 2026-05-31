"""Exp 037 — emit the round-1 ablation grid as thin configs over `_common.yaml`.

Factorial (15 cells), all sampler=mixsamp 50/50, 25 ep, steps_per_epoch=500 (inherited):
  KD regime  A = KD-on,  detach   (PL → head only;        use_perch_distill=true,  kd_detach_head=true)
             B = KD-off            (PL → whole student e2e; use_perch_distill=false)
             C = KD-on,  no-detach (PL → head + backbone;  use_perch_distill=true,  kd_detach_head=false)
  PL transform  raw (faithful 0.929 blend) | t2 (temperature-soften)
  PL aug        augF (control, no waveform noise) | augT (noise-the-student)
  → A/B/C × {raw,t2} × {augF,augT} = 12 soft-BCE cells
  + SoftAUC on {A,B,C} × raw × augT = 3 cells   (ranking loss, the deep-research recipe)

Run: ~/clef/birdclef-2026/.venv/bin/python scripts/gen_round1_configs.py
"""

from __future__ import annotations

from pathlib import Path

CFG_DIR = Path(__file__).resolve().parent.parent / "configs"
PL_ROOT = "/storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026"
PARQUET = {
    "raw": f"{PL_ROOT}/v037-ens-raw/ensemble_pseudo_labels.parquet",
    "t2": f"{PL_ROOT}/v037-ens-t2/ensemble_pseudo_labels.parquet",
}
# KD regime → (use_perch_distill, kd_detach_head, blurb)
KD = {
    "A": (True, True, "KD-on detach (PL->head only)"),
    "B": (False, None, "KD-off (PL->whole student e2e)"),
    "C": (True, False, "KD-on no-detach (PL->head+backbone)"),
}


def emit(kd: str, transform: str, aug: bool, softauc: bool) -> tuple[str, str]:
    use_kd, detach, blurb = KD[kd]
    aug_tag = "augT" if aug else "augF"
    name = f"r1_{kd}_{transform}_{aug_tag}" + ("_softauc" if softauc else "")
    run_name = f"2026-exp037-{name.replace('_', '-')}"
    lines = [
        f"# {blurb} | transform={transform} | PL aug={aug} | loss={'softauc' if softauc else 'bce'}",
        "# Inherits _common.yaml (sampler=mixsamp 50/50, 25 ep, steps_per_epoch=500, KD branch).",
        f'run_name: "{run_name}"',
        "pseudo_label:",
        f'  path: "{PARQUET[transform]}"',
        f"  aug: {str(aug).lower()}",
        "training:",
        f"  use_perch_distill: {str(use_kd).lower()}",
    ]
    if detach is not None:  # only meaningful when KD on
        lines.append(f"  kd_detach_head: {str(detach).lower()}")
    if softauc:
        lines.append('  loss_type: "softauc"')
    return name, "\n".join(lines) + "\n"


def main() -> None:
    CFG_DIR.mkdir(exist_ok=True)
    written = []
    # 12 soft-BCE cells
    for kd in ("A", "B", "C"):
        for transform in ("raw", "t2"):
            for aug in (False, True):
                name, body = emit(kd, transform, aug, softauc=False)
                (CFG_DIR / f"{name}.yaml").write_text(body)
                written.append(name)
    # 3 SoftAUC cells (raw, augT, one per KD regime)
    for kd in ("A", "B", "C"):
        name, body = emit(kd, "raw", aug=True, softauc=True)
        (CFG_DIR / f"{name}.yaml").write_text(body)
        written.append(name)
    print(f"Wrote {len(written)} configs to {CFG_DIR}:")
    for n in written:
        print(f"  {n}.yaml")


if __name__ == "__main__":
    main()
