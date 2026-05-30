#!/usr/bin/env python3
"""Post-hoc SWA (Stochastic Weight Averaging) for exp 036 — the selection-free
checkpoint candidate, robust to the untrustworthy CV (see proposal.md).

We train with `save_top_k=-1` (every epoch saved), then AVERAGE the `model.*` weights
of the last-K epoch checkpoints into one. Float tensors (incl. BN running_mean/var) are
averaged; integer buffers (e.g. BN `num_batches_tracked`) are copied from the LAST ckpt
(averaging a counter is meaningless). The output is a Lightning-style `{"state_dict": ...}`
with identical `model.*` keys → consumed UNCHANGED by
`experiments/010-submission-pipeline/scripts/export_xcpretrain.py` (which reads `state_dict`
and strips the `model.` prefix). last / best-by-val are just specific epoch ckpts; SWA is led.

Reference: exp 018 ref_yuriy_protossm_v5.py weight-averaging; Sydorskyi uses SWA.

Usage:
    python swa_average.py --ckpt-dir .../checkpoints/<run>/fold0 --last-k 5 \
        --output .../checkpoints/<run>/fold0/swa-last5.ckpt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def _epoch_num(path: Path) -> int | None:
    """Parse the epoch from a checkpoint filename; None for non-epoch ckpts (e.g. last.ckpt)."""
    if "last" in path.stem:
        return None
    nums = re.findall(r"\d+", path.stem)  # "epochepoch=43" or "epoch43" → ["43"]
    return int(nums[-1]) if nums else None


def select_ckpts(ckpt_dir: Path, last_k: int, glob: str = "*.ckpt") -> list[Path]:
    """The last-K epoch checkpoints, sorted ascending by epoch (excludes last.ckpt)."""
    epoch_ckpts = [(e, p) for p in ckpt_dir.glob(glob) if (e := _epoch_num(p)) is not None]
    if not epoch_ckpts:
        raise FileNotFoundError(f"No epoch checkpoints matching {glob!r} in {ckpt_dir}")
    epoch_ckpts.sort(key=lambda ep: ep[0])
    return [p for _, p in epoch_ckpts[-last_k:]]


def _model_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    """Load a Lightning ckpt and return only its `model.*` state_dict entries."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    return {k: v for k, v in state.items() if k.startswith("model.")}


def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Average float tensors across state_dicts; copy non-float (int buffers) from the last.

    All inputs must share the exact same key set. Returns the averaged state_dict.
    """
    if not state_dicts:
        raise ValueError("no state_dicts to average")
    keys = set(state_dicts[0])
    for sd in state_dicts[1:]:
        if set(sd) != keys:
            raise ValueError("state_dict key sets differ across checkpoints")
    n = len(state_dicts)
    last = state_dicts[-1]
    out: dict[str, torch.Tensor] = {}
    for k in state_dicts[0]:
        if last[k].is_floating_point():
            acc = torch.zeros_like(last[k], dtype=torch.float64)
            for sd in state_dicts:
                acc += sd[k].to(torch.float64)
            out[k] = (acc / n).to(last[k].dtype)
        else:
            out[k] = last[k].clone()  # int buffers (num_batches_tracked) — take last
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", required=True, help="dir with per-epoch *.ckpt (save_top_k=-1)")
    p.add_argument("--last-k", type=int, default=5, help="average the last K epoch checkpoints")
    p.add_argument("--glob", default="*.ckpt")
    p.add_argument("--output", required=True, help="output SWA checkpoint path")
    args = p.parse_args()

    ckpts = select_ckpts(Path(args.ckpt_dir), args.last_k, args.glob)
    print(f"SWA over {len(ckpts)} ckpts:")
    for c in ckpts:
        print(f"  {c.name}")
    states = [_model_state(c) for c in ckpts]
    avg = average_state_dicts(states)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": avg, "swa": {"source_ckpts": [c.name for c in ckpts], "k": len(ckpts)}},
        out,
    )
    print(f"Wrote SWA checkpoint ({len(avg)} model.* tensors) → {out}")


if __name__ == "__main__":
    main()
