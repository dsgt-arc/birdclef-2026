# v5j-era codebase diagnostic worktree

This is a **git worktree** of the main project repo pinned at commit
**`e420633`** — the commit at which v5j-original was trained (2026-03-31).

## Purpose

Investigate the v5j codebase regression discovered 2026-05-10 in exp 031:
today's training pipeline (HEAD) lands at LB **0.897** for v5j_control, but
v5j-original (Mar 31) hit LB **0.906**. v5j-resubmit (today's submission of
the v5j-original ep 19 ckpt) reproduces 0.906 exactly, so inference is clean.

The drift is somewhere in the training path between `e420633` (Mar 31) and
HEAD (May 10). 8 static diagnostics all passed (init bit-identical,
gradients bit-identical, splits bit-identical, libs identical, CUDA stack
identical, data unchanged, optimizer config identical). The drift requires
end-to-end training to manifest.

This worktree's purpose: **retrain v5j_control using the v5j-era source
code (this worktree) in today's environment**. The diagnostic config
`019-perch-guided-sed/config/hgnet_b0_sed_v5j_diag_eracode.yaml` adds
`save_top_k: -1` (preserves all 20 epochs) and a unique run_name, otherwise
identical to v5j-original's training config.

## Run

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-v5j-era/user/murilo/experiments/019-perch-guided-sed
bash sbatch/run.sh config/hgnet_b0_sed_v5j_diag_eracode.yaml
```

Expected walltime: ~40 min on rtx6000 (v5j_control_resave took 40m 36s).

Output:
- Checkpoints: `/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-exp019-v5j-diag-eracode-may10/`
- Logs: `logs/slurm_hgnet_b0_sed_v5j_diag_eracode_{JOBID}.log`
- W&B: project `birdclef-2026-exp031`, run name `2026-exp019-v5j-diag-eracode-may10`

## Interpret

After it finishes, **hash-compare** the diagnostic ep 19 weights vs v5j-original's ep 19 weights:

```bash
source ~/clef/birdclef-2026/.venv/bin/activate
python -c "
import torch, hashlib
def h(p):
    s = torch.load(p, map_location='cpu', weights_only=False)['state_dict']
    keys = sorted(k for k in s if not any(m in k for m in ['cmap','auroc','f1_macro','f1_micro','top1','top5','_val_sc']))
    return hashlib.sha256(b''.join(s[k].cpu().numpy().tobytes() for k in keys)).hexdigest()
print('v5j-original (Mar 31):', h('/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-hgnet-b0-sed-v5j/best-epochepoch=19.ckpt'))
print('diagnostic (May 10): ', h('/storage/project/ps-clef2026_birdclef-0/mgustineli3/experiments/017-mel-overhaul/checkpoints/2026-exp019-v5j-diag-eracode-may10/best-epochepoch=19.ckpt'))
"
```

| Hash result | LB submission needed? | Interpretation |
|---|---|---|
| Identical | NO (LB will be 0.906 deterministically) | Training is fully reproducible across May 10 vs Mar 31 runs of v5j-era code. **HEAD codebase IS the cause** of the 0.897 regression. Bisect the 8 PRs (exp 021/022/025/026/027/030/031) to localize. |
| Different but val/auroc matches | YES | Training is non-deterministic across runs. Submit ep 19 to LB. If 0.906, env-stable; if 0.897, env-shifted. |
| Different and val/auroc shifts | YES | Codebase + env both changed. Need to dig deeper. |

## Cleanup

After the experiment concludes and findings are documented:

```bash
cd /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-2026
git worktree remove /storage/project/ps-clef2026_birdclef-0/mgustineli3/birdclef-v5j-era
```

This removes the worktree (it's a detached HEAD; no upstream branch to delete).

## Cross-references

- `user/murilo/analysis/notes/2026-05-10-v5j-codebase-drift-analysis.md` — initial analysis with ranked hypotheses
- `user/murilo/analysis/notes/2026-05-10-v5j-drift-diagnostics-findings.md` — static diagnostic results (8 causes ruled out)
- `user/murilo/experiments/031-perch-kd-distillation/results.md` § Best-vs-last LB results — the LB data motivating this diagnostic
