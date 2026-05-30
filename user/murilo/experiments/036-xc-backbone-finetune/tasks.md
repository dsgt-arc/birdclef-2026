# Tasks — Exp 036: Unlock the XC backbone (faithful fine-tune recipe)

**Status: CLOSED — REJECT (2026-05-30).** Phase A tested on LB and flat; Phase B/C not pursued.
See `results.md § LB verdict` for the full read.

## Phase A — LR-sanity (sub-claim 1: schedule unlocks the backbone) — DONE, REJECTED

- [x] Build harness on exp 034 (metric suite, flexible optimizer, ClassBalancedMixSamp, SWA,
      configs, sbatch). 290 local tests pass (258 + 32 new).
- [x] Gate 6 (A0 parity): `a0-fullft-repro` reproduces exp 034 full-FT bit-identically (~1e-4
      epoch-by-epoch). Harness faithful; added metric suite did not perturb training. **PASS.**
- [x] Train 4 cells fold 0, 50 ep: a0 (lr 1e-4), a1 (5e-5), a2 (discriminative), a3 (gradual unfreeze).
- [x] Offline screen (best-by-val/auroc): **a1 0.921** > a3 0.914 > a0-peak 0.906 > a2 0.888.
      a1 wins (best offline, cleanest mechanism — lower LR prevents a0's overfit collapse,
      ideal SWA shape).
- [x] Build SWA candidate: `a1-lower-lr/fold0/swa-last5.ckpt` (avg ep45–49, 786 model.* tensors).
- [x] Wire export/upload/kernel for a1 (SWA + last) in `010-submission-pipeline` (4-files pattern;
      smoke-tested: both ckpts load 786 model.*, 0 missing/unexpected; titles slug-resolve;
      `xcfinetune` added to upload mapping guard so 234-identity isn't overwritten by 41-overlap).
- [x] **LB:** a1-SWA **0.881** / a1-last **0.882** vs 0.879 baseline = **flat (within noise).**
      SWA added nothing (≤ last). **Sub-claim 1 REJECTED.**

## Phase B — faithful Sydorskyi data pipeline (sub-claim 2, "main lever") — NOT PURSUED

- [ ] ~~Swap exp 032 datamodule for class-balanced WaveDataset; re-point `b1-class-balanced.yaml`
      `inherit:` to a1; train fold 0; LB A/B.~~ **Not pursued (decision 2026-05-30):** base rate
      ≈ 0 (exp 027 "16+ changes, 0 wins"); offline can't screen it (blind LB shot); deadline
      2026-06-03 EV favors final picks; even optimistic Phase B (~0.90) < our 0.906 single best.
      Assets remain built (`ClassBalancedMixSamp`, `b1-class-balanced.yaml`) if revisited.

## Phase C — PL + PP — NOT PURSUED (gated on Phase A/B ≥ ~0.92, never reached)

## Conclusion

The XC backbone's ~0.88 single-fold ceiling is **schedule-independent** (3 recipes: full-FT
0.879, +KD 0.878, +lower-LR 0.881/0.882). Confirms exp 034's "necessary-not-sufficient" from
the fine-tune angle — the gap to 0.906/0.929 is the **omitted stack** (PL + 5-fold + ensemble),
not the fine-tune. **Pivot:** secure the 0.929 ensemble + PP/TTA final picks; the XC model
stays available as a backbone-diverse ensemble member (separate, cheap test).
