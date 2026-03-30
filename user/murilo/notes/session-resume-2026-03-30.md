# Session Resume Prompt — 2026-03-30

Copy everything below this line into a new Claude session to resume work.

---

We're working on BirdCLEF 2026 (Kaggle competition) under user/murilo/. Read these files to get full context on where we left off:

1. `user/murilo/experiments/018-perch-probes/tasks.md` — exp 018 status (ProtoSSM v4.1 submitted, awaiting LB)
2. `user/murilo/experiments/019-perch-guided-sed/tasks.md` — exp 019 status (5 ablation runs training on PACE)
3. `user/murilo/experiments/019-perch-guided-sed/proposal.md` — exp 019 full plan (3 phases)
4. `user/murilo/MODEL-RUNS.md` — all model runs and LB scores
5. `user/murilo/notes/gap-analysis-hgnet-vs-effnet.md` — HGNet vs EfficientNet comparison driving exp 019

Branch: `mg/018-perch-ensemble`

## What's running RIGHT NOW (as of 2026-03-30 evening):

### Exp 018: Perch ProtoSSM v4.1 — running Kaggle inference (~90 min)
- Notebook version 4 submitted to competition, running on ~600 test files
- v4 scored **0.918 LB**; v4.1 has bug fixes (OOF prior by filename, per-stage clipping, train on all data)
- OOF metrics improved significantly: baseline AUROC 0.66→0.81, ensemble AUC 0.91→0.94
- Target: ~0.920-0.923 LB
- **Action when done**: Check LB score, update tasks.md step 9.10, update MODEL-RUNS.md

### Exp 019: 5 SED ablation runs — training on PACE (~4h each)
All start from v4b (0.860 LB) with single-variable changes:
- v5a: BCEWithLogitsLoss (replaces soft CE)
- v5b: MixUp α=1.0, θ=0.8, warmup=5 (stronger augmentation)
- v5c: weight_decay=1e-4 (from 0.01, 100x less regularization)
- v5d: LSEHead (Log-Sum-Exp temporal pooling, replaces AttHead)
- v5e: HGNetV2-B0 backbone (replaces EfficientNet-B0, embed_dim=1024)

**Action when done**:
1. Check val/auroc for all 5 runs (baseline v4b = 0.865)
2. Read TensorBoard logs: `scripts/tb_metrics.py`
3. Update tasks.md step 1.6 with results
4. Proceed to Step 2: combine winners on both EfficientNet + HGNet tracks
5. Best variants get submitted to LB (target: ≥0.870 single fold)

## Code changes made this session:

### New package: `src/birdclef/perch/` (8 modules, 140 tests)
- ProtoSSM v4 pipeline: models, training, losses, postprocessing, ensemble, data_utils, config
- Kernel: `018-perch-probes/kaggle_kernel/submission_perch_protossm.py` (2756 lines)

### SED infrastructure changes (for exp 019):
- `src/birdclef/encoders/efficientnet.py` — generic timm support via `in_chans=1` (enables HGNetV2)
- `src/birdclef/heads/sed_head.py` — added LSEPooling + LSEHead classes
- `src/birdclef/lightning/sed_finetune.py` — added mixup_warmup_epochs, mixup_theta, weighted label mixing
- `experiments/017-mel-overhaul/train.py` — registered LSEHead + new mixup params
- `experiments/019-perch-guided-sed/` — 5 YAML configs, copied train infrastructure

### Key reference files:
- `user/murilo/notes/ttahara-hgnetv2-b0-annotated.md` — HGNet reference (0.884 LB)
- `user/murilo/notes/yuriy-protossm-v5-annotated.md` — ProtoSSM reference (0.925 LB)
- `user/murilo/notes/competition-strategy.md` — overall strategy

## Competition context:
- Current #1: 0.939 LB
- Our best: 0.918 (Perch ProtoSSM v4), 0.860 (SED v4b)
- 2 months until deadline (June 3, 2026)
- 5 submissions/day, unlimited PACE GPU compute
- Top scores likely require: multi-model ensemble + pseudo-labeling on unlabeled data

## Next steps after results come in:
1. Record v4.1 LB score → close out exp 018
2. Analyze 5 ablation results → pick winners for Step 2 combo runs
3. Train combined models (EfficientNet track + HGNet track)
4. 4-fold ensemble of best variant
5. Phase B: Perch pseudo-label extraction on 10K+ unlabeled soundscapes (PACE GPU)
6. Phase C: Multi-round pseudo-label SED training (target: 0.900-0.915)
