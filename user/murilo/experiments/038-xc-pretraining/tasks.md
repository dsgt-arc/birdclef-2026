# Tasks — Exp 038 Broad Neotropical XC Pretraining → fine-tune on 2026

Claude builds + local-tests; **Murilo runs the HF download + A100 pretrain** ([[feedback_user_runs_commands]]).
Tests: `user/murilo/tests/038-xc-pretraining/`. Honest ceiling ~0.88–0.90 (paper + ensemble diversity).

## Stage 0 — curate (no GPU)
- [x] `scripts/curate_birdset_neotropical.py`: pure-pandas `curate()` (Neotropical-occurring ∪ 162 bird
      targets; lat/long bbox; cap 500/sp, min 10, q A/B; manifest + species_list + size report). 5 tests PASS.
- [x] **Curation DONE** (Murilo ran): metadata via `XCL_metadata.parquet` (rev=data, 99.9 MB, no audio;
      `datasets` 4.x can't run BirdSet.py → hf_hub_download direct). Result: **2,683 Neotropical bird
      species, 160,542 recordings, 2,202 h, 158/162 bird targets covered** (`palhor3`/`strher2` absent,
      2 thin). ebird_code is str (no decode). Manifest + species_list + report → `shared/pretrain/birdset-neotropical/`.
- [x] **Stage 0b downloader BUILT** — `scripts/download_birdset_audio.py` (stream-extract, --shard-dir)
      + `sbatch/download_birdset_audio.sh` (CPU SLURM **worker**, #SBATCH headers modeled on exp 037
      `_job.sh`; $TMPDIR transient shards, .env HF_TOKEN, reuses interactive venv). 2 extract-core
      tests PASS. filepath = bare XC id → all 98 shards pulled, extract our basenames, delete; peak
      ~155 GB; resumable. **Submit with `sbatch` (NOT `bash` — bash runs it in the current shell).**
- [x] **Download DONE** (Murilo ran via `sbatch`): all 98 shards pulled → extracted →
      `shared/pretrain/birdset-neotropical/audio/<ebird_code>/<XCid>.ogg`. **~151,261 files on disk /
      2,683 species; 9,281 (~5.8%) genuinely missing** (not in any tarball → `missing_files.json`).
      Audio confirmed **32 kHz mono OGG** (no resample). The datamodule's `require_audio=True` drops the
      missing rows → **148,427 train / 2,834 val** at val_fraction 0.02.

## Stage 1 — pretrain EffNetV2-S — BUILT + RUNNING (3-cell grid on rtx6000)
Decision: **SED-mirror** (reuse `DistilledSEDFinetuneModule` UNCHANGED, num_classes=2683, KD off) so
the backbone trains under the exact downstream regime. **Strategy = proven primary + parallel
insurance arms** (compute free; LB is the only real ranker → XC-val macro-AUC is a sanity gate only).
- [x] **New code**: `src/birdclef/data/xc_pretrain_dataset.py` (`XCPretrainDataset` multi-hot incl.
      in-vocab secondaries / random 5 s crop / `apply_aug`+mixup ported from `FocalDS`; + stratified
      per-species `XCPretrainDataModule`); gated additive `loss_type="focalbce"` + `label_smoothing`
      in `DistilledSEDFinetuneModule` (bce path bit-identical, lazy torchvision import); `train.py`;
      `configs/_common.yaml` + 3 cells; `pyproject.toml`; `sbatch/{run.sh,pretrain.sh}` (+ `--gpu-check`,
      `--batch-size`/`--accumulate` overrides, TF32); `save_metrics.py` (exp-037 metric parity).
- [x] **3 cells** (EffNetV2-S IN21k, Tucker mel n_mels=256, AttHeadLogitSpace, dual-supervision,
      25 ep save_top_k=-1, lr 5e-4, **effective batch 128**): **A** dual-BCE+LS0.05+full-aug (primary),
      **B** focal-BCE, **C** no-mixup.
- [x] **Verified local**: 23 new tests + 238 regression (032/033/037) pass; config merge; real-OGG
      dataset load+crop+mixup; CPU fwd→loss→backward through V2-S (embed_dim=1280) for bce+focalbce.
- [x] **fast-dev-run PASS** (a100, batch 128) + **gpu-check PASS** (rtx6000, batch 64): GPU-bound
      87% util, data-load 1.4% (num_workers=8 fine), **~10 min/epoch → ~4.2 h/cell**; peak reserved
      14.2 GB/24 GB. `outputs/gpu_check/*.{txt,png}`.
- [~] **RUNNING (rtx6000, launched ~00:14 May 31)**: A=job 9351785, B=9351788, C=9351789. eff batch
      128 (~1160 opt steps/ep). epoch 0: cellA val/auroc 0.513 (climbing); `epoch00.ckpt` saved. ETA ~04:30.
      **These were launched BEFORE the metric-parity commit** → live metrics in W&B/TB + `pretrain_results.json`
      (best_val_auroc), but NOT `outputs/test_results/` nor auto-`save_metrics`.
- [ ] **(next session) after they finish**: regenerate structured metrics —
      `python save_metrics.py --config configs/pretrain_cell{A,B,C}_*.yaml` (fallback reads
      `pretrain_results.json`). Compare the 3 `best_val_auroc` (sanity gate). Cell A backbone → Stage 2.

## Stage 2 — fine-tune on 2026 (all 234) — NOT YET BUILT
- [ ] Re-init 234-class head; reuse SED fine-tune recipe + domain-shift aug (MixUp/bg-mix/SpecAug).
      Non-birds learned here (pretraining didn't see them). Discard pretrain head, keep backbone.

## Stage 3 — eval / use
- [ ] Pretrain quality → held-out XC macro-AUC. Downstream → **LB** (labeled-SS anti-correlates).
- [ ] Single-fold LB vs 0.906 (KD baseline) + vs 0.879 (exp 034 XC-backbone). Blend into 0.929 ensemble.

## Out of scope (this experiment)
- The 72 non-bird targets / 25 focal-less insects (BirdSet is birds-only) → separate targeted-XC pull, deferred.
- PL rounds (exp 037 REJECT, 0/5). Pretraining is the lever here.
