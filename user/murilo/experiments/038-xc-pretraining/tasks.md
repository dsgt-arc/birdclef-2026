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
- [ ] **(Murilo, SLURM)** from the experiment dir: `mkdir -p logs` (one-time), then
      `sbatch sbatch/download_birdset_audio.sh --limit-shards 2` (smoke) → then `sbatch sbatch/download_birdset_audio.sh`
      (full). audio → `shared/pretrain/birdset-neotropical/audio/<ebird_code>/<XCid>.ogg`; `missing_files.json` at end.

## Stage 1 — pretrain EffNetV2-S (A100, overnight after smoke) — BUILT
Decision: **SED-mirror** (reuse `DistilledSEDFinetuneModule` UNCHANGED, num_classes=2683, KD off) so
the backbone trains under the exact downstream regime. **Strategy = proven primary + parallel
insurance arms** (compute free; LB is the only real ranker → XC-val macro-AUC is a sanity gate only).
- [x] **New code**: `src/birdclef/data/xc_pretrain_dataset.py` (`XCPretrainDataset` multi-hot incl.
      in-vocab secondaries / random 5 s crop / `apply_aug`+mixup ported from `FocalDS`; + stratified
      per-species `XCPretrainDataModule`); gated additive `loss_type="focalbce"` + `label_smoothing`
      in `DistilledSEDFinetuneModule` (bce path bit-identical, lazy torchvision import); `train.py`;
      `configs/_common.yaml` + 3 cells; `pyproject.toml`; `sbatch/{run.sh,pretrain.sh}`.
- [x] **3 cells** (EffNetV2-S IN21k, Tucker mel n_mels=256, AttHeadLogitSpace, 25 ep save_top_k=-1,
      lr 5e-4, batch 128): **A** dual-BCE+LS0.05+full-aug (primary), **B** focal-BCE, **C** no-mixup.
- [x] **Verified local**: 23 new tests + 238 regression (032/033/037) pass; config merge; real-OGG
      dataset load+crop+mixup; CPU fwd→loss→backward through V2-S (embed_dim=1280) for bce+focalbce.
- [ ] **(Murilo, GPU) smoke**: `bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml --fast-dev-run`
      → then a 1-epoch run to measure it/s + confirm checkpointing + XC-val/auroc, THEN launch all 3.
- [ ] **(Murilo, A100) full**: `bash sbatch/run.sh configs/pretrain_cellA_dualbce.yaml` (+ B, + C in
      parallel). Rank by XC-val macro-AUC (sanity); fine-tune Cell A backbone to LB first (Stage 2).

## Stage 2 — fine-tune on 2026 (all 234) — NOT YET BUILT
- [ ] Re-init 234-class head; reuse SED fine-tune recipe + domain-shift aug (MixUp/bg-mix/SpecAug).
      Non-birds learned here (pretraining didn't see them). Discard pretrain head, keep backbone.

## Stage 3 — eval / use
- [ ] Pretrain quality → held-out XC macro-AUC. Downstream → **LB** (labeled-SS anti-correlates).
- [ ] Single-fold LB vs 0.906 (KD baseline) + vs 0.879 (exp 034 XC-backbone). Blend into 0.929 ensemble.

## Out of scope (this experiment)
- The 72 non-bird targets / 25 focal-less insects (BirdSet is birds-only) → separate targeted-XC pull, deferred.
- PL rounds (exp 037 REJECT, 0/5). Pretraining is the lever here.
