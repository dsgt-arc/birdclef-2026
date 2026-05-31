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

## Stage 1 — pretrain EffNetV2-S (A100, overnight after smoke) — NOT YET BUILT
- [ ] Supervised-classification LightningModule (multi-label over N pretrain species; secondary labels
      as extra positives). Backbone **EffNetV2-S, ImageNet-21k init** (`tf_efficientnetv2_s.in21k`).
- [ ] Tucker mel (n_mels=256, sr=32000) to match the SED stack; BCE+Focal + LS 0.05; AdamW cosine→1e-6;
      ~15–20 ep; no class balancing; `save_top_k=-1`; held-out XC val (macro-AUC) = pretrain signal.
- [ ] **Smoke** (~50 sp, 1 ep): verify loop/mel/loss/throughput/checkpointing, THEN overnight scale-up.

## Stage 2 — fine-tune on 2026 (all 234) — NOT YET BUILT
- [ ] Re-init 234-class head; reuse SED fine-tune recipe + domain-shift aug (MixUp/bg-mix/SpecAug).
      Non-birds learned here (pretraining didn't see them). Discard pretrain head, keep backbone.

## Stage 3 — eval / use
- [ ] Pretrain quality → held-out XC macro-AUC. Downstream → **LB** (labeled-SS anti-correlates).
- [ ] Single-fold LB vs 0.906 (KD baseline) + vs 0.879 (exp 034 XC-backbone). Blend into 0.929 ensemble.

## Out of scope (this experiment)
- The 72 non-bird targets / 25 focal-less insects (BirdSet is birds-only) → separate targeted-XC pull, deferred.
- PL rounds (exp 037 REJECT, 0/5). Pretraining is the lever here.
