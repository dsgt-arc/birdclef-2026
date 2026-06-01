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
- [x] **DONE (rtx6000)**: A=9351785, B=9351788, C=9351789 all ran 25 ep clean on the first try
      (~5h50m/cell, no errors). (Jobs 9351455=fast_dev_run smoke, 9351577=gpu-check probe — pre-flight,
      NOT restarts.) **XC-val sanity gate PASS, decisively** — best `val/auroc`: **A 0.9843 (ep23) ·
      B 0.9847 (ep23) · C 0.9856 (ep22)** vs the ≳0.95 bar (top-5 ~0.89 over 2683 sp). AUROC ranks the 3
      within 0.0013 (un-rankable — sanity gate only). Sharpness separates: **C no-mixup sharpest** (val
      f1_micro 0.516 vs A 0.486 / B 0.456; train/auroc 0.994 vs 0.899) — the expected mixup-regularizer
      trade-off; transfer is an **LB-only** question → fine-tune **Cell A first**.
- [x] **Structured metrics regenerated**: `save_metrics.py --config configs/pretrain_cell{A,B,C}_*.yaml`
      → `artifacts/training/pretrain_cell{A,B,C}_*/metrics.json`. Fixed `save_metrics.py`: when
      test_results is absent it now reads best_val_auroc/ckpt from `pretrain_results.json` AND
      reconstructs the **final-epoch `val/*` vector from the W&B summary** (top metrics were null
      before). **`results.md` written + `MODEL-RUNS.md` updated.**

## Stage 2 — fine-tune on 2026 (all 234) — BUILT + LOCAL-TESTED (ready to launch)
Decision (2026-05-31, w/ Murilo): fine-tune **both Cell A & Cell C** and compare on **LB** (the A-vs-C
delta tests whether mixup-regularization in pretraining transfers to the focal→soundscape shift).
**KD-on (match the 0.906 recipe exactly)**; **all 5 folds for both** (compute free), **LB-submit
single-fold A + single-fold C first** (2 pulls = decisive read).
- [x] **Mechanism**: Stage 2 = exp 032's 0.906 KD-SED recipe with ONE arch change — `encoder.model_name`
      B0 → `tf_efficientnetv2_s.in21k` — + warm-start the encoder from our Stage-1 ckpt. Same module
      class (`DistilledSEDFinetuneModule`) → encoder.* keys identical → clean full-copy (780 backbone
      params verified in the real Cell A ckpt; the 4 `sed_head.att/cla` 2683-class params re-init at 234).
- [x] **New code**: `train_finetune.py` (copy of exp 032 train.py; added gated `init.encoder_only` to
      `transfer_pretrained_weights` = copy `encoder.*` only, re-init the whole head + distill_head;
      exp 032 behaviour bit-identical when off). `configs/finetune_cell{A,C}.yaml` (V2-S, KD-on,
      eff-batch 64 = batch32×accum2 to fit rtx6000 + GPU Perch teacher, init.from_checkpoint = Cell A
      epoch23 / Cell C epoch22, encoder_only). `sbatch/{finetune.sh,run_finetune.sh}` (--fold).
- [x] **Local-tested**: 3 new warm-start tests (encoder.* full-copy / head+distill re-init / class-count
      mismatch / exp 032 species-partial-copy parity) + 26 total exp 038 tests PASS; py_compile + bash -n
      clean; both configs parse (KD on, α1.0, tucker256, AttHeadLogitSpace, eff64, 25ep, monitor val/cmap).
- [x] **DONE (Murilo ran)**: pre-flight + 10-job grid (A & C × 5 folds) trained clean, 25 ep, rtx6000
      (~5–6 h). **Final-epoch offline val/auroc (held-out labeled-SS):** Cell A mean **0.9104** (±0.013),
      Cell C mean **0.9085** (±0.009) — a statistical TIE (Δ0.0019 ≪ fold spread), both ≈ the exp 032 B0
      baseline offline (0.908). Offline can't rank A vs C + anti-correlates w/ LB → **LB decides.**
      `test_results/2026-exp038-finetune-cell{A,C}-fold{0-4}.json`. results.md/MODEL-RUNS.md updated.
- [x] **Submission wiring BUILT** (exp 010, commit baecafb): cloned the exp 032 distilled-sed-kd
      pipeline (V2-S backbone only → export_distilled_sed.py + the kernel's Tucker path reused
      UNCHANGED). 4 models: `finetune-cell-{a,c}` (fold0) + `finetune-cell-{a,c}-5fold`. Touched
      `export.sbatch` (export_finetune_cell_single_fold/_5fold + dispatch + species-map glob),
      `upload_kaggle.py` (4 MODELS entries), 4 `kernel-metadata-finetune-cell-*.json`, `submission.py`
      (4 MODEL_SLUGS). Verified bash -n / py_compile / title-slug / cross-file naming.
- [x] **(Murilo) Export + LB — single-fold DONE (2026-05-31).** Exported/uploaded/pushed `finetune-cell-a`
      & `finetune-cell-c` from `experiments/010-submission-pipeline/`; both kernel logs printed the
      `Species mapping: 234 mapped` canary ([[feedback_species_mapping_canary]]) and ran clean. **LB:
      Cell A 0.907 (parity w/ 0.906 KD; +0.025–0.029 over the ~0.88 exp 034/036 XC-backbone ceiling →
      REFUTES "XC-pretrain caps at 0.88"), Cell C 0.901 (A>C by 0.006 → mixup-reg pretrain transfers
      better; offline fold0 anti-ranked: C 0.894 > A 0.887).** results.md/MODEL-RUNS.md updated.
- [x] **(Murilo) 5-fold LB — PARTIAL (2026-05-31).** **`finetune-cell-c-5fold` LB 0.911** (+0.010 over its
      0.901 single fold; **−0.004 below the 0.915 best Tucker-SED ensemble**, below 0.929). **`finetune-cell-a-5fold`
      TIMED OUT** — same EffNetV2-S/inference path as C-5fold → VM variance at the ~73-of-90-min CPU margin
      flagged pre-submit, not a bug. **A-vs-C at the ensemble level UNRESOLVED** (A led C at single-fold).
      results.md/MODEL-RUNS.md updated.
- [x] **(2026-05-31 late, w/ Murilo) A-5fold → A-4fold decision + build.** Chose **trim to 4-fold** (~58–72
      min, safe) over the coin-flip 5-fold retry. **Dropped fold2** (NOT the offline-weakest fold0) via
      `scripts/analyze_fold_coverage.py`: folds are coverage-equivalent; fold2 holds the most true singleton
      blind-spots (`23150`,`23724`); offline val/auroc spread is a val-composition artifact (29→45 val
      species/fold) so we KEPT the LB-verified fold0 (0.907). Built `finetune-cell-a-4fold` reusing the
      5-fold ONNX (no re-export): weights dir + kernel-metadata + upload_kaggle entry + MODEL_SLUGS;
      validated (compile/dry-run/ONNX smoke/commit-run canary). **Murilo submitted to LB — RUNNING.**
- [x] **(2026-05-31 late) Analyzed `ryutoyoda/...eos8-sidecar` (0.95 LB)** → `vendor/kaggle/notebooks/
      birdclef-2026/eos8-sidecar/`. Same ingredients as us (Perch + Tucker SED + ProtoSSM) recombined in
      rank space + heavy PP + taxonomy smoothing. Transferable: **rank-space Perch+SED blend** + **taxonomy
      smoothing** (cheap, on the 0.929 kernel) — next builds. See CONTEXT.md for line refs.
- [ ] **(Next session) (a)** read A-4fold LB (vs C-5fold 0.911 / 0.929; + runtime); **(b)** build the two PP
      variants of the 0.929 ensemble kernel (rank-space blend + taxonomy smoothing); **(c)** Perch + A-4fold
      blend (gated on A-4fold strength; ~76 min); **(d)** final 2 picks (0.929 + best); **(e)** merge
      `mg/038-xc-pretraining` (`--squash`) + ceiling-refute working-note.

## Stage 3 — eval / use
- [ ] Pretrain quality → held-out XC macro-AUC. Downstream → **LB** (labeled-SS anti-correlates).
- [ ] Single-fold LB vs 0.906 (KD baseline) + vs 0.879 (exp 034 XC-backbone). Blend into 0.929 ensemble.

## Out of scope (this experiment)
- The 72 non-bird targets / 25 focal-less insects (BirdSet is birds-only) → separate targeted-XC pull, deferred.
- PL rounds (exp 037 REJECT, 0/5). Pretraining is the lever here.
