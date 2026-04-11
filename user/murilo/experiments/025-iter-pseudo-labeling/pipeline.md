# Iterative Pseudo-Labeling Pipeline

End-to-end process for each pseudo-labeling round. Every round follows the same
6-step loop. The student from one round becomes the teacher (or informs the
teacher strategy) for the next.

## Pipeline Diagram

```
                        ┌─────────────────────────────────────┐
                        │          ITERATION LOOP              │
                        │                                      │
  ┌─────────┐     ┌─────▼──────┐     ┌────────────┐     ┌─────┴──────┐
  │ Teacher  │────►│ 1. Infer   │────►│ 2. EDA /   │────►│ 3. Generate│
  │ Model(s) │     │   Teacher  │     │  Diagnose  │     │   Pseudo-  │
  └─────────┘     └────────────┘     └────────────┘     │   Labels   │
                                                         └─────┬──────┘
                                                               │
  ┌─────────┐     ┌────────────┐     ┌────────────┐     ┌─────▼──────┐
  │ Next     │◄───│ 6. Submit  │◄───│ 5. Export   │◄───│ 4. Train   │
  │ Round    │     │  to LB     │     │   ONNX     │     │   Student  │
  └────┬─────┘     └────────────┘     └────────────┘     └────────────┘
       │
       │  Student becomes teacher
       │  (or informs teacher strategy)
       │
       └──────────────────────────────────────────────────────────┐
                                                                  │
  ┌─────────┐     ┌────────────┐     ┌────────────┐     ┌────────▼───┐
  │ Student  │────►│ 1. Infer   │────►│ 2. EDA /   │────►│ 3. Generate│
  │ (new     │     │  Student   │     │  Diagnose  │     │   v(N+1)   │
  │ teacher) │     │  on all    │     │            │     │   PLs      │
  └─────────┘     └────────────┘     └────────────┘     └────────────┘
                        ...continues...
```

## The 6 Steps

### Step 1: Infer — Run teacher/student on all data

Extract embeddings and logits from the model on both `train_audio` (focal)
and `train_soundscapes` (labeled + unlabeled).

**Script**: `user/murilo/inference/infer.py`
**Configs**: `user/murilo/inference/config/`
**SLURM**: `user/murilo/inference/sbatch/run.sh`

```bash
cd user/murilo/inference

# Create config (copy template, update checkpoint_path and run_name)
# Template: config/hgnet_b0_sed_v5j.yaml

# Run inference on all data
bash sbatch/run.sh config/<model_config>.yaml --source all
```

**Output**: `shared/inference/birdclef-2026/<run_name>/{train_audio,train_soundscapes}/`
- `logits.parquet` — clipwise logits (N x 234)
- `embeddings.parquet` — backbone embeddings (N x 1024)
- `framewise_logits.parquet` — frame-level logits

**Time**: ~30 min on RTX 6000

---

### Step 2: EDA / Diagnose — Analyze model predictions

Two scripts depending on context:

**A) First round (no previous student to compare)**:
Use `inference_eda.py` to understand teacher coverage, calibration, and
pseudo-label quality.

```bash
python user/murilo/scripts/inference_eda.py --model <teacher_model>
```

**B) After a round (compare baseline vs student)**:
Use `diagnose_regression.py` to do A/B comparison broken down by species tier.

```bash
python user/murilo/experiments/025-iter-pseudo-labeling/diagnose_regression.py \
  --baseline-model <baseline> \
  --student-model <student> \
  --pseudo-label-path shared/pseudo-labels/birdclef-2026/<version>/pseudo_labels.parquet \
  --tier-path shared/pseudo-labels/birdclef-2026/<version>/tier_assignment.csv
```

**Diagnostic sections**:

| Section | Question | Key metric |
|---------|----------|------------|
| A: Labeled AUC delta | Did PL hurt seen species (75)? | Macro AUC change |
| B: Focal accuracy delta | Did PL hurt focal discrimination? | Top-1 acc change |
| C: Confidence by tier | Which tiers gained/lost? | Coverage @>0.5 per tier |
| D: Per-species shift | Which species moved most? | Mean prob delta |
| E: PL agreement | Did student learn the PLs? | Agreement rate on PL>0.5 |

**Decision point**: Diagnosis results determine whether to proceed, adjust
parameters, or pivot strategy for the next round.

---

### Step 3: Generate Pseudo-Labels

Config-driven generation. Each round gets a versioned config and output dir.

**Script**: `user/murilo/experiments/025-iter-pseudo-labeling/generate_pseudo_labels.py`
**Configs**: `user/murilo/experiments/025-iter-pseudo-labeling/configs/`

```bash
cd user/murilo/experiments/025-iter-pseudo-labeling

# Generate pseudo-labels from config
python generate_pseudo_labels.py --config configs/<version>.yaml

# Verify stats
python generate_pseudo_labels.py --config configs/<version>.yaml --stats-only
```

**Output**: `shared/pseudo-labels/birdclef-2026/<version>/`
- `pseudo_labels.parquet` — merged soft labels (127K x 234)
- `sed_pseudo_labels.parquet` — SED component
- `perch_pseudo_labels.parquet` — Perch component
- `tier_assignment.csv` — per-species tier
- `stats.json` — coverage statistics
- `config.yaml` — copy of generation config

**Config template** (`configs/v1.yaml`):

```yaml
version: v1
parent_version: null

sed:
  source_dir: shared/inference/birdclef-2026/<teacher>/train_soundscapes
  n_folds: 4
  fold_strategy: max
  temperature: 0.6
  power_transform: 1.5
  tiers: [A, B]

perch:
  source_path: shared/inference/.../perch_234_raw_probs.parquet
  strategy: per_species_z_normalization
  z_threshold: 2.0       # Set to 999.0 to disable Perch
  min_std: 0.01
  power_transform: 1.5
  tiers: [C]

output_dir: shared/pseudo-labels/birdclef-2026/<version>
```

---

### Step 4: Train Student

Train v5j recipe on focal + labeled soundscapes + pseudo-labeled soundscapes.
The ONLY changes from baseline v5j are data-related parameters.

**Script**: `user/murilo/experiments/025-iter-pseudo-labeling/train.py`
**Configs**: `user/murilo/experiments/025-iter-pseudo-labeling/configs/`
**SLURM**: `user/murilo/experiments/025-iter-pseudo-labeling/sbatch/run.sh`

```bash
cd user/murilo/experiments/025-iter-pseudo-labeling

# Sanity check first (5 batches, verify batch composition)
bash sbatch/run.sh configs/<train_config>.yaml --fold 0 --fast-dev-run \
  --pl-path /storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/<version>/pseudo_labels.parquet

# Full training (single fold, 90/10 split)
bash sbatch/run.sh configs/<train_config>.yaml --fold 0 \
  --pl-path /storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/<version>/pseudo_labels.parquet

# 4-fold (if single fold improves on LB)
bash sbatch/run.sh configs/<train_config>.yaml --all-folds \
  --pl-path /storage/project/ps-clef2026_birdclef-0/shared/pseudo-labels/birdclef-2026/<version>/pseudo_labels.parquet
```

**Key training parameters** (in `<train_config>.yaml`):

| Parameter | v1 (failed) | v2-sed-only | Notes |
|-----------|-------------|-------------|-------|
| `pseudo_label_weight` | 0.5 | 0.5 | Loss weight for PL samples (GT=1.0) |
| `pseudo_label_batch_fraction` | 0.5 | 0.25 | Fraction of each batch from PL |
| All other params | v5j | v5j | Recipe unchanged |

**Output**:
- Checkpoint: `<project>/experiments/025-iter-pseudo-labeling/checkpoints/<run_name>/`
- TensorBoard: `outputs/tensorboard/`
- Metrics: `outputs/test_results/<run_name>.json`
- Structured: `artifacts/training/<config_stem>/metrics.json` (via `save_metrics.py`)

**Time**: ~1.5h on RTX 6000 (20 epochs, single fold)

---

### Step 5: Export to ONNX

Convert checkpoint to ONNX for Kaggle submission. Requires adding the model
to the exp 010 submission pipeline (4 files).

**Pipeline**: `user/murilo/experiments/010-submission-pipeline/`
**Guide**: `user/murilo/experiments/010-submission-pipeline/adding-new-models.md`

```bash
cd user/murilo/experiments/010-submission-pipeline

# Export checkpoint to ONNX (GPU job)
bash sbatch/run.sh export --model <model-slug>

# Upload weights dataset to Kaggle
python scripts/upload_kaggle.py --model <model-slug>

# Wait 2-3 min for new datasets, then push kernel
python scripts/upload_kaggle.py --model <model-slug> --push-kernel
```

**4 files to edit for each new model** (see `adding-new-models.md`):
1. `sbatch/export.sbatch` — export function + case dispatch (2 places)
2. `scripts/upload_kaggle.py` — MODELS dict entry
3. `kaggle_kernel/submission.py` — MODEL_SLUGS list
4. `kaggle_kernel/kernel-metadata-<slug>.json` — new file

---

### Step 6: Submit to LB

Submit the Kaggle kernel to the competition and record the result.

```bash
# Check kernel status
kaggle kernels status murilogustineli/birdclef-2026-<model-slug>
```

**Record in**: `results.md`, `tasks.md`, `CONTEXT.md`

**Decision gate**:

| LB Result | Interpretation | Next Action |
|-----------|---------------|-------------|
| Better than baseline | PL strategy works | 4-fold + ensemble, then next round |
| Matches baseline | PL neutral | Adjust parameters (weight, fraction, filtering) |
| Worse than baseline | PL hurts | Diagnose (Step 2), then adjust or pivot |

---

## Round History

| Round | Config | Teacher | PL Count | Batch Frac | LB | Delta vs 0.903 |
|-------|--------|---------|----------|------------|-----|----------------|
| v1 | `v1.yaml` | SED 4-fold + Perch z-norm | 347K | 0.50 | 0.897 | -0.006 |
| v2-sed-only | `v2_sed_only.yaml` | SED 4-fold only | ~198K | 0.25 | — | — |

## File Map

```
025-iter-pseudo-labeling/
├── pipeline.md                    # This file
├── proposal.md                    # Full methodology (12 sections)
├── tasks.md                       # Task tracking per phase
├── results.md                     # Results per round
├── README.md                      # Overview + directory structure
├── generate_pseudo_labels.py      # Step 3: PL generation
├── train.py                       # Step 4: Student training
├── save_metrics.py                # Post-training metrics extraction
├── diagnose_regression.py         # Step 2b: A/B diagnostic comparison
├── configs/
│   ├── v1.yaml                    # Round 1 PL generation config
│   ├── v1_train.yaml              # Round 1 training config
│   ├── v2_sed_only.yaml           # Round 2 PL generation (Perch disabled)
│   └── v2_sed_only_train.yaml     # Round 2 training (25% PL batch)
├── sbatch/
│   ├── _job.sh                    # SLURM job template
│   └── run.sh                     # Launcher (--pl-path, --fold, etc.)
├── artifacts/training/            # Structured metrics per variant
├── outputs/
│   ├── tensorboard/               # Training curves
│   └── test_results/              # Raw result JSONs
└── findings/
    └── v1-pseudo-labels.md        # v1 generation analysis

Inference (Step 1):
  user/murilo/inference/
  ├── infer.py                     # Inference script
  ├── config/                      # Model configs (1 per model)
  └── sbatch/run.sh                # SLURM launcher

EDA (Step 2a):
  user/murilo/scripts/inference_eda.py

Submission (Steps 5-6):
  user/murilo/experiments/010-submission-pipeline/
  └── adding-new-models.md         # 4-file checklist

Shared outputs (large, not version-controlled):
  shared/inference/birdclef-2026/<model>/          # Step 1 outputs
  shared/pseudo-labels/birdclef-2026/<version>/    # Step 3 outputs
```
