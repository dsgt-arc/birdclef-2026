# Kaggle baseline: exp 300 AST ridge head

The AST sibling of `../kaggle_kernel/` (Perch). Same backbone-agnostic ridge head, but
the encoder is **AST run in-kernel via HuggingFace `transformers`** (no ONNX) reusing the
exact `ASTEncoder` forward (ASTFeatureExtractor + ASTModel, **cls** pooling) that built the
training cache. The head covers 234/234 species, so there is **no logit fallback**.

## Pipeline
```
audio (16 kHz) → 12×5 s windows → AST (ASTFeatureExtractor mel → ASTModel) → cls emb(768)
              → (emb-mean)/std @ W → clip[0,1]            # head_ast.npz: 234 species
              → submission.csv (row_id + 234 species cols)
```

## Inputs (Kaggle, internet OFF)
| kind | slug | contents |
|------|------|----------|
| competition | `birdclef-2026` | `test_soundscapes/`, `sample_submission.csv` |
| dataset | `acmiyaguchi/birdclef-2026-exp300-ast-head` | `head_ast.npz` |
| dataset | `acmiyaguchi/ast-audioset-snapshot` | HF snapshot: `config.json`, `preprocessor_config.json`, weights |

**Parity:** the cache was built with `transformers<5` + torch 2.11 (PACE `.venv-300`). The
kernel reproduces that forward; `sbatch/dryrun-kernel-ast.sbatch` asserts a held-out
window's kernel embedding matches the cached cls embedding (cosine > 0.999). torch +
transformers + torchaudio + soundfile come from the Kaggle base image (no deps dataset).

## Build & push
```bash
# 1. extract AST focal embeddings (16-way GPU array) + assemble the features dir
ENCODER=ast VARIANT=frozen-cls POOLING=cls sbatch sbatch/extract-focal.sbatch

# 2. sweep λ + fit the head (CPU)
ENCODER=ast FEATURES_DIR=~/scratch/birdclef-2026/features/exp300-ast \
    sbatch sbatch/sweep-lambda.sbatch        # -> artifacts/head_ast_lambda_sweep.json
ENCODER=ast FEATURES_DIR=~/scratch/birdclef-2026/features/exp300-ast \
    LAMBDA=<best> OUT=artifacts/head_ast.npz sbatch sbatch/fit-head.sbatch

# 3. local parity + submission dry-run
sbatch sbatch/dryrun-kernel-ast.sbatch

# 4. stage datasets
cp ../artifacts/head_ast.npz head_dataset/
kaggle datasets create -p head_dataset            # first time (or: version -m "...")
#    AST snapshot: huggingface-cli download MIT/ast-finetuned-audioset-10-10-0.4593 \
#       --local-dir snapshot_dataset  ; then kaggle datasets create -p snapshot_dataset

# 5. push + submit the kernel
kaggle kernels push -p .
kaggle kernels status acmiyaguchi/birdclef-2026-exp300-ast-ridge-baseline
```
