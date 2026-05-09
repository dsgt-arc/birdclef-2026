# Vendored Kaggle datasets

This directory holds Kaggle-dataset bundles we depend on (typically frozen
ONNX teacher models or external corpora). The bundles themselves are
**gitignored** — they're large binaries that are trivially re-fetchable
via `kaggle datasets download`. Only this README is tracked, so the
directory exists in fresh clones and the re-fetch instructions are
preserved.

## How to re-fetch

```bash
# From repo root:
mkdir -p vendor/kaggle/datasets/<slug>
kaggle datasets download <owner>/<slug> -p vendor/kaggle/datasets/<slug> --unzip
```

The `<slug>` directory name should match the Kaggle dataset slug exactly so
references in code and docs resolve unambiguously.

## Currently-required bundles

| Slug | Used by | Notes |
|---|---|---|
| `tuckerarrants/perch-v2-no-dft-onnx` | exp 031 (Perch v2 KD teacher) | Contains `perch_v2_no_dft.onnx` (~395 MB) — the frozen Perch v2 backbone with the DFT layer stripped, used as the distillation teacher. Also bundles inference-side wheels (onnxruntime, openvino) that we do NOT use. |

## Why we don't track the bundles

- ONNX files routinely exceed GitHub's 100 MB per-file hard limit.
- Bundles are upstream artifacts, not our work — re-derivation is one
  Kaggle CLI call away.
- Re-fetched bundles are bit-identical to the originally-downloaded copy
  (Kaggle versions are immutable per dataset version).

## Verifying integrity after re-fetch

For Perch v2 specifically: run
`bash user/murilo/analysis/sbatch/verify_perch_teacher.sh` which exercises
all 6 sanity checks (load, schema, non-degenerate output, distinct
embeddings, CUDA provider). The expected schema is documented at
`user/murilo/experiments/031-perch-kd-distillation/phase0_perch_teacher_schema.md`.
