# Experiment 014: Teacher Pseudo-Labeling Pipeline

Generate soft pseudo-labels for 10,592 unlabeled soundscapes using exp 012 teacher
models (ProtoCLR + ConvNeXt). EAT excluded due to poor insect performance (error analysis).

## How It Works

```
For each 5s segment in each unlabeled soundscape:
  1. Load audio at model-specific sample rate
  2. ProtoCLR: mel → ONNX inference → sigmoid → probs (234,)
  3. ConvNeXt: mel → ONNX inference → sigmoid → probs (234,)
  4. Pseudo-label = mean(probs_pcr, probs_cnx)
```

Raw logits are also saved per model, enabling offline re-aggregation (temperature
scaling, weight tuning) without re-running inference.

## Quickstart

```bash
cd user/murilo/experiments/014-pseudo-labeling

# Validate on 66 labeled files first (~5 min on GPU)
bash sbatch/run.sh config/direct.yaml --split labeled

# Full run on unlabeled soundscapes (~1-2h on GPU)
bash sbatch/run.sh config/direct.yaml --split unlabeled

# Test with 5 files
bash sbatch/run.sh config/direct.yaml --split labeled --max-files 5

# Dry run
bash sbatch/run.sh config/direct.yaml --split labeled --dry-run
```

## Output

```
shared/pseudo-labels/birdclef-2026/direct/
├── logits/
│   ├── protoclr-2026-soundscape.npz   # (n_segments, 234) raw logits
│   └── convnext-2026-soundscape.npz   # (n_segments, 234) raw logits
├── pseudo_labels.parquet               # filename, end_sec, 234 species probs
└── config.json                         # run metadata
```

## Why ProtoCLR + ConvNeXt only (no EAT)

Error analysis (`013-mixit-separation/error-analysis.md`) showed:
- EAT insect AUROC: **0.673** (vs 0.856 ProtoCLR, 0.848 ConvNeXt)
- 3-model mean ensemble **hurts** (−0.039 vs best single model)
- EAT's 8kHz fmax misses insect stridulation

## Future Extensions

| Extension | Status | What it adds |
|-----------|--------|-------------|
| Temperature scaling | Deferred | Sharpen predictions with T < 1.0 (2025 1st used T=0.55) |
| MixIT enhancement | Deferred | Max-pool across 4 separated channels + original per model |
| Per-taxon weighting | Deferred | Different model weights per taxonomic class |
| Iterative self-training | Exp 015 | Student → teacher → re-pseudo-label → retrain |

## Dependencies

- Exp 012 ONNX models: `protoclr-2026-soundscape`, `convnext-2026-soundscape`
- Located in `010-submission-pipeline/artifacts/weights/`
