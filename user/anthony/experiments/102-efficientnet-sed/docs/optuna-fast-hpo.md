# Fast Hyperparameter Optimization with Optuna

## Idea

Use Optuna to search hyperparameters in ~5 minutes per trial instead of
4-8 hours. Train on a small subset, evaluate on a fast proxy metric, let
Optuna explore the space and prune bad trials early.

20-50 trials × 5 min = 2-4 hours total to explore a space that would
take weeks manually.

## How to get to 5 minutes

| Component | Full run | Fast trial |
|-----------|----------|------------|
| Training data | 220K chunks | 5K chunks (2% random sample) |
| Epochs | 30-50 | 3 |
| Steps/epoch | ~860 (batch 256) or ~6900 (batch 32) | ~156 (5K / 32) |
| Validation set | 24K chunks | 1K chunks |
| Total steps | ~25K-345K | ~468 |
| Wall time | 4-8 hours | ~5 minutes |

The subset should be stratified by species to maintain class distribution.

## What Optuna would tune

### Architecture (categorical)

| Parameter | Search space |
|-----------|-------------|
| pooling_mode | ["avg", "attention"] |
| head_hidden_dim | [0, 256, 512, 1024] |
| lora_rank | [4, 8, 16, 32] |
| lora_alpha | [2×rank] |

### Training (continuous/categorical)

| Parameter | Search space |
|-----------|-------------|
| lr | log-uniform [1e-4, 3e-3] |
| weight_decay | log-uniform [1e-4, 0.1] |
| head_dropout | [0.1, 0.3, 0.5] |
| focal_gamma | [0, 1, 2, 3] |
| warmup_epochs | [0, 1, 3] |

### Augmentation (continuous/categorical)

| Parameter | Search space |
|-----------|-------------|
| mixup_alpha | [0, 0.15, 0.3, 0.5, 1.0] |
| freq_mask_param | [20, 40, 60] |
| time_mask_param | [25, 50, 75] |
| label_smoothing | [0, 0.05, 0.1] |

### Loss / Weighting (categorical)

| Parameter | Search space |
|-----------|-------------|
| class_weight | ["none", "inverse", "sqrt_inverse"] |
| activity_weight | ["none", "soft", "hard_filter"] |

## Proxy metric

**Primary**: val/auroc on the 1K validation subset after 3 epochs.

AUROC is more stable than cmAP on small subsets (cmAP is sensitive to
per-species ranking on few examples). It's also the competition metric.

**Secondary**: train loss at step 300 as an early divergence detector.
If train loss > 0.05 after 300 steps, the trial is likely bad — prune it.

## Optuna setup

```python
import optuna

def objective(trial):
    # Sample hyperparameters
    config = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "lora_rank": trial.suggest_categorical("lora_rank", [4, 8, 16, 32]),
        "pooling_mode": trial.suggest_categorical("pooling", ["avg", "attention"]),
        "head_hidden_dim": trial.suggest_categorical("hidden", [0, 256, 512]),
        "mixup_alpha": trial.suggest_float("mixup", 0.0, 1.0),
        "focal_gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "class_weight": trial.suggest_categorical("cw", ["none", "inverse", "sqrt_inverse"]),
        # ...
    }

    # Build model + data (5K subset) + trainer (3 epochs)
    # ... (reuse existing CLI code with overrides)

    # Train with pruning callback
    val_auroc = train_fast(config, trial)
    return val_auroc

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=50)
```

## Implementation plan

1. **Subset sampler**: add a `max_chunks` parameter to `BirdCLEF102DataModule`
   that randomly samples N chunks (stratified by species) instead of using all.

2. **Trial runner**: a function that takes a config dict, builds model/data/trainer,
   trains for 3 epochs, returns val/auroc. Wraps the existing CLI code.

3. **Optuna integration**: Optuna `study.optimize()` calls the trial runner.
   Use `PyTorchLightningPruningCallback` for early pruning.

4. **SLURM**: run the study as a single GPU job (50 trials × 5 min = ~4 hours).
   Or distribute across multiple GPUs with Optuna's distributed backend.

## Validity concerns

**Does 5K chunks predict full-scale performance?**

Likely yes for:
- LR, optimizer settings (learning dynamics are similar at any scale)
- Architecture choices (pooling, hidden dim)
- Augmentation intensity (SpecAugment, mixup alpha)

Likely no for:
- Class weighting (subsampled data has different class balance)
- LoRA rank (capacity matters more with more data)
- Regularization (dropout, weight decay — overfitting looks different on 5K vs 220K)

**Mitigation**: after Optuna selects top 3 configs, run them full-scale
(the real 4-8 hour training) to confirm. The Optuna phase filters the
search space; the full runs validate the winners.

## When to use this

After Phase 1 (architecture) is done and we have a winning architecture.
Then use Optuna to jointly optimize Phases 2-6 (weighting, augmentation,
loss) instead of running them sequentially. This collapses 6+ sequential
experiments into a single 4-hour HPO run.

## Prerequisites

- `pip install optuna` (add to pyproject.toml)
- `max_chunks` parameter in data module
- Trial runner function wrapping existing training code
