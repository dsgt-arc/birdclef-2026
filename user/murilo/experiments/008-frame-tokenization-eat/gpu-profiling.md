# Experiment 008: GPU Profiling

## GPU Utilization Check (2 epochs, batch_size=32, RTX 6000 24GB)

All 4 Phase 1 configs profiled via `--gpu-check 2` on 2026-02-28.

| Metric | A: baseline (4422537) | B: control (4422578) | C1a: 2×128 (4422579) | C1b: 4×128 (4422796) |
|--------|----------------------:|---------------------:|---------------------:|---------------------:|
| GPU Util (avg) | 89.9% | 88.9% | 79.3% | 82.6% |
| Compute Time (avg) | 107.1 ms | 108.7 ms | 149.2 ms | 75.7 ms |
| Data Time (avg) | 58.2 ms | 56.3 ms | 36.2 ms | 21.5 ms |
| Data Loading % | 34.0% | 32.9% | 19.2% | 21.6% |
| Throughput (avg) | 202 samp/s | 202 samp/s | 175 samp/s | 334 samp/s |
| Mem Reserved (avg) | 5.36 GB | 5.35 GB | 5.35 GB | 2.60 GB |
| VRAM Used % | 2.4% | 2.4% | 2.4% | 2.5% |
| Wall Time (2 ep) | ~19 min | ~19 min | ~22 min | ~11 min |

## Key Observations

1. **C1b (4×128, 128 tokens) is fastest**: 334 samp/s, 75.7ms compute — half the tokens
   halves attention cost and memory reserved (2.60 vs 5.35 GB).
2. **C1a (2×128, 256 tokens) is slowest**: 175 samp/s, 149.2ms compute — same token count
   as A/B (256) but ~40% slower compute. Full-frequency Conv2d kernel or attention on a
   single-column grid likely causes different memory access patterns.
3. **A ≈ B**: expected — identical architecture (16×16), only weight init differs.
4. **All configs use <3.5% of 24GB VRAM**: batch_size=32 is far below capacity, but
   effective batch size is already 256 via accum=8.
5. **Data loading is a bottleneck** (19–34% of step time): could improve with more
   num_workers (up to 12, the max CPU:GPU ratio on gpu-rtx6000).

## Estimated Full-Training Wall Time

Extrapolated from 2-epoch runs. SLURM wall time limit is 16h.

| Config | Per-epoch | 50 epochs | 100 epochs |
|--------|----------:|----------:|-----------:|
| A/B | ~9.5 min | ~7.9 h | ~15.8 h |
| C1a | ~11 min | ~9.2 h | ~18.3 h |
| C1b | ~5.5 min | ~4.6 h | ~9.2 h |

All configs fit within 16h with early stopping (patience=15).
C1a is the tightest — would hit the limit around epoch 87, but early stopping
should kick in well before that.

## Batch Size Decision

**Keep batch_size=32, accum=8 (effective 256).** Rationale:

- Effective batch size 256 is already large for 206-class LoRA fine-tuning
- Matches exp 006 setup — avoids introducing confounding variables
- Wall time is not a concern with early stopping
- Primary bottleneck is data loading, not GPU compute
- Larger batches can hurt generalization with LoRA's small parameter count

If optimizing further, bump `num_workers` from 8→12 to reduce data loading overhead.
