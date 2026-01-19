# GPU Utilization Analysis - Task 3306049
**Experiment 003: Model Distillation (500k AudioSet)**
**Date**: January 17, 2026
**Duration**: 5.32 hours (00:19 - 05:39)

## Summary Statistics

### GPU Utilization
| Metric | Average | Min | Max | StdDev |
|--------|---------|-----|-----|--------|
| GPU Util % | **3.56%** | 0% | 100% | 15.10% |
| Memory Util % | 1.76% | 0% | 52% | 7.48% |
| Temperature °C | 44.0° | 27° | 51° | 2.38° |

### Memory Usage
- **Average Used**: 3,494 MB (14.2% of 24,576 MB total)
- **Min Used**: 1 MB
- **Max Used**: 3,530 MB
- **Total Available**: 24,576 MB (24 GB Quadro RTX 6000)

## Utilization Distribution

**GPU was SEVERELY UNDERUTILIZED:**

| Utilization Range | Sample Count | % of Time |
|-------------------|--------------|-----------|
| **0% (Idle)** | **3,513** | **92.67%** |
| 1-10% | 29 | 0.76% |
| 11-25% | 46 | 1.21% |
| 26-50% | 89 | 2.35% |
| 51-75% | 51 | 1.35% |
| 76-99% | 33 | 0.87% |
| 100% | 30 | 0.79% |

## Power State Analysis

| P-State | Sample Count | % of Time | Avg GPU Util | Avg Temp |
|---------|--------------|-----------|--------------|----------|
| P2 (Active) | 3,631 | 95.78% | 3.72% | 44.32°C |
| P8 (Idle) | 160 | 4.22% | 0.00% | 36.76°C |

## High Utilization Periods

When GPU utilization exceeded 25%:
- **Total samples**: 203 out of 3,791 (5.35%)
- **Average GPU util**: 61.77%
- **Average memory**: 3,530 MB
- **First occurrence**: 00:23:15
- **Last occurrence**: 05:38:22

## Key Findings

### Critical Bottleneck Identified
The GPU was **idle 92.67% of the time** with an average utilization of only **3.56%**. This indicates a severe **I/O bottleneck** where:

1. **Data loading is too slow** - GPU is waiting for data from storage
2. **Memory was loaded** (3.5 GB consistently), but not being processed
3. **Streaming cache inefficiency** - Despite the StreamingCacheDataset implementation, data wasn't flowing fast enough

### Pattern Observations
- GPU shows brief bursts of activity (reaching 100%) but immediately drops back to 0%
- Memory stays loaded at ~3.5 GB throughout training
- Temperature stable at ~44°C (indicating low sustained load)
- Most training time spent in P2 power state but with minimal compute

### Performance Impact
With only 3.56% average GPU utilization over 5.32 hours:
- **Effective GPU time**: ~11.4 minutes out of 5.32 hours
- **Wasted GPU time**: ~5.1 hours of idle GPU

## Recommendations

1. **Investigate DataLoader bottleneck**
   - Check if `num_workers=8` is optimal
   - Profile data loading pipeline
   - Consider prefetch_factor tuning

2. **Optimize Streaming Cache**
   - The sequential shard processing may not be streaming efficiently
   - Prefetching next 5 shards might not be fast enough
   - Consider parallel shard loading or larger prefetch buffer

3. **Storage I/O Analysis**
   - Check if source cache (scratch dir) I/O is saturated
   - Verify NVMe local cache is being utilized effectively
   - Monitor disk I/O metrics during training

4. **Alternative Strategies**
   - Pre-copy more shards to local NVMe before training
   - Increase `max_local_shards` from 20 to higher value
   - Use faster storage tier if available
   - Consider caching entire dataset to local NVMe first

5. **Code Profiling**
   - Add timing instrumentation to identify exact bottleneck
   - Profile `_load_shard()`, `__getitem__()`, and worker processes
   - Check for GIL contention in DataLoader workers

## DuckDB Query Reference

For further analysis, use these queries on `logs/gpu-3306049.csv`:

```sql
-- Basic statistics
SELECT
    ROUND(AVG("gpu_util_%")::NUMERIC, 2) as avg_gpu_util,
    ROUND(AVG("mem_util_%")::NUMERIC, 2) as avg_mem_util,
    ROUND(AVG(temp_c)::NUMERIC, 2) as avg_temp
FROM read_csv_auto('logs/gpu-3306049.csv');

-- Find periods of sustained high utilization (>50% for 30+ seconds)
SELECT
    timestamp,
    "gpu_util_%",
    mem_used_mb,
    temp_c
FROM read_csv_auto('logs/gpu-3306049.csv')
WHERE "gpu_util_%\" > 50
ORDER BY timestamp;

-- 5-minute bins
SELECT
    DATE_TRUNC('minute', timestamp::TIMESTAMP) +
        INTERVAL (EXTRACT('minute' FROM timestamp::TIMESTAMP)::INT / 5 * 5) MINUTE as time_bin,
    ROUND(AVG("gpu_util_%")::NUMERIC, 2) as avg_gpu_util,
    ROUND(AVG(mem_used_mb)::NUMERIC, 0) as avg_mem_mb
FROM read_csv_auto('logs/gpu-3306049.csv')
GROUP BY time_bin
ORDER BY time_bin;
```

## Data Source
- **Raw GPU logs**: `logs/gpu-3306049.csv`
- **Training logs**: `user/anthony/experiments/003-model-distillation/sbatch/logs/distill-500k-3306049.out`
- **SLURM job**: 3306049
