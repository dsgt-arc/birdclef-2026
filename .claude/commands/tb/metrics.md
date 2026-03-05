---
name: TensorBoard Metrics
description: Read and compare TensorBoard training metrics using DuckDB queries.
category: Training
tags: [tensorboard, metrics, training, duckdb]
---

**Purpose**

Query TensorBoard logs from experiment training runs using DuckDB. Supports single-run summaries, multi-run comparisons, and custom SQL queries.

**Usage**

```
/tb:metrics [experiment-path-or-id] [options]
```

**Arguments**

- `$ARGUMENTS` - Experiment path, ID, or log directory. Accepts:
  - Full path: `user/anthony/experiments/014-efficientnet-soft-labels`
  - ID: `014` or `015`
  - Direct log path: `user/anthony/experiments/014-efficientnet-soft-labels/logs/focal`
  - If no argument, ask user which experiment

**Steps**

1. **Resolve the experiment/log path**:
   - If a number (e.g., `014`), glob for `user/*/experiments/<number>-*/logs`
   - If an experiment path, append `/logs` if needed
   - Verify the directory exists and contains TensorBoard event files
   - If multiple runs exist (subdirectories), list them all

2. **Ensure dependencies are available**:
   - Use the shared venv python: `~/scratch/birdclef-2026/.venv/bin/python`
   - Check duckdb is installed; if not, install it: `uv pip install duckdb` (in activated venv)
   - TensorBoard should already be in the venv

3. **Run the metrics script**:
   ```bash
   ~/scratch/birdclef-2026/.venv/bin/python scripts/tb_metrics.py <log_dirs>
   ```

   Common patterns:
   ```bash
   # Summary of all runs in an experiment
   ~/scratch/birdclef-2026/.venv/bin/python scripts/tb_metrics.py logs/*/lightning_logs/

   # Specific metrics
   ~/scratch/birdclef-2026/.venv/bin/python scripts/tb_metrics.py logs/ --tags val/cmap,val/auroc,train/loss_epoch

   # Last N values to see trend
   ~/scratch/birdclef-2026/.venv/bin/python scripts/tb_metrics.py logs/ --tags val/cmap --last 5

   # Custom comparison query
   ~/scratch/birdclef-2026/.venv/bin/python scripts/tb_metrics.py logs/*/lightning_logs/ \
     --query "SELECT run, round(max(value),4) as best_cmap FROM tb WHERE tag='val/cmap' GROUP BY run ORDER BY best_cmap DESC"

   # Export for further analysis
   ~/scratch/birdclef-2026/.venv/bin/python scripts/tb_metrics.py logs/ --export metrics.parquet
   ```

4. **Present results**:
   - Show a clean summary table with run name, key metrics (val/cmap, val/auroc, train/loss), current epoch
   - If comparing runs, highlight the best performer
   - Note if training is still in progress (check `squeue` for active jobs)
   - Compare against known baselines from MEMORY.md if relevant

5. **If user asks for custom analysis**, use the `--query` flag with DuckDB SQL:
   - Table name is `tb` with columns: `run`, `tag`, `step`, `value`, `wall_time`
   - Example: learning curves, convergence analysis, cross-run comparison

**Example Output**

```
## Exp 014 Training Metrics

| Run              | Epoch | val/cmap | val/auroc | train/loss | Status  |
|------------------|-------|----------|-----------|------------|---------|
| focal            | 11    | 0.2312   | 0.8038    | 0.0142     | Running |
| focal-soundscape | 11    | 0.2338   | 0.8007    | 0.0156     | Running |

Baseline (exp 012): val/cmap = 0.336

Both runs are below baseline after 11/30 epochs.
```
