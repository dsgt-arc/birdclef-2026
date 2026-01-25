"""Data performance monitoring callback for PyTorch Lightning.

Tracks batch timing and dataset statistics to help identify I/O bottlenecks.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class DataPerfMonitor(Callback):
    """Monitor data loading performance and batch timing.

    Logs:
    - data/batch_time_ms: Time for each batch (end-to-end, including data loading)
    - data/epoch_mean_batch_time_ms: Mean batch time for the epoch
    - data/epoch_max_batch_time_ms: Max batch time for the epoch
    - data/epoch_min_batch_time_ms: Min batch time for the epoch
    - data/epoch_std_batch_time_ms: Std dev of batch time for the epoch
    - Worker stats from ShardAffinityDataset if available

    Args:
        log_every_n_batches: Log batch time every N batches (default: 100)
        log_worker_stats: Whether to attempt logging worker stats at epoch end
    """

    def __init__(
        self,
        log_every_n_batches: int = 100,
        log_worker_stats: bool = True,
    ):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.log_worker_stats = log_worker_stats
        self._batch_times: list[float] = []
        self._batch_start_time: float | None = None
        self._epoch_start_time: float | None = None

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Reset batch timing stats at epoch start."""
        self._batch_times = []
        self._epoch_start_time = time.perf_counter()

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record batch start time."""
        self._batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record batch end time and log metrics."""
        if self._batch_start_time is None:
            return

        batch_time_ms = (time.perf_counter() - self._batch_start_time) * 1000
        self._batch_times.append(batch_time_ms)

        # Log batch time periodically
        if (batch_idx + 1) % self.log_every_n_batches == 0:
            pl_module.log(
                "data/batch_time_ms",
                batch_time_ms,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log epoch-level timing statistics and worker stats."""
        if not self._batch_times:
            return

        import numpy as np

        times = np.array(self._batch_times)

        # Log epoch-level timing stats
        pl_module.log_dict(
            {
                "data/epoch_mean_batch_time_ms": float(np.mean(times)),
                "data/epoch_max_batch_time_ms": float(np.max(times)),
                "data/epoch_min_batch_time_ms": float(np.min(times)),
                "data/epoch_std_batch_time_ms": float(np.std(times)),
                "data/epoch_total_time_s": float(np.sum(times) / 1000),
            },
            on_step=False,
            on_epoch=True,
        )

        # Log wall clock epoch time
        if self._epoch_start_time is not None:
            epoch_wall_time = time.perf_counter() - self._epoch_start_time
            pl_module.log(
                "data/epoch_wall_time_s",
                epoch_wall_time,
                on_step=False,
                on_epoch=True,
            )

        # Try to get worker stats from dataset
        if self.log_worker_stats:
            self._log_worker_stats(trainer, pl_module)

    def _log_worker_stats(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Attempt to log worker stats from the dataset."""
        try:
            datamodule = trainer.datamodule
            if datamodule is None:
                return

            train_dataset = getattr(datamodule, "train_dataset", None)
            if train_dataset is None:
                return

            if not hasattr(train_dataset, "get_worker_stats"):
                return

            stats = train_dataset.get_worker_stats()
            if stats.get("mode") == "single_process":
                return

            # Log available stats
            stat_mapping = {
                "local_hits": "data/worker_local_hits",
                "source_loads": "data/worker_source_loads",
                "prefetch_hits": "data/worker_prefetch_hits",
                "cache_evictions": "data/worker_cache_evictions",
                "cached_shards": "data/worker_cached_shards",
            }

            logged = {}
            for key, log_name in stat_mapping.items():
                if key in stats:
                    logged[log_name] = float(stats[key])

            if logged:
                pl_module.log_dict(logged, on_step=False, on_epoch=True)

        except Exception:
            # Silently ignore errors - stats are optional
            pass
