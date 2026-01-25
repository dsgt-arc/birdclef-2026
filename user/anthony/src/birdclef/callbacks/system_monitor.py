"""System I/O monitoring callback for PyTorch Lightning.

Uses psutil to track disk I/O, memory usage, and network I/O.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SystemIOMonitor(Callback):
    """Monitor system-level I/O and memory statistics.

    Logs delta-based metrics between log intervals:
    - system/disk_read_mb: MB read since last log
    - system/disk_write_mb: MB written since last log
    - system/rss_mb: Current resident set size in MB
    - system/vms_mb: Current virtual memory size in MB
    - system/net_recv_mb: Network MB received (optional)
    - system/net_sent_mb: Network MB sent (optional)

    Args:
        log_every_n_batches: Log stats every N batches (default: 100)
        track_network: Whether to track network I/O (default: False)
    """

    def __init__(
        self,
        log_every_n_batches: int = 100,
        track_network: bool = False,
    ):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.track_network = track_network

        self._psutil_available = False
        self._prev_disk_io: Any = None
        self._prev_net_io: Any = None

        # Try to import psutil
        try:
            import psutil

            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            pass

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Initialize baseline counters."""
        if not self._psutil_available:
            return

        try:
            self._prev_disk_io = self._psutil.disk_io_counters()
            if self.track_network:
                self._prev_net_io = self._psutil.net_io_counters()
        except Exception:
            pass

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log system stats periodically."""
        if not self._psutil_available:
            return

        if (batch_idx + 1) % self.log_every_n_batches != 0:
            return

        try:
            self._log_system_stats(pl_module)
        except Exception:
            # Silently ignore errors
            pass

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Log epoch-end system stats."""
        if not self._psutil_available:
            return

        try:
            self._log_memory_stats(pl_module)
        except Exception:
            pass

    def _log_system_stats(self, pl_module: pl.LightningModule) -> None:
        """Log current system I/O stats with deltas."""
        metrics = {}

        # Disk I/O deltas
        current_disk = self._psutil.disk_io_counters()
        if current_disk and self._prev_disk_io:
            read_delta = current_disk.read_bytes - self._prev_disk_io.read_bytes
            write_delta = current_disk.write_bytes - self._prev_disk_io.write_bytes
            metrics["system/disk_read_mb"] = read_delta / (1024 * 1024)
            metrics["system/disk_write_mb"] = write_delta / (1024 * 1024)
        self._prev_disk_io = current_disk

        # Network I/O deltas (optional)
        if self.track_network:
            current_net = self._psutil.net_io_counters()
            if current_net and self._prev_net_io:
                recv_delta = current_net.bytes_recv - self._prev_net_io.bytes_recv
                sent_delta = current_net.bytes_sent - self._prev_net_io.bytes_sent
                metrics["system/net_recv_mb"] = recv_delta / (1024 * 1024)
                metrics["system/net_sent_mb"] = sent_delta / (1024 * 1024)
            self._prev_net_io = current_net

        # Memory stats
        process = self._psutil.Process()
        mem_info = process.memory_info()
        metrics["system/rss_mb"] = mem_info.rss / (1024 * 1024)
        metrics["system/vms_mb"] = mem_info.vms / (1024 * 1024)

        if metrics:
            pl_module.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=False)

    def _log_memory_stats(self, pl_module: pl.LightningModule) -> None:
        """Log memory stats at epoch end."""
        process = self._psutil.Process()
        mem_info = process.memory_info()

        pl_module.log_dict(
            {
                "system/epoch_rss_mb": mem_info.rss / (1024 * 1024),
                "system/epoch_vms_mb": mem_info.vms / (1024 * 1024),
            },
            on_step=False,
            on_epoch=True,
        )
