"""GPU utilization and throughput monitoring callback for PyTorch Lightning.

Logs timing, throughput, and GPU stats to TensorBoard for diagnosing
bottlenecks (data loading vs compute, GPU utilization, memory pressure).
"""

import time

import pytorch_lightning as pl
import torch


class GPUUtilizationMonitor(pl.Callback):
    """Log GPU utilization, throughput, and data/compute timing breakdown.

    Metrics logged per training step:
        timing/data_time_ms    - time between steps (data loading + overhead)
        timing/compute_time_ms - time for forward + backward + optimizer
        timing/data_pct        - % of step time spent on data loading
        throughput/samples_sec  - training samples per second
        throughput/batches_sec  - training batches per second
        gpu/utilization_pct    - GPU compute utilization (from nvidia-smi)
        gpu/memory_used_gb     - GPU memory in use
        gpu/memory_reserved_gb - GPU memory reserved by PyTorch
        gpu/memory_pct         - % of total GPU memory in use

    Requires pynvml for GPU utilization %. Falls back gracefully if unavailable.
    """

    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._step_start = None
        self._step_end = None
        self._nvml_available = False
        self._nvml_handle = None

    def setup(self, trainer, pl_module, stage=None):
        if stage != "fit":
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            device_idx = trainer.local_rank if trainer.local_rank >= 0 else 0
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self._nvml_available = True
        except Exception:
            self._nvml_available = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.monotonic()
        if self._step_end is not None:
            self._data_time = now - self._step_end
        else:
            self._data_time = 0.0
        self._step_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._step_end = time.monotonic()
        compute_time = self._step_end - self._step_start

        if batch_idx % self.log_every_n_steps != 0:
            return

        data_time = self._data_time
        total_time = data_time + compute_time
        data_pct = 100.0 * data_time / total_time if total_time > 0 else 0.0

        batch_size = batch["waveform"].shape[0] if isinstance(batch, dict) else batch[0].shape[0]
        samples_sec = batch_size / total_time if total_time > 0 else 0.0
        batches_sec = 1.0 / total_time if total_time > 0 else 0.0

        pl_module.log("timing/data_time_ms", data_time * 1000, on_step=True, on_epoch=False)
        pl_module.log("timing/compute_time_ms", compute_time * 1000, on_step=True, on_epoch=False)
        pl_module.log("timing/data_pct", data_pct, on_step=True, on_epoch=False)
        pl_module.log("throughput/samples_sec", samples_sec, on_step=True, on_epoch=False)
        pl_module.log("throughput/batches_sec", batches_sec, on_step=True, on_epoch=False)

        if torch.cuda.is_available():
            device = pl_module.device
            mem_used = torch.cuda.memory_allocated(device) / 1e9
            mem_reserved = torch.cuda.memory_reserved(device) / 1e9
            mem_total = torch.cuda.get_device_properties(device).total_memory / 1e9
            mem_pct = 100.0 * mem_used / mem_total if mem_total > 0 else 0.0

            pl_module.log("gpu/memory_used_gb", mem_used, on_step=True, on_epoch=False)
            pl_module.log("gpu/memory_reserved_gb", mem_reserved, on_step=True, on_epoch=False)
            pl_module.log("gpu/memory_pct", mem_pct, on_step=True, on_epoch=False)

        if self._nvml_available:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                pl_module.log("gpu/utilization_pct", float(util.gpu), on_step=True, on_epoch=False)
                pl_module.log("gpu/memory_util_pct", float(util.memory), on_step=True, on_epoch=False)
            except Exception:
                pass

    def teardown(self, trainer, pl_module, stage=None):
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
