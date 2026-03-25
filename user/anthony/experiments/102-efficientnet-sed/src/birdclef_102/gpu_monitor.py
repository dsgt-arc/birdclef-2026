"""GPU utilization monitoring callback. Vendored from exp 009."""

import time

import pytorch_lightning as pl
import torch


class GPUUtilizationMonitor(pl.Callback):
    """Log GPU utilization, throughput, and timing breakdown."""

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

        if isinstance(batch, dict):
            key = "mel" if "mel" in batch else "waveform"
            batch_size = batch[key].shape[0]
        else:
            batch_size = batch[0].shape[0]
        samples_sec = batch_size / total_time if total_time > 0 else 0.0

        pl_module.log("timing/data_time_ms", data_time * 1000, on_step=True, on_epoch=False)
        pl_module.log("timing/compute_time_ms", compute_time * 1000, on_step=True, on_epoch=False)
        pl_module.log("timing/data_pct", data_pct, on_step=True, on_epoch=False)
        pl_module.log("throughput/samples_sec", samples_sec, on_step=True, on_epoch=False)

        if torch.cuda.is_available():
            device = pl_module.device
            mem_used = torch.cuda.memory_allocated(device) / 1e9
            mem_pct = 100.0 * mem_used / (torch.cuda.get_device_properties(device).total_memory / 1e9)
            pl_module.log("gpu/memory_used_gb", mem_used, on_step=True, on_epoch=False)
            pl_module.log("gpu/memory_pct", mem_pct, on_step=True, on_epoch=False)

        if self._nvml_available:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                pl_module.log("gpu/utilization_pct", float(util.gpu), on_step=True, on_epoch=False)
            except Exception:
                pass

    def teardown(self, trainer, pl_module, stage=None):
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
