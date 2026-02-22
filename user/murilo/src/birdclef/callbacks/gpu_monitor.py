"""GPU utilization and throughput monitoring callback for PyTorch Lightning.

Logs timing, throughput, and GPU stats to TensorBoard for diagnosing
bottlenecks (data loading vs compute, GPU utilization, memory pressure).

Includes a print_summary() method that produces an actionable report
with min/avg/max statistics and concrete recommendations.
"""

import time
from pathlib import Path
from statistics import mean

import pytorch_lightning as pl
import torch


class GPUUtilizationMonitor(pl.Callback):
    """Log GPU utilization, throughput, and data/compute timing breakdown.

    Metrics logged per training step:
        timing/data_time_ms    - time between steps (data loading + overhead)
        timing/compute_time_ms - time for forward + backward + optimizer
        timing/data_pct        - % of step time spent on data loading
        throughput/samples_sec  - training samples per second
        gpu/memory_allocated_gb - active GPU memory used by tensors
        gpu/memory_reserved_gb  - GPU memory reserved by PyTorch allocator
        gpu/memory_pct         - % of total GPU VRAM allocated
        gpu/utilization_pct    - GPU compute utilization (from nvidia-smi)

    Requires pynvml for GPU utilization %. Falls back gracefully if unavailable.
    """

    def __init__(self, log_every_n_steps: int = 10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._step_start = None
        self._step_end = None
        self._nvml_available = False
        self._nvml_handle = None
        self._total_gpu_memory_gb = 0.0

        # History for summary statistics
        self._history = {
            "data_time_ms": [],
            "compute_time_ms": [],
            "data_pct": [],
            "samples_sec": [],
            "memory_allocated_gb": [],
            "memory_reserved_gb": [],
            "memory_pct": [],
            "gpu_util_pct": [],
        }

    def setup(self, trainer, pl_module, stage=None):
        if stage != "fit":
            return

        # Get total GPU memory
        if torch.cuda.is_available():
            device = (
                pl_module.device if pl_module.device.type == "cuda" else torch.device("cuda", 0)
            )
            self._total_gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

        # Initialize pynvml for GPU utilization %
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
        data_time = self._data_time
        total_time = data_time + compute_time
        data_pct = 100.0 * data_time / total_time if total_time > 0 else 0.0

        # Throughput
        if isinstance(batch, list | tuple):
            batch_size = batch[0].shape[0]
        else:
            batch_size = batch.shape[0]
        samples_sec = batch_size / total_time if total_time > 0 else 0.0

        # GPU memory from PyTorch
        mem_allocated_gb = 0.0
        mem_reserved_gb = 0.0
        mem_pct = 0.0
        if torch.cuda.is_available():
            device = pl_module.device
            mem_allocated_gb = torch.cuda.memory_allocated(device) / 1e9
            mem_reserved_gb = torch.cuda.memory_reserved(device) / 1e9
            if self._total_gpu_memory_gb > 0:
                mem_pct = 100.0 * mem_allocated_gb / self._total_gpu_memory_gb

        # GPU utilization from pynvml
        gpu_util_pct = None
        if self._nvml_available:
            try:
                import pynvml

                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util_pct = float(util.gpu)
            except Exception:
                pass

        # Always record history (skip first step where data_time is 0)
        if batch_idx > 0:
            self._history["data_time_ms"].append(data_time * 1000)
            self._history["compute_time_ms"].append(compute_time * 1000)
            self._history["data_pct"].append(data_pct)
            self._history["samples_sec"].append(samples_sec)
            self._history["memory_allocated_gb"].append(mem_allocated_gb)
            self._history["memory_reserved_gb"].append(mem_reserved_gb)
            self._history["memory_pct"].append(mem_pct)
            if gpu_util_pct is not None:
                self._history["gpu_util_pct"].append(gpu_util_pct)

        # Log to trainer every N steps
        if batch_idx % self.log_every_n_steps != 0:
            return

        pl_module.log("timing/data_time_ms", data_time * 1000, on_step=True, on_epoch=False)
        pl_module.log("timing/compute_time_ms", compute_time * 1000, on_step=True, on_epoch=False)
        pl_module.log("timing/data_pct", data_pct, on_step=True, on_epoch=False)
        pl_module.log("throughput/samples_sec", samples_sec, on_step=True, on_epoch=False)

        if torch.cuda.is_available():
            pl_module.log("gpu/memory_allocated_gb", mem_allocated_gb, on_step=True, on_epoch=False)
            pl_module.log("gpu/memory_reserved_gb", mem_reserved_gb, on_step=True, on_epoch=False)
            pl_module.log("gpu/memory_pct", mem_pct, on_step=True, on_epoch=False)

        if gpu_util_pct is not None:
            pl_module.log("gpu/utilization_pct", gpu_util_pct, on_step=True, on_epoch=False)

    def _build_summary(self) -> str:
        """Build the summary report as a string."""
        lines = []
        lines.append("=" * 60)
        lines.append("GPU Utilization Summary")
        lines.append("=" * 60)

        if not self._history["data_time_ms"]:
            lines.append("No data collected. Run training first.")
            lines.append("=" * 60)
            return "\n".join(lines)

        # Total GPU memory for context
        if self._total_gpu_memory_gb > 0:
            lines.append(f"Total GPU VRAM: {self._total_gpu_memory_gb:.1f} GB")
            lines.append(f"Steps recorded: {len(self._history['data_time_ms'])}")
            lines.append("")

        # Define metrics to display
        rows = [
            ("Data Time", "data_time_ms", "ms"),
            ("Compute Time", "compute_time_ms", "ms"),
            ("Data Loading %", "data_pct", "%"),
            ("Throughput", "samples_sec", "samples/s"),
            ("Memory Allocated", "memory_allocated_gb", "GB"),
            ("Memory Reserved", "memory_reserved_gb", "GB"),
            ("Memory %", "memory_pct", "%"),
            ("GPU Utilization", "gpu_util_pct", "%"),
        ]

        def _fmt(v, unit):
            if unit == "GB":
                return f"{v:.2f} {unit}"
            elif unit == "%":
                return f"{v:.1f}{unit}"
            elif unit == "ms":
                return f"{v:.1f} {unit}"
            else:
                return f"{v:.0f} {unit}"

        # Header
        lines.append(f"{'Metric':<20} {'Min':>10} {'Avg':>10} {'Max':>10}")
        lines.append("-" * 52)

        avg_mem_pct = None
        avg_data_pct = None

        for label, key, unit in rows:
            vals = self._history[key]
            if not vals:
                lines.append(f"{label:<20} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
                continue

            lo, avg, hi = min(vals), mean(vals), max(vals)
            lines.append(
                f"{label:<20} {_fmt(lo, unit):>10} {_fmt(avg, unit):>10} {_fmt(hi, unit):>10}"
            )

            if key == "memory_pct":
                avg_mem_pct = avg
            if key == "data_pct":
                avg_data_pct = avg

        # Recommendations
        lines.append("")
        lines.append("Recommendations:")
        lines.append("-" * 52)

        has_recommendation = False

        if avg_mem_pct is not None and avg_mem_pct < 50 and self._total_gpu_memory_gb > 0:
            headroom = int(100 / avg_mem_pct) if avg_mem_pct > 0 else 99
            avg_alloc = mean(self._history["memory_allocated_gb"])
            free_gb = self._total_gpu_memory_gb - avg_alloc
            lines.append(
                f"  Memory usage is {avg_mem_pct:.1f}% of "
                f"{self._total_gpu_memory_gb:.0f} GB "
                f"({free_gb:.1f} GB free)"
            )
            lines.append(f"  -> Batch size could be ~{headroom}x larger")
            has_recommendation = True

        if avg_data_pct is not None and avg_data_pct > 20:
            lines.append(f"  Data loading is {avg_data_pct:.1f}% of step time")
            lines.append("  -> Consider more num_workers or RAM caching")
            has_recommendation = True

        gpu_vals = self._history["gpu_util_pct"]
        if gpu_vals:
            avg_gpu = mean(gpu_vals)
            if avg_gpu < 50:
                lines.append(f"  GPU compute utilization is {avg_gpu:.1f}%")
                lines.append(
                    "  -> GPU is idle most of the time; increase batch size or reduce data stalls"
                )
                has_recommendation = True

        if not has_recommendation:
            lines.append("  No issues detected. GPU utilization looks healthy.")

        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self, save_path: str | Path | None = None):
        """Print a formatted summary table with recommendations.

        Args:
            save_path: If provided, also write the summary to this file.
        """
        summary = self._build_summary()
        print(f"\n{summary}\n")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text(summary + "\n")
            print(f"Summary saved to: {save_path}")

    def teardown(self, trainer, pl_module, stage=None):
        if self._nvml_available:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
