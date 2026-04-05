"""Perch v2 ONNX session wrapper.

Adapted from exp 104's `_load_perch`. Differences:
- Requests outputs by name (`['embedding']`) instead of by position.
- Sets `intra_op_num_threads` to silence the `pthread_setaffinity_np` spam
  seen in job 6275860 on cpu-small partition.
- Returns a small wrapper object that knows how to run a batch and return
  the named outputs the caller asked for.

Model interface (verified via job 6275860, see artifacts/onnx_inspection/summary.json):
    input  'inputs'              float  [batch, 160000]
    output 'embedding'           float  [batch, 1536]
    output 'spatial_embedding'   float  [batch, 16, 4, 1536]  (not used in phase 1)
    output 'spectrogram'         float  [batch, 500, 128]      (not used)
    output 'label'               float  [batch, 14795]         (not used)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

DEFAULT_REPO = "justinchuby/Perch-onnx"
DEFAULT_FILENAME = "perch_v2_no_dft.onnx"


class PerchSession:
    """Thin wrapper around an onnxruntime InferenceSession for Perch v2."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
        intra_op_num_threads: int | None = None,
    ):
        import onnxruntime as ort

        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        sess_options = ort.SessionOptions()
        if intra_op_num_threads is None:
            # Default to the allocation SLURM gave us, falling back to cpu_count.
            intra_op_num_threads = int(
                os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1
            )
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = {o.name for o in self.session.get_outputs()}

    def run(
        self,
        windows: np.ndarray,
        outputs: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run Perch on a batch of 5s windows.

        Args:
            windows: float32 array `(B, 160000)` at 32 kHz.
            outputs: Which output tensors to return. Defaults to `['embedding']`.

        Returns:
            Dict mapping output name → numpy array.
        """
        if outputs is None:
            outputs = ["embedding"]
        for name in outputs:
            if name not in self.output_names:
                raise ValueError(
                    f"Unknown Perch output {name!r}. Available: {sorted(self.output_names)}"
                )
        arrays = self.session.run(outputs, {self.input_name: windows.astype(np.float32, copy=False)})
        return dict(zip(outputs, arrays))
