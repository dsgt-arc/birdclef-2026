"""Perch v2 ONNX session wrapper.

Deduplicated from birdclef_106.perch_onnx / birdclef_105.perch_onnx. Kept
as a third copy here so the Kaggle kernel does not need to pull exp 106
into the deps layer (exp 106 imports polars/sklearn heavy for training).

Model interface (verified in exp 106 job 6275860):
    input  'inputs'              float  [batch, 160000]
    output 'embedding'           float  [batch, 1536]
    output 'spatial_embedding'   float  [batch, 16, 4, 1536]
    output 'spectrogram'         float  [batch, 500, 128]
    output 'label'               float  [batch, 14795]
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_REPO = "justinchuby/Perch-onnx"
DEFAULT_FILENAME = "perch_v2_no_dft.onnx"


class PerchSession:
    """Thin wrapper around onnxruntime.InferenceSession for Perch v2."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        repo_id: str = DEFAULT_REPO,
        filename: str = DEFAULT_FILENAME,
        intra_op_num_threads: int | None = None,
    ):
        import onnxruntime as ort

        if model_path is None:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model_path = str(model_path)

        sess_options = ort.SessionOptions()
        if intra_op_num_threads is None:
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
        """Run Perch on a batch of 5s windows at 32 kHz (float32 `(B, 160000)`)."""
        if outputs is None:
            outputs = ["embedding"]
        for name in outputs:
            if name not in self.output_names:
                raise ValueError(
                    f"Unknown Perch output {name!r}. Available: {sorted(self.output_names)}"
                )
        arrays = self.session.run(
            outputs,
            {self.input_name: windows.astype(np.float32, copy=False)},
        )
        return dict(zip(outputs, arrays))
