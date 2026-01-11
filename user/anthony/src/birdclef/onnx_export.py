"""ONNX export utilities for BEATs model.

Exports BEATs components to ONNX format for faster CPU inference.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Add vendor path for BEATs imports
VENDOR_PATH = Path(__file__).parents[4] / "vendor" / "unilm" / "beats"
if str(VENDOR_PATH) not in sys.path:
    sys.path.insert(0, str(VENDOR_PATH))


class BEATsFrontend(nn.Module):
    """BEATs frontend: patch_embed + layer_norm + proj.

    Takes fbank features as input (not raw audio) since torchaudio
    fbank doesn't export cleanly to ONNX.

    Input: fbank features of shape (batch, 1, time_frames, 128)
    Output: patch embeddings of shape (batch, num_patches, 768)
    """

    def __init__(self, beats_model):
        super().__init__()
        self.patch_embedding = beats_model.patch_embedding
        self.layer_norm = beats_model.layer_norm
        self.post_extract_proj = beats_model.post_extract_proj
        self.embed_dim = beats_model.cfg.encoder_embed_dim

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        """Forward pass through frontend.

        Args:
            fbank: Normalized fbank features (batch, 1, time_frames, 128)

        Returns:
            Patch embeddings (batch, num_patches, 768)
        """
        # Patch embedding: Conv2d with stride=patch_size
        features = self.patch_embedding(fbank)
        # (batch, embed_dim, h, w) -> (batch, embed_dim, num_patches)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        # -> (batch, num_patches, embed_dim)
        features = features.transpose(1, 2)

        # Layer norm
        features = self.layer_norm(features)

        # Project to encoder dimension
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        return features


class BEATsEncoder(nn.Module):
    """Full BEATs encoder for ONNX export.

    Takes fbank features as input and outputs pooled embeddings.
    """

    def __init__(self, beats_model):
        super().__init__()
        self.patch_embedding = beats_model.patch_embedding
        self.layer_norm = beats_model.layer_norm
        self.post_extract_proj = beats_model.post_extract_proj
        self.encoder = beats_model.encoder
        self.embed_dim = beats_model.cfg.encoder_embed_dim

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        """Forward pass through full encoder.

        Args:
            fbank: Normalized fbank features (batch, 1, time_frames, 128)

        Returns:
            Pooled embeddings (batch, 768)
        """
        # Frontend
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        # Transformer encoder
        x, _ = self.encoder(features, padding_mask=None)

        # Mean pooling
        x = x.mean(dim=1)

        return x


def export_frontend(
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export BEATs frontend to ONNX.

    Args:
        checkpoint_path: Path to BEATs checkpoint
        output_path: Path to save ONNX model
        opset_version: ONNX opset version

    Returns:
        Path to exported ONNX model
    """
    from BEATs import BEATs, BEATsConfig

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    beats = BEATs(cfg)
    beats.load_state_dict(checkpoint["model"])
    beats.eval()

    # Create frontend module
    frontend = BEATsFrontend(beats)
    frontend.eval()

    # Create dummy input
    # For 5 seconds of audio at 16kHz: 499 time frames, 128 mel bins
    # After patch embedding with patch_size=16: (499//16, 128//16) = (31, 8) = 248 patches
    batch_size = 1
    time_frames = 499  # ~5 seconds
    dummy_input = torch.randn(batch_size, 1, time_frames, 128)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX using legacy exporter for consistency
    torch.onnx.export(
        frontend,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["fbank"],
        output_names=["patch_embeddings"],
        dynamic_axes={
            "fbank": {0: "batch_size", 2: "time_frames"},
            "patch_embeddings": {0: "batch_size", 1: "num_patches"},
        },
        dynamo=False,  # Use legacy exporter for stability
    )

    print(f"Frontend exported to {output_path}")
    return output_path


def export_full_encoder(
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset_version: int = 17,
    fixed_batch_size: int | None = None,
) -> Path:
    """Export full BEATs encoder to ONNX.

    Args:
        checkpoint_path: Path to BEATs checkpoint
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        fixed_batch_size: If set, export with fixed batch size (avoids dynamic shape issues)

    Returns:
        Path to exported ONNX model
    """
    from BEATs import BEATs, BEATsConfig

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    beats = BEATs(cfg)
    beats.load_state_dict(checkpoint["model"])
    beats.eval()

    # Create encoder module
    encoder = BEATsEncoder(beats)
    encoder.eval()

    # Create dummy input
    batch_size = fixed_batch_size or 1
    time_frames = 499
    dummy_input = torch.randn(batch_size, 1, time_frames, 128)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use legacy exporter for transformer models (better dynamic shape support)
    # The new dynamo-based exporter has issues with attention mechanisms
    dynamic_axes = None if fixed_batch_size else {
        "fbank": {0: "batch_size", 2: "time_frames"},
        "embeddings": {0: "batch_size"},
    }

    torch.onnx.export(
        encoder,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["fbank"],
        output_names=["embeddings"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy exporter for better transformer support
    )

    print(f"Full encoder exported to {output_path}")
    return output_path


def verify_onnx_output(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    test_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> tuple[bool, float]:
    """Verify ONNX model output matches PyTorch.

    Args:
        pytorch_model: PyTorch model
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (matches, max_diff)
    """
    import onnxruntime as ort

    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input.numpy()})[0]

    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    matches = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)

    return matches, max_diff


def benchmark_onnx(
    onnx_path: str | Path,
    batch_sizes: list[int] = [1, 4, 16, 32],
    time_frames: int = 499,
    warmup_iters: int = 5,
    bench_iters: int = 20,
    num_threads: int = 4,
) -> dict:
    """Benchmark ONNX model with ONNX Runtime.

    Args:
        onnx_path: Path to ONNX model
        batch_sizes: Batch sizes to benchmark
        time_frames: Number of time frames (499 for 5 seconds)
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations
        num_threads: Number of CPU threads

    Returns:
        Dictionary of benchmark results
    """
    import time
    import onnxruntime as ort

    # Configure session
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = num_threads

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    results = {}

    for batch_size in batch_sizes:
        # Create input
        test_input = np.random.randn(batch_size, 1, time_frames, 128).astype(np.float32)

        # Warmup
        for _ in range(warmup_iters):
            _ = session.run(None, {input_name: test_input})

        # Benchmark
        times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            _ = session.run(None, {input_name: test_input})
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        throughput = (batch_size * 1000) / avg_ms

        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "avg_ms": avg_ms,
            "throughput_samples_per_sec": throughput,
            "min_ms": min(times),
            "max_ms": max(times),
        }

        print(f"Batch {batch_size}: {avg_ms:.2f} ms, {throughput:.1f} samples/sec")

    return results
