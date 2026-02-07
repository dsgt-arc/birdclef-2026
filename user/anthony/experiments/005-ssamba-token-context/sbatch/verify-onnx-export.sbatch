#!/bin/bash
#SBATCH --job-name=verify-onnx
#SBATCH --account=paceship-dsgt_clef2026
#SBATCH -N1 -n1 --cpus-per-task=4 --mem=16G
#SBATCH -t15 -qinferno
#SBATCH --output=logs/verify-onnx-%j.out

# Quick verification: export config 0 (coarse-128x24, fewest tokens) to ONNX,
# run inference via ONNX Runtime with mamba-onnx custom ops, and compare
# against PyTorch output for numerical correctness.
#
# Usage:
#   cd user/anthony/experiments/005-ssamba-token-context
#   mkdir -p logs
#   sbatch sbatch/verify-onnx-export.sbatch

set -eu

echo "=== ONNX Export Verification ==="
echo "Node: $SLURMD_NODENAME"
date

repo_dir=~/arc/birdclef-2026
project_dir=$repo_dir/user/anthony
exp_dir=$repo_dir/user/anthony/experiments/005-ssamba-token-context
PYTHON=$project_dir/.venv/bin/python

# 1. Verify mamba-onnx installation
echo ""
echo "--- Step 1: Check mamba-onnx ---"
$PYTHON -c "
import mamba_onnx
backends = mamba_onnx.available_backends()
print(f'mamba-onnx version: {mamba_onnx.__version__}')
print(f'Available backends: {backends}')
lib = mamba_onnx.get_library_path()
print(f'Library: {lib}')
assert backends, 'No backends available!'
print('OK')
"

# 2. Export smallest config and run ORT inference
echo ""
echo "--- Step 2: Export + ORT inference (config 0: coarse-128x24) ---"
$PYTHON -c "
import sys, os, tempfile, time
import torch
import numpy as np

sys.path.insert(0, '$exp_dir/scripts')
sys.path.insert(0, '$repo_dir/user/anthony/experiments/004-ssamba-hf-port/ssamba_hf')

from ssamba_onnx_export import export_ssamba
from ssamba_hf import SSAMBAConfig, SSAMBAModel

# Smallest config: ~250 tokens at 60s
config = SSAMBAConfig(
    embed_dim=192,
    depth=24,
    img_size=(128, 6000),
    patch_size=(128, 24),
    use_cuda_kernels=False,
)
print(f'Config: coarse-128x24, tokens={config.num_patches}, embed_dim=192')

model = SSAMBAModel(config)
model.eval()
params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {params:,}')

dummy = torch.randn(1, 6000, 128)

# Export
with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, 'test.onnx')

    t0 = time.perf_counter()
    export_ssamba(model, dummy, onnx_path)
    export_time = time.perf_counter() - t0
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f'ONNX export: {size_mb:.1f} MB in {export_time:.1f}s')

    # Load in ORT
    import onnxruntime as ort
    import mamba_onnx
    opts = ort.SessionOptions()
    lib = mamba_onnx.register_custom_ops(opts)
    print(f'ORT backend: {os.path.basename(lib)}')

    sess = ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])

    # Run ORT
    inp = np.random.randn(1, 6000, 128).astype(np.float32)
    t0 = time.perf_counter()
    out_ort = sess.run(None, {'input': inp})[0]
    ort_time = time.perf_counter() - t0
    print(f'ORT output: shape={out_ort.shape}, range=[{out_ort.min():.4f}, {out_ort.max():.4f}]')
    print(f'ORT inference: {ort_time*1000:.1f}ms')

    # Numerical comparison: PyTorch vs ORT on a short input (30 frames)
    print()
    print('--- Step 3: Numerical comparison (30 frames) ---')
    short_config = SSAMBAConfig(
        embed_dim=192,
        depth=24,
        img_size=(128, 30),
        patch_size=(128, 1),
        use_cuda_kernels=False,
    )
    short_model = SSAMBAModel(short_config)
    short_model.eval()

    short_dummy = torch.randn(1, 30, 128)

    # PyTorch reference
    with torch.no_grad():
        out_pt = short_model(short_dummy).numpy()

    # Export and run in ORT
    short_onnx = os.path.join(tmpdir, 'short.onnx')
    export_ssamba(short_model, short_dummy, short_onnx)

    opts2 = ort.SessionOptions()
    mamba_onnx.register_custom_ops(opts2)
    sess2 = ort.InferenceSession(short_onnx, opts2, providers=['CPUExecutionProvider'])
    out_ort2 = sess2.run(None, {'input': short_dummy.numpy()})[0]

    # Compare
    diff = np.abs(out_pt - out_ort2)
    print(f'PyTorch output: shape={out_pt.shape}, range=[{out_pt.min():.4f}, {out_pt.max():.4f}]')
    print(f'ORT output:     shape={out_ort2.shape}, range=[{out_ort2.min():.4f}, {out_ort2.max():.4f}]')
    print(f'Max abs diff: {diff.max():.6f}')
    print(f'Mean abs diff: {diff.mean():.6f}')

    if diff.max() < 1e-3:
        print('NUMERICAL CHECK: PASS (atol < 1e-3)')
    elif diff.max() < 1e-2:
        print('NUMERICAL CHECK: MARGINAL (atol < 1e-2)')
    else:
        print(f'NUMERICAL CHECK: FAIL (max diff = {diff.max():.4f})')

print()
print('=== Verification Complete ==='
)
"

# 3. Quick single-config benchmark test
echo ""
echo "--- Step 4: Benchmark test (config 0, single run) ---"
$PYTHON $exp_dir/scripts/benchmark_onnx_token_shapes.py \
    --config-index 0 \
    --results-dir $exp_dir/onnx-results

echo ""
echo "=== All Verification Steps Complete ==="
date
