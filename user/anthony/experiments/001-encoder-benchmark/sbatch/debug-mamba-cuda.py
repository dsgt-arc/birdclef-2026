#!/usr/bin/env python3
"""Debug mamba-ssm CUDA kernel availability."""

import sys
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")

# Check torch CUDA
import torch
print(f"\nTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

# Check mamba-ssm
print("\n--- mamba-ssm check ---")
try:
    import mamba_ssm
    print(f"mamba-ssm version: {mamba_ssm.__version__}")
    print(f"mamba-ssm location: {mamba_ssm.__file__}")
except ImportError as e:
    print(f"mamba-ssm import failed: {e}")
    sys.exit(1)

# Check selective_scan_fn specifically
print("\n--- selective_scan_fn check ---")
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print(f"selective_scan_fn: {selective_scan_fn}")
    print("CUDA kernels: AVAILABLE")
except ImportError as e:
    print(f"selective_scan_fn import failed: {e}")
    print("CUDA kernels: NOT AVAILABLE")

    # Try to get more details
    print("\n--- Detailed error ---")
    try:
        from mamba_ssm import ops
        print(f"mamba_ssm.ops: {ops}")
        print(f"ops contents: {dir(ops)}")
    except Exception as e2:
        print(f"ops import failed: {e2}")

    try:
        from mamba_ssm.ops import selective_scan_interface
        print(f"selective_scan_interface: {selective_scan_interface}")
    except Exception as e3:
        print(f"selective_scan_interface import failed: {e3}")

# Check if the CUDA extension exists
print("\n--- Extension check ---")
try:
    import mamba_ssm.ops.selective_scan_cuda as cuda_ext
    print(f"selective_scan_cuda: {cuda_ext}")
except ImportError as e:
    print(f"selective_scan_cuda not found: {e}")

# Check torch extensions
print("\n--- torch.utils.cpp_extension ---")
from torch.utils.cpp_extension import CUDA_HOME
print(f"CUDA_HOME: {CUDA_HOME}")
