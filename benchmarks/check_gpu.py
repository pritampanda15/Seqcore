#!/usr/bin/env python3
"""Quick GPU availability check for Seqcore.

Run this script to verify your GPU setup before running benchmarks.

Author: Dr. Pritam Kumar Panda @ Stanford University
"""

from __future__ import annotations

import sys


def check_gpu():
    """Check GPU availability and print setup instructions if needed."""
    print("=" * 60)
    print("SEQCORE GPU SETUP CHECK")
    print("=" * 60)
    print()

    # Check NumPy
    try:
        import numpy as np

        print(f"[OK] NumPy {np.__version__} installed")
    except ImportError:
        print("[ERROR] NumPy not installed")
        print("  Install with: pip install numpy")
        return False

    # Check CuPy
    try:
        import cupy as cp

        print(f"[OK] CuPy {cp.__version__} installed")
    except ImportError:
        print("[ERROR] CuPy not installed")
        print()
        print("  To install CuPy, first determine your CUDA version:")
        print("    nvidia-smi | grep 'CUDA Version'")
        print()
        print("  Then install the matching CuPy version:")
        print("    CUDA 11.x: pip install cupy-cuda11x")
        print("    CUDA 12.x: pip install cupy-cuda12x")
        print()
        print("  Or install with automatic detection:")
        print("    pip install cupy")
        return False

    # Check CUDA availability
    try:
        if cp.cuda.is_available():
            print("[OK] CUDA is available")
        else:
            print("[ERROR] CUDA not available")
            print("  Please check your NVIDIA driver installation")
            return False
    except Exception as e:
        print(f"[ERROR] CUDA check failed: {e}")
        return False

    # Check GPU device
    try:
        device = cp.cuda.Device(0)
        print(f"[OK] GPU device found")

        # Get device info
        try:
            name = device.name if hasattr(device, "name") else "Unknown"
            print(f"     Device: {name}")
        except Exception:
            pass

        # Get memory info
        mem_info = device.mem_info
        total_gb = mem_info[1] / 1e9
        free_gb = mem_info[0] / 1e9
        print(f"     Memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

        # Get compute capability
        attrs = device.attributes
        major = attrs.get("ComputeCapabilityMajor", "?")
        minor = attrs.get("ComputeCapabilityMinor", "?")
        print(f"     Compute Capability: {major}.{minor}")

    except Exception as e:
        print(f"[ERROR] GPU device error: {e}")
        return False

    # Quick computation test
    print()
    print("Running quick GPU test...")
    try:
        # Create array on GPU
        x = cp.random.rand(10000, 10000, dtype=cp.float32)
        y = cp.random.rand(10000, 10000, dtype=cp.float32)

        # Matrix multiplication
        cp.cuda.Stream.null.synchronize()
        import time

        start = time.perf_counter()
        z = cp.dot(x, y)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        print(f"[OK] GPU computation test passed")
        print(f"     10K x 10K matrix multiplication: {elapsed:.3f}s")

        # Cleanup
        del x, y, z
        cp.get_default_memory_pool().free_all_blocks()

    except Exception as e:
        print(f"[ERROR] GPU computation failed: {e}")
        return False

    print()
    print("=" * 60)
    print("GPU SETUP VERIFIED - Ready for benchmarks!")
    print("=" * 60)
    print()
    print("Run GPU benchmarks with:")
    print("  python benchmarks/benchmark_gpu.py")
    print()
    print("Get detailed GPU info with:")
    print("  python benchmarks/benchmark_gpu.py --info")
    print()

    return True


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)
