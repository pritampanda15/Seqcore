"""Device management for GPU acceleration and performance controls.

Provides a unified interface for CPU/GPU device selection and
memory management.
"""

from __future__ import annotations

import time
import warnings
from contextlib import contextmanager
from typing import Any

# Global state
_current_device = "cpu"
_batch_size = 10000
_memory_limit = None


def gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if CUDA-capable GPU is available, False otherwise.

    Example:
        >>> if sc.gpu_available():
        ...     print("GPU acceleration enabled")

    """
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def gpu_info() -> dict[str, Any]:
    """Get information about available GPU(s).

    Returns:
        Dictionary containing GPU device information.

    Example:
        >>> info = sc.gpu_info()
        >>> print(info['name'], info['memory_total'])

    """
    if not gpu_available():
        return {"available": False, "message": "No CUDA-capable GPU found"}

    try:
        import cupy as cp

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        mem_info = cp.cuda.runtime.memGetInfo()

        return {
            "available": True,
            "device_id": device.id,
            "name": props["name"].decode() if isinstance(props["name"], bytes) else props["name"],
            "compute_capability": f"{props['major']}.{props['minor']}",
            "memory_total": props["totalGlobalMem"],
            "memory_free": mem_info[0],
            "memory_used": mem_info[1] - mem_info[0],
            "multiprocessors": props["multiProcessorCount"],
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


@contextmanager
def device(device_name: str):
    """Context manager for setting computation device.

    Args:
        device_name: Device to use ("cpu", "cuda", "cuda:0", "cuda:1", etc.)

    Example:
        >>> with sc.device("cuda:0"):
        ...     result = sc.align(sequences, reference)

    """
    global _current_device
    old_device = _current_device

    if device_name.startswith("cuda"):
        if not gpu_available():
            warnings.warn("GPU not available, falling back to CPU", stacklevel=2)
            _current_device = "cpu"
        else:
            _current_device = device_name
            if ":" in device_name:
                device_id = int(device_name.split(":")[1])
                try:
                    import cupy as cp

                    cp.cuda.Device(device_id).use()
                except Exception:
                    pass
    else:
        _current_device = "cpu"

    try:
        yield
    finally:
        _current_device = old_device
        if old_device.startswith("cuda") and ":" in old_device:
            try:
                import cupy as cp

                device_id = int(old_device.split(":")[1])
                cp.cuda.Device(device_id).use()
            except Exception:
                pass


def get_current_device() -> str:
    """Get the currently active device."""
    return _current_device


def clear_gpu_cache():
    """Clear GPU memory cache.

    Frees unused memory blocks held by the GPU memory pool.

    Example:
        >>> sc.clear_gpu_cache()

    """
    if gpu_available():
        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        except Exception:
            pass


def set_memory_limit(limit: str):
    """Set GPU memory usage limit.

    Args:
        limit: Memory limit as string (e.g., "8GB", "4096MB")

    Example:
        >>> sc.set_memory_limit("8GB")

    """
    global _memory_limit

    # Parse memory limit string
    limit = limit.upper().strip()
    if limit.endswith("GB"):
        _memory_limit = int(float(limit[:-2]) * 1024 * 1024 * 1024)
    elif limit.endswith("MB"):
        _memory_limit = int(float(limit[:-2]) * 1024 * 1024)
    elif limit.endswith("KB"):
        _memory_limit = int(float(limit[:-2]) * 1024)
    else:
        _memory_limit = int(limit)

    if gpu_available():
        try:
            import cupy as cp

            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=_memory_limit)
        except Exception:
            pass


def set_batch_size(size: int):
    """Set default batch size for vectorized operations.

    Args:
        size: Number of sequences/items to process per batch.

    Example:
        >>> sc.set_batch_size(100_000)

    """
    global _batch_size
    _batch_size = size


def get_batch_size() -> int:
    """Get the current default batch size."""
    return _batch_size


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


@contextmanager
def timer():
    """Context manager for timing operations.

    Returns:
        Timer object with elapsed time in seconds.

    Example:
        >>> with sc.timer() as t:
        ...     result = sc.align(sequences, reference)
        >>> print(f"Aligned in {t.elapsed:.2f}s")

    """
    t = Timer()
    with t:
        yield t


def get_array_module():
    """Get the appropriate array module (numpy or cupy).

    Returns numpy by default, cupy if GPU is active.
    """
    if _current_device.startswith("cuda") and gpu_available():
        try:
            import cupy as cp

            return cp
        except ImportError:
            pass
    import numpy as np

    return np
