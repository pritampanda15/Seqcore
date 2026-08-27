#!/usr/bin/env python3
"""GPU headroom benchmark for Seqcore's kernels.

IMPORTANT -- read before quoting any number this produces
---------------------------------------------------------
Seqcore does **not** currently dispatch any analysis kernel to the GPU. Its
`gpu_available`, `gpu_info`, `device` and memory helpers manage CuPy devices,
but `gc_content`, `translate`, `reverse_complement` and `count_kmers` all
execute on the CPU through NumPy.

This script therefore measures two different things and labels them separately:

  * `seqcore-cpu`  -- Seqcore's real, shipping performance via its public API.
  * `cupy-gpu`     -- a CuPy transcription of the *same* kernel, written against
                      the same ordinal-matrix layout Seqcore uses.

The `cupy-gpu` column is an estimate of the headroom a future GPU backend could
reach. It is **not** Seqcore performance and must not be reported as such.

Because a GPU backend would have to keep the encoded matrix resident on the
device, two GPU timings are reported:

  * `compute`  -- kernel only, data already on the device. This is the ceiling.
  * `end2end`  -- host->device transfer + kernel + device->host result copy.
                  This is what a single call from CPU-resident data would cost,
                  and it is the honest number for a one-shot operation.

All GPU timings synchronize the stream before stopping the clock; CuPy kernel
launches are asynchronous and unsynchronized timings are meaningless.

Caveat on the transfer number: `end2end` copies from ordinary pageable host
memory, which is what a naive port would do. Pinned (page-locked) staging
buffers would raise the achieved PCIe bandwidth substantially, so `end2end`
should be read as a conservative bound rather than the best attainable. The gap
between `compute` and `end2end` is nevertheless the important signal: where it
is large, the operation is only worth offloading if the encoded matrix stays
resident on the device across several operations.

Requirements
------------
    pip install cupy-cuda12x     # or cupy-cuda11x, matching your CUDA runtime

Usage
-----
    python benchmarks/check_gpu.py            # verify the CUDA setup first
    python benchmarks/benchmark_gpu.py
    python benchmarks/benchmark_gpu.py --quick
    python benchmarks/benchmark_gpu.py --sizes 10000,50000 --seq-len 500

Exits with status 1 and a diagnostic if no usable GPU is present.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SEED = 20260825
DEFAULT_SIZES = [1000, 5000, 10000, 25000, 50000, 100000]
QUICK_SIZES = [1000, 5000, 10000]
DEFAULT_SEQ_LEN = 1000
DEFAULT_K = 8
REPEATS = 5
WARMUP = 2


# --------------------------------------------------------------------------
# GPU discovery
# --------------------------------------------------------------------------


def import_cupy():
    """Return the cupy module, or exit with an actionable message."""
    try:
        import cupy as cp
    except ImportError:
        sys.exit(
            "cupy is not installed.\n"
            "  Install the wheel matching your CUDA runtime, e.g.\n"
            "    pip install cupy-cuda12x   # CUDA 12.x\n"
            "    pip install cupy-cuda11x   # CUDA 11.x\n"
            "  Check your version with: nvidia-smi"
        )
    try:
        if not cp.cuda.is_available():
            sys.exit(
                "cupy is installed but no CUDA device is available.\n"
                "  Run `nvidia-smi` to confirm a GPU and driver are present, and\n"
                "  `python benchmarks/check_gpu.py` for a fuller diagnostic."
            )
    except Exception as exc:  # driver present but unusable
        sys.exit(f"CUDA initialization failed: {exc}")
    return cp


def gpu_description(cp) -> dict:
    """Collect device identification for the results file."""
    info = {"name": "unknown", "compute_capability": None, "total_memory_gb": None}
    try:
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        info["name"] = (
            props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        )
        info["compute_capability"] = f"{props['major']}.{props['minor']}"
        info["total_memory_gb"] = round(device.mem_info[1] / 1e9, 2)
        info["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        info["cupy_version"] = cp.__version__
    except Exception as exc:
        info["error"] = str(exc)
    return info


# --------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------


def time_cpu(fn, repeats: int = REPEATS) -> float:
    """Best-of-N wall clock for a CPU callable."""
    for _ in range(WARMUP):
        fn()
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def time_gpu(cp, fn, repeats: int = REPEATS) -> float:
    """Best-of-N wall clock for a GPU callable, synchronizing before stopping.

    CuPy launches kernels asynchronously, so the stream must be synchronized
    inside the timed region or the measurement records launch latency only.
    """
    for _ in range(WARMUP):
        fn()
    cp.cuda.Stream.null.synchronize()

    best = float("inf")
    for _ in range(repeats):
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - start)
    return best


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def make_sequences(n: int, length: int) -> list[str]:
    rng = random.Random(SEED)
    return ["".join(rng.choices("ACGT", k=length)) for _ in range(n)]


def encoded_matrix(sequences: list[str], length: int) -> np.ndarray:
    """Build the same uint8 ordinal matrix Seqcore uses internally."""
    import seqcore as sc

    return sc.DNAArray(sequences).encoded


# --------------------------------------------------------------------------
# CuPy transcriptions of Seqcore's kernels
#
# These mirror seqcore/core/operations.py and seqcore/core/kmers.py exactly,
# operating on the same ordinal encoding (A=0, C=1, G=2, T/U=3, N=4).
# --------------------------------------------------------------------------


def gpu_gc_content(cp, data_gpu, lengths_gpu):
    mask = (data_gpu == 1) | (data_gpu == 2)
    counts = mask.sum(axis=-1, dtype=cp.int64)
    return counts / lengths_gpu * 100.0


def gpu_reverse_complement(cp, data_gpu, comp_table_gpu):
    return comp_table_gpu[data_gpu[:, ::-1]]


def gpu_translate(cp, data_gpu, codon_table_gpu):
    n, width = data_gpu.shape
    n_codons = width // 3
    codons = data_gpu[:, : 3 * n_codons].reshape(n, n_codons, 3).astype(cp.int64)
    idx = codons[:, :, 0] * 25 + codons[:, :, 1] * 5 + codons[:, :, 2]
    return codon_table_gpu[idx]


def gpu_count_kmers(cp, data_gpu, k: int):
    width = data_gpu.shape[-1]
    n_windows = width - k + 1
    codes = data_gpu[:, :n_windows].astype(cp.int64)
    for offset in range(1, k):
        codes *= 5
        codes += data_gpu[:, offset : offset + n_windows]
    return cp.unique(codes.ravel(), return_counts=True)


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------


def _build_ops(cp, sc, arr, device, k: int):
    """Build the (name, cpu_fn, gpu_fn, gpu_transfer_fn) tuples for one batch.

    Kernels are bound explicitly through default arguments rather than captured
    from the enclosing loop, so each entry keeps its own references.
    """

    def cpu_gc():
        return sc.gc_content(arr)

    def cpu_rc():
        return sc.reverse_complement(arr)

    def cpu_tr():
        return sc.translate(arr)

    def cpu_km():
        return sc.count_kmers(arr, k=k)

    def gpu_gc(d=None):
        return gpu_gc_content(cp, device["data"] if d is None else d, device["lengths"])

    def gpu_rc(d=None):
        return gpu_reverse_complement(cp, device["data"] if d is None else d, device["comp"])

    def gpu_tr(d=None):
        return gpu_translate(cp, device["data"] if d is None else d, device["codon"])

    def gpu_km(d=None):
        return gpu_count_kmers(cp, device["data"] if d is None else d, k)

    return [
        ("gc_content", cpu_gc, gpu_gc),
        ("reverse_complement", cpu_rc, gpu_rc),
        ("translate", cpu_tr, gpu_tr),
        (f"count_kmers_k{k}", cpu_km, gpu_km),
    ]


def _make_end_to_end(cp, kernel, data_cpu):
    """Time host->device transfer + kernel + result copy back, as a real call would."""

    def end_to_end():
        dg = cp.asarray(data_cpu)
        out = kernel(dg)
        if isinstance(out, tuple):
            return tuple(cp.asnumpy(o) for o in out)
        return cp.asnumpy(out)

    return end_to_end


def run_size(cp, n: int, seq_len: int, k: int) -> list[dict]:
    """Benchmark every operation at one batch size."""
    import seqcore as sc
    from seqcore.core.operations import _CODON_TO_AA, _DNA_COMPLEMENT_TABLE

    sequences = make_sequences(n, seq_len)
    arr = sc.DNAArray(sequences)
    data_cpu = arr.encoded

    # Device-resident inputs are held in a dict so the timed closures never
    # capture names that are later unbound.
    device = {
        "data": cp.asarray(data_cpu),
        "lengths": cp.asarray(np.asarray(arr.lengths)),
        "comp": cp.asarray(_DNA_COMPLEMENT_TABLE),
        "codon": cp.asarray(_CODON_TO_AA),
    }

    rows = []
    for name, cpu_fn, gpu_fn in _build_ops(cp, sc, arr, device, k):
        cpu_t = time_cpu(cpu_fn)
        gpu_t = time_gpu(cp, gpu_fn)
        e2e_t = time_gpu(cp, _make_end_to_end(cp, gpu_fn, data_cpu), repeats=3)

        row = {
            "operation": name,
            "n_sequences": n,
            "seq_length": seq_len,
            "seqcore_cpu": cpu_t,
            "cupy_gpu_compute": gpu_t,
            "cupy_gpu_end2end": e2e_t,
            "headroom_compute": cpu_t / gpu_t if gpu_t else None,
            "headroom_end2end": cpu_t / e2e_t if e2e_t else None,
        }
        rows.append(row)
        print(
            f"    {name:22s} cpu {cpu_t:8.4f}s | gpu {gpu_t:8.4f}s "
            f"({row['headroom_compute']:6.1f}x) | gpu+transfer {e2e_t:8.4f}s "
            f"({row['headroom_end2end']:5.1f}x)"
        )

    device.clear()
    cp.get_default_memory_pool().free_all_blocks()
    return rows


def verify_agreement(cp, seq_len: int) -> bool:
    """Check the CuPy kernels reproduce Seqcore's CPU results before timing."""
    import seqcore as sc
    from seqcore.core.operations import _DNA_COMPLEMENT_TABLE

    sequences = make_sequences(64, seq_len)
    arr = sc.DNAArray(sequences)
    data_gpu = cp.asarray(arr.encoded)
    lengths_gpu = cp.asarray(np.asarray(arr.lengths))

    ok = True

    cpu_gc = sc.gc_content(arr)
    gpu_gc = cp.asnumpy(gpu_gc_content(cp, data_gpu, lengths_gpu))
    if not np.allclose(cpu_gc, gpu_gc, atol=1e-3):
        print("    ! gc_content mismatch between CPU and GPU kernels")
        ok = False

    comp_gpu = cp.asarray(_DNA_COMPLEMENT_TABLE)
    cpu_rc = sc.reverse_complement(arr).encoded
    gpu_rc = cp.asnumpy(gpu_reverse_complement(cp, data_gpu, comp_gpu))
    if not np.array_equal(cpu_rc, gpu_rc):
        print("    ! reverse_complement mismatch between CPU and GPU kernels")
        ok = False

    print(f"    kernel agreement: {'OK' if ok else 'FAILED'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--quick", action="store_true", help="use smaller batch sizes")
    ap.add_argument("--sizes", default="", help="comma-separated batch sizes")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="k for k-mer counting")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    cp = import_cupy()
    info = gpu_description(cp)

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES

    print("=" * 78)
    print("SEQCORE GPU HEADROOM BENCHMARK")
    print(
        f"  GPU     : {info['name']} "
        f"(cc {info.get('compute_capability')}, {info.get('total_memory_gb')} GB)"
    )
    print(f"  CuPy    : {info.get('cupy_version')}")
    print(f"  Host    : {platform.platform()} | Python {platform.python_version()}")
    print("=" * 78)
    print()
    print("  NOTE: 'gpu' columns are CuPy transcriptions of Seqcore's kernels.")
    print("        Seqcore itself computes on the CPU. Do not report the GPU")
    print("        numbers as Seqcore performance -- they are headroom only.")
    print()

    try:
        import seqcore as sc

        if sc.gpu_available():
            print(f"  seqcore.gpu_available() -> True ({sc.gpu_info()})")
        print("  seqcore compute path    -> CPU (NumPy) for all operations")
    except Exception:
        pass
    print()

    if not args.skip_verify:
        print("  Verifying CuPy kernels against Seqcore's CPU results...")
        if not verify_agreement(cp, args.seq_len):
            sys.exit("Aborting: GPU kernels disagree with the CPU implementation.")
        print()

    results = []
    for n in sizes:
        print(f"  Batch: {n} x {args.seq_len} bp")
        try:
            results.extend(run_size(cp, n, args.seq_len, args.k))
        except cp.cuda.memory.OutOfMemoryError:
            print(f"    ! out of GPU memory at n={n}; stopping here")
            break
        print()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "gpu-headroom",
        "warning": (
            "cupy_gpu_* columns are CuPy transcriptions of Seqcore's kernels, "
            "not Seqcore performance. Seqcore computes on the CPU."
        ),
        "gpu": info,
        "host": platform.platform(),
        "python": platform.python_version(),
        "metric": "best-of-N wall clock (seconds), stream-synchronized",
        "seq_length": args.seq_len,
        "k": args.k,
        "results": results,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
