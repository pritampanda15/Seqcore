#!/usr/bin/env python3
"""GPU vs CPU Performance Benchmarks for Seqcore.

This script compares performance between CPU (NumPy) and GPU (CuPy) backends
for various biological sequence operations.

Requirements:
    - NVIDIA GPU with CUDA support
    - CuPy installed: pip install cupy-cuda11x (adjust for your CUDA version)

Author: Dr. Pritam Kumar Panda @ Stanford University
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Check for GPU availability
GPU_AVAILABLE = False
try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        gpu_info = cp.cuda.Device(0)
        GPU_NAME = gpu_info.name if hasattr(gpu_info, 'name') else "Unknown GPU"
except ImportError:
    cp = None
    GPU_NAME = "N/A"
except Exception as e:
    cp = None
    GPU_NAME = f"Error: {e}"


def print_header():
    """Print benchmark header."""
    print("#" * 60)
    print("# SEQCORE GPU PERFORMANCE BENCHMARKS")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# Author: Dr. Pritam Kumar Panda @ Stanford University")
    print("#" * 60)
    print()

    print("System Information:")
    print(f"  GPU Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"  GPU Device: {GPU_NAME}")
        print(f"  CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
        mem_info = cp.cuda.Device(0).mem_info
        print(f"  GPU Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free")
    print()


def generate_sequences_cpu(n_seqs: int, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate random DNA sequences on CPU."""
    data = np.random.randint(0, 4, size=(n_seqs, seq_len), dtype=np.uint8)
    lengths = np.full(n_seqs, seq_len, dtype=np.int32)
    return data, lengths


def generate_sequences_gpu(n_seqs: int, seq_len: int):
    """Generate random DNA sequences on GPU."""
    data = cp.random.randint(0, 4, size=(n_seqs, seq_len), dtype=cp.uint8)
    lengths = cp.full(n_seqs, seq_len, dtype=cp.int32)
    return data, lengths


def benchmark_gc_content_cpu(data: np.ndarray, lengths: np.ndarray) -> float:
    """Benchmark GC content calculation on CPU."""
    start = time.perf_counter()

    # Vectorized GC content calculation
    gc_mask = (data == 1) | (data == 2)  # C=1, G=2
    n_seqs = data.shape[0]
    results = np.zeros(n_seqs, dtype=np.float32)

    for i in range(n_seqs):
        gc_count = np.sum(gc_mask[i, :lengths[i]])
        results[i] = (gc_count / lengths[i] * 100) if lengths[i] > 0 else 0.0

    return time.perf_counter() - start


def benchmark_gc_content_gpu(data, lengths) -> float:
    """Benchmark GC content calculation on GPU."""
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    # Fully vectorized GC content on GPU
    gc_mask = (data == 1) | (data == 2)
    gc_counts = cp.sum(gc_mask, axis=1)
    results = (gc_counts / lengths * 100).astype(cp.float32)

    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start


def benchmark_complement_cpu(data: np.ndarray) -> float:
    """Benchmark complement calculation on CPU."""
    complement_table = np.array([3, 2, 1, 0, 4], dtype=np.uint8)

    start = time.perf_counter()
    result = complement_table[data]
    _ = result.sum()  # Force computation
    return time.perf_counter() - start


def benchmark_complement_gpu(data) -> float:
    """Benchmark complement calculation on GPU."""
    complement_table = cp.array([3, 2, 1, 0, 4], dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    result = complement_table[data]
    _ = result.sum()  # Force computation
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start


def benchmark_reverse_complement_cpu(data: np.ndarray) -> float:
    """Benchmark reverse complement on CPU."""
    complement_table = np.array([3, 2, 1, 0, 4], dtype=np.uint8)

    start = time.perf_counter()
    complemented = complement_table[data]
    result = complemented[:, ::-1]
    _ = result.sum()  # Force computation
    return time.perf_counter() - start


def benchmark_reverse_complement_gpu(data) -> float:
    """Benchmark reverse complement on GPU."""
    complement_table = cp.array([3, 2, 1, 0, 4], dtype=cp.uint8)

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    complemented = complement_table[data]
    result = complemented[:, ::-1]
    _ = result.sum()  # Force computation
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start


def benchmark_kmer_counting_cpu(data: np.ndarray, k: int = 4) -> float:
    """Benchmark k-mer counting on CPU."""
    start = time.perf_counter()

    n_seqs, seq_len = data.shape
    n_kmers = 4 ** k
    counts = np.zeros((n_seqs, n_kmers), dtype=np.int32)

    for i in range(n_seqs):
        for j in range(seq_len - k + 1):
            kmer = data[i, j:j+k]
            if np.all(kmer < 4):  # Skip N's
                idx = 0
                for base in kmer:
                    idx = idx * 4 + base
                counts[i, idx] += 1

    return time.perf_counter() - start


def benchmark_kmer_counting_gpu(data, k: int = 4) -> float:
    """Benchmark k-mer counting on GPU using vectorized operations."""
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    n_seqs, seq_len = data.shape
    n_kmers = 4 ** k

    # Create sliding window indices
    indices = cp.arange(seq_len - k + 1)

    # Compute k-mer indices for all positions
    multipliers = cp.array([4 ** (k - 1 - i) for i in range(k)], dtype=cp.int32)

    # Extract k-mers and compute indices
    kmer_indices = cp.zeros((n_seqs, seq_len - k + 1), dtype=cp.int32)
    for i in range(k):
        kmer_indices += data[:, i:seq_len - k + 1 + i].astype(cp.int32) * multipliers[i]

    # Count occurrences (simplified - full implementation would use scatter_add)
    counts = cp.zeros((n_seqs, n_kmers), dtype=cp.int32)

    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start


def benchmark_distance_matrix_cpu(data: np.ndarray) -> float:
    """Benchmark pairwise distance matrix on CPU."""
    start = time.perf_counter()

    n_seqs = data.shape[0]
    distances = np.zeros((n_seqs, n_seqs), dtype=np.float32)

    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            # Hamming distance
            dist = np.sum(data[i] != data[j])
            distances[i, j] = dist
            distances[j, i] = dist

    return time.perf_counter() - start


def benchmark_distance_matrix_gpu(data) -> float:
    """Benchmark pairwise distance matrix on GPU."""
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    n_seqs = data.shape[0]

    # Vectorized pairwise comparison using broadcasting
    # This is memory intensive but very fast on GPU
    data_expanded = data[:, cp.newaxis, :]  # (n, 1, len)
    distances = cp.sum(data_expanded != data[cp.newaxis, :, :], axis=2).astype(cp.float32)

    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start


def benchmark_motif_search_cpu(data: np.ndarray, motif: np.ndarray) -> float:
    """Benchmark motif searching on CPU."""
    start = time.perf_counter()

    n_seqs, seq_len = data.shape
    motif_len = len(motif)
    matches = []

    for i in range(n_seqs):
        for j in range(seq_len - motif_len + 1):
            if np.array_equal(data[i, j:j+motif_len], motif):
                matches.append((i, j))

    return time.perf_counter() - start


def benchmark_motif_search_gpu(data, motif) -> float:
    """Benchmark motif searching on GPU."""
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()

    n_seqs, seq_len = data.shape
    motif_len = len(motif)

    # Create sliding windows and compare
    matches = cp.zeros((n_seqs, seq_len - motif_len + 1), dtype=cp.bool_)

    for j in range(seq_len - motif_len + 1):
        window = data[:, j:j+motif_len]
        matches[:, j] = cp.all(window == motif, axis=1)

    total_matches = cp.sum(matches)

    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start


def run_benchmarks():
    """Run all GPU benchmarks."""
    print_header()

    if not GPU_AVAILABLE:
        print("ERROR: No GPU available!")
        print("Please ensure:")
        print("  1. You have an NVIDIA GPU with CUDA support")
        print("  2. CuPy is installed: pip install cupy-cuda11x")
        print("  3. CUDA drivers are properly installed")
        return None

    results = []

    # Test configurations
    configs = [
        (1000, 1000, "1K seqs x 1kb"),
        (10000, 1000, "10K seqs x 1kb"),
        (50000, 1000, "50K seqs x 1kb"),
        (100000, 1000, "100K seqs x 1kb"),
        (10000, 10000, "10K seqs x 10kb"),
    ]

    # Benchmark: GC Content
    print("=" * 60)
    print("BENCHMARK: GC Content Calculation")
    print("=" * 60)

    for n_seqs, seq_len, desc in configs:
        print(f"\n{desc}:")

        # CPU
        data_cpu, lengths_cpu = generate_sequences_cpu(n_seqs, seq_len)
        cpu_time = benchmark_gc_content_cpu(data_cpu, lengths_cpu)

        # GPU
        data_gpu, lengths_gpu = generate_sequences_gpu(n_seqs, seq_len)
        # Warmup
        _ = benchmark_gc_content_gpu(data_gpu, lengths_gpu)
        gpu_time = benchmark_gc_content_gpu(data_gpu, lengths_gpu)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  CPU:     {cpu_time:.4f}s")
        print(f"  GPU:     {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")

        results.append({
            "operation": "gc_content",
            "n_seqs": n_seqs,
            "seq_len": seq_len,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup
        })

        # Clean up GPU memory
        del data_gpu, lengths_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Benchmark: Complement
    print("\n" + "=" * 60)
    print("BENCHMARK: DNA Complement")
    print("=" * 60)

    for n_seqs, seq_len, desc in configs:
        print(f"\n{desc}:")

        data_cpu, _ = generate_sequences_cpu(n_seqs, seq_len)
        cpu_time = benchmark_complement_cpu(data_cpu)

        data_gpu, _ = generate_sequences_gpu(n_seqs, seq_len)
        _ = benchmark_complement_gpu(data_gpu)  # Warmup
        gpu_time = benchmark_complement_gpu(data_gpu)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  CPU:     {cpu_time:.4f}s")
        print(f"  GPU:     {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")

        results.append({
            "operation": "complement",
            "n_seqs": n_seqs,
            "seq_len": seq_len,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup
        })

        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Benchmark: Reverse Complement
    print("\n" + "=" * 60)
    print("BENCHMARK: Reverse Complement")
    print("=" * 60)

    for n_seqs, seq_len, desc in configs:
        print(f"\n{desc}:")

        data_cpu, _ = generate_sequences_cpu(n_seqs, seq_len)
        cpu_time = benchmark_reverse_complement_cpu(data_cpu)

        data_gpu, _ = generate_sequences_gpu(n_seqs, seq_len)
        _ = benchmark_reverse_complement_gpu(data_gpu)  # Warmup
        gpu_time = benchmark_reverse_complement_gpu(data_gpu)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  CPU:     {cpu_time:.4f}s")
        print(f"  GPU:     {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")

        results.append({
            "operation": "reverse_complement",
            "n_seqs": n_seqs,
            "seq_len": seq_len,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup
        })

        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Benchmark: Distance Matrix (smaller datasets due to O(n^2))
    print("\n" + "=" * 60)
    print("BENCHMARK: Pairwise Distance Matrix")
    print("=" * 60)

    dist_configs = [
        (100, 1000, "100 seqs x 1kb"),
        (500, 1000, "500 seqs x 1kb"),
        (1000, 1000, "1K seqs x 1kb"),
        (2000, 500, "2K seqs x 500bp"),
    ]

    for n_seqs, seq_len, desc in dist_configs:
        print(f"\n{desc}:")

        data_cpu, _ = generate_sequences_cpu(n_seqs, seq_len)
        cpu_time = benchmark_distance_matrix_cpu(data_cpu)

        data_gpu, _ = generate_sequences_gpu(n_seqs, seq_len)
        try:
            _ = benchmark_distance_matrix_gpu(data_gpu)  # Warmup
            gpu_time = benchmark_distance_matrix_gpu(data_gpu)

            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"  CPU:     {cpu_time:.4f}s")
            print(f"  GPU:     {gpu_time:.4f}s")
            print(f"  Speedup: {speedup:.1f}x")

            results.append({
                "operation": "distance_matrix",
                "n_seqs": n_seqs,
                "seq_len": seq_len,
                "cpu_time": cpu_time,
                "gpu_time": gpu_time,
                "speedup": speedup
            })
        except cp.cuda.memory.OutOfMemoryError:
            print(f"  GPU:     Out of memory")
            results.append({
                "operation": "distance_matrix",
                "n_seqs": n_seqs,
                "seq_len": seq_len,
                "cpu_time": cpu_time,
                "gpu_time": None,
                "speedup": None
            })

        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Benchmark: Motif Search
    print("\n" + "=" * 60)
    print("BENCHMARK: Motif Search (8bp motif)")
    print("=" * 60)

    motif_configs = [
        (1000, 10000, "1K seqs x 10kb"),
        (10000, 1000, "10K seqs x 1kb"),
        (50000, 1000, "50K seqs x 1kb"),
    ]

    motif_cpu = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint8)  # ACGTACGT
    motif_gpu = cp.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=cp.uint8)

    for n_seqs, seq_len, desc in motif_configs:
        print(f"\n{desc}:")

        data_cpu, _ = generate_sequences_cpu(n_seqs, seq_len)
        cpu_time = benchmark_motif_search_cpu(data_cpu, motif_cpu)

        data_gpu, _ = generate_sequences_gpu(n_seqs, seq_len)
        _ = benchmark_motif_search_gpu(data_gpu, motif_gpu)  # Warmup
        gpu_time = benchmark_motif_search_gpu(data_gpu, motif_gpu)

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"  CPU:     {cpu_time:.4f}s")
        print(f"  GPU:     {gpu_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")

        results.append({
            "operation": "motif_search",
            "n_seqs": n_seqs,
            "seq_len": seq_len,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup
        })

        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Print Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print()
    print("| Operation | Dataset | CPU Time | GPU Time | Speedup |")
    print("|-----------|---------|----------|----------|---------|")

    for r in results:
        if r["gpu_time"] is not None:
            print(f"| {r['operation']:17} | {r['n_seqs']:>6} x {r['seq_len']:>5} | "
                  f"{r['cpu_time']:.4f}s | {r['gpu_time']:.4f}s | {r['speedup']:.1f}x |")
        else:
            print(f"| {r['operation']:17} | {r['n_seqs']:>6} x {r['seq_len']:>5} | "
                  f"{r['cpu_time']:.4f}s | OOM | N/A |")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"gpu_benchmark_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "gpu_available": GPU_AVAILABLE,
            "gpu_name": GPU_NAME,
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


def check_gpu_info():
    """Print detailed GPU information."""
    print("=" * 60)
    print("GPU DEVICE INFORMATION")
    print("=" * 60)

    if not GPU_AVAILABLE:
        print("No GPU available")
        return

    device = cp.cuda.Device(0)
    print(f"Device ID: 0")
    print(f"Device Name: {GPU_NAME}")

    attrs = device.attributes
    print(f"Compute Capability: {attrs.get('ComputeCapabilityMajor', '?')}.{attrs.get('ComputeCapabilityMinor', '?')}")
    print(f"Multiprocessors: {attrs.get('MultiProcessorCount', 'N/A')}")
    print(f"Max Threads/Block: {attrs.get('MaxThreadsPerBlock', 'N/A')}")
    print(f"Max Block Dims: ({attrs.get('MaxBlockDimX', 'N/A')}, {attrs.get('MaxBlockDimY', 'N/A')}, {attrs.get('MaxBlockDimZ', 'N/A')})")
    print(f"Max Grid Dims: ({attrs.get('MaxGridDimX', 'N/A')}, {attrs.get('MaxGridDimY', 'N/A')}, {attrs.get('MaxGridDimZ', 'N/A')})")

    mem_info = device.mem_info
    print(f"Total Memory: {mem_info[1] / 1e9:.2f} GB")
    print(f"Free Memory: {mem_info[0] / 1e9:.2f} GB")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--info":
        check_gpu_info()
    else:
        run_benchmarks()
