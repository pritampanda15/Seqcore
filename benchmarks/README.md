# Seqcore Benchmarks

Performance benchmarks comparing Seqcore against other bioinformatics libraries.

## Files

- `benchmark_comparison.py` - CPU benchmarks comparing Seqcore vs Biopython
- `benchmark_gpu.py` - GPU vs CPU performance comparison
- `check_gpu.py` - Verify GPU setup before running benchmarks
- `results/` - JSON benchmark results

## Running CPU Benchmarks

```bash
# Compare against Biopython
python benchmarks/benchmark_comparison.py
```

## Running GPU Benchmarks

### Prerequisites

1. NVIDIA GPU with CUDA support
2. CUDA drivers installed
3. CuPy installed

### Setup

```bash
# Check your CUDA version
nvidia-smi | grep "CUDA Version"

# Install CuPy (choose based on your CUDA version)
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x

# Or let pip auto-detect
pip install cupy
```

### Verify GPU Setup

```bash
python benchmarks/check_gpu.py
```

Expected output:
```
[OK] NumPy installed
[OK] CuPy installed
[OK] CUDA is available
[OK] GPU device found
     Device: NVIDIA RTX 4090
     Memory: 20.5 GB free / 24.0 GB total
     Compute Capability: 8.9
[OK] GPU computation test passed

GPU SETUP VERIFIED - Ready for benchmarks!
```

### Run GPU Benchmarks

```bash
# Full benchmark suite
python benchmarks/benchmark_gpu.py

# Get detailed GPU info
python benchmarks/benchmark_gpu.py --info
```

## Benchmark Operations

### CPU Benchmarks
- GC content calculation
- DNA translation
- Reverse complement
- FASTA I/O
- Pairwise alignment

### GPU Benchmarks
- GC content calculation
- DNA complement
- Reverse complement
- Pairwise distance matrix
- Motif search

## Expected Speedups

### CPU (Seqcore vs Biopython)
| Operation | Speedup |
|-----------|---------|
| GC Content | 2-3x |
| Translation | 1.1x |

### GPU (vs CPU NumPy)
| Operation | Expected Speedup |
|-----------|-----------------|
| GC Content | 10-50x |
| Distance Matrix | 50-200x |
| Motif Search | 20-100x |

*Actual speedups depend on dataset size and GPU hardware.*

## Results

Benchmark results are saved as JSON files in `results/`:
- `benchmark_YYYYMMDD_HHMMSS.json` - CPU benchmark results
- `gpu_benchmark_YYYYMMDD_HHMMSS.json` - GPU benchmark results

## Author

Dr. Pritam Kumar Panda @ Stanford University
