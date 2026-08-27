# Seqcore Benchmarks

## Files

- `benchmark_suite.py` — the main benchmark. Compares Seqcore against Biopython
  and idiomatic pure Python across batch operations, pairwise alignment and
  k-mer counting. Produces the JSON that `paper/make_figures.py` turns into the
  manuscript figures and table.
- `benchmark_gpu.py` — estimates the headroom a future GPU backend could reach
  (see the warning below).
- `check_gpu.py` — verifies a CUDA setup before running GPU benchmarks.
- `benchmark_comparison.py` — an older, simpler Seqcore-vs-Biopython script,
  kept for continuity.
- `results/` — committed JSON output. Figures are generated from these files.

## Running

```bash
python benchmarks/benchmark_suite.py
```

Options:

```bash
python benchmarks/benchmark_suite.py --quick          # smaller sizes, fast check
python benchmarks/benchmark_suite.py --baseline DIR   # dev only, see below
```

## Methodology

Each configuration runs in a freshly spawned subprocess, so import order,
allocator state and warmed caches cannot leak between implementations. Reported
times are best-of-N wall clock. Sequences are uniformly random over `{A,C,G,T}`
from a fixed seed, so runs are comparable across machines.

The pure-Python comparison uses the fastest obvious standard-library approach
for each operation (`str.count`, `str.translate`, `collections.Counter`), not a
deliberately slow one. For several operations it beats Biopython, and it is the
more honest baseline.

### Comparing against another checkout (development only)

`--baseline` adds a second Seqcore checkout as an extra series, for tracking
performance across commits:

```bash
mkdir -p /tmp/sc_base && git archive <commit> | tar -x -C /tmp/sc_base
python benchmarks/benchmark_suite.py --baseline /tmp/sc_base
```

This is a development tool. It is not part of the published comparison and the
manuscript does not use it.

## GPU benchmarks

> **Seqcore does not currently compute on the GPU.** `gpu_available`, `gpu_info`,
> `device` and the memory helpers manage CuPy devices, but `gc_content`,
> `translate`, `reverse_complement` and `count_kmers` all run on the CPU through
> NumPy.

`benchmark_gpu.py` therefore measures **headroom**, not Seqcore performance. It
times CuPy transcriptions of Seqcore's exact kernels — written against the same
ordinal-matrix layout — against Seqcore's real CPU path, to estimate what a GPU
backend could achieve.

Do not report its GPU columns as Seqcore performance.

### Requirements

```bash
pip install cupy-cuda12x   # or cupy-cuda11x, matching your CUDA runtime
```

Check your CUDA version with `nvidia-smi`.

### Running

```bash
python benchmarks/check_gpu.py          # verify the setup first
python benchmarks/benchmark_gpu.py
```

Options:

```bash
python benchmarks/benchmark_gpu.py --quick
python benchmarks/benchmark_gpu.py --sizes 10000,50000 --seq-len 500 --k 8
```

### What it reports

For each operation and batch size:

| Column | Meaning |
|---|---|
| `seqcore_cpu` | Seqcore's real performance through its public API |
| `cupy_gpu_compute` | Kernel only, data already resident on the device. The ceiling. |
| `cupy_gpu_end2end` | Host→device transfer + kernel + result copy back. What a single call from CPU-resident data actually costs. |

Both GPU timings synchronize the CUDA stream inside the timed region — CuPy
launches kernels asynchronously, and an unsynchronized timing measures launch
latency rather than execution.

Before timing anything, the script verifies that its CuPy kernels reproduce
Seqcore's CPU results, and aborts if they do not.

The `end2end` column is the one to take seriously for judging whether a GPU
backend is worthwhile: for memory-bandwidth-bound operations, PCIe transfer can
dominate and erase the kernel's advantage unless the data stays resident on the
device across many operations.

Transfers use ordinary pageable host memory, as a naive port would. Pinned
(page-locked) buffers would achieve higher PCIe bandwidth, so treat `end2end` as
a conservative bound. What matters is the *gap* between the two columns: a large
gap means the operation only pays off if the encoded matrix stays on the device
across several calls.
