#!/usr/bin/env python3
"""Benchmark suite for Seqcore.

Compares Seqcore against Biopython and an idiomatic pure-Python implementation
across the operations Seqcore is designed for. Produces the JSON consumed by
``paper/make_figures.py``.

Every configuration runs in a freshly spawned subprocess so that import order,
allocator state and warmed caches cannot leak between implementations. Reported
times are best-of-N wall clock, which is the appropriate statistic for
throughput comparisons on a machine that is not otherwise quiesced.

Usage::

    python benchmarks/benchmark_suite.py                 # full sweep
    python benchmarks/benchmark_suite.py --quick         # smaller sizes
    python benchmarks/benchmark_suite.py --baseline DIR  # dev only, see below

``--baseline`` points at another checkout of Seqcore and adds it as an extra
series. It exists for tracking performance across commits during development
and is not part of the published comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SEED = 20260825
SEQ_LEN = 1000

BATCH_SIZES = [1000, 5000, 10000, 25000, 50000, 100000]
BATCH_SIZES_QUICK = [1000, 5000, 10000]
ALIGN_LENGTHS = [100, 200, 400, 800, 1600, 3200]
ALIGN_LENGTHS_QUICK = [100, 200, 400]
KMER_KS = [3, 5, 8, 10, 12, 15, 21]
KMER_N = 2000

BATCH_OPS = ["gc_content", "translate", "reverse_complement"]
IMPLS = ["seqcore", "biopython", "python"]


# Best-effort friendly names for Apple silicon desktops/laptops. Anything not
# listed falls back to the raw sysctl identifier, which is unambiguous anyway.
_APPLE_MODELS = {
    "Mac16,10": "Mac mini",
    "Mac16,11": "Mac mini",
    "Mac15,12": "MacBook Air",
    "Mac16,1": "MacBook Pro",
    "Mac16,5": "MacBook Pro",
    "Mac16,6": "MacBook Pro",
    "Mac16,7": "MacBook Pro",
    "Mac16,8": "MacBook Pro",
    "Mac16,9": "Mac Studio",
    "Mac16,3": "iMac",
    "Mac14,3": "Mac mini",
    "Mac14,12": "Mac mini",
    "Mac14,13": "Mac Studio",
    "Mac14,14": "Mac Studio",
}


# Standard genetic code, defined here so the pure-Python baseline is genuinely
# independent of the library being benchmarked.
_STANDARD_CODONS = dict(
    zip(
        [a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"],
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    )
)


def _sysctl(key: str) -> str:
    """Read one sysctl value, returning an empty string if unavailable."""
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _hardware_model() -> str:
    """Machine model, e.g. 'Mac mini (Mac16,11)' or an EC2 instance type."""
    if platform.system() == "Darwin":
        ident = _sysctl("hw.model")
        if ident:
            friendly = _APPLE_MODELS.get(ident)
            return f"{friendly} ({ident})" if friendly else ident
    elif platform.system() == "Linux":
        # EC2 and most cloud providers expose this via DMI.
        for path in (
            "/sys/devices/virtual/dmi/id/product_name",
            "/sys/devices/virtual/dmi/id/board_vendor",
        ):
            try:
                value = Path(path).read_text().strip()
                if value and value.lower() not in ("", "none", "default string"):
                    return value
            except Exception:
                continue
    return ""


def _cpu_model() -> str:
    """Best-effort human-readable CPU name, for the results header."""
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            if out.returncode == 0:
                return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _library_versions() -> dict:
    """Record the versions actually used, so results are self-describing."""
    versions = {}
    for name, mod in (("numpy", "numpy"), ("biopython", "Bio"), ("seqcore", "seqcore")):
        try:
            versions[name] = __import__(mod).__version__
        except Exception:
            versions[name] = None
    return versions


def make_sequences(n: int, length: int, seed: int = SEED) -> list[str]:
    """Generate n uniformly random DNA sequences of the given length."""
    rng = random.Random(seed)
    return ["".join(rng.choices("ACGT", k=length)) for _ in range(n)]


def timeit(fn, repeats: int = 5) -> float:
    """Best-of-N wall clock, in seconds."""
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


# --------------------------------------------------------------------- workers


def batch_seqcore(op: str, n: int) -> float:
    import seqcore as sc

    arr = sc.DNAArray(make_sequences(n, SEQ_LEN))
    fns = {
        "gc_content": lambda: sc.gc_content(arr),
        "translate": lambda: sc.translate(arr),
        "reverse_complement": lambda: sc.reverse_complement(arr),
    }
    return timeit(fns[op])


def batch_biopython(op: str, n: int) -> float | None:
    try:
        from Bio.Seq import Seq
        from Bio.SeqUtils import gc_fraction
    except ImportError:
        return None

    bio = [Seq(s) for s in make_sequences(n, SEQ_LEN)]
    fns = {
        "gc_content": lambda: [gc_fraction(s) * 100 for s in bio],
        "translate": lambda: [s.translate() for s in bio],
        "reverse_complement": lambda: [s.reverse_complement() for s in bio],
    }
    return timeit(fns[op])


def batch_python(op: str, n: int) -> float | None:
    """Idiomatic pure Python, using the fastest obvious stdlib approach."""
    seqs = make_sequences(n, SEQ_LEN)
    if op == "gc_content":
        return timeit(lambda: [(s.count("G") + s.count("C")) / len(s) * 100 for s in seqs])
    if op == "reverse_complement":
        table = str.maketrans("ACGT", "TGCA")
        return timeit(lambda: [s.translate(table)[::-1] for s in seqs])
    if op == "translate":

        def run():
            table = _STANDARD_CODONS
            out = []
            for s in seqs:
                out.append("".join(table.get(s[i : i + 3], "X") for i in range(0, len(s) - 2, 3)))
            return out

        return timeit(run, repeats=3)
    return None


def align_seqcore(length: int) -> float:
    import seqcore as sc

    a, b = make_sequences(2, length)
    return timeit(lambda: sc.align(a, b), repeats=3)


def align_biopython(length: int) -> float | None:
    try:
        from Bio import Align
    except ImportError:
        return None

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score, aligner.mismatch_score = 2, -1
    aligner.open_gap_score = aligner.extend_gap_score = -5
    a, b = make_sequences(2, length)
    return timeit(lambda: aligner.score(a, b), repeats=3)


def align_python(length: int) -> float | None:
    """Scalar Needleman-Wunsch. Skipped above 400 bp, where it is impractical."""
    if length > 400:
        return None
    a, b = make_sequences(2, length)

    def run():
        m, n = len(a), len(b)
        prev = [j * -5 for j in range(n + 1)]
        for i in range(1, m + 1):
            cur = [i * -5] + [0] * n
            ai = a[i - 1]
            for j in range(1, n + 1):
                cur[j] = max(
                    prev[j - 1] + (2 if ai == b[j - 1] else -1),
                    prev[j] - 5,
                    cur[j - 1] - 5,
                )
            prev = cur
        return prev[n]

    return timeit(run, repeats=1)


def kmer_seqcore(k: int) -> float:
    import seqcore as sc

    arr = sc.DNAArray(make_sequences(KMER_N, SEQ_LEN))
    return timeit(lambda: sc.count_kmers(arr, k=k), repeats=3)


def kmer_python(k: int) -> float:
    from collections import Counter

    seqs = make_sequences(KMER_N, SEQ_LEN)

    def run():
        counts: Counter = Counter()
        for s in seqs:
            for i in range(len(s) - k + 1):
                counts[s[i : i + k]] += 1
        return counts

    return timeit(run, repeats=3)


def kmer_biopython(k: int) -> None:
    """Biopython has no batch k-mer counter; reported as not applicable."""
    return None


DISPATCH = {
    ("batch", "seqcore"): batch_seqcore,
    ("batch", "biopython"): batch_biopython,
    ("batch", "python"): batch_python,
    ("align", "seqcore"): align_seqcore,
    ("align", "biopython"): align_biopython,
    ("align", "python"): align_python,
    ("kmer", "seqcore"): kmer_seqcore,
    ("kmer", "biopython"): kmer_biopython,
    ("kmer", "python"): kmer_python,
}


def worker_main() -> None:
    kind, impl, arg, syspath = sys.argv[2:6]
    if syspath:
        sys.path.insert(0, syspath)
    fn = DISPATCH[(kind, impl)]
    if kind == "batch":
        op, n = arg.split(":")
        value = fn(op, int(n))
    else:
        value = fn(int(arg))
    print(json.dumps({"seconds": value}))


# ---------------------------------------------------------------------- driver


def launch(kind: str, impl: str, arg, syspath: str = "") -> float | None:
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", kind, impl, str(arg), syspath],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1:] or ["unknown error"]
        print(f"      ! {impl} failed: {tail[0][:90]}")
        return None
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])["seconds"]
    except (ValueError, IndexError, KeyError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smaller sizes for a fast check")
    ap.add_argument("--baseline", default="", help="dev only: extra checkout to compare")
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    sizes = BATCH_SIZES_QUICK if args.quick else BATCH_SIZES
    lengths = ALIGN_LENGTHS_QUICK if args.quick else ALIGN_LENGTHS
    impls = list(IMPLS) + (["baseline"] if args.baseline else [])

    def path_for(impl: str) -> str:
        return args.baseline if impl == "baseline" else ""

    def run_impl(impl: str) -> str:
        return "seqcore" if impl == "baseline" else impl

    print("=" * 70)
    print("SEQCORE BENCHMARK SUITE")
    print(f"{platform.platform()} | Python {platform.python_version()}")
    print("=" * 70)

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_model": _cpu_model(),
        "hardware_model": _hardware_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "library_versions": _library_versions(),
        "metric": "best-of-N wall clock (seconds)",
        "seq_length": SEQ_LEN,
        "batch": {},
        "align": {"lengths": lengths},
        "kmer": {"ks": KMER_KS, "n_sequences": KMER_N},
    }

    for op in BATCH_OPS:
        print(f"\n[batch] {op}  ({SEQ_LEN} bp sequences)")
        data["batch"][op] = {"sizes": sizes}
        for impl in impls:
            series = [launch("batch", run_impl(impl), f"{op}:{n}", path_for(impl)) for n in sizes]
            data["batch"][op][impl] = series
            shown = ", ".join("--" if v is None else f"{v:.4f}" for v in series)
            print(f"    {impl:10s} {shown}")

    print("\n[align] global alignment")
    for impl in impls:
        series = [launch("align", run_impl(impl), L, path_for(impl)) for L in lengths]
        data["align"][impl] = series
        shown = ", ".join("--" if v is None else f"{v:.4f}" for v in series)
        print(f"    {impl:10s} {shown}")

    print(f"\n[kmer] counting  ({KMER_N} x {SEQ_LEN} bp)")
    for impl in impls:
        series = [launch("kmer", run_impl(impl), k, path_for(impl)) for k in KMER_KS]
        data["kmer"][impl] = series
        shown = ", ".join("--" if v is None else f"{v:.4f}" for v in series)
        print(f"    {impl:10s} {shown}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(data, indent=2))
    print(f"\nSaved: {out_file}")
    print("Regenerate figures with: python paper/make_figures.py")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main()
    else:
        main()
