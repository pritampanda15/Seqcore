#!/usr/bin/env python3
"""
Seqcore Performance Benchmarks
Author: Dr. Pritam Kumar Panda @ Stanford University

Compares seqcore performance against:
- Biopython (sequence analysis)
- Standard Python (baseline)

Run: python benchmarks/benchmark_comparison.py
"""

import gc
import json
import random
import string
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def generate_random_dna(length: int) -> str:
    """Generate random DNA sequence."""
    return "".join(random.choices("ACGT", k=length))


def generate_test_sequences(n_sequences: int, seq_length: int) -> list[str]:
    """Generate test sequences."""
    return [generate_random_dna(seq_length) for _ in range(n_sequences)]


def benchmark_gc_content():
    """Benchmark GC content calculation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: GC Content Calculation")
    print("=" * 60)

    results = {"operation": "gc_content", "sizes": [], "seqcore": [], "biopython": [], "python": []}

    for n_seq in [100, 1000, 10000, 50000]:
        seq_length = 1000
        sequences = generate_test_sequences(n_seq, seq_length)
        print(f"\n{n_seq} sequences x {seq_length} bp:")
        results["sizes"].append(n_seq)

        # Seqcore
        import seqcore as sc

        dna = sc.DNAArray(sequences)
        gc.collect()
        start = time.perf_counter()
        gc_sc = sc.gc_content(dna)
        sc_time = time.perf_counter() - start
        results["seqcore"].append(sc_time)
        print(f"  Seqcore:   {sc_time:.4f}s")

        # Biopython
        try:
            from Bio.SeqUtils import gc_fraction

            gc.collect()
            start = time.perf_counter()
            gc_bp = [gc_fraction(seq) * 100 for seq in sequences]
            bp_time = time.perf_counter() - start
            results["biopython"].append(bp_time)
            print(f"  Biopython: {bp_time:.4f}s")
        except ImportError:
            results["biopython"].append(None)
            print("  Biopython: not installed")

        # Pure Python
        gc.collect()
        start = time.perf_counter()
        gc_py = [(s.count("G") + s.count("C")) / len(s) * 100 for s in sequences]
        py_time = time.perf_counter() - start
        results["python"].append(py_time)
        print(f"  Python:    {py_time:.4f}s")

        # Speedup
        if results["biopython"][-1]:
            print(f"  Speedup vs Biopython: {bp_time / sc_time:.1f}x")
        print(f"  Speedup vs Python: {py_time / sc_time:.1f}x")

    return results


def benchmark_translation():
    """Benchmark DNA to protein translation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: DNA Translation")
    print("=" * 60)

    results = {"operation": "translation", "sizes": [], "seqcore": [], "biopython": [], "python": []}

    # Standard genetic code
    codon_table = {
        "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
        "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
        "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
        "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
        "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
        "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
        "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
        "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
        "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
        "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
        "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
        "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
        "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
        "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
        "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
        "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    }

    for n_seq in [100, 1000, 10000]:
        seq_length = 999  # Divisible by 3
        sequences = generate_test_sequences(n_seq, seq_length)
        print(f"\n{n_seq} sequences x {seq_length} bp:")
        results["sizes"].append(n_seq)

        # Seqcore
        import seqcore as sc

        dna = sc.DNAArray(sequences)
        gc.collect()
        start = time.perf_counter()
        proteins = sc.translate(dna)
        sc_time = time.perf_counter() - start
        results["seqcore"].append(sc_time)
        print(f"  Seqcore:   {sc_time:.4f}s")

        # Biopython
        try:
            from Bio.Seq import Seq

            gc.collect()
            start = time.perf_counter()
            proteins_bp = [str(Seq(s).translate()) for s in sequences]
            bp_time = time.perf_counter() - start
            results["biopython"].append(bp_time)
            print(f"  Biopython: {bp_time:.4f}s")
        except ImportError:
            results["biopython"].append(None)
            print("  Biopython: not installed")

        # Pure Python
        def translate_py(seq):
            return "".join(codon_table.get(seq[i : i + 3], "X") for i in range(0, len(seq) - 2, 3))

        gc.collect()
        start = time.perf_counter()
        proteins_py = [translate_py(s) for s in sequences]
        py_time = time.perf_counter() - start
        results["python"].append(py_time)
        print(f"  Python:    {py_time:.4f}s")

        # Speedup
        if results["biopython"][-1]:
            print(f"  Speedup vs Biopython: {bp_time / sc_time:.1f}x")
        print(f"  Speedup vs Python: {py_time / sc_time:.1f}x")

    return results


def benchmark_reverse_complement():
    """Benchmark reverse complement."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Reverse Complement")
    print("=" * 60)

    results = {
        "operation": "reverse_complement",
        "sizes": [],
        "seqcore": [],
        "biopython": [],
        "python": [],
    }

    complement = {"A": "T", "T": "A", "G": "C", "C": "G"}

    for n_seq in [100, 1000, 10000, 50000]:
        seq_length = 1000
        sequences = generate_test_sequences(n_seq, seq_length)
        print(f"\n{n_seq} sequences x {seq_length} bp:")
        results["sizes"].append(n_seq)

        # Seqcore
        import seqcore as sc

        dna = sc.DNAArray(sequences)
        gc.collect()
        start = time.perf_counter()
        rc = sc.reverse_complement(dna)
        sc_time = time.perf_counter() - start
        results["seqcore"].append(sc_time)
        print(f"  Seqcore:   {sc_time:.4f}s")

        # Biopython
        try:
            from Bio.Seq import Seq

            gc.collect()
            start = time.perf_counter()
            rc_bp = [str(Seq(s).reverse_complement()) for s in sequences]
            bp_time = time.perf_counter() - start
            results["biopython"].append(bp_time)
            print(f"  Biopython: {bp_time:.4f}s")
        except ImportError:
            results["biopython"].append(None)
            print("  Biopython: not installed")

        # Pure Python
        gc.collect()
        start = time.perf_counter()
        rc_py = ["".join(complement[b] for b in s[::-1]) for s in sequences]
        py_time = time.perf_counter() - start
        results["python"].append(py_time)
        print(f"  Python:    {py_time:.4f}s")

        # Speedup
        if results["biopython"][-1]:
            print(f"  Speedup vs Biopython: {bp_time / sc_time:.1f}x")
        print(f"  Speedup vs Python: {py_time / sc_time:.1f}x")

    return results


def benchmark_file_io():
    """Benchmark file I/O operations."""
    print("\n" + "=" * 60)
    print("BENCHMARK: FASTA File I/O")
    print("=" * 60)

    import tempfile

    results = {"operation": "fasta_io", "sizes": [], "seqcore": [], "biopython": []}

    for n_seq in [1000, 10000, 50000]:
        seq_length = 1000
        sequences = generate_test_sequences(n_seq, seq_length)
        print(f"\n{n_seq} sequences x {seq_length} bp:")
        results["sizes"].append(n_seq)

        # Create temp FASTA file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")
            temp_path = f.name

        # Seqcore
        import seqcore as sc

        gc.collect()
        start = time.perf_counter()
        data = sc.read(temp_path)
        sc_time = time.perf_counter() - start
        results["seqcore"].append(sc_time)
        print(f"  Seqcore:   {sc_time:.4f}s")

        # Biopython
        try:
            from Bio import SeqIO

            gc.collect()
            start = time.perf_counter()
            records = list(SeqIO.parse(temp_path, "fasta"))
            bp_time = time.perf_counter() - start
            results["biopython"].append(bp_time)
            print(f"  Biopython: {bp_time:.4f}s")
            print(f"  Speedup: {bp_time / sc_time:.1f}x")
        except ImportError:
            results["biopython"].append(None)
            print("  Biopython: not installed")

        # Cleanup
        Path(temp_path).unlink()

    return results


def benchmark_pairwise_alignment():
    """Benchmark pairwise sequence alignment."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Pairwise Alignment")
    print("=" * 60)

    results = {"operation": "pairwise_alignment", "sizes": [], "seqcore": [], "biopython": []}

    for seq_length in [100, 500, 1000]:
        seq1 = generate_random_dna(seq_length)
        seq2 = generate_random_dna(seq_length)
        print(f"\nSequence length: {seq_length} bp")
        results["sizes"].append(seq_length)

        # Seqcore
        import seqcore as sc

        gc.collect()
        start = time.perf_counter()
        result = sc.align(seq1, seq2)
        sc_time = time.perf_counter() - start
        results["seqcore"].append(sc_time)
        print(f"  Seqcore:   {sc_time:.4f}s (score: {result.score})")

        # Biopython
        try:
            from Bio import pairwise2

            gc.collect()
            start = time.perf_counter()
            alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
            bp_time = time.perf_counter() - start
            results["biopython"].append(bp_time)
            print(f"  Biopython: {bp_time:.4f}s")
            print(f"  Speedup: {bp_time / sc_time:.1f}x")
        except ImportError:
            results["biopython"].append(None)
            print("  Biopython: not installed")

    return results


def generate_summary_table(all_results: list[dict]):
    """Generate summary table."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print("\n| Operation | Dataset Size | Seqcore | Biopython | Speedup |")
    print("|-----------|--------------|---------|-----------|---------|")

    for result in all_results:
        op = result["operation"]
        for i, size in enumerate(result["sizes"]):
            sc_val = result["seqcore"][i]
            bp = result.get("biopython", [None] * len(result["sizes"]))[i]
            if bp:
                speedup = f"{bp / sc_val:.1f}x"
                print(f"| {op:20} | {size:12} | {sc_val:.4f}s | {bp:.4f}s | {speedup:7} |")
            else:
                print(f"| {op:20} | {size:12} | {sc_val:.4f}s | N/A | N/A |")


def main():
    """Run all benchmarks."""
    print("#" * 60)
    print("# SEQCORE PERFORMANCE BENCHMARKS")
    print(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("# Author: Dr. Pritam Kumar Panda @ Stanford University")
    print("#" * 60)

    all_results = []

    all_results.append(benchmark_gc_content())
    all_results.append(benchmark_translation())
    all_results.append(benchmark_reverse_complement())
    all_results.append(benchmark_file_io())
    all_results.append(benchmark_pairwise_alignment())

    generate_summary_table(all_results)

    # Save results
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
