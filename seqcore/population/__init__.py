"""Population genetics operations.

Provides functions for variant analysis, population statistics,
and genetic diversity calculations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from seqcore.core.arrays import BioArray


def allele_frequency(
    variants: dict[str, Any],
    sample_indices: list[int] | None = None,
) -> np.ndarray:
    """Calculate allele frequencies from variant data.

    Args:
        variants: Variant dictionary from read_vcf.
        sample_indices: Subset of samples to analyze.

    Returns:
        Array of alternate allele frequencies.

    Example:
        >>> af = sc.allele_frequency(variants)

    """
    if "samples" not in variants or not variants["samples"]:
        return np.array([])

    n_variants = len(variants["pos"])
    frequencies = np.zeros(n_variants)

    for i in range(n_variants):
        sample_data = variants["samples"][i]

        if sample_indices is not None:
            sample_data = [sample_data[j] for j in sample_indices if j < len(sample_data)]

        alt_count = 0
        total_alleles = 0

        for gt_str in sample_data:
            # Parse genotype (e.g., "0/1", "1|1")
            gt = gt_str.split(":")[0] if ":" in gt_str else gt_str
            alleles = gt.replace("|", "/").split("/")

            for allele in alleles:
                if allele.isdigit():
                    total_alleles += 1
                    if int(allele) > 0:
                        alt_count += 1

        if total_alleles > 0:
            frequencies[i] = alt_count / total_alleles

    return frequencies


def minor_allele_frequency(variants: dict[str, Any]) -> np.ndarray:
    """Calculate minor allele frequencies.

    Args:
        variants: Variant dictionary from read_vcf.

    Returns:
        Array of minor allele frequencies (0 to 0.5).

    Example:
        >>> maf = sc.minor_allele_frequency(variants)

    """
    af = allele_frequency(variants)
    return np.minimum(af, 1 - af)


def heterozygosity(variants: dict[str, Any]) -> np.ndarray:
    """Calculate observed heterozygosity per variant.

    Args:
        variants: Variant dictionary from read_vcf.

    Returns:
        Array of heterozygosity values.

    Example:
        >>> het = sc.heterozygosity(variants)

    """
    if "samples" not in variants or not variants["samples"]:
        return np.array([])

    n_variants = len(variants["pos"])
    het_values = np.zeros(n_variants)

    for i in range(n_variants):
        sample_data = variants["samples"][i]

        het_count = 0
        total_samples = 0

        for gt_str in sample_data:
            gt = gt_str.split(":")[0] if ":" in gt_str else gt_str
            alleles = gt.replace("|", "/").split("/")

            if len(alleles) == 2 and all(a.isdigit() for a in alleles):
                total_samples += 1
                if alleles[0] != alleles[1]:
                    het_count += 1

        if total_samples > 0:
            het_values[i] = het_count / total_samples

    return het_values


def fst(
    population1: dict[str, Any],
    population2: dict[str, Any],
    sample_indices1: list[int] | None = None,
    sample_indices2: list[int] | None = None,
) -> float:
    """Calculate Weir-Cockerham Fst between populations.

    Args:
        population1: Variants for population 1.
        population2: Variants for population 2 (same variants).
        sample_indices1: Sample indices for population 1.
        sample_indices2: Sample indices for population 2.

    Returns:
        Fst value (0 to 1).

    Example:
        >>> fst_value = sc.fst(pop1_variants, pop2_variants)

    """
    af1 = allele_frequency(population1, sample_indices1)
    af2 = allele_frequency(population2, sample_indices2)

    if len(af1) == 0 or len(af2) == 0:
        return 0.0

    # Weir-Cockerham Fst
    # Simplified calculation
    p_bar = (af1 + af2) / 2
    het_total = 2 * p_bar * (1 - p_bar)
    het_within = (af1 * (1 - af1) + af2 * (1 - af2)) / 2

    valid = het_total > 0
    if not np.any(valid):
        return 0.0

    fst_per_site = np.zeros_like(het_total)
    fst_per_site[valid] = (het_total[valid] - het_within[valid]) / het_total[valid]

    return float(np.mean(fst_per_site[valid]))


def nucleotide_diversity(
    sequences: BioArray | list[str],
) -> float:
    """Calculate nucleotide diversity (pi).

    Args:
        sequences: Aligned DNA sequences.

    Returns:
        Nucleotide diversity value.

    Example:
        >>> pi = sc.nucleotide_diversity(sequences)

    """
    from seqcore.core.arrays import BioArray

    seqs = sequences.sequences if isinstance(sequences, BioArray) else sequences

    if len(seqs) < 2:
        return 0.0

    n = len(seqs)
    seq_len = len(seqs[0])

    # Count pairwise differences
    total_diff = 0
    n_comparisons = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff = sum(a != b for a, b in zip(seqs[i], seqs[j]) if a != "-" and b != "-")
            total_diff += diff
            n_comparisons += 1

    if n_comparisons == 0 or seq_len == 0:
        return 0.0

    # Pi = average pairwise differences / sequence length
    pi = total_diff / (n_comparisons * seq_len)

    return float(pi)


def tajimas_d(
    sequences: BioArray | list[str],
) -> float:
    """Calculate Tajima's D statistic.

    Args:
        sequences: Aligned DNA sequences.

    Returns:
        Tajima's D value.

    Example:
        >>> d = sc.tajimas_d(sequences)

    """
    from seqcore.core.arrays import BioArray

    seqs = sequences.sequences if isinstance(sequences, BioArray) else sequences

    n = len(seqs)
    if n < 4:
        return 0.0

    seq_len = len(seqs[0])

    # Count segregating sites
    S = 0
    for pos in range(seq_len):
        alleles = {seq[pos] for seq in seqs if seq[pos] != "-"}
        if len(alleles) > 1:
            S += 1

    if S == 0:
        return 0.0

    # Calculate theta_W (Watterson's estimator)
    a1 = sum(1 / i for i in range(1, n))
    theta_W = S / a1

    # Calculate theta_pi (from nucleotide diversity)
    pi = nucleotide_diversity(seqs)
    theta_pi = pi * seq_len

    # Calculate variance components
    a2 = sum(1 / (i**2) for i in range(1, n))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1**2)
    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)

    # Tajima's D
    var_d = e1 * S + e2 * S * (S - 1)
    if var_d <= 0:
        return 0.0

    d = (theta_pi - theta_W) / np.sqrt(var_d)

    return float(d)


def linkage_disequilibrium(
    variants: dict[str, Any],
    max_distance: int | None = None,
) -> np.ndarray:
    """Calculate linkage disequilibrium (r^2) matrix.

    Args:
        variants: Variant dictionary from read_vcf.
        max_distance: Maximum distance between variants (bp).

    Returns:
        Square matrix of r^2 values.

    Example:
        >>> ld_matrix = sc.linkage_disequilibrium(variants)

    """
    if "samples" not in variants or not variants["samples"]:
        return np.array([[]])

    n_variants = len(variants["pos"])
    positions = np.array(variants["pos"])

    ld_matrix = np.zeros((n_variants, n_variants))
    np.fill_diagonal(ld_matrix, 1.0)

    for i in range(n_variants):
        for j in range(i + 1, n_variants):
            # Check distance
            if max_distance is not None and abs(positions[i] - positions[j]) > max_distance:
                continue

            r2 = r_squared(variants, i, j)
            ld_matrix[i, j] = r2
            ld_matrix[j, i] = r2

    return ld_matrix


def r_squared(
    variants: dict[str, Any],
    variant1_idx: int,
    variant2_idx: int,
) -> float:
    """Calculate r^2 between two variants.

    Args:
        variants: Variant dictionary.
        variant1_idx: Index of first variant.
        variant2_idx: Index of second variant.

    Returns:
        r^2 value (0 to 1).

    Example:
        >>> r2 = sc.r_squared(variants, 0, 1)

    """
    if "samples" not in variants or not variants["samples"]:
        return 0.0

    samples1 = variants["samples"][variant1_idx]
    samples2 = variants["samples"][variant2_idx]

    # Parse genotypes
    def parse_gt(gt_str):
        gt = gt_str.split(":")[0] if ":" in gt_str else gt_str
        alleles = gt.replace("|", "/").split("/")
        if len(alleles) == 2 and all(a.isdigit() for a in alleles):
            return int(alleles[0]), int(alleles[1])
        return None

    # Count haplotype frequencies
    hap_counts = {}
    total = 0

    for s1, s2 in zip(samples1, samples2):
        gt1 = parse_gt(s1)
        gt2 = parse_gt(s2)

        if gt1 is None or gt2 is None:
            continue

        # Add both haplotypes (assuming diploid)
        for a1 in gt1:
            for a2 in gt2:
                hap = (min(a1, 1), min(a2, 1))  # Convert to 0/1
                hap_counts[hap] = hap_counts.get(hap, 0) + 0.5
                total += 0.5

    if total == 0:
        return 0.0

    # Calculate frequencies
    p_A = (hap_counts.get((1, 0), 0) + hap_counts.get((1, 1), 0)) / total
    p_B = (hap_counts.get((0, 1), 0) + hap_counts.get((1, 1), 0)) / total
    p_AB = hap_counts.get((1, 1), 0) / total

    # D and D^2
    D = p_AB - p_A * p_B

    # r^2 = D^2 / (p_A * (1-p_A) * p_B * (1-p_B))
    denom = p_A * (1 - p_A) * p_B * (1 - p_B)
    if denom <= 0:
        return 0.0

    return float((D**2) / denom)


def phase_genotypes(
    variants: dict[str, Any],
    window_size: int = 100,
) -> dict[str, Any]:
    """Phase genotypes to reconstruct haplotypes.

    Args:
        variants: Variant dictionary from read_vcf.
        window_size: Window size for phasing.

    Returns:
        Dictionary with phased haplotypes.

    Example:
        >>> haplotypes = sc.phase_genotypes(variants)

    """
    # Simplified phasing - just separate alleles
    if "samples" not in variants or not variants["samples"]:
        return {"haplotypes": [], "sample_names": []}

    n_variants = len(variants["pos"])
    n_samples = len(variants["samples"][0]) if variants["samples"] else 0

    haplotypes = []

    for sample_idx in range(n_samples):
        hap1 = []
        hap2 = []

        for var_idx in range(n_variants):
            gt_str = variants["samples"][var_idx][sample_idx]
            gt = gt_str.split(":")[0] if ":" in gt_str else gt_str
            alleles = gt.replace("|", "/").split("/")

            if len(alleles) == 2:
                hap1.append(alleles[0])
                hap2.append(alleles[1])
            else:
                hap1.append(".")
                hap2.append(".")

        haplotypes.append(("".join(hap1), "".join(hap2)))

    return {
        "haplotypes": haplotypes,
        "sample_names": variants.get("sample_names", []),
        "positions": variants["pos"],
    }


def haplotype_frequency(
    haplotypes: dict[str, Any],
) -> dict[str, float]:
    """Calculate haplotype frequencies.

    Args:
        haplotypes: Phased haplotype data.

    Returns:
        Dictionary mapping haplotypes to frequencies.

    Example:
        >>> freq = sc.haplotype_frequency(haplotypes)

    """
    from collections import Counter

    if "haplotypes" not in haplotypes:
        return {}

    all_haps = []
    for hap1, hap2 in haplotypes["haplotypes"]:
        all_haps.append(hap1)
        all_haps.append(hap2)

    total = len(all_haps)
    if total == 0:
        return {}

    counts = Counter(all_haps)
    return {hap: count / total for hap, count in counts.items()}


__all__ = [
    "allele_frequency",
    "minor_allele_frequency",
    "heterozygosity",
    "fst",
    "nucleotide_diversity",
    "tajimas_d",
    "linkage_disequilibrium",
    "r_squared",
    "phase_genotypes",
    "haplotype_frequency",
]
