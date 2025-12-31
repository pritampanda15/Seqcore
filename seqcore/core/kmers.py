"""K-mer extraction and analysis operations.

Provides efficient k-mer counting, extraction, and spectrum analysis.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seqcore.core.arrays import BioArray


def extract_kmers(
    sequences: BioArray | str | list[str],
    k: int,
    step: int = 1,
) -> list[list[str]]:
    """Extract k-mers from sequence(s).

    Args:
        sequences: Sequences to extract k-mers from.
        k: K-mer length.
        step: Step size for sliding window.

    Returns:
        List of k-mer lists for each sequence.

    Example:
        >>> kmers = sc.extract_kmers(sequences, k=21)

    """
    from seqcore.core.arrays import BioArray

    def _extract(seq: str) -> list[str]:
        return [seq[i : i + k] for i in range(0, len(seq) - k + 1, step)]

    if isinstance(sequences, str):
        return [_extract(sequences)]
    elif isinstance(sequences, BioArray):
        return [_extract(seq) for seq in sequences.sequences]
    else:
        return [_extract(seq) for seq in sequences]


def count_kmers(
    sequences: BioArray | str | list[str],
    k: int,
    normalize: bool = False,
) -> dict[str, int | float]:
    """Count k-mer frequencies across all sequences.

    Args:
        sequences: Sequences to count k-mers from.
        k: K-mer length.
        normalize: If True, return frequencies instead of counts.

    Returns:
        Dictionary mapping k-mers to counts/frequencies.

    Example:
        >>> kmer_counts = sc.count_kmers(sequences, k=21)

    """
    all_kmers = []
    for kmer_list in extract_kmers(sequences, k):
        all_kmers.extend(kmer_list)

    counts = Counter(all_kmers)

    if normalize:
        total = sum(counts.values())
        if total > 0:
            return {kmer: count / total for kmer, count in counts.items()}
        return {}

    return dict(counts)


def kmer_spectrum(
    sequences: BioArray | str | list[str],
    k: int,
    max_count: int = 100,
) -> np.ndarray:
    """Calculate k-mer spectrum (frequency of frequencies).

    Args:
        sequences: Sequences to analyze.
        k: K-mer length.
        max_count: Maximum count to include in spectrum.

    Returns:
        Array where index i contains number of k-mers with count i.

    Example:
        >>> spectrum = sc.kmer_spectrum(sequences, k=21)
        >>> # spectrum[5] = number of k-mers appearing exactly 5 times

    """
    counts = count_kmers(sequences, k, normalize=False)

    spectrum = np.zeros(max_count + 1, dtype=np.int64)
    for count in counts.values():
        if count <= max_count:
            spectrum[count] += 1

    return spectrum


def kmer_distance(
    seq1: BioArray | str,
    seq2: BioArray | str,
    k: int,
    metric: str = "jaccard",
) -> float:
    """Calculate k-mer based distance between two sequences.

    Args:
        seq1: First sequence.
        seq2: Second sequence.
        k: K-mer length.
        metric: Distance metric ("jaccard", "cosine", "euclidean").

    Returns:
        Distance value.

    Example:
        >>> dist = sc.kmer_distance(seq1, seq2, k=5)

    """
    from seqcore.core.arrays import BioArray

    if isinstance(seq1, BioArray):
        seq1 = seq1.sequences[0] if len(seq1) == 1 else seq1.sequences
    if isinstance(seq2, BioArray):
        seq2 = seq2.sequences[0] if len(seq2) == 1 else seq2.sequences

    kmers1 = set(extract_kmers(seq1, k)[0])
    kmers2 = set(extract_kmers(seq2, k)[0])

    if metric == "jaccard":
        intersection = len(kmers1 & kmers2)
        union = len(kmers1 | kmers2)
        return 1.0 - (intersection / union) if union > 0 else 1.0

    elif metric == "cosine":
        counts1 = count_kmers(seq1, k)
        counts2 = count_kmers(seq2, k)
        all_kmers = set(counts1.keys()) | set(counts2.keys())

        vec1 = np.array([counts1.get(kmer, 0) for kmer in all_kmers])
        vec2 = np.array([counts2.get(kmer, 0) for kmer in all_kmers])

        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 1.0
        return 1.0 - (dot / (norm1 * norm2))

    elif metric == "euclidean":
        counts1 = count_kmers(seq1, k)
        counts2 = count_kmers(seq2, k)
        all_kmers = set(counts1.keys()) | set(counts2.keys())

        vec1 = np.array([counts1.get(kmer, 0) for kmer in all_kmers])
        vec2 = np.array([counts2.get(kmer, 0) for kmer in all_kmers])

        return float(np.linalg.norm(vec1 - vec2))

    else:
        raise ValueError(f"Unknown metric: {metric}")
