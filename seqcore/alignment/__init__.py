"""Sequence alignment and pattern matching operations.

Provides pairwise alignment, pattern searching, and motif finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from seqcore.core.arrays import BioArray

from seqcore.core.device import get_array_module

# Default scoring matrices
DNA_MATCH = 2
DNA_MISMATCH = -1
DNA_GAP_OPEN = -5
DNA_GAP_EXTEND = -1

BLOSUM62 = {
    ("A", "A"): 4,
    ("A", "R"): -1,
    ("A", "N"): -2,
    ("A", "D"): -2,
    ("A", "C"): 0,
    ("A", "Q"): -1,
    ("A", "E"): -1,
    ("A", "G"): 0,
    ("A", "H"): -2,
    ("A", "I"): -1,
    ("A", "L"): -1,
    ("A", "K"): -1,
    ("A", "M"): -1,
    ("A", "F"): -2,
    ("A", "P"): -1,
    ("A", "S"): 1,
    ("A", "T"): 0,
    ("A", "W"): -3,
    ("A", "Y"): -2,
    ("A", "V"): 0,
    # ... (simplified - full BLOSUM62 would be included)
}


@dataclass
class AlignmentResult:
    """Result of sequence alignment."""

    score: float
    aligned_query: str
    aligned_reference: str
    identity: float
    cigar: str
    start_query: int = 0
    end_query: int = 0
    start_reference: int = 0
    end_reference: int = 0

    def __repr__(self):
        return f"AlignmentResult(score={self.score:.1f}, identity={self.identity:.1%})"


def _needleman_wunsch(
    seq1: str,
    seq2: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> AlignmentResult:
    """Needleman-Wunsch global alignment."""
    m, n = len(seq1), len(seq2)

    # Initialize score matrix
    score = np.zeros((m + 1, n + 1), dtype=np.float32)
    score[0, :] = np.arange(n + 1) * gap
    score[:, 0] = np.arange(m + 1) * gap

    # Fill score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            score[i, j] = max(
                score[i - 1, j - 1] + match_score,
                score[i - 1, j] + gap,
                score[i, j - 1] + gap,
            )

    # Traceback
    aligned1, aligned2 = [], []
    i, j = m, n
    matches = 0
    total = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match_score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            if score[i, j] == score[i - 1, j - 1] + match_score:
                aligned1.append(seq1[i - 1])
                aligned2.append(seq2[j - 1])
                if seq1[i - 1] == seq2[j - 1]:
                    matches += 1
                total += 1
                i -= 1
                j -= 1
                continue

        if i > 0 and score[i, j] == score[i - 1, j] + gap:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            total += 1
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            total += 1
            j -= 1

    aligned1 = "".join(reversed(aligned1))
    aligned2 = "".join(reversed(aligned2))

    # Generate CIGAR
    cigar = _generate_cigar(aligned1, aligned2)

    return AlignmentResult(
        score=float(score[m, n]),
        aligned_query=aligned1,
        aligned_reference=aligned2,
        identity=matches / total if total > 0 else 0.0,
        cigar=cigar,
        end_query=m,
        end_reference=n,
    )


def _smith_waterman(
    seq1: str,
    seq2: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> AlignmentResult:
    """Smith-Waterman local alignment."""
    m, n = len(seq1), len(seq2)

    # Initialize score matrix
    score = np.zeros((m + 1, n + 1), dtype=np.float32)

    # Track max score position
    max_score = 0
    max_i, max_j = 0, 0

    # Fill score matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = match if seq1[i - 1] == seq2[j - 1] else mismatch
            score[i, j] = max(
                0,
                score[i - 1, j - 1] + match_score,
                score[i - 1, j] + gap,
                score[i, j - 1] + gap,
            )
            if score[i, j] > max_score:
                max_score = score[i, j]
                max_i, max_j = i, j

    # Traceback from max score
    aligned1, aligned2 = [], []
    i, j = max_i, max_j
    matches = 0
    total = 0

    while i > 0 and j > 0 and score[i, j] > 0:
        match_score = match if seq1[i - 1] == seq2[j - 1] else mismatch

        if score[i, j] == score[i - 1, j - 1] + match_score:
            aligned1.append(seq1[i - 1])
            aligned2.append(seq2[j - 1])
            if seq1[i - 1] == seq2[j - 1]:
                matches += 1
            total += 1
            i -= 1
            j -= 1
        elif score[i, j] == score[i - 1, j] + gap:
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            total += 1
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            total += 1
            j -= 1

    start_i, start_j = i, j
    aligned1 = "".join(reversed(aligned1))
    aligned2 = "".join(reversed(aligned2))

    cigar = _generate_cigar(aligned1, aligned2)

    return AlignmentResult(
        score=float(max_score),
        aligned_query=aligned1,
        aligned_reference=aligned2,
        identity=matches / total if total > 0 else 0.0,
        cigar=cigar,
        start_query=start_i,
        end_query=max_i,
        start_reference=start_j,
        end_reference=max_j,
    )


def _generate_cigar(aligned1: str, aligned2: str) -> str:
    """Generate CIGAR string from alignment."""
    cigar = []
    current_op = None
    count = 0

    for c1, c2 in zip(aligned1, aligned2):
        if c1 == "-":
            op = "I"  # Insertion
        elif c2 == "-":
            op = "D"  # Deletion
        elif c1 == c2:
            op = "M"  # Match
        else:
            op = "X"  # Mismatch

        if op == current_op:
            count += 1
        else:
            if current_op:
                cigar.append(f"{count}{current_op}")
            current_op = op
            count = 1

    if current_op:
        cigar.append(f"{count}{current_op}")

    return "".join(cigar)


def align(
    query: BioArray | str | list[str],
    reference: BioArray | str,
    method: str = "global",
    match: int = 2,
    mismatch: int = -1,
    gap_open: int = -5,
    gap_extend: int = -1,
) -> AlignmentResult | list[AlignmentResult]:
    """Align sequence(s) to reference.

    Args:
        query: Query sequence(s).
        reference: Reference sequence.
        method: Alignment method ("global" for Needleman-Wunsch, "local" for Smith-Waterman).
        match: Match score.
        mismatch: Mismatch penalty.
        gap_open: Gap opening penalty.
        gap_extend: Gap extension penalty.

    Returns:
        AlignmentResult or list of results.

    Example:
        >>> result = sc.align(query, reference)
        >>> print(result.score, result.identity)

    """
    from seqcore.core.arrays import BioArray

    # Get reference string
    if isinstance(reference, BioArray):
        ref_str = reference[0] if len(reference) == 1 else reference.sequences[0]
    else:
        ref_str = reference

    # Simple gap penalty for basic implementation
    gap = gap_open

    # Get query sequences
    if isinstance(query, str):
        queries = [query]
    elif isinstance(query, BioArray):
        queries = query.sequences
    else:
        queries = query

    # Select alignment function
    align_func = _needleman_wunsch if method == "global" else _smith_waterman

    results = []
    for q in queries:
        result = align_func(q, ref_str, match, mismatch, gap)
        results.append(result)

    if len(results) == 1:
        return results[0]
    return results


def pairwise_distance(
    sequences: BioArray | list[str],
    metric: str = "edit",
) -> np.ndarray:
    """Calculate pairwise distance matrix.

    Args:
        sequences: Sequences to compare.
        metric: Distance metric ("edit", "hamming", "identity").

    Returns:
        Square distance matrix.

    Example:
        >>> dm = sc.pairwise_distance(sequences, metric="edit")

    """
    from seqcore.core.arrays import BioArray

    seqs = sequences.sequences if isinstance(sequences, BioArray) else sequences

    n = len(seqs)
    dm = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            if metric == "edit":
                dist = _edit_distance(seqs[i], seqs[j])
            elif metric == "hamming":
                dist = _hamming_distance(seqs[i], seqs[j])
            elif metric == "identity":
                result = _needleman_wunsch(seqs[i], seqs[j])
                dist = 1.0 - result.identity
            else:
                raise ValueError(f"Unknown metric: {metric}")

            dm[i, j] = dist
            dm[j, i] = dist

    return dm


def _edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)

    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return int(dp[m, n])


def _hamming_distance(s1: str, s2: str) -> int:
    """Calculate Hamming distance (requires equal length)."""
    if len(s1) != len(s2):
        raise ValueError("Hamming distance requires equal length sequences")
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def find_pattern(
    sequences: BioArray | str | list[str],
    pattern: str,
) -> list[list[dict[str, Any]]]:
    """Find regex pattern matches in sequences.

    Args:
        sequences: Sequences to search.
        pattern: Regular expression pattern.

    Returns:
        List of match lists for each sequence.

    Example:
        >>> matches = sc.find_pattern(sequences, "ATG[ACGT]{30,1000}T(AA|AG|GA)")

    """
    from seqcore.core.arrays import BioArray

    if isinstance(sequences, str):
        seqs = [sequences]
    elif isinstance(sequences, BioArray):
        seqs = sequences.sequences
    else:
        seqs = sequences

    compiled = re.compile(pattern)
    results = []

    for seq in seqs:
        matches = []
        for match in compiled.finditer(seq):
            matches.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "sequence": match.group(),
                    "groups": match.groups(),
                }
            )
        results.append(matches)

    return results


def find_motifs(
    sequences: BioArray | list[str],
    min_length: int = 6,
    max_length: int = 12,
    min_occurrences: int = 2,
) -> list[dict[str, Any]]:
    """Find overrepresented motifs in sequences.

    Args:
        sequences: Sequences to analyze.
        min_length: Minimum motif length.
        max_length: Maximum motif length.
        min_occurrences: Minimum number of occurrences.

    Returns:
        List of motif dictionaries with sequence and count.

    Example:
        >>> motifs = sc.find_motifs(sequences, min_length=6, max_length=12)

    """
    from collections import Counter

    from seqcore.core.arrays import BioArray

    seqs = sequences.sequences if isinstance(sequences, BioArray) else sequences

    # Count all k-mers of each length
    motif_counts = Counter()

    for seq in seqs:
        seen_in_seq = set()
        for k in range(min_length, max_length + 1):
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if kmer not in seen_in_seq:
                    motif_counts[kmer] += 1
                    seen_in_seq.add(kmer)

    # Filter by minimum occurrences
    motifs = [
        {"sequence": motif, "count": count}
        for motif, count in motif_counts.items()
        if count >= min_occurrences
    ]

    # Sort by count
    motifs.sort(key=lambda x: x["count"], reverse=True)

    return motifs


def position_weight_matrix(
    sequences: BioArray | list[str],
    pseudocount: float = 0.5,
) -> np.ndarray:
    """Calculate position weight matrix from aligned sequences.

    Args:
        sequences: Aligned sequences (must be same length).
        pseudocount: Pseudocount for smoothing.

    Returns:
        Position weight matrix (4 x length for DNA, 20 x length for protein).

    Example:
        >>> pwm = sc.position_weight_matrix(aligned_sequences)

    """
    from seqcore.core.arrays import BioArray, DNAArray, RNAArray

    if isinstance(sequences, BioArray):
        seqs = sequences.sequences
        is_dna = isinstance(sequences, (DNAArray, RNAArray))
    else:
        seqs = sequences
        # Detect if DNA/RNA
        sample = seqs[0] if seqs else ""
        is_dna = all(c in "ACGTU" for c in sample.upper())

    if not seqs:
        return np.array([])

    length = len(seqs[0])
    if not all(len(s) == length for s in seqs):
        raise ValueError("All sequences must have the same length for PWM")

    alphabet = "ACGT" if is_dna else "ACDEFGHIKLMNPQRSTVWY"

    n_chars = len(alphabet)
    char_to_idx = {c: i for i, c in enumerate(alphabet)}

    # Count matrix
    counts = np.zeros((n_chars, length), dtype=np.float64)

    for seq in seqs:
        for i, c in enumerate(seq.upper()):
            if c in char_to_idx:
                counts[char_to_idx[c], i] += 1

    # Add pseudocounts and normalize
    counts += pseudocount
    pwm = counts / counts.sum(axis=0)

    return pwm


__all__ = [
    "align",
    "pairwise_distance",
    "find_pattern",
    "find_motifs",
    "position_weight_matrix",
    "AlignmentResult",
]
