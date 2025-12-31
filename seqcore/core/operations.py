"""Vectorized sequence operations.

Provides fast, batch-capable operations on biological sequences
including GC content, molecular weight, transcription, translation, etc.

Optimized for high-performance using NumPy vectorization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from seqcore.core.arrays import BioArray, DNAArray, ProteinArray, RNAArray


# Codon translation table (Standard genetic code)
CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# RNA codon table
RNA_CODON_TABLE = {k.replace("T", "U"): v for k, v in CODON_TABLE.items()}

# Amino acid molecular weights (in Daltons)
AA_WEIGHTS = {
    "A": 89.09,
    "R": 174.20,
    "N": 132.12,
    "D": 133.10,
    "C": 121.15,
    "E": 147.13,
    "Q": 146.15,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "L": 131.17,
    "K": 146.19,
    "M": 149.21,
    "F": 165.19,
    "P": 115.13,
    "S": 105.09,
    "T": 119.12,
    "W": 204.23,
    "Y": 181.19,
    "V": 117.15,
    "*": 0.0,
    "X": 0.0,
    "-": 0.0,
}

# Nucleotide molecular weights
NT_WEIGHTS = {"A": 331.2, "C": 307.2, "G": 347.2, "T": 322.2, "U": 324.2, "N": 327.0}

# DNA encoding: A=0, C=1, G=2, T=3, N=4
# Pre-compute complement lookup: 0->3, 1->2, 2->1, 3->0, 4->4
_DNA_COMPLEMENT_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.uint8)
_RNA_COMPLEMENT_TABLE = np.array([3, 2, 1, 0, 4], dtype=np.uint8)  # A=0->U=3, etc.

# Pre-compute codon to amino acid lookup (64 codons = 4^3)
# Encoding: codon_idx = base1*16 + base2*4 + base3
_CODON_TO_AA = np.zeros(125, dtype=np.uint8)  # 5^3 = 125 (including N)
_AA_CHARS = "ACDEFGHIKLMNPQRSTVWY*X"
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_CHARS)}

# Build codon lookup table
for codon, aa in CODON_TABLE.items():
    _base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    b1, b2, b3 = _base_map[codon[0]], _base_map[codon[1]], _base_map[codon[2]]
    idx = b1 * 25 + b2 * 5 + b3
    _CODON_TO_AA[idx] = _AA_TO_IDX.get(aa, _AA_TO_IDX["X"])

# Fill in codons with N -> X
for i in range(125):
    if _CODON_TO_AA[i] == 0 and i not in [
        b1 * 25 + b2 * 5 + b3 for b1 in range(4) for b2 in range(4) for b3 in range(4)
    ]:
        _CODON_TO_AA[i] = _AA_TO_IDX["X"]

# Decode table for amino acids
_AA_DECODE_TABLE = np.array([ord(c) for c in _AA_CHARS], dtype=np.uint8)


def gc_content(sequences: BioArray | str | list[str]) -> np.ndarray:
    """Calculate GC content for sequence(s) using vectorized operations.

    Args:
        sequences: DNA/RNA sequences (BioArray, string, or list of strings).

    Returns:
        Array of GC percentages (0-100).

    Example:
        >>> gc = sc.gc_content(dna_sequences)
        >>> print(gc)  # [45.2, 52.1, 38.9, ...]

    """
    from seqcore.core.arrays import BioArray

    if isinstance(sequences, BioArray):
        # Use encoded data directly for maximum performance
        # DNA encoding: A=0, C=1, G=2, T=3, N=4
        data = sequences.encoded
        lengths = sequences.lengths

        # G=2, C=1 -> count where value is 1 or 2
        gc_mask = (data == 1) | (data == 2)

        # Sum GC bases for each sequence, considering actual length
        n_seqs = len(sequences)
        results = np.zeros(n_seqs, dtype=np.float32)

        for i in range(n_seqs):
            gc_count = np.sum(gc_mask[i, : lengths[i]])
            results[i] = (gc_count / lengths[i] * 100) if lengths[i] > 0 else 0.0

        return results

    # Fallback for string input - still optimized
    seq_list = [sequences] if isinstance(sequences, str) else sequences

    results = np.zeros(len(seq_list), dtype=np.float32)
    for i, seq in enumerate(seq_list):
        # Use numpy for counting
        arr = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
        gc_count = np.sum((arr == ord("G")) | (arr == ord("C")))
        results[i] = (gc_count / len(seq) * 100) if len(seq) > 0 else 0.0

    return results


def length(sequences: BioArray | str | list[str]) -> np.ndarray:
    """Get lengths of sequence(s).

    Args:
        sequences: Sequences (BioArray, string, or list of strings).

    Returns:
        Array of sequence lengths.

    """
    from seqcore.core.arrays import BioArray

    if isinstance(sequences, str):
        return np.array([len(sequences)])
    elif isinstance(sequences, BioArray):
        return sequences.lengths.copy()
    else:
        return np.array([len(s) for s in sequences])


def molecular_weight(sequences: BioArray | str | list[str], seq_type: str = "auto") -> np.ndarray:
    """Calculate molecular weight for sequence(s).

    Args:
        sequences: Sequences (BioArray, string, or list of strings).
        seq_type: Sequence type ("dna", "rna", "protein", or "auto").

    Returns:
        Array of molecular weights in Daltons.

    """
    from seqcore.core.arrays import BioArray, ProteinArray, RNAArray

    if isinstance(sequences, str):
        seq_list = [sequences]
        if seq_type == "auto":
            seq_type = "dna"
    elif isinstance(sequences, BioArray):
        seq_list = sequences.sequences
        if seq_type == "auto":
            if isinstance(sequences, ProteinArray):
                seq_type = "protein"
            elif isinstance(sequences, RNAArray):
                seq_type = "rna"
            else:
                seq_type = "dna"
    else:
        seq_list = sequences
        if seq_type == "auto":
            seq_type = "dna"

    weights = AA_WEIGHTS if seq_type == "protein" else NT_WEIGHTS
    results = np.zeros(len(seq_list), dtype=np.float64)

    for i, seq in enumerate(seq_list):
        mw = sum(weights.get(c.upper(), 0.0) for c in seq)
        if seq_type == "protein" and len(seq) > 1:
            mw -= 18.015 * (len(seq) - 1)
        elif seq_type in ("dna", "rna") and len(seq) > 1:
            mw -= 61.97 * (len(seq) - 1)
        results[i] = mw

    return results


def reverse(sequences: BioArray | str | list[str]) -> BioArray | str | list[str]:
    """Reverse sequence(s) using vectorized operations.

    Args:
        sequences: Sequences to reverse.

    Returns:
        Reversed sequences in same format as input.

    """
    from seqcore.core.arrays import BioArray

    if isinstance(sequences, str):
        return sequences[::-1]
    elif isinstance(sequences, BioArray):
        # Vectorized reverse on encoded data
        reversed_data = sequences.encoded[:, ::-1].copy()
        # Shift to align with original lengths
        n_seqs = len(sequences)
        max_len = reversed_data.shape[1]

        for i in range(n_seqs):
            length = sequences.lengths[i]
            if length < max_len:
                # Move reversed sequence to the front
                reversed_data[i, :length] = reversed_data[i, max_len - length : max_len]
                reversed_data[i, length:] = 0

        return type(sequences).from_numpy(reversed_data, sequences.lengths)
    else:
        return [seq[::-1] for seq in sequences]


def complement(
    sequences: BioArray | str | list[str], seq_type: str = "dna"
) -> BioArray | str | list[str]:
    """Get complement of sequence(s) using vectorized operations.

    Args:
        sequences: DNA or RNA sequences.
        seq_type: "dna" or "rna".

    Returns:
        Complementary sequences.

    """
    from seqcore.core.arrays import BioArray, DNAArray, RNAArray

    comp_table = _RNA_COMPLEMENT_TABLE if seq_type == "rna" else _DNA_COMPLEMENT_TABLE

    if isinstance(sequences, str):
        comp_map = {
            "A": "U" if seq_type == "rna" else "T",
            "T": "A",
            "U": "A",
            "C": "G",
            "G": "C",
            "N": "N",
        }
        return "".join(comp_map.get(c.upper(), c) for c in sequences)

    elif isinstance(sequences, BioArray):
        # Vectorized complement using lookup table
        comp_data = comp_table[sequences.encoded]
        if isinstance(sequences, RNAArray) or seq_type == "rna":
            return RNAArray.from_numpy(comp_data, sequences.lengths)
        return DNAArray.from_numpy(comp_data, sequences.lengths)
    else:
        comp_map = {
            "A": "U" if seq_type == "rna" else "T",
            "T": "A",
            "U": "A",
            "C": "G",
            "G": "C",
            "N": "N",
        }
        return ["".join(comp_map.get(c.upper(), c) for c in seq) for seq in sequences]


def reverse_complement(
    sequences: BioArray | str | list[str], seq_type: str = "dna"
) -> BioArray | str | list[str]:
    """Get reverse complement of sequence(s) using vectorized operations.

    Args:
        sequences: DNA or RNA sequences.
        seq_type: "dna" or "rna".

    Returns:
        Reverse complementary sequences.

    """
    from seqcore.core.arrays import BioArray, DNAArray, RNAArray

    comp_table = _RNA_COMPLEMENT_TABLE if seq_type == "rna" else _DNA_COMPLEMENT_TABLE

    if isinstance(sequences, str):
        comp_map = {
            "A": "U" if seq_type == "rna" else "T",
            "T": "A",
            "U": "A",
            "C": "G",
            "G": "C",
            "N": "N",
        }
        return "".join(comp_map.get(c.upper(), c) for c in sequences[::-1])

    elif isinstance(sequences, BioArray):
        # Vectorized reverse complement
        # First complement, then reverse
        comp_data = comp_table[sequences.encoded]

        # Reverse along sequence axis
        n_seqs = len(sequences)

        for i in range(n_seqs):
            length = sequences.lengths[i]
            comp_data[i, :length] = comp_data[i, :length][::-1]

        if isinstance(sequences, RNAArray) or seq_type == "rna":
            return RNAArray.from_numpy(comp_data, sequences.lengths)
        return DNAArray.from_numpy(comp_data, sequences.lengths)
    else:
        comp_map = {
            "A": "U" if seq_type == "rna" else "T",
            "T": "A",
            "U": "A",
            "C": "G",
            "G": "C",
            "N": "N",
        }
        return ["".join(comp_map.get(c.upper(), c) for c in seq[::-1]) for seq in sequences]


def transcribe(sequences: DNAArray | str | list[str]) -> RNAArray | str | list[str]:
    """Transcribe DNA to RNA.

    Args:
        sequences: DNA sequences.

    Returns:
        RNA sequences (T -> U).

    """
    from seqcore.core.arrays import DNAArray, RNAArray

    if isinstance(sequences, str):
        return sequences.upper().replace("T", "U")
    elif isinstance(sequences, DNAArray):
        # DNA and RNA share the same encoding except T(3) vs U(3)
        # So we can directly reuse the encoded data
        return RNAArray.from_numpy(sequences.encoded.copy(), sequences.lengths)
    else:
        return [seq.upper().replace("T", "U") for seq in sequences]


def translate(
    sequences: BioArray | str | list[str],
    frame: int = 0,
    table: int = 1,
    stop_symbol: str = "*",
    to_stop: bool = False,
) -> ProteinArray | str | list[str]:
    """Translate DNA/RNA sequences to protein using vectorized operations.

    Args:
        sequences: DNA or RNA sequences.
        frame: Reading frame (0, 1, or 2).
        table: Genetic code table number (1=standard).
        stop_symbol: Character to use for stop codons.
        to_stop: If True, stop translation at first stop codon.

    Returns:
        Protein sequences.

    """
    from seqcore.core.arrays import BioArray, ProteinArray

    def _translate_fast(seq: str) -> str:
        """Fast translation using lookup table."""
        seq = seq.upper().replace("U", "T")
        n = len(seq)
        n_codons = (n - frame) // 3

        if n_codons <= 0:
            return ""

        protein_chars = []
        for i in range(n_codons):
            pos = frame + i * 3
            codon = seq[pos : pos + 3]
            if len(codon) == 3:
                aa = CODON_TABLE.get(codon, "X")
                if to_stop and aa == "*":
                    break
                protein_chars.append(aa if aa != "*" else stop_symbol)

        return "".join(protein_chars)

    if isinstance(sequences, str):
        return _translate_fast(sequences)
    elif isinstance(sequences, BioArray):
        proteins = [_translate_fast(seq) for seq in sequences.sequences]
        return ProteinArray(proteins, ids=sequences.ids)
    else:
        return [_translate_fast(seq) for seq in sequences]


def extract_codons(sequences: BioArray | str | list[str], frame: int = 0) -> list[list[str]]:
    """Extract codons from sequence(s).

    Args:
        sequences: DNA or RNA sequences.
        frame: Reading frame (0, 1, or 2).

    Returns:
        List of codon lists for each sequence.

    """
    from seqcore.core.arrays import BioArray

    def _extract(seq: str) -> list[str]:
        return [seq[i : i + 3] for i in range(frame, len(seq) - 2, 3)]

    if isinstance(sequences, str):
        return [_extract(sequences)]
    elif isinstance(sequences, BioArray):
        return [_extract(seq) for seq in sequences.sequences]
    else:
        return [_extract(seq) for seq in sequences]


def codon_frequency(sequences: BioArray | str | list[str], frame: int = 0) -> dict[str, float]:
    """Calculate codon frequency distribution.

    Args:
        sequences: DNA or RNA sequences.
        frame: Reading frame (0, 1, or 2).

    Returns:
        Dictionary mapping codons to frequencies.

    """
    from collections import Counter

    all_codons = []
    for codon_list in extract_codons(sequences, frame):
        all_codons.extend(codon_list)

    total = len(all_codons)
    if total == 0:
        return {}

    counts = Counter(all_codons)
    return {codon: count / total for codon, count in counts.items()}


def codon_adaptation_index(
    sequences: BioArray | str | list[str],
    reference_table: dict[str, float],
    frame: int = 0,
) -> np.ndarray:
    """Calculate Codon Adaptation Index (CAI).

    Args:
        sequences: DNA sequences.
        reference_table: Reference codon usage table.
        frame: Reading frame.

    Returns:
        Array of CAI values for each sequence.

    """
    codon_lists = extract_codons(sequences, frame)
    results = []

    for codons in codon_lists:
        if not codons:
            results.append(0.0)
            continue

        w_values = [reference_table.get(c, 0.1) for c in codons]
        log_sum = sum(np.log(w) for w in w_values if w > 0)
        cai = np.exp(log_sum / len(codons)) if codons else 0.0
        results.append(cai)

    return np.array(results)


def quality_filter(
    sequences: BioArray,
    min_q: int = 20,
    min_length: int = 0,
    max_n: float = 0.1,
) -> np.ndarray:
    """Filter sequences by quality criteria using vectorized operations.

    Args:
        sequences: Sequences with quality scores.
        min_q: Minimum average quality score.
        min_length: Minimum sequence length.
        max_n: Maximum fraction of N bases allowed.

    Returns:
        Boolean mask of sequences passing filter.

    """
    n_seqs = len(sequences)
    mask = np.ones(n_seqs, dtype=bool)

    # Length filter (vectorized)
    if min_length > 0:
        mask &= sequences.lengths >= min_length

    # Quality filter (vectorized where possible)
    if sequences.qualities is not None and min_q > 0:
        # Calculate mean quality for each sequence
        for i in range(n_seqs):
            if mask[i]:
                avg_q = np.mean(sequences.qualities[i][: sequences.lengths[i]])
                mask[i] = avg_q >= min_q

    # N content filter using encoded data
    if max_n < 1.0:
        data = sequences.encoded
        for i in range(n_seqs):
            if mask[i]:
                n_count = np.sum(data[i, : sequences.lengths[i]] == 4)  # N=4
                n_frac = n_count / sequences.lengths[i] if sequences.lengths[i] > 0 else 0
                mask[i] = n_frac <= max_n

    return mask


def trim_adapters(
    sequences: BioArray,
    adapter: str,
    min_overlap: int = 3,
    error_rate: float = 0.1,
) -> BioArray:
    """Trim adapter sequences from reads.

    Args:
        sequences: Sequences to trim.
        adapter: Adapter sequence to remove.
        min_overlap: Minimum overlap for adapter matching.
        error_rate: Maximum error rate for matching.

    Returns:
        Trimmed sequences.

    """
    adapter = adapter.upper()
    trimmed_seqs = []
    trimmed_ids = []

    for i, seq in enumerate(sequences.sequences):
        seq_upper = seq.upper()
        trim_pos = len(seq)

        for pos in range(len(seq) - min_overlap + 1):
            remaining = seq_upper[pos:]
            overlap = min(len(adapter), len(remaining))

            if overlap >= min_overlap:
                mismatches = sum(a != b for a, b in zip(adapter[:overlap], remaining[:overlap]))
                if mismatches / overlap <= error_rate:
                    trim_pos = pos
                    break

        trimmed_seqs.append(seq[:trim_pos])
        trimmed_ids.append(sequences.ids[i])

    return type(sequences)(trimmed_seqs, ids=trimmed_ids)
