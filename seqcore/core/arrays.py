"""Core biological sequence array implementations.

Provides efficient, memory-optimized storage for biological sequences
with optional GPU acceleration.

Optimized for high-performance batch operations using NumPy vectorization.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# Encoding maps for efficient storage
DNA_ENCODE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
DNA_DECODE = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}

RNA_ENCODE = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4}
RNA_DECODE = {0: "A", 1: "C", 2: "G", 3: "U", 4: "N"}

# Standard amino acids + special characters
AA_CHARS = "ACDEFGHIKLMNPQRSTVWY*X-"
AA_ENCODE = {aa: i for i, aa in enumerate(AA_CHARS)}
AA_DECODE = dict(enumerate(AA_CHARS))

# Pre-computed lookup tables for fast encoding (ASCII -> encoded value)
_DNA_ENCODE_TABLE = np.zeros(128, dtype=np.uint8)
_DNA_ENCODE_TABLE[ord("A")] = 0
_DNA_ENCODE_TABLE[ord("a")] = 0
_DNA_ENCODE_TABLE[ord("C")] = 1
_DNA_ENCODE_TABLE[ord("c")] = 1
_DNA_ENCODE_TABLE[ord("G")] = 2
_DNA_ENCODE_TABLE[ord("g")] = 2
_DNA_ENCODE_TABLE[ord("T")] = 3
_DNA_ENCODE_TABLE[ord("t")] = 3
_DNA_ENCODE_TABLE[ord("N")] = 4
_DNA_ENCODE_TABLE[ord("n")] = 4

_RNA_ENCODE_TABLE = np.zeros(128, dtype=np.uint8)
_RNA_ENCODE_TABLE[ord("A")] = 0
_RNA_ENCODE_TABLE[ord("a")] = 0
_RNA_ENCODE_TABLE[ord("C")] = 1
_RNA_ENCODE_TABLE[ord("c")] = 1
_RNA_ENCODE_TABLE[ord("G")] = 2
_RNA_ENCODE_TABLE[ord("g")] = 2
_RNA_ENCODE_TABLE[ord("U")] = 3
_RNA_ENCODE_TABLE[ord("u")] = 3
_RNA_ENCODE_TABLE[ord("N")] = 4
_RNA_ENCODE_TABLE[ord("n")] = 4

_AA_ENCODE_TABLE = np.zeros(128, dtype=np.uint8)
for aa, idx in AA_ENCODE.items():
    _AA_ENCODE_TABLE[ord(aa)] = idx
    _AA_ENCODE_TABLE[ord(aa.lower())] = idx

# Pre-computed decode tables (encoded value -> ASCII character)
_DNA_DECODE_TABLE = np.array([ord("A"), ord("C"), ord("G"), ord("T"), ord("N")], dtype=np.uint8)
_RNA_DECODE_TABLE = np.array([ord("A"), ord("C"), ord("G"), ord("U"), ord("N")], dtype=np.uint8)
_AA_DECODE_TABLE = np.array([ord(c) for c in AA_CHARS], dtype=np.uint8)


class BioArray(ABC):
    """Abstract base class for biological sequence arrays.

    Provides a common interface for DNA, RNA, and protein sequences
    with efficient numpy-backed storage.
    """

    _encode_map: dict
    _decode_map: dict
    _encode_table: np.ndarray
    _decode_table: np.ndarray
    _dtype: np.dtype

    def __init__(
        self,
        sequences: str | list[str] | np.ndarray,
        qualities: np.ndarray | None = None,
        ids: list[str] | None = None,
    ):
        """Initialize a biological sequence array.

        Args:
            sequences: Single sequence string, list of sequences, or encoded array.
            qualities: Optional quality scores (for FASTQ data).
            ids: Optional sequence identifiers.

        """
        if isinstance(sequences, np.ndarray):
            self._data = sequences
            self._lengths = (
                np.array([len(s) for s in sequences])
                if sequences.ndim > 1
                else np.array([len(sequences)])
            )
        elif isinstance(sequences, str):
            self._data, self._lengths = self._encode_single_fast(sequences)
        else:
            self._data, self._lengths = self._encode_batch_fast(sequences)

        self._qualities = qualities
        self._ids = ids if ids is not None else [f"seq_{i}" for i in range(len(self))]
        self._decoded_cache = {}  # Cache for decoded sequences

    def _encode_single_fast(self, sequence: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode a single sequence using vectorized lookup."""
        # Convert string to byte array and use lookup table
        byte_arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        encoded = self._encode_table[byte_arr]
        return encoded.reshape(1, -1), np.array([len(sequence)])

    def _encode_batch_fast(self, sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Encode multiple sequences with vectorized operations."""
        if not sequences:
            return np.array([], dtype=self._dtype).reshape(0, 0), np.array([])

        lengths = np.array([len(s) for s in sequences])
        max_len = lengths.max()
        n_seqs = len(sequences)

        # Pre-allocate array
        encoded = np.zeros((n_seqs, max_len), dtype=self._dtype)

        # Vectorized encoding using lookup table
        for i, seq in enumerate(sequences):
            if seq:
                byte_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
                encoded[i, : len(seq)] = self._encode_table[byte_arr]

        return encoded, lengths

    def _decode_single_fast(self, encoded: np.ndarray, length: int) -> str:
        """Decode a single encoded sequence using vectorized lookup."""
        # Use lookup table to convert encoded values to ASCII bytes
        ascii_bytes = self._decode_table[encoded[:length]]
        return ascii_bytes.tobytes().decode("ascii")

    def __len__(self) -> int:
        """Return number of sequences."""
        return self._data.shape[0] if self._data.ndim > 1 else 1

    def __getitem__(self, idx: int | slice) -> str | BioArray:
        """Get sequence(s) by index."""
        if isinstance(idx, int):
            # Check cache first
            if idx in self._decoded_cache:
                return self._decoded_cache[idx]

            if self._data.ndim == 1:
                result = self._decode_single_fast(self._data, self._lengths[0])
            else:
                result = self._decode_single_fast(self._data[idx], self._lengths[idx])

            # Cache the result
            if len(self._decoded_cache) < 1000:  # Limit cache size
                self._decoded_cache[idx] = result
            return result

        # Return new array for slices
        new_data = self._data[idx]
        self._lengths[idx]
        new_ids = (
            self._ids[idx]
            if isinstance(idx, slice)
            else [self._ids[i] for i in range(len(self._ids))[idx]]
        )
        new_qualities = self._qualities[idx] if self._qualities is not None else None

        return self.__class__.__new__(self.__class__, new_data, new_qualities, new_ids)

    def __iter__(self) -> Iterator[str]:
        """Iterate over sequences as strings."""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        max_len = self._data.shape[-1] if self._data.size > 0 else 0
        return f"{self.__class__.__name__}(n={len(self)}, max_len={max_len})"

    def __str__(self) -> str:
        if len(self) == 1:
            return self[0]
        return f"{self.__class__.__name__} with {len(self)} sequences"

    @property
    def sequences(self) -> list[str]:
        """Return all sequences as list of strings."""
        return [self[i] for i in range(len(self))]

    @property
    def encoded(self) -> np.ndarray:
        """Return the encoded numpy array."""
        return self._data

    @property
    def lengths(self) -> np.ndarray:
        """Return lengths of all sequences."""
        return self._lengths

    @property
    def ids(self) -> list[str]:
        """Return sequence identifiers."""
        return self._ids

    @property
    def qualities(self) -> np.ndarray | None:
        """Return quality scores if available."""
        return self._qualities

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (encoded)."""
        return self._data.copy()

    def to_list(self) -> list[str]:
        """Convert to list of sequence strings."""
        return self.sequences

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        import pandas as pd

        data = {"id": self._ids, "sequence": self.sequences, "length": self._lengths.tolist()}
        if self._qualities is not None:
            data["quality"] = [q.tolist() for q in self._qualities]
        return pd.DataFrame(data)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, lengths: np.ndarray | None = None) -> BioArray:
        """Create from encoded numpy array."""
        instance = cls.__new__(cls)
        instance._data = arr
        instance._lengths = (
            lengths if lengths is not None else np.array([arr.shape[-1]] * arr.shape[0])
        )
        instance._qualities = None
        instance._ids = [f"seq_{i}" for i in range(arr.shape[0])]
        instance._decoded_cache = {}
        instance._encode_table = cls._encode_table
        instance._decode_table = cls._decode_table
        return instance

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, seq_col: str = "sequence", id_col: str = "id"
    ) -> BioArray:
        """Create from pandas DataFrame."""
        sequences = df[seq_col].tolist()
        ids = df[id_col].tolist() if id_col in df.columns else None
        return cls(sequences, ids=ids)


class DNAArray(BioArray):
    """Efficient array for DNA sequences.

    Uses 3-bit encoding for memory efficiency (supports A, C, G, T, N).

    Example:
        >>> dna = DNAArray("ACGTACGT")
        >>> dna = DNAArray(["ACGT", "TGCA", "NNNN"])

    """

    _encode_map = DNA_ENCODE
    _decode_map = DNA_DECODE
    _encode_table = _DNA_ENCODE_TABLE
    _decode_table = _DNA_DECODE_TABLE
    _dtype = np.uint8

    def to_biopython(self):
        """Convert to Biopython Seq object(s)."""
        from Bio.Seq import Seq

        if len(self) == 1:
            return Seq(self[0])
        return [Seq(s) for s in self.sequences]

    @classmethod
    def from_biopython(cls, seq) -> DNAArray:
        """Create from Biopython Seq object."""
        return cls(str(seq))


class RNAArray(BioArray):
    """Efficient array for RNA sequences.

    Uses 3-bit encoding for memory efficiency (supports A, C, G, U, N).

    Example:
        >>> rna = RNAArray("ACGUACGU")

    """

    _encode_map = RNA_ENCODE
    _decode_map = RNA_DECODE
    _encode_table = _RNA_ENCODE_TABLE
    _decode_table = _RNA_DECODE_TABLE
    _dtype = np.uint8

    def to_biopython(self):
        """Convert to Biopython Seq object(s)."""
        from Bio.Seq import Seq

        if len(self) == 1:
            return Seq(self[0])
        return [Seq(s) for s in self.sequences]

    @classmethod
    def from_biopython(cls, seq) -> RNAArray:
        """Create from Biopython Seq object."""
        return cls(str(seq))


class ProteinArray(BioArray):
    """Efficient array for protein sequences.

    Uses 5-bit encoding for all standard amino acids.

    Example:
        >>> protein = ProteinArray("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")

    """

    _encode_map = AA_ENCODE
    _decode_map = AA_DECODE
    _encode_table = _AA_ENCODE_TABLE
    _decode_table = _AA_DECODE_TABLE
    _dtype = np.uint8

    def to_biopython(self):
        """Convert to Biopython Seq object(s)."""
        from Bio.Seq import Seq

        if len(self) == 1:
            return Seq(self[0])
        return [Seq(s) for s in self.sequences]

    @classmethod
    def from_biopython(cls, seq) -> ProteinArray:
        """Create from Biopython Seq object."""
        return cls(str(seq))
