"""Core data structures and operations for seqcore."""

from seqcore.core.arrays import BioArray, DNAArray, ProteinArray, RNAArray
from seqcore.core.device import (
    clear_gpu_cache,
    device,
    gpu_available,
    gpu_info,
    set_batch_size,
    set_memory_limit,
    timer,
)
from seqcore.core.kmers import count_kmers, extract_kmers, kmer_spectrum
from seqcore.core.operations import (
    codon_adaptation_index,
    codon_frequency,
    complement,
    extract_codons,
    gc_content,
    length,
    molecular_weight,
    quality_filter,
    reverse,
    reverse_complement,
    transcribe,
    translate,
    trim_adapters,
)
from seqcore.core.structure import StructureArray

__all__ = [
    "DNAArray",
    "RNAArray",
    "ProteinArray",
    "BioArray",
    "StructureArray",
    "gpu_available",
    "gpu_info",
    "device",
    "clear_gpu_cache",
    "set_memory_limit",
    "set_batch_size",
    "timer",
    "gc_content",
    "length",
    "molecular_weight",
    "reverse",
    "complement",
    "reverse_complement",
    "transcribe",
    "translate",
    "extract_codons",
    "codon_frequency",
    "codon_adaptation_index",
    "quality_filter",
    "trim_adapters",
    "extract_kmers",
    "count_kmers",
    "kmer_spectrum",
]
