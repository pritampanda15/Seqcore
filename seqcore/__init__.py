"""Seqcore: High-performance biological sequence analysis library.

A unified, GPU-accelerated library for genomics, proteomics,
structural biology, and drug design.

Example:
    >>> import seqcore as sc
    >>> dna = sc.DNAArray("ACGTACGT")
    >>> gc = sc.gc_content(dna)

"""

__version__ = "0.1.0"
__author__ = "Dr. Pritam Kumar Panda @ Stanford University"

# Alignment operations
from seqcore.alignment import (
    align,
    find_motifs,
    find_pattern,
    pairwise_distance,
    position_weight_matrix,
)

# Core data structures
from seqcore.core.arrays import (
    BioArray,
    DNAArray,
    ProteinArray,
    RNAArray,
)
from seqcore.core.device import (
    clear_gpu_cache,
    device,
    gpu_available,
    gpu_info,
    set_batch_size,
    set_memory_limit,
    timer,
)

# K-mer operations
from seqcore.core.kmers import (
    count_kmers,
    extract_kmers,
    kmer_spectrum,
)

# Sequence operations
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

# I/O operations
from seqcore.io import (
    fetch,
    read,
    read_fasta,
    read_fastq,
    read_mmcif,
    read_pdb,
    read_sdf,
    read_stream,
    write,
)

# Molecule operations
from seqcore.molecules import (
    Molecule,
    bbb_filter,
    find_interactions,
    find_similar,
    generate_conformers,
    h_bond_acceptors,
    h_bond_donors,
    lipinski_filter,
    logp,
    maccs_fingerprint,
    minimize_energy,
    morgan_fingerprint,
    rdkit_fingerprint,
    rotatable_bonds,
    substructure_search,
    tanimoto_similarity,
    tpsa,
    veber_filter,
)

# Phylogenetics
from seqcore.phylogenetics import (
    neighbor_joining,
    place_sequence,
    upgma,
)

# Population genetics
from seqcore.population import (
    allele_frequency,
    fst,
    haplotype_frequency,
    heterozygosity,
    linkage_disequilibrium,
    minor_allele_frequency,
    nucleotide_diversity,
    phase_genotypes,
    r_squared,
    tajimas_d,
)

# Structure operations
from seqcore.structure import (
    compare_contacts,
    distance_matrix,
    druggability_score,
    find_contacts,
    find_neighbors,
    find_pockets,
    flexibility_profile,
    normalize_bfactors,
    rmsd,
    sasa,
    secondary_structure,
    surface_residues,
)

__all__ = [
    # Version
    "__version__",
    # Core arrays
    "DNAArray",
    "RNAArray",
    "ProteinArray",
    "BioArray",
    "StructureArray",
    # Device management
    "gpu_available",
    "gpu_info",
    "device",
    "clear_gpu_cache",
    "set_memory_limit",
    "set_batch_size",
    "timer",
    # Sequence operations
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
    # K-mers
    "extract_kmers",
    "count_kmers",
    "kmer_spectrum",
    # Alignment
    "align",
    "pairwise_distance",
    "find_pattern",
    "find_motifs",
    "position_weight_matrix",
    # I/O
    "read",
    "write",
    "read_fasta",
    "read_fastq",
    "read_pdb",
    "read_mmcif",
    "read_sdf",
    "read_stream",
    "fetch",
    # Structure
    "distance_matrix",
    "find_contacts",
    "find_neighbors",
    "rmsd",
    "secondary_structure",
    "sasa",
    "surface_residues",
    "normalize_bfactors",
    "flexibility_profile",
    "find_pockets",
    "druggability_score",
    "compare_contacts",
    # Molecules
    "Molecule",
    "logp",
    "h_bond_donors",
    "h_bond_acceptors",
    "tpsa",
    "rotatable_bonds",
    "lipinski_filter",
    "veber_filter",
    "bbb_filter",
    "morgan_fingerprint",
    "rdkit_fingerprint",
    "maccs_fingerprint",
    "tanimoto_similarity",
    "find_similar",
    "substructure_search",
    "generate_conformers",
    "minimize_energy",
    "find_interactions",
    # Phylogenetics
    "neighbor_joining",
    "upgma",
    "place_sequence",
    # Population genetics
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
