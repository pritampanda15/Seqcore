#!/usr/bin/env python3
"""
Seqcore Usage Examples - Real-World Scenarios
Author: Dr. Pritam Kumar Panda @ Stanford University

Run these examples to see seqcore in action with real biological data.
"""

import seqcore as sc
import numpy as np


def example_1_sequence_analysis():
    """Example 1: Analyze DNA sequences from FASTA file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: DNA Sequence Analysis")
    print("=" * 60)

    # Load sequences
    sequences = sc.read("tests/data/test_sequences.fasta")
    print(f"Loaded {len(sequences)} sequences")

    # Calculate GC content
    gc = sc.gc_content(sequences)
    for seq_id, gc_val in zip(sequences.ids, gc):
        print(f"  {seq_id}: GC content = {gc_val:.1f}%")

    # Translate to protein
    proteins = sc.translate(sequences)
    print(f"\nTranslated {len(proteins)} proteins")
    for i, prot in enumerate(proteins.sequences[:2]):
        print(f"  {sequences.ids[i]}: {prot[:30]}...")

    # Find ATG start codons
    matches = sc.find_pattern(sequences, "ATG")
    print(f"\nATG start codon positions:")
    for seq_id, positions in zip(sequences.ids, matches):
        print(f"  {seq_id}: {len(positions)} occurrences")

    return sequences


def example_2_protein_structure():
    """Example 2: Analyze protein structure from PDB file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Protein Structure Analysis")
    print("=" * 60)

    # Load structure
    structure = sc.read("tests/data/test_protein.pdb")
    print(f"Loaded structure with {len(structure.atoms)} atoms")
    print(f"Chains: {set(structure.chains)}")

    # Calculate distance matrix for CA atoms
    dm = sc.distance_matrix(structure, selection="CA")
    print(f"CA distance matrix shape: {dm.shape}")
    print(f"Min/Max CA distances: {dm[dm > 0].min():.2f} / {dm.max():.2f} Å")

    # Find contacts
    contacts = sc.find_contacts(structure, cutoff=4.0)
    print(f"Found {len(contacts)} contacts within 4.0 Å")

    # Select chain A
    chain_a = structure.select(chain="A")
    print(f"Chain A: {len(chain_a.atoms)} atoms")

    return structure


def example_3_variant_analysis():
    """Example 3: Analyze genetic variants from VCF file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Genetic Variant Analysis")
    print("=" * 60)

    # Load variants
    variants = sc.read("tests/data/test_variants.vcf")
    print(f"Loaded {len(variants['pos'])} variants")
    print(f"Samples: {variants['sample_names']}")

    # Calculate allele frequencies
    af = sc.allele_frequency(variants)
    print(f"\nAllele frequencies:")
    for i in range(min(5, len(af))):
        print(f"  {variants['chrom'][i]}:{variants['pos'][i]} "
              f"{variants['ref'][i]}>{variants['alt'][i]}: AF={af[i]:.3f}")

    # Calculate heterozygosity
    het = sc.heterozygosity(variants)
    print(f"\nHeterozygosity range: {min(het):.3f} - {max(het):.3f}")

    # Linkage disequilibrium matrix
    ld = sc.linkage_disequilibrium(variants)
    print(f"LD matrix shape: {ld.shape}")

    return variants


def example_4_molecule_analysis():
    """Example 4: Analyze small molecules from SDF file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Small Molecule Analysis")
    print("=" * 60)

    # Load molecules
    molecules = sc.read("tests/data/test_molecules.sdf")
    print(f"Loaded {len(molecules)} molecules")

    for mol in molecules:
        print(f"  {mol.name}: {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")

    # Generate fingerprints
    fps = sc.morgan_fingerprint(molecules)
    print(f"\nFingerprint matrix shape: {fps.shape}")

    # Calculate similarity matrix
    sim = sc.tanimoto_similarity(fps)
    print("Tanimoto similarity matrix:")
    for i, mol1 in enumerate(molecules):
        for j, mol2 in enumerate(molecules):
            if i <= j:
                print(f"  {mol1.name} vs {mol2.name}: {sim[i, j]:.3f}")

    return molecules


def example_5_phylogenetics():
    """Example 5: Build phylogenetic tree."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Phylogenetic Tree Construction")
    print("=" * 60)

    # Load sequences
    sequences = sc.read("tests/data/test_sequences.fasta")

    # Calculate pairwise distances
    dm = sc.pairwise_distance(sequences, metric="edit")
    print(f"Distance matrix shape: {dm.shape}")

    # Build Neighbor-Joining tree
    tree = sc.neighbor_joining(sequences)
    print(f"NJ tree leaves: {tree.get_leaves()}")
    print(f"Newick: {tree.newick()[:80]}...")

    # Build UPGMA tree
    upgma_tree = sc.upgma(sequences)
    print(f"UPGMA tree leaves: {upgma_tree.get_leaves()}")

    return tree


def example_6_sequencing_qc():
    """Example 6: Quality control for sequencing reads."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Sequencing QC")
    print("=" * 60)

    # Load FASTQ reads
    reads = sc.read("tests/data/test_reads.fastq")
    print(f"Loaded {len(reads)} reads")

    # Quality statistics
    mean_quals = [np.mean(q[q > 0]) for q in reads.qualities]
    print(f"Mean quality scores: {[f'{q:.1f}' for q in mean_quals]}")

    # Quality filtering
    passed = sc.quality_filter(reads, min_q=20, max_n=0.1)
    print(f"Reads passing QC (Q>=20, N<10%): {sum(passed)}/{len(passed)}")

    return reads


def example_7_streaming_large_files():
    """Example 7: Stream processing for large files."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Streaming Large Files")
    print("=" * 60)

    # Simulate streaming (using small test file)
    total_seqs = 0
    total_gc = 0

    print("Processing file in batches...")
    for batch in sc.read_stream("tests/data/test_sequences.fasta", batch_size=2):
        batch_gc = np.mean(sc.gc_content(batch))
        total_seqs += len(batch)
        total_gc += batch_gc * len(batch)
        print(f"  Batch: {len(batch)} sequences, mean GC: {batch_gc:.1f}%")

    print(f"Total: {total_seqs} sequences, overall mean GC: {total_gc/total_seqs:.1f}%")


def example_8_alignment_formats():
    """Example 8: Working with different alignment formats."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Multiple Alignment Formats")
    print("=" * 60)

    # Stockholm format (common for protein families)
    sto = sc.read("tests/data/test_alignment.sto")
    print(f"Stockholm: {len(sto)} alignment(s)")
    if isinstance(sto, list):
        for aln in sto:
            print(f"  Sequences: {len(aln.get('sequences', {}))}")

    # SAM format (read alignments)
    sam = sc.read("tests/data/test_alignment.sam")
    print(f"SAM: {len(sam['reads'])} aligned reads")
    print(f"  References: {list(sam['references'].keys())}")

    # Newick trees
    trees = sc.read("tests/data/test_tree.nwk")
    print(f"Newick: {len(trees)} tree(s)")


def example_9_annotation_formats():
    """Example 9: Working with annotation files."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Genomic Annotations")
    print("=" * 60)

    # GFF3 annotations
    gff = sc.read("tests/data/test_annotation.gff3")
    print(f"GFF3: {len(gff['seqid'])} features")
    feature_types = set(gff["type"])
    print(f"  Feature types: {feature_types}")

    # BED regions
    bed = sc.read("tests/data/test_regions.bed")
    print(f"BED: {len(bed['chrom'])} regions")
    for i in range(min(3, len(bed["chrom"]))):
        print(f"  {bed['chrom'][i]}:{bed['start'][i]}-{bed['end'][i]} ({bed['name'][i]})")

    # GenBank
    gb = sc.read("tests/data/test_sequence.gb")
    print(f"GenBank: {len(gb)} records")
    for rec in gb:
        print(f"  {rec['locus']}: {len(rec['sequence'])} bp")


def example_10_population_genetics():
    """Example 10: Population genetics analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Population Genetics")
    print("=" * 60)

    # Nucleotide diversity
    seqs = ["ACGTACGT", "ACGTACGT", "ACGTATGT", "ACGTACGT"]
    pi = sc.nucleotide_diversity(seqs)
    print(f"Nucleotide diversity (π): {pi:.4f}")

    # Tajima's D
    d = sc.tajimas_d(seqs)
    print(f"Tajima's D: {d:.4f}")

    # Using VCF data
    variants = sc.read("tests/data/test_variants.vcf")
    af = sc.allele_frequency(variants)
    print(f"\nVariant allele frequencies: {[f'{f:.3f}' for f in af[:5]]}")


def main():
    """Run all examples."""
    print("\n" + "#" * 60)
    print("# SEQCORE - Biological Sequence Analysis Library")
    print("# Author: Dr. Pritam Kumar Panda @ Stanford University")
    print("#" * 60)

    example_1_sequence_analysis()
    example_2_protein_structure()
    example_3_variant_analysis()
    example_4_molecule_analysis()
    example_5_phylogenetics()
    example_6_sequencing_qc()
    example_7_streaming_large_files()
    example_8_alignment_formats()
    example_9_annotation_formats()
    example_10_population_genetics()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
