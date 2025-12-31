"""Comprehensive tests using real biological data files.

Author: Dr. Pritam Kumar Panda @ Stanford University
"""

import os

import numpy as np
import pytest

# Get the test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestFastaIO:
    """Tests for FASTA file I/O with real sequences."""

    def test_read_fasta(self):
        """Test reading real FASTA file."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        assert len(sequences) == 5
        assert "BRCA1_human" in sequences.ids
        assert "TP53_human" in sequences.ids

    def test_gc_content_real_sequences(self):
        """Test GC content on real gene sequences."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        gc = bc.gc_content(sequences)

        assert len(gc) == 5
        assert all(0 <= g <= 100 for g in gc)
        # EGFR should have high GC content
        egfr_idx = sequences.ids.index("EGFR_human")
        assert gc[egfr_idx] > 50

    def test_translation_real_sequences(self):
        """Test translation of real gene sequences."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        proteins = bc.translate(sequences)

        assert len(proteins) == 5
        # All sequences start with ATG, so proteins should start with M
        for prot in proteins.sequences:
            assert prot[0] == "M"

    def test_reverse_complement(self):
        """Test reverse complement of real sequences."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        rc = bc.reverse_complement(sequences)

        assert len(rc) == len(sequences)
        # Double reverse complement should give original
        rc2 = bc.reverse_complement(rc)
        for orig, double_rc in zip(sequences.sequences, rc2.sequences):
            assert orig == double_rc


class TestFastqIO:
    """Tests for FASTQ file I/O."""

    def test_read_fastq(self):
        """Test reading real FASTQ file."""
        import seqcore as bc

        reads = bc.read_fastq(os.path.join(TEST_DATA_DIR, "test_reads.fastq"))
        assert len(reads) == 5
        assert reads.qualities is not None

    def test_quality_scores(self):
        """Test quality score parsing."""
        import seqcore as bc

        reads = bc.read_fastq(os.path.join(TEST_DATA_DIR, "test_reads.fastq"))

        # Check that quality scores are in valid Phred range
        for i, q in enumerate(reads.qualities):
            valid_quals = q[: reads.lengths[i]]
            assert all(0 <= qv <= 41 for qv in valid_quals)

    def test_quality_filter(self):
        """Test quality filtering."""
        import seqcore as bc

        reads = bc.read_fastq(os.path.join(TEST_DATA_DIR, "test_reads.fastq"))
        passed = bc.quality_filter(reads, min_q=20, max_n=0.1)

        assert len(passed) == len(reads)
        assert passed.dtype == bool


class TestPDBStructure:
    """Tests for PDB structure operations."""

    def test_read_pdb(self):
        """Test reading real PDB file."""
        import seqcore as bc

        structure = bc.read_pdb(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))
        assert len(structure) > 0
        assert len(structure.chains) == 2  # A and B chains

    def test_distance_matrix(self):
        """Test CA distance matrix calculation."""
        import seqcore as bc

        structure = bc.read_pdb(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))
        dm = bc.distance_matrix(structure, selection="CA")

        # Distance matrix should be square and symmetric
        assert dm.shape[0] == dm.shape[1]
        assert np.allclose(dm, dm.T)
        # Diagonal should be zero
        assert np.allclose(np.diag(dm), 0)

    def test_find_contacts(self):
        """Test contact finding."""
        import seqcore as bc

        structure = bc.read_pdb(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))
        contacts = bc.find_contacts(structure, cutoff=4.0)

        assert isinstance(contacts, list)
        for contact in contacts:
            assert contact.distance <= 4.0

    def test_chain_selection(self):
        """Test chain selection."""
        import seqcore as bc

        structure = bc.read_pdb(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))
        chain_a = structure.select(chain="A")

        assert len(chain_a) < len(structure)
        assert all(c == "A" for c in chain_a._chain_ids)

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        pytest.importorskip("pandas", reason="pandas required for DataFrame tests")
        import seqcore as bc

        structure = bc.read_pdb(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))
        df = structure.to_dataframe()

        assert len(df) == len(structure)
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns


class TestVCFPopulation:
    """Tests for VCF and population genetics."""

    def test_read_vcf(self):
        """Test reading real VCF file."""
        from seqcore.io import read_vcf

        variants = read_vcf(os.path.join(TEST_DATA_DIR, "test_variants.vcf"))
        assert len(variants["pos"]) == 10
        assert len(variants["sample_names"]) == 3

    def test_allele_frequency(self):
        """Test allele frequency calculation."""
        import seqcore as bc
        from seqcore.io import read_vcf

        variants = read_vcf(os.path.join(TEST_DATA_DIR, "test_variants.vcf"))
        af = bc.allele_frequency(variants)

        assert len(af) == 10
        assert all(0 <= f <= 1 for f in af)

    def test_heterozygosity(self):
        """Test heterozygosity calculation."""
        import seqcore as bc
        from seqcore.io import read_vcf

        variants = read_vcf(os.path.join(TEST_DATA_DIR, "test_variants.vcf"))
        het = bc.heterozygosity(variants)

        assert len(het) == 10
        assert all(0 <= h <= 1 for h in het)

    def test_linkage_disequilibrium(self):
        """Test LD matrix calculation."""
        import seqcore as bc
        from seqcore.io import read_vcf

        variants = read_vcf(os.path.join(TEST_DATA_DIR, "test_variants.vcf"))
        ld = bc.linkage_disequilibrium(variants)

        assert ld.shape == (10, 10)
        # Diagonal should be 1
        assert np.allclose(np.diag(ld), 1.0)


class TestSDFMolecules:
    """Tests for SDF molecule operations."""

    def test_read_sdf(self):
        """Test reading real SDF file."""
        import seqcore as bc

        molecules = bc.read_sdf(os.path.join(TEST_DATA_DIR, "test_molecules.sdf"))
        assert len(molecules) == 3
        assert molecules[0].name == "Aspirin"

    def test_molecule_properties(self):
        """Test molecule has atoms and bonds."""
        import seqcore as bc

        molecules = bc.read_sdf(os.path.join(TEST_DATA_DIR, "test_molecules.sdf"))

        for mol in molecules:
            assert len(mol.atoms) > 0
            assert len(mol.bonds) > 0
            assert mol.coordinates is not None

    def test_fingerprints(self):
        """Test fingerprint generation."""
        import seqcore as bc

        aspirin = bc.Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        caffeine = bc.Molecule.from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

        fps = bc.morgan_fingerprint([aspirin, caffeine])
        assert fps.shape == (2, 2048)

    def test_tanimoto_similarity(self):
        """Test Tanimoto similarity."""
        import seqcore as bc

        # Use molecules from file which have actual atom data
        molecules = bc.read_sdf(os.path.join(TEST_DATA_DIR, "test_molecules.sdf"))

        fps = bc.morgan_fingerprint(molecules[:2])
        sim = bc.tanimoto_similarity(fps)

        assert sim.shape == (2, 2)
        # Similarity matrix should be symmetric
        assert np.allclose(sim, sim.T)
        # All similarities should be between 0 and 1
        assert np.all(sim >= 0) and np.all(sim <= 1)


class TestAlignment:
    """Tests for sequence alignment."""

    def test_pairwise_alignment(self):
        """Test pairwise alignment with real sequences."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        result = bc.align(sequences[0], sequences[1])

        assert result.score != 0
        assert 0 <= result.identity <= 1

    def test_distance_matrix(self):
        """Test pairwise distance matrix."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        dm = bc.pairwise_distance(sequences, metric="edit")

        assert dm.shape == (5, 5)
        assert np.allclose(dm, dm.T)  # Symmetric
        assert np.allclose(np.diag(dm), 0)  # Zero diagonal

    def test_pattern_finding(self):
        """Test pattern finding."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        matches = bc.find_pattern(sequences, "ATG")

        assert len(matches) == 5
        # All sequences should have at least one ATG (start codon)
        for m in matches:
            assert len(m) >= 1


class TestPhylogenetics:
    """Tests for phylogenetic tree construction."""

    def test_neighbor_joining(self):
        """Test NJ tree construction."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        tree = bc.neighbor_joining(sequences)

        assert len(tree.get_leaves()) == 5
        assert tree.newick().endswith(";")

    def test_upgma(self):
        """Test UPGMA tree construction."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        tree = bc.upgma(sequences)

        assert len(tree.get_leaves()) == 5
        assert tree.newick().endswith(";")

    def test_tree_distance(self):
        """Test patristic distance."""
        import seqcore as bc

        sequences = bc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        tree = bc.neighbor_joining(sequences)

        leaves = tree.get_leaves()
        dist = tree.distance(leaves[0], leaves[1])
        assert dist >= 0


class TestPopulationSequences:
    """Tests for population genetics with sequences."""

    def test_nucleotide_diversity(self):
        """Test nucleotide diversity calculation."""
        import seqcore as bc

        # Identical sequences should have pi = 0
        identical = ["ACGTACGT", "ACGTACGT", "ACGTACGT"]
        pi = bc.nucleotide_diversity(identical)
        assert pi == 0.0

        # Different sequences should have pi > 0
        different = ["ACGTACGT", "TGCATGCA"]
        pi = bc.nucleotide_diversity(different)
        assert pi > 0

    def test_tajimas_d(self):
        """Test Tajima's D calculation."""
        import seqcore as bc

        sequences = [
            "ACGTACGTACGTACGT",
            "ACGTACGTACGTACGT",
            "ACGTACGTACGTACGT",
            "ACGTACGTACGTACGT",
        ]
        d = bc.tajimas_d(sequences)
        assert isinstance(d, float)
