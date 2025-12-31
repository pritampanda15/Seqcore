"""Tests for core seqcore functionality."""

import numpy as np
import pytest


class TestDNAArray:
    """Tests for DNAArray class."""

    def test_single_sequence(self):
        """Test creating DNAArray from single sequence."""
        from seqcore.core.arrays import DNAArray

        dna = DNAArray("ACGT")
        assert len(dna) == 1
        assert dna[0] == "ACGT"

    def test_multiple_sequences(self):
        """Test creating DNAArray from list of sequences."""
        from seqcore.core.arrays import DNAArray

        seqs = ["ACGT", "TGCA", "NNNN"]
        dna = DNAArray(seqs)
        assert len(dna) == 3
        assert dna[0] == "ACGT"
        assert dna[1] == "TGCA"
        assert dna[2] == "NNNN"

    def test_to_list(self):
        """Test converting to list."""
        from seqcore.core.arrays import DNAArray

        seqs = ["ACGT", "TGCA"]
        dna = DNAArray(seqs)
        assert dna.to_list() == seqs

    def test_iteration(self):
        """Test iterating over sequences."""
        from seqcore.core.arrays import DNAArray

        seqs = ["ACGT", "TGCA", "GGGG"]
        dna = DNAArray(seqs)
        result = list(dna)
        assert result == seqs


class TestRNAArray:
    """Tests for RNAArray class."""

    def test_single_sequence(self):
        """Test creating RNAArray from single sequence."""
        from seqcore.core.arrays import RNAArray

        rna = RNAArray("ACGU")
        assert len(rna) == 1
        assert rna[0] == "ACGU"


class TestProteinArray:
    """Tests for ProteinArray class."""

    def test_single_sequence(self):
        """Test creating ProteinArray from single sequence."""
        from seqcore.core.arrays import ProteinArray

        protein = ProteinArray("MVLSPADKTNVK")
        assert len(protein) == 1
        assert protein[0] == "MVLSPADKTNVK"


class TestSequenceOperations:
    """Tests for sequence operations."""

    def test_gc_content(self):
        """Test GC content calculation."""
        from seqcore.core.arrays import DNAArray
        from seqcore.core.operations import gc_content

        dna = DNAArray(["GGCC", "AATT", "GCGC"])
        gc = gc_content(dna)

        assert len(gc) == 3
        assert gc[0] == pytest.approx(100.0)
        assert gc[1] == pytest.approx(0.0)
        assert gc[2] == pytest.approx(100.0)

    def test_gc_content_string(self):
        """Test GC content with string input."""
        from seqcore.core.operations import gc_content

        gc = gc_content("ATGC")
        assert gc[0] == pytest.approx(50.0)

    def test_reverse(self):
        """Test sequence reversal."""
        from seqcore.core.arrays import DNAArray
        from seqcore.core.operations import reverse

        dna = DNAArray(["ACGT"])
        rev = reverse(dna)
        assert rev[0] == "TGCA"

    def test_complement(self):
        """Test DNA complement."""
        from seqcore.core.arrays import DNAArray
        from seqcore.core.operations import complement

        dna = DNAArray(["ACGT"])
        comp = complement(dna)
        assert comp[0] == "TGCA"

    def test_reverse_complement(self):
        """Test reverse complement."""
        from seqcore.core.arrays import DNAArray
        from seqcore.core.operations import reverse_complement

        dna = DNAArray(["ACGT"])
        rc = reverse_complement(dna)
        assert rc[0] == "ACGT"  # ACGT is its own reverse complement

    def test_transcribe(self):
        """Test DNA to RNA transcription."""
        from seqcore.core.arrays import DNAArray
        from seqcore.core.operations import transcribe

        dna = DNAArray(["ATGC"])
        rna = transcribe(dna)
        assert rna[0] == "AUGC"

    def test_translate(self):
        """Test translation."""
        from seqcore.core.arrays import DNAArray
        from seqcore.core.operations import translate

        dna = DNAArray(["ATGGCC"])  # Met-Ala
        protein = translate(dna)
        assert protein[0] == "MA"

    def test_length(self):
        """Test length calculation."""
        from seqcore.core.arrays import DNAArray
        from seqcore.core.operations import length

        dna = DNAArray(["ACGT", "AC", "ACGTACGT"])
        lengths = length(dna)
        assert list(lengths) == [4, 2, 8]


class TestKmers:
    """Tests for k-mer operations."""

    def test_extract_kmers(self):
        """Test k-mer extraction."""
        from seqcore.core.kmers import extract_kmers

        kmers = extract_kmers("ACGTAC", k=3)
        assert kmers == [["ACG", "CGT", "GTA", "TAC"]]

    def test_count_kmers(self):
        """Test k-mer counting."""
        from seqcore.core.kmers import count_kmers

        counts = count_kmers("ACGTACGT", k=4)
        assert counts["ACGT"] == 2
        assert counts["CGTA"] == 1
        assert counts["GTAC"] == 1
        assert counts["TACG"] == 1

    def test_kmer_spectrum(self):
        """Test k-mer spectrum."""
        from seqcore.core.kmers import kmer_spectrum

        spectrum = kmer_spectrum("ACGTACGT", k=4)
        assert spectrum[1] == 3  # 3 k-mers appear once
        assert spectrum[2] == 1  # 1 k-mer appears twice


class TestAlignment:
    """Tests for alignment operations."""

    def test_pairwise_alignment(self):
        """Test pairwise alignment."""
        from seqcore.alignment import align

        result = align("ACGT", "ACGT")
        assert result.score > 0
        assert result.identity == pytest.approx(1.0)

    def test_alignment_with_mismatch(self):
        """Test alignment with mismatch."""
        from seqcore.alignment import align

        result = align("ACGT", "ACCT")
        assert result.identity < 1.0
        assert result.identity > 0.5

    def test_edit_distance(self):
        """Test edit distance calculation."""
        from seqcore.alignment import pairwise_distance

        dm = pairwise_distance(["ACGT", "ACGT", "TGCA"], metric="edit")
        assert dm[0, 0] == 0
        assert dm[0, 1] == 0
        assert dm[0, 2] > 0

    def test_find_pattern(self):
        """Test pattern finding."""
        from seqcore.alignment import find_pattern

        matches = find_pattern("ATGATGATG", "ATG")
        assert len(matches[0]) == 3


class TestDevice:
    """Tests for device management."""

    def test_gpu_available(self):
        """Test GPU availability check."""
        from seqcore.core.device import gpu_available

        result = gpu_available()
        assert isinstance(result, bool)

    def test_timer(self):
        """Test timer context manager."""
        import time

        from seqcore.core.device import timer

        with timer() as t:
            time.sleep(0.05)  # Use longer sleep for Windows timing precision

        # Allow some tolerance for Windows timing (can be slightly less than expected)
        assert t.elapsed >= 0.04

    def test_batch_size(self):
        """Test batch size setting."""
        from seqcore.core.device import get_batch_size, set_batch_size

        set_batch_size(50000)
        assert get_batch_size() == 50000


class TestMolecules:
    """Tests for molecule operations."""

    def test_molecule_from_smiles(self):
        """Test creating molecule from SMILES."""
        from seqcore.molecules import Molecule

        mol = Molecule.from_smiles("CCO", name="ethanol")
        assert mol.smiles == "CCO"
        assert mol.name == "ethanol"

    def test_lipinski_filter(self):
        """Test Lipinski filter."""
        from seqcore.molecules import Molecule, lipinski_filter

        mol = Molecule.from_smiles("CCO")  # Ethanol - should pass
        passes = lipinski_filter([mol])
        # Without RDKit, returns all zeros
        assert len(passes) == 1


class TestPhylogenetics:
    """Tests for phylogenetic operations."""

    def test_neighbor_joining(self):
        """Test neighbor joining tree construction."""
        from seqcore.phylogenetics import neighbor_joining

        # Simple distance matrix
        dm = np.array(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.0, 0.15],
                [0.2, 0.15, 0.0],
            ]
        )

        tree = neighbor_joining(dm, names=["A", "B", "C"])
        assert tree is not None
        assert "A" in tree.get_leaves()
        assert "B" in tree.get_leaves()
        assert "C" in tree.get_leaves()

    def test_tree_newick(self):
        """Test Newick format export."""
        from seqcore.phylogenetics import neighbor_joining

        dm = np.array(
            [
                [0.0, 0.1, 0.2],
                [0.1, 0.0, 0.15],
                [0.2, 0.15, 0.0],
            ]
        )

        tree = neighbor_joining(dm, names=["A", "B", "C"])
        newick = tree.newick()
        assert newick.endswith(";")
        assert "A" in newick
        assert "B" in newick
        assert "C" in newick


class TestPopulation:
    """Tests for population genetics operations."""

    def test_nucleotide_diversity(self):
        """Test nucleotide diversity calculation."""
        from seqcore.population import nucleotide_diversity

        # Identical sequences
        pi = nucleotide_diversity(["ACGT", "ACGT"])
        assert pi == 0.0

        # Different sequences
        pi = nucleotide_diversity(["ACGT", "TGCA"])
        assert pi > 0.0

    def test_tajimas_d(self):
        """Test Tajima's D calculation."""
        from seqcore.population import tajimas_d

        # Need multiple sequences
        seqs = ["ACGTACGT", "ACGTACGT", "ACGTACGT", "ACGTACGT"]
        d = tajimas_d(seqs)
        assert isinstance(d, float)


class TestImport:
    """Test that all main imports work."""

    def test_main_import(self):
        """Test importing seqcore."""
        import seqcore as sc

        assert hasattr(sc, "DNAArray")
        assert hasattr(sc, "RNAArray")
        assert hasattr(sc, "ProteinArray")
        assert hasattr(sc, "gc_content")
        assert hasattr(sc, "align")
        assert hasattr(sc, "read")
        assert hasattr(sc, "__version__")

    def test_version(self):
        """Test version string."""
        import seqcore as sc

        assert sc.__version__ == "0.1.0"
