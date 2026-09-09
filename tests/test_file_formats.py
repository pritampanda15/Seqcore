"""Comprehensive tests for all supported file formats.

Author: Dr. Pritam Kumar Panda @ Stanford University
"""

import os

import numpy as np
import pytest

# Get the test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestSequenceFormats:
    """Tests for sequence file formats."""

    def test_fasta_format(self):
        """Test FASTA format reading."""
        import seqcore as sc

        sequences = sc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        assert len(sequences) == 5
        assert "BRCA1_human" in sequences.ids

    def test_fastq_format(self):
        """Test FASTQ format reading."""
        import seqcore as sc

        reads = sc.read(os.path.join(TEST_DATA_DIR, "test_reads.fastq"))
        assert len(reads) == 5
        assert reads.qualities is not None

    def test_genbank_format(self):
        """Test GenBank format reading."""
        import seqcore as sc

        records = sc.read(os.path.join(TEST_DATA_DIR, "test_sequence.gb"))
        assert len(records) == 2
        assert records[0]["locus"] == "BRCA1_HUMAN"
        assert len(records[0]["sequence"]) == 1863

    def test_embl_format(self):
        """Test EMBL format reading."""
        import seqcore as sc

        records = sc.read(os.path.join(TEST_DATA_DIR, "test_sequence.embl"))
        assert len(records) == 2
        assert records[0]["id"] == "BRCA1_TEST"
        assert len(records[0]["sequence"]) == 500


class TestStructureFormats:
    """Tests for structure file formats."""

    def test_pdb_format(self):
        """Test PDB format reading."""
        import seqcore as sc

        structure = sc.read(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))
        assert len(structure.atoms) > 0
        assert len(structure.chains) == 2

    def test_mmcif_format(self):
        """Test mmCIF format reading."""
        import seqcore as sc

        structure = sc.read(os.path.join(TEST_DATA_DIR, "test_structure.cif"))
        assert len(structure.atoms) == 34


class TestMoleculeFormats:
    """Tests for molecule file formats."""

    def test_sdf_format(self):
        """Test SDF format reading."""
        import seqcore as sc

        molecules = sc.read(os.path.join(TEST_DATA_DIR, "test_molecules.sdf"))
        assert len(molecules) == 3
        assert molecules[0].name == "Aspirin"

    def test_mol2_format(self):
        """Test MOL2 format reading."""
        import seqcore as sc

        molecules = sc.read(os.path.join(TEST_DATA_DIR, "test_molecule.mol2"))
        assert len(molecules) == 2
        assert molecules[0].name == "Aspirin"
        assert len(molecules[0].atoms) == 13
        assert len(molecules[0].bonds) == 13


class TestAnnotationFormats:
    """Tests for annotation file formats."""

    def test_vcf_format(self):
        """Test VCF format reading."""
        import seqcore as sc

        variants = sc.read(os.path.join(TEST_DATA_DIR, "test_variants.vcf"))
        assert len(variants["chrom"]) == 10
        assert len(variants["sample_names"]) == 3

    def test_gff3_format(self):
        """Test GFF3 format reading."""
        import seqcore as sc

        annotations = sc.read(os.path.join(TEST_DATA_DIR, "test_annotation.gff3"))
        assert len(annotations["seqid"]) > 0
        assert "gene" in annotations["type"]

    def test_bed_format(self):
        """Test BED format reading."""
        import seqcore as sc

        regions = sc.read(os.path.join(TEST_DATA_DIR, "test_regions.bed"))
        assert len(regions["chrom"]) == 7
        assert "chr1" in regions["chrom"]


class TestAlignmentFormats:
    """Tests for alignment file formats."""

    def test_sam_format(self):
        """Test SAM format reading."""
        import seqcore as sc

        alignments = sc.read(os.path.join(TEST_DATA_DIR, "test_alignment.sam"))
        assert len(alignments["reads"]) == 8
        assert len(alignments["header"]) == 5

    def test_stockholm_format(self):
        """Test Stockholm format reading."""
        import seqcore as sc

        alignments = sc.read(os.path.join(TEST_DATA_DIR, "test_alignment.sto"))
        assert len(alignments) == 2
        assert len(alignments[0]["sequences"]) == 5

    def test_phylip_format(self):
        """Test PHYLIP format reading."""
        import seqcore as sc

        alignment = sc.read(os.path.join(TEST_DATA_DIR, "test_alignment.phy"))
        assert alignment is not None


class TestTreeFormats:
    """Tests for tree file formats."""

    def test_newick_format(self):
        """Test Newick format reading."""
        import seqcore as sc

        trees = sc.read(os.path.join(TEST_DATA_DIR, "test_tree.nwk"))
        assert len(trees) == 3
        # All trees should be valid Newick strings
        for tree in trees:
            assert tree.startswith("(")
            assert tree.endswith(";")


class TestSingleCellFormats:
    """Tests for single-cell data formats."""

    def test_h5ad_format(self):
        """Test HDF5/AnnData format reading."""
        pytest.importorskip("h5py", reason="h5py required for HDF5 tests")
        import seqcore as sc

        data = sc.read(os.path.join(TEST_DATA_DIR, "test_single_cell.h5ad"))
        assert data["X"].shape == (100, 50)
        assert "cell_type" in data["obs"]
        assert "_index" in data["var"]

    def test_tenx_matrix_layout(self, tmp_path):
        """A 10x /matrix file reads back transposed to cells x genes."""
        pytest.importorskip("h5py", reason="h5py required for HDF5 tests")
        pytest.importorskip("scipy", reason="scipy required to assemble sparse X")
        import h5py
        import numpy as np
        from scipy.sparse import csc_matrix

        import seqcore as sc

        rng = np.random.default_rng(0)
        dense = (rng.random((5, 3)) < 0.6) * rng.random((5, 3)).astype(np.float32)
        genes_by_cells = csc_matrix(dense)  # 10x orients genes as rows

        path = tmp_path / "filtered_feature_bc_matrix.h5"
        with h5py.File(path, "w") as f:
            m = f.create_group("matrix")
            m.create_dataset("data", data=genes_by_cells.data)
            m.create_dataset("indices", data=genes_by_cells.indices)
            m.create_dataset("indptr", data=genes_by_cells.indptr)
            m.create_dataset("shape", data=np.array([5, 3]))
            m.create_dataset("barcodes", data=np.array([b"c1", b"c2", b"c3"]))
            feat = m.create_group("features")
            feat.create_dataset("name", data=np.array([f"G{i}".encode() for i in range(5)]))
            feat.create_dataset("id", data=np.array([f"ENSG{i}".encode() for i in range(5)]))

        data = sc.read(str(path))

        # Transposed to the AnnData convention: rows are cells.
        assert data["X"].shape == (3, 5)
        assert (data["n_obs"], data["n_vars"]) == (3, 5)
        np.testing.assert_allclose(data["X"].toarray(), dense.T)

        # Byte strings are decoded.
        assert list(data["obs"]["_index"]) == ["c1", "c2", "c3"]
        assert list(data["var"]["_index"]) == ["G0", "G1", "G2", "G3", "G4"]
        assert list(data["var"]["id"])[0] == "ENSG0"

    def test_tenx_cellranger_2x_layout(self, tmp_path):
        """CellRanger 2.x named the feature datasets differently."""
        pytest.importorskip("h5py", reason="h5py required for HDF5 tests")
        pytest.importorskip("scipy", reason="scipy required to assemble sparse X")
        import h5py
        import numpy as np
        from scipy.sparse import csc_matrix

        import seqcore as sc

        m2 = csc_matrix(np.eye(3, dtype=np.float32))
        path = tmp_path / "old_cellranger.h5"
        with h5py.File(path, "w") as f:
            m = f.create_group("matrix")
            m.create_dataset("data", data=m2.data)
            m.create_dataset("indices", data=m2.indices)
            m.create_dataset("indptr", data=m2.indptr)
            m.create_dataset("shape", data=np.array([3, 3]))
            m.create_dataset("gene_names", data=np.array([b"A", b"B", b"C"]))
            m.create_dataset("genes", data=np.array([b"E1", b"E2", b"E3"]))

        data = sc.read(str(path))
        assert list(data["var"]["_index"]) == ["A", "B", "C"]
        assert list(data["var"]["id"]) == ["E1", "E2", "E3"]

    def test_sparse_anndata_x_is_assembled(self, tmp_path):
        """A CSR /X group is returned as a matrix, not loose components."""
        pytest.importorskip("h5py", reason="h5py required for HDF5 tests")
        pytest.importorskip("scipy", reason="scipy required to assemble sparse X")
        import h5py
        import numpy as np
        from scipy.sparse import csr_matrix

        import seqcore as sc

        dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
        sparse = csr_matrix(dense)

        path = tmp_path / "sparse.h5ad"
        with h5py.File(path, "w") as f:
            g = f.create_group("X")
            g.create_dataset("data", data=sparse.data)
            g.create_dataset("indices", data=sparse.indices)
            g.create_dataset("indptr", data=sparse.indptr)
            g.attrs["shape"] = np.array([2, 3])

        data = sc.read(str(path))
        np.testing.assert_allclose(data["X"].toarray(), dense)
        # The raw components stay available for callers that used them.
        assert "X_data" in data and "X_indptr" in data

    def test_unknown_hdf5_layout_raises(self, tmp_path):
        """An unrecognised layout must fail loudly, not return an empty matrix."""
        pytest.importorskip("h5py", reason="h5py required for HDF5 tests")
        import h5py
        import numpy as np

        import seqcore as sc

        path = tmp_path / "mystery.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("counts", data=np.zeros((4, 4)))

        with pytest.raises(ValueError, match="no '/X'.*no '/matrix'"):
            sc.read(str(path))


class TestFormatDetection:
    """Tests for automatic format detection."""

    @pytest.mark.parametrize(
        "filename,expected_format",
        [
            ("test.fasta", "fasta"),
            ("test.fa", "fasta"),
            ("test.fna", "fasta"),
            ("test.fastq", "fastq"),
            ("test.fq", "fastq"),
            ("test.pdb", "pdb"),
            ("test.cif", "mmcif"),
            ("test.sdf", "sdf"),
            ("test.mol2", "mol2"),
            ("test.vcf", "vcf"),
            ("test.gff", "gff"),
            ("test.gff3", "gff3"),
            ("test.bed", "bed"),
            ("test.gb", "genbank"),
            ("test.genbank", "genbank"),
            ("test.embl", "embl"),
            ("test.nwk", "newick"),
            ("test.newick", "newick"),
            ("test.sto", "stockholm"),
            ("test.stockholm", "stockholm"),
            ("test.phy", "phylip"),
            ("test.sam", "sam"),
            ("test.h5ad", "h5ad"),
            ("test.hdf5", "hdf5"),
        ],
    )
    def test_format_detection(self, filename, expected_format):
        """Test automatic format detection from file extension."""
        from seqcore.io import _detect_format

        detected = _detect_format(filename)
        assert detected == expected_format


class TestIOFunctions:
    """Tests for I/O module functions."""

    def test_read_function_auto_detect(self):
        """Test that read() auto-detects formats."""
        import seqcore as sc

        # FASTA
        seqs = sc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))
        assert len(seqs) > 0

        # PDB
        struct = sc.read(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))
        assert len(struct.atoms) > 0

        # VCF
        vcf = sc.read(os.path.join(TEST_DATA_DIR, "test_variants.vcf"))
        assert len(vcf["chrom"]) > 0

    def test_explicit_format_parameter(self):
        """Test explicit format parameter."""
        import seqcore as sc

        # Read FASTA with explicit format
        seqs = sc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"), format="fasta")
        assert len(seqs) > 0


class TestDataIntegrity:
    """Tests for data integrity across formats."""

    def test_sequence_data_preserved(self):
        """Test that sequence data is preserved correctly."""
        import seqcore as sc

        seqs = sc.read(os.path.join(TEST_DATA_DIR, "test_sequences.fasta"))

        # BRCA1 should start with ATG
        brca1_idx = seqs.ids.index("BRCA1_human")
        assert seqs.sequences[brca1_idx].startswith("ATG")

    def test_structure_coordinates_valid(self):
        """Test that structure coordinates are valid numbers."""
        import seqcore as sc

        struct = sc.read(os.path.join(TEST_DATA_DIR, "test_protein.pdb"))

        # Coordinates should be valid floats
        for coord in struct.coordinates:
            assert not np.isnan(coord).any()
            assert not np.isinf(coord).any()

    def test_variant_positions_valid(self):
        """Test that variant positions are valid."""
        import seqcore as sc

        vcf = sc.read(os.path.join(TEST_DATA_DIR, "test_variants.vcf"))

        # All positions should be positive integers
        for pos in vcf["pos"]:
            assert pos > 0
            assert isinstance(pos, int)

    def test_molecule_atoms_valid(self):
        """Test that molecule atoms are valid."""
        import seqcore as sc

        mols = sc.read(os.path.join(TEST_DATA_DIR, "test_molecules.sdf"))

        for mol in mols:
            # Each molecule should have atoms
            assert len(mol.atoms) > 0
            # Each atom should have valid coordinates (atoms can be dicts or objects)
            for atom in mol.atoms:
                if isinstance(atom, dict):
                    assert "x" in atom
                    assert "y" in atom
                    assert "z" in atom
                else:
                    assert hasattr(atom, "x")
                    assert hasattr(atom, "y")
                    assert hasattr(atom, "z")
