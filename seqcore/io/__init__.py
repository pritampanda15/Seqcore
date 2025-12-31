"""Universal I/O operations for biological data.

Provides smart format detection and streaming capabilities.
"""

from __future__ import annotations

import gzip
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib.parse import urlparse

if TYPE_CHECKING:
    from seqcore.core.arrays import BioArray, DNAArray, ProteinArray, RNAArray
    from seqcore.core.structure import StructureArray


def _safe_urlopen(url: str, allowed_schemes: tuple = ("https",)):
    """Safely open a URL after validating the scheme.

    This function validates that the URL uses an allowed scheme (default: https only)
    to prevent potential security issues with file:// or other schemes.
    """
    import urllib.request

    parsed = urlparse(url)
    if parsed.scheme not in allowed_schemes:
        raise ValueError(f"URL scheme '{parsed.scheme}' not allowed. Allowed: {allowed_schemes}")
    return urllib.request.urlopen(url)  # nosec B310 - scheme validated above


def _detect_format(filepath: str) -> str:
    """Detect file format from extension."""
    path_lower = filepath.lower()

    # Remove compression extension first
    if path_lower.endswith(".gz"):
        path_lower = path_lower[:-3]
    elif path_lower.endswith(".bz2"):
        path_lower = path_lower[:-4]

    if path_lower.endswith((".fa", ".fasta", ".fna", ".faa")):
        return "fasta"
    elif path_lower.endswith((".fq", ".fastq")):
        return "fastq"
    elif path_lower.endswith(".pdb"):
        return "pdb"
    elif path_lower.endswith((".cif", ".mmcif")):
        return "mmcif"
    elif path_lower.endswith(".sdf"):
        return "sdf"
    elif path_lower.endswith(".mol2"):
        return "mol2"
    elif path_lower.endswith(".gff3"):
        return "gff3"
    elif path_lower.endswith(".gff"):
        return "gff"
    elif path_lower.endswith(".gtf"):
        return "gtf"
    elif path_lower.endswith(".vcf"):
        return "vcf"
    elif path_lower.endswith(".bam"):
        return "bam"
    elif path_lower.endswith(".sam"):
        return "sam"
    elif path_lower.endswith(".bed"):
        return "bed"
    elif path_lower.endswith((".gb", ".genbank", ".gbk")):
        return "genbank"
    elif path_lower.endswith(".embl"):
        return "embl"
    elif path_lower.endswith((".nwk", ".newick", ".tree")):
        return "newick"
    elif path_lower.endswith((".sto", ".stockholm")):
        return "stockholm"
    elif path_lower.endswith((".phy", ".phylip")):
        return "phylip"
    elif path_lower.endswith((".h5", ".hdf5")):
        return "hdf5"
    elif path_lower.endswith(".h5ad"):
        return "h5ad"
    else:
        raise ValueError(f"Could not detect format for: {filepath}")


def _open_file(filepath: str, mode: str = "r"):
    """Open file, handling compression."""
    if filepath.endswith(".gz"):
        return gzip.open(filepath, mode + "t")
    elif filepath.endswith(".bz2"):
        import bz2

        return bz2.open(filepath, mode + "t")
    else:
        return open(filepath, mode)


def read(filepath: str, format: str | None = None) -> Any:
    """Read biological data from file with automatic format detection.

    Args:
        filepath: Path to input file.
        format: File format (auto-detected if not specified).

    Returns:
        Appropriate data structure based on file format.

    Example:
        >>> data = sc.read("sequences.fasta")
        >>> data = sc.read("structure.pdb")
        >>> data = sc.read("reads.fastq.gz")

    """
    if format is None:
        format = _detect_format(filepath)

    if format == "fasta":
        return read_fasta(filepath)
    elif format == "fastq":
        return read_fastq(filepath)
    elif format == "pdb":
        return read_pdb(filepath)
    elif format == "mmcif":
        return read_mmcif(filepath)
    elif format == "sdf":
        return read_sdf(filepath)
    elif format == "mol2":
        return read_mol2(filepath)
    elif format == "vcf":
        return read_vcf(filepath)
    elif format in ("gff", "gff3", "gtf"):
        return read_gff(filepath)
    elif format == "bed":
        return read_bed(filepath)
    elif format == "genbank":
        return read_genbank(filepath)
    elif format == "embl":
        return read_embl(filepath)
    elif format == "newick":
        return read_newick(filepath)
    elif format == "stockholm":
        return read_stockholm(filepath)
    elif format == "phylip":
        return read_phylip(filepath)
    elif format == "sam":
        return read_sam(filepath)
    elif format in ("hdf5", "h5ad"):
        return read_h5ad(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def read_fasta(filepath: str) -> DNAArray:
    """Read sequences from FASTA file.

    Args:
        filepath: Path to FASTA file.

    Returns:
        DNAArray (or ProteinArray if detected).

    Example:
        >>> sequences = sc.read_fasta("sequences.fa")

    """
    from seqcore.core.arrays import DNAArray, ProteinArray

    sequences = []
    ids = []
    current_seq = []
    current_id = None

    with _open_file(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append("".join(current_seq))
                current_id = line[1:].split()[0]
                ids.append(current_id)
                current_seq = []
            elif line:
                current_seq.append(line.upper())

        if current_id is not None:
            sequences.append("".join(current_seq))

    if not sequences:
        return DNAArray([])

    # Detect if protein based on amino acids not in DNA
    sample = sequences[0][:100] if sequences else ""
    protein_chars = set("DEFHIKLMNPQRSVWY")
    if any(c in protein_chars for c in sample.upper()):
        return ProteinArray(sequences, ids=ids)

    return DNAArray(sequences, ids=ids)


def read_fastq(
    filepath: str,
    quality_encoding: str = "phred33",
) -> DNAArray:
    """Read sequences from FASTQ file.

    Args:
        filepath: Path to FASTQ file.
        quality_encoding: Quality encoding ("phred33" or "phred64").

    Returns:
        DNAArray with quality scores.

    Example:
        >>> reads = sc.read_fastq("reads.fq.gz")

    """
    import numpy as np

    from seqcore.core.arrays import DNAArray

    offset = 33 if quality_encoding == "phred33" else 64

    sequences = []
    ids = []
    qualities = []

    with _open_file(filepath, "r") as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            if not header.startswith("@"):
                continue

            seq = f.readline().strip()
            f.readline()  # Plus line
            qual = f.readline().strip()

            ids.append(header[1:].split()[0])
            sequences.append(seq.upper())
            qualities.append(np.array([ord(c) - offset for c in qual], dtype=np.int8))

    if not sequences:
        return DNAArray([])

    # Pad quality arrays to same length
    max_len = max(len(q) for q in qualities)
    padded_qualities = np.zeros((len(qualities), max_len), dtype=np.int8)
    for i, q in enumerate(qualities):
        padded_qualities[i, : len(q)] = q

    return DNAArray(sequences, qualities=padded_qualities, ids=ids)


def read_pdb(filepath: str) -> StructureArray:
    """Read structure from PDB file.

    Args:
        filepath: Path to PDB file.

    Returns:
        StructureArray instance.

    Example:
        >>> structure = sc.read_pdb("protein.pdb")

    """
    from seqcore.core.structure import StructureArray

    return StructureArray.from_pdb(filepath)


def read_mmcif(filepath: str) -> StructureArray:
    """Read structure from mmCIF file.

    Args:
        filepath: Path to mmCIF file.

    Returns:
        StructureArray instance.

    Example:
        >>> structure = sc.read_mmcif("structure.cif")

    """
    from seqcore.core.structure import StructureArray

    return StructureArray.from_mmcif(filepath)


def read_sdf(filepath: str) -> list:
    """Read molecules from SDF file.

    Args:
        filepath: Path to SDF file.

    Returns:
        List of Molecule objects.

    Example:
        >>> mols = sc.read_sdf("molecules.sdf")

    """
    from seqcore.molecules import Molecule

    molecules = []

    with _open_file(filepath, "r") as f:
        mol_block = []
        for line in f:
            mol_block.append(line)
            if line.strip() == "$$$$":
                mol_str = "".join(mol_block)
                try:
                    mol = Molecule.from_mol_block(mol_str)
                    molecules.append(mol)
                except Exception:
                    pass
                mol_block = []

    return molecules


def read_vcf(filepath: str) -> dict:
    """Read variants from VCF file.

    Args:
        filepath: Path to VCF file.

    Returns:
        Dictionary with variant data.

    Example:
        >>> variants = sc.read_vcf("variants.vcf")

    """
    variants = {
        "chrom": [],
        "pos": [],
        "id": [],
        "ref": [],
        "alt": [],
        "qual": [],
        "filter": [],
        "info": [],
        "samples": [],
    }
    sample_names = []

    with _open_file(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("##"):
                continue
            elif line.startswith("#CHROM"):
                parts = line.split("\t")
                if len(parts) > 9:
                    sample_names = parts[9:]
            elif not line.startswith("#") and line:
                parts = line.split("\t")
                if len(parts) >= 8:
                    variants["chrom"].append(parts[0])
                    variants["pos"].append(int(parts[1]))
                    variants["id"].append(parts[2])
                    variants["ref"].append(parts[3])
                    variants["alt"].append(parts[4].split(","))
                    variants["qual"].append(float(parts[5]) if parts[5] != "." else None)
                    variants["filter"].append(parts[6])
                    variants["info"].append(parts[7])
                    if len(parts) > 9:
                        variants["samples"].append(parts[9:])

    variants["sample_names"] = sample_names
    return variants


def read_mol2(filepath: str) -> list:
    """Read molecules from MOL2 (Tripos) file.

    Args:
        filepath: Path to MOL2 file.

    Returns:
        List of Molecule objects.

    Example:
        >>> mols = sc.read_mol2("molecules.mol2")

    """
    import numpy as np

    from seqcore.molecules import Molecule

    molecules = []

    with _open_file(filepath, "r") as f:
        content = f.read()

    # Split by @<TRIPOS>MOLECULE
    mol_blocks = content.split("@<TRIPOS>MOLECULE")

    for block in mol_blocks[1:]:  # Skip first empty block
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        name = lines[0].strip()
        atoms = []
        bonds = []
        coordinates = []

        in_atoms = False
        in_bonds = False

        for line in lines:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atoms = True
                in_bonds = False
                continue
            elif line.startswith("@<TRIPOS>BOND"):
                in_atoms = False
                in_bonds = True
                continue
            elif line.startswith("@<TRIPOS>"):
                in_atoms = False
                in_bonds = False
                continue

            if in_atoms and line.strip():
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        atom_id = int(parts[0])
                        atom_name = parts[1]
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        atom_type = parts[5]
                        element = atom_type.split(".")[0]
                        atoms.append(
                            {
                                "id": atom_id,
                                "name": atom_name,
                                "element": element,
                                "x": x,
                                "y": y,
                                "z": z,
                                "type": atom_type,
                            }
                        )
                        coordinates.append([x, y, z])
                    except (ValueError, IndexError):
                        pass

            elif in_bonds and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        bond_id = int(parts[0])
                        atom1 = int(parts[1]) - 1
                        atom2 = int(parts[2]) - 1
                        bond_type = parts[3]
                        bonds.append(
                            {"id": bond_id, "atom1": atom1, "atom2": atom2, "type": bond_type}
                        )
                    except (ValueError, IndexError):
                        pass

        if atoms:
            mol = Molecule(
                smiles="",
                name=name,
                atoms=atoms,
                bonds=bonds,
                coordinates=np.array(coordinates) if coordinates else None,
            )
            molecules.append(mol)

    return molecules


def read_gff(filepath: str) -> dict:
    """Read annotations from GFF/GFF3/GTF file.

    Args:
        filepath: Path to GFF file.

    Returns:
        Dictionary with annotation data.

    Example:
        >>> annotations = sc.read_gff("annotation.gff3")

    """
    annotations = {
        "seqid": [],
        "source": [],
        "type": [],
        "start": [],
        "end": [],
        "score": [],
        "strand": [],
        "phase": [],
        "attributes": [],
    }

    with _open_file(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) >= 9:
                annotations["seqid"].append(parts[0])
                annotations["source"].append(parts[1])
                annotations["type"].append(parts[2])
                annotations["start"].append(int(parts[3]))
                annotations["end"].append(int(parts[4]))
                annotations["score"].append(float(parts[5]) if parts[5] != "." else None)
                annotations["strand"].append(parts[6])
                annotations["phase"].append(int(parts[7]) if parts[7] != "." else None)

                # Parse attributes
                attrs = {}
                for attr in parts[8].split(";"):
                    attr = attr.strip()
                    if "=" in attr:
                        key, value = attr.split("=", 1)
                        attrs[key] = value
                    elif " " in attr:
                        key, value = attr.split(" ", 1)
                        attrs[key] = value.strip('"')
                annotations["attributes"].append(attrs)

    return annotations


def read_bed(filepath: str) -> dict:
    """Read genomic regions from BED file.

    Args:
        filepath: Path to BED file.

    Returns:
        Dictionary with BED data.

    Example:
        >>> regions = sc.read_bed("regions.bed")

    """
    regions = {
        "chrom": [],
        "start": [],
        "end": [],
        "name": [],
        "score": [],
        "strand": [],
        "thick_start": [],
        "thick_end": [],
        "item_rgb": [],
        "block_count": [],
        "block_sizes": [],
        "block_starts": [],
    }

    with _open_file(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if (
                not line
                or line.startswith("#")
                or line.startswith("track")
                or line.startswith("browser")
            ):
                continue

            parts = line.split("\t")
            n_cols = len(parts)

            if n_cols >= 3:
                regions["chrom"].append(parts[0])
                regions["start"].append(int(parts[1]))
                regions["end"].append(int(parts[2]))
                regions["name"].append(parts[3] if n_cols > 3 else "")
                regions["score"].append(int(parts[4]) if n_cols > 4 else 0)
                regions["strand"].append(parts[5] if n_cols > 5 else ".")
                regions["thick_start"].append(int(parts[6]) if n_cols > 6 else int(parts[1]))
                regions["thick_end"].append(int(parts[7]) if n_cols > 7 else int(parts[2]))
                regions["item_rgb"].append(parts[8] if n_cols > 8 else "0,0,0")
                regions["block_count"].append(int(parts[9]) if n_cols > 9 else 1)
                regions["block_sizes"].append(
                    parts[10] if n_cols > 10 else str(int(parts[2]) - int(parts[1]))
                )
                regions["block_starts"].append(parts[11] if n_cols > 11 else "0")

    return regions


def read_genbank(filepath: str) -> list:
    """Read sequences from GenBank file.

    Args:
        filepath: Path to GenBank file.

    Returns:
        List of dictionaries with sequence data and features.

    Example:
        >>> records = sc.read_genbank("sequence.gb")

    """
    records = []
    current_record = None
    in_sequence = False
    sequence_lines = []

    with _open_file(filepath, "r") as f:
        for line in f:
            if line.startswith("LOCUS"):
                if current_record is not None:
                    current_record["sequence"] = "".join(sequence_lines).upper()
                    records.append(current_record)
                parts = line.split()
                current_record = {
                    "locus": parts[1] if len(parts) > 1 else "",
                    "length": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0,
                    "molecule_type": "",
                    "topology": "",
                    "division": "",
                    "date": "",
                    "definition": "",
                    "accession": "",
                    "version": "",
                    "keywords": "",
                    "source": "",
                    "organism": "",
                    "features": [],
                    "sequence": "",
                }
                sequence_lines = []
                in_sequence = False

            elif current_record is not None:
                if line.startswith("DEFINITION"):
                    current_record["definition"] = line[12:].strip()
                elif line.startswith("ACCESSION"):
                    current_record["accession"] = line[12:].strip().split()[0]
                elif line.startswith("VERSION"):
                    current_record["version"] = line[12:].strip()
                elif line.startswith("KEYWORDS"):
                    current_record["keywords"] = line[12:].strip()
                elif line.startswith("SOURCE"):
                    current_record["source"] = line[12:].strip()
                elif line.startswith("  ORGANISM"):
                    current_record["organism"] = line[12:].strip()
                elif line.startswith("ORIGIN"):
                    in_sequence = True
                elif line.startswith("//"):
                    in_sequence = False
                elif in_sequence:
                    # Remove numbers and spaces from sequence lines
                    seq = "".join(c for c in line if c.isalpha())
                    sequence_lines.append(seq)

        if current_record is not None:
            current_record["sequence"] = "".join(sequence_lines).upper()
            records.append(current_record)

    return records


def read_embl(filepath: str) -> list:
    """Read sequences from EMBL file.

    Args:
        filepath: Path to EMBL file.

    Returns:
        List of dictionaries with sequence data.

    Example:
        >>> records = sc.read_embl("sequence.embl")

    """
    records = []
    current_record = None
    in_sequence = False
    sequence_lines = []

    with _open_file(filepath, "r") as f:
        for line in f:
            if line.startswith("ID"):
                if current_record is not None:
                    current_record["sequence"] = "".join(sequence_lines).upper()
                    records.append(current_record)
                parts = line[5:].strip().split(";")
                current_record = {
                    "id": parts[0].strip() if parts else "",
                    "accession": "",
                    "description": "",
                    "keywords": "",
                    "organism": "",
                    "features": [],
                    "sequence": "",
                }
                sequence_lines = []
                in_sequence = False

            elif current_record is not None:
                if line.startswith("AC"):
                    current_record["accession"] = line[5:].strip().rstrip(";")
                elif line.startswith("DE"):
                    current_record["description"] += line[5:].strip() + " "
                elif line.startswith("KW"):
                    current_record["keywords"] += line[5:].strip() + " "
                elif line.startswith("OS"):
                    current_record["organism"] = line[5:].strip()
                elif line.startswith("SQ"):
                    in_sequence = True
                elif line.startswith("//"):
                    in_sequence = False
                elif in_sequence and line.startswith("     "):
                    seq = "".join(c for c in line if c.isalpha())
                    sequence_lines.append(seq)

        if current_record is not None:
            current_record["sequence"] = "".join(sequence_lines).upper()
            records.append(current_record)

    return records


def read_newick(filepath: str) -> list:
    """Read phylogenetic trees from Newick file.

    Args:
        filepath: Path to Newick file.

    Returns:
        List of tree strings (one per line).

    Example:
        >>> trees = sc.read_newick("trees.nwk")

    """
    trees = []

    with _open_file(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                trees.append(line)

    return trees


def read_stockholm(filepath: str) -> dict:
    """Read multiple sequence alignment from Stockholm format.

    Args:
        filepath: Path to Stockholm file.

    Returns:
        Dictionary with alignment data.

    Example:
        >>> msa = sc.read_stockholm("alignment.sto")

    """
    alignments = []
    current_aln = None

    with _open_file(filepath, "r") as f:
        sequences = {}
        metadata = {}

        for line in f:
            line = line.rstrip()

            if line.startswith("# STOCKHOLM"):
                if current_aln is not None:
                    current_aln["sequences"] = sequences
                    alignments.append(current_aln)
                current_aln = {"metadata": {}, "sequences": {}}
                sequences = {}
                metadata = {}

            elif line.startswith("#=GF"):
                parts = line[5:].split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    metadata[key] = value
                    if current_aln:
                        current_aln["metadata"][key] = value

            elif line.startswith("#=GS"):
                pass  # Sequence-specific metadata

            elif line.startswith("#=GC"):
                pass  # Column annotation

            elif line.startswith("//"):
                if current_aln is not None:
                    current_aln["sequences"] = sequences
                    alignments.append(current_aln)
                current_aln = None
                sequences = {}

            elif line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 2:
                    seq_id = parts[0]
                    seq = parts[1]
                    if seq_id in sequences:
                        sequences[seq_id] += seq
                    else:
                        sequences[seq_id] = seq

    if current_aln is not None and sequences:
        current_aln["sequences"] = sequences
        alignments.append(current_aln)

    return (
        alignments
        if len(alignments) > 1
        else (alignments[0] if alignments else {"sequences": {}, "metadata": {}})
    )


def read_phylip(filepath: str) -> dict:
    """Read multiple sequence alignment from PHYLIP format.

    Args:
        filepath: Path to PHYLIP file.

    Returns:
        Dictionary with alignment data.

    Example:
        >>> msa = sc.read_phylip("alignment.phy")

    """
    sequences = {}
    names = []

    with _open_file(filepath, "r") as f:
        lines = f.readlines()

    if not lines:
        return {"sequences": {}, "n_taxa": 0, "n_sites": 0}

    # Parse header
    header = lines[0].strip().split()
    n_taxa = int(header[0])
    n_sites = int(header[1]) if len(header) > 1 else 0

    # Parse sequences (interleaved or sequential)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            seq = "".join(parts[1:])

            if name not in sequences:
                sequences[name] = seq
                names.append(name)
            else:
                sequences[name] += seq
        elif len(parts) == 1 and names:
            # Continuation of sequence (interleaved format)
            # Add to sequences in order
            pass

    return {"sequences": sequences, "n_taxa": n_taxa, "n_sites": n_sites, "names": names}


def read_sam(filepath: str) -> dict:
    """Read alignments from SAM file.

    Args:
        filepath: Path to SAM file.

    Returns:
        Dictionary with alignment data.

    Example:
        >>> alignments = sc.read_sam("alignments.sam")

    """
    alignments = {
        "header": [],
        "references": {},
        "reads": [],
    }

    with _open_file(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("@"):
                alignments["header"].append(line)
                if line.startswith("@SQ"):
                    parts = line.split("\t")
                    ref_name = None
                    ref_len = 0
                    for part in parts[1:]:
                        if part.startswith("SN:"):
                            ref_name = part[3:]
                        elif part.startswith("LN:"):
                            ref_len = int(part[3:])
                    if ref_name:
                        alignments["references"][ref_name] = ref_len
            else:
                parts = line.split("\t")
                if len(parts) >= 11:
                    read = {
                        "qname": parts[0],
                        "flag": int(parts[1]),
                        "rname": parts[2],
                        "pos": int(parts[3]),
                        "mapq": int(parts[4]),
                        "cigar": parts[5],
                        "rnext": parts[6],
                        "pnext": int(parts[7]) if parts[7] != "*" else 0,
                        "tlen": int(parts[8]),
                        "seq": parts[9],
                        "qual": parts[10],
                        "tags": parts[11:] if len(parts) > 11 else [],
                    }
                    alignments["reads"].append(read)

    return alignments


def read_h5ad(filepath: str) -> dict:
    """Read single-cell data from HDF5/AnnData format.

    Args:
        filepath: Path to H5AD file.

    Returns:
        Dictionary with single-cell data.

    Example:
        >>> adata = sc.read_h5ad("data.h5ad")

    """
    try:
        import h5py
    except ImportError as err:
        raise ImportError(
            "h5py is required for HDF5 files. Install with: pip install h5py"
        ) from err

    data = {
        "X": None,
        "obs": {},
        "var": {},
        "uns": {},
        "n_obs": 0,
        "n_vars": 0,
    }

    with h5py.File(filepath, "r") as f:
        # Read main matrix
        if "X" in f:
            x = f["X"]
            if isinstance(x, h5py.Dataset):
                data["X"] = x[:]
                data["n_obs"] = x.shape[0]
                data["n_vars"] = x.shape[1] if len(x.shape) > 1 else 1
            elif isinstance(x, h5py.Group):
                # Sparse matrix
                if "data" in x:
                    data["X_data"] = x["data"][:]
                if "indices" in x:
                    data["X_indices"] = x["indices"][:]
                if "indptr" in x:
                    data["X_indptr"] = x["indptr"][:]
                if "shape" in x.attrs:
                    shape = x.attrs["shape"]
                    data["n_obs"] = shape[0]
                    data["n_vars"] = shape[1]

        # Read observations metadata
        if "obs" in f:
            obs = f["obs"]
            for key in obs:
                if isinstance(obs[key], h5py.Dataset):
                    data["obs"][key] = obs[key][:]

        # Read variables metadata
        if "var" in f:
            var = f["var"]
            for key in var:
                if isinstance(var[key], h5py.Dataset):
                    data["var"][key] = var[key][:]

        # Read unstructured data
        if "uns" in f:
            uns = f["uns"]
            for key in uns:
                if isinstance(uns[key], h5py.Dataset):
                    data["uns"][key] = uns[key][:]

    return data


def read_stream(
    filepath: str,
    batch_size: int = 10000,
    format: str | None = None,
) -> Iterator[BioArray]:
    """Stream sequences from large file in batches.

    Args:
        filepath: Path to input file.
        batch_size: Number of sequences per batch.
        format: File format (auto-detected if not specified).

    Yields:
        BioArray batches.

    Example:
        >>> for batch in sc.read_stream("huge.fastq.gz", batch_size=100000):
        ...     results = process(batch)

    """
    import numpy as np

    from seqcore.core.arrays import DNAArray

    if format is None:
        format = _detect_format(filepath)

    if format == "fasta":
        sequences = []
        ids = []
        current_seq = []
        current_id = None

        with _open_file(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None:
                        sequences.append("".join(current_seq))
                        if len(sequences) >= batch_size:
                            yield DNAArray(sequences, ids=ids)
                            sequences = []
                            ids = []
                    current_id = line[1:].split()[0]
                    ids.append(current_id)
                    current_seq = []
                elif line:
                    current_seq.append(line.upper())

            if current_id is not None:
                sequences.append("".join(current_seq))
            if sequences:
                yield DNAArray(sequences, ids=ids)

    elif format == "fastq":
        sequences = []
        ids = []
        qualities = []

        with _open_file(filepath, "r") as f:
            while True:
                header = f.readline().strip()
                if not header:
                    break
                if not header.startswith("@"):
                    continue

                seq = f.readline().strip()
                f.readline()
                qual = f.readline().strip()

                ids.append(header[1:].split()[0])
                sequences.append(seq.upper())
                qualities.append(np.array([ord(c) - 33 for c in qual], dtype=np.int8))

                if len(sequences) >= batch_size:
                    max_len = max(len(q) for q in qualities)
                    padded = np.zeros((len(qualities), max_len), dtype=np.int8)
                    for i, q in enumerate(qualities):
                        padded[i, : len(q)] = q
                    yield DNAArray(sequences, qualities=padded, ids=ids)
                    sequences = []
                    ids = []
                    qualities = []

            if sequences:
                max_len = max(len(q) for q in qualities)
                padded = np.zeros((len(qualities), max_len), dtype=np.int8)
                for i, q in enumerate(qualities):
                    padded[i, : len(q)] = q
                yield DNAArray(sequences, qualities=padded, ids=ids)

    else:
        raise ValueError(f"Streaming not supported for format: {format}")


def write(data: Any, filepath: str, format: str | None = None, append: bool = False):
    """Write data to file.

    Args:
        data: Data to write (BioArray, StructureArray, etc.).
        filepath: Output file path.
        format: File format (auto-detected if not specified).
        append: If True, append to existing file.

    Example:
        >>> sc.write(sequences, "output.fasta")
        >>> sc.write(structure, "output.pdb")

    """
    from seqcore.core.arrays import BioArray
    from seqcore.core.structure import StructureArray

    if format is None:
        format = _detect_format(filepath)

    mode = "a" if append else "w"

    if format == "fasta":
        with _open_file(filepath, mode) as f:
            if isinstance(data, BioArray):
                for i, seq in enumerate(data.sequences):
                    f.write(f">{data.ids[i]}\n")
                    # Write sequence in lines of 80 characters
                    for j in range(0, len(seq), 80):
                        f.write(f"{seq[j:j+80]}\n")
            else:
                for i, seq in enumerate(data):
                    f.write(f">seq_{i}\n")
                    for j in range(0, len(seq), 80):
                        f.write(f"{seq[j:j+80]}\n")

    elif format == "pdb":
        if isinstance(data, StructureArray):
            data.to_pdb(filepath)
        else:
            raise ValueError("PDB format requires StructureArray")

    else:
        raise ValueError(f"Write not supported for format: {format}")


def fetch(identifier: str, database: str | None = None) -> Any:
    """Fetch data from online database.

    Args:
        identifier: Sequence/structure identifier.
        database: Database name (auto-detected if not specified).

    Returns:
        Fetched data (sequence or structure).

    Example:
        >>> seq = sc.fetch("NP_000509")  # UniProt/NCBI
        >>> structure = sc.fetch("1ABC")  # PDB

    """
    import tempfile
    import urllib.request

    # Auto-detect database
    if database is None:
        if len(identifier) == 4 and identifier[0].isdigit():
            database = "pdb"
        elif identifier.startswith(("NP_", "XP_", "YP_", "NC_", "NM_")):
            database = "ncbi"
        elif identifier.startswith(("P", "Q", "O", "A", "B", "C")) and len(identifier) >= 6:
            database = "uniprot"
        else:
            database = "pdb"  # Default guess

    if database == "pdb":
        url = f"https://files.rcsb.org/download/{identifier.upper()}.pdb"
        with _safe_urlopen(url) as response:
            content = response.read().decode("utf-8")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(content)
            temp_path = f.name
        try:
            result = read_pdb(temp_path)
        finally:
            os.unlink(temp_path)
        return result

    elif database == "uniprot":
        url = f"https://rest.uniprot.org/uniprotkb/{identifier}.fasta"
        with _safe_urlopen(url) as response:
            content = response.read().decode("utf-8")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(content)
            temp_path = f.name
        try:
            result = read_fasta(temp_path)
        finally:
            os.unlink(temp_path)
        return result

    elif database == "ncbi":
        # Use NCBI E-utilities
        from seqcore.core.arrays import ProteinArray

        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={identifier}&rettype=fasta&retmode=text"
        with _safe_urlopen(url) as response:
            content = response.read().decode("utf-8")
        lines = content.strip().split("\n")
        if lines:
            seq_id = lines[0][1:].split()[0] if lines[0].startswith(">") else identifier
            seq = "".join(lines[1:])
            return ProteinArray([seq], ids=[seq_id])
        raise ValueError(f"Could not fetch {identifier} from NCBI")

    else:
        raise ValueError(f"Unknown database: {database}")


__all__ = [
    "read",
    "write",
    "read_fasta",
    "read_fastq",
    "read_pdb",
    "read_mmcif",
    "read_sdf",
    "read_mol2",
    "read_vcf",
    "read_gff",
    "read_bed",
    "read_genbank",
    "read_embl",
    "read_newick",
    "read_stockholm",
    "read_phylip",
    "read_sam",
    "read_h5ad",
    "read_stream",
    "fetch",
]
