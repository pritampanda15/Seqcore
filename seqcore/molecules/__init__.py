"""Small molecule handling and drug design operations.

Provides molecular property calculations, fingerprints, similarity analysis,
and ADMET filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from seqcore.core.structure import StructureArray


@dataclass
class Molecule:
    """Represents a small molecule.

    Stores molecular structure and provides property calculations.
    """

    smiles: str
    name: str = ""
    atoms: list[dict[str, Any]] = field(default_factory=list)
    bonds: list[dict[str, Any]] = field(default_factory=list)
    coordinates: np.ndarray | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_smiles(cls, smiles: str, name: str = "") -> Molecule:
        """Create molecule from SMILES string.

        Args:
            smiles: SMILES string.
            name: Molecule name.

        Returns:
            Molecule instance.

        Example:
            >>> mol = sc.Molecule.from_smiles("CCO")  # Ethanol

        """
        return cls(smiles=smiles, name=name)

    @classmethod
    def from_sdf(cls, filepath: str) -> Molecule:
        """Create molecule from SDF file.

        Args:
            filepath: Path to SDF file.

        Returns:
            Molecule instance.

        """
        with open(filepath) as f:
            content = f.read()
        return cls.from_mol_block(content)

    @classmethod
    def from_mol_block(cls, mol_block: str) -> Molecule:
        """Create molecule from MOL/SDF block.

        Args:
            mol_block: MOL block string.

        Returns:
            Molecule instance.

        """
        lines = mol_block.strip().split("\n")

        name = lines[0].strip() if lines else ""

        # Parse counts line (line 4)
        if len(lines) > 3:
            counts_line = lines[3]
            try:
                n_atoms = int(counts_line[0:3])
                n_bonds = int(counts_line[3:6])
            except (ValueError, IndexError):
                n_atoms = 0
                n_bonds = 0
        else:
            n_atoms = 0
            n_bonds = 0

        atoms = []
        coordinates = []

        # Parse atoms
        for i in range(4, 4 + n_atoms):
            if i < len(lines):
                line = lines[i]
                try:
                    x = float(line[0:10])
                    y = float(line[10:20])
                    z = float(line[20:30])
                    element = line[31:34].strip()
                    atoms.append({"element": element, "x": x, "y": y, "z": z})
                    coordinates.append([x, y, z])
                except (ValueError, IndexError):
                    pass

        bonds = []
        # Parse bonds
        for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
            if i < len(lines):
                line = lines[i]
                try:
                    a1 = int(line[0:3]) - 1
                    a2 = int(line[3:6]) - 1
                    bond_type = int(line[6:9])
                    bonds.append({"atom1": a1, "atom2": a2, "type": bond_type})
                except (ValueError, IndexError):
                    pass

        return cls(
            smiles="",
            name=name,
            atoms=atoms,
            bonds=bonds,
            coordinates=np.array(coordinates) if coordinates else None,
        )

    def to_rdkit(self):
        """Convert to RDKit Mol object.

        Returns:
            RDKit Mol object.

        """
        try:
            from rdkit import Chem

            if self.smiles:
                return Chem.MolFromSmiles(self.smiles)
            else:
                raise ValueError("SMILES required for RDKit conversion")
        except ImportError as err:
            raise ImportError("RDKit required. Install with: pip install rdkit") from err

    @classmethod
    def from_rdkit(cls, mol) -> Molecule:
        """Create from RDKit Mol object.

        Args:
            mol: RDKit Mol object.

        Returns:
            Molecule instance.

        """
        try:
            from rdkit import Chem

            smiles = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            return cls(smiles=smiles, name=name)
        except ImportError as err:
            raise ImportError("RDKit required. Install with: pip install rdkit") from err

    def __repr__(self):
        return f"Molecule(smiles='{self.smiles}', name='{self.name}')"


def _try_rdkit(func_name: str, molecules: Molecule | list[Molecule], *args, **kwargs) -> Any:
    """Try to use RDKit for calculations, fall back to basic if unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

        return True, Chem, Descriptors, rdMolDescriptors, AllChem
    except ImportError:
        return False, None, None, None, None


def _get_mol_list(molecules: Molecule | list[Molecule]) -> list[Molecule]:
    """Convert single molecule to list."""
    if isinstance(molecules, Molecule):
        return [molecules]
    return molecules


def molecular_weight(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Calculate molecular weight.

    Args:
        molecules: Molecule(s) to analyze.

    Returns:
        Array of molecular weights.

    Example:
        >>> mw = sc.molecular_weight(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, Descriptors, _, _ = _try_rdkit("molecular_weight", molecules)

    if has_rdkit:
        weights = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    weights.append(Descriptors.MolWt(rdkit_mol))
                else:
                    weights.append(0.0)
            else:
                weights.append(0.0)
        return np.array(weights)

    # Basic calculation from atoms
    element_weights = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "S": 32.065,
        "P": 30.974,
        "F": 18.998,
        "Cl": 35.453,
        "Br": 79.904,
        "I": 126.904,
    }

    weights = []
    for mol in mols:
        if mol.atoms:
            mw = sum(element_weights.get(a["element"], 0) for a in mol.atoms)
            weights.append(mw)
        else:
            weights.append(0.0)
    return np.array(weights)


def logp(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Calculate LogP (partition coefficient).

    Args:
        molecules: Molecule(s) to analyze.

    Returns:
        Array of LogP values.

    Example:
        >>> logp_values = sc.logp(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, Descriptors, _, _ = _try_rdkit("logp", molecules)

    if has_rdkit:
        values = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    values.append(Descriptors.MolLogP(rdkit_mol))
                else:
                    values.append(0.0)
            else:
                values.append(0.0)
        return np.array(values)

    # Placeholder without RDKit
    return np.zeros(len(mols))


def h_bond_donors(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Count hydrogen bond donors.

    Args:
        molecules: Molecule(s) to analyze.

    Returns:
        Array of HBD counts.

    Example:
        >>> hbd = sc.h_bond_donors(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, Descriptors, rdMolDescriptors, _ = _try_rdkit("hbd", molecules)

    if has_rdkit:
        values = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    values.append(rdMolDescriptors.CalcNumHBD(rdkit_mol))
                else:
                    values.append(0)
            else:
                values.append(0)
        return np.array(values)

    return np.zeros(len(mols), dtype=int)


def h_bond_acceptors(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Count hydrogen bond acceptors.

    Args:
        molecules: Molecule(s) to analyze.

    Returns:
        Array of HBA counts.

    Example:
        >>> hba = sc.h_bond_acceptors(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, Descriptors, rdMolDescriptors, _ = _try_rdkit("hba", molecules)

    if has_rdkit:
        values = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    values.append(rdMolDescriptors.CalcNumHBA(rdkit_mol))
                else:
                    values.append(0)
            else:
                values.append(0)
        return np.array(values)

    return np.zeros(len(mols), dtype=int)


def tpsa(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Calculate topological polar surface area.

    Args:
        molecules: Molecule(s) to analyze.

    Returns:
        Array of TPSA values.

    Example:
        >>> tpsa_values = sc.tpsa(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, Descriptors, rdMolDescriptors, _ = _try_rdkit("tpsa", molecules)

    if has_rdkit:
        values = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    values.append(rdMolDescriptors.CalcTPSA(rdkit_mol))
                else:
                    values.append(0.0)
            else:
                values.append(0.0)
        return np.array(values)

    return np.zeros(len(mols))


def rotatable_bonds(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Count rotatable bonds.

    Args:
        molecules: Molecule(s) to analyze.

    Returns:
        Array of rotatable bond counts.

    Example:
        >>> rot = sc.rotatable_bonds(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, Descriptors, rdMolDescriptors, _ = _try_rdkit("rotatable", molecules)

    if has_rdkit:
        values = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    values.append(rdMolDescriptors.CalcNumRotatableBonds(rdkit_mol))
                else:
                    values.append(0)
            else:
                values.append(0)
        return np.array(values)

    return np.zeros(len(mols), dtype=int)


def lipinski_filter(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Apply Lipinski's Rule of Five filter.

    Args:
        molecules: Molecule(s) to filter.

    Returns:
        Boolean array (True = passes filter).

    Example:
        >>> passes = sc.lipinski_filter(molecules)

    """
    mols = _get_mol_list(molecules)

    mw = molecular_weight(mols)
    lp = logp(mols)
    hbd = h_bond_donors(mols)
    hba = h_bond_acceptors(mols)

    # Lipinski criteria: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10
    passes = (mw <= 500) & (lp <= 5) & (hbd <= 5) & (hba <= 10)

    return passes


def veber_filter(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Apply Veber's filter for oral bioavailability.

    Args:
        molecules: Molecule(s) to filter.

    Returns:
        Boolean array (True = passes filter).

    Example:
        >>> passes = sc.veber_filter(molecules)

    """
    mols = _get_mol_list(molecules)

    rot = rotatable_bonds(mols)
    tp = tpsa(mols)

    # Veber criteria: rotatable bonds <= 10, TPSA <= 140
    passes = (rot <= 10) & (tp <= 140)

    return passes


def bbb_filter(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Apply blood-brain barrier permeability filter.

    Args:
        molecules: Molecule(s) to filter.

    Returns:
        Boolean array (True = likely BBB permeable).

    Example:
        >>> passes = sc.bbb_filter(molecules)

    """
    mols = _get_mol_list(molecules)

    mw = molecular_weight(mols)
    lp = logp(mols)
    tp = tpsa(mols)
    hbd = h_bond_donors(mols)

    # BBB criteria (simplified)
    passes = (mw <= 400) & (lp >= 1) & (lp <= 3) & (tp <= 90) & (hbd <= 3)

    return passes


def morgan_fingerprint(
    molecules: Molecule | list[Molecule],
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Calculate Morgan (circular) fingerprints.

    Args:
        molecules: Molecule(s) to fingerprint.
        radius: Fingerprint radius.
        n_bits: Fingerprint length.

    Returns:
        Array of fingerprints (n_molecules x n_bits).

    Example:
        >>> fps = sc.morgan_fingerprint(molecules, radius=2)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, _, rdMolDescriptors, AllChem = _try_rdkit("morgan", molecules)

    if has_rdkit:
        fps = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(rdkit_mol, radius, nBits=n_bits)
                    fps.append(np.array(fp))
                else:
                    fps.append(np.zeros(n_bits))
            else:
                fps.append(np.zeros(n_bits))
        return np.array(fps)

    # Placeholder without RDKit
    return np.zeros((len(mols), n_bits))


def rdkit_fingerprint(
    molecules: Molecule | list[Molecule],
    n_bits: int = 2048,
) -> np.ndarray:
    """Calculate RDKit topological fingerprints.

    Args:
        molecules: Molecule(s) to fingerprint.
        n_bits: Fingerprint length.

    Returns:
        Array of fingerprints.

    Example:
        >>> fps = sc.rdkit_fingerprint(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, _, _, _ = _try_rdkit("rdkit_fp", molecules)

    if has_rdkit:
        from rdkit.Chem import RDKFingerprint

        fps = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    fp = RDKFingerprint(rdkit_mol, fpSize=n_bits)
                    fps.append(np.array(fp))
                else:
                    fps.append(np.zeros(n_bits))
            else:
                fps.append(np.zeros(n_bits))
        return np.array(fps)

    return np.zeros((len(mols), n_bits))


def maccs_fingerprint(molecules: Molecule | list[Molecule]) -> np.ndarray:
    """Calculate MACCS keys fingerprints.

    Args:
        molecules: Molecule(s) to fingerprint.

    Returns:
        Array of 166-bit MACCS fingerprints.

    Example:
        >>> fps = sc.maccs_fingerprint(molecules)

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, _, _, _ = _try_rdkit("maccs", molecules)

    if has_rdkit:
        from rdkit.Chem import MACCSkeys

        fps = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol:
                    fp = MACCSkeys.GenMACCSKeys(rdkit_mol)
                    fps.append(np.array(fp))
                else:
                    fps.append(np.zeros(167))
            else:
                fps.append(np.zeros(167))
        return np.array(fps)

    return np.zeros((len(mols), 167))


def tanimoto_similarity(
    fingerprints1: np.ndarray,
    fingerprints2: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate Tanimoto similarity between fingerprints.

    Args:
        fingerprints1: First fingerprint array.
        fingerprints2: Second fingerprint array (or None for self-similarity).

    Returns:
        Similarity matrix.

    Example:
        >>> sim = sc.tanimoto_similarity(fps1, fps2)

    """
    if fingerprints2 is None:
        fingerprints2 = fingerprints1

    # Ensure 2D arrays
    if fingerprints1.ndim == 1:
        fingerprints1 = fingerprints1.reshape(1, -1)
    if fingerprints2.ndim == 1:
        fingerprints2 = fingerprints2.reshape(1, -1)

    # Calculate Tanimoto: |A & B| / |A | B|
    # For binary fingerprints: intersection / (sum1 + sum2 - intersection)

    n1 = len(fingerprints1)
    n2 = len(fingerprints2)
    similarity = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            fp1 = fingerprints1[i]
            fp2 = fingerprints2[j]

            intersection = np.sum(np.logical_and(fp1, fp2))
            union = np.sum(np.logical_or(fp1, fp2))

            if union > 0:
                similarity[i, j] = intersection / union
            else:
                similarity[i, j] = 0.0

    return similarity


def find_similar(
    query: Molecule,
    library: list[Molecule],
    threshold: float = 0.7,
    fingerprint_type: str = "morgan",
) -> list[dict[str, Any]]:
    """Find similar molecules in library.

    Args:
        query: Query molecule.
        library: Library of molecules.
        threshold: Similarity threshold.
        fingerprint_type: Type of fingerprint to use.

    Returns:
        List of similar molecules with scores.

    Example:
        >>> similar = sc.find_similar(query, library, threshold=0.7)

    """
    if fingerprint_type == "morgan":
        query_fp = morgan_fingerprint([query])
        lib_fps = morgan_fingerprint(library)
    elif fingerprint_type == "rdkit":
        query_fp = rdkit_fingerprint([query])
        lib_fps = rdkit_fingerprint(library)
    else:
        query_fp = maccs_fingerprint([query])
        lib_fps = maccs_fingerprint(library)

    similarities = tanimoto_similarity(query_fp, lib_fps)[0]

    results = []
    for i, sim in enumerate(similarities):
        if sim >= threshold:
            results.append(
                {
                    "molecule": library[i],
                    "similarity": float(sim),
                    "index": i,
                }
            )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def substructure_search(
    molecules: Molecule | list[Molecule],
    pattern: str,
) -> np.ndarray:
    """Search for substructure pattern (SMARTS).

    Args:
        molecules: Molecule(s) to search.
        pattern: SMARTS pattern.

    Returns:
        Boolean array of matches.

    Example:
        >>> matches = sc.substructure_search(molecules, "c1ccccc1")

    """
    mols = _get_mol_list(molecules)
    has_rdkit, Chem, _, _, _ = _try_rdkit("substructure", molecules)

    if has_rdkit:
        query = Chem.MolFromSmarts(pattern)
        if query is None:
            query = Chem.MolFromSmiles(pattern)

        matches = []
        for mol in mols:
            if mol.smiles:
                rdkit_mol = Chem.MolFromSmiles(mol.smiles)
                if rdkit_mol and query:
                    matches.append(rdkit_mol.HasSubstructMatch(query))
                else:
                    matches.append(False)
            else:
                matches.append(False)
        return np.array(matches)

    return np.zeros(len(mols), dtype=bool)


def generate_conformers(
    molecule: Molecule,
    n_conformers: int = 10,
    random_seed: int = 42,
) -> list[np.ndarray]:
    """Generate 3D conformers for molecule.

    Args:
        molecule: Molecule to generate conformers for.
        n_conformers: Number of conformers to generate.
        random_seed: Random seed for reproducibility.

    Returns:
        List of coordinate arrays.

    Example:
        >>> conformers = sc.generate_conformers(mol, n_conformers=100)

    """
    has_rdkit, Chem, _, _, AllChem = _try_rdkit("conformers", molecule)

    if has_rdkit and molecule.smiles:
        rdkit_mol = Chem.MolFromSmiles(molecule.smiles)
        if rdkit_mol:
            rdkit_mol = Chem.AddHs(rdkit_mol)
            AllChem.EmbedMultipleConfs(
                rdkit_mol,
                numConfs=n_conformers,
                randomSeed=random_seed,
            )

            conformers = []
            for conf in rdkit_mol.GetConformers():
                coords = conf.GetPositions()
                conformers.append(coords)
            return conformers

    return []


def minimize_energy(
    conformers: list[np.ndarray],
    molecule: Molecule | None = None,
) -> list[np.ndarray]:
    """Minimize energy of conformers.

    Args:
        conformers: List of conformer coordinates.
        molecule: Source molecule (for force field).

    Returns:
        List of minimized coordinate arrays.

    Example:
        >>> minimized = sc.minimize_energy(conformers)

    """
    # Placeholder - actual implementation would use force field
    return conformers


def find_interactions(
    protein: StructureArray,
    ligand: Molecule | StructureArray,
    cutoff: float = 4.0,
) -> dict[str, list[dict[str, Any]]]:
    """Find protein-ligand interactions.

    Args:
        protein: Protein structure.
        ligand: Ligand molecule or structure.
        cutoff: Distance cutoff for interactions.

    Returns:
        Dictionary of interaction types and details.

    Example:
        >>> interactions = sc.find_interactions(protein, ligand)

    """
    from seqcore.core.structure import StructureArray

    interactions = {
        "hydrogen_bonds": [],
        "salt_bridges": [],
        "pi_stacking": [],
        "hydrophobic": [],
    }

    # Get ligand coordinates
    if isinstance(ligand, StructureArray):
        lig_coords = ligand.coordinates
        lig_elements = ligand._elements
    elif isinstance(ligand, Molecule) and ligand.coordinates is not None:
        lig_coords = ligand.coordinates
        lig_elements = np.array([a.get("element", "C") for a in ligand.atoms])
    else:
        return interactions

    prot_coords = protein.coordinates
    prot_elements = protein._elements

    # Find contacts
    for i, pc in enumerate(prot_coords):
        for j, lc in enumerate(lig_coords):
            dist = np.sqrt(np.sum((pc - lc) ** 2))

            if dist <= cutoff:
                prot_elem = str(prot_elements[i]).strip().upper()
                lig_elem = str(lig_elements[j]).strip().upper() if j < len(lig_elements) else "C"

                # Hydrogen bond donors/acceptors (N, O)
                if prot_elem in ("N", "O") and lig_elem in ("N", "O"):
                    if dist <= 3.5:
                        interactions["hydrogen_bonds"].append(
                            {
                                "protein_atom": i,
                                "ligand_atom": j,
                                "distance": float(dist),
                                "residue": int(protein._residue_numbers[i]),
                            }
                        )

                # Hydrophobic contacts (C-C)
                elif prot_elem == "C" and lig_elem == "C" and dist <= 4.0:
                    interactions["hydrophobic"].append(
                        {
                            "protein_atom": i,
                            "ligand_atom": j,
                            "distance": float(dist),
                            "residue": int(protein._residue_numbers[i]),
                        }
                    )

    return interactions


__all__ = [
    "Molecule",
    "molecular_weight",
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
]
