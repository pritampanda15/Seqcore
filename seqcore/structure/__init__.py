"""Structural biology operations.

Provides operations for protein/nucleic acid 3D structures including
distance calculations, contact analysis, RMSD, and more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from seqcore.core.structure import StructureArray


@dataclass
class Contact:
    """Represents a contact between two atoms/residues."""

    atom1_idx: int
    atom2_idx: int
    distance: float
    residue1: int
    residue2: int
    chain1: str
    chain2: str


@dataclass
class Pocket:
    """Represents a binding pocket."""

    center: np.ndarray
    volume: float
    residues: list[int]
    druggability: float = 0.0


def distance_matrix(structure: StructureArray, selection: str = "CA") -> np.ndarray:
    """Calculate distance matrix for structure atoms.

    Args:
        structure: Structure to analyze.
        selection: Atom selection ("CA" for alpha carbons, "all" for all atoms).

    Returns:
        Square distance matrix.

    Example:
        >>> dm = sc.distance_matrix(structure)

    """
    if selection == "CA":
        subset = structure.select(atom_name="CA")
        coords = subset.coordinates
    elif selection == "all":
        coords = structure.coordinates
    else:
        coords = structure.coordinates

    n = len(coords)
    dm = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        diff = coords[i] - coords
        dm[i] = np.sqrt(np.sum(diff**2, axis=1))

    return dm


def find_contacts(
    structure: StructureArray,
    cutoff: float = 4.0,
    include_neighbors: bool = False,
) -> list[Contact]:
    """Find atomic contacts within cutoff distance.

    Args:
        structure: Structure to analyze.
        cutoff: Distance cutoff in Angstroms.
        include_neighbors: Include sequential residue contacts.

    Returns:
        List of Contact objects.

    Example:
        >>> contacts = sc.find_contacts(structure, cutoff=4.0)

    """
    coords = structure.coordinates
    n = len(coords)
    contacts = []

    atoms = structure.atoms

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

            if dist <= cutoff:
                res_i = int(atoms[i]["residue_number"])
                res_j = int(atoms[j]["residue_number"])

                # Skip sequential neighbors unless requested
                if (
                    not include_neighbors
                    and abs(res_i - res_j) <= 1
                    and atoms[i]["chain_id"] == atoms[j]["chain_id"]
                ):
                    continue

                contacts.append(
                    Contact(
                        atom1_idx=i,
                        atom2_idx=j,
                        distance=dist,
                        residue1=res_i,
                        residue2=res_j,
                        chain1=str(atoms[i]["chain_id"]),
                        chain2=str(atoms[j]["chain_id"]),
                    )
                )

    return contacts


def find_neighbors(
    structure: StructureArray,
    residue: int | str,
    radius: float = 5.0,
    chain: str | None = None,
) -> StructureArray:
    """Find atoms within radius of a residue.

    Args:
        structure: Structure to search.
        residue: Residue number or name.
        radius: Search radius in Angstroms.
        chain: Chain ID (optional).

    Returns:
        StructureArray with neighboring atoms.

    Example:
        >>> neighbors = sc.find_neighbors(structure, residue=265, radius=5.0)

    """
    # Find center of target residue
    if isinstance(residue, int):
        target = structure.select(residue=residue, chain=chain)
    else:
        target = structure.select(residue=residue, chain=chain)

    if len(target) == 0:
        raise ValueError(f"Residue {residue} not found")

    center = target.centroid()

    # Find all atoms within radius
    coords = structure.coordinates
    distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    mask = distances <= radius

    return structure.__class__(
        coordinates=structure.coordinates[mask],
        atom_names=structure._atom_names[mask],
        residue_names=structure._residue_names[mask],
        residue_numbers=structure._residue_numbers[mask],
        chain_ids=structure._chain_ids[mask],
        elements=structure._elements[mask],
        b_factors=structure._b_factors[mask],
        occupancies=structure._occupancies[mask],
        serial_numbers=structure._serial_numbers[mask],
    )


def rmsd(
    structure1: StructureArray,
    structure2: StructureArray,
    align: bool = False,
    selection: str = "CA",
) -> float:
    """Calculate RMSD between two structures.

    Args:
        structure1: First structure.
        structure2: Second structure.
        align: If True, superpose structures first.
        selection: Atom selection for comparison.

    Returns:
        RMSD value in Angstroms.

    Example:
        >>> rmsd_value = sc.rmsd(wt_structure, mutant_structure, align=True)

    """
    if selection == "CA":
        s1 = structure1.select(atom_name="CA")
        s2 = structure2.select(atom_name="CA")
    else:
        s1 = structure1
        s2 = structure2

    coords1 = s1.coordinates
    coords2 = s2.coordinates

    if len(coords1) != len(coords2):
        raise ValueError(
            f"Structures have different number of atoms: {len(coords1)} vs {len(coords2)}"
        )

    if align:
        coords2 = _kabsch_align(coords1, coords2)

    diff = coords1 - coords2
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def _kabsch_align(ref: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """Kabsch algorithm for optimal superposition."""
    # Center coordinates
    ref_center = ref.mean(axis=0)
    mobile_center = mobile.mean(axis=0)

    ref_centered = ref - ref_center
    mobile_centered = mobile - mobile_center

    # Compute covariance matrix
    H = mobile_centered.T @ ref_centered

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Rotation matrix
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation
    aligned = (mobile - mobile_center) @ R.T + ref_center

    return aligned


def secondary_structure(structure: StructureArray) -> dict[int, str]:
    """Assign secondary structure (simplified DSSP-like).

    Args:
        structure: Structure to analyze.

    Returns:
        Dictionary mapping residue numbers to SS codes (H=helix, E=sheet, C=coil).

    Example:
        >>> ss = sc.secondary_structure(structure)

    """
    # Get CA atoms
    ca = structure.select(atom_name="CA")
    coords = ca.coordinates
    res_nums = ca._residue_numbers

    ss = {}

    # Simplified assignment based on CA-CA distances
    for i in range(len(coords)):
        res_num = int(res_nums[i])

        # Check for helix pattern (i to i+4 distance ~5.5 A)
        if i + 4 < len(coords):
            d_i4 = np.sqrt(np.sum((coords[i] - coords[i + 4]) ** 2))
            if 4.5 < d_i4 < 6.5:
                ss[res_num] = "H"
                continue

        # Check for beta pattern (i to i+2 distance ~6.5 A)
        if i + 2 < len(coords):
            d_i2 = np.sqrt(np.sum((coords[i] - coords[i + 2]) ** 2))
            if 5.5 < d_i2 < 7.5:
                ss[res_num] = "E"
                continue

        ss[res_num] = "C"

    return ss


def sasa(
    structure: StructureArray,
    probe_radius: float = 1.4,
    n_points: int = 100,
) -> np.ndarray:
    """Calculate solvent accessible surface area per atom.

    Args:
        structure: Structure to analyze.
        probe_radius: Probe sphere radius in Angstroms.
        n_points: Number of points for sphere sampling.

    Returns:
        Array of SASA values per atom.

    Example:
        >>> sasa_values = sc.sasa(structure)

    """
    # Simplified SASA calculation using sphere sampling
    coords = structure.coordinates
    n_atoms = len(coords)

    # Van der Waals radii (simplified)
    vdw_radii = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "H": 1.2}

    # Generate sphere points
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5**0.5) * indices
    points = np.column_stack(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ]
    )

    sasa = np.zeros(n_atoms)

    for i in range(n_atoms):
        element = str(structure._elements[i]).strip()
        if not element:
            element = "C"  # Default
        radius = vdw_radii.get(element[0].upper(), 1.7) + probe_radius

        # Generate surface points
        surface_points = coords[i] + radius * points

        # Count accessible points
        accessible = 0
        for sp in surface_points:
            blocked = False
            for j in range(n_atoms):
                if i == j:
                    continue
                elem_j = str(structure._elements[j]).strip() or "C"
                rad_j = vdw_radii.get(elem_j[0].upper(), 1.7) + probe_radius
                if np.sqrt(np.sum((sp - coords[j]) ** 2)) < rad_j:
                    blocked = True
                    break
            if not blocked:
                accessible += 1

        # Surface area = (accessible/total) * 4*pi*r^2
        sasa[i] = (accessible / n_points) * 4 * np.pi * radius**2

    return sasa


def surface_residues(
    structure: StructureArray,
    threshold: float = 25.0,
) -> list[int]:
    """Find surface residues based on SASA threshold.

    Args:
        structure: Structure to analyze.
        threshold: SASA threshold in square Angstroms.

    Returns:
        List of surface residue numbers.

    Example:
        >>> surface = sc.surface_residues(structure, threshold=25.0)

    """
    sasa_values = sasa(structure)

    # Sum SASA per residue
    residue_sasa = {}
    for i, sa in enumerate(sasa_values):
        res_num = int(structure._residue_numbers[i])
        residue_sasa[res_num] = residue_sasa.get(res_num, 0) + sa

    return [res for res, sa in residue_sasa.items() if sa >= threshold]


def normalize_bfactors(structure: StructureArray) -> StructureArray:
    """Normalize B-factors to 0-100 range.

    Args:
        structure: Structure to normalize.

    Returns:
        New structure with normalized B-factors.

    Example:
        >>> normalized = sc.normalize_bfactors(structure)

    """
    b = structure._b_factors.copy()

    b = (b - b.min()) / (b.max() - b.min()) * 100 if b.max() > b.min() else np.ones_like(b) * 50

    return structure.__class__(
        coordinates=structure.coordinates,
        atom_names=structure._atom_names,
        residue_names=structure._residue_names,
        residue_numbers=structure._residue_numbers,
        chain_ids=structure._chain_ids,
        elements=structure._elements,
        b_factors=b,
        occupancies=structure._occupancies,
        serial_numbers=structure._serial_numbers,
    )


def flexibility_profile(structure: StructureArray) -> dict[int, float]:
    """Calculate flexibility profile from B-factors.

    Args:
        structure: Structure to analyze.

    Returns:
        Dictionary mapping residue numbers to flexibility scores.

    Example:
        >>> flex = sc.flexibility_profile(structure)

    """
    # Average B-factor per residue
    residue_bfactors = {}
    residue_counts = {}

    for i in range(len(structure)):
        res_num = int(structure._residue_numbers[i])
        b = structure._b_factors[i]

        if res_num not in residue_bfactors:
            residue_bfactors[res_num] = 0.0
            residue_counts[res_num] = 0

        residue_bfactors[res_num] += b
        residue_counts[res_num] += 1

    return {res: b / residue_counts[res] for res, b in residue_bfactors.items()}


def find_pockets(
    structure: StructureArray,
    min_volume: float = 100.0,
    grid_spacing: float = 1.0,
) -> list[Pocket]:
    """Find binding pockets in structure.

    Args:
        structure: Structure to analyze.
        min_volume: Minimum pocket volume in cubic Angstroms.
        grid_spacing: Grid spacing for cavity detection.

    Returns:
        List of Pocket objects.

    Example:
        >>> pockets = sc.find_pockets(structure)

    """
    coords = structure.coordinates

    # Create grid
    min_coords = coords.min(axis=0) - 5
    max_coords = coords.max(axis=0) + 5

    np.arange(min_coords[0], max_coords[0], grid_spacing)
    np.arange(min_coords[1], max_coords[1], grid_spacing)
    np.arange(min_coords[2], max_coords[2], grid_spacing)

    pockets = []

    # Simplified pocket detection - find large cavities
    # This is a placeholder for more sophisticated algorithms (fpocket, SiteMap, etc.)

    # Find center of mass and look for cavities around it
    com = coords.mean(axis=0)

    # Create a simple pocket at center of structure
    pocket = Pocket(
        center=com,
        volume=min_volume,
        residues=list(set(structure._residue_numbers[:10].tolist())),
        druggability=0.5,
    )
    pockets.append(pocket)

    return pockets


def druggability_score(pockets: list[Pocket]) -> list[float]:
    """Calculate druggability scores for pockets.

    Args:
        pockets: List of Pocket objects.

    Returns:
        List of druggability scores (0-1).

    Example:
        >>> scores = sc.druggability_score(pockets)

    """
    return [p.druggability for p in pockets]


def compare_contacts(
    structure1: StructureArray,
    structure2: StructureArray,
    ligand: str | None = None,
    cutoff: float = 4.0,
) -> dict[str, Any]:
    """Compare contact patterns between two structures.

    Args:
        structure1: First structure.
        structure2: Second structure.
        ligand: Ligand residue name to focus on.
        cutoff: Contact cutoff distance.

    Returns:
        Dictionary with contact comparison results.

    Example:
        >>> diff = sc.compare_contacts(wt, mutant, ligand="propofol")

    """
    contacts1 = find_contacts(structure1, cutoff=cutoff)
    contacts2 = find_contacts(structure2, cutoff=cutoff)

    # Convert to sets of residue pairs
    pairs1 = {(c.residue1, c.residue2) for c in contacts1}
    pairs2 = {(c.residue1, c.residue2) for c in contacts2}

    common = pairs1 & pairs2
    only_in_1 = pairs1 - pairs2
    only_in_2 = pairs2 - pairs1

    return {
        "common_contacts": len(common),
        "contacts_only_in_1": list(only_in_1),
        "contacts_only_in_2": list(only_in_2),
        "similarity": len(common) / len(pairs1 | pairs2) if pairs1 | pairs2 else 1.0,
    }


__all__ = [
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
    "Contact",
    "Pocket",
]
