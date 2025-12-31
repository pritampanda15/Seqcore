"""Structure array for 3D molecular structures.

Provides efficient storage and operations for protein/nucleic acid structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Atom:
    """Represents a single atom in a structure."""

    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_factor: float = 0.0
    element: str = ""


class StructureArray:
    """Efficient array for 3D molecular structures.

    Stores atomic coordinates and metadata in structured numpy arrays
    for fast vectorized operations.

    Example:
        >>> structure = sc.StructureArray.from_pdb("protein.pdb")
        >>> print(structure.chains)  # ['A', 'B', 'C']
        >>> coords = structure.coordinates  # (n_atoms, 3) array

    """

    def __init__(
        self,
        coordinates: np.ndarray,
        atom_names: np.ndarray,
        residue_names: np.ndarray,
        residue_numbers: np.ndarray,
        chain_ids: np.ndarray,
        elements: np.ndarray | None = None,
        b_factors: np.ndarray | None = None,
        occupancies: np.ndarray | None = None,
        serial_numbers: np.ndarray | None = None,
    ):
        """Initialize a structure array.

        Args:
            coordinates: (N, 3) array of atomic coordinates.
            atom_names: Array of atom names (CA, CB, N, etc.).
            residue_names: Array of residue names (ALA, GLY, etc.).
            residue_numbers: Array of residue numbers.
            chain_ids: Array of chain identifiers.
            elements: Array of element symbols.
            b_factors: Array of B-factors (temperature factors).
            occupancies: Array of occupancies.
            serial_numbers: Array of atom serial numbers.

        """
        self._coordinates = np.asarray(coordinates, dtype=np.float64)
        self._atom_names = np.asarray(atom_names, dtype="U4")
        self._residue_names = np.asarray(residue_names, dtype="U4")
        self._residue_numbers = np.asarray(residue_numbers, dtype=np.int32)
        self._chain_ids = np.asarray(chain_ids, dtype="U1")
        self._elements = (
            np.asarray(elements, dtype="U2")
            if elements is not None
            else np.array([""] * len(coordinates), dtype="U2")
        )
        self._b_factors = (
            np.asarray(b_factors, dtype=np.float64)
            if b_factors is not None
            else np.zeros(len(coordinates))
        )
        self._occupancies = (
            np.asarray(occupancies, dtype=np.float64)
            if occupancies is not None
            else np.ones(len(coordinates))
        )
        self._serial_numbers = (
            np.asarray(serial_numbers, dtype=np.int32)
            if serial_numbers is not None
            else np.arange(1, len(coordinates) + 1)
        )

    @classmethod
    def from_pdb(cls, filepath: str) -> StructureArray:
        """Load structure from PDB file.

        Args:
            filepath: Path to PDB file.

        Returns:
            StructureArray instance.

        Example:
            >>> structure = sc.StructureArray.from_pdb("1asc.pdb")

        """
        coords = []
        atom_names = []
        residue_names = []
        residue_numbers = []
        chain_ids = []
        elements = []
        b_factors = []
        occupancies = []
        serial_numbers = []

        with open(filepath) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    serial_numbers.append(int(line[6:11].strip()))
                    atom_names.append(line[12:16].strip())
                    residue_names.append(line[17:20].strip())
                    chain_ids.append(line[21].strip() or "A")
                    residue_numbers.append(int(line[22:26].strip()))
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                    occupancies.append(
                        float(line[54:60].strip())
                        if len(line) > 54 and line[54:60].strip()
                        else 1.0
                    )
                    b_factors.append(
                        float(line[60:66].strip())
                        if len(line) > 60 and line[60:66].strip()
                        else 0.0
                    )
                    elements.append(line[76:78].strip() if len(line) > 76 else "")

        return cls(
            coordinates=np.array(coords),
            atom_names=np.array(atom_names),
            residue_names=np.array(residue_names),
            residue_numbers=np.array(residue_numbers),
            chain_ids=np.array(chain_ids),
            elements=np.array(elements),
            b_factors=np.array(b_factors),
            occupancies=np.array(occupancies),
            serial_numbers=np.array(serial_numbers),
        )

    @classmethod
    def from_mmcif(cls, filepath: str) -> StructureArray:
        """Load structure from mmCIF file.

        Args:
            filepath: Path to mmCIF file.

        Returns:
            StructureArray instance.

        """
        coords = []
        atom_names = []
        residue_names = []
        residue_numbers = []
        chain_ids = []
        elements = []
        b_factors = []
        occupancies = []
        serial_numbers = []

        in_atom_site = False
        columns = {}
        col_names = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()

                if line.startswith("_atom_site."):
                    in_atom_site = True
                    col_name = line.split(".")[1].split()[0]
                    col_names.append(col_name)
                    columns[col_name] = len(col_names) - 1
                elif (
                    in_atom_site and line and not line.startswith("_") and not line.startswith("#")
                ):
                    if line.startswith("loop_"):
                        continue

                    parts = line.split()
                    if len(parts) < len(col_names):
                        continue

                    try:
                        serial_numbers.append(int(parts[columns.get("id", 0)]))
                        atom_names.append(
                            parts[columns.get("label_atom_id", columns.get("auth_atom_id", 1))]
                        )
                        residue_names.append(
                            parts[columns.get("label_comp_id", columns.get("auth_comp_id", 2))]
                        )
                        chain_ids.append(
                            parts[columns.get("label_asym_id", columns.get("auth_asym_id", 3))]
                        )
                        residue_numbers.append(
                            int(parts[columns.get("label_seq_id", columns.get("auth_seq_id", 4))])
                        )
                        x = float(parts[columns.get("Cartn_x", 5)])
                        y = float(parts[columns.get("Cartn_y", 6)])
                        z = float(parts[columns.get("Cartn_z", 7)])
                        coords.append([x, y, z])
                        occupancies.append(
                            float(parts[columns.get("occupancy", 8)])
                            if "occupancy" in columns
                            else 1.0
                        )
                        b_factors.append(
                            float(parts[columns.get("B_iso_or_equiv", 9)])
                            if "B_iso_or_equiv" in columns
                            else 0.0
                        )
                        elements.append(
                            parts[columns.get("type_symbol", 10)]
                            if "type_symbol" in columns
                            else ""
                        )
                    except (ValueError, IndexError):
                        continue
                elif in_atom_site and (
                    line.startswith("#")
                    or line.startswith("_")
                    and not line.startswith("_atom_site")
                ):
                    in_atom_site = False

        if not coords:
            raise ValueError(f"No atomic coordinates found in {filepath}")

        return cls(
            coordinates=np.array(coords),
            atom_names=np.array(atom_names),
            residue_names=np.array(residue_names),
            residue_numbers=np.array(residue_numbers),
            chain_ids=np.array(chain_ids),
            elements=np.array(elements),
            b_factors=np.array(b_factors),
            occupancies=np.array(occupancies),
            serial_numbers=np.array(serial_numbers),
        )

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._coordinates)

    def __repr__(self) -> str:
        return (
            f"StructureArray(atoms={len(self)}, chains={self.chains}, residues={self.n_residues})"
        )

    @property
    def coordinates(self) -> np.ndarray:
        """Return (N, 3) coordinate array."""
        return self._coordinates

    @property
    def atoms(self) -> np.ndarray:
        """Return structured array of atom data."""
        dtype = [
            ("serial", np.int32),
            ("name", "U4"),
            ("residue_name", "U4"),
            ("chain_id", "U1"),
            ("residue_number", np.int32),
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("occupancy", np.float64),
            ("b_factor", np.float64),
            ("element", "U2"),
        ]
        arr = np.zeros(len(self), dtype=dtype)
        arr["serial"] = self._serial_numbers
        arr["name"] = self._atom_names
        arr["residue_name"] = self._residue_names
        arr["chain_id"] = self._chain_ids
        arr["residue_number"] = self._residue_numbers
        arr["x"] = self._coordinates[:, 0]
        arr["y"] = self._coordinates[:, 1]
        arr["z"] = self._coordinates[:, 2]
        arr["occupancy"] = self._occupancies
        arr["b_factor"] = self._b_factors
        arr["element"] = self._elements
        return arr

    @property
    def chains(self) -> list[str]:
        """Return list of unique chain IDs."""
        return list(np.unique(self._chain_ids))

    @property
    def residues(self) -> np.ndarray:
        """Return unique residue identifiers."""
        # Combine chain and residue number for unique residues
        unique_res = np.unique(
            np.array(
                list(zip(self._chain_ids, self._residue_numbers, self._residue_names)),
                dtype=[("chain", "U1"), ("resnum", np.int32), ("resname", "U4")],
            )
        )
        return unique_res

    @property
    def n_residues(self) -> int:
        """Return number of residues."""
        return len(self.residues)

    @property
    def b_factors(self) -> np.ndarray:
        """Return B-factor array."""
        return self._b_factors

    def select(
        self,
        chain: str | None = None,
        residue: int | str | None = None,
        atom_name: str | None = None,
        element: str | None = None,
    ) -> StructureArray:
        """Select subset of atoms.

        Args:
            chain: Chain ID to select.
            residue: Residue number or name to select.
            atom_name: Atom name to select (CA, CB, etc.).
            element: Element symbol to select.

        Returns:
            New StructureArray with selected atoms.

        Example:
            >>> backbone = structure.select(atom_name="CA")
            >>> chain_a = structure.select(chain="A")

        """
        mask = np.ones(len(self), dtype=bool)

        if chain is not None:
            mask &= self._chain_ids == chain

        if residue is not None:
            if isinstance(residue, int):
                mask &= self._residue_numbers == residue
            else:
                mask &= self._residue_names == residue

        if atom_name is not None:
            mask &= self._atom_names == atom_name

        if element is not None:
            mask &= self._elements == element

        return StructureArray(
            coordinates=self._coordinates[mask],
            atom_names=self._atom_names[mask],
            residue_names=self._residue_names[mask],
            residue_numbers=self._residue_numbers[mask],
            chain_ids=self._chain_ids[mask],
            elements=self._elements[mask],
            b_factors=self._b_factors[mask],
            occupancies=self._occupancies[mask],
            serial_numbers=self._serial_numbers[mask],
        )

    def to_pdb(self, filepath: str):
        """Write structure to PDB file.

        Args:
            filepath: Output file path.

        """
        with open(filepath, "w") as f:
            for i in range(len(self)):
                record = "ATOM" if self._residue_names[i] not in ("HOH", "WAT") else "HETATM"
                line = (
                    f"{record:6s}{self._serial_numbers[i]:5d} "
                    f"{self._atom_names[i]:4s} "
                    f"{self._residue_names[i]:3s} "
                    f"{self._chain_ids[i]:1s}"
                    f"{self._residue_numbers[i]:4d}    "
                    f"{self._coordinates[i, 0]:8.3f}"
                    f"{self._coordinates[i, 1]:8.3f}"
                    f"{self._coordinates[i, 2]:8.3f}"
                    f"{self._occupancies[i]:6.2f}"
                    f"{self._b_factors[i]:6.2f}          "
                    f"{self._elements[i]:2s}\n"
                )
                f.write(line)
            f.write("END\n")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        Returns:
            DataFrame with atomic data.

        """
        import pandas as pd

        return pd.DataFrame(
            {
                "serial": self._serial_numbers,
                "atom_name": self._atom_names,
                "residue_name": self._residue_names,
                "chain_id": self._chain_ids,
                "residue_number": self._residue_numbers,
                "x": self._coordinates[:, 0],
                "y": self._coordinates[:, 1],
                "z": self._coordinates[:, 2],
                "occupancy": self._occupancies,
                "b_factor": self._b_factors,
                "element": self._elements,
            }
        )

    def to_mdanalysis(self):
        """Convert to MDAnalysis Universe.

        Returns:
            MDAnalysis Universe object.

        """
        try:
            import MDAnalysis as mda

            n_atoms = len(self)
            universe = mda.Universe.empty(n_atoms, trajectory=True)
            universe.atoms.positions = self._coordinates
            universe.add_TopologyAttr("names", self._atom_names)
            universe.add_TopologyAttr("resnames", self._residue_names)
            universe.add_TopologyAttr("resids", self._residue_numbers)
            universe.add_TopologyAttr("chainIDs", self._chain_ids)

            return universe
        except ImportError as err:
            raise ImportError(
                "MDAnalysis is required for this operation. Install with: pip install MDAnalysis"
            ) from err

    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        # Simplified - uses uniform weights
        return np.mean(self._coordinates, axis=0)

    def centroid(self) -> np.ndarray:
        """Calculate geometric centroid."""
        return np.mean(self._coordinates, axis=0)
