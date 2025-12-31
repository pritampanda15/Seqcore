"""Phylogenetic tree construction and analysis.

Provides distance-based tree building methods and tree operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from seqcore.core.arrays import BioArray


@dataclass
class TreeNode:
    """Represents a node in a phylogenetic tree."""

    name: str = ""
    children: list[TreeNode] = field(default_factory=list)
    branch_length: float = 0.0
    parent: TreeNode | None = None

    def is_leaf(self) -> bool:
        """Check if node is a leaf."""
        return len(self.children) == 0

    def get_leaves(self) -> list[TreeNode]:
        """Get all leaf nodes under this node."""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def newick(self) -> str:
        """Generate Newick format string."""
        if self.is_leaf():
            return f"{self.name}:{self.branch_length:.6f}"
        children_str = ",".join(c.newick() for c in self.children)
        if self.name:
            return f"({children_str}){self.name}:{self.branch_length:.6f}"
        return f"({children_str}):{self.branch_length:.6f}"


@dataclass
class PhyloTree:
    """Phylogenetic tree."""

    root: TreeNode

    def newick(self) -> str:
        """Export tree as Newick format string.

        Returns:
            Newick format string.

        Example:
            >>> tree.newick()
            '((A:0.1,B:0.2):0.3,C:0.4);'

        """
        return self.root.newick() + ";"

    def get_leaves(self) -> list[str]:
        """Get all leaf names."""
        return [n.name for n in self.root.get_leaves()]

    def distance(self, leaf1: str, leaf2: str) -> float:
        """Calculate patristic distance between two leaves.

        Args:
            leaf1: First leaf name.
            leaf2: Second leaf name.

        Returns:
            Patristic distance.

        Example:
            >>> dist = tree.distance("Species_A", "Species_B")

        """
        # Find paths to both leaves
        path1 = self._find_path(self.root, leaf1, [])
        path2 = self._find_path(self.root, leaf2, [])

        if path1 is None or path2 is None:
            raise ValueError(f"Leaf not found: {leaf1 if path1 is None else leaf2}")

        # Find common ancestor
        common_idx = 0
        for i, (n1, n2) in enumerate(zip(path1, path2)):
            if n1 != n2:
                break
            common_idx = i

        # Sum distances from LCA to both leaves
        dist = 0.0
        for node in path1[common_idx + 1 :]:
            dist += node.branch_length
        for node in path2[common_idx + 1 :]:
            dist += node.branch_length

        return dist

    def _find_path(
        self, node: TreeNode, target: str, path: list[TreeNode]
    ) -> list[TreeNode] | None:
        """Find path from node to target leaf."""
        current_path = path + [node]

        if node.is_leaf() and node.name == target:
            return current_path

        for child in node.children:
            result = self._find_path(child, target, current_path)
            if result is not None:
                return result

        return None

    def lca(self, leaf1: str, leaf2: str) -> TreeNode:
        """Find last common ancestor of two leaves.

        Args:
            leaf1: First leaf name.
            leaf2: Second leaf name.

        Returns:
            LCA TreeNode.

        Example:
            >>> ancestor = tree.lca("Species_A", "Species_B")

        """
        path1 = self._find_path(self.root, leaf1, [])
        path2 = self._find_path(self.root, leaf2, [])

        if path1 is None or path2 is None:
            raise ValueError("Leaf not found")

        lca_node = self.root
        for n1, n2 in zip(path1, path2):
            if n1 == n2:
                lca_node = n1
            else:
                break

        return lca_node

    def prune(self, keep_leaves: list[str]) -> PhyloTree:
        """Prune tree to keep only specified leaves.

        Args:
            keep_leaves: List of leaf names to keep.

        Returns:
            Pruned PhyloTree.

        Example:
            >>> subtree = tree.prune(["A", "B", "C"])

        """
        new_root = self._prune_node(self.root, set(keep_leaves))
        if new_root is None:
            raise ValueError("No leaves to keep")
        return PhyloTree(root=new_root)

    def _prune_node(self, node: TreeNode, keep: set) -> TreeNode | None:
        """Recursively prune tree."""
        if node.is_leaf():
            if node.name in keep:
                return TreeNode(name=node.name, branch_length=node.branch_length)
            return None

        new_children = []
        for child in node.children:
            new_child = self._prune_node(child, keep)
            if new_child is not None:
                new_children.append(new_child)

        if not new_children:
            return None

        if len(new_children) == 1:
            # Collapse single-child nodes
            new_children[0].branch_length += node.branch_length
            return new_children[0]

        new_node = TreeNode(
            name=node.name,
            branch_length=node.branch_length,
            children=new_children,
        )
        for child in new_children:
            child.parent = new_node
        return new_node


def _compute_distance_matrix(
    sequences: BioArray | list[str],
    metric: str = "identity",
) -> tuple[np.ndarray, list[str]]:
    """Compute distance matrix from sequences."""
    from seqcore.alignment import pairwise_distance
    from seqcore.core.arrays import BioArray

    if isinstance(sequences, BioArray):
        seqs = sequences.sequences
        names = sequences.ids
    else:
        seqs = sequences
        names = [f"seq_{i}" for i in range(len(seqs))]

    dm = pairwise_distance(seqs, metric="identity")
    return dm, names


def neighbor_joining(
    sequences_or_matrix: BioArray | list[str] | np.ndarray,
    names: list[str] | None = None,
) -> PhyloTree:
    """Build tree using Neighbor-Joining algorithm.

    Args:
        sequences_or_matrix: Sequences or precomputed distance matrix.
        names: Leaf names (required if using distance matrix).

    Returns:
        PhyloTree object.

    Example:
        >>> tree = sc.neighbor_joining(sequences)
        >>> print(tree.newick())

    """
    if isinstance(sequences_or_matrix, np.ndarray):
        dm = sequences_or_matrix.copy()
        if names is None:
            names = [f"seq_{i}" for i in range(len(dm))]
    else:
        dm, names = _compute_distance_matrix(sequences_or_matrix)

    n = len(dm)
    nodes = [TreeNode(name=name) for name in names]
    active = list(range(n))

    while len(active) > 2:
        # Calculate Q matrix
        r = len(active)
        row_sums = np.zeros(r)
        for i, ai in enumerate(active):
            for j, aj in enumerate(active):
                row_sums[i] += dm[ai, aj]

        q = np.zeros((r, r))
        for i in range(r):
            for j in range(i + 1, r):
                ai, aj = active[i], active[j]
                q[i, j] = (r - 2) * dm[ai, aj] - row_sums[i] - row_sums[j]
                q[j, i] = q[i, j]

        # Find minimum Q
        min_val = float("inf")
        min_i, min_j = 0, 1
        for i in range(r):
            for j in range(i + 1, r):
                if q[i, j] < min_val:
                    min_val = q[i, j]
                    min_i, min_j = i, j

        # Join nodes
        ai, aj = active[min_i], active[min_j]

        # Calculate branch lengths
        dist_ij = dm[ai, aj]
        if r > 2:
            branch_i = 0.5 * dist_ij + (row_sums[min_i] - row_sums[min_j]) / (2 * (r - 2))
            branch_j = dist_ij - branch_i
        else:
            branch_i = branch_j = dist_ij / 2

        branch_i = max(0, branch_i)
        branch_j = max(0, branch_j)

        nodes[ai].branch_length = branch_i
        nodes[aj].branch_length = branch_j

        # Create new node
        new_node = TreeNode(children=[nodes[ai], nodes[aj]])
        nodes[ai].parent = new_node
        nodes[aj].parent = new_node

        # Update distance matrix
        new_idx = len(nodes)
        nodes.append(new_node)

        # Expand distance matrix
        new_dm = np.zeros((new_idx + 1, new_idx + 1))
        new_dm[:new_idx, :new_idx] = dm

        for k in active:
            if k != ai and k != aj:
                new_dist = 0.5 * (dm[ai, k] + dm[aj, k] - dm[ai, aj])
                new_dm[new_idx, k] = new_dist
                new_dm[k, new_idx] = new_dist

        dm = new_dm

        # Update active list
        active.remove(ai)
        active.remove(aj)
        active.append(new_idx)

    # Connect final two nodes
    if len(active) == 2:
        ai, aj = active
        dist = dm[ai, aj] / 2
        nodes[ai].branch_length = dist
        nodes[aj].branch_length = dist
        root = TreeNode(children=[nodes[ai], nodes[aj]])
        nodes[ai].parent = root
        nodes[aj].parent = root
    else:
        root = nodes[active[0]]

    return PhyloTree(root=root)


def upgma(
    sequences_or_matrix: BioArray | list[str] | np.ndarray,
    names: list[str] | None = None,
) -> PhyloTree:
    """Build tree using UPGMA algorithm.

    Args:
        sequences_or_matrix: Sequences or precomputed distance matrix.
        names: Leaf names (required if using distance matrix).

    Returns:
        PhyloTree object.

    Example:
        >>> tree = sc.upgma(sequences)

    """
    if isinstance(sequences_or_matrix, np.ndarray):
        dm = sequences_or_matrix.copy()
        if names is None:
            names = [f"seq_{i}" for i in range(len(dm))]
    else:
        dm, names = _compute_distance_matrix(sequences_or_matrix)

    n = len(dm)
    nodes = [TreeNode(name=name) for name in names]
    heights = [0.0] * n
    sizes = [1] * n
    active = list(range(n))

    while len(active) > 1:
        # Find minimum distance
        min_dist = float("inf")
        min_i, min_j = 0, 1

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ai, aj = active[i], active[j]
                if dm[ai, aj] < min_dist:
                    min_dist = dm[ai, aj]
                    min_i, min_j = i, j

        ai, aj = active[min_i], active[min_j]

        # New node height
        new_height = min_dist / 2

        # Set branch lengths
        nodes[ai].branch_length = new_height - heights[ai]
        nodes[aj].branch_length = new_height - heights[aj]

        # Create new node
        new_node = TreeNode(children=[nodes[ai], nodes[aj]])
        nodes[ai].parent = new_node
        nodes[aj].parent = new_node

        new_idx = len(nodes)
        nodes.append(new_node)
        heights.append(new_height)

        # Calculate new distances (UPGMA average)
        new_size = sizes[ai] + sizes[aj]
        sizes.append(new_size)

        new_dm = np.zeros((new_idx + 1, new_idx + 1))
        new_dm[:new_idx, :new_idx] = dm

        for k in active:
            if k != ai and k != aj:
                new_dist = (sizes[ai] * dm[ai, k] + sizes[aj] * dm[aj, k]) / new_size
                new_dm[new_idx, k] = new_dist
                new_dm[k, new_idx] = new_dist

        dm = new_dm

        # Update active list
        active.remove(ai)
        active.remove(aj)
        active.append(new_idx)

    return PhyloTree(root=nodes[active[0]])


def place_sequence(
    new_sequence: str,
    reference_tree: PhyloTree,
    reference_alignment: BioArray | list[str],
) -> dict[str, Any]:
    """Place a new sequence onto existing tree.

    Args:
        new_sequence: Sequence to place.
        reference_tree: Reference phylogenetic tree.
        reference_alignment: Aligned reference sequences.

    Returns:
        Dictionary with placement information.

    Example:
        >>> placement = sc.place_sequence(new_seq, tree, alignment)

    """
    from seqcore.alignment import align

    # Align to each reference sequence
    from seqcore.core.arrays import BioArray

    if isinstance(reference_alignment, BioArray):
        ref_seqs = reference_alignment.sequences
        ref_names = reference_alignment.ids
    else:
        ref_seqs = reference_alignment
        ref_names = [f"seq_{i}" for i in range(len(ref_seqs))]

    distances = []
    for ref_seq in ref_seqs:
        result = align(new_sequence, ref_seq)
        # Convert identity to distance
        dist = 1.0 - result.identity
        distances.append(dist)

    # Find closest reference
    min_idx = np.argmin(distances)
    closest_name = ref_names[min_idx]
    closest_dist = distances[min_idx]

    return {
        "closest_sequence": closest_name,
        "distance": float(closest_dist),
        "all_distances": {name: float(d) for name, d in zip(ref_names, distances)},
    }


__all__ = [
    "TreeNode",
    "PhyloTree",
    "neighbor_joining",
    "upgma",
    "place_sequence",
]
