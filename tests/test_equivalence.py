"""Differential tests pinning the vectorized fast paths to reference semantics.

Each optimized routine in Seqcore has a straightforward scalar definition. These
tests generate randomized inputs -- including ragged batches, empty sequences,
ambiguous ``N`` bases and every reading frame -- and assert that the vectorized
implementation agrees with that definition exactly. They are the guard that lets
the fast paths be rewritten without silently changing results.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

import seqcore as sc
from seqcore.alignment import _dp_fill
from seqcore.core.operations import CODON_TABLE

SEED = 20260825


def random_batch(rng, max_seqs=6, max_len=60, alphabet="ACGTN"):
    """Generate a ragged batch of sequences, possibly containing empty strings."""
    return [
        "".join(rng.choices(alphabet, k=rng.randint(0, max_len)))
        for _ in range(rng.randint(1, max_seqs))
    ]


# --------------------------------------------------------------------------
# Reference implementations (deliberately naive)
# --------------------------------------------------------------------------


def ref_gc(seq: str) -> float:
    """Percent G+C, the textbook definition."""
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq) * 100


def ref_revcomp(seq: str) -> str:
    """Reverse complement by explicit per-base substitution."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp[c] for c in reversed(seq))


def ref_translate(seq: str, frame: int = 0, to_stop: bool = False) -> str:
    """Translate codon by codon through the standard genetic code dict."""
    seq = seq.upper().replace("U", "T")
    out = []
    for i in range((len(seq) - frame) // 3):
        codon = seq[frame + i * 3 : frame + i * 3 + 3]
        aa = CODON_TABLE.get(codon, "X")
        if to_stop and aa == "*":
            break
        out.append(aa)
    return "".join(out)


def ref_kmers(seqs: list[str], k: int) -> Counter:
    """Count k-mers by slicing strings, one window at a time."""
    counts: Counter = Counter()
    for s in seqs:
        for i in range(len(s) - k + 1):
            counts[s[i : i + k]] += 1
    return counts


def ref_dp(s1: str, s2: str, match: int, mismatch: int, gap: int, local: bool) -> np.ndarray:
    """Fill the DP matrix with the plain scalar recurrence."""
    m, n = len(s1), len(s2)
    score = np.zeros((m + 1, n + 1), dtype=np.float32)
    if not local:
        score[0, :] = np.arange(n + 1) * gap
        score[:, 0] = np.arange(m + 1) * gap
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sub = match if s1[i - 1] == s2[j - 1] else mismatch
            best = max(score[i - 1, j - 1] + sub, score[i - 1, j] + gap, score[i, j - 1] + gap)
            score[i, j] = max(0.0, best) if local else best
    return score


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_encode_decode_round_trip():
    """Batched encoding must reproduce the input strings exactly."""
    rng = random.Random(SEED)
    for _ in range(200):
        seqs = random_batch(rng)
        arr = sc.DNAArray(seqs)
        assert [arr[i] for i in range(len(seqs))] == seqs


def test_gc_content_matches_reference():
    """Vectorized GC content must equal the per-sequence definition."""
    rng = random.Random(SEED + 1)
    for _ in range(200):
        seqs = [s for s in random_batch(rng) if s]
        if not seqs:
            continue
        got = sc.gc_content(sc.DNAArray(seqs))
        expected = np.array([ref_gc(s) for s in seqs], dtype=np.float32)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-3)


def test_reverse_complement_matches_reference():
    """Gather-based reverse complement must equal per-base substitution."""
    rng = random.Random(SEED + 2)
    for _ in range(200):
        seqs = random_batch(rng)
        rc = sc.reverse_complement(sc.DNAArray(seqs))
        assert [rc[i] for i in range(len(seqs))] == [ref_revcomp(s) for s in seqs]


@pytest.mark.parametrize("frame", [0, 1, 2])
@pytest.mark.parametrize("to_stop", [False, True])
def test_translate_matches_reference(frame, to_stop):
    """The table-driven fast path must agree for every frame and stop policy."""
    rng = random.Random(SEED + 3 + frame)
    for _ in range(60):
        seqs = random_batch(rng, max_len=45)
        got = sc.translate(sc.DNAArray(seqs), frame=frame, to_stop=to_stop)
        expected = [ref_translate(s, frame, to_stop) for s in seqs]
        assert [got[i] for i in range(len(seqs))] == expected


@pytest.mark.parametrize("k", [1, 2, 3, 5, 8, 12, 21, 30])
def test_count_kmers_matches_reference(k):
    """Both sides of the 4^k dispatch guard must produce identical counts."""
    rng = random.Random(SEED + 4 + k)
    for _ in range(40):
        seqs = random_batch(rng, max_len=60)
        got = sc.count_kmers(sc.DNAArray(seqs), k=k)
        assert got == dict(ref_kmers(seqs, k))


def test_count_kmers_normalize_sums_to_one():
    """Normalized k-mer frequencies form a probability distribution."""
    counts = sc.count_kmers(sc.DNAArray(["ACGTACGT", "TTTTAAAA"]), k=3, normalize=True)
    assert pytest.approx(sum(counts.values()), abs=1e-9) == 1.0


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("scoring", [(2, -1, -2), (1, -3, -5), (5, -4, -1)])
def test_wavefront_dp_matches_scalar_recurrence(local, scoring):
    """The anti-diagonal fill must reproduce the scalar matrix bit for bit."""
    match, mismatch, gap = scoring
    rng = random.Random(SEED + 5)
    for _ in range(25):
        s1 = "".join(rng.choices("ACGT", k=rng.randint(0, 35)))
        s2 = "".join(rng.choices("ACGT", k=rng.randint(0, 35)))
        got = _dp_fill(s1, s2, match, mismatch, gap, local)
        expected = ref_dp(s1, s2, match, mismatch, gap, local)
        np.testing.assert_array_equal(got, expected)


def test_alignment_score_matches_scalar_recurrence():
    """End-to-end global alignment scores match the reference matrix corner.

    ``align`` applies a linear gap penalty equal to ``gap_open``; ``gap_extend``
    is accepted but not yet used, so the reference uses the same single penalty.
    """
    rng = random.Random(SEED + 6)
    for _ in range(25):
        s1 = "".join(rng.choices("ACGT", k=rng.randint(5, 40)))
        s2 = "".join(rng.choices("ACGT", k=rng.randint(5, 40)))
        result = sc.align(s1, s2, gap_open=-5)
        expected = ref_dp(s1, s2, 2, -1, -5, local=False)[len(s1), len(s2)]
        assert result.score == pytest.approx(float(expected), abs=1e-4)


def test_align_gap_extend_is_documented_as_unused():
    """Guard the known limitation: gap_extend currently has no effect."""
    s1, s2 = "ACGTACGTAC", "ACGTTTACGT"
    assert sc.align(s1, s2, gap_extend=-1).score == sc.align(s1, s2, gap_extend=-99).score


@pytest.mark.parametrize("n_seqs", [1, 2, 5, 12])
def test_pairwise_hamming_matches_per_pair_reference(n_seqs):
    """The vectorized all-pairs Hamming must equal the per-pair definition."""
    from seqcore.alignment import _hamming_distance, pairwise_distance

    rng = random.Random(SEED + 7 + n_seqs)
    for _ in range(20):
        length = rng.randint(0, 60)
        seqs = ["".join(rng.choices("ACGTN", k=length)) for _ in range(n_seqs)]
        got = pairwise_distance(seqs, metric="hamming")
        expected = np.zeros((n_seqs, n_seqs), dtype=np.float32)
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                expected[i, j] = expected[j, i] = _hamming_distance(seqs[i], seqs[j])
        np.testing.assert_array_equal(got, expected)


def test_pairwise_hamming_rejects_ragged_input():
    """Hamming is undefined for unequal lengths and must say so."""
    from seqcore.alignment import pairwise_distance

    with pytest.raises(ValueError, match="equal length"):
        pairwise_distance(["AC", "ACG"], metric="hamming")


def test_tree_metric_default_is_unchanged():
    """Adding the metric option must not alter the default tree."""
    rng = random.Random(SEED + 8)
    seqs = sc.DNAArray(["".join(rng.choices("ACGT", k=120)) for _ in range(8)])
    assert sc.neighbor_joining(seqs).newick() == (
        sc.neighbor_joining(seqs, metric="identity").newick()
    )
    assert sc.upgma(seqs).newick() == sc.upgma(seqs, metric="identity").newick()


def test_tanimoto_matches_pairwise_definition():
    """Vectorized Tanimoto must equal the per-pair intersection/union form."""
    from seqcore.molecules import tanimoto_similarity

    rng = np.random.default_rng(SEED)
    for trial in range(30):
        n1, n2 = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        width = int(rng.integers(1, 40))
        a = (rng.random((n1, width)) < 0.3).astype(np.float64)
        b = (rng.random((n2, width)) < 0.3).astype(np.float64)
        if trial % 5 == 0:
            a[0] = 0  # exercise the empty-union branch

        got = tanimoto_similarity(a, b)
        expected = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                inter = np.sum(np.logical_and(a[i], b[j]))
                union = np.sum(np.logical_or(a[i], b[j]))
                expected[i, j] = inter / union if union > 0 else 0.0
        np.testing.assert_allclose(got, expected)


def test_tanimoto_accepts_one_dimensional_input():
    """A single fingerprint is treated as a batch of one."""
    from seqcore.molecules import tanimoto_similarity

    sim = tanimoto_similarity(np.array([1, 0, 1, 1]), np.array([1, 1, 0, 1]))
    assert sim.shape == (1, 1)
    assert sim[0, 0] == pytest.approx(2 / 4)


def test_morgan_fingerprint_matches_rdkit():
    """The fast bit-vector conversion must reproduce RDKit's own bits."""
    pytest.importorskip("rdkit")
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    from seqcore.molecules import Molecule, morgan_fingerprint

    RDLogger.DisableLog("rdApp.*")
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]
    got = morgan_fingerprint([Molecule.from_smiles(s) for s in smiles])

    expected = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        expected.append(
            np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            if mol is not None
            else np.zeros(2048)
        )
    np.testing.assert_array_equal(got, np.array(expected))


def test_molecule_caches_its_parsed_form():
    """to_rdkit parses once; repeated property calls reuse the same object."""
    pytest.importorskip("rdkit")
    from seqcore.molecules import Molecule

    mol = Molecule.from_smiles("CCO")
    assert mol.to_rdkit() is mol.to_rdkit()


def test_empty_and_degenerate_batches():
    """Degenerate inputs must not raise."""
    assert len(sc.DNAArray([])) == 0
    empty = sc.DNAArray(["", ""])
    assert list(sc.gc_content(empty)) == [0.0, 0.0]
    assert sc.count_kmers(empty, k=3) == {}
    assert [sc.translate(empty)[i] for i in range(2)] == ["", ""]
