#!/usr/bin/env python3
"""Benchmark Seqcore's domain modules: molecules, structure, phylogenetics, population.

These are the parts of the library the README advertises but the main benchmark
suite does not cover. Each operation is compared against an appropriate
reference implementation where one exists:

  molecules      -> RDKit directly (Seqcore wraps it, so this measures overhead)
  structure      -> SciPy (cdist, cKDTree)
  phylogenetics  -> Biopython's DistanceTreeConstructor
  population     -> a straightforward NumPy implementation

Where no fair reference exists the absolute time and its scaling are reported,
which is still the useful number for deciding whether an operation is usable at
a realistic input size.

Usage::

    python benchmarks/benchmark_modules.py
    python benchmarks/benchmark_modules.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

SEED = 20260825


def timeit(fn, repeats=3):
    """Best-of-N wall clock; returns (seconds, error_string)."""
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # a failure is a result worth recording
            return None, f"{type(exc).__name__}: {exc}"[:90]
        best = min(best, time.perf_counter() - start)
    return best, None


def row(records, module, op, size, seqcore_t, ref_t, ref_name, err=None):
    rec = {
        "module": module,
        "operation": op,
        "size": size,
        "seqcore": seqcore_t,
        "reference": ref_t,
        "reference_name": ref_name,
        "error": err,
    }
    if seqcore_t and ref_t:
        rec["speedup"] = ref_t / seqcore_t
    records.append(rec)

    if err:
        print(f"    {op:26s} n={size:<6} FAILED  {err}")
        return
    ratio = ""
    if seqcore_t and ref_t:
        ratio = f"| {ref_name} {ref_t:8.4f}s  ->  {ref_t/seqcore_t:7.2f}x"
    elif ref_name:
        ratio = f"| {ref_name}: n/a"
    print(f"    {op:26s} n={size:<6} {seqcore_t:9.4f}s {ratio}")


# ------------------------------------------------------------------ molecules
SMILES_POOL = [
    "CCO",
    "CCN",
    "c1ccccc1",
    "CC(=O)O",
    "CCCCC",
    "c1ccncc1",
    "CC(C)O",
    "CCOC",
    "CC(=O)Nc1ccc(O)cc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
    "COc1cc2c(cc1OC)CCN(C)C2",
    "OC(=O)c1ccccc1O",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",
]


def bench_molecules(records, sizes):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.DataStructs import BulkTanimotoSimilarity

    import seqcore.molecules as scm

    RDLogger.DisableLog("rdApp.*")

    print("\n[molecules]  reference: RDKit directly")
    rng = random.Random(SEED)
    for n in sizes:
        smis = [rng.choice(SMILES_POOL) for _ in range(n)]
        mols = [scm.Molecule.from_smiles(s) for s in smis]
        rd = [Chem.MolFromSmiles(s) for s in smis]

        t, e = timeit(lambda m=mols: scm.molecular_weight(m))
        r, _ = timeit(lambda r=rd: [Descriptors.MolWt(m) for m in r])
        row(records, "molecules", "molecular_weight", n, t, r, "RDKit", e)

        t, e = timeit(lambda m=mols: scm.morgan_fingerprint(m))
        r, _ = timeit(lambda r=rd: [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in r])
        row(records, "molecules", "morgan_fingerprint", n, t, r, "RDKit", e)

        fps = scm.morgan_fingerprint(mols)
        rdfps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in rd]
        t, e = timeit(lambda f=fps: scm.tanimoto_similarity(f))
        r, _ = timeit(lambda ref=rdfps: [BulkTanimotoSimilarity(f, ref) for f in ref])
        row(records, "molecules", "tanimoto_similarity", n, t, r, "RDKit bulk", e)


# ------------------------------------------------------------------ structure
def synthetic_pdb(n_atoms: int, path: Path) -> Path:
    """Write a synthetic all-CA PDB with plausible spacing."""
    rng = np.random.default_rng(SEED)
    xyz = np.cumsum(rng.normal(0, 2.0, size=(n_atoms, 3)), axis=0)
    lines = []
    for i, (x, y, z) in enumerate(xyz, start=1):
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    " f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def bench_structure(records, sizes, tmp: Path):
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist

    import seqcore as sc

    print("\n[structure]  reference: SciPy")
    for n in sizes:
        pdb = synthetic_pdb(n, tmp / f"synth_{n}.pdb")
        st = sc.read(str(pdb))
        xyz = np.array([[a["x"], a["y"], a["z"]] for a in st.atoms], dtype=float)

        t, e = timeit(lambda x=st: sc.distance_matrix(x))
        r, _ = timeit(lambda c=xyz: cdist(c, c))
        row(records, "structure", "distance_matrix", n, t, r, "scipy cdist", e)

        t, e = timeit(lambda x=st: sc.find_contacts(x, cutoff=8.0))
        r, _ = timeit(lambda c=xyz: cKDTree(c).query_pairs(8.0))
        row(records, "structure", "find_contacts", n, t, r, "scipy cKDTree", e)

        # No fair drop-in reference; absolute time and scaling are the point.
        t, e = timeit(lambda x=st: sc.sasa(x), repeats=1)
        row(records, "structure", "sasa", n, t, None, "", e)


# -------------------------------------------------------------- phylogenetics
def bench_phylogenetics(records, sizes):
    from Bio import Phylo  # noqa: F401
    from Bio.Align import MultipleSeqAlignment
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    import seqcore as sc

    print("\n[phylogenetics]  reference: Biopython DistanceTreeConstructor")
    rng = random.Random(SEED)
    for n in sizes:
        seqs = ["".join(rng.choices("ACGT", k=300)) for _ in range(n)]
        arr = sc.DNAArray(seqs)
        aln = MultipleSeqAlignment([SeqRecord(Seq(s), id=f"t{i}") for i, s in enumerate(seqs)])
        calc = DistanceCalculator("identity")

        for name, sc_fn, bio_method in (
            ("neighbor_joining", sc.neighbor_joining, "nj"),
            ("upgma", sc.upgma, "upgma"),
        ):
            t, e = timeit(lambda f=sc_fn, a=arr: f(a), repeats=1)

            def bio(m=bio_method, c=calc, a=aln):
                DistanceTreeConstructor(c, m).build_tree(a)

            r, _ = timeit(bio, repeats=1)
            row(records, "phylogenetics", name, n, t, r, "Biopython", e)


# ----------------------------------------------------------------- population
def synthetic_variants(n_variants: int, n_samples: int = 20) -> dict:
    """Build the variant dict shape Seqcore's VCF reader produces."""
    rng = random.Random(SEED)
    gts = ["0/0", "0/1", "1/1", "0|1", "./."]
    samples = []
    for _ in range(n_variants):
        samples.append({f"S{j}": {"GT": rng.choice(gts)} for j in range(n_samples)})
    return {
        "chrom": ["chr1"] * n_variants,
        "pos": list(range(1, n_variants + 1)),
        "id": [f"v{i}" for i in range(n_variants)],
        "ref": ["A"] * n_variants,
        "alt": [["G"]] * n_variants,
        "qual": [100.0] * n_variants,
        "filter": ["PASS"] * n_variants,
        "info": ["DP=100"] * n_variants,
        "samples": samples,
        "sample_names": [f"S{j}" for j in range(n_samples)],
    }


def numpy_allele_frequency(variants: dict) -> np.ndarray:
    """Vectorized reference: alt-allele frequency per variant."""
    names = variants["sample_names"]
    out = np.zeros(len(variants["samples"]), dtype=np.float64)
    for i, row_ in enumerate(variants["samples"]):
        alt = tot = 0
        for s in names:
            gt = row_[s].get("GT", "./.").replace("|", "/")
            for a in gt.split("/"):
                if a.isdigit():
                    tot += 1
                    alt += int(a) > 0
        out[i] = alt / tot if tot else 0.0
    return out


def bench_population(records, variant_sizes, seq_sizes):
    """Variant-based and sequence-based statistics scale on different axes."""
    import seqcore as sc

    print("\n[population]  reference: NumPy/pure-Python implementation")
    rng = random.Random(SEED)

    for n in variant_sizes:
        variants = synthetic_variants(n)
        t, e = timeit(lambda v=variants: sc.allele_frequency(v))
        r, _ = timeit(lambda v=variants: numpy_allele_frequency(v))
        row(records, "population", "allele_frequency", n, t, r, "NumPy ref", e)

    # These scale with the number of sequence pairs, so they need their own
    # size axis; holding the count fixed would report a meaningless exponent.
    for n in seq_sizes:
        seqs = sc.DNAArray(["".join(rng.choices("ACGT", k=400)) for _ in range(n)])
        t, e = timeit(lambda q=seqs: sc.nucleotide_diversity(q), repeats=1)
        row(records, "population", "nucleotide_diversity", n, t, None, "", e)

        t, e = timeit(lambda q=seqs: sc.tajimas_d(q), repeats=1)
        row(records, "population", "tajimas_d", n, t, None, "", e)


# --------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    if args.quick:
        mol_sizes, struct_sizes, phylo_sizes = [64], [200], [16]
        pop_sizes, popseq_sizes = [200], [20]
    else:
        mol_sizes = [100, 500, 2000]
        struct_sizes = [100, 250, 500, 1000]
        phylo_sizes = [16, 24, 32, 48]
        pop_sizes = [100, 500, 2000]
        popseq_sizes = [20, 40, 80, 160]

    print("=" * 78)
    print("SEQCORE DOMAIN MODULE BENCHMARK")
    print(f"{platform.platform()} | Python {platform.python_version()}")
    print("=" * 78)

    records: list[dict] = []
    tmp = Path(args.out).parent / "_tmp_structures"
    tmp.mkdir(parents=True, exist_ok=True)

    for name, fn, sizes in (
        ("molecules", bench_molecules, mol_sizes),
        ("structure", lambda r, s: bench_structure(r, s, tmp), struct_sizes),
        ("phylogenetics", bench_phylogenetics, phylo_sizes),
        ("population", lambda r, sz: bench_population(r, sz, popseq_sizes), pop_sizes),
    ):
        try:
            fn(records, sizes)
        except Exception as exc:
            print(f"\n[{name}]  SKIPPED: {type(exc).__name__}: {exc}")

    for f in tmp.glob("*.pdb"):
        f.unlink()
    tmp.rmdir()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "domain-modules",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "metric": "best-of-N wall clock (seconds)",
        "results": records,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"modules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
