# Seqcore
<p align="center">
  <a href="https://github.com/pritampanda15/Seqcore">
    <img src="https://raw.githubusercontent.com/pritampanda15/Seqcore/main/logo/seqcore_logo.png" width="400" alt="Seqcore Logo"/>
  </a>
</p>

High-performance biological sequence analysis library for Python.

A unified, NumPy-vectorized library for genomics, proteomics, structural biology, and drug design.

> **Note on GPU support:** Seqcore ships CuPy device-management utilities
> (`gpu_available`, `gpu_info`, `device`, `set_memory_limit`, `clear_gpu_cache`),
> but the analysis kernels themselves still execute on NumPy. GPU dispatch for
> the compute functions is planned, not implemented -- see
> [Roadmap](#roadmap).

## Installation

```bash
pip install seqcore
```

With GPU support:
```bash
pip install seqcore[gpu]
```

With all optional dependencies:
```bash
pip install seqcore[full]
```

## Quick Start

```python
import seqcore as sc

# DNA sequences - efficient 2-bit encoding
dna = sc.DNAArray("ACGTACGTACGT" * 1_000_000)

# Batch operations
sequences = sc.DNAArray([
    "ACGTACGT",
    "TGCATGCA",
    "GGGGCCCC",
])

# Vectorized operations
gc = sc.gc_content(sequences)
lengths = sc.length(sequences)
rev_comp = sc.reverse_complement(sequences)

# Translation
proteins = sc.translate(sequences)
```

## Features

### Sequence Operations

```python
# GC content, molecular weight, length
gc = sc.gc_content(dna)
mw = sc.molecular_weight(protein)

# Transcription and translation
rna = sc.transcribe(dna)
protein = sc.translate(dna, frame=0)

# K-mer operations
kmers = sc.extract_kmers(sequences, k=21)
kmer_counts = sc.count_kmers(sequences, k=21)
```

### Sequence Alignment

```python
# Pairwise alignment
result = sc.align(query, reference)
print(result.score, result.identity, result.cigar)

# Distance matrices
dm = sc.pairwise_distance(sequences, metric="edit")

# Pattern matching
matches = sc.find_pattern(sequences, "ATG[ACGT]{30,100}TAA")
```

### File I/O

```python
# Auto-detect format
data = sc.read("sequences.fasta")
data = sc.read("structure.pdb")
data = sc.read("reads.fastq.gz")

# Streaming for large files
for batch in sc.read_stream("huge.fastq.gz", batch_size=100_000):
    results = process(batch)

# Database fetching
seq = sc.fetch("NP_000509")      # NCBI/UniProt
structure = sc.fetch("1ABC")     # PDB
```

### Structural Biology

```python
# Load structure
structure = sc.read("protein.pdb")

# Access data
print(structure.chains)      # ['A', 'B']
print(structure.n_residues)  # 265

# Distance matrix
dm = sc.distance_matrix(structure, selection="CA")

# Find contacts
contacts = sc.find_contacts(structure, cutoff=4.0)

# RMSD calculation
rmsd = sc.rmsd(structure1, structure2, align=True)

# Surface analysis
sasa = sc.sasa(structure)
surface = sc.surface_residues(structure, threshold=25.0)

# Binding pockets
pockets = sc.find_pockets(structure)
```

### Drug Design

```python
# Small molecules
mol = sc.Molecule.from_smiles("CCO")

# Molecular properties
mw = sc.molecular_weight(molecules)
logp = sc.logp(molecules)
hbd = sc.h_bond_donors(molecules)

# ADMET filters
passes_lipinski = sc.lipinski_filter(molecules)
bbb_permeable = sc.bbb_filter(molecules)

# Fingerprints and similarity
fps = sc.morgan_fingerprint(molecules, radius=2)
similarity = sc.tanimoto_similarity(fps)

# Substructure search
matches = sc.substructure_search(molecules, "c1ccccc1")
```

### Phylogenetics

```python
# Tree construction
tree = sc.neighbor_joining(sequences)
tree = sc.upgma(sequences)

# Tree operations
print(tree.newick())
dist = tree.distance("Species_A", "Species_B")
subtree = tree.prune(["A", "B", "C"])
```

### Population Genetics

```python
# Variant analysis
variants = sc.read("variants.vcf")
af = sc.allele_frequency(variants)
maf = sc.minor_allele_frequency(variants)

# Population statistics
fst = sc.fst(pop1, pop2)
pi = sc.nucleotide_diversity(sequences)
d = sc.tajimas_d(sequences)

# Linkage disequilibrium
ld = sc.linkage_disequilibrium(variants)
```

### GPU Utilities

These manage CuPy devices and memory. They do **not** move Seqcore's analysis
functions onto the GPU -- those run on NumPy today.

```python
# Check GPU availability
if sc.gpu_available():
    print(sc.gpu_info())

# Select the active CuPy device for your own CuPy code
with sc.device("cuda:0"):
    xp = sc.core.device.get_array_module()  # cupy when a GPU is active

# Memory management
sc.set_memory_limit("8GB")
sc.clear_gpu_cache()

# Timing (works on any code)
with sc.timer() as t:
    result = sc.align(sequences, reference)
print(f"Completed in {t.elapsed:.2f}s")
```

### Interoperability

```python
# NumPy
arr = sequences.to_numpy()
sequences = sc.DNAArray.from_numpy(arr)

# pandas
df = sequences.to_dataframe()
df = structure.to_dataframe()

# Biopython
bio_seq = sequences[0].to_biopython()
sc_seq = sc.DNAArray.from_biopython(bio_seq)

# RDKit
rdkit_mol = molecule.to_rdkit()
sc_mol = sc.Molecule.from_rdkit(rdkit_mol)
```

## Performance

Seqcore stores a batch of sequences as one padded integer matrix, so batch-wide
operations are single NumPy calls rather than per-sequence loops. That wins
decisively where work amortizes across the batch, and loses where a compiled
per-element implementation already exists. Both cases are reported below.

100,000 sequences x 1000 bp. `Speedup` is Seqcore vs the **faster** of Biopython
and idiomatic pure Python; values below 1.0 mean Seqcore is slower.

| Operation | Mac mini (M4 Pro) | AWS g5.2xlarge (EPYC 7R32) |
|-----------|------------------:|---------------------------:|
| GC content | **16.2x** | **6.4x** |
| Translation | **27.9x** | **18.1x** |
| k-mer counting (k=8, 2000 seqs) | **4.4x** | **9.2x** |
| k-mer counting (k=12) | 1.7x | 1.4x |
| Reverse complement | 0.94x | 0.88x |
| Global alignment (L=3200) | 0.07x | 0.03x |

*Both machines run Python 3.12, NumPy 2.5.2, Biopython 1.88; best of 5 runs,
each implementation in a fresh subprocess. Raw results in `benchmarks/results/`.*

Every conclusion holds on both machines — the same operations win and lose, in
the same order — but absolute times differ by 2.4–4.5x, so treat any single
speedup figure as a point estimate.

Reproduce with:

```bash
python benchmarks/benchmark_suite.py
```

### When to use something else

- **Pairwise alignment** is a NumPy anti-diagonal wavefront. Far faster than a
  scalar Python loop, but still 15–29x slower than Biopython's C aligner. For
  alignment-bound work, use a dedicated aligner.
- **Reverse complement** is memory-bandwidth bound and sits at rough parity with
  Biopython's `str.translate`. No vectorization advantage is available.
- **k-mer counting** switches to a `Counter` over strings once `4**k` exceeds the
  number of windows, because past that point almost every k-mer is unique.
- **GPU**: not used by any compute path. See below.

### GPU

Seqcore ships CuPy device-management helpers but **does not compute on the GPU**.
`benchmarks/benchmark_gpu.py` measures what a GPU backend would be worth by
timing CuPy transcriptions of Seqcore's kernels (NVIDIA A10G):

| | Data resident on device | Per call, incl. host transfer |
|---|---|---|
| Reverse complement | 326x | **8.5x** |
| GC content | 36x | **5.1x** |
| Translation | 32x | **13.3x** |
| k-mers (k=8) | 23x | **20.4x** |

PCIe transfer, not arithmetic, decides the outcome — for reverse complement it is
97% of per-call time. A GPU backend is only worth building if the encoded matrix
stays resident on the device across many operations.

## Requirements

- Python 3.9+
- NumPy 1.22+

Optional:
- CuPy (GPU acceleration)
- Biopython (interoperability)
- RDKit (molecular operations)
- MDAnalysis (structure analysis)

## Roadmap

- GPU dispatch for the core kernels (`gc_content`, `translate`,
  `reverse_complement`, k-mer counting) via CuPy, wired through
  `get_array_module`.
- Compiled pairwise alignment kernel to replace the current pure-NumPy
  dynamic programming implementation.
- Published API reference on Read the Docs.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE).

## Author

**Dr. Pritam Kumar Panda**
Stanford University
Email: pritam@stanford.edu

## Citation

If you use Seqcore in your research, please cite:

```bibtex
@software{seqcore,
  author = {Panda, Pritam Kumar},
  title = {Seqcore: High-performance biological sequence analysis},
  url = {https://github.com/pritampanda15/seqcore},
  version = {0.4.0},
  year = {2026},
  institution = {Stanford University}
}
```
