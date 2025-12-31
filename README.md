# Seqcore
<p align="center">
  <a href="https://github.com/pritampanda15/Seqcore">
    <img src="https://raw.githubusercontent.com/pritampanda15/Seqcore/main/logo/seqcore_logo.png" width="400" alt="Seqcore Logo"/>
  </a>
</p>

High-performance biological sequence analysis library for Python.

A unified, GPU-accelerated library for genomics, proteomics, structural biology, and drug design.

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

### GPU Acceleration

```python
# Check GPU availability
if sc.gpu_available():
    print(sc.gpu_info())

# Device context
with sc.device("cuda:0"):
    result = sc.align(sequences, reference)

# Memory management
sc.set_memory_limit("8GB")
sc.clear_gpu_cache()

# Timing
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

Seqcore provides significant speedups over traditional libraries:

| Operation | Biopython | Seqcore | Speedup |
|-----------|-----------|---------|---------|
| GC Content (1M seqs) | 45s | 0.8s | 56x |
| Reverse Complement | 12s | 0.1s | 120x |
| Translation | 38s | 0.5s | 76x |
| K-mer Counting | 89s | 1.2s | 74x |

*Benchmarks on AMD Ryzen 9 5900X, 32GB RAM. GPU benchmarks show additional 10-50x speedup.*

## Requirements

- Python 3.9+
- NumPy 1.21+

Optional:
- CuPy (GPU acceleration)
- Biopython (interoperability)
- RDKit (molecular operations)
- MDAnalysis (structure analysis)

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
  version = {0.3.0},
  year = {2025},
  institution = {Stanford University}
}
```
