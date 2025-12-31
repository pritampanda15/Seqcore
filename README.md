# Seqcore
<p align="center">
  <a href="https://github.com/pritampanda15/Seqcore">
    <img src="https://github.com/pritampanda15/Seqcore/blob/main/logo/seqcore_logo.png" width="400" alt="Seqcore Logo"/> 
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
import seqcore as bc

# DNA sequences - efficient 2-bit encoding
dna = bc.DNAArray("ACGTACGTACGT" * 1_000_000)

# Batch operations
sequences = bc.DNAArray([
    "ACGTACGT",
    "TGCATGCA",
    "GGGGCCCC",
])

# Vectorized operations
gc = bc.gc_content(sequences)
lengths = bc.length(sequences)
rev_comp = bc.reverse_complement(sequences)

# Translation
proteins = bc.translate(sequences)
```

## Features

### Sequence Operations

```python
# GC content, molecular weight, length
gc = bc.gc_content(dna)
mw = bc.molecular_weight(protein)

# Transcription and translation
rna = bc.transcribe(dna)
protein = bc.translate(dna, frame=0)

# K-mer operations
kmers = bc.extract_kmers(sequences, k=21)
kmer_counts = bc.count_kmers(sequences, k=21)
```

### Sequence Alignment

```python
# Pairwise alignment
result = bc.align(query, reference)
print(result.score, result.identity, result.cigar)

# Distance matrices
dm = bc.pairwise_distance(sequences, metric="edit")

# Pattern matching
matches = bc.find_pattern(sequences, "ATG[ACGT]{30,100}TAA")
```

### File I/O

```python
# Auto-detect format
data = bc.read("sequences.fasta")
data = bc.read("structure.pdb")
data = bc.read("reads.fastq.gz")

# Streaming for large files
for batch in bc.read_stream("huge.fastq.gz", batch_size=100_000):
    results = process(batch)

# Database fetching
seq = bc.fetch("NP_000509")      # NCBI/UniProt
structure = bc.fetch("1ABC")     # PDB
```

### Structural Biology

```python
# Load structure
structure = bc.read("protein.pdb")

# Access data
print(structure.chains)      # ['A', 'B']
print(structure.n_residues)  # 265

# Distance matrix
dm = bc.distance_matrix(structure, selection="CA")

# Find contacts
contacts = bc.find_contacts(structure, cutoff=4.0)

# RMSD calculation
rmsd = bc.rmsd(structure1, structure2, align=True)

# Surface analysis
sasa = bc.sasa(structure)
surface = bc.surface_residues(structure, threshold=25.0)

# Binding pockets
pockets = bc.find_pockets(structure)
```

### Drug Design

```python
# Small molecules
mol = bc.Molecule.from_smiles("CCO")

# Molecular properties
mw = bc.molecular_weight(molecules)
logp = bc.logp(molecules)
hbd = bc.h_bond_donors(molecules)

# ADMET filters
passes_lipinski = bc.lipinski_filter(molecules)
bbb_permeable = bc.bbb_filter(molecules)

# Fingerprints and similarity
fps = bc.morgan_fingerprint(molecules, radius=2)
similarity = bc.tanimoto_similarity(fps)

# Substructure search
matches = bc.substructure_search(molecules, "c1ccccc1")
```

### Phylogenetics

```python
# Tree construction
tree = bc.neighbor_joining(sequences)
tree = bc.upgma(sequences)

# Tree operations
print(tree.newick())
dist = tree.distance("Species_A", "Species_B")
subtree = tree.prune(["A", "B", "C"])
```

### Population Genetics

```python
# Variant analysis
variants = bc.read("variants.vcf")
af = bc.allele_frequency(variants)
maf = bc.minor_allele_frequency(variants)

# Population statistics
fst = bc.fst(pop1, pop2)
pi = bc.nucleotide_diversity(sequences)
d = bc.tajimas_d(sequences)

# Linkage disequilibrium
ld = bc.linkage_disequilibrium(variants)
```

### GPU Acceleration

```python
# Check GPU availability
if bc.gpu_available():
    print(bc.gpu_info())

# Device context
with bc.device("cuda:0"):
    result = bc.align(sequences, reference)

# Memory management
bc.set_memory_limit("8GB")
bc.clear_gpu_cache()

# Timing
with bc.timer() as t:
    result = bc.align(sequences, reference)
print(f"Completed in {t.elapsed:.2f}s")
```

### Interoperability

```python
# NumPy
arr = sequences.to_numpy()
sequences = bc.DNAArray.from_numpy(arr)

# pandas
df = sequences.to_dataframe()
df = structure.to_dataframe()

# Biopython
bio_seq = sequences[0].to_biopython()
bc_seq = bc.DNAArray.from_biopython(bio_seq)

# RDKit
rdkit_mol = molecule.to_rdkit()
bc_mol = bc.Molecule.from_rdkit(rdkit_mol)
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
  version = {0.1.0},
  year = {2026},
  institution = {Stanford University}
}
```
