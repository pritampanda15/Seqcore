# Changelog

All notable changes to Seqcore are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-09-09

### Fixed

- `read_h5ad()` silently returned an empty result for 10x CellRanger `.h5`
  files. Those store the matrix under `/matrix`, not `/X`, so the reader found
  nothing and handed back `X=None, n_obs=0, n_vars=0` without raising. A caller
  had no way to tell that apart from a dataset with no cells in it.

  10x files are now read, including the CellRanger 2.x layout that names its
  feature datasets `gene_names` and `genes`. 10x stores genes as rows, the
  opposite of AnnData, so the matrix is transposed on read: `X` is always cells
  by genes. Barcodes land in `obs["_index"]`, and feature names, ids, genome and
  type in `var`.

- A file with neither `/X` nor `/matrix` now raises `ValueError` naming its
  top-level keys, instead of returning an empty matrix.

- A sparse `/X` group is now assembled into a SciPy sparse matrix rather than
  left as `X=None`. The `X_data`, `X_indices` and `X_indptr` entries are still
  present for callers that were reading them.

- String columns in `obs` and `var` are decoded from HDF5 fixed-width bytes to
  `str`. Gene names previously came back as `b"GENE1"`.

- Reading a sparse matrix without SciPy installed now raises `ImportError`
  naming the fix, rather than returning `X=None`. That was the same
  silent-empty failure in a narrower form.

### Changed

- A sparse `/X` read without SciPy previously returned `X=None` alongside the
  component arrays. It now raises. Callers that were reading `X_data`,
  `X_indices` and `X_indptr` directly on a machine without SciPy will need to
  install SciPy; the components are still populated when it is present.

## [0.5.0] - 2026-08-27

Performance work on the domain modules -- molecules, structural biology and
phylogenetics -- which had never been benchmarked. Every change is pinned to the
previous implementation by a differential test; no result changes.

### Performance

- **SASA is roughly 15,800x faster at 5,000 atoms** (an extrapolated 43 minutes
  down to 0.17 s). It was a triple loop over atoms x sphere points x atoms that
  also re-derived each atom's van der Waals radius from its element string
  inside the innermost loop. Radii are now resolved once and each atom is tested
  only against neighbours whose inflated sphere can reach it, making the cost
  roughly linear rather than quadratic.
- **`find_contacts` is 161x faster at 5,000 atoms.** Candidate pairs come from a
  spatial index instead of a Python double loop; pairs, their ordering and their
  distances are unchanged.
- **`tanimoto_similarity` is 216x faster** (16.4 s to 0.076 s at 2000x2000). For
  binary fingerprints the intersection is a dot product, so the whole matrix is
  one BLAS call.
- **`morgan_fingerprint` is 26x faster.** It spent 95% of its time in
  `np.array()` walking RDKit bit vectors one bit at a time;
  `DataStructs.ConvertToNumpyArray` does the same work inside RDKit.
- **Molecule property functions are ~46x faster.** Every one of them re-parsed
  the SMILES with `Chem.MolFromSmiles`; `Molecule` now caches its parsed form.
  These are RDKit wrappers and are not expected to beat RDKit, but they no
  longer charge a large multiple for the convenience.
- **All-pairs Hamming distance is 377x faster**, as a single matrix product over
  the one-hot encoding.

### Added

- `neighbor_joining()` and `upgma()` accept `metric=`. The default `"identity"`
  aligns every pair with Needleman-Wunsch, costing O(n^2 L^2), and is unchanged.
  `"hamming"` compares equal-length sequences column by column and is far
  faster, but requires input that is already aligned. With `metric="hamming"`,
  a 48-taxon tree goes from 6.2 s to 0.016 s, and Seqcore is 3-4x faster than
  Biopython on the like-for-like comparison rather than ~200x slower.

  The two metrics are **not** equivalent: once sequences diverge enough for the
  aligner to open gaps they disagree by up to 0.24 in our tests, which is why
  this is opt-in.
- `benchmarks/benchmark_modules.py`, covering the four domain modules against
  RDKit, SciPy, Biopython and a NumPy reference.
- 21 further differential tests (133 -> 154).

### Fixed

- `sc.molecular_weight(molecules)`, shown in the README's drug-design section,
  raised `TypeError`. The name is exported by both `core.operations` (for
  sequences) and `molecules`, and the sequence version shadowed the other. Use
  `seqcore.molecules.molecular_weight` for molecules; this is the only such
  collision in the public API.

### Known limitations

- `nucleotide_diversity` and `tajimas_d` remain O(n^2) over sequence pairs
  (0.21 s for 160 sequences).
- Tree building itself is still an O(n^3) Python loop; with `metric="hamming"`
  that, rather than the distance matrix, is now the bottleneck.
- `distance_matrix` is left as-is. It already vectorizes each row, and the
  faster BLAS formulation introduces ~1e-12 error and a non-zero diagonal.

## [0.4.0] - 2026-08-26

> **Note:** this release changes performance substantially but not results. The
> optimized routines are pinned to naive scalar references by randomized
> differential tests, and the alignment score matrices are bit-identical to the
> previous implementation. Version 0.3.0 on PyPI predates all of this work.

### Performance

Removed per-sequence Python iteration from the hot paths. All speedups are
against the previous implementation, measured by
`benchmarks/benchmark_suite.py`; results are committed under
`benchmarks/results/` and every change is pinned to a scalar reference by
`tests/test_equivalence.py`.

- **Translation is 21.6x faster** (100k x 1000 bp: 4.49s -> 0.21s). It no longer
  decodes sequences to Python strings or loops per codon; codons index a
  precomputed table directly on the encoded matrix, and the result is already a
  valid protein encoding.
- **Global alignment is 19.9x faster** at L=3200 (4.22s -> 0.21s). The scalar
  double loop over DP cells was replaced with an anti-diagonal wavefront, cutting
  interpreter-level iterations from O(mn) to O(m+n). The recurrence is unchanged
  and the score matrix is bit-identical. Still slower than Biopython's C aligner.
- **GC content is 5.1x faster** (100k x 1000 bp: 0.187s -> 0.037s) via a masked
  2-D reduction instead of a per-sequence loop.
- **k-mer counting is 4.7x faster at k=8** through base-5 integer encoding plus
  `np.unique`, with only distinct k-mers decoded back to strings. Guarded by a
  `4**k <= windows` dispatch rule: past that point nearly every k-mer is unique,
  the integer path would be a 1.3x regression, and the string path is used
  instead.
- **Batch encoding is 1.8x faster** — one concatenation and one table gather for
  the whole batch rather than three dispatches per sequence.
- **Reverse complement is 1.5x faster** by folding the complement table into the
  reversal so the batch is traversed once. Still slower than Biopython; this path
  is memory-bandwidth bound, not dispatch bound, so there is no vectorization
  advantage to gain.

Comparisons against Biopython and pure Python are platform dependent and are
reported, with the machine attached, in the README and in `paper/`. The speedups
above are against Seqcore's own previous implementation on one machine, which is
what a changelog should measure.

### Added

- Benchmarks now run on two machines (Mac mini M4 Pro and an AWS g5.2xlarge)
  with matched Python/NumPy/Biopython versions, so hardware is the only
  difference. Every conclusion holds on both; absolute times differ by 2.4-4.5x.
- Benchmark results are self-describing: each JSON records the CPU model,
  hardware model (Mac model identifier or EC2 instance type), core count, Python
  version and library versions.
- `paper/make_figures.py` generates `numbers.tex`, a set of LaTeX macros for
  every value the manuscript quotes including the platform sentence, so the text
  cannot drift from the data it describes.
- `tests/test_equivalence.py`: randomized differential tests asserting every
  vectorized fast path agrees exactly with a naive scalar reference, across
  ragged batches, empty sequences, `N` bases, all reading frames, both
  stop-codon policies, k from 1 to 30, and three alignment scoring schemes.
  DP matrices are compared for exact equality. Test count went from 106 to 133.
- `benchmarks/benchmark_suite.py`: the publication benchmark, comparing Seqcore
  against Biopython and pure Python across batch operations, alignment and
  k-mer counting. Each implementation runs in a fresh subprocess. A `--baseline`
  flag compares against another checkout for tracking performance across
  commits during development.
- The pure-Python benchmark baseline no longer imports its codon table from
  Seqcore; it defines its own, so the baseline is genuinely independent of the
  library under test. The two tables agree on all 64 codons.
- `benchmarks/benchmark_gpu.py`: rewritten to measure the headroom a GPU backend
  could reach, by benchmarking CuPy transcriptions of Seqcore's kernels against
  Seqcore's CPU path. It verifies the CuPy kernels reproduce the CPU results
  before timing, synchronizes the stream around every measurement, and reports
  both kernel-only and transfer-inclusive times. It does not report Seqcore GPU
  performance, because Seqcore does not compute on the GPU.
- `paper/`: a bioRxiv-format manuscript describing the optimization work, with
  figures and Table 1 generated from the committed benchmark JSON by
  `paper/make_figures.py`.

### Fixed

- Corrected the performance claims in the README. The previously published
  speedup table (56x-120x over Biopython) was not reproducible; the table now
  reports measured results from `benchmarks/benchmark_suite.py`, including
  the operations where Seqcore is slower than Biopython.
- Documented that the GPU utilities manage CuPy devices only -- the analysis
  kernels still execute on NumPy. `seqcore.device()` never moved `align`,
  `gc_content` or `translate` onto the GPU, and the docs no longer imply it does.
- Removed the stale `numpy<2.0.0` and `pandas<2.2.0` upper bounds. The test
  suite passes on NumPy 2.x and pandas 2.x, and the old caps made Seqcore
  uninstallable alongside a current scientific Python stack.
- `fetch()` now applies a network timeout (30s by default, configurable via the
  `timeout` argument) so an unresponsive server cannot hang the caller
  indefinitely.
- Removed the invalid `affiliation` key from `[project.authors]` in
  `pyproject.toml`, which is not part of the project metadata specification.
- Removed 26 dead imports across the package and narrowed the blanket
  `__init__.py` F401 lint exemption that had been hiding them.
- Corrected the stale version (`0.1.0`), release date and placeholder ORCID in
  `CITATION.cff`.
- Documented that `align()` accepts a `gap_extend` argument that has no effect:
  scoring is linear-gap using `gap_open` alone, and affine gap penalties are not
  implemented. Previously the parameter was silently ignored.

### Added

- Documentation pages for every entry in the Sphinx table of contents.
  `installation`, `quickstart`, `tutorials`, `contributing`, `changelog` and the
  seven `api/*` pages were all referenced but missing, producing 13 build
  warnings and an empty site.
- Python 3.13 to the supported classifiers, CI matrix and Black target versions.
- `tests/`, `LICENSE`, `README.md`, `CITATION.cff`, `CONTRIBUTING.md` and this
  changelog to the source distribution, which previously shipped only the
  package directory.
- A roadmap section in the README covering GPU kernel dispatch and a compiled
  pairwise alignment implementation.

### Changed

- The CI documentation job now builds the Sphinx site with `-W` instead of
  echoing a placeholder string.
- Dropped the deprecated `safety check` step from the security workflow;
  `pip-audit` and Bandit are retained and `pip-audit` now reports failures
  rather than being suppressed with `|| true`.

## [0.3.0] - 2025-12-31

- Single-cell RNA-seq example and HDF5/`.h5ad` reading.
- Expanded file format coverage to 16 formats.

## [0.2.0] - 2025-12-31

- Structural biology, drug design, phylogenetics and population genetics modules.

## [0.1.0] - 2025-12-30

- Initial release: sequence arrays, vectorized operations, k-mers, alignment and
  core file I/O.

[0.5.1]: https://github.com/pritampanda15/Seqcore/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/pritampanda15/Seqcore/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/pritampanda15/Seqcore/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pritampanda15/Seqcore/releases/tag/v0.3.0
[0.2.0]: https://github.com/pritampanda15/Seqcore/releases/tag/v0.2.0
[0.1.0]: https://github.com/pritampanda15/Seqcore/releases/tag/v0.1.0
