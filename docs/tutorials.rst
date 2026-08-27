Tutorials
=========

Worked, runnable examples live in the ``examples/`` directory of the
repository.

``examples/usage_examples.py``
------------------------------

Ten self-contained examples covering sequence operations, k-mers, alignment,
file I/O across all supported formats, structural analysis, phylogenetics and
population genetics. It runs against the test fixtures committed under
``tests/data/`` and needs no external downloads::

   python examples/usage_examples.py

``examples/data/single_cell/single_cell_analysis.py``
-----------------------------------------------------

A single-cell RNA-seq walkthrough using the 10x Genomics PBMC 3k dataset. This
example requires a data file that is **not** committed to the repository
because of its size. Download ``pbmc3k_molecule_info.h5`` from
`10x Genomics <https://www.10xgenomics.com/datasets>`_ and place it in
``examples/data/single_cell/data/`` before running::

   python examples/data/single_cell/single_cell_analysis.py

The script exits with a descriptive message if the file is absent.

Benchmarks
----------

``benchmarks/benchmark_suite.py`` compares Seqcore against Biopython and
idiomatic pure Python across batch operations, pairwise alignment and ``k``-mer
counting. It produces the JSON that the manuscript figures are built from.

``benchmarks/benchmark_gpu.py`` estimates how much a future GPU backend could
gain, by timing CuPy transcriptions of Seqcore's kernels against Seqcore's CPU
path. It requires an NVIDIA GPU and CuPy; run ``benchmarks/check_gpu.py`` first
to verify the CUDA setup. Note that Seqcore itself computes on the CPU, so those
GPU numbers are headroom estimates rather than Seqcore performance.
