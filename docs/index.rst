Seqcore Documentation
=====================

**Seqcore** is a high-performance Python library for biological sequence analysis,
providing unified APIs for genomics, proteomics, structural biology, and drug design.

.. image:: https://img.shields.io/pypi/v/seqcore.svg
   :target: https://pypi.org/project/seqcore/

.. image:: https://img.shields.io/github/license/pritampanda15/seqcore.svg
   :target: https://github.com/pritampanda15/seqcore/blob/main/LICENSE

Features
--------

- **Vectorized Operations**: Fast sequence analysis using NumPy
- **GPU Acceleration**: Optional CuPy backend for GPU computing
- **16+ File Formats**: FASTA, FASTQ, PDB, VCF, SDF, GenBank, and more
- **Structural Biology**: PDB/mmCIF parsing, RMSD, contacts
- **Drug Design**: Fingerprints, similarity, ADMET filters
- **Phylogenetics**: Neighbor-Joining, UPGMA trees
- **Population Genetics**: Allele frequency, Tajima's D, LD

Quick Start
-----------

Installation::

    pip install seqcore

Basic usage::

    import seqcore as sc

    # Read sequences
    sequences = sc.read("sequences.fasta")

    # Calculate GC content
    gc = sc.gc_content(sequences)

    # Translate to protein
    proteins = sc.translate(sequences)

    # Build phylogenetic tree
    tree = sc.neighbor_joining(sequences)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/io
   api/alignment
   api/structure
   api/molecules
   api/phylogenetics
   api/population

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Implementation notes
--------------------

The ``paper/`` directory of the repository contains a manuscript describing the
array layout and the vectorization techniques Seqcore uses (batched ordinal
encoding, masked reductions, table-driven translation, and an anti-diagonal
wavefront formulation of pairwise dynamic programming), together with measured
benchmarks and the cases where Seqcore is not the fastest option.

Citation
--------

If you use Seqcore in your research, please cite::

    @software{seqcore,
      author = {Panda, Pritam Kumar},
      title = {Seqcore: A High-Performance Biological Sequence Analysis Library},
      year = {2025},
      url = {https://github.com/pritampanda15/seqcore}
    }

License
-------

Seqcore is released under the MIT License.
