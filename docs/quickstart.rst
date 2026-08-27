Quickstart
==========

Sequence arrays
---------------

Seqcore's central type is a batched sequence array. Operations are applied to
the whole batch at once rather than sequence by sequence.

.. code-block:: python

   import seqcore as sc

   sequences = sc.DNAArray([
       "ACGTACGT",
       "TGCATGCA",
       "GGGGCCCC",
   ])

   sc.gc_content(sequences)         # array([50., 50., 100.])
   sc.length(sequences)             # array([8, 8, 8])
   sc.reverse_complement(sequences)
   sc.translate(sequences)

Reading files
-------------

:func:`seqcore.read` detects the format from the file extension and
transparently handles ``.gz`` and ``.bz2`` compression.

.. code-block:: python

   sequences = sc.read("sequences.fasta")
   structure = sc.read("structure.pdb")
   variants  = sc.read("variants.vcf")

For inputs too large to hold in memory, stream them in batches:

.. code-block:: python

   for batch in sc.read_stream("huge.fastq.gz", batch_size=100_000):
       gc = sc.gc_content(batch)

Structures
----------

.. code-block:: python

   structure = sc.read("protein.pdb")

   structure.chains
   structure.n_residues
   sc.distance_matrix(structure, selection="CA")
   sc.find_contacts(structure, cutoff=4.0)

GPU utilities
-------------

.. warning::
   Seqcore ships CuPy device-management helpers, but the analysis kernels
   themselves currently execute on NumPy. Entering a ``cuda`` device context
   does **not** move ``align``, ``gc_content`` or ``translate`` onto the GPU.
   GPU dispatch for those functions is planned but not yet implemented.

.. code-block:: python

   if sc.gpu_available():
       print(sc.gpu_info())

   with sc.device("cuda:0"):
       xp = sc.core.device.get_array_module()  # cupy when a GPU is active

Performance expectations
------------------------

Seqcore is fastest on whole-batch columnar work, where a single NumPy call
covers the entire batch: roughly 38x Biopython for GC content and 26x for
translation on 100,000 sequences of 1000 bp.

It is *not* the fastest option everywhere. Reverse complement sits at rough
parity with Biopython (~0.86x) because the operation is memory-bandwidth bound.
Pairwise alignment uses an anti-diagonal NumPy wavefront that is far faster than
a scalar Python loop but still roughly 16x slower than Biopython's compiled
aligner; use a dedicated aligner for alignment-bound work. ``k``-mer counting
falls back to a string-based counter once ``4**k`` exceeds the number of
windows, since past that point nearly every ``k``-mer is unique.

See ``benchmarks/`` for reproducible measurements and ``paper/`` for a full
write-up of the implementation techniques.
