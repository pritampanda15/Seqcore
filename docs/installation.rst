Installation
============

Requirements
------------

* Python 3.9 or newer
* NumPy 1.22 or newer

Basic install
-------------

.. code-block:: bash

   pip install seqcore

Optional extras
---------------

Seqcore keeps its required dependency set to NumPy alone. Additional
functionality is enabled through extras:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Install
     - Enables
   * - ``full``
     - ``pip install seqcore[full]``
     - Biopython interoperability, pandas ``to_dataframe`` export, HDF5/``.h5ad`` reading
   * - ``gpu``
     - ``pip install seqcore[gpu]``
     - CuPy device-management utilities (see :doc:`quickstart`)
   * - ``structure``
     - ``pip install seqcore[structure]``
     - MDAnalysis-backed structure analysis
   * - ``molecules``
     - ``pip install seqcore[molecules]``
     - RDKit-backed fingerprints, descriptors and substructure search (Python 3.10+)

Extras can be combined::

   pip install "seqcore[full,structure,molecules]"

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/pritampanda15/Seqcore.git
   cd Seqcore
   pip install -e ".[dev]"
   pytest

Verifying the install
---------------------

.. code-block:: python

   import seqcore as sc

   print(sc.__version__)
   print(sc.gc_content(sc.DNAArray("ACGTACGT")))
