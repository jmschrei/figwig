figwig
======

A fast, multithreaded reader of many bigWig windows at once, into numpy.

It reads the per-base values of tens or hundreds of thousands of windows in
one call, straight into a float32 numpy array. The work runs on several
threads, because zlib's ``uncompress()`` and the numba decoder both run
without the GIL. It depends only on numpy and numba.

.. code-block:: python

    import numpy
    from figwig import BigWig

    bw = BigWig("ENCFF830RWF.bigWig")
    chroms = numpy.array(["chr1", "chr1", "chr2"])
    starts = numpy.array([1_000_000, 2_500_000, 300_000])
    y = bw.read(chroms, starts, width=1000, n_jobs=8)   # (3, 1000) float32

Installation
============

figwig is not on PyPI yet. Install it from a clone:

.. code-block:: bash

    git clone https://github.com/jmschrei/figwig.git
    cd figwig
    pip install .

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/jmschrei/figwig.git
    cd figwig
    uv sync --extra dev
    uv run pytest


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   whats_new

.. toctree::
   :maxdepth: 1
   :caption: API

   api/bigwig
