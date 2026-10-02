figwig
======

A fast, multithreaded reader of many bigWig windows at once, into numpy.

It reads the per-base values of tens or hundreds of thousands of windows in
one call, straight into a float32 numpy array, from one bigWig or from several
at once. The work runs on several threads, because zlib's ``uncompress()``
and the numba decoder both run without the GIL. It depends only on numpy and
numba. The README on `GitHub <https://github.com/jmschrei/figwig>`_ has
examples for data loaders and the comparison with other readers, and the API
page describes every value figwig gives and every file it refuses.

.. code-block:: python

    import numpy
    from figwig import BigWig
    from figwig import read_windows

    chroms = numpy.array(["chr1", "chr1", "chr2"])
    starts = numpy.array([1_000_000, 2_500_000, 300_000])

    bw = BigWig("ENCFF830RWF.bigWig")
    y = bw.read(chroms, starts, width=1000, n_jobs=8)   # (3, 1000) float32

    y = read_windows(["ENCFF830RWF.bigWig", "ENCFF989SAK.bigWig"], chroms,
        starts, width=1000)                              # (3, 2, 1000) float32

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
