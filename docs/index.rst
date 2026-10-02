figwig
======

A fast, multithreaded reader and writer of bigWig files, into and out of
numpy.

It reads the per-base values of tens or hundreds of thousands of windows in
one call, straight into a float32 numpy array, from one bigWig or from several
at once, and writes bigWigs from intervals, single bases or dense arrays. The
work runs on several threads, because zlib's ``uncompress()`` and
``compress2()`` and the numba kernels all run without the GIL. It depends only
on numpy and numba; libdeflate, through the optional ``deflate`` package,
makes writing faster. The README on `GitHub <https://github.com/jmschrei/figwig>`_
has examples for data loaders and the comparisons with other readers and
writers, and the API pages describe every value figwig gives, every file it
refuses and how it lays out the files it writes.

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

.. code-block:: python

    import numpy
    from figwig import BigWigWriter

    with BigWigWriter("out.bw", {"chr1": 248_956_422, "chr2": 242_193_529}) as bw:
        bw.add("chr1", [100, 250], [200, 300], [1.5, 2.0])     # intervals
        bw.add("chr1", numpy.array([1000, 1005]), values=[3, 1])  # single bases
        bw.add("chr2", 5000, values=numpy.random.rand(2000))      # a dense array

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
   api/writer
