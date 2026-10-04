figwig
======

A fast, multithreaded reader and writer of bigWig files, into and out of
numpy.

It reads the per-base values of tens or hundreds of thousands of windows in
one call, straight into a float32 numpy array, from one bigWig or from several
at once, and writes bigWigs from the same windows, or from intervals. The
work runs on several threads, because zlib's ``uncompress()`` and
``compress2()`` and the numba kernels all run without the GIL. It depends only
on numpy and numba; libdeflate, through the optional ``deflate`` package,
makes writing faster. The README on `GitHub <https://github.com/jmschrei/figwig>`_
has examples for data loaders and the comparisons with other readers and
writers, and the API pages describe every value figwig gives, every file it
refuses and how it lays out the files it writes. Its command,
:doc:`figwig bam2bw <bam2bw>`, turns SAM/BAM files of reads, or BED/tsv files
of fragments, into bigWigs of per-base counts.

.. code-block:: python

    import numpy
    from figwig import BigWigReader
    from figwig import read_bigwig

    chroms = numpy.array(["chr1", "chr1", "chr2"])
    starts = numpy.array([1_000_000, 2_500_000, 300_000])

    bw = BigWigReader("ENCFF830RWF.bigWig")
    y = bw.read(chroms, starts, width=1000, n_jobs=8)   # (3, 1000) float32

    y = read_bigwig(["ENCFF830RWF.bigWig", "ENCFF989SAK.bigWig"], chroms,
        starts, width=1000)                              # (3, 2, 1000) float32

.. code-block:: python

    import numpy
    from figwig import BigWigWriter
    from figwig import write_bigwig

    chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
    chroms = numpy.array(["chr1", "chr1", "chr2"])
    starts = numpy.array([1_000_000, 2_500_000, 300_000])
    y_hat = numpy.random.default_rng(0).random((3, 2, 1000), dtype=numpy.float32)

    write_bigwig(["plus.bw", "minus.bw"], chrom_sizes, chroms, starts, y_hat)

    with BigWigWriter("out.bw", chrom_sizes) as bw:
        bw.write("chr1", [100, 250], [1.5, 2.0], ends=[200, 300])  # intervals
        bw.write("chr1", [1000, 1005], [[3], [1]])                  # single bases
        bw.write("chr2", [5000], y_hat[:1, 0])                      # a window

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
   :caption: Command line

   bam2bw

.. toctree::
   :maxdepth: 1
   :caption: API

   api/bigwig
   api/writer
