figwig
======

A fast, multithreaded reader and writer of bigWig files, into and out of
numpy.

figwig reads the per-base values of tens or hundreds of thousands of windows
in one call, straight into a float32 numpy array, from one bigWig or from
several at once, and writes bigWigs from the same windows, from intervals, or
from per-base counts. Its command, :doc:`figwig bam2bw <bam2bw>`, turns
SAM/BAM files of reads, or BED/tsv files of fragments, into bigWigs of
per-base counts. The work runs on threads that run in parallel, because
zlib's ``uncompress()`` and ``compress2()``, libdeflate and figwig's numba
kernels all run without the GIL. It depends only on numpy and numba. A file
it cannot read with certainty raises a ``ValueError`` that says what it found,
rather than being guessed at, so that a pipeline can fall back to another
reader.

==============================================================  ===============================  ==================================
task                                                            figwig                           fastest other tool
==============================================================  ===============================  ==================================
read 167,750 windows of 1,000 bp from an ATAC-seq bigWig        0.135 s on 8 threads             1.25 s, pybigtools
write 15.9 million per-base counts                              0.18 s on 8 threads              4.20 s, pyBigWig
write 21.2 million intervals with zoom levels                   0.55 s on 8 threads              5.47 s, pybigtools
convert a 2.4 GB ATAC-seq BAM to two stranded bigWigs           4.65 s and 910 MB at ``-p 2``    58.48 s and 3,039 MB, bam2bw 0.5.1
==============================================================  ===============================  ==================================

These were measured on a 2x AMD EPYC 9575F; :doc:`benchmarks` has the
setup, every comparison, and how the times fall with threads.

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

It needs Python 3.10 or later, numpy 1.23 or later, and numba 0.58 or later.
To write faster with libdeflate, install the ``fast`` extra, which adds the
``deflate`` package, and for ``figwig bam2bw``, the ``bam2bw`` extra, which
adds pysam, pyfaidx, biopython, tqdm, isal and deflate:

.. code-block:: bash

    pip install ".[fast]"
    pip install ".[bam2bw]"

Development install
-------------------

.. code-block:: bash

    git clone https://github.com/jmschrei/figwig.git
    cd figwig
    uv sync --extra dev
    uv run pytest

Claude Code skill
=================

figwig ships a `Claude Code <https://claude.com/claude-code>`_ skill that
teaches a coding agent to use figwig in any project: reading windows from one
bigWig or several, writing predictions, intervals and per-base counts,
converting reads with ``figwig bam2bw``, the rules a write follows, what each
error message means, and how much threads gain. ``figwig install-skill``
copies it into ``~/.claude/skills/figwig``:

.. code-block:: bash

    figwig install-skill                 # into ~/.claude/skills
    figwig install-skill -d DIRECTORY    # into DIRECTORY/figwig
    figwig install-skill --force         # replace an installed copy

An installed copy raises ``FileExistsError`` unless ``--force`` is given, so
after upgrading figwig, run it with ``--force`` to pick up a changed skill.
``--symlink`` links to the copy inside the installed package instead, so that
the skill follows the package; the link breaks if the package moves or is
uninstalled.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   whats_new

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   reading
   writing
   threads
   benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Command line

   bam2bw

.. toctree::
   :maxdepth: 1
   :caption: API

   api/bigwig
   api/writer
