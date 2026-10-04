Writing bigWigs
===============

figwig writes bigWigs from the same windows it reads, such as a model's
predictions for them, from intervals, or from single bases, such as per-base
counts. ``write_bigwig`` writes all of the values in one call, and
``BigWigWriter`` writes them a batch at a time.

Writing windows
---------------

.. code-block:: python

    import numpy
    from figwig import read_bigwig
    from figwig import write_bigwig

    chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
    chroms = numpy.array(["chr2", "chr1", "chr1"])
    starts = numpy.array([300_000, 2_500_000, 1_000_000])
    y_hat = numpy.random.default_rng(0).random((3, 2, 1000), dtype=numpy.float32)

    write_bigwig(["plus.bw", "minus.bw"], chrom_sizes, chroms, starts, y_hat)
    y = read_bigwig(["plus.bw", "minus.bw"], chroms, starts, width=1000)
    print(numpy.array_equal(y, y_hat))
    # True

``write_bigwig`` takes what ``read_bigwig`` gives: windows on any
chromosomes, in any order, and their values, of shape ``(n, width)`` for one
file or ``(n, len(paths), width)`` for several, channel ``i`` going to the
``i``-th file. That is the layout of a model's predictions for the windows,
here for the two strands of a stranded assay. A list of one path still takes
values of shape ``(n, 1, width)``. Windows must not overlap, counting their
full width even where their values are ``missing`` or NaN.
``chrom_sizes`` is a dict of chromosome lengths, or a list of ``(name,
length)`` pairs; ``BigWigReader(path).chrom_sizes`` copies a file's.

A window may run past the end of its chromosome only where its values are
NaN, as ``read_bigwig`` gives them there. Predictions for a window over a
chromosome's end raise a ``ValueError`` until the values past the end are set
to NaN.

Intervals and single bases
--------------------------

With ``ends``, ``write`` and ``write_bigwig`` take intervals and one value
for each, of shape ``(n,)``, or ``(n, len(paths))`` for several files.
Single bases, such as per-base counts, are windows of width 1:

.. code-block:: python

    import numpy
    from figwig import BigWigReader
    from figwig import write_bigwig

    chrom_sizes = {"chr1": 10_000}
    write_bigwig("intervals.bw", chrom_sizes, "chr1", [100, 250], [1.5, 2.0],
        ends=[200, 300])

    positions = numpy.array([1000, 1003, 1004, 2050])
    counts = numpy.array([3, 1, 1, 7], dtype=numpy.float32)
    write_bigwig("counts.bw", chrom_sizes, "chr1", positions, counts[:, None])

    print(BigWigReader("intervals.bw").read("chr1", [195], width=10))
    # [[1.5 1.5 1.5 1.5 1.5 0.  0.  0.  0.  0. ]]
    print(BigWigReader("counts.bw").read("chr1", [1000], width=6))
    # [[3. 0. 0. 1. 1. 0.]]

An interval must end after its start and at most at the length of its
chromosome. :doc:`figwig bam2bw <bam2bw>` makes count tracks from reads.

What is left out
----------------

A base or interval whose value is ``missing``, 0.0 unless given, or NaN, is
left out of the file, so that reading with the same ``missing`` gives the
values back. Where ``missing`` is 0.0, a base whose value is -0.0 is left out
too, and reads back as 0.0. To store zeros as data, write and read with
``missing=numpy.nan``. Values are written as float32, and must be finite or
NaN, and within float32's range.

Writing a batch at a time
-------------------------

.. code-block:: python

    from figwig import BigWigReader
    from figwig import BigWigWriter

    chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
    with BigWigWriter("example.bw", chrom_sizes) as writer:
        writer.write("chr1", [1000, 2000], [0.5, 2.0], ends=[1500, 2100])
        writer.write("chr1", [5000, 5003, 5004], [[3], [1], [1]])
        writer.write("chr2", [10_000], [[0.0, 1.5, 1.5, 0.0, 2.5]])

    y = BigWigReader("example.bw").read(["chr1", "chr2"], [5000, 10_000], width=6)
    print(y)
    # [[3.  0.  0.  1.  1.  0. ]
    #  [0.  1.5 1.5 0.  2.5 0. ]]

``BigWigWriter`` writes values as they come, so that a track larger than
memory, such as a model's predictions over a genome, can be written a batch
of windows at a time. Within one call, windows or intervals may come in any
order. Each call's items come after the last call's: on a chromosome later in
``chrom_sizes``, or on the same one at or after the end of the last call's
last item. A chromosome may be written over many calls.

The order of ``chrom_sizes`` is the order the calls must follow. The
``chrom_sizes`` of an ENCODE or UCSC file is sorted by name (``chr1``,
``chr10``, ..., ``chr19``, ``chr2``), so writing ``chr1``, ``chr2``, ... in
one call each, with those sizes, raises at ``chr10``. Loop over ``writer.chrom_sizes``, or build
``chrom_sizes`` in the order the values come in.

A call that raises writes nothing, and the writer can still be used. The
header, the index and the zoom levels are written when the writer is closed,
at the end of the ``with`` block or by ``close()``, so the file is not a
bigWig until then. An exception inside the ``with`` block leaves the file
without a header, and a reader then says it is not a bigWig file.
``write_bigwig`` writes several files one after another; if writing one
fails, that one is left without a header and the files after it are not
written.

Section types
-------------

figwig lays a bigWig out the way libBigWig, the C library inside pyBigWig,
does: the header, the chromosome tree, data blocks of at most 32,768 bytes
before compression, the R-tree index over them, and the zoom levels with their
indexes. Intervals become bedGraph sections, as they are given. Each window,
or each 65,536 bases of a wider one, becomes whichever section type takes it
in the fewest bytes before compression, counting each data block's header and
index entry: fixedStep of span 1 for runs of bases, which start a new block at
every gap; varStep of span 1 for scattered bases; or bedGraph for stretches of
one value. A track of 10 million bases, 7% of them covered in stretches of 10
to 30 bases of one value, took 0.2 MB this way, against 2.0 MB as fixedStep
alone and 1.1 MB as varStep alone.

Zoom levels
-----------

``zooms``, 10 by default, is the most zoom levels to write. Genome browsers
draw zoomed-out views from them; figwig's reader does not use them. The
levels' sizes are libBigWig's: 16 times the mean width of an item, or 10 bases
if that is more, then 4 times larger at each level, up to the longest
chromosome. figwig skips a level that has no fewer records than the last one
it kept, and goes on to the coarser ones. The levels are built when the writer
is closed, from the data blocks read back from the file, which adds a pass
over the data: writing 15.9 million per-base counts on 8 threads took 0.18 s
without zoom levels and 0.75 s with them (:doc:`benchmarks`). Pass
``zooms=0`` for a file no genome browser will draw.

Compression
-----------

Blocks are compressed with zlib, or with libdeflate when the ``deflate``
package (the ``fast`` extra) is installed and ``engine`` is ``'auto'``, its
default, except on Windows, where libdeflate's functions cannot be loaded.
Both write zlib streams that any bigWig reader can read, and both write the
same file whatever ``n_jobs`` is. libdeflate is faster: on the data blocks of
a track of 15.9 million per-base counts, at level 6 on one thread, it
compressed 142 MB/s to zlib's 32 MB/s, into blocks 0.4% smaller.

``level`` is 6 by default, zlib's own default and what pyBigWig uses; it runs
from 0 to 9 with zlib and 0 to 12 with libdeflate. ``figwig bam2bw`` writes
at libdeflate's level 1, into files 3% larger than pyBigWig's.
``engine='isal'`` uses ISA-L, which takes levels 0 to 3 and is faster still
at levels 1 to 3, but at levels 1 and 2 its output can differ from one run to the next on the same input, so
its files are not reproducible byte for byte; their values are.

Blocks are compressed on up to ``n_jobs`` threads, besides the calling
thread, which lays out the next batch of about a million items meanwhile.
:doc:`threads` shows how the time falls with threads.

Files identical to pyBigWig's
-----------------------------

A file of intervals, or of single bases written as windows of width 1,
written with ``missing=numpy.nan`` so that none is left out, without zoom
levels, and with ``engine='zlib'`` at level 6, is byte for byte the file
pyBigWig writes from the same values, and the tests check that it is.

Where libBigWig writes a wrong value, figwig writes the right one, so those
files differ from pyBigWig's there:

- the maximum in the header, which libBigWig misses when the first value is
  the largest, and leaves at the smallest positive double, about 2.2e-308,
  when no value is positive;
- the end of the last data block of each fixedStep call, which libBigWig puts
  6 bases past its last item;
- the sum and sum of squares of the last zoom record of each zoom block,
  which libBigWig leaves at 0. pyBigWig's mean over a whole chromosome, which
  it takes from the zoom levels, then drifts from the exact one: by 2.2% on
  one chromosome of a test file.

libBigWig also writes an empty data block where it changes section type,
which figwig leaves out, and stops making zoom levels at the first level that
needs as many zoom blocks as the one before it, which can leave a sparse track
with only its finest level.
