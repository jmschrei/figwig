Reading bigWigs
===============

figwig reads the per-base values of many windows of one width in one call,
straight into a float32 numpy array, from one bigWig or from several at once.
The examples on this page read two public ENCODE bigWigs: the ATAC-seq signal
of HG02943 (ENCFF830RWF, 238 MB) and the DNase-seq signal of dorsolateral
prefrontal cortex (ENCFF989SAK, 645 MB).

.. code-block:: bash

    wget https://www.encodeproject.org/files/ENCFF830RWF/@@download/ENCFF830RWF.bigWig
    wget https://www.encodeproject.org/files/ENCFF989SAK/@@download/ENCFF989SAK.bigWig

Reading windows
---------------

.. code-block:: python

    import numpy
    from figwig import BigWigReader

    bw = BigWigReader("ENCFF830RWF.bigWig")
    print(len(bw.chrom_sizes), bw.chrom_sizes["chr1"])
    # 149 248956422

    chroms = numpy.array(["chr1", "chr1", "chr2"])
    starts = numpy.array([1_000_000, 2_500_000, 300_000])
    y = bw.read(chroms, starts, width=1000, n_jobs=8)
    print(y.shape, y.dtype)
    # (3, 1000) float32
    print(y.sum(axis=1))
    # [116.  43.  10.]

Coordinates are 0-based and half-open, as in BED files: window ``j`` covers
the bases ``starts[j]`` to ``starts[j] + width - 1`` on ``chroms[j]``, and is
written into row ``j``, whatever order the windows are given in. figwig sorts
them by position itself. The windows may overlap or repeat. ``chroms`` can
also be a single name for every window, and both can be lists, numpy arrays
or pandas Series. ``starts`` must be integers: a pandas column that held a
NaN is float64, and raises a ``TypeError`` until it is cast with
``.astype("int64")``. A window that starts before 0 raises a ``ValueError``,
so windows centred on summits near the start of a chromosome need clipping.
Every window has the same width; regions of several lengths can be read at
the longest width and masked, or with one call per width.

``BigWigReader`` reads the file's header and chromosome tree when it is
opened, and its data index on its first read, and keeps the index, so reading
one file many times, as a data loader does, pays for the index once.
``chrom_sizes`` holds the length of each chromosome, in the order of the
file's chromosome tree: sorted by name (``chr1``, ``chr10``, ``chr11``,
...) in ENCODE's and UCSC's files, and in the order they were given to the
writer in figwig's and pyBigWig's. One ``BigWigReader`` can be read from several
threads at once, and holds no open file between reads.

Reading several bigWigs into one array
--------------------------------------

.. code-block:: python

    import numpy
    from figwig import read_bigwig

    chroms = numpy.array(["chr1", "chr1", "chr2"])
    starts = numpy.array([1_000_000, 2_500_000, 300_000])
    y = read_bigwig(["ENCFF830RWF.bigWig", "ENCFF989SAK.bigWig"], chroms, starts,
        width=1000)
    print(y.shape)
    # (3, 2, 1000)
    print(y.sum(axis=2))
    # [[116. 351.]
    #  [ 43. 240.]
    #  [ 10.  63.]]

Channel ``i`` holds the values from the ``i``-th file, in the
``(batch, channels, length)`` layout that sequence models take: the plus and
minus strands of a stranded assay, say, or one track per task. Reading into
that array directly avoids stacking one array per file, which takes time and
twice the memory. Every file's work shares one pool of threads.

Given one path or one ``BigWigReader``, ``read_bigwig`` gives
``(n, width)``, as ``BigWigReader.read`` does. Given a list or tuple, even of
one file, it gives ``(n, len(bigwigs), width)``.

Reading the same files repeatedly
---------------------------------

A path given to ``read_bigwig`` is opened, and its data index read, on every
call. To read the same files repeatedly, pass ``BigWigReader`` objects
instead, which keep their indexes. Both functions take ``out=``, a writeable,
C-contiguous float32 array of the result's shape to fill rather than
allocate a new one each time:

.. code-block:: python

    import numpy
    from figwig import BigWigReader
    from figwig import read_bigwig

    readers = [BigWigReader("ENCFF830RWF.bigWig"), BigWigReader("ENCFF989SAK.bigWig")]
    starts = numpy.arange(1_000_000, 2_000_000, 1000)
    out = numpy.empty((100, 2, 1000), dtype=numpy.float32)
    totals = numpy.zeros(2)
    for i in range(0, len(starts), 100):
        read_bigwig(readers, "chr1", starts[i:i + 100], 1000, out=out)
        totals += out.sum(axis=(0, 2))
    print(totals)
    # [ 28857. 195167.]

Values and coordinates
----------------------

Every value is copied from the file's float32, not computed.

==========================================  ===================================
base                                        value
==========================================  ===================================
covered by an interval                      the interval's value
covered by no interval                      ``missing``, 0.0 unless given
covered by an interval whose value is NaN   ``missing``
past the end of its chromosome              NaN
on a chromosome the file does not have      ``missing``, with a warning
==========================================  ===================================

The first four rows are what pybigtools' ``values(chrom, start, end,
missing=missing)`` gives, cast to float32, and the tests check figwig against
it bit for bit. pybigtools raises for a chromosome the file does not have.
pyBigWig's ``values()`` gives NaN for a base that no interval covers, which
``missing=numpy.nan`` matches, and raises for a window that runs past the end
of its chromosome or is on a chromosome the file does not have.

.. code-block:: python

    import numpy
    from figwig import BigWigReader

    bw = BigWigReader("ENCFF830RWF.bigWig")
    y = bw.read("chr1", [1_000_000], width=1000, missing=numpy.nan)
    print(numpy.isnan(y).sum(), (y == 0).sum())
    # 907 0

    end = bw.chrom_sizes["chr1"]
    print(bw.read("chr1", [end - 3], width=6))
    # [[ 0.  0.  0. nan nan nan]]

Chromosomes the file does not have
----------------------------------

A window on a chromosome that the file does not have is ``missing``
throughout, and one warning per read names the chromosomes:

.. code-block:: python

    from figwig import BigWigReader

    bw = BigWigReader("ENCFF830RWF.bigWig")
    y = bw.read(["1", "2"], [1_000_000, 300_000], width=1000)

.. code-block:: text

    UserWarning: 2 windows are on chromosomes not in ENCFF830RWF.bigWig, and are
    0.0 throughout: '1', '2'. Its chromosomes include 'chr1', 'chr10', 'chr11'.

Writers leave out chromosomes without data, so a sample without reads on chrY,
say, has no chrY in its file, and that is not an error. A different naming
scheme (``1`` against ``chr1``) looks the same, which is why the warning
lists some of the file's own names. ``set(chroms) - set(bw.chrom_sizes)``
finds the names before reading. Under pytest's ``filterwarnings = error``,
the warning is an exception.

Binned values
-------------

.. code-block:: python

    from figwig import BigWigReader

    bw = BigWigReader("ENCFF830RWF.bigWig")
    y = bw.read(["chr1", "chr1", "chr2"], [1_000_000, 2_500_000, 300_000],
        width=1024)
    binned = y.reshape(len(y), -1, 32).mean(axis=-1)
    print(binned.shape)
    # (3, 32)

figwig reads every base and does not use a file's zoom levels. A mean over
bins of bases, as above, gives binned targets; use ``numpy.nanmean`` if
windows run past the end of a chromosome.

Training with PyTorch
---------------------

.. code-block:: python

    import numpy
    import torch
    from figwig import BigWigReader

    class Windows(torch.utils.data.Dataset):
        def __init__(self, path, chroms, starts, width):
            self.bw, self.chroms, self.starts = BigWigReader(path), chroms, starts
            self.width = width
            self.bw.read(chroms[:1], starts[:1], width)   # reads the index once

        def __len__(self):
            return len(self.starts)

        def __getitem__(self, idx):                        # a batch of indices
            return torch.from_numpy(self.bw.read(self.chroms[idx], self.starts[idx],
                self.width, n_jobs=1))

    if __name__ == '__main__':
        starts = numpy.arange(1_000_000, 2_000_000, 1000)
        chroms = numpy.full(len(starts), "chr1")
        data = Windows("ENCFF830RWF.bigWig", chroms, starts, 1000)
        batches = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(data),
            batch_size=64, drop_last=False)
        loader = torch.utils.data.DataLoader(data, sampler=batches, batch_size=None,
            num_workers=2, persistent_workers=True, multiprocessing_context="spawn")
        y = torch.cat(list(loader))
        print(y.shape, float(y.sum()))
        # torch.Size([1000, 1000]) 28857.0

Each item is a whole batch, read in one call: the ``BatchSampler`` hands
``__getitem__`` a new mix of indices each epoch. A ``BigWigReader`` can be
pickled, carrying its index if it has read it, so it can go to DataLoader
workers under any start method; reading once in the main process, as
``__init__`` does here, saves each worker from reading the index again. The
workers already read in parallel, so each reads its batch on one thread;
:doc:`threads` has the numbers behind that choice. For several tracks, keep a
list of readers and call ``read_bigwig`` with it.

When every window fits in memory, reading all of them once is faster still,
and ``torch.from_numpy`` wraps the result without a copy: 88,971 windows of
1,000 bp from both bigWigs above (712 MB) took 0.27 s on 8 threads, where an
epoch read by 4 workers, one batch of 64 at a time, took about 1.7 s.

What it reads, and what it refuses
----------------------------------

figwig reads bigWig files with bedGraph, varStep or fixedStep sections,
compressed or not, at base-pair resolution. Everything else raises a
``ValueError`` saying what was found, rather than being guessed at:

- a file that is not a little-endian bigWig, such as a bigBed or a big-endian
  bigWig;
- a chromosome tree or data index that is corrupt, or cut short by a
  truncated file;
- a data index whose entries are unsorted or span two chromosomes;
- a window that starts before 0 or ends past 2**32 - 1;
- a window overlapping a data block that cannot be decompressed, holds a
  section of another type, or has intervals that are unsorted, overlap each
  other or those of a neighbouring block, or lie outside the block's index
  entry. Overlapping intervals give a base two values, and readers disagree
  on which to report: pybigtools sums them.

A read that raises because a data block cannot be read may already have
written other windows into ``out``. figwig does not read bigWigs over HTTP: a
URL is taken for a path, and raises ``FileNotFoundError``. It does not return
binned summaries or use the zoom levels when it reads.

Falling back to another reader
------------------------------

.. code-block:: python

    import numpy
    import pybigtools
    from figwig import BigWigReader

    def read(path, chroms, starts, width):
        try:
            return BigWigReader(path).read(chroms, starts, width)
        except ValueError:
            bw = pybigtools.open(path)
            return numpy.array([bw.values(chrom, start, start + width) for chrom,
                start in zip(chroms, starts)], dtype=numpy.float32)

figwig's tests compare it against pybigtools' ``values()``, bit for bit,
which makes pybigtools a natural fallback for the files figwig refuses.
