Threads, memory and the first call
==================================

figwig's speed comes from threads that run in parallel: zlib's
``uncompress()`` and ``compress2()``, libdeflate and figwig's numba kernels
all run without Python's GIL. This page says how each part uses its threads,
how much they gain, and what memory they take. The measurements are from the
thread sweep on :doc:`benchmarks`, on a 2x AMD EPYC 9575F; the gains on
another machine will differ, and ``benchmarks/compare_readers.py`` and
``benchmarks/compare_writers.py`` in the repository measure them.

Reading
-------

``n_jobs`` is the most threads a read uses, 8 by default, and -1 gives one per
CPU the process may run on. A read sorts its windows, finds the data blocks
that they overlap in the file's index, and splits those blocks into batches of
256. Each batch is read, decompressed and decoded on one thread, straight into
the windows' rows of the output, and a block that several windows of a batch
share is decompressed once. A read therefore uses more threads only when its
windows span more batches, and the result does not depend on ``n_jobs``.

One ``BigWigReader``, opened once and warmed up, read random windows of
1,000 bp from the 238 MB ATAC-seq bigWig, given unsorted, in these times
(median of five):

=================  ==========  ===========  ===========  ============  ============
windows per read   1 thread    2 threads    4 threads    8 threads     32 threads
=================  ==========  ===========  ===========  ============  ============
100                3.5 ms      3.0 ms       3.0 ms       3.0 ms        3.0 ms
1,000              29 ms       24 ms        19 ms        19 ms         18 ms
10,000             235 ms      132 ms       74 ms        52 ms         53 ms
100,000            756 ms      405 ms       221 ms       128 ms        74 ms
=================  ==========  ===========  ===========  ============  ============

A read of a thousand windows or fewer gains little from threads. In a PyTorch
DataLoader the workers already read in parallel, so each can read its batch on
one thread (``n_jobs=1``), as in :doc:`reading`. To read every window at
once, 8 to 32 threads are where the gains were; on the full set of 167,750
windows, 8 threads were 6.1 times faster than one, and 32 were 11 times
faster.

Writing
-------

A writer lays out values on the calling thread in batches of about a million
items, and compresses each batch's data blocks on up to ``n_jobs`` other
threads while it lays out the next one. With libdeflate, the calling thread
becomes the limit: writing 15.9 million per-base counts took 1.07 s on one
thread and 0.18 s on 8, and no less on 16 or 32. zlib compresses about four
times slower, so it keeps gaining up to 32 threads, where it took 0.20 s.
Zoom levels are built when the writer is closed, with the same threads, and
add a pass over the data.

``write`` converts the values of one call a batch at a time as well, so its
working memory does not grow with the call: on 8 threads without zoom levels,
writing windows of 25 million and of 100 million float32 values in one call
took 166 and 173 MB above the values themselves, as tracemalloc counted it.
The finest zoom level's blocks are written as they are compressed. The other
levels' are kept compressed in memory until every level is built, since
whether a level is written depends on how many records it has, and so are the
finest level's where Python has no ``os.pread``, as on Windows. With zoom
levels, peak memory grew with threads in the sweep: 580 MB on one thread and
1,028 MB on 32 for the counts track, against about 580 MB at every thread
count without them.

figwig bam2bw
-------------

``-p`` is a number of cores, 1 by default. A BAM file's BGZF blocks are
inflated by libdeflate on ``-p`` threads, and its records walked by a numba
kernel, and a BED/tsv file that is BGZF or not compressed is scanned on
``-p`` threads. On a 2.4 GB ATAC-seq BAM, ``figwig bam2bw`` took 9.35 s at
``-p 1``, 1.74 s at ``-p 8``, and no less at ``-p 16`` or ``-p 32``. Peak
memory grew a little with ``-p``, from 831 MB to 1,032 MB. Some files take
one core each, whatever ``-p`` is: SAM files, ``--mate_pairs`` and remote
files, which pysam's loop reads, and gzipped BED/tsv files that are not BGZF,
which must be inflated in order before the numba kernel scans them.
:doc:`bam2bw` says how files are shared out. For many files, several
processes at a low ``-p`` use the cores better than one at a time at a high
``-p``, and each should run with ``OPENBLAS_NUM_THREADS=1``: numpy's OpenBLAS
otherwise starts a thread per CPU when it is imported, which costs several
CPU-seconds per process on a many-core machine.

Memory of a read
----------------

The output takes ``n * width * 4`` bytes for one file, times the number of
files for several: 671 MB per file for 167,750 windows of 1,000 bp. Each
thread also holds the decompressed blocks of the batch it is reading: 256
blocks and any more that its windows run into, each at most the file's
``uncompressBufSize``, which is 32 KB in ENCODE's files. ``out=`` reuses one
output array from read to read.

Threads across processes
------------------------

Threads multiply across processes: 8 DataLoader workers at the default
``n_jobs=8`` start up to 64 threads. On a shared machine, keep the number of
processes times ``n_jobs`` within the cores that are free.

The first call
--------------

The first read in a new environment compiles figwig's numba kernels, which
took 0.7 s on the machine above, and 1.1 s for ``figwig bam2bw``. numba
caches them on disk, next to the installed package, or in the user's cache
directory when the package directory cannot be written, and later processes
load them in well under a second. Warm up with a small read before timing
figwig. One ``BigWigReader`` can be read from several threads at once, and
holds no open file between reads.

Platforms
---------

figwig needs a little-endian machine, which every common one is, and Python
3.10 to 3.14. Its tests run on Linux under each of those versions, and on
macOS and Windows under 3.10 and 3.13. On Windows, Python has no
``os.pread``, so blocks are read by seeking the file under a lock. Where
zlib's library cannot be loaded, as on Windows, blocks are decompressed and
compressed by Python's zlib module instead, with the same values. libdeflate's
functions cannot be loaded from the ``deflate`` package on Windows either, so
there ``engine='auto'`` is zlib and ``engine='libdeflate'`` raises a
``ValueError``. pyBigWig, which the writer's tests compare files with byte for
byte, does not build on Windows, so those comparisons run on Linux and macOS.
