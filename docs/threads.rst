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

A figwig file is read the same way, in batches of chunks on up to ``n_jobs``
threads, each batch read with one ``pread`` per run of adjacent chunks and
inflated by libdeflate, or by zlib where the ``deflate`` package is not
installed; where libc's ``pread`` and libdeflate both load, as everywhere
but Windows, a batch is read, checked and inflated in one compiled call that
holds no GIL. A read of fewer than one in 16 of a group's tracks, as one
cell's of a file of thousands, checks only those tracks' offsets and passes
over the chunks in which none of them has a base. A read uses one thread per
64 chunks it touches, up to ``n_jobs``, since a read of a few dozen windows
is done in under a millisecond and gains nothing from waking threads, and no
more than one per 256 KB of the chunks it decodes and per 2 MB of a dense
output it fills, but at least 4 where it asks for that many, since a read of
a few hundred KB of chunks took longer on 16 or 32 threads than on 2 to 8,
while a read summed per bin over fewer windows than threads keeps them all,
to split its windows' batches, as does a float sum that they would cut
across tasks, whose sums in a copy of the output per thread depend on the
threads; it
keeps its threads between reads, and on a read of several groups each thread
takes the next batch as it finishes one. Windows packed into a few tiles are
cut over the threads by window rather than by chunk, and a read of few large
chunks, as dense float tracks have, is cut into at least two batches per
thread rather than batches of 32 MB. A read of many small groups, whose own
two batches per thread would be a chunk or two each, takes batches of up to
128 KB per thread of the whole read's chunks instead, since a batch costs
the calling thread its dispatch whatever its size. On the six genome-wide
count tracks of :doc:`benchmarks`, 8,192 windows took 0.12 s on one thread
and 0.05 s on 8, and nothing more was gained past 8; on three dense
conservation tracks over chr22 they took 0.51, 0.070 and 0.034 s on 1, 8 and
32 threads. Sums per label run a group's batches at a time, each writing its
own cells of the output, so they take no copy of the output per thread in
position order; sums per bin without labels, as ``sums`` takes, run every
group's batches at once, since each group's tracks have their own cells,
unless each group has one batch and together they hold less than 1 MB of
chunks. In track order sums take a copy of the output per thread, which the
8 GiB guard counts. In position order ``sums(..., paired=True)`` sums its
batches of tracks side by side, a thread each, where their sums are exact in
any order, as integer tracks' are, and over integer tracks and intervals of
a few tiles each it is one read, each interval with its own track's chunks
alone. A read with labels over several groups of integer tracks runs in
bands of windows side by side, where it ran a group at a time. A few long
windows summed per bin, as a genome browser's views are, are read as parts
of whole bins, a part per thread, whose float sums differ from one thread's
only in rounding. The indexes a read needs load on the calling thread
while each spans fewer than 131,072 tiles, since threads that lay out small
indexes mostly wait for each other, and from the first that spans more the
rest load on up to 8 threads, each holding the one it lays out: a process
running a browser's session over six genome-wide count tracks peaked at
169 MB on 8 threads, where loading them one at a time peaked at 146 MB.

Writing
-------

A ``FigWigWriter`` encodes whole tiles that a ``write`` call's windows
cover on its pool, in up to one task per thread of at least 4 MB of tiles
per call, and intervals and windows' partial tiles there too, but for a
tile that an earlier call left open or a later one may continue and a
single partial tile, which it encodes on the calling thread; it compresses
on the pool, and holds about 0.5 GB plus two to three times the call's
values, so a call of a few hundred MB at a time is the size to aim for;
``bw2fw`` and ``fw2bw`` read each piece of a chromosome on a thread of
their own while the writers take the last, and ``bw2fw`` stops gaining at
``-p 8``. A bigWig writer takes values in
batches of about a million items on one thread, or 786,432 with threads,
where a planning thread lays each batch out as data blocks, up to ``n_jobs``
pool threads encode and compress them, and a summary thread folds the batch
into the file's totals in file order, so the calling thread only takes the
next batch, with at most two batches in flight. With libdeflate the encoding
is the limit: writing 15.9 million per-base counts took 1.03 s on one thread and 0.14 s on 8, against 1.11 and 0.21 s when the calling thread laid out the blocks (:doc:`benchmarks`). zlib
compresses about four times slower, so it keeps gaining up to 32 threads,
where it took 0.20 s.
Zoom levels are built when the writer is closed, with the same threads, and
add a pass over the data.

``write_sparse`` sorts a call's entries by a radix sort on the pool, from
above a tile's offsets where the call's positions ascend, and takes entries
already in the writer's order without sorting them; it runs its passes over
the entries, its range and dtype checks and its search for two entries on
one base, in parts on the pool, and each task of entries is encoded and
compressed in two compiled calls that hold no GIL. ``write`` checks its
values for infinity in parts on the pool, and windows' values against
declared integer types too, and copies each group's block once, every
group's of a call in one pass on the pool, in waves of at most 4 MB per
thread, the next copied once the tiles of all but the last are encoded, so
that a call holds two waves of copies at most; the calling thread takes a
part itself where the pool's threads are busy, and they join in as they
come free. A writer with threads writes its chunks and blocks to the file
in order on a thread of its own, with at most 64 MB queued for it, so that
the calling thread does not wait on the file between batches. ``fw2bw``
writes the bigWigs of a group of at most ``-p`` tracks each on a thread of
its own, with at most two pieces of the group in flight, and those of a
larger group ``-p`` at a time, each piece whole before the next is read.

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
took 1.1 s for the bigWig reader on the machine above, and 1.8 s for
``figwig bam2bw``. A figwig file's kernels are larger, and compiled once per
element type of a chunk and per output dtype. Alone in a new process,
the first dense read compiled for 21 s, a sparse read for 22 s, a sum per
label for 21 s, ``read_entries`` for 20 s, a track-order read for 3.6 s
and a write for 7.9 s, each with no output while it compiles. Kinds of call
share kernels, so a process that reads and writes every way compiled for
87 s in all: after its dense read of float32 values, a sparse read of
uint16 values took 19 s more, a sum per label 19 s, ``read_entries`` 17 s,
a track-order read 2.1 s and a write 8.7 s.
numba caches them on
disk, next to the installed package,
or in the user's cache directory when the package directory cannot be
written, and later processes load them in a tenth of a second. Warm up with
a small read before timing figwig, and before starting DataLoader workers
under ``spawn``, each of which loads the cache. One ``BigWigReader`` or
``FigWigReader`` can be read from several threads at once, and holds no
open file between reads; a ``FigWigReader`` keeps each group's index, 17
bytes per tile of the genome, from the first read that touches it.

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
