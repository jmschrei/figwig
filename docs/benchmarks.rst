Benchmarks
==========

Every measurement on this page was taken on one machine, 2x AMD EPYC 9575F
(256 cores), with Python 3.13.5, numpy 2.5.3 and numba 0.68. ``benchmarks/compare_readers.py`` and
``benchmarks/compare_writers.py`` in the repository run the reader and writer
comparisons on any bigWigs and BED files.

Reading
-------

The test read 167,750 windows of 1,000 bp, centred on the fold-0 training
peaks and negatives of ENCODE ATAC-seq experiment ENCSR123WME, from two
public bigWigs held on tmpfs: the ATAC-seq signal of HG02943 (ENCFF830RWF,
238 MB) and the DNase-seq signal of dorsolateral prefrontal cortex
(ENCFF989SAK, 645 MB). Of the 167.75 million bases, 11.9 million are nonzero
in the ATAC file and 19.0 million in the DNase file.

Each configuration ran in its own process, three times, and the table gives
the median, file opening included. To give the other libraries more threads,
the windows were split into 8 chunks read on a thread pool, each chunk with
its own file handle. The table gives every library the windows sorted by
position, which helps the others' caching. figwig sorts them itself: given
unsorted, it took 0.144 s and 0.292 s on 8 threads. Every library returned
the same values on every window, once pyBigWig's NaN for bases without an
interval was set to 0.

===========================================  ==============  ===============  ===============  ================
library                                      ATAC, 1 thread  ATAC, 8 threads  DNase, 1 thread  DNase, 8 threads
===========================================  ==============  ===============  ===============  ================
pybigtools 0.2.5, ``values()`` per window    1.25 s          1.42 s           1.98 s           2.21 s
pyBigWig 0.3.26, ``values()`` per window     3.95 s          4.08 s           4.50 s           4.64 s
pybbi 0.4.2, ``stackup``                     9.61 s          9.48 s           10.75 s          10.63 s
figwig 0.1.0                                 0.81 s          **0.135 s**      1.85 s           **0.277 s**
===========================================  ==============  ===============  ===============  ================

On one thread figwig took 0.81 s to pybigtools' 1.25 s on the ATAC file, and
1.85 s to 1.98 s on the denser DNase file. Its advantage is that its threads
run in parallel.

Writing
-------

The writers wrote the values of two tracks again, held in memory as numpy
arrays: the 5' ends of the reads of ENCODE ATAC-seq BAM ENCFF877LRY on the
plus strand, as counts at 15,875,960 single bases, and the fold-change signal
of ENCODE snATAC-seq pseudobulk ENCSR206UWN, ENCFF932UQM, as 21,212,477
intervals. figwig wrote one ``BigWigWriter.write`` per chromosome, single
bases as windows of width 1, pyBigWig one ``addEntries`` per chromosome, and
pybigtools one ``write()`` from an iterator of tuples. Every write, opening
the file and closing it included, ran in its own process after a warm-up
write of the first 1,000 values of each chromosome, three times, onto tmpfs,
and the table gives the median, at compression level 6. pybigtools 0.2.5's
``write()`` takes no option for zoom levels and writes them, so it was timed
with them only. The libraries were pyBigWig 0.3.26, pybigtools 0.2.5 and
deflate 0.9.0. Every writer's file gave the same values on 2,000 windows of
1,000 bases read back with figwig.

================================  ==========  =====================  ==========  =====================
writer                            counts      counts, zoom levels    signal      signal, zoom levels
================================  ==========  =====================  ==========  =====================
pyBigWig 0.3.26                   4.20 s      12.80 s                7.41 s      10.48 s
pybigtools 0.2.5                              4.45 s                             5.47 s
figwig, zlib, 1 thread            4.25 s      12.09 s                6.21 s      8.10 s
figwig, zlib, 8 threads           0.53 s      1.63 s                 0.78 s      1.09 s
figwig, libdeflate, 1 thread      1.07 s      4.85 s                 1.80 s      3.27 s
figwig, libdeflate, 8 threads     **0.18 s**  **0.75 s**             **0.30 s**  **0.55 s**
================================  ==========  =====================  ==========  =====================

figwig's files were the size of pyBigWig's: 35.7 and 107.1 MB without zoom
levels, against 35.8 and 108.0 MB, and 214.0 and 126.5 MB with them, against
213.9 and 127.4 MB. pybigtools chooses its zoom levels differently, and wrote
96.4 and 162.2 MB. Building zoom levels took from a quarter to three quarters
of figwig's time. Its peak memory was above pyBigWig's without zoom levels,
554 MB against 354 MB for the counts on one thread, of which the arrays of
values held 318 MB, and below it with them, 739 MB on 8 threads against
1,089 MB.

figwig bam2bw
-------------

Two inputs
~~~~~~~~~~

``figwig bam2bw`` and bam2bw 0.5.1 converted the reads of ENCODE ATAC-seq BAM
ENCFF877LRY (2.4 GB) with default flags, to two stranded bigWigs, and the
18.9 million scATAC-seq fragments of ``Alpha_unt.fragments.tsv.gz`` with
``-f -u``, to one bigWig. Each call ran in its own process after a warm-up
call on a tiny input and with the input in the page cache, with
``OPENBLAS_NUM_THREADS`` set to ``-p``, and its peak memory is that of the
process and its children, sampled every 50 ms. The table gives figwig's
median of five runs at ``-p 2`` and one run at ``-p 8``, and one run of
bam2bw, which reads one file in one process whatever ``-p`` is. pysam was
0.24.1, and the files were on a disk rather than tmpfs.

=======================  ==========  ==========  ================  ======================  ===================
\                        BAM         fragments   peak memory, BAM  peak memory, fragments  bigWigs, both calls
=======================  ==========  ==========  ================  ======================  ===================
bam2bw 0.5.1, ``-p 2``   58.48 s     28.45 s     3,039 MB          2,866 MB                135.5 MB
figwig bam2bw, ``-p 2``  **4.65 s**  **1.73 s**  910 MB            1,061 MB                139.8 MB
figwig bam2bw, ``-p 8``  **1.82 s**  **1.31 s**  949 MB            1,180 MB                139.8 MB
=======================  ==========  ==========  ================  ======================  ===================

figwig's bigWigs are 3% larger because they are compressed at level 1
rather than pyBigWig's level 6. The first call in a new environment also
compiles figwig's kernels, which took 1.1 s on the tiny input.

Every file of a test-data collection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both also ran on every file of a collection of public test data that bam2bw
takes: 591 BAM, SAM, BED and TSV files from 28 assay directories, from 3
records to a 10 GB ATAC-seq BAM, each under 5 to 9 sets of flags. Each run was
timed once, 16 at a time, interpreter start-up included. In all 3,781 runs
that bam2bw completed, ``figwig bam2bw`` gave the same exit code, messages
and decoded bigWig entries.

.. image:: figures/bam2bw-timings.png
   :alt: Wall time of figwig bam2bw against bam2bw 0.5.1, log scales
   :width: 600px

In the 155 runs where bam2bw took a second or more, ``figwig bam2bw`` was a
median 3.2 times faster, and 7 times faster on the 10 GB BAM at ``-p 1``
(40 s against 282 s), or 27 times at ``-p 4``. In the 3,511 runs that bam2bw
finished in under half a second, it took a median 0.12 s longer, most of it
importing numba and loading its compiled kernels. With ``--mate_pairs``
(hollow points) both read with pysam's loop and take the same time. The
points well above the line are TSV files that are not coordinates, such as
quantification tables and 10x feature lists, whose every line names a
different sequence that is not in the sizes file: the fast reader returns to
Python for each new name, where bam2bw's loop skips the line.

The runs were repeated with their memory measured: the peak is the larger of
the whole process tree's resident set, sampled every 20 ms, and the largest
single process's exact peak (GNU time's maximum resident set size); the mean
is the process tree's resident set averaged over the run.

.. image:: figures/bam2bw-memory.png
   :alt: Peak and mean memory of figwig bam2bw against bam2bw 0.5.1, log scales
   :width: 750px

Where bam2bw's peak was under 250 MB, ``figwig bam2bw``'s start-up held about
55 MB more, numba and its compiled kernels: a median of 98 MB against 43 MB.
On the 10 GB BAM its peak was 4.3 GB against bam2bw's 11.6 GB, and 7.5 GB
against 20.2 GB with ``-u -f``; with ``--mate_pairs``, read by pysam's loop in
both, the two were the same (6.6 and 6.7 GB). These runs were made before
``figwig bam2bw`` dropped the keys it had counted while still reading a BAM,
which lowers its peak on BAM files: on the 10 GB BAM at ``-p 1``, run one at a
time, it is now 2.9 GB, and 4.9 GB with ``-u -f``, in the same time.

figwig files
------------

``benchmarks/compare_formats.py`` times a figwig file against the bigWigs
of the same tracks. The tracks were six genome-wide count tracks on hg38:
the two strands of the 5' ends of ENCODE ATAC-seq BAM ENCFF877LRY, written
by ``figwig bam2bw``; the two strands of PRO-cap, ENCFF994CSC and
ENCFF253HUA; and the unstranded cut counts of ATAC-seq ENCFF830RWF and
DNase-seq ENCFF989SAK. Past six, track i is track i mod 6 with every
chromosome rolled by a random offset, written as a bigWig by figwig, so
that it has the statistics of a real track and shares no position with it.
``figwig bw2fw`` wrote the figwig file of each set, in groups of 8 tracks
at a tile of 16,384 bases and compression level 6, and that conversion and
``figwig fw2bw`` back are the write timings, one run each; every file was
on tmpfs.

The read was 8,192 windows of 2,114 bp centred on the summits of the K562
ATAC-seq IDR peaks ENCFF855PCP, drawn at random, in one call:
``read_bigwig`` over the N bigWigs into float32, and ``FigWigReader.read``
into the tracks' own dtype (uint8 or uint16) and into float32. Each
configuration ran in its own process, which warmed up on 1,000 windows,
then opened the files again and timed one read, opening included; the
table gives the median of three. Every configuration of one N gave the same
values.

======  =======  =============  ==========  ==========
tracks  threads  bigWigs, f32   figwig      figwig, f32
======  =======  =============  ==========  ==========
1       1        0.191 s        0.057 s     0.064 s
1       8        0.040 s        0.022 s     0.027 s
1       32       0.028 s        0.034 s     0.032 s
2       1        0.393 s        0.095 s     0.103 s
2       8        0.065 s        0.026 s     0.037 s
2       32       0.038 s        0.043 s     0.037 s
6       1        0.921 s        0.254 s     0.279 s
6       8        0.152 s        0.054 s     0.057 s
6       32       0.115 s        0.046 s     0.052 s
24      1        3.351 s        0.666 s     0.769 s
24      8        0.516 s        0.122 s     0.151 s
24      32       0.312 s        0.116 s     0.124 s
96      1        13.093 s       2.415 s     2.757 s
96      8        1.930 s        0.425 s     0.498 s
96      32       1.087 s        0.406 s     0.432 s
======  =======  =============  ==========  ==========

On one thread the figwig file read 3.4 to 5.4 times faster than the
bigWigs, and on 8 threads 1.8 to 4.5 times. A read of 8,192 windows from a
figwig file is done in tens of milliseconds, and gained nothing past 8
threads, so at 16 and 32 threads one or two bigWigs read as fast as the
figwig file; at 24 and 96 tracks the figwig file stayed 2.7 times faster at
32 threads. Reading into float32 costs a few percent over the tracks' own
types, which are half or a quarter of the bytes.

The pattern of a PyTorch DataLoader worker, 32 batches of 64 windows read
one after another on one thread, is where the figwig file gains most, since
a batch of windows on a sparse track shares few bigWig blocks:

======  ===============  ===========
tracks  bigWigs, f32     figwig
======  ===============  ===========
1       0.127 s          0.018 s
2       0.253 s          0.029 s
6       0.538 s          0.070 s
24      2.507 s          0.185 s
96      10.458 s         0.661 s
======  ===============  ===========

Sizes, and the conversions each way, one run each:

======  ==============  =============  ====================================  ====================================
tracks  bigWigs         figwig file    ``bw2fw``, 1 / 8 / 32 threads         ``fw2bw``, 1 / 8 / 32 threads
======  ==============  =============  ====================================  ====================================
1       36.9 MB         30.1 MB        6.2 s / 2.6 s / 3.1 s                 6.1 s / 5.5 s / 7.9 s
2       73.9 MB         54.0 MB        11.4 s / 4.6 s / 5.2 s                12.8 s / 8.4 s / 8.1 s
6       971.4 MB        266.4 MB       40.5 s / 12.8 s / 12.7 s              44.8 s / 28.8 s / 20.4 s
24      2,195.1 MB      1,054.6 MB     152.2 s / 38.6 s / 38.6 s             178.0 s / 109.2 s / 74.9 s
96      7,089.9 MB      4,219.7 MB     604.0 s / 149.4 s / 150.0 s           717.5 s / 441.9 s / 313.7 s
======  ==============  =============  ====================================  ====================================

The six real tracks are 3.6 times smaller as a figwig file, most of the
gain on the two ENCODE tracks written as bedGraph, 883 MB of the 971; the
rolled copies past six are bigWigs figwig wrote as varStep, and the figwig
file is 0.6 times their size. ``bw2fw`` reads the bigWigs and encodes whole
tiles on its threads and gained nothing past 8; ``fw2bw`` is bound by
laying each track's windows out as bigWig sections, on one thread per
track. ``figwig fw2bw`` wrote the six tracks back as 408 MB of bigWigs.

``figwig bam2fw`` and ``figwig bam2bw`` converted the 2.4 GB ATAC-seq BAM
ENCFF877LRY, on tmpfs, to one figwig file of two tracks and to two bigWigs,
median of three:

=================  ========  ========  ========  =========
command            1 core    8 cores   32 cores  output
=================  ========  ========  ========  =========
``figwig bam2bw``  9.42 s    1.88 s    1.84 s    73.9 MB
``figwig bam2fw``  11.79 s   4.13 s    4.05 s    53.9 MB
=================  ========  ========  ========  =========

Both read the BAM the same way; ``figwig bam2fw`` spends the difference
merging the two strands' positions and encoding their tiles, which runs on
the calling thread.

Thread scaling
--------------

The reader, the writer and ``figwig bam2bw`` ran again at 1 to 32 threads,
one run at a time, on the inputs above, while other jobs kept the machine's
load at about 20: the reader through
``compare_readers.py`` on the ATAC and DNase files, three times; the writer
through ``compare_writers.py`` on the counts and signal tracks, three times;
and ``figwig bam2bw`` on the BAM and the fragments, three times, with
GNU time's maximum resident set size as its peak memory. Each figure is a
median. Every run of each input gave the same values, and ``figwig bam2bw``
the same bytes, whatever the number of threads.

.. image:: figures/thread-scaling.png
   :alt: Wall time against threads for reading, writing and figwig bam2bw, log scales
   :width: 100%

Reading 167,750 windows of 1,000 bp, sorted by position, file opening
included:

==============  ========  =========  =========  =========  ==========  ==========
bigWig          1 thread  2 threads  4 threads  8 threads  16 threads  32 threads
==============  ========  =========  =========  =========  ==========  ==========
ATAC, 238 MB    0.816 s   0.427 s    0.228 s    0.133 s    0.088 s     0.074 s
DNase, 645 MB   1.852 s   0.952 s    0.499 s    0.272 s    0.180 s     0.164 s
==============  ========  =========  =========  =========  ==========  ==========

Given unsorted, the same reads took 2% to 11% longer. :doc:`threads` has the
times by the number of windows per read.

Writing, at level 6, file opening and closing included:

===============================  ========  ========  ========  ========  ========  ========
track, engine, zoom levels       1         2         4         8         16        32
===============================  ========  ========  ========  ========  ========  ========
counts, libdeflate, none         1.07 s    0.47 s    0.25 s    0.18 s    0.19 s    0.23 s
counts, zlib, none               4.22 s    2.04 s    1.04 s    0.53 s    0.28 s    0.20 s
counts, libdeflate, 10           4.85 s    2.50 s    1.37 s    0.75 s    0.53 s    0.55 s
counts, zlib, 10                 12.11 s   6.08 s    3.16 s    1.63 s    0.87 s    0.55 s
signal, libdeflate, none         1.78 s    0.77 s    0.41 s    0.30 s    0.32 s    0.34 s
signal, zlib, none               6.18 s    2.97 s    1.52 s    0.77 s    0.42 s    0.32 s
signal, libdeflate, 10           3.25 s    1.64 s    0.85 s    0.56 s    0.50 s    0.55 s
signal, zlib, 10                 8.11 s    4.16 s    2.06 s    1.09 s    0.61 s    0.52 s
===============================  ========  ========  ========  ========  ========  ========

``figwig bam2bw``, with peak memory:

====================  ================  ================  ================  ================  ================  ================
input                 ``-p 1``          ``-p 2``          ``-p 4``          ``-p 8``          ``-p 16``         ``-p 32``
====================  ================  ================  ================  ================  ================  ================
BAM, default flags    9.35 s, 831 MB    4.65 s, 874 MB    2.71 s, 888 MB    1.74 s, 925 MB    1.75 s, 968 MB    1.82 s, 1,032 MB
fragments, ``-f -u``  2.91 s, 1,039 MB  1.74 s, 1,326 MB  1.41 s, 1,386 MB  1.25 s, 1,496 MB  1.30 s, 1,414 MB  1.41 s, 1,495 MB
====================  ================  ================  ================  ================  ================  ================
