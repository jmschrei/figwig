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

The figwig file was measured against the other ways of holding many
tracks on 2026-10-23 (``claude_comments/2026-10-23-format-benchmarks/``,
``bench.py``): the same tracks as one bigWig per track read by figwig, by
pybigtools 0.2.5 and by pyBigWig 0.3.26 (those two on a thread pool of
window chunks, one ``values()`` per window, since neither reads on
threads itself), as a zarr 3 array (zstd, chunks of tracks x 65,536,
zarr's own pool set to the thread count), as an HDF5 dataset (h5py, gzip
4, the same chunks) and as a raw numpy memmap, and the figwig file in
both orders. Every configuration ran in its own process under ``nice``,
files on tmpfs, OMP, MKL and OpenBLAS pinned to one thread, while other
jobs kept the machine's load at 55 to 70. Reads are the median of up to
five repeats after a warm-up that compiles the kernels and opens the
files; writes the best of two. In every configuration each reader's
output was checked against the others' (one SHA-256 per read), and they
agreed.

Count tracks, 1 to 1,024, over chr1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tracks are chr1 of the six count tracks the examples use (the two
strands of the 5' ends of ENCODE ATAC-seq BAM ENCFF877LRY, the two
strands of PRO-cap ENCFF994CSC and ENCFF253HUA, and the unstranded cut
counts of ATAC-seq ENCFF830RWF and DNase-seq ENCFF989SAK), uint16, 1.1
percent of bases nonzero; past six, track i is track i mod 6 rolled by a
seeded offset, so it has the statistics of a real track and shares no
position with it. The figwig file is one group of all tracks at a tile of
16,384 and libdeflate level 6; zarr, HDF5 and npy store uint16 as figwig
does. The read is 8,192 random windows of 2,048 bases into float32, or
2,048 and 1,024 windows at 512 and 1,024 tracks, so that the dense output
stays under 8 GB. Seconds at 1 / 8 / 32 threads:

.. list-table::
   :header-rows: 1

   * - format, read
     - 6 tracks
     - 48 tracks
     - 512 tracks
     - 1,024 tracks
   * - figwig file, position order
     - 0.065 / 0.017 / 0.021
     - 0.43 / 0.085 / 0.044
     - 1.16 / 0.22 / 0.11
     - 1.31 / 0.21 / 0.13
   * - figwig file, track order
     - 0.046 / 0.029 / 0.030
     - 0.36 / 0.17 / 0.18
     - 2.77 / 0.66 / 0.56
     - 3.75 / 1.15 / 0.86
   * - bigWigs, figwig
     - 0.24 / 0.048 / 0.036
     - 1.85 / 0.27 / 0.11
     - 13.1 / 1.76 / 0.70
     - 19.1 / 2.62 / 1.28
   * - bigWigs, pybigtools
     - 0.77 / 1.41 / 1.74
     - 5.9 / 11.0 / 13.6
     - 21 / 36 / 42
     - 25 / 39 / 49
   * - bigWigs, pyBigWig
     - 2.23 / 2.56 / 2.75
     - 17.5 / 20.0 / 21.7
     - not run
     - not run
   * - zarr, zstd
     - 1.95 / 2.25 / 2.61
     - 8.1 / 4.0 / 3.7
     - 46 / 15 / 12
     - 55 / 18 / 15
   * - HDF5, gzip 4
     - 3.8 / 9.5 / 9.5
     - 31 / 78 / 75
     - 229 / 230 / 230
     - 232 / 233 / 233
   * - npy memmap, raw
     - 0.26 / 0.15 / 0.20
     - 0.90 / 0.43 / 0.48
     - not run
     - not run

.. list-table::
   :header-rows: 1

   * - format, write, and size
     - 6 tracks
     - 48 tracks
     - 512 tracks
     - 1,024 tracks
   * - figwig file, position order
     - 1.8 / 1.1 / 1.5, 20 MB
     - 14.7 / 8.0 / 7.9, 148 MB
     - 172 / 82 / 82, 1.5 GB
     - 342 / 167 / 169, 3.1 GB
   * - figwig file, track order
     - 3.3 / 2.6 / 3.1, 84 MB
     - 23 / 24 / 25, 676 MB
     - 248 / 255 / 235, 7.2 GB
     - 551 / 575 / 520, 14.4 GB
   * - bigWigs, figwig (with zoom levels)
     - 5.9 / 2.6 / 2.6, 156 MB
     - 47 / 21 / 20, 1.3 GB
     - 504 / 221 / 211, 13.3 GB
     - 1,020 / 454 / 421, 26.6 GB
   * - bigWigs, pyBigWig (1 thread)
     - 18.3, 193 MB
     - 143, 1.5 GB
     - not run
     - not run
   * - zarr, zstd
     - 1.6 / 1.7 / 1.4, 33 MB
     - 9.2 / 4.6 / 5.3, 267 MB
     - 100 / 53 / 52, 2.8 GB
     - 185 / 129 / 140, 5.7 GB
   * - HDF5, gzip 4
     - 6.2 / 6.4 / 6.5, 39 MB
     - 52 / 52 / 53, 309 MB
     - 604 / 603 / 617, 3.3 GB
     - 1,320 / 1,210 / 1,250, 6.6 GB
   * - npy memmap, raw
     - 1.1 / 1.0 / 1.1, 3.0 GB
     - 7.2 / 7.0 / 5.9, 23.9 GB
     - not run
     - not run

.. image:: figures/format-scaling.png
   :alt: Write and read time against threads for 1 to 1,024 count tracks in each format, log scales
   :width: 100%

The figwig file reads 10 times faster than the same tracks as bigWigs on
one thread and 2 to 6 times on 32, and is 8 times smaller than bigWigs
figwig writes; both of figwig's readers gain from threads, where
pyBigWig, pybigtools and HDF5 get slower with more of them (a lock or the
GIL) and zarr stops at about 4 times, its reads being one Python call per
window. Track order is the faster of the two for a few tracks, where its
raw blocks need no inflating, and position order past about 48. The
figwig writer is the fastest of the compressed formats up to 128 tracks
and 1.6 times slower than zarr at 512 and 1,024, where it stops gaining
past 8 threads because sorting and the float64 buffer copies stay on the
calling thread; the track-order writer encodes on the calling thread and
does not gain at all. pyBigWig and pybigtools were not run at 512 tracks
and above, nor the memmap, whose file would be 255 GB.

Real experiments, 1 to 512 ENCODE tracks, over chr1:0-30 Mb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

512 hg38 bigWigs sampled from the ChromBPNet collection on this machine,
mostly float signal with 6 to 8 percent of bases nonzero (three sampled
files held infinite values, which no figwig writer accepts, and the next
candidates stood in), over the first 30 Mb of chr1 so that the dense
forms fit the budget; the figwig file is one group at the defaults, zarr
and HDF5 store float32. The same reads as above. Seconds at 1 / 8 / 32
threads:

.. list-table::
   :header-rows: 1

   * - format, read
     - 8 tracks
     - 128 tracks
     - 512 tracks
   * - figwig file, position order
     - 0.18 / 0.034 / 0.022
     - 3.07 / 0.46 / 0.33
     - 8.7 / 1.15 / 0.56
   * - figwig file, track order
     - 0.066 / 0.032 / 0.030
     - 0.94 / 0.48 / 0.39
     - 2.59 / 1.14 / 0.79
   * - bigWigs, figwig
     - 0.21 / 0.048 / 0.048
     - 3.81 / 0.72 / 0.80
     - 7.1 / 1.50 / 1.62
   * - bigWigs, pybigtools
     - 0.68 / 1.01 / 1.43
     - 10.9 / 14.3 / 19.9
     - 24 / 19.5 / 25
   * - bigWigs, pyBigWig
     - 1.57 / 1.97 / 2.21
     - not run
     - not run
   * - zarr, zstd
     - 3.3 / 3.1 / 3.2
     - 90 / 32 / 25
     - 97 / 24 / 19
   * - HDF5, gzip 4
     - 1.8 / 17.8 / 32
     - 533 / 538 / 539
     - 534 / 536 / 539
   * - npy memmap, raw
     - 0.11 / 0.11 / 0.095
     - not run
     - not run

.. list-table::
   :header-rows: 1

   * - format, write, and size
     - 8 tracks
     - 128 tracks
     - 512 tracks
   * - figwig file, position order
     - 1.9 / 1.4 / 1.4, 44 MB
     - 17.9 / 5.0 / 4.1, 1.0 GB
     - 80 / 16.6 / 17.7, 4.2 GB
   * - figwig file, track order
     - 2.0 / 1.8 / 1.8, 113 MB
     - 15.0 / 12.2 / 11.5, 2.5 GB
     - 70 / 50 / 50, 11.1 GB
   * - bigWigs, figwig (with zoom levels)
     - 11.4 / 9.7 / 10.2, 79 MB
     - 182 / 159 / 168, 1.7 GB
     - 791 / 703 / 707, 7.1 GB
   * - bigWigs, pyBigWig (1 thread)
     - 21.1, 115 MB
     - not run
     - not run
   * - zarr, zstd
     - 1.8 / 1.5 / 1.6, 47 MB
     - 9.9 / 4.8 / 4.5, 1.0 GB
     - 40 / 21 / 30, 4.3 GB
   * - HDF5, gzip 4
     - 4.2 / 3.3 / 3.4, 50 MB
     - 59 / 55 / 55, 1.1 GB
     - 244 / 240 / 240, 4.5 GB
   * - npy memmap, raw
     - 1.6 / 0.33 / 0.37, 960 MB
     - not run
     - not run

Dense float signal is the generic formats' best case and figwig's
hardest: the figwig file is the size of zarr's and HDF5's here, and still
reads 15 to 30 times faster than zarr and 500 times faster than HDF5 at
128 tracks and above. Track order reads these dense tracks faster on one
thread, since its blocks are raw, and the two orders meet at 32 threads.
figwig's bigWig writer, with zoom levels, is 5 to 10 times slower than
the figwig writer on dense signal.

Conversions, genome-wide
~~~~~~~~~~~~~~~~~~~~~~~~

``benchmarks/compare_formats.py`` converted the six count tracks above,
genome-wide this time, and the rolled copies past six, written as bigWigs
by figwig: ``figwig bw2fw`` wrote the figwig file of each set in groups of
8 tracks at a tile of 16,384 bases and level 6, and ``figwig fw2bw`` wrote
it back, one run each, on tmpfs, at a load of 48 to 112:

======  ==========  ===========  =============================  =============================
tracks  bigWigs     figwig file  ``bw2fw``, 1 / 8 / 32 threads  ``fw2bw``, 1 / 8 / 32 threads
======  ==========  ===========  =============================  =============================
1       36.9 MB     25.2 MB      3.0 s / 1.7 s / 2.2 s          5.7 s / 4.2 s / 4.5 s        
2       73.9 MB     46.9 MB      5.0 s / 2.9 s / 3.1 s          11.2 s / 5.4 s / 6.0 s       
6       971.4 MB    224.8 MB     13.3 s / 7.1 s / 7.2 s         38.9 s / 21.8 s / 12.9 s     
24      2,195.1 MB  888.7 MB     55.4 s / 20.3 s / 20.8 s       162.1 s / 82.7 s / 47.3 s    
96      7,089.9 MB  3,555.3 MB   216.8 s / 68.6 s / 71.8 s      604.5 s / 302.2 s / 174.8 s  
======  ==========  ===========  =============================  =============================

The six real tracks are 4.3 times smaller as a figwig file than as
ENCODE's bigWigs, most of the gain on the two ENCODE tracks written as
bedGraph, 883 MB of the 971, and 1.8 times smaller than the same tracks
as bigWigs figwig writes (408 MB); the rolled copies past six are bigWigs
figwig wrote as varStep, and the figwig file is half their size.
``bw2fw`` reads each piece of a chromosome on a thread of its own while
the writer encodes and compresses the piece before it on ``-p`` threads,
and gained nothing past 8; ``fw2bw`` is bound by laying each track's
windows out as bigWig sections, on one thread per track.

``figwig bam2fw`` and ``figwig bam2bw`` converted the 2.4 GB ATAC-seq BAM
ENCFF877LRY, on tmpfs, to one figwig file of two tracks and to two bigWigs:

=================  =======  =======  ========  =======
command            1 core   8 cores  32 cores  output 
=================  =======  =======  ========  =======
``figwig bam2bw``  10.80 s  2.11 s   1.94 s    73.9 MB
``figwig bam2fw``  10.60 s  2.32 s   2.58 s    47.4 MB
=================  =======  =======  ========  =======

Both read the BAM the same way, and ``figwig bam2fw`` writes its counts as
entries on the same cores; it took longer only on 32, where its sort of
each chromosome's entries stays on one.

Compression levels and codecs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The six tracks were written at each libdeflate level, and read as the
8,192 windows above, on one and eight threads:

=====  ========  ===================  ========  =========
level  file      ``bw2fw -p 8``       1 thread  8 threads
=====  ========  ===================  ========  =========
0      412.3 MB  7.6 s                0.062 s   0.047 s  
1      230.1 MB  7.3 s                0.121 s   0.055 s  
3      232.9 MB  7.5 s                0.119 s   0.056 s  
6      227.3 MB  7.3 s                0.123 s   0.048 s  
9      225.2 MB  7.3 s                0.124 s   0.053 s  
12     217.6 MB  9.0 s                0.122 s   0.050 s  
=====  ========  ===================  ========  =========

Every other chunk of the uncompressed file, 205 MB of payload, was then
compressed through each codec's Python binding on one thread, and
decompressed:

==========================  =====  ==========  ============
codec                       ratio  compress    decompress  
==========================  =====  ==========  ============
zlib 6                      0.547  5.6 s       0.88 s      
libdeflate 1                0.554  1.1 s       0.45 s      
libdeflate 6, the default   0.547  2.0 s       0.45 s      
libdeflate 12               0.524  23.7 s      0.44 s      
ISA-L 0                     0.752  0.3 s       0.37 s      
ISA-L 1                     0.558  1.2 s       0.55 s      
ISA-L 3                     0.566  1.3 s       0.54 s      
zstd 3                      0.542  0.8 s       0.36 s      
zstd 19                     0.523  25.0 s      0.35 s      
shuffle by 4, libdeflate 6  0.556  2.3 s       0.47 s      
==========================  =====  ==========  ============

The level makes little difference: 1 to 12 span 230 to 218 MB and read in
the same time, and the uncompressed file is 1.8 times larger and twice as
fast to read on one thread, the same on eight. ISA-L's streams are 2%
larger than libdeflate's and inflate 20% slower, so the reader keeps
libdeflate; zstd's are 1% smaller at level 3 and 4% at 19 and inflate 25%
faster, which, with the inflate about half of a one-thread read and none
of an eight-thread one, would be at most a tenth of a one-thread read for
another dependency. A byte shuffle before deflate grows the chunks, which
are varints and bytes. The bytes are the values: each of the six count
tracks takes 1.1 to 1.7 bytes per nonzero base as sparse chunks of uint8
(the PRO-cap tracks 3.5, on a fifth as many bases), and choosing dense or
sparse by the compressed rather than the raw size would have saved nothing
on 20,000 sampled tiles of each of three of them.

Sums over intervals of any length
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 172,732 IDR peaks of the set, which differ in length, summed over
the six tracks, on tmpfs, median of three at load about 60:

=========================================================  ========  ========
method                                                     seconds   peak RSS
=========================================================  ========  ========
``FigWigReader.sums``, 8 threads                           0.25 s    345 MB
``read(bins=width)`` once per distinct width, 8 threads    2.70 s    233 MB
pyBigWig ``stats(type="sum", exact=True)``, one process    46 s      201 MB
=========================================================  ========  ========

Over 3,000 single cells as tracks (chr1 to chr5 of the PBMC fragments,
46 million cut sites, one group), the 58,705 peaks on those chromosomes
summed in 0.96 s on 8 threads and 1.5 s on one, the same sums as a sparse
read of the windows gives in 1.2 s; before the entries were binned
without a scratch tile this read was refused above 2,048 tracks and took
73 s in two halves.

Dense float tracks and the layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The defaults, a tile of 16,384 and groups of 8, suit count tracks, whose
chunks are mostly zero. Nine chr22 annotation tracks (phyloP, phastCons,
GC content and mappability, 742 MB as bigWigs) were converted at the
defaults and at ``-t 1024 --groups 1 -l 1``, and read as 10,000 random 1
kb windows over every track on one thread, at load about 53:

==================================  ========  ========  =========
layout                              file      1 thread  8 threads
==================================  ========  ========  =========
tile 16,384, groups of 8, level 6   190 MB    9.6 s     2.3 s
tile 1,024, groups of 1, level 1    290 MB    1.33 s    0.39 s
nine bigWigs, ``read_bigwig``       742 MB    1.5 s     0.5 s
==================================  ========  ========  =========

A second run of six chr22 tracks at 500 bp and 1 kb windows gave the
same picture (9.7 s at the defaults against 0.81 s at ``-t 1024 --groups
1`` for 10,000 500 bp windows, and 1.17 s for the bigWigs), and found
``-l 1`` no faster than level 6 for a file 8% larger, so the tile and
groups are the advice. Count tracks go the other way: 8 of 48 count
tracks read in short windows from one group of 48 in a quarter to a half
of the time from 48 groups of one, since a sparse chunk's other tracks are
stepped over by their byte counts.

A 1 kb window from a 16,384-base tile decodes 16 kb of float32 per
track, and every track of its group with it; a dense GC track grouped
with two phyloP tracks made a read of one of those 12 times slower (0.69
s against 0.057 s for 20,000 windows) than with ``--groups 1``, for a
file 3% smaller. The same tracks read in 100 bp bins over the whole of
chr22 took 0.09 s from the small-tile file against 9.8 s for pyBigWig's
``stats``.

Single cells
~~~~~~~~~~~~

The 10,246 called cells of the 10x PBMC ATAC-seq experiment
(``10k_pbmc_ATACv2_nextgem_Chromium_Controller``), 488 million Tn5 cut
sites genome-wide, written with ``write_sparse`` as one uint8 track per
cell in one group (the layout for cells), in track order, as one bigWig
per cell with figwig's writer, and as a scipy CSR matrix of positions x
cells per chromosome saved with ``save_npz`` and held in memory for the
reads. 8,192 windows of 2,048 bases centred on K562 ATAC-seq peaks.
Seconds at 1 / 8 / 32 threads (scipy at 1 / 8):

.. list-table::
   :header-rows: 1

   * - workload
     - figwig, position order (1.0 GB)
     - figwig, track order (2.4 GB)
     - scipy CSR (16.3 GB, in memory)
     - one bigWig per cell (1.4 GB)
   * - write
     - 68 / 37 / 36
     - 71 / 72 / 91
     - 10.6 / 11.4
     - 116 / 78 / 82
   * - 10 pseudobulks over the windows (``labels=``)
     - 1.69 / 0.30 / 0.17
     - 21.7 / 10.0 / refused
     - 5.1 / 4.9
     - 136 / 95 / 96
   * - 128 cells dense over the windows
     - 0.96 / 0.16 / 0.082
     - 0.58 / 0.24 / 0.18
     - 6.6 / 4.9
     - 0.79 / 0.28 / 0.26
   * - every cell's cut sites in 1,024 windows (``sparse=True``)
     - 0.39 / 0.16 / 0.14
     - 7.1 / 5.4 / 5.0
     - 4.3 / 4.7
     - not run

Pseudobulks in the read are 30 times faster than from the sparse matrix
and 500 times faster than from one bigWig per cell, and the file is a
sixteenth of the matrix's size; a tile of cut sites whose nonzero values
are all 1 stores no values. Track order sums 10,246 tracks one block task
at a time, and at 16 and 32 threads its per-thread output copies pass the
8 GiB guard, so it is refused there. The scipy writer only sorts and
saves, and its 16 GB are loaded before any read.

Reads as tracks
~~~~~~~~~~~~~~~

The m6A calls (probability at least 0.5) of the 90,543 reads of a K562
Fiber-seq BAM, 49.6 million calls, one uint8 track per read with the
read's name, span and strand as attributes, in position order in groups of
1,024 consecutive reads (62 MB), in track order (226 MB), and as a scipy
CSR matrix per chromosome (12.7 GB). Windows of 2,048 bases on reads.
Seconds at 1 / 8 / 32 threads (scipy at 1 / 8):

.. list-table::
   :header-rows: 1

   * - workload
     - figwig, position order
     - figwig, track order
     - scipy CSR
   * - write
     - 4.0 / 2.6 / 2.7
     - 6.5 / 9.1 / 10.6
     - 6.8
   * - sum over all reads, 8,192 windows (``labels=``)
     - 0.42 / 0.43 / 0.42
     - 15.7 / 15.8 / 16.0
     - 3.8 / 4.0
   * - calls of every read in 1,024 windows (``sparse=True``)
     - 0.17 / 0.16 / 0.17
     - 7.1 / 7.2 / 7.2
     - 27 / 3.5
   * - each read over its own span (``sums(paired=True)``)
     - 2.7 / 2.7 / 2.7
     - 9.6 / 9.7 / 9.6
     - 5.8 / 6.8

One group of every read is refused, since a chunk holds at most 2**29
values, and a group per read made the sum over 8,192 windows take 230 s,
each read paying a planning step per group; groups of consecutive reads
are the layout. These reads do not gain from threads, since each touches
few chunks and the work is the planning.

To a GPU
~~~~~~~~

``gpu_bench.py`` timed six paths from a file to a CUDA tensor on an
H200 NVL (PCIe), with ``torch.cuda.synchronize()`` before the clock
stopped: the bigWigs read by pybigtools, by pyBigWig and by figwig, each
followed by ``torch.from_numpy`` and ``.cuda()``; the figwig file read
into a numpy array then ``.cuda()``; into a pinned torch buffer through
``out=`` then ``.to("cuda", non_blocking=True)``; and with
``device="cuda"`` then ``torch.from_dlpack``. A DataLoader batch of 64
windows and the full reads above; the single-cell and Fiber-seq files on
the figwig paths. Seconds at 1 / 8 / 32 threads (pybigtools and pyBigWig
at 1 / 8, where they only get slower):

.. list-table::
   :header-rows: 1

   * - path
     - 6 tracks, batch of 64
     - 48 tracks, batch of 64
     - 48 tracks, 8,192 windows
     - 512 tracks, 2,048 windows
     - 128 of 10,246 cells, 8,192 windows
     - 10 pseudobulks of 10,246 cells
   * - bigWigs, pybigtools, ``.cuda()``
     - 0.015 / 0.021
     - 0.12 / 0.16
     - 15.1 / 11.3
     - 43 / 39
     - not run
     - not run
   * - bigWigs, pyBigWig, ``.cuda()``
     - 0.023 / 0.027
     - 0.18 / 0.21
     - 17.7 / 21.1
     - not run
     - not run
     - not run
   * - bigWigs, figwig, ``.cuda()``
     - 0.016 / 0.0093 / 0.0099
     - 0.13 / 0.030 / 0.035
     - 2.0 / 0.37 / 0.21
     - 13.1 / 2.1 / 1.14
     - not run
     - not run
   * - figwig file, numpy, ``.cuda()``
     - 0.00079 / 0.00079 / 0.00078
     - 0.0048 / 0.0047 / 0.0049
     - 0.65 / 0.21 / 0.14
     - 1.67 / 1.56 / 0.36
     - 1.21 / 0.41 / 0.33
     - 1.70 / 0.31 / 0.20
   * - figwig file, pinned ``out=``, ``non_blocking``
     - 0.00075 / 0.00074 / 0.00074
     - 0.0043 / 0.0043 / 0.0043
     - 0.51 / 0.13 / 0.10
     - 1.47 / 0.35 / 0.26
     - 0.88 / 0.30 / 0.25
     - 1.67 / 0.30 / 0.19
   * - figwig file, ``device="cuda"``, ``from_dlpack``
     - 0.00081 / 0.00078 / 0.00077
     - 0.0045 / 0.0043 / 0.0043
     - 0.50 / 0.14 / 0.10
     - 1.34 / 0.36 / 0.30
     - 0.68 / 0.099 / 0.048
     - 1.69 / 0.30 / 0.19

.. image:: figures/gpu-loading.png
   :alt: Time to a CUDA tensor against threads for each path and dataset, log scales
   :width: 100%

For a DataLoader batch the figwig file reaches the GPU in a millisecond
for 6 tracks and 4 ms for 48, 15 to 40 times ahead of the bigWig readers,
and the three figwig paths are the same there since a batch's transfer is
small. For full reads the transfer is the large part: 3.2 GB of 48 tracks
reaches the GPU in 0.10 s through the pinned buffer against 0.14 s
through a pageable numpy array. ``device="cuda"`` matches the pinned path
where it copies, and wins by 5 times where it scatters: 128 of 10,246
cells over 8,192 windows arrive in 0.048 s against 0.25 s, since the bus
carries 0.002 percent of the bases as entries instead of 8.6 GB of zeros.

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
