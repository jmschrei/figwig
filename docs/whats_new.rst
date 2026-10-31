.. currentmodule:: figwig


===============
Release History
===============


Unreleased
==========

figwig files
------------

	- ``FigWigWriter`` and ``write_figwig`` accept ``summary_index=True`` (default ``False``) for responsive interactive browser use without whole-level summary downloads and decompression. Existing v1 readers remain compatible. See :doc:`summary_restarts` for usage and limits.

	- ``FigWigReader`` uses the index automatically for selected-chromosome ``tile_sums`` and exact integer ``sums``, reading only requested ranges. Precision, missing-value semantics and ordinary-file fallback are unchanged.

	- ``sums`` over a track stored as runs, such as the coverage ``figwig bam2fw -c`` writes, only decodes the changes of value before each interval, adds a run of integers once rather than base by base, and takes the tiles an interval covers whole from the file's tile sums when the track holds integers whose sums those keep exactly. On one thread of the benchmarks' server, the coverage of ENCODE ATAC-seq BAM ENCFF877LRY sums over 78,733 gene bodies in 0.47 s, where it took 7.9 s, and over 100,548 peaks in 0.41 s, where it took 1.6 s; a fold-change track over the gene bodies in 1.55 s, where it took 6.8 s; 48 count tracks over the gene bodies in 6.8 s, where they took 14.4 s. The sums are the same to the bit.

	- ``write_sparse``, and so ``figwig reorder``, stores a tile as a dense or runs chunk when either takes fewer bytes than a sparse one, as ``write`` does. Reordering the ATAC-seq coverage file gives the same 79 MB of runs chunks, where it gave 119 MB of sparse chunks. Scattered entries, such as cells' cut sites, are stored as before: the 10,246-cell PBMC file is the same to the byte.

Converters and commands
-----------------------

	- ``figwig bam2fw -c`` on threads writes each 4 Mb of a chromosome's coverage as one dense window, whose tiles the writer encodes on its threads, where it encoded every run on the calling thread, unless the 4 Mb hold fewer runs than one per 1,024 bases, as sparse coverage such as RNA-seq's does, which goes in as runs as before: the 2.4 GB ATAC-seq BAM takes 7.2 s on 8 threads, where it took 14.2 s. The file is the same to the byte, and one thread writes it as before.

	- ``figwig bam2bw -c`` and ``figwig bam2fw -c`` read a coordinate-sorted BAM file with a ``.bai`` no older than it a region at a time on every core, from the index's offset for each region's first base, and write each chromosome while the next is read, rather than keeping two events per aligned block of every read until the file ends. They do so at ``-p 8`` or more, for ``bam2bw`` from 128 MiB and 5 million mapped reads in its files, for ``bam2fw`` from 128 MiB and 50,000 mapped reads in each file; below these, set from timings on 25 BAMs of the test-data collection with a margin on the whole-file reader's side, and for a file the region reader cannot read exactly as the whole-file reader would, the whole-file reader runs as before. The 2.4 GB ATAC-seq BAM takes 1.6 s with ``bam2bw -c -u -p 8``, where it took 2.5 s, and 1.05 s at ``-p 32``, where it took 2.5 s; 4.9 s with ``bam2fw -c -u -p 8``, where it took 7.2 s, and 2.3 s at ``-p 32``, where it took 7.1 s. At ``-p 8`` the peak memory falls from 2.0 to 0.54 GB (``bam2bw``) and from 1.9 to 0.63 GB (``bam2fw``); at ``-p 32`` it is the regions in flight, 2.4 GB against 2.0 GB for ``bam2bw``. The bigWigs and unstranded figwig files are the same to the byte; a stranded figwig file holds the same values and the same bytes of chunks, the two strands' interleaved, so its index holds more runs: 0.14% more bytes in all on 8 million ChIP-seq reads.

Documentation
-------------

	- The benchmarks page's count-track tables have a row for zarrs 0.2.3, the Rust codec pipeline for zarr-python, on the same arrays as the zarr row: it writes them 1.1 to 3.5 times faster than zarr-python and reads one window per call at zarr-python's one-thread speed, without gaining from the Python threads, with its chunk concurrency pinned to one chunk (its default read twice as slowly on these calls).


Version 0.2.0
=============

figwig 0.2.0 adds the figwig file format: any number of tracks over one genome in one file, read in windows from any subset of them, with converters from bigWigs and BAMs and a Claude Code skill that covers it; and faster bigWig reading and writing. The bigWig API only gains keywords, and files written by 0.1.0 read as before.

figwig files
------------

	- A new file format, the figwig file (``.fw``), holds the per-base values of any number of tracks over one genome in one file, with a JSON record of attributes per track and the name of the genome, and is read many windows at a time from any subset of its tracks into one ``(windows, tracks, width)`` array in the tracks' own dtype. ``FigWigReader``, ``read_figwig``, ``FigWigWriter`` and ``write_figwig`` mirror the bigWig API with a ``tracks=`` argument; ``select(**attributes)`` gives the tracks whose records match, ``stats`` each track's sum, nonzero bases, minimum and maximum over its nonzero bases. Version 1 of the format is specified and frozen on the format page, with a reference reader written from the page alone and a Kaitai Struct definition in the repository; a reader refuses a file of a later version, or one with a required feature it does not know.

	- Layout. Every track lies on a grid of ``tile`` bases, and the tiles of a group of tracks are one chunk, stored as nothing, dense, sparse, runs or a per-tile matrix, whichever is smallest, in the narrowest element type that holds the values exactly, compressed with zlib or libdeflate on ``n_jobs`` threads (``level=0`` keeps chunks raw). The index holds only the tiles with values, so a file of 100,000 single-track groups opens in a tenth of a second. ``order='tracks'`` stores each track's nonzero bases in blocks per chromosome instead, for reading a few tracks over the genome, and ``figwig reorder`` rewrites a file in the other order.

	- Reads. ``read(..., sparse=True)`` returns the nonzero bases as entries, ``labels=`` sums the tracks per label, which turns cells into pseudobulks, ``bins=`` sums each window per bin, ``read_entries`` returns every nonzero base of tracks over chromosomes, and ``channels=``/``items=`` arrange the tracks read by two attributes. ``sums(chroms, starts, ends, nan=, paired=)`` sums intervals of any lengths, with NaN bases propagated, skipped or counted. ``tile_sums`` gives each track's sum per tile from the file's summaries, and ``tile_sums(level=k)`` the zoom levels a file written with ``zooms=`` keeps, without decoding a chunk. A dense read whose output would pass 8 GiB raises instead of allocating it.

	- Writes. ``write`` takes windows or intervals of whole groups and ``write_sparse`` nonzero bases as entries, both encoded and compressed on the ``n_jobs`` threads. A track's dtype is declared or inferred; a float32 track keeps float32's rounding, and a value beyond float32, a 64-bit integer beyond 2**53 or an infinite value is refused before anything is written. The file is written beside its path and moved there at ``close``, so a failed write leaves the path as it was. ``append=True`` adds tracks to a file in place under its lock, and ``add_file`` and ``figwig merge`` copy the tracks of files written with the same chromosomes, tile, level and order.

	- Integrity. Every chunk carries a CRC-32 that ``verify=True`` checks, the header, tables and index records carry CRC-32s checked at open, and every count and pointer is bounded against the file before its bytes are read, so a damaged, crafted or truncated file raises ``ValueError`` rather than crashing. A reader checks on every read that the file at its path is the one it opened.

	- ``device="cuda"`` on ``read``, ``sums`` and ``read_figwig`` lands the output in GPU memory as a ``cupy.ndarray``, which PyTorch, JAX and TensorFlow take without a copy through DLPack; the new ``cuda`` extra adds cupy.

	- Dense float tracks, such as conservation or fold change, read 3 to 6 times faster and write up to twice as fast on many threads: three chr22 conservation tracks, 8,192 windows of 2,048 bp, read in 0.50, 0.07 and 0.03 s on 1, 8 and 32 threads of the benchmarks' server, against 0.80, 0.12 and 0.11 s from their bigWigs.

	- A first call compiles a quarter fewer kernel specializations, since the kernels pass constants to each other as plain values rather than literals that numba compiled a copy per value for: about 85 s for a process that reads and writes every way, where it was 116 s.

Converters and commands
-----------------------

	- ``figwig bw2fw`` writes bigWigs as the tracks of one figwig file, named and given attributes by a metadata table, with ``--missing nan`` for tracks whose zeros are measurements, ``--append`` and ``-z`` zoom levels; ``figwig fw2bw`` writes a file's tracks back as bigWigs with such a table; ``figwig bam2fw`` takes what ``figwig bam2bw`` takes and writes each input file as its own stranded pair of tracks, or with ``--barcode-column N`` a fragments file as one track per cell, ``--cells`` listing the cells to keep with their attributes: the 257.6 million fragments of the 10x PBMC experiment become its 10,246 called cells in 9 s on 8 threads of the benchmarks' server. ``figwig merge`` and ``figwig info`` complete the set. Every command refuses to write over an existing output without ``-f``, writes beside its path and moves the output into place when complete, and prints a message rather than a traceback for an input that is missing, not a BAM or cut short.

	- ``-c/--coverage`` on ``figwig bam2bw`` and ``figwig bam2fw`` counts every aligned base of each read (the CIGAR's M, = and X runs, as pysam's ``get_blocks()`` and deepTools' ``bamCoverage`` count them), or every base of each BED/tsv interval, in place of one end, with ``-ps``/``-ns`` moving a read's first base and its end and ``-r`` dividing by the reads; ``bam2fw -c`` writes the coverage as runs, each track its own group. On the 2.4 GB ATAC-seq BAM and ten more experiments the bigWig holds the same value at every base as ``bamCoverage --binSize 1``, 3 to 8 times faster; the 2.4 GB BAM takes 3.4 s on 8 threads of the benchmarks' server against deepTools' 14.9 s.

	- ``figwig bam2fw`` and ``figwig bam2bw`` check each BAM's header against the sizes before reading: a shared chromosome of another length, or no shared chromosome, is refused by ``bam2fw`` and reported on stderr by ``bam2bw``, which goes on as bam2bw 0.5.1 does; an hg19 BAM against hg38 sizes converted silently before. ``figwig bw2fw -s`` refuses a sizes file that shares no chromosome with a bigWig, where it wrote a file of zeros. An empty track survives the round trip through ``fw2bw`` and ``bw2fw``: ``BigWigReader`` opens a bigWig without a data index, which it refused as unreadable.

Reading
-------

	- The bigWig reader takes a file whose index leaves lie in natural chromosome order (chr1, chr2, ..., chr10), which it refused as unsorted where pyBigWig read it. ``keep_nan=`` on ``BigWigReader.read`` and ``read_bigwig`` reads an interval whose value is NaN as NaN rather than ``missing``.

	- ``engine='auto'``, ``'zlib'`` or ``'isal'`` on ``read`` and ``read_bigwig``: with python-isal 1.8 built on ISA-L 2.31.1 installed (the new ``isal`` extra, which the ``bam2bw`` extra includes), data blocks inflate through ISA-L from the numba kernel without the GIL, 1.3 to 1.5 times faster on the benchmarks' bigWigs on 1 to 8 threads; every engine gives the same values. Thanks @YaoxiangLi!

Writing
-------

	- The kernel that summarizes items into a zoom level holds its open record in locals, so levels summarized on different threads no longer share a cache line, which made the kernel many times slower in some processes. The files written are unchanged. Thanks @YaoxiangLi!

	- With threads, ``BigWigWriter`` plans, encodes and summarizes each batch on threads of its own, so the calling thread only takes the next call's values, and sorted, valid input is confirmed in one compiled pass. The files written are unchanged byte for byte; writing the benchmarks' tracks with libdeflate and no zoom levels on 8 threads takes 1.4 to 1.5 times less. Thanks @YaoxiangLi!

Claude Code skill
-----------------

	- The skill covers figwig files and the converters, routes by task and by symptom, lists every error message in ``references/errors.md``, and ``references/recipes.md`` walks through whole jobs: a training set from many bigWigs or BAMs, genome-wide predictions, a single-cell file, summaries over peaks, a lossless signal round trip and reads as tracks. Re-run ``figwig install-skill --force`` to pick it up.

Tests
-----

	- The test suite runs in a quarter of the time, on several cores with ``uv run pytest -n 8`` (pytest-xdist is in the dev extra), and CI keeps numba's compiled kernels between runs.


Version 0.1.0
=============

Reading
-------

	- Initial release. ``BigWigReader(path).read(chroms, starts, width, out=None, n_jobs=8, missing=0.0)`` reads the per-base values of many windows of one width into a float32 numpy array of shape ``(n, width)``, on up to ``n_jobs`` threads, or one per CPU with ``n_jobs=-1``. ``read_bigwig`` does the same in one call, and given a list of bigWigs reads them all into one array of shape ``(n, len(bigwigs), width)``, with every file's batches on one pool of threads.

	- The values are those pybigtools' ``values(chrom, start, end, missing=missing)`` gives: ``missing``, 0 unless given, where no interval covers a base or an interval's value is NaN, and NaN past the end of a chromosome. A window on a chromosome that the file does not have, which writers leave out when it has no data, is ``missing`` throughout, and a warning names the chromosomes.

	- bedGraph, varStep and fixedStep sections are read, compressed or not, including pyBigWig's fixedStep blocks whose index entries overlap by a few bases. A file or data block that figwig does not read, such as a bigBed, a big-endian bigWig, a corrupt or truncated index, or a block with overlapping intervals, raises a ``ValueError`` that says what it found.

	- A ``BigWigReader`` can be read from several threads at once and pickled, so it can be part of a PyTorch Dataset read by DataLoader workers under any start method.

	- The reader began inside tangermeme's ``extract_loci`` (tangermeme #107).

Writing
-------

	- ``write_bigwig(paths, chrom_sizes, chroms, starts, values, ends=None, missing=0.0)`` writes what ``read_bigwig`` reads: windows on any chromosomes, in any order, with values of shape ``(n, width)`` for one file, or ``(n, len(paths), width)`` for several, channel ``i`` to the ``i``-th file. With ``ends``, it writes intervals and one value for each. Bases and intervals whose value is ``missing`` or NaN are left out, so that reading with the same ``missing`` gives the values back, and bases past the end of a chromosome, which ``read_bigwig`` gives as NaN, are accepted there. ``BigWigWriter(path, chrom_sizes, zooms=10, level=6, engine='auto', n_jobs=8)`` writes the same values a batch at a time with ``write(chroms, starts, values, ends=None, missing=0.0)``, each call after the last. Values are converted and laid out about a million at a time, however many one call holds. Data blocks are compressed on up to ``n_jobs`` threads by zlib's ``compress2()``, or by libdeflate when the ``deflate`` package is installed (the new ``fast`` extra), both called without the GIL, except on Windows, where neither library can be loaded and Python's zlib module compresses the blocks; ``engine='isal'`` uses ISA-L. Zoom levels are built when the writer is closed.

	- Each window, or each 65,536 bases of a wider one, is written as whichever section type takes it in the fewest bytes: fixedStep for runs of bases, varStep for scattered bases, or bedGraph for stretches of one value. Intervals are written as bedGraph, as they are given.

	- Written files are laid out as libBigWig, pyBigWig's writer, lays them out, and a file of intervals, or of single bases as windows of width 1, written with ``missing=numpy.nan``, without zoom levels and with zlib at level 6, is pyBigWig's byte for byte. Where libBigWig writes a wrong value, figwig writes the correct one: the header's maximum when the first value is the largest or no value is positive, the end of the last fixedStep block of each call, which libBigWig puts 6 bases past it, and the sum and sum of squares of the last zoom record of each zoom block, which libBigWig leaves at 0. libBigWig's empty blocks are not written, and zoom levels go on past a level that is no smaller than the one before it, where libBigWig stops.

figwig bam2bw
-------------

	- ``figwig bam2bw``, a command, is bam2bw 0.5.1 with the same arguments, outputs and messages, reading files as the winner of a speed search over bam2bw's code reads them, a BAM file with libdeflate and a numba kernel on the ``-p`` threads and a BED/tsv file with a numba kernel, and writing the counts with ``BigWigWriter``, with libdeflate at level 1. Its bigWigs hold bam2bw's entries, but not its bytes, and ``-z`` writes figwig's zoom levels. ``-p`` is a number of cores: files are read one after another, each on every core, and files read on one thread (SAM, ``--mate_pairs``, remote files, gzipped BED/tsv files that are not BGZF) by a pool of processes alongside them. It needs the new ``bam2bw`` extra. A BAM file's keys, one per counted read end, are counted a group of chromosomes at a time while the file is read and then dropped, so that for a coordinate-sorted BAM the array that holds them is the size of its largest chromosome's rather than the whole file's.

Claude Code skill
-----------------

	- ``figwig install-skill`` installs figwig's Claude Code skill, which teaches a coding agent to read and write bigWigs with figwig and to convert reads with ``figwig bam2bw``: the signatures and shapes, the rules a write follows, the error messages and their causes, and the measured gain from threads. It copies the skill into ``~/.claude/skills/figwig``, or ``-d DIRECTORY/figwig``; ``--symlink`` links to the package's copy instead, and ``--force`` replaces an installed copy, which an upgrade of figwig needs to pick up a changed skill.

Platforms
---------

	- Tested on Linux with Python 3.10 to 3.14, and on macOS and Windows with Python 3.10 and 3.13. On Windows, which has no ``os.pread`` and where neither zlib's library nor libdeflate's functions can be loaded, blocks are read by seeking the file, and decompressed and compressed by Python's zlib module, with the same values. ``figwig bam2bw`` runs on Linux and macOS, since pysam does not support Windows.
