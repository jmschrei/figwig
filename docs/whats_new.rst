.. currentmodule:: figwig


===============
Release History
===============


Version 0.1.0 (unreleased)
==========================

Highlights
----------

	- Initial release. ``BigWigReader(path).read(chroms, starts, width, out=None, n_jobs=8, missing=0.0)`` reads the per-base values of many windows of one width into a float32 numpy array of shape ``(n, width)``, on up to ``n_jobs`` threads, or one per CPU with ``n_jobs=-1``. ``read_bigwig`` does the same in one call, and given a list of bigWigs reads them all into one array of shape ``(n, len(bigwigs), width)``, with every file's batches on one pool of threads.

	- The values are those pybigtools' ``values(chrom, start, end, missing=missing)`` gives: ``missing``, 0 unless given, where no interval covers a base or an interval's value is NaN, and NaN past the end of a chromosome. A window on a chromosome that the file does not have, which writers leave out when it has no data, is ``missing`` throughout, and a warning names the chromosomes.

	- bedGraph, varStep and fixedStep sections are read, compressed or not, including pyBigWig's fixedStep blocks whose index entries overlap by a few bases. A file or data block that figwig does not read, such as a bigBed, a big-endian bigWig, a corrupt or truncated index, or a block with overlapping intervals, raises a ``ValueError`` that says what it found.

	- A ``BigWigReader`` can be read from several threads at once and pickled, so it can be part of a PyTorch Dataset read by DataLoader workers under any start method.

	- ``write_bigwig(paths, chrom_sizes, chroms, starts, values, ends=None, missing=0.0)`` writes what ``read_bigwig`` reads: windows on any chromosomes, in any order, with values of shape ``(n, width)`` for one file, or ``(n, len(paths), width)`` for several, channel ``i`` to the ``i``-th file. With ``ends``, it writes intervals and one value for each. Bases and intervals whose value is ``missing`` or NaN are left out, so that reading with the same ``missing`` gives the values back, and bases past the end of a chromosome, which ``read_bigwig`` gives as NaN, are accepted there. ``BigWigWriter(path, chrom_sizes, zooms=10, level=6, engine='auto', n_jobs=8)`` writes the same values a batch at a time with ``write(chroms, starts, values, ends=None, missing=0.0)``, each call after the last. Values are converted and laid out about a million at a time, however many one call holds. Data blocks are compressed on up to ``n_jobs`` threads by zlib's ``compress2()``, or by libdeflate when the ``deflate`` package is installed (the new ``fast`` extra), both called without the GIL, except on Windows, where neither library can be loaded and Python's zlib module compresses the blocks; ``engine='isal'`` uses ISA-L. Zoom levels are built when the writer is closed.

	- Each window, or each 65,536 bases of a wider one, is written as whichever section type takes it in the fewest bytes: fixedStep for runs of bases, varStep for scattered bases, or bedGraph for stretches of one value. Intervals are written as bedGraph, as they are given.

	- Written files are laid out as libBigWig, pyBigWig's writer, lays them out, and a file of intervals, or of single bases as windows of width 1, written with ``missing=numpy.nan``, without zoom levels and with zlib at level 6, is pyBigWig's byte for byte. Where libBigWig writes a wrong value, figwig writes the correct one: the header's maximum when the first value is the largest or no value is positive, the end of the last fixedStep block of each call, which libBigWig puts 6 bases past it, and the sum and sum of squares of the last zoom record of each zoom block, which libBigWig leaves at 0. libBigWig's empty blocks are not written, and zoom levels go on past a level that is no smaller than the one before it, where libBigWig stops.

	- ``figwig bam2bw``, a command, is bam2bw 0.5.1 with the same arguments, outputs and messages, reading files as the winner of a speed search over bam2bw's code reads them, a BAM file with libdeflate and a numba kernel on the ``-p`` threads and a BED/tsv file with a numba kernel, and writing the counts with ``BigWigWriter``, with libdeflate at level 1. Its bigWigs hold bam2bw's entries, but not its bytes, and ``-z`` writes figwig's zoom levels. ``-p`` is a number of cores: files are read one after another, each on every core, and files read on one thread (SAM, ``--mate_pairs``, remote files, gzipped BED/tsv files that are not BGZF) by a pool of processes alongside them. It needs the new ``bam2bw`` extra. A BAM file's keys, one per counted read end, are counted a group of chromosomes at a time while the file is read and then dropped, so that for a coordinate-sorted BAM the array that holds them is the size of its largest chromosome's rather than the whole file's.

	- ``figwig install-skill`` installs figwig's Claude Code skill, which teaches a coding agent to read and write bigWigs with figwig and to convert reads with ``figwig bam2bw``: the signatures and shapes, the rules a write follows, the error messages and their causes, and the measured gain from threads. It copies the skill into ``~/.claude/skills/figwig``, or ``-d DIRECTORY/figwig``; ``--symlink`` links to the package's copy instead, and ``--force`` replaces an installed copy, which an upgrade of figwig needs to pick up a changed skill.

	- Tested on Linux with Python 3.10 to 3.14, and on macOS and Windows with Python 3.10 and 3.13. On Windows, which has no ``os.pread`` and where neither zlib's library nor libdeflate's functions can be loaded, blocks are read by seeking the file, and decompressed and compressed by Python's zlib module, with the same values.

	- The reader began inside tangermeme's ``extract_loci`` (tangermeme #107).
