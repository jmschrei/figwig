.. currentmodule:: figwig


===============
Release History
===============


Unreleased
==========

figwig files
------------

	- A new file format, the figwig file (``.fw``), holds the per-base values of any number of tracks over one genome in one file, with a JSON record of attributes for each track and the name or URL of the reference genome, and is read many windows at a time from any subset of its tracks into one array of shape ``(windows, tracks, width)``, in the type the tracks were written as. ``FigWigReader``, ``read_figwig``, ``FigWigWriter`` and ``write_figwig`` mirror the bigWig API, with ``tracks=`` on ``read(chroms, starts, width)`` and ``write(chroms, starts, values, ends=None)``; ``FigWigReader.select(**attributes)`` gives the indices of the tracks whose records match, ``stats`` each track's sum, nonzero bases (NaN bases are not counted), minimum and maximum, and ``written_by`` the figwig that wrote the file. This is version 1 of the format: the header carries ``required`` and ``optional`` feature bits, so that a later addition can be marked for the readers that must know it, and a reader refuses a file of a later version, or with a required bit it does not know, saying a newer figwig wrote it.

	- Layout. Every track lies on one grid of ``tile`` bases, so a window's tiles are found by division. The tiles of a group of tracks are one chunk, stored as nothing when every value is zero, and otherwise densely, as nonzero bases (a *sparse* chunk, whose offsets are varints with the bytes each track's take, so that a read of some of a group's tracks steps over the others'), as the bases where a track's value changes and the value from each on (a *runs* chunk, a few dozen bytes per tile for tracks of bins or fold change), or as a per-tile matrix of the tracks present, whose size grows with the values rather than the tracks, in the narrowest element type that holds the values exactly, including a ``one`` type that stores no values when every nonzero value is 1, as a tile of single-cell cut sites is, and a float64 type that keeps an integer above 2**24 exact beside a fraction. Chunks are compressed with zlib or libdeflate on ``n_jobs`` threads, or stored as they are when deflate would grow them, and ``level=0`` writes an uncompressed file. The index holds only the tiles with values, deflated, so a group costs bytes for its data rather than per genome, the reader keeps 17 bytes per tile, a pickled reader carries no index, and a file of 100,000 single-track groups opens in a tenth of a second.

	- Two orders. ``order='positions'`` (``'csc'``), the tile layout above, reads windows across many tracks; ``order='tracks'`` (``'csr'``) stores each track's nonzero bases in blocks per chromosome, for reading a few tracks over the genome; ``figwig reorder`` rewrites a file in the other order, keeping its tile, groups and level, and may write over its input.

	- Reads. ``read(..., sparse=True)`` returns the nonzero bases as a ``SparseWindows`` of entries, ``labels=`` sums the tracks per label in the read, which turns cells into pseudobulks, ``bins=`` sums each window per bin of bases without laying the bases out, ``read_entries(tracks, chroms)`` returns every nonzero base of tracks over chromosomes, and ``channels="channel", items="read"`` arranges the tracks read by two attributes into ``(windows, items, channels, width)``, for reads, cells or channels as tracks, with no change to the file. The default dtype of a read of mixed tracks is the narrowest type that holds every track exactly; sums of integer tracks come back float64, and sums into an integer dtype are taken in int64 and checked; a dense read whose output would pass 8 GiB raises instead of allocating it. A read uses one thread per 64 chunks it touches up to ``n_jobs``, keeps its pool between reads, reads each batch's bytes without the GIL, splits the pairs of a few chunks over the threads, and decodes a chunk that many windows fall on once into a scratch tile.

	- Sums. ``FigWigReader.sums(chroms, starts, ends, tracks=None, dtype=None, n_jobs=8, nan='propagate', paired=False)`` sums intervals of any lengths, such as peaks or genes; ``nan='skip'`` leaves NaN bases out and ``nan='count'`` gives the bases that are not NaN, so a mean over the bases a track covers, as pyBigWig and deepTools give it, is the sums with ``'skip'`` over those with ``'count'`` on a file whose uncovered bases are NaN; ``paired=True`` sums interval i over track i alone, for a reads dimension where each read's span is its own track. ``FigWigReader.tile_sums(tracks, chroms, level=0)`` gives each track's sum over every tile from the file's summaries, kept while they take at most 256 MB, without decoding a chunk: a genome-wide profile at the tile's resolution, which is what a bigWig's zoom levels serve. A file written with ``zooms=k`` (``bw2fw -z``, ``bam2fw -z``) keeps ``k`` zoom levels, each 4 times coarser than the last, in float64, which ``tile_sums(level=j)`` returns, ``FigWigReader.zooms`` lists and ``info`` prints, announced by an optional header bit; ``reorder``, ``merge`` and ``append=True`` keep them.

	- Writes. ``write`` takes windows or intervals of whole groups, and ``write_sparse(chroms, positions, tracks, values)`` takes nonzero bases as entries, the form of single-cell cut sites and of a sparse matrix's triplets, sorted once by one composite key; both encode and compress on the ``n_jobs`` threads, each task converting its own tiles, so that 512 count tracks over chr1 write in 38 s on 8 threads and 1,024 in 75 s. A track's ``dtype`` may be declared (``uint8`` to ``uint32``, ``int8`` to ``int32``, ``float32``, ``float64``) or inferred; a float32 track keeps float32's rounding, a value beyond float32 raises unless the track is float64, a 64-bit integer beyond 2**53 is refused rather than rounded, an infinite value is refused before anything is written (a base is a finite value or NaN, in figwig files as in the bigWigs figwig writes), -0.0 is 0, and the sum statistic is compensated, so it does not depend on how the values were split between calls. The file is written beside its path and moved there at ``close``, so a failed write, a writer never closed, or a conversion over its own input leaves the path as it was; a writer garbage collected with values written and no ``close`` warns.

	- Adding tracks. ``FigWigWriter(path, None, tracks, append=True)`` adds tracks to the file at the path in place, writing the new chunks, tables, index and last the header after its end, under the file's lock, so that the cost is the new chunks, a writer that fails or is killed leaves the file readable as it was, and one writer at a time appends. ``FigWigWriter.add_file(path)`` adds every track of another file written with the same chromosomes, tile, level and order, copying its chunks; ``figwig merge -o all.fw a.fw b.fw`` does that from the command line, refusing files that name different genomes unless ``-g`` says which; and ``figwig bw2fw --append`` adds bigWigs to a figwig file.

	- Integrity. Every chunk's CRC-32 is kept and ``FigWigReader(path, verify=True)`` checks each chunk read against it; a chunk stored as it is inside a compressed file is checked on every read; the header keeps a CRC-32 of itself and one of the tables and directory, and each index record one of its own, all checked at open, so a flipped byte anywhere outside the chunks is refused rather than read by. Every count in the header and the index is bounded against the file before its bytes are asked for, runs may not overlap, the tile count is capped, and every pointer of a chunk is checked before any offset is read by it, so a crafted or truncated file raises ``ValueError`` rather than a ``MemoryError`` or a crash. The reader holds the file by its inode and header and checks on every read that the file is the one it opened, so a file replaced by a conversion raises instead of reading with a stale index, while a reader keeps reading through an append in place. A directory, a FIFO, a deeply nested metadata document or a non-object ``attributes`` are refused as ``ValueError``; any other top-level metadata key is kept through an append.

	- Reading onto a GPU. ``FigWigReader.read``, ``sums`` and ``read_figwig`` take ``device="cuda"`` (or ``"cuda:N"``, or an index) and land their output in that GPU's memory as a ``cupy.ndarray``, which PyTorch, JAX, TensorFlow and CuPy take without a copy through DLPack (``torch.from_dlpack(y)``); the new ``cuda`` extra adds cupy. Sparse tracks are listed as entries, moved and scattered on the device, so the bus carries the entries rather than the dense windows; dense tracks are decoded into a pinned host buffer kept on the reader and copied; the device's stream is synchronized before the array is returned. Without ``device``, ``out=`` may be a numpy view of a pinned torch tensor, the shorter path than a DataLoader's ``pin_memory``.

Converters and commands
-----------------------

	- ``figwig bw2fw`` writes bigWigs as the tracks of one figwig file, named after their files or by a metadata table whose other columns become the tracks' attributes, reading each piece of a chromosome on a thread of its own while the piece before it is written. It reads with ``keep_nan=True``, so the NaN gaps of a Hi-C insulation or compartment track are NaN in the figwig file rather than 0, and ``--missing nan`` writes the bases no interval covers as NaN, for methylation and conservation tracks whose zeros are measurements, recording the convention in the file's attributes; ``-s`` is checked against every bigWig's chromosome lengths. ``figwig fw2bw`` writes a figwig file's tracks back as bigWigs with such a table, so the two round-trip names and attributes, following the file's ``missing`` convention by default and giving tracks whose names collide one file each. ``figwig bam2fw`` takes what ``figwig bam2bw`` takes and writes each input file as its own pair of stranded tracks, or one unstranded track, as entries on the ``-p`` threads. ``figwig info`` prints what a file holds, every value on one printable line, and ``-b`` counts the bytes its chunks, tables and unused space take. Every command refuses to write over an output that exists unless ``-f`` is given (``--force`` for ``bam2fw``), writes its output beside its path and moves it into place when complete, prints a message rather than a traceback for a missing input, and exits quietly on Ctrl-C or a closed pipe. A bigWig's values read from the figwig file exactly as ``read_bigwig`` gives them with ``missing=0.0``.

Reading
-------

	- The bigWig reader takes a file whose index leaves lie in natural chromosome order, chr1, chr2, ..., chr10, where the ids run chr1, chr10, chr2, by sorting the leaves; two bigWigs of the test collection were refused as unsorted where pyBigWig read them. ``BigWigReader.read`` and ``read_bigwig`` take ``keep_nan=``, with which an interval whose value is NaN reads as NaN rather than ``missing``.

Writing
-------

	- ``_zoom_records``, the kernel that summarizes items into one zoom level's records, holds the open record in locals and stores it back when it returns, instead of writing it to the level's state arrays for every item. Zoom levels are summarized on different threads at once, and their small state arrays can share a cache line, which makes the kernel many times slower in the processes where the arrays happen to be allocated next to each other. The files written are unchanged. Thanks @YaoxiangLi!

Claude Code skill
-----------------

	- The skill covers figwig files: ``references/figwig-files.md`` gives the API, the layout that count tracks want (one group of many) and the one for reads, cells or channels as tracks, the lossless round trip of signal tracks through ``--missing nan``, the path onto a GPU, the converters and every error message.


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
