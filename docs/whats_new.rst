.. currentmodule:: figwig


===============
Release History
===============


Version 0.1.0 (unreleased)
==========================

Highlights
----------

	- Initial release. ``BigWig(path).read(chroms, starts, width, out=None, n_jobs=8, missing=0.0)`` reads the per-base values of many windows of one width into a float32 numpy array of shape ``(n, width)``, on up to ``n_jobs`` threads, or one per CPU with ``n_jobs=-1``. ``read_windows`` does the same in one call, and given a list of bigWigs reads them all into one array of shape ``(n, len(bigwigs), width)``, with every file's batches on one pool of threads.

	- The values are those pybigtools' ``values(chrom, start, end, missing=missing)`` gives: ``missing``, 0 unless given, where no interval covers a base or an interval's value is NaN, and NaN past the end of a chromosome. A window on a chromosome that the file does not have, which writers leave out when it has no data, is ``missing`` throughout, and a warning names the chromosomes.

	- bedGraph, varStep and fixedStep sections are read, compressed or not, including pyBigWig's fixedStep blocks whose index entries overlap by a few bases. A file or data block that figwig does not read, such as a bigBed, a big-endian bigWig, a corrupt or truncated index, or a block with overlapping intervals, raises a ``ValueError`` that says what it found.

	- A ``BigWig`` can be read from several threads at once and pickled, so it can be part of a PyTorch Dataset read by DataLoader workers under any start method.

	- The reader began inside tangermeme's ``extract_loci`` (tangermeme #107).
