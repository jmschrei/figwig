.. currentmodule:: figwig


===============
Release History
===============


Version 0.1.0 (unreleased)
==========================

Highlights
----------

	- Initial release. ``BigWig(path).read(chroms, starts, width, out=None, n_jobs=8)`` reads the per-base values of many windows of one width into a float32 numpy array of shape ``(n, width)``, on up to ``n_jobs`` threads, and ``read_windows`` does the same in one call. The values are those pybigtools' ``values()`` gives with its defaults: 0 where no interval covers a base, and NaN past the end of a chromosome.

	- bedGraph, varStep and fixedStep sections are read, compressed or not. A file or data block that figwig does not read, such as a bigBed, a big-endian bigWig, or a block with overlapping intervals, raises a ``ValueError`` that says what it found.

	- The reader began inside tangermeme's ``extract_loci`` (tangermeme #107).
