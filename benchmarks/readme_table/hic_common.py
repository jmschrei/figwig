# hic_common.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Helpers shared by the Hi-C rows (`hic.py`) and the single-cell Hi-C row
(`schic.py`).

Both rows start every writer from a pixel table on disk: the upper-triangle
raw counts of one resolution, sorted by (bin1, bin2), as flat int32 files
(`bin1.i32`, `bin2.i32`, `count.i32`) and a `meta.json` with the chromosomes,
their bin offsets and the totals. For single cells the table also has
`cell.i32` and `cell_offsets.i64`, sorted by (cell, bin1, bin2). Writers read
it in chunks with `numpy.fromfile`, so a process holds only what it asked for.
"""

import os
import json
import time
import hashlib

import numpy


def load_meta(directory):
	with open(os.path.join(directory, 'meta.json')) as handle:
		meta = json.load(handle)
	meta['chrom_offsets'] = numpy.array(meta['chrom_offsets'], dtype=numpy.int64)
	path = os.path.join(directory, 'cell_offsets.i64')
	if os.path.exists(path):
		meta['cell_offsets'] = numpy.fromfile(path, dtype=numpy.int64)
	return meta


def read_slice(directory, column, start, stop):
	"""Pixels [start, stop) of one column of a pixel table, as int32."""

	with open(os.path.join(directory, column + '.i32'), 'rb') as handle:
		return numpy.fromfile(handle, dtype=numpy.int32, count=stop - start,
			offset=4 * start)


def chunks(directory, size=20_000_000):
	"""(bin1, bin2, count) chunks of a pixel table in its order."""

	n = load_meta(directory)['nnz']
	for start in range(0, n, size):
		stop = min(n, start + size)
		yield tuple(read_slice(directory, c, start, stop) for c in ('bin1', 'bin2', 'count'))


def bin_starts(meta):
	"""The start (bp) and chromosome index of every bin."""

	offsets = meta['chrom_offsets']
	chrom = numpy.repeat(numpy.arange(len(meta['chroms'])), numpy.diff(offsets))
	starts = (numpy.arange(int(offsets[-1])) - offsets[chrom]) * meta['resolution']
	return starts.astype(numpy.int64), chrom


def tile_calls(positions, tile, size):
	"""Cut points that split sorted positions into calls of about `size`
	entries, each ending on a tile boundary."""

	tiles = positions // tile
	cuts = [0]
	while cuts[-1] < len(positions):
		target = cuts[-1] + size
		if target >= len(positions):
			cuts.append(len(positions))
			break
		cuts.append(int(numpy.searchsorted(tiles, tiles[target], side='right')))
	return cuts


def digest(*arrays):
	"""A short hash of arrays, as int64, for the cross-format check."""

	h = hashlib.sha1()
	for array in arrays:
		h.update(numpy.ascontiguousarray(array, dtype=numpy.int64).tobytes())
	return h.hexdigest()[:16]


def mirror(m):
	"""The symmetric matrix of an upper triangle."""

	return m + m.T - numpy.diag(numpy.diag(m))


def upper_sorted(rows, cols, vals):
	"""The pixels on or above the diagonal, sorted by (row, column)."""

	keep = rows <= cols
	rows, cols, vals = rows[keep], cols[keep], vals[keep]
	order = numpy.lexsort((cols, rows))
	return rows[order], cols[order], vals[order]


def time_queries(run, queue, warmup, budget, verify, check):
	"""Time `run(item)` over `queue` one query at a time, after running the
	`warmup` items untimed, and stop once `budget` seconds have passed. The
	first `verify` queries are then run again, untimed, and their output
	hashed with `check`. Returns the median per query, the number timed and
	the hashes."""

	for item in warmup:
		run(item)
	times = []
	start = time.perf_counter()
	for item in queue:
		t0 = time.perf_counter()
		run(item)
		times.append(time.perf_counter() - t0)
		if time.perf_counter() - start > budget:
			break
	hashes = [check(run(item)) for item in queue[:min(verify, len(times))]]
	return {'seconds': float(numpy.median(times)), 'n_done': len(times),
		'min_seconds': float(min(times)), 'hashes': hashes}


def same_hashes(label, a, b):
	"""Raise if two readers disagree on the queries both ran; returns how
	many queries were compared."""

	n = min(len(a), len(b))
	if n == 0 or a[:n] != b[:n]:
		raise RuntimeError('{}: the outputs of the two formats differ over {} '
			'queries ({} vs {})'.format(label, n, a[:n], b[:n]))
	return n
