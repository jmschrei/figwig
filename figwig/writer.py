# writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import os
import sys
import zlib
import numbers

from concurrent.futures import FIRST_COMPLETED
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait

import numpy

from ._kernels import _decode_items
from ._kernels import _deflate_blocks_libdeflate
from ._kernels import _deflate_blocks_zlib
from ._kernels import _inflate_blocks
from ._kernels import _libdeflate
from ._kernels import _pread
from ._kernels import _update_summary
from ._kernels import _zlib_compress
from ._kernels import _zlib_uncompress
from ._kernels import _zoom_records
from .bigwig import _BIGWIG_MAGIC
from .bigwig import _CHROM_TREE_MAGIC
from .bigwig import _INDEX_MAGIC
from .bigwig import _check_missing
from .bigwig import _check_n_jobs


# The layout of a file is libBigWig's (pyBigWig's writer): a data block is at
# most 32768 bytes before compression, a 24-byte section header and as many
# items as fit, which is 2728 bedGraph, 4092 varStep or 8185 fixedStep items.
_BUF_SIZE = 32768
_BEDGRAPH, _VARSTEP, _FIXEDSTEP = 1, 2, 3
_BLOCK_ITEMS = {_BEDGRAPH: 2728, _VARSTEP: 4092, _FIXEDSTEP: 8185}
_ITEM_WORDS = {_BEDGRAPH: 3, _VARSTEP: 2, _FIXEDSTEP: 1}

# An index node has at most 64 children, and a zoom block 1023 records.
_RTREE_BLOCK_SIZE = 64
_ZOOM_BLOCK_RECORDS = 1023

# Items are laid out and compressed in batches of about this many, so that
# many small chromosomes cost a few numpy calls, and a long one does not
# have to be held twice in memory.
_BATCH_ITEMS = 1 << 20

# The data blocks read back in one task to build the zoom levels, the most
# of these chunks read past the one the slowest level is on, and the zoom
# blocks compressed in one task.
_ZOOM_CHUNK_BLOCKS = 64
_ZOOM_READ_AHEAD = 8
_ZOOM_COMPRESS_BLOCKS = 64

# The records the record kernel writes before it stops, into one buffer that
# each zoom level reuses; the records made are copied out of it.
_ZOOM_SCRATCH_RECORDS = 1 << 16

_FLT_MAX = float(numpy.finfo(numpy.float32).max)
_DBL_MAX = sys.float_info.max
_DBL_MIN = sys.float_info.min

_ENGINES = ('auto', 'zlib', 'libdeflate', 'isal')
_LEVELS = {'zlib': (0, 9), 'libdeflate': (0, 12), 'isal': (0, 3)}

_LEAF_ITEM = numpy.dtype([('start_tid', '<u4'), ('start', '<u4'),
	('end_tid', '<u4'), ('end', '<u4'), ('offset', '<u8'), ('size', '<u8')])

_HEADER = numpy.dtype([('magic', '<u4'), ('version', '<u2'),
	('zoom_levels', '<u2'), ('chrom_tree', '<u8'), ('data', '<u8'),
	('index', '<u8'), ('field_count', '<u2'), ('defined_field_count', '<u2'),
	('autosql', '<u8'), ('summary', '<u8'), ('buf_size', '<u4'),
	('extension', '<u8')])


def _check_chrom_sizes(chrom_sizes):
	"""The `chrom_sizes` argument as a list of (name, length) pairs, in
	order."""

	if isinstance(chrom_sizes, dict):
		pairs = list(chrom_sizes.items())
	else:
		try:
			pairs = [tuple(pair) for pair in chrom_sizes]
		except TypeError:
			raise TypeError("chrom_sizes must be a dict of chromosome lengths, or "
				"a list of (name, length) pairs.")

	if len(pairs) == 0:
		raise ValueError("chrom_sizes must name at least one chromosome.")
	if len(pairs) > 1073676289:
		raise ValueError("a bigWig can hold at most 1,073,676,289 chromosomes.")

	names = set()
	for pair in pairs:
		if len(pair) != 2:
			raise ValueError("each chromosome must be a (name, length) pair, not "
				"{!r}.".format(pair))

		name, length = pair
		if not isinstance(name, str):
			raise TypeError("chromosome names must be strings, not {}.".format(
				type(name).__name__))
		if len(name) == 0 or '\x00' in name:
			raise ValueError("chromosome names must be non-empty and contain no "
				"null characters: {!r}.".format(name))
		if name in names:
			raise ValueError("chromosome {!r} is named twice.".format(name))
		if isinstance(length, bool) or not isinstance(length, (int,
				numpy.integer)):
			raise TypeError("the length of chromosome {!r} must be an integer, "
				"not {}.".format(name, type(length).__name__))
		if not 1 <= length <= 0xFFFFFFFF:
			raise ValueError("the length of chromosome {!r} must be from 1 to "
				"2**32 - 1, not {}.".format(name, length))

		names.add(name)

	return [(name, int(length)) for name, length in pairs]


def _check_engine(engine, level):
	"""The compression engine `engine` resolves to, with `level` checked."""

	if engine not in _ENGINES:
		raise ValueError("engine must be one of {}, not {!r}.".format(
			', '.join(map(repr, _ENGINES)), engine))

	if engine == 'auto':
		engine = 'libdeflate' if _libdeflate() is not None else 'zlib'
	elif engine == 'libdeflate' and _libdeflate() is None:
		try:
			import deflate  # noqa: F401
		except ImportError:
			raise ValueError("engine='libdeflate' needs the deflate package: "
				"pip install deflate.")

		raise ValueError("engine='libdeflate' needs libdeflate's functions, "
			"which could not be loaded from the deflate package here, as on "
			"Windows; use engine='zlib'.")
	elif engine == 'isal':
		try:
			import isal.isal_zlib  # noqa: F401
		except ImportError:
			raise ValueError("engine='isal' needs the isal package: pip install "
				"isal.")

	low, high = _LEVELS[engine]
	if isinstance(level, bool) or not isinstance(level, (int, numpy.integer)):
		raise TypeError("level must be an integer.")
	if not low <= level <= high:
		raise ValueError("level must be from {} to {} with engine {!r}, not "
			"{}.".format(low, high, engine, level))

	return engine, int(level)


def _compress_blocks(engine, level, data, bounds):
	"""Compress the blocks data[bounds[k]:bounds[k + 1]] one after another.

	Returns (out, sizes): the blocks' zlib streams are back to back in
	out[:sizes.sum()], and sizes[k] is the length of block k's. zlib and
	libdeflate are called from numba without the GIL, so that several threads
	can compress at once; ISA-L through its Python module, which also lets
	other threads run while it compresses.
	"""

	n = len(bounds) - 1
	sizes = numpy.zeros(n, dtype=numpy.int64)
	view = memoryview(data)

	zlib_compress = _zlib_compress() if engine == 'zlib' else None
	if engine == 'isal' or (engine == 'zlib' and zlib_compress is None):
		if engine == 'isal':
			from isal.isal_zlib import compress
		else:
			compress = zlib.compress

		streams = [compress(view[bounds[k]:bounds[k + 1]], level) for k in
			range(n)]
		sizes[:] = [len(stream) for stream in streams]
		return numpy.frombuffer(b''.join(streams), dtype=numpy.uint8), sizes

	# A bound on the zlib stream of every block, with room to spare: zlib's
	# compressBound() is n + n/4096 + n/16384 + 13 for n bytes, and
	# libdeflate's is smaller still.
	out = numpy.empty(int(bounds[-1]) + int(bounds[-1]) // 8 + 64 * n,
		dtype=numpy.uint8)

	if engine == 'libdeflate':
		alloc, compress, free = _libdeflate()
		compressor = alloc(level)
		if compressor == 0:
			raise MemoryError("libdeflate could not allocate a compressor.")

		try:
			status = _deflate_blocks_libdeflate(compress, compressor, data, bounds,
				out, sizes)
		finally:
			free(compressor)
	else:
		compress2, ulong = zlib_compress
		status = _deflate_blocks_zlib(compress2, level, data, bounds, out, sizes,
			numpy.zeros(1, dtype=ulong))

	if status != -1:
		raise RuntimeError("block {} could not be compressed.".format(status))

	return out, sizes


def _chrom_tree(chroms, offset):
	"""The chromosome B+ tree, to be written at `offset`, as libBigWig lays
	it out: at most 0x7FFF keys per node, a non-leaf root only when more than
	one leaf is needed, and every node padded to the full block size."""

	n = len(chroms)
	keys = [name.encode('utf-8') for name, _ in chroms]
	key_size = max(len(key) for key in keys)
	per_block = min(n, 0x7FFF)
	n_blocks = -(-n // per_block)

	parts = [numpy.array([_CHROM_TREE_MAGIC, per_block, key_size, 8],
		dtype='<u4').tobytes(), numpy.array([n, 0], dtype='<u8').tobytes()]

	if n_blocks > 1:
		non_leaf_end = offset + 32 + 4 + per_block * (key_size + 8)
		leaf_size = per_block * (key_size + 8) + 4
		root = numpy.zeros(per_block, dtype=[('key', 'S{}'.format(key_size)),
			('offset', '<u8')])
		root['key'][:n_blocks] = keys[::per_block]
		root['offset'][:n_blocks] = non_leaf_end + leaf_size * numpy.arange(
			n_blocks, dtype=numpy.uint64)
		parts.append(bytes([0, 0]) + numpy.uint16(n_blocks).tobytes())
		parts.append(root.tobytes())

	leaves = numpy.zeros(n_blocks * per_block, dtype=[('key',
		'S{}'.format(key_size)), ('id', '<u4'), ('length', '<u4')])
	leaves['key'][:n] = keys
	leaves['id'][:n] = numpy.arange(n, dtype=numpy.uint32)
	leaves['length'][:n] = [length for _, length in chroms]

	for i in range(n_blocks):
		count = min(per_block, n - i * per_block)
		parts.append(bytes([1, 0]) + numpy.uint16(count).tobytes())
		parts.append(leaves[i * per_block:(i + 1) * per_block].tobytes())

	return b''.join(parts)


class _Node:
	"""An R-tree node: a leaf over a run of consecutive blocks, or a node over
	child nodes."""

	def __init__(self, first=0, count=0, children=None):
		self.first = first
		self.count = count
		self.children = children
		self.offset = 0

	def is_leaf(self):
		return self.children is None

	def first_block(self):
		return self.first if self.is_leaf() else self.children[0].first_block()

	def last_block(self):
		return self.first + self.count - 1 if self.is_leaf() else \
			self.children[-1].last_block()

	def n_items(self):
		return self.count if self.is_leaf() else len(self.children)

	def size(self):
		return 4 + (32 if self.is_leaf() else 24) * self.n_items()


def _add_leaves(leaves, position, to_process, size):
	"""libBigWig's addLeaves: a subtree over the next `to_process` leaves,
	spread as evenly as its order of ceil() calls allows. `size` adds up the
	bytes of the nodes it makes, as libBigWig's does."""

	node = _Node(children=[])
	if to_process <= _RTREE_BLOCK_SIZE:
		for _ in range(to_process):
			leaf = leaves[position[0]]
			position[0] += 1
			node.children.append(leaf)
			size[0] += 4 + 32 * leaf.count
	else:
		for i in range(_RTREE_BLOCK_SIZE):
			n = -(-to_process // (_RTREE_BLOCK_SIZE - i))
			node.children.append(_add_leaves(leaves, position, n, size))
			to_process -= n

	size[0] += 4 + 24 * len(node.children)
	return node


def _index(blocks, offset, items_per_slot):
	"""The R-tree index over `blocks`, to be written at `offset`, as
	libBigWig's writeIndex lays it out: the root first, then each level below
	it in order. `items_per_slot` is the header field libBigWig sets to 1 for
	the data and to 1024 for a zoom level."""

	n_blocks = blocks.size
	leaves = [_Node(first, min(_RTREE_BLOCK_SIZE, n_blocks - first)) for first in
		range(0, n_blocks, _RTREE_BLOCK_SIZE)]

	if len(leaves) == 1:
		root = leaves[0]
		index_size = 4 + 24 * root.count
	else:
		size = [0]
		root = _add_leaves(leaves, [0], len(leaves), size)
		index_size = size[0]

	order = [root]
	level = [root]
	while level:
		level = [child for node in level if not node.is_leaf() for child in
			node.children]
		order.extend(level)

	position = offset + 48
	for node in order:
		node.offset = position
		position += node.size()

	first = blocks[root.first_block()]
	last = blocks[root.last_block()]
	parts = [numpy.array([_INDEX_MAGIC, _RTREE_BLOCK_SIZE], dtype='<u4').tobytes(),
		numpy.uint64(n_blocks).tobytes(),
		numpy.array([first['start_tid'], first['start'], last['end_tid'],
			last['end']], dtype='<u4').tobytes(),
		numpy.uint64(index_size).tobytes(),
		numpy.array([items_per_slot, 0], dtype='<u4').tobytes()]

	for node in order:
		parts.append(bytes([1 if node.is_leaf() else 0, 0]) +
			numpy.uint16(node.n_items()).tobytes())

		if node.is_leaf():
			parts.append(blocks[node.first:node.first + node.count].tobytes())
		else:
			items = numpy.zeros(len(node.children), dtype=[('bounds', '<u4', 4),
				('offset', '<u8')])
			for i, child in enumerate(node.children):
				a, b = blocks[child.first_block()], blocks[child.last_block()]
				items[i] = ((a['start_tid'], a['start'], b['end_tid'], b['end']),
					child.offset)
			parts.append(items.tobytes())

	return b''.join(parts)


def _zoom_reductions(width_sum, n_items, max_length, zooms):
	"""The bases each zoom level's records summarize, as libBigWig's
	makeZoomLevels chooses them: 4 times 4 times the mean item width, or 10
	if that is less, then 4 times the last, while no more than the longest
	chromosome, for at most `zooms` levels. Where 4 times the mean width is
	more than 2**30 - 1, there are none; libBigWig's 32-bit arithmetic wraps
	around there instead."""

	mean = int(width_sum / n_items) * 4
	if 0xFFFFFFFF >> 2 < mean:
		return []

	zoom = max(10, 4 * mean)
	zoom = min(zoom, max_length)

	reductions = []
	for _ in range(zooms):
		if zoom > max_length:
			break

		reductions.append(zoom)
		if 0xFFFFFFFF // 4 < zoom:
			break
		zoom *= 4

	return reductions


class _Run:
	"""Items of one section type on one chromosome, added one after another,
	that go into consecutive data blocks. libBigWig, and pyBigWig through it,
	keeps adding to the open block while the next call is of the same type
	on the same chromosome and, for fixedStep, starts where the last one
	ended; anything else starts a new block."""

	def __init__(self, kind, tid, start):
		self.kind = kind
		self.tid = tid
		self.start = start
		self.starts = []
		self.ends = []
		self.values = []
		self.n = 0
		self.end = start

	def take(self, n):
		"""Remove the first n items, as arrays (starts, ends, values)."""

		starts = numpy.concatenate(self.starts) if self.kind != _FIXEDSTEP else \
			self.start + numpy.arange(self.n, dtype=numpy.int64)
		ends = numpy.concatenate(self.ends) if self.kind == _BEDGRAPH else \
			starts + 1
		values = numpy.concatenate(self.values)

		if self.kind != _FIXEDSTEP:
			self.starts = [starts[n:]]
		if self.kind == _BEDGRAPH:
			self.ends = [ends[n:]]

		self.values = [values[n:]]
		self.start += n if self.kind == _FIXEDSTEP else 0
		self.n -= n
		return starts[:n], ends[:n], values[:n]


class BigWigWriter:
	"""Write a bigWig file, chromosome by chromosome, on several threads.

	`BigWigWriter` takes the values of one chromosome at a time, in the order
	the chromosomes are given, and writes them as they come, so that a track
	larger than memory can be written a chromosome, or a part of one, at a
	time. Use it as a context manager, or call `close` when every value has
	been added; the header and the index are written there.

	Values can be added three ways, each with `add`:

	- intervals, each a start, an end and a value, written as bedGraph
	  sections;
	- single bases, each a position and a value, written as varStep sections
	  of span 1, which is what per-base counts, such as those of reads'
	  5' ends, are;
	- a dense array of values from one start, written as fixedStep sections
	  of span 1, leaving out the bases equal to `missing` or NaN, which is
	  what a model's predictions along a region are.

	The file is laid out the way libBigWig, pyBigWig's writer, lays it out:
	its header, chromosome tree, data blocks of at most 32768 bytes before
	compression, the R-tree index over them, and the zoom levels and their
	indexes. A file of intervals or single bases written without zoom levels,
	with engine='zlib' and level 6, is byte for byte the file pyBigWig writes
	from the same calls, given the same zlib, unless libBigWig gets one of
	the values below wrong. Where it does, the correct ones are written: the
	maximum in the header, which libBigWig misses when the first value is the
	largest, and leaves at the smallest positive double when no value is
	positive; the end of the last block of each fixedStep call, which
	libBigWig puts 6 bases past it; and the sum and sum of squares of the
	last zoom record of each zoom block, which libBigWig leaves at 0.
	libBigWig also writes an empty block, a section header without items,
	where it changes section type; those are not written here.

	Each data block is compressed on its own, so blocks are compressed on up
	to `n_jobs` threads at once while the calling thread lays out the next
	batch, and the file does not depend on the number of threads. zlib and
	libdeflate are called without holding the GIL.

	Parameters
	----------
	path: str or os.PathLike
		The file to write. It is created, or emptied, when the writer is made.

	chrom_sizes: dict or list of (str, int)
		The chromosomes and their lengths, in the order their values will be
		added: a dict such as `BigWig.chrom_sizes`, or a list of (name, length)
		pairs. Chromosomes that get no values are still in the file's header.

	zooms: int, optional
		The most zoom levels to write. Genome browsers draw zoomed-out views
		from them. Their sizes are chosen the way libBigWig chooses them, from
		16 times the mean width of an item, 4 times larger at each level, up
		to the longest chromosome; a level that would hold no fewer records
		than the last level kept is skipped. They are made when the writer is
		closed, from the data written, and add a pass over it. 0 writes none.
		Default is 10.

	level: int, optional
		The compression level: 0 to 9 with zlib, 0 to 12 with libdeflate, 0
		to 3 with ISA-L. Default is 6, zlib's own default and what pyBigWig
		uses.

	engine: str, optional
		What compresses the blocks: zlib's own library, 'zlib', or
		'libdeflate', which needs the `deflate` package, in figwig's `fast`
		extra, and is several times faster than zlib at the same level, for
		files as small or smaller. Both write the zlib streams every bigWig
		reader reads. Where zlib's library cannot be loaded, as on Windows,
		'zlib' compresses with Python's zlib module, with the same bytes, and
		libdeflate's functions cannot be loaded there either. 'auto' is
		libdeflate when its functions can be loaded, and zlib otherwise. 'isal' needs the `isal` package, and is faster still at
		levels 1 to 3, but at levels 1 and 2 its output has been seen to
		differ from one run to the next on the same input, so files written
		with it are not reproducible byte for byte; their values are.
		Default is 'auto'.

	n_jobs: int, optional
		The most threads that compress blocks at once, besides the calling
		thread, or -1 for one per CPU. Default is 8.
	"""

	def __init__(self, path: str | os.PathLike, chrom_sizes: dict | list,
		zooms: int = 10, level: int = 6, engine: str = 'auto', n_jobs: int = 8):
		self._file = None
		self._pool = None

		self.chrom_sizes = dict(_check_chrom_sizes(chrom_sizes))
		self._chrom_list = list(self.chrom_sizes.items())
		self._tids = {name: i for i, name in enumerate(self.chrom_sizes)}

		if isinstance(zooms, bool) or not isinstance(zooms, (int, numpy.integer)):
			raise TypeError("zooms must be an integer.")
		if not 0 <= zooms <= 0xFFFF:
			raise ValueError("zooms must be from 0 to 65535, not {}.".format(zooms))

		self.zooms = int(zooms)
		self.engine, self.level = _check_engine(engine, level)
		self.n_jobs = _check_n_jobs(n_jobs)
		self.path = path

		self._summary_offset = 64 + 24 * self.zooms
		self._chrom_tree_offset = self._summary_offset + 40

		self._file = open(path, 'w+b')
		self._file.write(bytes(self._chrom_tree_offset))
		self._file.write(_chrom_tree(self._chrom_list, self._chrom_tree_offset))
		self._data_offset = self._file.tell()
		self._file.write(bytes(8))

		self._run = None
		self._pending = []
		self._pending_n = 0
		self._last_tid = -1
		self._last_end = 0

		self._blocks = []
		self._sizes = []
		self._n_blocks = 0
		self._inflight = []

		self._counts = numpy.zeros(2, dtype=numpy.uint64)
		self._stats = numpy.array([numpy.inf, -numpy.inf, 0.0, 0.0])

		if self.n_jobs > 1:
			self._pool = ThreadPoolExecutor(self.n_jobs)

	def __repr__(self):
		return "BigWigWriter({!r})".format(os.fspath(self.path))

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, traceback):
		if exc_type is None:
			self.close()
		else:
			self._abandon()

	def _abandon(self):
		"""Stop without finishing the file, which is left without a header."""

		if self._pool is not None:
			self._pool.shutdown(cancel_futures=True)
			self._pool = None
		if self._file is not None:
			self._file.close()
			self._file = None

	def add(self, chrom: str, starts, ends=None, values=None,
		missing: float = 0.0):
		"""Add values on one chromosome.

		Chromosomes are added in the order of `chrom_sizes`, and within one,
		every call starts at or after the end of the last; a chromosome may be
		added over several calls. Positions are 0-based and ends are exclusive,
		as in BED files, and must lie within the chromosome. Values are written
		as float32, and must be finite.

		Parameters
		----------
		chrom: str
			The chromosome the values are on.

		starts: int or array-like of ints
			With `ends`, the starts of intervals; without, the positions of
			single bases. A single integer means that `values` is a dense array
			of the values of the bases from that position on.

		ends: array-like of ints or None, optional
			The ends of the intervals, each after its start and at or before
			the start of the next. Default is None.

		values: array-like of numbers
			The value of each interval, base, or base of the dense array.

		missing: float, optional
			Only for a dense array: bases whose value is `missing`, or NaN, are
			not written, so that `BigWig.read` with the same `missing` reads
			the array back. A base whose value is -0.0 is left out where
			`missing` is 0.0, and so reads back as 0.0. Default is 0.0.
		"""

		if self._file is None:
			raise ValueError("this BigWigWriter has been closed.")

		if not isinstance(chrom, str):
			raise TypeError("chrom must be a string, not {}.".format(
				type(chrom).__name__))
		if chrom not in self._tids:
			raise ValueError("chromosome {!r} is not one of the writer's "
				"chromosomes.".format(chrom))
		if values is None:
			raise TypeError("add needs values.")

		tid = self._tids[chrom]
		length = self.chrom_sizes[chrom]

		dense = isinstance(starts, (numbers.Integral, numpy.integer)) and \
			not isinstance(starts, bool)
		if dense:
			if ends is not None:
				raise ValueError("a dense array takes a single start and no ends.")

			runs = self._dense(chrom, int(starts), values, missing, length)
		else:
			runs = self._intervals(chrom, starts, ends, values, length)

		if len(runs) == 0:
			return

		if tid < self._last_tid:
			raise ValueError("chromosome {!r} is added after {!r}, but comes "
				"before it in chrom_sizes.".format(chrom, self._chrom_list[
				self._last_tid][0]))
		if tid == self._last_tid and runs[0][1] < self._last_end:
			raise ValueError("values on {!r} start at {}, before the end of the "
				"values already added on it, {}.".format(chrom, runs[0][1],
				self._last_end))

		for kind, first, run_starts, run_ends, run_values in runs:
			self._append(kind, tid, first, run_starts, run_ends, run_values)

		kind, first, _, run_ends, run_values = runs[-1]
		self._last_tid = tid
		self._last_end = first + len(run_values) if kind == _FIXEDSTEP else \
			int(run_ends[-1])

		if self._pending_n + (self._run.n if self._run is not None else 0) >= \
				_BATCH_ITEMS:
			self._flush(final=False)

	def _values(self, values, name):
		"""`values` as a 1-D float32 array, from any real numbers within
		float32's range."""

		values = numpy.asarray(values)
		if values.ndim != 1:
			raise ValueError("{} must be one-dimensional, not of shape {}.".format(
				name, values.shape))
		if values.dtype.kind not in 'biuf':
			raise TypeError("{} must be numbers, not {}.".format(name,
				values.dtype))

		if values.dtype.kind == 'f' and values.dtype.itemsize > 4:
			finite = values[numpy.isfinite(values)]
			if finite.size and (finite.max() > _FLT_MAX or finite.min() < -_FLT_MAX):
				raise ValueError("{} holds values outside float32's range.".format(
					name))

		return values.astype(numpy.float32)

	def _positions(self, positions, name):
		positions = numpy.asarray(positions)
		if positions.ndim != 1:
			raise ValueError("{} must be one-dimensional, not of shape {}.".format(
				name, positions.shape))
		if positions.size and positions.dtype.kind not in 'iu':
			raise TypeError("{} must be integers, not {}.".format(name,
				positions.dtype))

		return positions.astype(numpy.int64)

	def _intervals(self, chrom, starts, ends, values, length):
		"""Intervals or single bases as one run, after checking them."""

		starts = self._positions(starts, 'starts')
		values = self._values(values, 'values')
		if starts.shape != values.shape:
			raise ValueError("starts and values must be the same length, not {} "
				"and {}.".format(starts.size, values.size))

		if ends is None:
			kind = _VARSTEP
			ends = starts + 1
		else:
			kind = _BEDGRAPH
			ends = self._positions(ends, 'ends')
			if ends.shape != starts.shape:
				raise ValueError("starts and ends must be the same length, not {} "
					"and {}.".format(starts.size, ends.size))

		if starts.size == 0:
			return []

		if not numpy.isfinite(values).all():
			raise ValueError("values on {!r} must be finite; leave out the bases "
				"or intervals without a value.".format(chrom))
		if starts[0] < 0:
			raise ValueError("values on {!r} start before 0, at {}.".format(chrom,
				starts[0]))
		if (ends <= starts).any():
			j = int(numpy.argmax(ends <= starts))
			raise ValueError("interval {} on {!r} ends at {}, at or before its "
				"start, {}.".format(j, chrom, ends[j], starts[j]))
		if (starts[1:] < ends[:-1]).any():
			j = int(numpy.argmax(starts[1:] < ends[:-1])) + 1
			raise ValueError("{} {} on {!r}, at {}, starts before the end of the "
				"one before it, {}; they must be sorted and must not overlap."
				.format('interval' if kind == _BEDGRAPH else 'position', j, chrom,
				starts[j], ends[j - 1]))
		if ends[-1] > length:
			raise ValueError("values on {!r} end at {}, past its length, {}."
				.format(chrom, ends[-1], length))

		return [(kind, int(starts[0]), starts, ends, values)]

	def _dense(self, chrom, start, values, missing, length):
		"""The runs of a dense array's bases that are not `missing` or NaN."""

		values = self._values(values, 'values')
		missing = _check_missing(missing)

		if start < 0:
			raise ValueError("values on {!r} start before 0, at {}.".format(chrom,
				start))
		if start + values.size > length:
			raise ValueError("values on {!r} end at {}, past its length, {}."
				.format(chrom, start + values.size, length))
		if numpy.isinf(values).any():
			raise ValueError("values on {!r} must be finite or NaN.".format(chrom))

		keep = ~numpy.isnan(values)
		if not numpy.isnan(missing):
			keep &= values != missing

		edges = numpy.flatnonzero(numpy.diff(keep.astype(numpy.int8), prepend=0,
			append=0))
		return [(_FIXEDSTEP, start + int(a), None, None, values[a:b]) for a, b in
			zip(edges[::2], edges[1::2])]

	def _append(self, kind, tid, first, starts, ends, values):
		"""Add one run of items, continuing the open run where libBigWig
		would continue its open block."""

		run = self._run
		if run is None or run.kind != kind or run.tid != tid or (
				kind == _FIXEDSTEP and first != run.end):
			if run is not None:
				self._pending.append(run)
				self._pending_n += run.n

			run = self._run = _Run(kind, tid, first)

		if kind != _FIXEDSTEP:
			run.starts.append(starts)
		if kind == _BEDGRAPH:
			run.ends.append(ends)

		run.values.append(values)
		run.n += len(values)
		run.end = first + len(values) if kind == _FIXEDSTEP else int(ends[-1])

	def _flush(self, final):
		"""Lay out and compress the pending runs and, unless this is the final
		flush, the whole blocks of the open run, and write every batch that
		has finished compressing."""

		runs = []
		for run in self._pending:
			runs.append((run.kind, run.tid) + run.take(run.n))

		self._pending = []
		self._pending_n = 0

		run = self._run
		if run is not None:
			n = run.n if final else run.n - run.n % _BLOCK_ITEMS[run.kind]
			if n > 0:
				runs.append((run.kind, run.tid) + run.take(n))
			if final:
				self._run = None

		if len(runs) > 0:
			self._submit(runs)

		self._write(len(self._inflight) - (0 if final or self._pool is None else 1))

	def _submit(self, runs):
		"""Lay one batch of runs out as data blocks, fold it into the summary,
		and compress it, on the pool when there is one."""

		words, bounds, blocks, spans, values = _layout(runs)
		_update_summary(spans, values, self._counts, self._stats)
		self._sizes.append(numpy.diff(bounds))

		data = words.view(numpy.uint8)
		if self._pool is None:
			parts = [_compress_blocks(self.engine, self.level, data, bounds)]
		else:
			# Contiguous shares of about equal bytes, one per thread.
			cuts = numpy.searchsorted(bounds[1:], bounds[-1] * numpy.arange(1,
				self.n_jobs) / self.n_jobs).tolist()
			parts = []
			for a, b in zip([0] + cuts, cuts + [blocks.size]):
				if b > a:
					parts.append(self._pool.submit(_compress_blocks, self.engine,
						self.level, data[bounds[a]:bounds[b]], bounds[a:b + 1] -
						bounds[a]))

		self._inflight.append((parts, blocks))

	def _write(self, n_batches):
		"""Write the first `n_batches` batches in flight, in order, waiting for
		them to be compressed."""

		for parts, blocks in self._inflight[:n_batches]:
			offset = self._file.tell()
			sizes = []
			for part in parts:
				out, part_sizes = part if isinstance(part, tuple) else part.result()
				self._file.write(out[:part_sizes.sum()])
				sizes.append(part_sizes)

			sizes = numpy.concatenate(sizes)
			blocks['offset'] = offset + numpy.cumsum(sizes) - sizes
			blocks['size'] = sizes
			self._blocks.append(blocks)
			self._n_blocks += blocks.size

		del self._inflight[:n_batches]

	def close(self):
		"""Write what is left, the index, the zoom levels and the header, and
		close the file. Calling it again does nothing."""

		if self._file is None:
			return

		try:
			self._flush(final=True)

			end = self._file.tell()
			blocks = numpy.concatenate(self._blocks) if self._n_blocks else None
			if blocks is not None:
				self._file.write(_index(blocks, end, 1))
				index_offset = end
			else:
				index_offset = 0

			levels = []
			if self.zooms > 0 and blocks is not None:
				levels = self._write_zoom_levels(blocks)

			self._file.write(numpy.uint32(_BIGWIG_MAGIC).tobytes())

			header = numpy.zeros(1, dtype=_HEADER)
			header['magic'] = _BIGWIG_MAGIC
			header['version'] = 4
			header['zoom_levels'] = len(levels)
			header['chrom_tree'] = self._chrom_tree_offset
			header['data'] = self._data_offset
			header['index'] = index_offset
			header['summary'] = self._summary_offset
			header['buf_size'] = _BUF_SIZE

			zoom_headers = numpy.zeros(len(levels), dtype=[('reduction', '<u4'),
				('reserved', '<u4'), ('data', '<u8'), ('index', '<u8')])
			for i, level in enumerate(levels):
				zoom_headers[i] = level

			# A file without values has libBigWig's starting minimum and maximum.
			stats = self._stats.copy()
			if self._counts[1] == 0:
				stats[:2] = _DBL_MAX, _DBL_MIN

			summary = numpy.uint64(self._counts[0]).tobytes() + numpy.array(
				stats, dtype='<f8').tobytes()

			self._file.seek(0)
			self._file.write(header.tobytes())
			self._file.write(zoom_headers.tobytes())
			self._file.seek(self._summary_offset)
			self._file.write(summary)
			self._file.seek(self._data_offset)
			self._file.write(numpy.uint64(self._n_blocks).tobytes())
		finally:
			self._abandon()

	def _write_zoom_levels(self, blocks):
		"""Build the zoom levels from the data blocks, which are read back from
		the file, and write those worth keeping. Returns (reduction, 0, data
		offset, index offset) for each level written."""

		reductions = _zoom_reductions(int(self._counts[0]), int(self._counts[1]),
			max(length for _, length in self._chrom_list), self.zooms)
		if len(reductions) == 0:
			return []

		zooms = [_ZoomLevel(size) for size in reductions]

		self._file.flush()
		fd = self._file.fileno()
		lengths = numpy.concatenate(self._sizes)

		def read(k):
			"""Start reading chunk `k` of the blocks."""

			a = k * _ZOOM_CHUNK_BLOCKS
			b = a + _ZOOM_CHUNK_BLOCKS
			return self._task(_read_items, fd, blocks[a:b], lengths[a:b])

		# The finest level is always kept, so it is written as its blocks are
		# compressed rather than held to the end, where reading back leaves the
		# file's position alone.
		streamed = None
		if hasattr(os, 'preadv') or hasattr(os, 'pread'):
			streamed = zooms[0]
			self._start_zoom_level(streamed)

		# A level summarizes the chunks one after another, since its open record
		# carries over, but each starts its next chunk as soon as it is done
		# with one and the chunk has been read, without waiting for the other
		# levels. Chunks are read up to `ahead` past the one the slowest level
		# is on, and the zoom blocks the summaries complete are compressed on
		# the pool meanwhile.
		ahead = min(self.n_jobs, _ZOOM_READ_AHEAD)
		n_chunks = -(-blocks.size // _ZOOM_CHUNK_BLOCKS)
		chunks = {}
		n_read = 0
		positions = [0] * len(zooms)
		running = [None] * len(zooms)
		while min(positions) < n_chunks:
			while n_read < min(n_chunks, min(positions) + ahead + 1):
				chunks[n_read] = read(n_read)
				n_read += 1

			progressed = False
			for i, zoom in enumerate(zooms):
				if running[i] is not None and running[i].done():
					self._compress_zoom(zoom, *running[i].result())
					running[i] = None
					positions[i] += 1
					progressed = True

				k = positions[i]
				if running[i] is None and k < n_read and chunks[k].done():
					running[i] = self._task(zoom.add, *chunks[k].result())
					progressed = True

			for k in [k for k in chunks if k < min(positions)]:
				del chunks[k]

			if streamed is not None:
				self._write_zoom_blocks(streamed, False)

			if not progressed:
				wait([summary for summary in running if summary is not None] + [chunk
					for chunk in chunks.values() if not chunk.done()],
					return_when=FIRST_COMPLETED)

		for zoom in zooms:
			self._compress_zoom(zoom, *zoom.finish())

		# Where there is no pread, as on Windows, reading moved the file's
		# position, so the zoom levels are written from its end explicitly.
		self._file.seek(0, os.SEEK_END)
		written = []
		fewest = None
		for zoom in zooms:
			if fewest is not None and zoom.n_records >= fewest:
				continue

			fewest = zoom.n_records
			if zoom is not streamed:
				self._start_zoom_level(zoom)

			self._write_zoom_blocks(zoom, True)
			index_offset = self._file.tell()
			self._file.seek(zoom.data_offset)
			self._file.write(numpy.uint32(len(zoom.entries)).tobytes())
			self._file.seek(index_offset)

			index = numpy.array(zoom.entries, dtype=_LEAF_ITEM)
			self._file.write(_index(index, index_offset, _BUF_SIZE // 32))
			written.append((zoom.size, 0, zoom.data_offset, index_offset))

		return written

	def _start_zoom_level(self, zoom):
		"""Write a zoom level's block count, to be filled in once its blocks
		are written, at the end of the file."""

		zoom.data_offset = self._file.tell()
		self._file.write(numpy.uint32(0).tobytes())

	def _write_zoom_blocks(self, zoom, wait):
		"""Write a zoom level's compressed blocks in order, and their index
		entries, waiting for those not yet compressed or, unless `wait`,
		stopping at the first of them."""

		while zoom.n_written < len(zoom.blocks):
			part = zoom.blocks[zoom.n_written]
			if not wait and not part.done():
				break

			for stream, entry in part.result():
				zoom.entries.append(entry + (self._file.tell(), len(stream)))
				self._file.write(stream)

			zoom.blocks[zoom.n_written] = None
			zoom.n_written += 1

	def _task(self, function, *args):
		"""Run function(*args) on the pool, or here when there is none."""

		if self._pool is not None:
			return self._pool.submit(function, *args)

		return _Done(function(*args))

	def _compress_zoom(self, zoom, ints, floats):
		"""Start compressing whole zoom blocks of records, in parts of
		_ZOOM_COMPRESS_BLOCKS blocks, and add the parts to the level's."""

		step = _ZOOM_BLOCK_RECORDS * _ZOOM_COMPRESS_BLOCKS
		for a in range(0, len(ints), step):
			zoom.blocks.append(self._task(_compress_zoom_blocks, self.engine,
				self.level, ints[a:a + step], floats[a:a + step]))


class _Done:
	"""The result of a function that has already run, read as a future's."""

	def __init__(self, value):
		self.value = value

	def result(self):
		return self.value

	def done(self):
		return True


def _layout(runs):
	"""Lay runs of items out as data blocks, each one section.

	`runs` are (kind, chromosome id, starts, ends, values), in file order.
	A run is cut into blocks of as many items as fit, the last of which may
	be short. Returns the blocks' 32-bit words, back to back; the byte offset
	of each block in them and of their end; an index entry for each block,
	without its offset and size; and every item's span and value in file
	order, for the total summary.
	"""

	kinds = numpy.array([run[0] for run in runs])
	tids = numpy.array([run[1] for run in runs], dtype=numpy.int64)
	counts = numpy.array([len(run[4]) for run in runs], dtype=numpy.int64)
	capacity = numpy.array([_BLOCK_ITEMS[kind] for kind in kinds],
		dtype=numpy.int64)

	starts = numpy.concatenate([run[2] for run in runs])
	ends = numpy.concatenate([run[3] for run in runs])
	values = numpy.concatenate([run[4] for run in runs])

	run_blocks = -(-counts // capacity)
	n_blocks = int(run_blocks.sum())
	block_run = numpy.repeat(numpy.arange(len(runs)), run_blocks)
	block_rank = numpy.arange(n_blocks) - numpy.repeat(numpy.cumsum(run_blocks)
		- run_blocks, run_blocks)
	run_first = numpy.cumsum(counts) - counts
	first = run_first[block_run] + block_rank * capacity[block_run]
	n_items = numpy.minimum(first + capacity[block_run], (run_first +
		counts)[block_run]) - first
	block_kinds = kinds[block_run]

	# A fixedStep block of n items ends n bases after its start, the end of
	# its last item, as for the other types.
	headers = numpy.zeros((n_blocks, 6), dtype='<u4')
	headers[:, 0] = tids[block_run]
	headers[:, 1] = starts[first]
	headers[:, 2] = ends[first + n_items - 1]
	headers[:, 3] = block_kinds == _FIXEDSTEP
	headers[:, 4] = block_kinds != _BEDGRAPH
	headers[:, 5] = block_kinds | (n_items << 16)

	# The items' words, in groups of consecutive runs of the same type.
	item_words = []
	group_starts = numpy.flatnonzero(numpy.diff(kinds, prepend=-1))
	for g, a in enumerate(group_starts):
		b = group_starts[g + 1] if g + 1 < len(group_starts) else len(runs)
		lo, hi = run_first[a], run_first[b - 1] + counts[b - 1]

		kind = kinds[a]
		words = numpy.empty((hi - lo, _ITEM_WORDS[kind]), dtype='<u4')
		if kind == _FIXEDSTEP:
			words[:, 0] = values[lo:hi].view('<u4')
		else:
			words[:, 0] = starts[lo:hi]
			words[:, -1] = values[lo:hi].view('<u4')
			if kind == _BEDGRAPH:
				words[:, 1] = ends[lo:hi]

		item_words.append(words.ravel())

	block_words = 6 + numpy.array([_ITEM_WORDS[kind] for kind in block_kinds],
		dtype=numpy.int64) * n_items
	block_offsets = numpy.cumsum(block_words) - block_words
	header_slots = (block_offsets[:, None] + numpy.arange(6)).ravel()

	words = numpy.empty(int(block_words.sum()), dtype='<u4')
	is_item = numpy.ones(words.size, dtype=bool)
	is_item[header_slots] = False
	words[header_slots] = headers.ravel()
	words[is_item] = numpy.concatenate(item_words)

	bounds = 4 * numpy.concatenate([[0], numpy.cumsum(block_words)])

	blocks = numpy.zeros(n_blocks, dtype=_LEAF_ITEM)
	blocks['start_tid'] = headers[:, 0]
	blocks['start'] = headers[:, 1]
	blocks['end_tid'] = headers[:, 0]
	blocks['end'] = headers[:, 2]

	spans = (ends - starts).astype(numpy.uint32)
	return words, bounds, blocks, spans, values


def _read_items(fd, blocks, lengths):
	"""The items of the data blocks `blocks`, read back from the file, as
	(chromosome ids, starts, ends, values) in file order. lengths[k] is block
	k's length before compression. The blocks are consecutive in the file."""

	first = int(blocks['offset'][0])
	total = int(blocks['offset'][-1] + blocks['size'][-1]) - first
	data = numpy.frombuffer(_pread(fd, total, first), dtype=numpy.uint8)
	starts = (blocks['offset'] - first).astype(numpy.int64)
	sizes = blocks['size'].astype(numpy.int64)

	buffer = numpy.empty(int(lengths.sum()), dtype=numpy.uint8)
	inflated = 0
	uncompress = _zlib_uncompress()
	if uncompress is not None:
		flags = numpy.zeros((blocks.size, 6), dtype=numpy.int64)
		flags[:, 5] = 1
		inflated = _inflate_blocks(uncompress[0], data, starts, sizes, buffer,
			flags, numpy.zeros(1, dtype=uncompress[1]))

	if inflated != buffer.size:
		buffer = numpy.frombuffer(b''.join(zlib.decompress(data[a:a + n]) for a, n
			in zip(starts, sizes)), dtype=numpy.uint8)

	words = buffer.view('<u4')
	bounds = numpy.concatenate([[0], numpy.cumsum(lengths // 4)])
	n = int((words[bounds[:-1] + 5] >> 16).astype(numpy.int64).sum())

	tids = numpy.empty(n, dtype=numpy.int64)
	item_starts = numpy.empty(n, dtype=numpy.int64)
	item_ends = numpy.empty(n, dtype=numpy.int64)
	values = numpy.empty(n, dtype=numpy.float32)
	_decode_items(words, bounds, tids, item_starts, item_ends, values)
	return tids, item_starts, item_ends, values


class _ZoomLevel:
	"""The records of one zoom level, built from items as they are read back.

	`add` and `finish` give back the records of the zoom blocks they complete,
	to be compressed elsewhere; `blocks` collects the futures of their
	compressed streams, in order, and the first `n_written` have been written
	to the file at `data_offset`, with `entries` their index entries.
	"""

	def __init__(self, size):
		self.size = size
		self.open_ints = numpy.zeros(6, dtype=numpy.int64)
		self.open_floats = numpy.zeros(4, dtype=numpy.float64)
		self.pending_ints = numpy.zeros((0, 4), dtype=numpy.uint32)
		self.pending_floats = numpy.zeros((0, 4), dtype=numpy.float32)
		self.scratch_ints = numpy.empty((_ZOOM_SCRATCH_RECORDS, 4),
			dtype=numpy.uint32)
		self.scratch_floats = numpy.empty((_ZOOM_SCRATCH_RECORDS, 4),
			dtype=numpy.float32)
		self.blocks = []
		self.n_records = 0
		self.n_written = 0
		self.entries = []
		self.data_offset = None

	def add(self, tids, starts, ends, values):
		"""Summarize items, given by chromosome id, start, end and value in file
		order, and return the records of every zoom block they complete, as
		(ints, floats)."""

		records = []
		progress = numpy.array([0, -1], dtype=numpy.int64)
		while progress[0] < len(values):
			n = _zoom_records(self.size, tids, starts, ends, values, progress,
				self.open_ints, self.open_floats, self.scratch_ints,
				self.scratch_floats)
			records.append((self.scratch_ints[:n].copy(),
				self.scratch_floats[:n].copy()))

		return self._take(records, False)

	def finish(self):
		"""Close the last record, and return the records of the last blocks."""

		ints = numpy.zeros((0, 4), dtype=numpy.uint32)
		floats = numpy.zeros((0, 4), dtype=numpy.float32)
		if self.open_ints[0] == 1:
			ints = self.open_ints[None, 2:6].astype(numpy.uint32)
			floats = self.open_floats[None].astype(numpy.float32)
			self.open_ints[0] = 0

		return self._take([(ints, floats)], True)

	def _take(self, records, final):
		self.n_records += sum(len(ints) for ints, _ in records)
		ints = numpy.concatenate([self.pending_ints] + [i for i, _ in records])
		floats = numpy.concatenate([self.pending_floats] + [f for _, f in records])

		n = len(ints) if final else len(ints) - len(ints) % _ZOOM_BLOCK_RECORDS
		self.pending_ints = ints[n:]
		self.pending_floats = floats[n:]
		return ints[:n], floats[:n]


def _compress_zoom_blocks(engine, level, ints, floats):
	"""Lay zoom records out as zoom blocks of 1023 records, the last of which
	may be short, and compress each. Returns (stream, index entry) for each
	block, its entry being the first record's chromosome and start and the
	last record's chromosome and end."""

	records = numpy.empty((len(ints), 8), dtype='<u4')
	records[:, :4] = ints
	records[:, 4:] = floats.view('<u4')

	firsts = numpy.arange(0, len(ints), _ZOOM_BLOCK_RECORDS)
	lasts = numpy.minimum(firsts + _ZOOM_BLOCK_RECORDS, len(ints)) - 1
	bounds = 32 * numpy.append(firsts, len(ints)).astype(numpy.int64)
	out, sizes = _compress_blocks(engine, level, records.view(numpy.uint8).ravel(),
		bounds)

	offsets = numpy.cumsum(sizes) - sizes
	return [(out[offset:offset + size].tobytes(), (int(ints[first, 0]),
		int(ints[first, 1]), int(ints[last, 0]), int(ints[last, 2]))) for first,
		last, offset, size in zip(firsts, lasts, offsets, sizes)]


def write_bigwig(path: str | os.PathLike, chrom_sizes: dict | list, data: dict,
	zooms: int = 10, level: int = 6, engine: str = 'auto', n_jobs: int = 8,
	missing: float = 0.0):
	"""Write a bigWig file from values held in memory, in one call.

	`write_bigwig` writes every chromosome of `data` with one
	`BigWigWriter`, in the order of `chrom_sizes`, whatever order `data` is
	in. Each chromosome's values are given the way `BigWigWriter.add` takes
	them: a tuple of (starts, ends, values) for intervals, a tuple of
	(positions, values) for single bases, and anything else, such as a numpy
	array or a list, for a dense array of the values of every base from the start of the
	chromosome. A dense array may be shorter than its chromosome, and its
	bases equal to `missing`, or NaN, are left out of the file.

	If writing fails part way, the file is left without a header, which no
	reader takes for a bigWig.

	Parameters
	----------
	path: str or os.PathLike
		The file to write. It is created, or emptied.

	chrom_sizes: dict or list of (str, int)
		The chromosomes and their lengths, in the order they are written: a
		dict such as `BigWig.chrom_sizes`, or a list of (name, length) pairs. A
		chromosome without values is still in the file's header.

	data: dict
		The values of each chromosome that has any, keyed by name.

	zooms: int, optional
		The most zoom levels to write; see `BigWigWriter`. Default is 10.

	level: int, optional
		The compression level; see `BigWigWriter`. Default is 6.

	engine: str, optional
		What compresses the blocks: 'auto', 'zlib', 'libdeflate' or 'isal';
		see `BigWigWriter`. Default is 'auto'.

	n_jobs: int, optional
		The most threads that compress blocks at once, besides the calling
		thread, or -1 for one per CPU. Default is 8.

	missing: float, optional
		The value of the bases of dense arrays that are not written. Default
		is 0.0.
	"""

	if not isinstance(data, dict):
		raise TypeError("data must be a dict of each chromosome's values, not "
			"{}.".format(type(data).__name__))

	with BigWigWriter(path, chrom_sizes, zooms=zooms, level=level, engine=engine,
			n_jobs=n_jobs) as writer:
		unknown = [chrom for chrom in data if chrom not in writer.chrom_sizes]
		if len(unknown) > 0:
			raise ValueError("data has chromosomes that are not in chrom_sizes: {}."
				.format(', '.join(map(repr, unknown))))

		for chrom in writer.chrom_sizes:
			if chrom not in data:
				continue

			values = data[chrom]
			if isinstance(values, tuple):
				if len(values) == 3:
					writer.add(chrom, values[0], values[1], values[2])
				elif len(values) == 2:
					writer.add(chrom, values[0], values=values[1])
				else:
					raise ValueError("the values of {!r} are a tuple of {} arrays; "
						"give (starts, ends, values) or (positions, values)."
						.format(chrom, len(values)))
			else:
				writer.add(chrom, 0, values=values, missing=missing)
