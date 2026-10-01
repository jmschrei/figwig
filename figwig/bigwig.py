# bigwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import os
import sys
import zlib
import struct
import warnings
import threading

from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy

from ._kernels import _inflate_blocks
from ._kernels import _pread
from ._kernels import _pread_into
from ._kernels import _read_windows
from ._kernels import _zlib_uncompress


# The number of data blocks each task decompresses at once. A block holds at
# most uncompressBufSize bytes, 32 KB in ENCODE's bigWigs, so n_jobs tasks
# hold at most n_jobs * 256 * 32 KB of decompressed data.
_BATCH_BLOCKS = 256

# A task inflates its blocks into a buffer of uncompressBufSize bytes per
# block, or of this many when uncompressBufSize is larger. A block that does
# not fit in what is left of the buffer is inflated by zlib.decompress.
_MAX_BLOCK_BYTES = 2 ** 20

_BIGWIG_MAGIC = 0x888FFC26
_BIGBED_MAGIC = 0x8789F2EB
_CHROM_TREE_MAGIC = 0x78CA8C91
_INDEX_MAGIC = 0x2468ACE0

_LEAF_TYPE = numpy.dtype([('start_chrom', '<u4'), ('start', '<u4'),
	('end_chrom', '<u4'), ('end', '<u4'), ('offset', '<u8'), ('size', '<u8')])


class BigWig:
	"""A bigWig file, read many windows at a time on several threads.

	`BigWig` reads the per-base values of many windows of the same width in
	one call, straight into a float32 numpy array. It is built for loading
	training data: tens or hundreds of thousands of windows, such as 1,000 bp
	around every peak and background region of an experiment.

	The file's header and chromosome tree are read when it is opened, and its
	data index on the first read, after which the index is kept for every
	later read. A read sorts its windows by position, so the order they are
	given in does not matter. It finds the data blocks overlapping each window
	in the index, and works through those blocks in batches of 256. Each
	batch is read from the file, decompressed and decoded straight into the
	windows' rows of the output. A block that several windows of one batch
	share, because they overlap or repeat, is decompressed once. Batches run
	on up to `n_jobs` threads. zlib's own `uncompress()` and the numba decoder
	both run without the GIL, so the threads run in parallel.

	Each base gets the value of the interval that covers it. A base that no
	interval covers gets `missing`, 0 unless given, and a base past the end of
	its chromosome is NaN. An interval whose value is NaN is treated as
	covering nothing, so its bases get `missing` too. These are the values
	pybigtools' `values(chrom, start, end, missing=missing)` gives, cast to
	float32. pyBigWig's `values()` gives NaN where no interval covers a base,
	which `missing=numpy.nan` matches.

	A writer leaves out a chromosome that has no data, so a sample without
	reads on chrY, say, has no chrY in its file. A window on a chromosome the
	file does not have is `missing` throughout, and a warning names the
	chromosomes, so that a naming mismatch (1 against chr1) does not go
	unnoticed.

	bigWigs with bedGraph, varStep and fixedStep sections are read, whether
	compressed or not. Anything else raises a ValueError rather than being
	guessed at:

		- a file that is not a little-endian bigWig, such as a bigBed or a
		  big-endian bigWig;
		- a chromosome tree or data index that is corrupt, or cut short by a
		  truncated file;
		- a data index whose entries are unsorted or span two chromosomes;
		- a window that starts before 0 or ends past 2**32 - 1;
		- a window overlapping a data block that cannot be decompressed, that
		  holds a section of another type, or whose intervals are unsorted,
		  overlap each other or those of a neighbouring block, or lie outside
		  the block's index entry. Overlapping intervals give a base two
		  values, and readers disagree on which to report: pybigtools sums
		  them.

	A `BigWig` holds no open file between reads, and one object can be read
	from several Python threads at once. It can be pickled, so it can be
	part of a PyTorch Dataset read by DataLoader workers.


	Parameters
	----------
	path: str or os.PathLike
		The path to a bigWig file.


	Attributes
	----------
	path: str
		The path the file was opened from.

	chroms: dict
		The length of each chromosome, keyed by its name, in the order of the
		file's chromosome tree.
	"""

	def __init__(self, path: str | os.PathLike):
		self.path = os.fspath(path)

		with open(self.path, 'rb') as handle:
			header = handle.read(64)
			if len(header) < 64:
				raise ValueError("{} is not a bigWig file.".format(self.path))

			magic = struct.unpack('<I', header[:4])[0]
			if magic == _BIGBED_MAGIC:
				raise ValueError("{} is a bigBed file; figwig reads bigWig "
					"files.".format(self.path))
			if struct.unpack('>I', header[:4])[0] in (_BIGWIG_MAGIC, _BIGBED_MAGIC):
				raise ValueError("{} is a big-endian file, which figwig does not "
					"read.".format(self.path))
			if magic != _BIGWIG_MAGIC:
				raise ValueError("{} is not a bigWig file.".format(self.path))
			if sys.byteorder != 'little':
				raise ValueError("figwig reads bigWig files only on little-endian "
					"machines.")

			(_, _, _, chrom_tree, _, data_index, _, _, _, _, self._buffer_size,
				_) = struct.unpack('<IHHQQQHHQQIQ', header)

			self._data_index = data_index
			self._chroms = self._read_chrom_tree(handle, chrom_tree)

		self.chroms = {name: length for name, (_, length) in self._chroms.items()}
		self._index = None
		self._index_lock = threading.Lock()

	def __repr__(self):
		return "BigWig('{}', {} chromosomes)".format(self.path, len(self.chroms))

	def __getstate__(self):
		# The lock cannot be pickled, and a copy needs its own. The index, if
		# it has been read, goes with the copy, so that a DataLoader worker
		# that receives one does not read it again.
		state = self.__dict__.copy()
		del state['_index_lock']
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		self._index_lock = threading.Lock()

	def _read_chrom_tree(self, handle, offset):
		"""The chromosome B+ tree, as {name: (chromosome id, length)}.

		Every node is checked against the length of the file before it is
		read, so that a corrupt count or key size cannot ask for more bytes
		than the file holds.
		"""

		error = "The chromosome tree of {} cannot be read".format(self.path)
		file_size = os.fstat(handle.fileno()).st_size
		if offset + 32 > file_size:
			raise ValueError(error + ".")

		handle.seek(offset)
		data = handle.read(32)

		magic, _, key_size, value_size, _, _ = struct.unpack('<IIIIQQ', data)
		if magic != _CHROM_TREE_MAGIC or value_size != 8:
			raise ValueError(error + ".")

		chroms, nodes, seen = {}, [offset + 32], set()
		while nodes:
			node = nodes.pop()
			if node in seen:
				raise ValueError(error + ": it has a cycle.")

			seen.add(node)
			if node + 4 > file_size:
				raise ValueError(error + ": the file ends inside it.")

			handle.seek(node)
			is_leaf, _, count = struct.unpack('<BBH', handle.read(4))
			if node + 4 + count * (key_size + 8) > file_size:
				raise ValueError(error + ": the file ends inside it.")

			data = handle.read(count * (key_size + 8))
			children = []
			for k in range(count):
				item = data[k * (key_size + 8): (k + 1) * (key_size + 8)]
				if is_leaf:
					try:
						name = item[:key_size].rstrip(b'\0').decode()
					except UnicodeDecodeError:
						raise ValueError(error + ": a chromosome name is not "
							"UTF-8.") from None

					chroms[name] = struct.unpack('<II', item[key_size:])
				else:
					children.append(struct.unpack('<Q', item[key_size:])[0])

			# Children go on the stack last first, so that the leaves, and the
			# chromosomes in them, are read in the tree's order.
			nodes.extend(children[::-1])

		return chroms

	def _read_index(self):
		"""The data blocks in file order, from the R-tree.

		Each entry of the R-tree's leaves gives a block's range, from
		(chromosome, start) to (chromosome, end), and its offset and size in
		the file. Each entry must be on one chromosome, and both the starts
		and the ends of the entries must be sorted, so that the blocks
		overlapping a window can be found by binary search.

		Neighbouring entries may overlap. pyBigWig writes some fixedStep
		blocks whose entry ends a few bases past the block's last interval,
		into the next block's range, though no two intervals overlap.
		`_read_windows` checks the intervals themselves.
		"""

		error = "The data index of {} cannot be read".format(self.path)
		try:
			with open(self.path, 'rb') as handle:
				fd = handle.fileno()
				file_size = os.fstat(fd).st_size
				magic = struct.unpack('<I', _pread(fd, 4, self._data_index))[0]
				if magic != _INDEX_MAGIC:
					raise ValueError(error + ".")

				leaves, nodes, seen = [], [self._data_index + 48], set()
				while nodes:
					offset = nodes.pop()
					if offset in seen:
						raise ValueError(error + ": it has a cycle.")

					seen.add(offset)
					is_leaf, _, count = struct.unpack('<BBH', _pread(fd, 4,
						offset))
					size = 32 if is_leaf else 24
					data = _pread(fd, count * size, offset + 4)
					if len(data) != count * size:
						raise ValueError(error + ": the file ends inside it.")

					if is_leaf:
						leaves.append(numpy.frombuffer(data, dtype=_LEAF_TYPE))
					else:
						children = numpy.frombuffer(data, dtype='<u8').reshape(
							count, 3)[:, 2]
						nodes.extend(children[::-1].tolist())
		except (OSError, struct.error, OverflowError) as e:
			raise ValueError(error + ": {}".format(e)) from e

		leaves = numpy.concatenate(leaves) if leaves else numpy.empty(0,
			dtype=_LEAF_TYPE)

		# Checked before the uint64 offsets and sizes become int64, where 2**63
		# and more would turn negative.
		size = numpy.uint64(file_size)
		if ((leaves['size'] > size) | (leaves['offset'] > size -
				numpy.minimum(leaves['size'], size))).any():
			raise ValueError(error + ": a data block lies past the end of the "
				"file, which may be truncated.")

		chroms = leaves['start_chrom'].astype(numpy.int64)
		starts = (chroms << 32) | leaves['start'].astype(numpy.int64)
		ends = (leaves['end_chrom'].astype(numpy.int64) << 32) | \
			leaves['end'].astype(numpy.int64)

		if (leaves['end_chrom'] != leaves['start_chrom']).any():
			raise ValueError(error + ": a data block spans two chromosomes.")
		if (ends < starts).any() or (starts[1:] < starts[:-1]).any() or \
				(ends[1:] < ends[:-1]).any():
			raise ValueError(error + ": its data blocks are unsorted.")

		return {'starts': starts, 'ends': ends, 'chroms': chroms,
			'bases': numpy.stack([leaves['start'], leaves['end']], axis=1).astype(
				numpy.int64),
			'offsets': leaves['offset'].astype(numpy.int64),
			'sizes': leaves['size'].astype(numpy.int64)}

	def _get_index(self):
		with self._index_lock:
			if self._index is None:
				self._index = self._read_index()

		return self._index

	def read(self, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
		numpy.ndarray, width: int, out: numpy.ndarray | None = None,
		n_jobs: int = 8, missing: float = 0.0) -> numpy.ndarray:
		"""Read the per-base values of many windows of one width.

		Window j covers [starts[j], starts[j] + width) on chroms[j], and its
		values are written into row j of the output, whatever order the
		windows are given in. The windows may overlap or repeat.

		The result does not depend on `n_jobs`. Passing `out` reuses an array
		from one read to the next, which a data loader can do to avoid
		allocating a new one for every batch.


		Parameters
		----------
		chroms: str, list of str, or numpy.ndarray of str
			The chromosome of each window, or one name for every window. A
			window on a chromosome that is not one of `self.chroms` is
			`missing` throughout, with a warning.

		starts: list of int or numpy.ndarray of int, shape=(n,)
			The start of each window, inclusive and base-0. Every window must
			start at 0 or later, and its end, start + width, must be at most
			2**32 - 1. A window may run past the end of its chromosome, and
			the bases past it are NaN.

		width: int
			The length of every window, in bases. Must be at least 1.

		out: numpy.ndarray, shape=(n, width), dtype=float32, or None, optional
			A writeable, C-contiguous array to write the values into, which is
			returned. If None, a new array is returned. Default is None.

		n_jobs: int, optional
			The largest number of threads to decompress and decode data blocks
			on, or -1 for as many as there are CPUs that this process may run
			on. Default is 8.

		missing: float, optional
			The value of a base that no interval covers, or that an interval
			whose value is NaN covers, and of every base of a window on a
			chromosome the file does not have. It is cast to float32. Pass
			numpy.nan to tell these bases apart from a measured 0, as
			pyBigWig's `values()` does. Default is 0.0.


		Returns
		-------
		out: numpy.ndarray, shape=(n, width), dtype=float32
			The value of every base of every window: the value of the interval
			covering it, `missing` where no interval does, and NaN past the
			end of the chromosome.
		"""

		return _read([self], True, chroms, starts, width, out, n_jobs, missing)

	def _warn_absent(self, windows, missing, stacklevel):
		"""Warn about windows on chromosomes the file does not have.

		A writer leaves out a chromosome that has no data, so a sample without
		reads on chrY, say, has no chrY. A misspelt or differently named
		chromosome (1 against chr1) is absent too, so the warning lists some
		of the file's own names. `stacklevel` counts from the caller of this
		method.
		"""

		absent = [k for k, name in enumerate(windows.names) if name not in
			self._chroms]
		if len(absent) == 0:
			return

		n = int(numpy.isin(windows.codes, absent).sum())
		# Names are quoted, so that an empty name or one with stray spaces can
		# be told from the name it resembles.
		shown = ", ".join(repr(windows.names[k]) for k in absent[:10])
		if len(absent) > 10:
			shown += ", and {} more".format(len(absent) - 10)

		warnings.warn("{} windows are on chromosomes not in {}, and are {} "
			"throughout: {}. Its chromosomes include {}.".format(n, self.path,
			float(missing), shown, ", ".join(repr(name) for name in
			list(self.chroms)[:3])),
			stacklevel=stacklevel + 1)

	def _read_blocks(self, fd, leaves, index, uncompress=None):
		"""Read and decompress data blocks into one array of 32-bit words.

		Returns the words and a (len(leaves), 6) array of each block's first
		and last word, its index entry's chromosome, start and end, and 1 if
		it was read or 0 if it could not be.

		The compressed blocks are read with one positional read per run of
		adjacent blocks, into one array. `uncompress`, the result of
		`_zlib_uncompress()`, inflates them without the GIL. The blocks it
		leaves, and every block when it is None or the file is not
		compressed, are inflated with zlib.decompress, with the same result.
		"""

		offsets, sizes = index['offsets'][leaves], index['sizes'][leaves]
		breaks = numpy.flatnonzero(offsets[1:] != offsets[:-1] + sizes[:-1]) + 1
		breaks = [0] + breaks.tolist() + [len(leaves)] if len(leaves) > 0 else []

		# Blocks in a run are adjacent in the file, so each run is read to
		# the positions its blocks have when they are packed end to end. A
		# block that a short read does not reach in full is not read.
		ends = numpy.cumsum(sizes)
		starts = ends - sizes
		data = numpy.empty(int(sizes.sum()), dtype=numpy.uint8)
		read = numpy.ones(len(leaves), dtype=numpy.int64)
		for r0, r1 in pairwise(breaks):
			position, end = int(starts[r0]), int(ends[r1 - 1])
			got = _pread_into(fd, data[position:end], int(offsets[r0]))
			read[r0:r1][ends[r0:r1] > position + got] = 0

		zeros = numpy.zeros(len(leaves), dtype=numpy.int64)
		blocks = numpy.stack([zeros, zeros, index['chroms'][leaves],
			index['bases'][leaves, 0], index['bases'][leaves, 1], read], axis=1)

		used, buffer = 0, numpy.empty(0, dtype=numpy.uint8)
		if uncompress is not None and self._buffer_size > 0:
			function, ulong = uncompress
			buffer = numpy.empty(len(leaves) * min(self._buffer_size,
				_MAX_BLOCK_BYTES), dtype=numpy.uint8)
			used = _inflate_blocks(function, data, starts, sizes, buffer, blocks,
				numpy.zeros(1, dtype=ulong))
		else:
			blocks[:, 5] *= 2

		parts, position = [], used
		for k in numpy.flatnonzero(blocks[:, 5] == 2).tolist():
			raw = data[starts[k]:ends[k]]
			try:
				block = zlib.decompress(raw) if self._buffer_size > 0 else \
					raw.tobytes()
			except zlib.error:
				block = b''

			if len(block) % 4 != 0 or len(block) == 0:
				blocks[k, 5] = 0
				continue

			blocks[k, 0], blocks[k, 1], blocks[k, 5] = position // 4, (position +
				len(block)) // 4, 1
			position += len(block)
			parts.append(block)

		words = buffer[:used]
		if len(parts) > 0:
			words = numpy.concatenate([words, numpy.frombuffer(b''.join(parts),
				dtype=numpy.uint8)])

		return words.view(numpy.uint32), blocks


def _cpu_count():
	"""The number of CPUs this process may run on."""

	if hasattr(os, 'sched_getaffinity'):
		return len(os.sched_getaffinity(0))

	return os.cpu_count() or 1


def _check_n_jobs(n_jobs):
	"""The `n_jobs` argument as a number of threads."""

	if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, numpy.integer)):
		raise TypeError("n_jobs must be an integer.")
	if n_jobs == -1:
		return _cpu_count()
	if n_jobs < 1:
		raise ValueError("n_jobs must be at least 1, or -1 for every CPU.")

	return int(n_jobs)


def _check_missing(missing):
	"""The `missing` argument as a float32."""

	if isinstance(missing, bool) or not isinstance(missing, (int, float,
			numpy.integer, numpy.floating)):
		raise TypeError("missing must be a number, not {}.".format(
			type(missing).__name__))

	return numpy.float32(missing)


class _Windows:
	"""The windows of a read, with their arguments checked.

	`names` are the chromosome names the windows are on, `codes` each
	window's position in `names`, and `starts` each window's start as an
	int64. Windows that start before 0 or end past 2**32 - 1 are found here
	and raised on by `check_range`, which comes after the check that every
	chromosome is in the file.
	"""

	def __init__(self, chroms, starts, width):
		if isinstance(width, bool) or not isinstance(width, (int, numpy.integer)):
			raise TypeError("width must be an integer.")
		if width < 1:
			raise ValueError("width must be at least 1.")

		starts = numpy.asarray(starts)
		if starts.ndim != 1:
			raise ValueError("starts must be one-dimensional.")
		if starts.dtype.kind not in 'iu' and len(starts) > 0:
			raise TypeError("starts must be integers, not {}.".format(starts.dtype))

		# Windows must end at or before 2**32 - 1. This is checked on the
		# starts as given, comparing each with 2**32 - 1 - width rather than
		# adding width to it, so that neither a uint64 of 2**63 or more nor
		# an int64 close to it overflows into a value that passes.
		n, width, limit = len(starts), int(width), 2**32 - 1 - int(width)
		self._bad = numpy.flatnonzero((starts < 0) | (starts > limit)) if \
			limit >= 0 else numpy.arange(n)
		self._given = starts

		if isinstance(chroms, str):
			names, codes = [chroms], numpy.zeros(n, dtype=numpy.int64)
		else:
			chroms = numpy.asarray(chroms, dtype=str)
			if chroms.shape != (n,):
				raise ValueError("chroms must have one name per start, or be a "
					"single name.")

			names, codes = numpy.unique(chroms, return_inverse=True)
			names = names.tolist()

		self.n, self.width = n, width
		self.names, self.codes = names, codes
		self.starts = starts.astype(numpy.int64)

	def describe(self, j):
		"""Window j as chrom:start-end."""

		start = int(self._given[j])
		return "{}:{}-{}".format(self.names[self.codes[j]], start, start +
			self.width)

	def check_range(self):
		if len(self._bad) > 0:
			raise ValueError("{} windows start before 0 or end past 2**32 - 1, "
				"such as {}.".format(len(self._bad), self.describe(self._bad[0])))


class _Read:
	"""One file's share of a read: its windows, sorted into batches.

	The windows are sorted by position and matched to the data blocks of the
	file that they overlap, and split into batches by their first block.
	`run(k)` reads batch k into channel `signal` of the windows' rows of
	`out`, of shape (n, channels, width), from the file opened by `open`.
	`finish` raises if any window could not be read.

	Windows on a chromosome the file does not have are filled with `missing`
	here and left out of the batches. Without the chromosome's length there
	is no end past which their bases would be NaN.
	"""

	def __init__(self, bigwig, windows, out, signal, missing):
		self.bigwig, self.windows = bigwig, windows
		self.out, self.signal, self.missing = out, signal, missing

		index = bigwig._get_index()
		names, codes, starts = windows.names, windows.codes, windows.starts
		chroms = [bigwig._chroms.get(name, (-1, 0)) for name in names]
		ids = numpy.array([chrom[0] for chrom in chroms], dtype=numpy.int64)[codes]
		sizes = numpy.array([chrom[1] for chrom in chroms],
			dtype=numpy.int64)[codes]

		present = numpy.flatnonzero(ids >= 0)
		out[ids < 0, signal] = missing
		order = present[numpy.lexsort((starts[present], ids[present]))]
		n = len(order)

		# The blocks [lo, hi) overlap a window: those that end after it starts
		# and start before it ends. The starts and the ends of the index
		# entries are each sorted, so both bounds are binary searches.
		keys = (ids[order] << 32) | starts[order]
		lo = numpy.searchsorted(index['ends'], keys, side='right')
		hi = numpy.searchsorted(index['starts'], keys + windows.width,
			side='left')
		hi = numpy.maximum(lo, hi)

		# The blocks any window needs, and each window's blocks as positions
		# in that list. Windows are split into batches by their first block.
		n_blocks = len(index['starts'])
		cover = numpy.cumsum(numpy.bincount(lo, minlength=n_blocks + 1) -
			numpy.bincount(hi, minlength=n_blocks + 1))
		needed = numpy.flatnonzero(cover[:n_blocks] > 0)
		lo, hi = numpy.searchsorted(needed, lo), numpy.searchsorted(needed, hi)

		batch = lo // _BATCH_BLOCKS
		self.index, self.needed, self.order = index, needed, order
		self.rows = numpy.stack([lo, hi, starts[order], sizes[order], order],
			axis=1)
		self.bounds = numpy.append(numpy.flatnonzero(numpy.diff(batch,
			prepend=-1)), n).tolist()
		self.n_batches = len(self.bounds) - 1
		self.failed = numpy.zeros(n, dtype=numpy.bool_)
		self.uncompress = _zlib_uncompress() if bigwig._buffer_size > 0 else \
			None
		self.fd = None

	def open(self):
		# Every batch shares one fd, read without moving its position where
		# there is pread. Windows opens a file in text mode unless asked for
		# binary, and has O_BINARY to ask; elsewhere it does not exist.
		self.fd = os.open(self.bigwig.path, os.O_RDONLY | getattr(os,
			'O_BINARY', 0))

	def close(self):
		if self.fd is not None:
			os.close(self.fd)
			self.fd = None

	def run(self, k):
		w0, w1 = self.bounds[k], self.bounds[k + 1]
		b0, b1 = int(self.rows[w0, 0]), int(self.rows[w0:w1, 1].max())
		words, blocks = self.bigwig._read_blocks(self.fd, self.needed[b0:b1],
			self.index, self.uncompress)
		local = self.rows[w0:w1].copy()
		local[:, :2] -= b0
		_read_windows(words, blocks, local, self.out, self.signal,
			self.failed[w0:w1], self.missing)

	def finish(self):
		if self.failed.any():
			j = self.order[numpy.flatnonzero(self.failed)[0]]
			raise ValueError("{} windows overlap data blocks that figwig cannot "
				"read in {}: a block that cannot be decompressed, holds a section "
				"that is not bedGraph, varStep or fixedStep, or holds intervals "
				"that are unsorted, overlap, or lie outside its index entry. The "
				"first is {}.".format(int(self.failed.sum()), self.bigwig.path,
				self.windows.describe(j)))


def _read(bigwigs, single, chroms, starts, width, out, n_jobs, missing):
	"""Read windows from one or more open bigWigs into one array.

	With `single`, `bigwigs` holds one BigWig and the output has shape
	(n, width); otherwise it has shape (n, len(bigwigs), width), and channel
	i holds the values from bigwigs[i]. Every file's batches run on one
	pool of threads. This is called by `BigWig.read` and `read_windows`, and
	the warnings it raises name the line that called them.
	"""

	n_jobs = _check_n_jobs(n_jobs)
	missing = _check_missing(missing)
	windows = _Windows(chroms, starts, width)
	n, width = windows.n, windows.width

	shape = (n, width) if single else (n, len(bigwigs), width)
	if out is None:
		out = numpy.empty(shape, dtype=numpy.float32)
	elif not isinstance(out, numpy.ndarray):
		raise TypeError("out must be a numpy array, not {}.".format(
			type(out).__name__))
	elif out.shape != shape or out.dtype != numpy.float32 or \
			not out.flags['C_CONTIGUOUS'] or not out.flags['WRITEABLE']:
		raise ValueError("out must be a writeable, C-contiguous float32 array "
			"of shape {}.".format(shape))

	if n == 0:
		return out

	windows.check_range()
	for bigwig in bigwigs:
		bigwig._warn_absent(windows, missing, stacklevel=3)

	out3 = out.reshape(n, 1, width) if single else out
	_run([_Read(bigwig, windows, out3, i, missing) for i, bigwig in
		enumerate(bigwigs)], n_jobs)
	return out


def _run(reads, n_jobs):
	"""Run every batch of every read, on up to `n_jobs` threads.

	A batch is read on this thread first when the decoder, or the inflater
	that a compressed file needs, has not been compiled yet, so that each is
	compiled once and before any thread starts. With NUMBA_DISABLE_JIT=1 the
	kernels are plain functions, which have no signatures and need no
	compiling.
	"""

	tasks = [(read, k) for read in reads for k in range(read.n_batches)]
	first = []
	if tasks and not getattr(_read_windows, 'signatures', True):
		first.append(tasks[0])

	compressed = [task for task in tasks if task[0].uncompress is not None]
	if compressed and not getattr(_inflate_blocks, 'signatures', True) and \
			compressed[0] not in first:
		first.append(compressed[0])

	rest = [task for task in tasks if task not in first]
	try:
		for read in reads:
			read.open()

		for read, k in first:
			read.run(k)

		if n_jobs == 1 or len(rest) <= 1:
			for read, k in rest:
				read.run(k)
		else:
			with ThreadPoolExecutor(min(n_jobs, len(rest))) as pool:
				list(pool.map(lambda task: task[0].run(task[1]), rest))
	finally:
		for read in reads:
			read.close()

	for read in reads:
		read.finish()


def read_windows(bigwigs: str | os.PathLike | BigWig | list | tuple,
	chroms: str | list[str] | numpy.ndarray, starts: list[int] | numpy.ndarray,
	width: int, out: numpy.ndarray | None = None, n_jobs: int = 8,
	missing: float = 0.0) -> numpy.ndarray:
	"""Read the per-base values of many windows of one width from bigWigs.

	Given one bigWig, this reads the windows as `BigWig.read` does, into an
	array of shape (n, width). Given a list of them, such as the plus and
	minus strands of a stranded assay or one track per task of a model, it
	reads every file into one array of shape (n, len(bigwigs), width), in
	which channel i holds the values from bigwigs[i]. That is the
	(batch, channels, length) layout that sequence models take, and reading
	into it directly costs neither a stack of one array per file nor the
	memory for both. The batches of every file run on one pool of up to
	`n_jobs` threads.

	A path is opened, and its data index read, on every call. To read the
	same files more than once, as a data loader does, open them with
	`BigWig` and pass those, which keep their indexes.

	Each file is read as `BigWig.read` describes: a window on a chromosome
	that one file does not have is `missing` throughout in that file's
	channel, with a warning, and the files need not share chromosomes.


	Parameters
	----------
	bigwigs: str, os.PathLike, BigWig, or list or tuple of these
		One bigWig, or several, as paths or as open `BigWig` objects.

	chroms: str, list of str, or numpy.ndarray of str
		The chromosome of each window, or one name for every window.

	starts: list of int or numpy.ndarray of int, shape=(n,)
		The start of each window, inclusive and base-0.

	width: int
		The length of every window, in bases.

	out: numpy.ndarray, dtype=float32, or None, optional
		A writeable, C-contiguous array to write the values into, of shape
		(n, width) for one bigWig and (n, len(bigwigs), width) for a list.
		Default is None.

	n_jobs: int, optional
		The largest number of threads to use, or -1 for one per CPU. Default
		is 8.

	missing: float, optional
		The value of a base that no interval covers, and of every base of a
		window on a chromosome a file does not have. Default is 0.0.


	Returns
	-------
	out: numpy.ndarray, dtype=float32, shape=(n, width) or (n, len(bigwigs), width)
		The value of every base of every window, as `BigWig.read` gives it.
	"""

	single = not isinstance(bigwigs, (list, tuple))
	files = [bigwigs] if single else list(bigwigs)
	if len(files) == 0:
		raise ValueError("bigwigs must hold at least one bigWig.")

	for bigwig in files:
		if not isinstance(bigwig, (str, os.PathLike, BigWig)):
			raise TypeError("bigwigs must be a path, a BigWig, or a list or tuple "
				"of them, not {}.".format(type(bigwig).__name__))

	files = [bigwig if isinstance(bigwig, BigWig) else BigWig(bigwig) for
		bigwig in files]
	return _read(files, single, chroms, starts, width, out, n_jobs, missing)
