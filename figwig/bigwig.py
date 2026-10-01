# bigwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from __future__ import annotations

import os
import sys
import zlib
import struct
import threading

from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy

from ._kernels import _inflate_blocks
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
	windows' rows of the output. A block that several windows share, because
	they overlap or repeat, is decompressed once. Batches run on up to
	`n_jobs` threads. zlib's own `uncompress()` and the numba decoder both run
	without the GIL, so the threads run in parallel.

	Each base gets the value of the interval that covers it. A base that no
	interval covers is 0, and a base past the end of its chromosome is NaN.
	An interval whose value is NaN is treated as covering nothing, so its
	bases are 0. These are the values pybigtools' `values(chrom, start, end)`
	gives with its defaults, cast to float32.

	bigWigs with bedGraph, varStep and fixedStep sections are read, whether
	compressed or not. Anything else raises a ValueError rather than being
	guessed at:

		- a file that is not a little-endian bigWig, such as a bigBed or a
		  big-endian bigWig;
		- a data index whose entries are unsorted, overlap, or span two
		  chromosomes;
		- a window on a chromosome the file does not have, or one that starts
		  before 0 or ends past 2**32 - 1;
		- a window overlapping a data block that cannot be decompressed, that
		  holds a section of another type, or whose intervals are unsorted,
		  overlap each other, or lie outside the block's index entry.
		  Overlapping intervals give a base two values, and readers disagree
		  on which to report: pybigtools sums them.

	A `BigWig` holds no open file between reads, and one object can be read
	from several Python threads at once.


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

	@staticmethod
	def _read_chrom_tree(handle, offset):
		"""The chromosome B+ tree, as {name: (chromosome id, length)}."""

		handle.seek(offset)
		data = handle.read(32)
		if len(data) < 32:
			raise ValueError("The chromosome tree cannot be read.")

		magic, _, key_size, value_size, _, _ = struct.unpack('<IIIIQQ', data)
		if magic != _CHROM_TREE_MAGIC or value_size != 8:
			raise ValueError("The chromosome tree cannot be read.")

		chroms, nodes, seen = {}, [offset + 32], set()
		while nodes:
			node = nodes.pop()
			if node in seen:
				raise ValueError("The chromosome tree has a cycle.")

			seen.add(node)
			handle.seek(node)
			is_leaf, _, count = struct.unpack('<BBH', handle.read(4))
			data = handle.read(count * (key_size + 8))
			children = []
			for k in range(count):
				item = data[k * (key_size + 8): (k + 1) * (key_size + 8)]
				if is_leaf:
					name = item[:key_size].rstrip(b'\0').decode()
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
		the file. The entries must be sorted, non-overlapping and each on one
		chromosome, so that the blocks overlapping a window can be found by
		binary search.
		"""

		error = "The data index of {} cannot be read".format(self.path)
		try:
			with open(self.path, 'rb') as handle:
				fd = handle.fileno()
				magic = struct.unpack('<I', os.pread(fd, 4, self._data_index))[0]
				if magic != _INDEX_MAGIC:
					raise ValueError(error + ".")

				leaves, nodes, seen = [], [self._data_index + 48], set()
				while nodes:
					offset = nodes.pop()
					if offset in seen:
						raise ValueError(error + ": it has a cycle.")

					seen.add(offset)
					is_leaf, _, count = struct.unpack('<BBH', os.pread(fd, 4,
						offset))
					size = 32 if is_leaf else 24
					data = os.pread(fd, count * size, offset + 4)
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
		chroms = leaves['start_chrom'].astype(numpy.int64)
		starts = (chroms << 32) | leaves['start'].astype(numpy.int64)
		ends = (leaves['end_chrom'].astype(numpy.int64) << 32) | \
			leaves['end'].astype(numpy.int64)

		if (leaves['end_chrom'] != leaves['start_chrom']).any():
			raise ValueError(error + ": a data block spans two chromosomes.")
		if (ends < starts).any() or (starts[1:] < ends[:-1]).any():
			raise ValueError(error + ": its data blocks are unsorted or overlap.")

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
		n_jobs: int = 8) -> numpy.ndarray:
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
			The chromosome of each window, or one name for every window.
			Every name must be one of `self.chroms`.

		starts: list of int or numpy.ndarray of int, shape=(n,)
			The start of each window, inclusive and base-0. Every window must
			start at 0 or later and end before 2**32 - 1. A window may run
			past the end of its chromosome, and the bases past it are NaN.

		width: int
			The length of every window, in bases. Must be at least 1.

		out: numpy.ndarray, shape=(n, width), dtype=float32, or None, optional
			A C-contiguous array to write the values into, which is returned.
			If None, a new array is returned. Default is None.

		n_jobs: int, optional
			The largest number of threads to decompress and decode data blocks
			on. Must be at least 1. Default is 8.


		Returns
		-------
		out: numpy.ndarray, shape=(n, width), dtype=float32
			The value of every base of every window: the value of the interval
			covering it, 0 where no interval does, and NaN past the end of the
			chromosome.
		"""

		if isinstance(width, bool) or not isinstance(width, (int, numpy.integer)):
			raise TypeError("width must be an integer.")
		if width < 1:
			raise ValueError("width must be at least 1.")
		if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, numpy.integer)):
			raise TypeError("n_jobs must be an integer.")
		if n_jobs < 1:
			raise ValueError("n_jobs must be at least 1.")

		width, n_jobs = int(width), int(n_jobs)
		starts = numpy.asarray(starts)
		if starts.ndim != 1:
			raise ValueError("starts must be one-dimensional.")
		if starts.dtype.kind not in 'iu':
			raise TypeError("starts must be integers, not {}.".format(starts.dtype))

		n = len(starts)
		starts = starts.astype(numpy.int64)

		if isinstance(chroms, str):
			names, codes = [chroms], numpy.zeros(n, dtype=numpy.int64)
		else:
			chroms = numpy.asarray(chroms, dtype=str)
			if chroms.shape != (n,):
				raise ValueError("chroms must have one name per start, or be a "
					"single name.")

			names, codes = numpy.unique(chroms, return_inverse=True)
			names = names.tolist()

		if out is None:
			out = numpy.empty((n, width), dtype=numpy.float32)
		elif out.shape != (n, width) or out.dtype != numpy.float32 or \
				not out.flags['C_CONTIGUOUS']:
			raise ValueError("out must be a C-contiguous float32 array of shape "
				"({}, {}).".format(n, width))

		if n == 0:
			return out

		missing = [name for name in names if name not in self._chroms]
		if missing:
			raise ValueError("Chromosomes not in {}: {}.".format(self.path,
				", ".join(sorted(missing))))

		bad = numpy.flatnonzero((starts < 0) | (starts + width >= 2**32))
		if len(bad) > 0:
			j = bad[0]
			raise ValueError("{} windows start before 0 or end past 2**32 - 1, "
				"such as {}:{}-{}.".format(len(bad), names[codes[j]], starts[j],
				starts[j] + width))

		index = self._get_index()
		ids = numpy.array([self._chroms[name][0] for name in names],
			dtype=numpy.int64)[codes]
		sizes = numpy.array([self._chroms[name][1] for name in names],
			dtype=numpy.int64)[codes]

		order = numpy.lexsort((starts, ids))

		# The blocks [lo, hi) overlap a window. The index entries are sorted
		# and do not overlap, so both their starts and their ends are sorted.
		keys = (ids[order] << 32) | starts[order]
		lo = numpy.searchsorted(index['ends'], keys, side='right')
		hi = numpy.searchsorted(index['starts'], keys + width, side='left')
		hi = numpy.maximum(lo, hi)

		# The blocks any window needs, and each window's blocks as positions
		# in that list. Windows are split into batches by their first block.
		n_blocks = len(index['starts'])
		cover = numpy.cumsum(numpy.bincount(lo, minlength=n_blocks + 1) -
			numpy.bincount(hi, minlength=n_blocks + 1))
		needed = numpy.flatnonzero(cover[:n_blocks] > 0)
		lo, hi = numpy.searchsorted(needed, lo), numpy.searchsorted(needed, hi)

		windows = numpy.stack([lo, hi, starts[order], sizes[order], order], axis=1)
		failed = numpy.zeros(n, dtype=numpy.bool_)
		out3 = out.reshape(n, 1, width)

		batch = lo // _BATCH_BLOCKS
		bounds = numpy.append(numpy.flatnonzero(numpy.diff(batch, prepend=-1)),
			n).tolist()

		def read_batch(k):
			w0, w1 = bounds[k], bounds[k + 1]
			b0, b1 = int(windows[w0, 0]), int(windows[w0:w1, 1].max())
			words, blocks = self._read_blocks(fd, needed[b0:b1], index,
				uncompress)
			local = windows[w0:w1].copy()
			local[:, :2] -= b0
			_read_windows(words, blocks, local, out3, 0, failed[w0:w1])

		# The first batch is read on this thread when the decoder or the
		# inflater has not been compiled yet, so that each is compiled once and
		# before any thread starts. pread takes no file position, so the
		# threads share one fd.
		uncompress = _zlib_uncompress() if self._buffer_size > 0 else None
		n_batches = len(bounds) - 1
		first = int(n_batches > 0 and not (_read_windows.signatures and
			(uncompress is None or _inflate_blocks.signatures)))
		fd = os.open(self.path, os.O_RDONLY)
		try:
			if first:
				read_batch(0)

			if n_jobs == 1 or n_batches - first <= 1:
				for k in range(first, n_batches):
					read_batch(k)
			else:
				with ThreadPoolExecutor(min(n_jobs, n_batches - first)) as pool:
					list(pool.map(read_batch, range(first, n_batches)))
		finally:
			os.close(fd)

		if failed.any():
			j = order[numpy.flatnonzero(failed)[0]]
			raise ValueError("{} windows overlap data blocks that figwig cannot "
				"read: a block that cannot be decompressed, holds a section that "
				"is not bedGraph, varStep or fixedStep, or holds intervals that "
				"are unsorted, overlap, or lie outside its index entry. The first "
				"is {}:{}-{}.".format(int(failed.sum()), names[codes[j]],
				starts[j], starts[j] + width))

		return out

	def _read_blocks(self, fd, leaves, index, uncompress=None):
		"""Read and decompress data blocks into one array of 32-bit words.

		Returns the words and a (len(leaves), 6) array of each block's first
		and last word, its index entry's chromosome, start and end, and 1 if
		it was read or 0 if it could not be.

		The compressed blocks are read with one pread per run of adjacent
		blocks, into one array. `uncompress`, the result of
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


def read_windows(path: str | os.PathLike, chroms: str | list[str] |
	numpy.ndarray, starts: list[int] | numpy.ndarray, width: int,
	out: numpy.ndarray | None = None, n_jobs: int = 8) -> numpy.ndarray:
	"""Read the per-base values of many windows of one width from a bigWig.

	This opens the file and reads its data index on every call. To read the
	same file more than once, open it with `BigWig` and call `read`, which
	keeps the index.


	Parameters
	----------
	path: str or os.PathLike
		The path to a bigWig file.

	chroms: str, list of str, or numpy.ndarray of str
		The chromosome of each window, or one name for every window.

	starts: list of int or numpy.ndarray of int, shape=(n,)
		The start of each window, inclusive and base-0.

	width: int
		The length of every window, in bases.

	out: numpy.ndarray, shape=(n, width), dtype=float32, or None, optional
		A C-contiguous array to write the values into. Default is None.

	n_jobs: int, optional
		The largest number of threads to use. Default is 8.


	Returns
	-------
	out: numpy.ndarray, shape=(n, width), dtype=float32
		The value of every base of every window, as `BigWig.read` gives it.
	"""

	return BigWig(path).read(chroms, starts, width, out=out, n_jobs=n_jobs)
