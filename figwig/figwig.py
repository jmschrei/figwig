# figwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""figwig files: many per-base tracks in one file, read in windows.

A figwig file holds the per-base values of any number of tracks over one
genome, with a JSON record of attributes for each track and for the file,
and is read the way `BigWigReader` reads a bigWig: many windows of one
width at a time, into one array of shape (windows, tracks, width), or as a
sparse list of the nonzero bases, or summed per label of each track.

A file is written in one of two orders. In position order, every track is
cut on one grid of tiles of `tile` bases, so a window's tiles are found by
division rather than by a tree search; the tiles of a group of tracks at
one position are one chunk, so a window read decodes each chunk once for
every track in it; a chunk whose values are all zero takes no space; and
the other chunks hold their values densely, as nonzero bases per track, or
as a sparse matrix of the tracks present, in the narrowest integer that
holds them. That is the order for reading windows across many tracks. In
track order, each track's nonzero bases lie in blocks per chromosome, so
one track's values over the genome are a few slices, and a window read
searches each requested track's blocks. That is the order for reading a
few of many tracks, such as cells of a single-cell experiment.

The file is, in order: a 128-byte header; the chunks or blocks, as they
were written; the chromosome table; a table of each track's sum, nonzero
bases, minimum and maximum; the metadata, a JSON document holding the
genome, the file's attributes and every track's record; and the index, one
record per group of tracks holding, in position order, the kind byte and
the compressed and uncompressed sizes of every tile and the runs of tiles
that lie back to back in the file, and in track order the blocks of each
track. `_figwig_kernels` describes the chunks and blocks.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import zlib

from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy

from ._figwig_kernels import _BLOCK_ENTRIES
from ._figwig_kernels import _check_blocks
from ._figwig_kernels import _check_chunks
from ._figwig_kernels import _encode_blocks
from ._figwig_kernels import _encode_dense_tiles
from ._figwig_kernels import _encode_entries
from ._figwig_kernels import _encode_tiles
from ._figwig_kernels import _inflate_chunks
from ._figwig_kernels import _inflate_chunks_libdeflate
from ._figwig_kernels import _list_tiles
from ._figwig_kernels import _read_blocks
from ._figwig_kernels import _read_tiles
from ._kernels import _libdeflate_inflate
from ._kernels import _pread
from ._kernels import _pread_into
from ._kernels import _zlib_uncompress
from .bigwig import _check_n_jobs
from .bigwig import _chrom_codes
from .bigwig import _Windows
from .writer import _check_chrom_sizes
from .writer import _check_engine
from .writer import _compress_blocks


_FIGWIG_MAGIC = int.from_bytes(b'FIGW', 'little')
_VERSION = 1
_CODEC_NONE, _CODEC_ZLIB = 0, 1
_FLAG_TRACKS = 1
_ORDERS = {'positions': 'positions', 'csc': 'positions', 'tracks': 'tracks',
	'csr': 'tracks'}

# Tiles are a power of two of bases from 16 to 65536, so that a sparse
# chunk's offsets fit uint16; a group's chunk must fit a uint32 size.
_MIN_TILE, _MAX_TILE = 16, 65536
_DEFAULT_TILE = 16384
_DEFAULT_GROUPS = 8
_MAX_GROUP_BASES = 1 << 29

# The most bytes of chunks before compression that one task encodes or
# decodes, the most and the fewest chunks one read task takes, how many
# tasks each thread of a read should get, and the most bytes a dense read
# allocates before asking for a sparse or summed one instead.
_BATCH_BYTES = 1 << 25
_BATCH_CHUNKS = 256
_MIN_BATCH_CHUNKS = 16
_BATCHES_PER_THREAD = 8
_MAX_DENSE_BYTES = 1 << 33

# Whether a tile whose few tracks have values may be stored as a matrix
# chunk in an uncompressed file. Compression shrinks a sparse chunk's zero
# counts to almost nothing, where a matrix chunk's track ids stay, so a
# compressed file is laid out without matrix chunks either way.
_MATRIX_CHUNKS = True

_DTYPES = ('uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'float32')
_RESERVED = ('name', 'dtype', 'group')

_HEADER = numpy.dtype([('magic', '<u4'), ('version', '<u2'), ('flags', '<u2'),
	('tile', '<u4'), ('codec', '<u2'), ('level', '<u2'), ('n_chroms', '<u4'),
	('n_tracks', '<u4'), ('n_groups', '<u4'), ('reserved', '<u4'),
	('chroms', '<u8'), ('tracks', '<u8'), ('index', '<u8'), ('data', '<u8'),
	('metadata', '<u8'), ('metadata_length', '<u8'), ('zooms', '<u8'),
	('length', '<u8'), ('padding', 'V32')])
_STATS = numpy.dtype([('sum', '<f8'), ('nonzero', '<u8'), ('min', '<f8'),
	('max', '<f8')])
_RUN = numpy.dtype([('first', '<u8'), ('last', '<u8'), ('offset', '<u8'),
	('bytes', '<u8')])
_BLOCK = numpy.dtype([('chrom', '<u4'), ('first', '<u4'), ('last', '<u4'),
	('n', '<u4'), ('offset', '<u8'), ('size', '<u4'), ('raw', '<u4'),
	('kind', '<u4'), ('pad', '<u4')])


class SparseWindows(NamedTuple):
	"""The nonzero bases of a read of windows, as entries.

	Entry k is base offsets[k] of window rows[k] in channel channels[k], with
	the value values[k]. `shape` is (windows, channels, width), the shape of
	the dense array the same read would give. Entries are sorted by row,
	channel and offset. `scipy.sparse.coo_array((values, (rows * width +
	offsets, channels)))` and `torch.sparse_coo_tensor` take them as they
	are.
	"""

	rows: numpy.ndarray
	channels: numpy.ndarray
	offsets: numpy.ndarray
	values: numpy.ndarray
	shape: tuple


class SparseEntries(NamedTuple):
	"""The nonzero bases of tracks over chromosomes, as entries.

	Entry k is base positions[k] of chromosome chroms[k], an index into the
	reader's `chrom_sizes`, in track tracks[k], an index into its `tracks`,
	with the value values[k]. Entries are sorted by chromosome, track and
	position.
	"""

	chroms: numpy.ndarray
	positions: numpy.ndarray
	tracks: numpy.ndarray
	values: numpy.ndarray


def _check_tile(tile):
	if isinstance(tile, bool) or not isinstance(tile, (int, numpy.integer)):
		raise TypeError("tile must be an integer.")
	if not _MIN_TILE <= tile <= _MAX_TILE or tile & (tile - 1) != 0:
		raise ValueError("tile must be a power of two from {} to {}, not {}."
			.format(_MIN_TILE, _MAX_TILE, tile))

	return int(tile)


def _check_order(order):
	if not isinstance(order, str) or order not in _ORDERS:
		raise ValueError("order must be 'positions' (or 'csc') or 'tracks' (or "
			"'csr'), not {!r}.".format(order))

	return _ORDERS[order]


def _check_tracks(tracks):
	"""The `tracks` argument as a list of dicts with a `name`, a `dtype` that
	may be None, and the attributes given."""

	if isinstance(tracks, (str, dict)) or not hasattr(tracks, '__iter__'):
		raise TypeError("tracks must be a list of names or of dicts, not {}."
			.format(type(tracks).__name__))

	records, names = [], set()
	for track in tracks:
		if isinstance(track, str):
			track = {'name': track}
		elif isinstance(track, dict):
			track = dict(track)
		else:
			raise TypeError("each track must be a name or a dict, not {}.".format(
				type(track).__name__))

		name = track.get('name')
		if not isinstance(name, str) or len(name) == 0:
			raise ValueError("each track needs a non-empty string 'name', not "
				"{!r}.".format(name))
		if name in names:
			raise ValueError("track {!r} is named twice.".format(name))
		if 'group' in track:
			raise ValueError("track {!r} has a 'group' key, which the writer "
				"assigns from `groups`.".format(name))

		dtype = track.get('dtype')
		if dtype is not None:
			try:
				dtype = numpy.dtype(dtype).name
			except TypeError:
				raise TypeError("the dtype of track {!r} is not a dtype: {!r}."
					.format(name, dtype)) from None
			if dtype not in _DTYPES:
				raise ValueError("the dtype of track {!r} must be one of {}, not "
					"{!r}.".format(name, ', '.join(_DTYPES), dtype))

		track['dtype'] = dtype
		try:
			json.dumps(track)
		except TypeError as e:
			raise TypeError("the attributes of track {!r} must be JSON "
				"serializable: {}".format(name, e)) from None

		records.append(track)
		names.add(name)

	if len(records) == 0:
		raise ValueError("tracks must hold at least one track.")

	return records


def _check_groups(groups, tracks):
	"""The group of each track, as a list of arrays of track ids."""

	n = len(tracks)
	if groups is None:
		groups = _DEFAULT_GROUPS

	if isinstance(groups, bool):
		raise TypeError("groups must be an integer or a list of lists of names.")
	if isinstance(groups, (int, numpy.integer)):
		if groups < 1:
			raise ValueError("groups must be at least 1 when it is a number of "
				"tracks per group.")
		return [numpy.arange(a, min(n, a + int(groups))) for a in range(0, n,
			int(groups))]

	ids = {track['name']: i for i, track in enumerate(tracks)}
	seen, out = set(), []
	try:
		members = [list(group) for group in groups]
	except TypeError:
		raise TypeError("groups must be an integer or a list of lists of names.")

	for group in members:
		if len(group) == 0:
			raise ValueError("a group must hold at least one track.")
		for name in group:
			if name not in ids:
				raise ValueError("group member {!r} is not one of the tracks."
					.format(name))
			if name in seen:
				raise ValueError("track {!r} is in two groups.".format(name))
			seen.add(name)
		out.append(numpy.array([ids[name] for name in group], dtype=numpy.int64))

	if len(seen) != n:
		missing = [track['name'] for track in tracks if track['name'] not in seen]
		raise ValueError("every track must be in a group; {!r} is not.".format(
			missing[0]))

	return out


def _check_attributes(attributes):
	if attributes is None:
		return {}
	if not isinstance(attributes, dict):
		raise TypeError("attributes must be a dict, not {}.".format(
			type(attributes).__name__))

	try:
		json.dumps(attributes)
	except TypeError as e:
		raise TypeError("attributes must be JSON serializable: {}".format(
			e)) from None

	return dict(attributes)


def _tile_offsets(lengths, tile):
	"""The first tile of each chromosome in the file's tile numbering, and
	the number of tiles in all."""

	counts = (lengths + tile - 1) // tile
	offsets = numpy.concatenate([[0], numpy.cumsum(counts)])
	return offsets[:-1], int(offsets[-1])


def _infer_dtype(sum_, nonzero, low, high, nonint):
	if nonint:
		return 'float32'
	if nonzero == 0:
		return 'uint8'
	if low >= 0:
		return 'uint8' if high <= 255 else 'uint16' if high <= 65535 else \
			'uint32'
	if low >= -128 and high <= 127:
		return 'int8'
	if low >= -32768 and high <= 32767:
		return 'int16'
	if high <= 2147483647:
		return 'int32'

	return 'float32'


def _positions(positions, name):
	"""`positions` as a 1-D int64 array."""

	positions = numpy.asarray(positions)
	if positions.ndim != 1:
		raise ValueError("{} must be one-dimensional, not of shape {}.".format(
			name, positions.shape))
	if positions.size and positions.dtype.kind not in 'iu':
		raise TypeError("{} must be integers, not {}.".format(name,
			positions.dtype))

	return positions.astype(numpy.int64)


def _check_values_fit(name, dtype, values):
	"""Raise if the values of a track with a declared dtype do not fit it."""

	if dtype is None or dtype == 'float32':
		return

	block = values.astype(numpy.float64, copy=False)
	if numpy.isnan(block).any():
		raise ValueError("track {!r} is {}, and a value is NaN.".format(name,
			dtype))
	if (block != numpy.floor(block)).any():
		raise ValueError("track {!r} is {}, and a value is not an integer."
			.format(name, dtype))

	info = numpy.iinfo(dtype)
	if block.size and (block.min() < info.min or block.max() > info.max):
		raise ValueError("track {!r} is {}, and a value is outside {} to {}."
			.format(name, dtype, info.min, info.max))


class FigWigWriter:
	"""Write a figwig file, a batch of windows, intervals or nonzero bases at
	a time, on several threads.

	`FigWigWriter` takes the values of several tracks at once with `write`,
	as the windows `FigWigReader.read` reads, of shape (n, tracks, width),
	or as intervals, and the nonzero bases of tracks with `write_sparse`,
	and writes them as they come, so that tracks larger than memory, such as
	a model's predictions over a genome, can be written a batch at a time.
	Use it as a context manager, or call `close` when every value has been
	written; the tables and the index are written there.

	`order` decides the layout. In position order, the default, tracks are
	written in groups whose tiles are stored together: a read of any track
	in a group decodes the whole group's chunk, so the tracks of a group
	should be those read together, such as the two strands of an assay, and
	a file for hundreds of tracks read a few at a time wants small groups,
	or track order. Each `write` or `write_sparse` call covers whole groups,
	and within a group its items come in position order on each chromosome,
	after the items of earlier calls, and must not overlap; a group may move
	to a new chromosome in any order, but cannot return to one it has left,
	and the tile a call ends in is sealed by `write_sparse`, so that
	windows and nonzero bases of one tile come in one call. Writing several
	groups in turn on one chromosome is allowed, and costs nothing but a
	longer index.

	Each tile of a group is encoded as the writer reaches it: as nothing
	when every value is zero, and otherwise densely, as nonzero bases per
	track, or as a sparse matrix of the tracks present, whichever takes the
	fewest bytes, in the narrowest integer type that holds the tile's
	values, or float32 where they are not integers. A tile that `write`
	fills only part of waits for the next call. The chunks are compressed on
	up to `n_jobs` threads while the calling thread encodes the next batch,
	with zlib or libdeflate, as `BigWigWriter` compresses its blocks, and
	the file does not depend on the number of threads or on how the values
	were split between calls, only on the values and on which calls wrote
	which groups.

	In track order, every track is its own group, and its nonzero bases on
	each chromosome are stored in blocks of up to 65,536, in position
	order. Windows and intervals are reduced to their nonzero bases, and
	each track's bases must come in position order on each chromosome
	across calls. The blocks are stored uncompressed unless `level` is
	given, since a window read over many tracks decodes a block of each.

	A track's `dtype` is what `FigWigReader.read` returns its values as.
	Given, it is checked against every value written; left out, it is set
	when the writer closes, to the narrowest type that holds the track's
	values. Either way each chunk is stored as narrowly as its own values
	allow.

	Parameters
	----------
	path: str or os.PathLike
		The file to write. It is created, or emptied, when the writer is made.

	chrom_sizes: dict or list of (str, int)
		The chromosomes and their lengths: a dict such as
		`BigWigReader.chrom_sizes`, or a list of (name, length) pairs.

	tracks: list of str or of dict
		The tracks, in order: a name for each, or a dict with a 'name', an
		optional 'dtype' (one of uint8, uint16, uint32, int8, int16, int32 or
		float32), and any JSON-serializable attributes, such as the assay,
		the biosample or the strand, which `FigWigReader.select` matches.

	genome: str or None, optional
		The reference genome the coordinates are on, as a name such as
		'hg38' or a URL to its sequence. Default is None.

	attributes: dict or None, optional
		JSON-serializable attributes of the file as a whole, such as how it
		was made. Default is None.

	tile: int, optional
		The bases per tile in position order, a power of two from 16 to
		65536. Smaller tiles decode less for each window and compress a
		little less well. Default is 16384.

	groups: int or list of lists of str, optional
		In position order, the tracks whose tiles are stored together: a
		number of consecutive tracks per group, or a list of groups of track
		names that holds every track once. In track order every track is its
		own group, whatever is given. Default is 8.

	level: int or None, optional
		The compression level, as `BigWigWriter` takes it, or 0 to store the
		chunks without compressing them, which decodes fastest. None is 6 in
		position order and 0 in track order. Default is None.

	engine: str, optional
		What compresses the chunks: 'zlib', 'libdeflate', 'isal' or 'auto';
		see `BigWigWriter`. Default is 'auto'.

	n_jobs: int, optional
		The most threads that encode and compress at once, besides the
		calling thread, or -1 for one per CPU. Default is 8.

	order: str, optional
		'positions' (or 'csc'), the layout for reading windows across many
		tracks, or 'tracks' (or 'csr'), the layout for reading a few tracks
		over the genome. Default is 'positions'.
	"""

	def __init__(self, path: str | os.PathLike, chrom_sizes: dict | list,
		tracks: list, genome: str | None = None, attributes: dict | None = None,
		tile: int = _DEFAULT_TILE, groups: int | list = _DEFAULT_GROUPS,
		level: int | None = None, engine: str = 'auto', n_jobs: int = 8,
		order: str = 'positions'):
		self._file = None
		self._pool = None

		self.chrom_sizes = dict(_check_chrom_sizes(chrom_sizes))
		self._chrom_list = list(self.chrom_sizes.items())
		self._tids = {name: i for i, name in enumerate(self.chrom_sizes)}
		self._lengths = numpy.array(list(self.chrom_sizes.values()),
			dtype=numpy.int64)

		self.order = _check_order(order)
		self.tracks = _check_tracks(tracks)
		if self.order == 'tracks':
			self._groups = [numpy.array([i]) for i in range(len(self.tracks))]
		else:
			self._groups = _check_groups(groups, self.tracks)
		self.tile = _check_tile(tile)
		for group in self._groups:
			if len(group) * self.tile > _MAX_GROUP_BASES:
				raise ValueError("a group of {} tracks at a tile of {} bases "
					"holds more than 2**29 values per chunk; use smaller groups "
					"or tiles.".format(len(group), self.tile))

		self._track_ids = {track['name']: i for i, track in enumerate(self.tracks)}
		self._group_of = numpy.zeros(len(self.tracks), dtype=numpy.int64)
		self._slot_of = numpy.zeros(len(self.tracks), dtype=numpy.int64)
		for g, group in enumerate(self._groups):
			self._group_of[group] = g
			self._slot_of[group] = numpy.arange(len(group))
			for i in group:
				self.tracks[i]['group'] = g

		if genome is not None and not isinstance(genome, str):
			raise TypeError("genome must be a string or None, not {}.".format(
				type(genome).__name__))

		self.genome = genome
		self.attributes = _check_attributes(attributes)
		if level is None:
			level = 6 if self.order == 'positions' else 0
		self.engine, self.level = _check_engine(engine, level)
		self._matrix = _MATRIX_CHUNKS and self.level == 0
		self.n_jobs = _check_n_jobs(n_jobs)
		self.path = path

		self._tile_offsets, self._n_tiles = _tile_offsets(self._lengths,
			self.tile)

		n_groups = len(self._groups)
		self._kinds = [numpy.zeros(self._n_tiles, dtype=numpy.uint8) for _ in
			range(n_groups)] if self.order == 'positions' else None
		self._sizes = [numpy.zeros(self._n_tiles, dtype=numpy.uint32) for _ in
			range(n_groups)] if self.order == 'positions' else None
		self._raw = [numpy.zeros(self._n_tiles, dtype=numpy.uint32) for _ in
			range(n_groups)] if self.order == 'positions' else None
		self._runs = [[] for _ in range(n_groups)]
		self._blocks = [[] for _ in range(n_groups)]
		self._pending = [None] * n_groups
		self._buffers = [None] * n_groups
		self._open = [None] * n_groups
		self._sealed = [None] * n_groups
		self._last = [None] * n_groups
		self._done = [set() for _ in range(n_groups)]
		self._stats = numpy.zeros((len(self.tracks), 5), dtype=numpy.float64)
		self._stats[:, 2] = numpy.inf
		self._stats[:, 3] = -numpy.inf
		self._inflight = []
		self._copies = True

		self._file = open(path, 'w+b')
		self._file.write(bytes(_HEADER.itemsize))

		if self.n_jobs > 1:
			self._pool = ThreadPoolExecutor(self.n_jobs)

	def __repr__(self):
		return "FigWigWriter({!r}, {} tracks, {} order)".format(os.fspath(
			self.path), len(self.tracks), self.order[:-1])

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

	def _select(self, tracks):
		"""The tracks a call names, as (track ids in channel order, whether a
		single track was named)."""

		if tracks is None:
			return numpy.arange(len(self.tracks)), False

		single = isinstance(tracks, (str, int, numpy.integer)) and \
			not isinstance(tracks, bool)
		items = [tracks] if single else tracks
		if isinstance(items, (str, dict)) or not hasattr(items, '__iter__'):
			raise TypeError("tracks must be a name, an index, or a list of "
				"them, not {}.".format(type(tracks).__name__))

		ids = []
		for item in items:
			if isinstance(item, str):
				if item not in self._track_ids:
					raise ValueError("{!r} is not one of the tracks.".format(item))
				ids.append(self._track_ids[item])
			elif isinstance(item, (int, numpy.integer)) and not isinstance(item,
					bool):
				if not 0 <= item < len(self.tracks):
					raise ValueError("track {} is out of range; there are {} "
						"tracks.".format(item, len(self.tracks)))
				ids.append(int(item))
			else:
				raise TypeError("tracks must be names or indices, not {}.".format(
					type(item).__name__))

		return numpy.array(ids, dtype=numpy.int64), single

	def _whole_groups(self, ids):
		"""The groups the track ids make up, as {group: {track id: channel}},
		raising if a group is only partly among them."""

		if len(ids) != len(set(ids.tolist())):
			raise ValueError("a track is named twice in tracks.")

		groups = {}
		for channel, i in enumerate(ids.tolist()):
			groups.setdefault(int(self._group_of[i]), {})[i] = channel

		for g, members in groups.items():
			if len(members) != len(self._groups[g]):
				missing = [self.tracks[i]['name'] for i in self._groups[g] if i
					not in members]
				raise ValueError("tracks must make up whole groups: {!r} is in "
					"the group of {!r}, which is not among them.".format(
					missing[0], self.tracks[list(members)[0]]['name']))

		return groups

	def _check_chroms(self, chroms, n):
		names, codes = _chrom_codes(chroms, n)
		unknown = [name for name in names if name not in self._tids]
		if len(unknown) > 0:
			raise ValueError("chromosome {!r} is not one of the writer's "
				"chromosomes.".format(unknown[0]))

		tids = numpy.array([self._tids[name] for name in names],
			dtype=numpy.int64)[codes]
		return names, codes, tids

	def _check_order_of(self, g, tids, starts, ends, bounds, describe, order):
		"""Raise if a group's items on the chromosomes of a call, sorted,
		come before what was written to it earlier."""

		last, done, sealed = self._last[g], self._done[g], self._sealed[g]
		for a, b in zip(bounds[:-1], bounds[1:]):
			tid = int(tids[a])
			if last is not None and tid != last[0]:
				done = done | {last[0]}
			if tid in done:
				raise ValueError("chromosome {!r} is written again after the "
					"group of {!r} moved on from it.".format(self._chrom_list[tid][
					0], self.tracks[self._groups[g][0]]['name']))
			if last is not None and tid == last[0] and starts[a] < last[1]:
				raise ValueError("{} starts at {}, before the end of what was "
					"written on it before, {}.".format(describe(a if order is None
					else int(order[a])), starts[a], last[1]))
			if sealed is not None and tid == sealed[0] and starts[a] // self.tile \
					<= sealed[1]:
				raise ValueError("{} falls in tile {} of {!r}, which an earlier "
					"call already wrote; the windows and nonzero bases of one "
					"tile must come in one call.".format(describe(a if order is
					None else int(order[a])), starts[a] // self.tile,
					self._chrom_list[tid][0]))
			last = (tid, int(ends[b - 1]))

	def write(self, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
		numpy.ndarray, values, ends: list[int] | numpy.ndarray | None = None,
		tracks: str | int | list | None = None):
		"""Write the values of windows, or of intervals, for whole groups of
		tracks.

		Without `ends`, item j is the window [starts[j], starts[j] + width)
		on chroms[j], and values[j], of shape (tracks, width), holds the
		value of each of its bases in each track, as `FigWigReader.read`
		returns them. With `ends`, item j is the interval [starts[j],
		ends[j]), and values[j], of shape (tracks,), is the value of every
		base in it in each track. Positions are 0-based and ends exclusive,
		as in BED files.

		Items may be given in any order, and are sorted by chromosome and
		start, but must not overlap. Within each group, the items of a call
		come after those of earlier calls on the same chromosome, and a
		group cannot return to a chromosome it has left. A window may run
		past the end of its chromosome only where its values are NaN or 0,
		as reads give them there.

		A base that is not written is 0. NaN is a value for a float32 track,
		and an error for an integer one. In track order, the items' nonzero
		bases are what is written.

		Parameters
		----------
		chroms: str, list of str, or numpy.ndarray of str
			The chromosome of each item, or one name for every item.

		starts: list of int or numpy.ndarray of int, shape=(n,)
			The start of each item.

		values: array-like of numbers
			The values: of shape (n, tracks, width) for windows and (n,
			tracks) for intervals, with one channel for each track named by
			`tracks`, in its order; or (n, width) and (n,) when `tracks` is
			one name or index.

		ends: list of int or numpy.ndarray of int, shape=(n,), or None, optional
			The end of each interval. If None, the items are windows. Default
			is None.

		tracks: str, int, list of str or int, or None, optional
			The tracks the channels of `values` are for, which must make up
			whole groups. None is every track, in order. Default is None.
		"""

		if self._file is None:
			raise ValueError("the writer is closed.")

		ids, single = self._select(tracks)
		groups = self._whole_groups(ids)

		values = numpy.asarray(values)
		if values.dtype.kind not in 'iuf':
			raise TypeError("values must be numbers, not {}.".format(values.dtype))

		starts = _positions(starts, 'starts')
		n = len(starts)
		item = 'window' if ends is None else 'interval'
		if single and ends is None and values.ndim == 2:
			values = values[:, None, :]
		elif single and ends is not None and values.ndim == 1:
			values = values[:, None]
		expected = 2 if ends is not None else 3
		if values.ndim != expected or values.shape[0] != n or values.shape[1] \
				!= len(ids):
			raise ValueError("values must be of shape (n, tracks{}) with one "
				"{} per start and one channel per track, ({}, {}{}), not {}."
				.format('' if ends is not None else ', width', item, n, len(ids),
				'' if ends is not None else ', width', values.shape))

		names, codes, tids = self._check_chroms(chroms, n)
		if n == 0:
			return
		lengths = self._lengths[tids]

		def describe(j):
			return "{} {} on {!r}".format(item, j, names[codes[j]])

		if starts.min() < 0:
			j = int(numpy.argmax(starts < 0))
			raise ValueError("{} starts before 0, at {}.".format(describe(j),
				starts[j]))

		if ends is None:
			width = values.shape[2]
			if width < 1:
				raise ValueError("windows must be at least one base wide.")
			ends = starts + width
		else:
			ends = _positions(ends, 'ends')
			if ends.shape != starts.shape:
				raise ValueError("ends must have one end per start.")
			if (ends <= starts).any():
				j = int(numpy.argmax(ends <= starts))
				raise ValueError("{} ends at {}, at or before its start, {}."
					.format(describe(j), ends[j], starts[j]))

		over = numpy.flatnonzero(ends > lengths)
		if len(over) > 0:
			if values.ndim == 2:
				j = int(over[0])
				raise ValueError("{} ends at {}, past its length, {}.".format(
					describe(j), ends[j], lengths[j]))

			# A window past the end is accepted where its values are NaN or
			# 0, as reads give them there, and is cut there.
			for j in over.tolist():
				tail = values[j, :, max(0, lengths[j] - starts[j]):].astype(
					numpy.float64)
				if not (numpy.isnan(tail) | (tail == 0)).all():
					raise ValueError("{} ends at {}, past its length, {}; a "
						"window may run past the end only where its values are "
						"NaN or 0.".format(describe(j), ends[j], lengths[j]))
			ends = numpy.minimum(ends, lengths)
			if (ends <= starts).any():
				j = int(numpy.argmax(ends <= starts))
				raise ValueError("{} starts at {}, at or past its length, {}."
					.format(describe(j), starts[j], lengths[j]))

		# Items that already come in order, as counts from a sorted BAM do,
		# are not sorted again.
		keys = (tids << 32) | starts
		order = None
		if (keys[1:] <= keys[:-1]).any():
			order = numpy.lexsort((starts, tids))
			tids, starts, ends = tids[order], starts[order], ends[order]
			values = values[order]

		overlap = (tids[1:] == tids[:-1]) & (starts[1:] < ends[:-1])
		if overlap.any():
			k = int(numpy.argmax(overlap)) + 1
			j = k if order is None else int(order[k])
			i = k - 1 if order is None else int(order[k - 1])
			raise ValueError("{}, at {}, starts before {} {} ends, at {}; they "
				"must not overlap.".format(describe(j), starts[k], item, i,
				ends[k - 1]))

		# Each group's ordering is checked for every chromosome of the call
		# before anything is encoded, so that a call that raises writes
		# nothing.
		bounds = numpy.concatenate([[0], numpy.flatnonzero(tids[1:] !=
			tids[:-1]) + 1, [n]])
		for g in groups:
			self._check_order_of(g, tids, starts, ends, bounds, describe, order)

		# A declared dtype is checked against the values inside the
		# chromosome; the bases of a window past its end are NaN or 0.
		inside = None
		if values.ndim == 3 and len(over) > 0 and any(self.tracks[i]['dtype']
				not in (None, 'float32') for members in groups.values() for i in
				members):
			inside = numpy.arange(values.shape[2])[None, :] < (ends - starts)[:,
				None]
		for g, members in groups.items():
			for i, channel in members.items():
				if self.tracks[i]['dtype'] in (None, 'float32'):
					continue
				block = values[:, channel]
				if inside is not None:
					block = block[inside]
				_check_values_fit(self.tracks[i]['name'], self.tracks[i]['dtype'],
					block)

		if self.order == 'tracks':
			for g, members in groups.items():
				(i, channel), = members.items()
				for a, b in zip(bounds[:-1], bounds[1:]):
					positions, entry_values = _items_to_entries(starts[a:b],
						ends[a:b], values[a:b, channel])
					self._write_track_entries(i, int(tids[a]), positions,
						entry_values)
			self._write(max(0, len(self._inflight) - (0 if self._pool is None
				else 2 * self.n_jobs)))
			return

		# The encoders take float32 and integers as they are, and other types
		# as float64, from a copy, since encoding goes on after this call
		# returns and the caller may then change `values`; a caller that
		# never touches its arrays again, as the converters are, turns the
		# copy off with `_copies`, and each call then costs no pass over
		# the values on this thread.
		if values.dtype == numpy.float32:
			dtype = numpy.float32
		elif values.dtype.kind in 'iu' and values.dtype.itemsize < 8 or \
				values.dtype == numpy.int64:
			dtype = numpy.int64
		else:
			dtype = numpy.float64

		for g, members in groups.items():
			slots = numpy.array([members[int(i)] for i in self._groups[g]],
				dtype=numpy.int64)
			identity = len(slots) == values.shape[1] and bool((slots ==
				numpy.arange(len(slots))).all())
			for a, b in zip(bounds[:-1], bounds[1:]):
				block = values[a:b] if identity else values[a:b, slots]
				if block.ndim == 2:
					block = block[:, :, None]
				if self._copies:
					block = numpy.array(block, dtype=dtype, order='C')
				else:
					block = numpy.ascontiguousarray(block, dtype=dtype)
				self._write_group(g, int(tids[a]), starts[a:b], ends[a:b], block,
					values.ndim == 3)

		# Batches are written as they finish compressing, keeping a few in
		# flight per thread, so that a long call does not hold every chunk
		# of its values in memory at once.
		self._write(max(0, len(self._inflight) - (0 if self._pool is None else
			2 * self.n_jobs)))

	def write_sparse(self, chroms: str | list[str] | numpy.ndarray,
		positions: list[int] | numpy.ndarray, tracks: str | int | list |
		numpy.ndarray, values):
		"""Write nonzero bases, one entry each, for whole groups of tracks.

		Entry k is base positions[k] of chroms[k] in track tracks[k], with
		the value values[k]. This is the form of per-cell cut sites, of a
		sparse matrix's triplets and of what `FigWigReader.read` gives with
		`sparse=True`. The call writes the groups of the tracks named over
		the tiles its entries fall in, so every other base of those groups'
		tracks is 0 there, and a group's entries come after its earlier
		items on each chromosome; in position order they also seal the tile
		they end in, so that a later call cannot add to it. No two entries
		may name one base of one track.

		Parameters
		----------
		chroms: str, list of str, or numpy.ndarray of str
			The chromosome of each entry, or one name for every entry.

		positions: list of int or numpy.ndarray of int, shape=(n,)
			The base of each entry.

		tracks: str, int, list or numpy.ndarray of str or int
			The track of each entry, by name or index, or one for every
			entry.

		values: array-like of numbers, shape=(n,)
			The value of each entry.
		"""

		if self._file is None:
			raise ValueError("the writer is closed.")

		positions = _positions(positions, 'positions')
		n = len(positions)
		values = numpy.asarray(values)
		if values.dtype.kind not in 'iuf':
			raise TypeError("values must be numbers, not {}.".format(values.dtype))
		if values.shape != (n,):
			raise ValueError("values must have one value per position, shape ({},"
				"), not {}.".format(n, values.shape))
		if n == 0:
			return

		if isinstance(tracks, (str, int, numpy.integer)) and not isinstance(
				tracks, bool):
			track_ids = numpy.full(n, self._select(tracks)[0][0], dtype=numpy.int64)
		else:
			tracks = numpy.asarray(tracks)
			if tracks.shape != (n,):
				raise ValueError("tracks must name one track per position, or one "
					"for all.")
			if tracks.dtype.kind in 'US':
				unique, inverse = numpy.unique(tracks, return_inverse=True)
				track_ids = self._select(unique.tolist())[0][inverse]
			elif tracks.dtype.kind in 'iu':
				track_ids = tracks.astype(numpy.int64)
				if n > 0 and (track_ids.min() < 0 or track_ids.max() >= len(
						self.tracks)):
					raise ValueError("a track index is out of range; there are {} "
						"tracks.".format(len(self.tracks)))
			else:
				raise TypeError("tracks must be names or indices, not {}.".format(
					tracks.dtype))

		names, codes, tids = self._check_chroms(chroms, n)
		if n == 0:
			return
		present = numpy.zeros(len(self.tracks), dtype=numpy.bool_)
		present[track_ids] = True
		ids = numpy.flatnonzero(present)

		lengths = self._lengths[tids]
		if positions.min() < 0 or (positions >= lengths).any():
			j = int(numpy.argmax((positions < 0) | (positions >= lengths)))
			raise ValueError("entry {} on {!r} is at {}, outside the chromosome."
				.format(j, names[codes[j]], positions[j]))

		# Declared dtypes are checked over all their tracks' entries at once,
		# with each track's range found in one pass.
		declared = [i for i in ids.tolist() if self.tracks[i]['dtype'] not in
			(None, 'float32')]
		if declared:
			mask = numpy.isin(track_ids, declared)
			block = values[mask].astype(numpy.float64)
			if numpy.isnan(block).any() or (block != numpy.floor(block)).any():
				bad = numpy.flatnonzero(mask)[numpy.flatnonzero(numpy.isnan(block) |
					(block != numpy.floor(block)))[0]]
				_check_values_fit(self.tracks[int(track_ids[bad])]['name'],
					self.tracks[int(track_ids[bad])]['dtype'], values[bad:bad + 1])
			lows = numpy.full(len(self.tracks), numpy.inf)
			highs = numpy.full(len(self.tracks), -numpy.inf)
			numpy.minimum.at(lows, track_ids[mask], block)
			numpy.maximum.at(highs, track_ids[mask], block)
			for i in declared:
				info = numpy.iinfo(self.tracks[i]['dtype'])
				if lows[i] < info.min or highs[i] > info.max:
					raise ValueError("track {!r} is {}, and a value is outside {} to "
						"{}.".format(self.tracks[i]['name'], self.tracks[i]['dtype'],
						info.min, info.max))

		# One sort by chromosome, group, tile, track and position puts the
		# entries in the order the encoders take and two entries on one base
		# of one track next to each other. The sort is of one composite key
		# per entry where the chromosome and group are the same throughout,
		# as in a call per chromosome, and of two otherwise; a lexsort of the
		# five would take several times longer.
		group_ids = self._group_of[track_ids]
		order = self._entry_order(tids, group_ids, track_ids, positions)
		tids, positions, track_ids, group_ids = tids[order], positions[order], \
			track_ids[order], group_ids[order]
		values = values[order].astype(numpy.float64)
		same = (tids[1:] == tids[:-1]) & (track_ids[1:] == track_ids[:-1]) & (
			positions[1:] == positions[:-1])
		if same.any():
			k = int(numpy.argmax(same)) + 1
			raise ValueError("entries {} and {} both name base {} of {!r} in "
				"track {!r}.".format(int(order[k - 1]), int(order[k]),
				positions[k], names[codes[int(order[k])]], self.tracks[int(
				track_ids[k])]['name']))

		def describe(j):
			return "entry {} on {!r}".format(j, names[codes[j]])

		# The entries of each (chromosome, group), checked against what the
		# group wrote before, then encoded.
		bounds = numpy.concatenate([[0], numpy.flatnonzero((tids[1:] != tids[
			:-1]) | (group_ids[1:] != group_ids[:-1])) + 1, [n]])
		for a, b in zip(bounds[:-1], bounds[1:]):
			low = int(positions[a:b].min())
			high = int(positions[a:b].max())
			self._check_order_of(int(group_ids[a]), tids[a:a + 1], numpy.array([
				low]), numpy.array([high + 1]), numpy.array([0, 1]), lambda j, c=a:
				describe(int(order[c])), None)

		for a, b in zip(bounds[:-1], bounds[1:]):
			g, tid = int(group_ids[a]), int(tids[a])
			if self.order == 'tracks':
				self._write_track_entries(int(self._groups[g][0]), tid,
					positions[a:b], values[a:b])
			else:
				self._write_group_entries(g, tid, self._slot_of[track_ids[a:b]],
					positions[a:b], values[a:b])

		self._write(max(0, len(self._inflight) - (0 if self._pool is None else
			2 * self.n_jobs)))

	def _entry_order(self, tids, group_ids, track_ids, positions):
		"""The order that sorts entries by chromosome, group, tile, track
		and position."""

		shift = self.tile.bit_length() - 1
		longest = int(self._lengths.max()).bit_length()
		slots = self._slot_of[track_ids]
		if longest + 25 > 63:
			return numpy.lexsort((positions, slots, positions >> shift, group_ids,
				tids))

		# A group holds at most 2**25 tracks (_MAX_GROUP_BASES at the
		# smallest tile), so the slot takes 25 bits between the tile and the
		# offset in it, which together take the chromosome's bits.
		low = ((positions >> shift) << (shift + 25)) | (slots << shift) | (
			positions & (self.tile - 1))
		high = tids * len(self._groups) + group_ids
		if (high[1:] == high[:-1]).all():
			return numpy.argsort(low, kind='stable')
		return numpy.lexsort((low, high))

	def _write_group(self, g, tid, starts, ends, values, by_base):
		"""Encode one group's items on one chromosome, in order, into chunks,
		and hand them to the compressors."""

		tile = self.tile
		n_tracks = len(self._groups[g])
		if self._buffers[g] is None:
			self._buffers[g] = numpy.zeros((n_tracks, tile), dtype=numpy.float64)

		first_tile = int(starts[0] // tile)
		open_ = self._open[g]
		preset = False
		if open_ is not None:
			if open_[0] == tid and open_[1] == first_tile:
				preset = True
			else:
				self._encode(g, open_[0], numpy.array([open_[1]]), None, None,
					None, True, False)
			self._open[g] = None

		# The tiles the items touch; the others stay zero and take nothing. A
		# tile that one window covers whole is encoded from the window's
		# values; the rest are filled into the group's buffer.
		k0, k1 = starts // tile, (ends - 1) // tile
		counts = k1 - k0 + 1
		total = int(counts.sum())
		tiles = numpy.repeat(k0, counts) + numpy.arange(total) - numpy.repeat(
			numpy.cumsum(counts) - counts, counts)
		tiles = numpy.unique(tiles)
		last_tile = int(tiles[-1])

		# The last tile stays open unless its last item ends on its edge, or
		# at the end of the chromosome.
		last_full = int(ends[-1]) == (last_tile + 1) * tile or \
			int(ends[-1]) == int(self._lengths[tid])
		bound = n_tracks * tile * 4 + 16
		per_batch = max(1, _BATCH_BYTES // bound)

		# Whole tiles belong to the window that covers them, and the segments
		# of whole and of partial tiles are encoded in tile order, so that
		# each run of chunks in the index spans only its own tiles.
		full_tiles = numpy.zeros(0, dtype=numpy.int64)
		full_items = numpy.zeros(0, dtype=numpy.int64)
		if by_base:
			f0 = -(-starts // tile)
			f1 = ends // tile - 1
			whole = numpy.flatnonzero(f1 >= f0)
			if len(whole) > 0:
				spans = f1[whole] - f0[whole] + 1
				full_tiles = numpy.repeat(f0[whole], spans) + numpy.arange(int(
					spans.sum())) - numpy.repeat(numpy.cumsum(spans) - spans, spans)
				full_items = numpy.repeat(whole, spans)

		is_full = numpy.isin(tiles, full_tiles)
		cuts = numpy.concatenate([[0], numpy.flatnonzero(is_full[1:] !=
			is_full[:-1]) + 1, [len(tiles)]])
		for a, b in zip(cuts[:-1], cuts[1:]):
			if is_full[a]:
				items = full_items[numpy.searchsorted(full_tiles, tiles[a:b])]
				columns = tiles[a:b] * tile - starts[items]
				for e in range(a, b, per_batch):
					f = min(b, e + per_batch)
					self._encode_dense(g, tid, values, items[e - a:f - a],
						columns[e - a:f - a], tiles[e:f])
			else:
				segment = tiles[a:b]
				lo = numpy.searchsorted(ends, segment * tile, side='right')
				hi = numpy.searchsorted(starts, (segment + 1) * tile, side='left')
				for e in range(0, len(segment), per_batch):
					f = min(len(segment), e + per_batch)
					keep_last = b == len(tiles) and f == len(segment) and \
						not last_full
					self._encode(g, tid, segment[e:f], lo[e:f], hi[e:f], (starts,
						ends, values, by_base), preset and a == 0 and e == 0,
						keep_last)

		if not last_full:
			self._open[g] = (tid, last_tile)

		self._note_written(g, tid, int(ends[-1]))

	def _note_written(self, g, tid, end):
		last = self._last[g]
		if last is not None and last[0] != tid:
			self._done[g].add(last[0])
		self._last[g] = (tid, end)

	def _write_group_entries(self, g, tid, slots, positions, values):
		"""Encode one group's nonzero bases on one chromosome, sorted by
		tile, track and position, into chunks on the pool."""

		tile = self.tile
		n_tracks = len(self._groups[g])
		open_ = self._open[g]
		if open_ is not None:
			self._encode(g, open_[0], numpy.array([open_[1]]), None, None, None,
				True, False)
			self._open[g] = None
			if open_[0] == tid and open_[1] == int(positions.min() // tile):
				raise ValueError("tile {} of {!r} was begun by write, and "
					"write_sparse cannot add to it; the windows and nonzero bases "
					"of one tile must come in one call.".format(open_[1],
					self._chrom_list[tid][0]))

		tiles = positions // tile
		slots = slots.astype(numpy.int64)
		first = numpy.concatenate([[0], numpy.flatnonzero(tiles[1:] != tiles[
			:-1]) + 1, [len(tiles)]])
		unique = tiles[first[:-1]]

		# Batches of tiles that hold about _BATCH_BYTES of entries, or fewer
		# so that each thread gets several.
		per_batch = _BATCH_BYTES // 8
		if self._pool is not None:
			per_batch = max(_BLOCK_ENTRIES, min(per_batch, len(positions) // (
				_BATCHES_PER_THREAD * self.n_jobs)))
		a = 0
		while a < len(unique):
			b = a
			total = 0
			while b < len(unique) and (b == a or total < per_batch):
				total += int(first[b + 1] - first[b])
				b += 1
			e0, e1 = int(first[a]), int(first[b])
			task = (self._entries_task, g, tid, unique[a:b], first[a:b + 1] - e0,
				slots[e0:e1], positions[e0:e1], values[e0:e1], n_tracks)
			if self._pool is None:
				self._inflight.append((task[0](*task[1:]), g, None))
			else:
				self._inflight.append((self._pool.submit(*task), g, None))
			a = b

		self._sealed[g] = (tid, int(unique[-1]))
		self._note_written(g, tid, int(positions[-1]) + 1)

	def _entries_task(self, g, tid, tiles, first, slots, positions, values,
		n_tracks):
		n = len(tiles)
		n_entries = len(positions)
		out = numpy.empty(n * (8 * n_tracks + 8 * min(n_tracks, n_entries) + 32) +
			7 * n_entries, dtype=numpy.uint8)
		bounds = numpy.zeros(n + 1, dtype=numpy.int64)
		kinds = numpy.zeros(n, dtype=numpy.uint8)
		stats = numpy.zeros((n_tracks, 5), dtype=numpy.float64)
		stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf
		_encode_entries(self.tile, n_tracks, tiles, first, slots, positions,
			values, self._matrix, out, bounds, kinds, stats)
		return self._compress(g, tid, tiles, n, out, bounds, kinds) + (stats,)

	def _encode_dense(self, g, tid, values, items, columns, tiles):
		"""Encode whole tiles covered by windows, each tile from window
		items[t] at column columns[t], on the pool when there is one, and
		compress them in the same task."""

		task = (self._dense_task, g, tid, values, items, columns, tiles)
		if self._pool is None:
			self._inflight.append((task[0](*task[1:]), g, None))
		else:
			self._inflight.append((self._pool.submit(*task), g, None))

	def _dense_task(self, g, tid, values, items, columns, tiles):
		"""Encode and compress one batch of whole tiles. Returns (compressed
		bytes, sizes, raw sizes, kind bytes, file tiles, stats) of the live
		tiles."""

		n_tracks = len(self._groups[g])
		tile = self.tile
		n = len(tiles)
		out = numpy.empty(n * (n_tracks * tile * 4 + 16), dtype=numpy.uint8)
		bounds = numpy.zeros(n + 1, dtype=numpy.int64)
		kinds = numpy.zeros(n, dtype=numpy.uint8)
		stats = numpy.zeros((n_tracks, 5), dtype=numpy.float64)
		stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf
		_encode_dense_tiles(tile, values, items, columns, self._matrix, out,
			bounds, kinds, stats)
		return self._compress(g, tid, tiles, n, out, bounds, kinds) + (stats,)

	def _encode(self, g, tid, tiles, lo, hi, items, preset, keep_last):
		"""Run the encoder on a batch of tiles of one group and chromosome,
		on this thread, since it fills the group's buffer, and hand the
		chunks it made to the compressors."""

		n_tracks = len(self._groups[g])
		tile = self.tile
		if self._buffers[g] is None:
			self._buffers[g] = numpy.zeros((n_tracks, tile), dtype=numpy.float64)
		if items is None:
			starts = numpy.zeros(0, dtype=numpy.int64)
			ends = numpy.zeros(0, dtype=numpy.int64)
			values = numpy.zeros((0, n_tracks, 1), dtype=numpy.float64)
			by_base = False
			lo = hi = numpy.zeros(len(tiles), dtype=numpy.int64)
		else:
			starts, ends, values, by_base = items

		n = len(tiles)
		out = numpy.empty(n * (n_tracks * tile * 4 + 16), dtype=numpy.uint8)
		bounds = numpy.zeros(n + 1, dtype=numpy.int64)
		kinds = numpy.zeros(n, dtype=numpy.uint8)
		stats = self._stats[self._groups[g]]
		n_done = _encode_tiles(tile, tiles, lo, hi, starts, ends, values, by_base,
			self._buffers[g], preset, keep_last, self._matrix, out, bounds,
			kinds, stats)
		self._stats[self._groups[g]] = stats

		live = numpy.flatnonzero(kinds[:n_done])
		if len(live) == 0:
			return

		args = (g, tid, tiles, n_done, out, bounds, kinds)
		if self._pool is None:
			self._inflight.append((self._compress(*args), g, None))
		else:
			self._inflight.append((self._pool.submit(self._compress, *args), g,
				None))

	def _compress(self, g, tid, tiles, n_done, out, bounds, kinds):
		"""Compress the first `n_done` encoded tiles of a batch. Returns
		(compressed bytes, sizes, raw sizes, kind bytes, file tiles) of the
		live tiles."""

		live = numpy.flatnonzero(kinds[:n_done])
		gtiles = self._tile_offsets[tid] + tiles[:n_done][live]
		if len(live) == 0:
			empty = numpy.zeros(0, dtype=numpy.int64)
			return numpy.zeros(0, dtype=numpy.uint8), empty, empty, kinds[:0], gtiles

		cuts = numpy.append(bounds[live], bounds[n_done])
		raw = numpy.diff(cuts)
		data = out[:bounds[n_done]]
		if self.level == 0:
			return data, raw, raw, kinds[:n_done][live], gtiles

		compressed, sizes = _compress_blocks(self.engine, self.level, data, cuts)
		return compressed, sizes, raw, kinds[:n_done][live], gtiles

	def _write_track_entries(self, t, tid, positions, values):
		"""Add one track's nonzero bases on one chromosome, in position
		order, to its pending block, and encode the blocks that are full."""

		pending = self._pending[t]
		if pending is not None and pending[0] != tid:
			self._flush_track(t)
			pending = None
		if pending is None:
			pending = self._pending[t] = [tid, [], [], 0]

		keep = values != 0
		if not keep.all():
			positions, values = positions[keep], values[keep]
		if len(positions) == 0:
			self._note_written(t, tid, int(positions.max()) + 1 if len(positions)
				else (self._last[t][1] if self._last[t] is not None and self._last[
				t][0] == tid else 0))
			return

		pending[1].append(positions)
		pending[2].append(values.astype(numpy.float64, copy=False))
		pending[3] += len(positions)
		if pending[3] >= _BLOCK_ENTRIES:
			self._flush_track(t, final=False)

		self._note_written(t, tid, int(positions[-1]) + 1)

	def _flush_track(self, t, final=True):
		"""Encode the pending entries of track t as blocks: every full block,
		and the rest too when `final`."""

		pending = self._pending[t]
		if pending is None or pending[3] == 0:
			self._pending[t] = None
			return

		tid = pending[0]
		positions = numpy.concatenate(pending[1])
		values = numpy.concatenate(pending[2])
		n = len(positions)
		n_blocks = n // _BLOCK_ENTRIES if not final else -(-n // _BLOCK_ENTRIES)
		cut = n_blocks * _BLOCK_ENTRIES if not final else n
		if n_blocks == 0:
			return
		if cut < n:
			self._pending[t] = [tid, [positions[cut:]], [values[cut:]], n - cut]
		else:
			self._pending[t] = None

		first = numpy.arange(0, cut + 1, _BLOCK_ENTRIES)
		if first[-1] != cut:
			first = numpy.append(first, cut)
		task = (self._blocks_task, t, tid, positions[:cut], values[:cut], first)
		if self._pool is None:
			self._inflight.append((task[0](*task[1:]), t, 'blocks'))
		else:
			self._inflight.append((self._pool.submit(*task), t, 'blocks'))

	def _blocks_task(self, t, tid, positions, values, first):
		"""Encode and compress one track's blocks. Returns (compressed bytes,
		sizes, raw sizes, kind bytes, block records, stats)."""

		n = len(first) - 1
		out = numpy.empty(16 * n + 12 * len(positions) + 16, dtype=numpy.uint8)
		bounds = numpy.zeros(n + 1, dtype=numpy.int64)
		kinds = numpy.zeros(n, dtype=numpy.uint8)
		stats = numpy.zeros((1, 5), dtype=numpy.float64)
		stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf
		_encode_blocks(positions, values, first, out, bounds, kinds, stats)
		raw = numpy.diff(bounds)
		data = out[:bounds[-1]]
		if self.level == 0:
			compressed, sizes = data, raw
		else:
			compressed, sizes = _compress_blocks(self.engine, self.level, data,
				bounds)

		records = numpy.zeros(n, dtype=_BLOCK)
		records['chrom'] = tid
		records['first'] = positions[first[:-1]]
		records['last'] = positions[first[1:] - 1]
		records['n'] = numpy.diff(first)
		records['size'] = sizes
		records['raw'] = raw
		records['kind'] = kinds
		return compressed, sizes, raw, kinds, records, stats

	def _write(self, n_batches):
		"""Write the first `n_batches` batches in flight to the file, in
		order, record their chunks or blocks in the index, and fold in the
		stats of those encoded on the pool."""

		for result, g, kind in self._inflight[:n_batches]:
			if not isinstance(result, tuple):
				result = result.result()
			if len(result) == 6:
				stats = result[5]
				rows = self._groups[g]
				self._stats[rows, 0] += stats[:, 0]
				self._stats[rows, 1] += stats[:, 1]
				self._stats[rows, 2] = numpy.minimum(self._stats[rows, 2],
					stats[:, 2])
				self._stats[rows, 3] = numpy.maximum(self._stats[rows, 3],
					stats[:, 3])
				self._stats[rows, 4] = numpy.maximum(self._stats[rows, 4],
					stats[:, 4])

			compressed, sizes, raw, kinds, where = result[:5]
			if len(where) == 0:
				continue

			offset = self._file.tell()
			self._file.write(compressed[:sizes.sum()])
			total = int(sizes.sum())

			if kind == 'blocks':
				where['offset'] = offset + numpy.cumsum(sizes) - sizes
				self._blocks[g].append(where)
				continue

			self._kinds[g][where] = kinds
			self._sizes[g][where] = sizes
			self._raw[g][where] = raw

			runs = self._runs[g]
			if len(runs) > 0 and runs[-1][2] + runs[-1][3] == offset and \
					runs[-1][1] < where[0] and runs[-1][3] + total < 2 ** 40:
				runs[-1][1] = int(where[-1])
				runs[-1][3] += total
			else:
				runs.append([int(where[0]), int(where[-1]), offset, total])

		del self._inflight[:n_batches]

	def close(self):
		"""Write the open tiles and pending blocks, the tables, the index and
		the header, and close the file. Calling it again does nothing."""

		if self._file is None:
			return

		try:
			for g, open_ in enumerate(self._open):
				if open_ is not None:
					self._encode(g, open_[0], numpy.array([open_[1]]), None, None,
						None, True, False)
					self._open[g] = None
			for t in range(len(self._pending)):
				if self._pending[t] is not None:
					self._flush_track(t)
			self._write(len(self._inflight))

			for i, track in enumerate(self.tracks):
				if track['dtype'] is None:
					sum_, nonzero, low, high, nonint = self._stats[i]
					track['dtype'] = _infer_dtype(sum_, nonzero, low, high,
						nonint > 0)

			chroms_offset = self._file.tell()
			for name, length in self._chrom_list:
				encoded = name.encode()
				self._file.write(struct.pack('<H', len(encoded)) + encoded +
					struct.pack('<Q', length))

			tracks_offset = self._file.tell()
			stats = numpy.zeros(len(self.tracks), dtype=_STATS)
			stats['sum'] = self._stats[:, 0]
			stats['nonzero'] = self._stats[:, 1]
			stats['min'] = numpy.where(self._stats[:, 1] > 0, self._stats[:, 2], 0)
			stats['max'] = numpy.where(self._stats[:, 1] > 0, self._stats[:, 3], 0)
			self._file.write(stats.tobytes())

			metadata_offset = self._file.tell()
			metadata = json.dumps({'genome': self.genome, 'attributes':
				self.attributes, 'order': self.order, 'tracks': self.tracks}).encode()
			self._file.write(metadata)

			index_offset = self._file.tell()
			directory = numpy.zeros(len(self._groups), dtype='<u8')
			position = index_offset + directory.nbytes
			records = []
			for g in range(len(self._groups)):
				if self.order == 'tracks':
					blocks = numpy.concatenate(self._blocks[g]) if self._blocks[g] \
						else numpy.zeros(0, dtype=_BLOCK)
					record = struct.pack('<I', len(blocks)) + blocks.tobytes()
				else:
					runs = numpy.zeros(len(self._runs[g]), dtype=_RUN)
					for r, run in enumerate(self._runs[g]):
						runs[r] = tuple(run)
					record = struct.pack('<I', len(runs)) + runs.tobytes() + \
						self._kinds[g].tobytes() + self._sizes[g].tobytes() + \
						self._raw[g].tobytes()
				directory[g] = position
				position += len(record)
				records.append(record)

			self._file.write(directory.tobytes())
			for record in records:
				self._file.write(record)
			length = self._file.tell()

			header = numpy.zeros(1, dtype=_HEADER)
			header['magic'] = _FIGWIG_MAGIC
			header['version'] = _VERSION
			header['flags'] = _FLAG_TRACKS if self.order == 'tracks' else 0
			header['tile'] = self.tile
			header['codec'] = _CODEC_NONE if self.level == 0 else _CODEC_ZLIB
			header['level'] = self.level
			header['n_chroms'] = len(self._chrom_list)
			header['n_tracks'] = len(self.tracks)
			header['n_groups'] = len(self._groups)
			header['chroms'] = chroms_offset
			header['tracks'] = tracks_offset
			header['index'] = index_offset
			header['data'] = _HEADER.itemsize
			header['metadata'] = metadata_offset
			header['metadata_length'] = len(metadata)
			header['length'] = length
			self._file.seek(0)
			self._file.write(header.tobytes())
		finally:
			self._abandon()


def _items_to_entries(starts, ends, values):
	"""The nonzero bases of one track's windows or intervals, as (positions,
	values), in position order."""

	if values.ndim == 2:
		widths = ends - starts
		inside = numpy.arange(values.shape[1])[None, :] < widths[:, None]
		rows, columns = numpy.nonzero((values != 0) & inside)
		return starts[rows] + columns, values[rows, columns]

	keep = numpy.flatnonzero(values != 0)
	lengths = ends[keep] - starts[keep]
	positions = numpy.repeat(starts[keep], lengths) + numpy.arange(int(
		lengths.sum())) - numpy.repeat(numpy.cumsum(lengths) - lengths, lengths)
	return positions, numpy.repeat(values[keep], lengths)


class FigWigReader:
	"""A figwig file, read many windows and tracks at a time on several
	threads.

	`FigWigReader` reads the per-base values of many windows of one width,
	from any of the file's tracks, in one call, into an array of shape
	(windows, tracks, width): the (batch, channels, length) layout that
	sequence models take. It is built for loading training data from many
	tracks at once, such as the counts of every assay of a cell type under
	a batch of windows. The same read gives the nonzero bases as a sparse
	list with `sparse=True`, or the tracks summed per label with `labels=`,
	which is what a file of thousands of cells is read as.

	The file's tables, metadata and the directory of its index are read when
	it is opened, and each group's index on its first read, after which the
	index is kept for every later read. In position order a read sorts its
	windows, finds the chunks they overlap by division on the tile grid, and
	works through those chunks in batches; in track order it finds each
	requested track's blocks over the windows' chromosomes and searches
	them. Each batch is read from the file, decompressed and decoded
	straight into the windows' rows of the output, on up to `n_jobs`
	threads; zlib's `uncompress()` and the numba decoders run without the
	GIL. Values come back in each track's dtype, or in the one asked for.

	A `FigWigReader` holds no open file between reads, can be read from
	several Python threads at once, and can be pickled, so it can be part of
	a PyTorch Dataset read by DataLoader workers.

	Parameters
	----------
	path: str or os.PathLike
		The path to a figwig file.

	Attributes
	----------
	path: str
		The path the file was opened from.

	chrom_sizes: dict
		The length of each chromosome, keyed by its name, in the file's order.

	tracks: list of dict
		Each track's record: its 'name', 'dtype' and 'group', and the
		attributes it was written with.

	genome: str or None
		The reference genome the file names, or None.

	attributes: dict
		The file's attributes.

	order: str
		'positions' or 'tracks', the layout the file was written in.

	tile: int
		The bases per tile.

	stats: numpy.ndarray, shape=(tracks,)
		A structured array with each track's 'sum', 'nonzero' bases, 'min'
		and 'max'.
	"""

	def __init__(self, path: str | os.PathLike):
		self.path = os.fspath(path)
		error = "{} cannot be read as a figwig file".format(self.path)

		with open(self.path, 'rb') as handle:
			fd = handle.fileno()
			file_size = os.fstat(fd).st_size
			raw = handle.read(_HEADER.itemsize)
			if len(raw) < _HEADER.itemsize:
				raise ValueError(error + ": it is too short to hold a header.")

			header = numpy.frombuffer(raw, dtype=_HEADER)[0]
			if int(header['magic']) != _FIGWIG_MAGIC:
				raise ValueError("{} is not a figwig file.".format(self.path))
			if int(header['version']) != _VERSION:
				raise ValueError(error + ": it is version {}, and this figwig "
					"reads version {}.".format(int(header['version']), _VERSION))

			tile, codec = int(header['tile']), int(header['codec'])
			if not _MIN_TILE <= tile <= _MAX_TILE or tile & (tile - 1) != 0:
				raise ValueError(error + ": its tile size, {}, is not a power of "
					"two from {} to {}.".format(tile, _MIN_TILE, _MAX_TILE))
			if codec not in (_CODEC_NONE, _CODEC_ZLIB):
				raise ValueError(error + ": its chunks are compressed with codec "
					"{}, which this figwig does not know.".format(codec))

			length = int(header['length'])
			if length > file_size or length < _HEADER.itemsize:
				raise ValueError(error + ": the file is {} bytes, and its header "
					"says {}, so it may be truncated.".format(file_size, length))

			offsets = [int(header[key]) for key in ('chroms', 'tracks',
				'metadata', 'index')]
			if any(not _HEADER.itemsize <= o <= length for o in offsets):
				raise ValueError(error + ": a table lies outside the file.")

			n_chroms, n_tracks, n_groups = (int(header[key]) for key in (
				'n_chroms', 'n_tracks', 'n_groups'))
			if n_chroms == 0 or n_tracks == 0 or n_groups == 0:
				raise ValueError(error + ": it has no chromosomes, tracks or "
					"groups.")
			order = 'tracks' if int(header['flags']) & _FLAG_TRACKS else 'positions'
			if order == 'tracks' and n_groups != n_tracks:
				raise ValueError(error + ": a file in track order needs a group "
					"per track.")

			chroms = self._read_chroms(fd, offsets[0], n_chroms, offsets[1],
				error)
			stats = _pread(fd, n_tracks * _STATS.itemsize, offsets[1])
			if len(stats) != n_tracks * _STATS.itemsize:
				raise ValueError(error + ": the file ends inside its track "
					"table.")
			stats = numpy.frombuffer(stats, dtype=_STATS).copy()

			raw = _pread(fd, int(header['metadata_length']), offsets[2])
			try:
				metadata = json.loads(raw.decode())
			except (UnicodeDecodeError, ValueError):
				raise ValueError(error + ": its metadata is not a JSON "
					"document.") from None

			directory = _pread(fd, 8 * n_groups, offsets[3])
			if len(directory) != 8 * n_groups:
				raise ValueError(error + ": the file ends inside its index.")

		tracks = metadata.get('tracks') if isinstance(metadata, dict) else None
		if not isinstance(tracks, list) or len(tracks) != n_tracks:
			raise ValueError(error + ": its metadata does not describe its {} "
				"tracks.".format(n_tracks))

		names = set()
		for track in tracks:
			if not isinstance(track, dict) or not isinstance(track.get('name'),
					str) or track.get('dtype') not in _DTYPES or not isinstance(
					track.get('group'), int) or not 0 <= track['group'] < n_groups:
				raise ValueError(error + ": a track's record is missing its name, "
					"dtype or group.")
			if track['name'] in names:
				raise ValueError(error + ": two tracks are named {!r}.".format(
					track['name']))
			names.add(track['name'])

		self.tracks = tracks
		self.genome = metadata.get('genome')
		self.attributes = metadata.get('attributes') or {}
		self.order = order
		self.stats = stats
		self.tile = tile
		self.chrom_sizes = dict(chroms)
		self._codec = codec
		self._file_size = length
		self._directory = numpy.frombuffer(directory, dtype='<u8').astype(
			numpy.int64)
		self._track_ids = {track['name']: i for i, track in enumerate(tracks)}
		self._group_of = numpy.array([track['group'] for track in tracks],
			dtype=numpy.int64)
		self._groups = [numpy.flatnonzero(self._group_of == g) for g in
			range(n_groups)]
		if any(len(group) == 0 for group in self._groups):
			raise ValueError(error + ": a group has no tracks.")
		self._slot_of = numpy.zeros(n_tracks, dtype=numpy.int64)
		for group in self._groups:
			self._slot_of[group] = numpy.arange(len(group))
		self._dtype_codes = numpy.array([_DTYPES.index(track['dtype']) for
			track in tracks], dtype=numpy.int64)

		self._tids = {name: i for i, (name, _) in enumerate(chroms)}
		self._lengths = numpy.array([length for _, length in chroms],
			dtype=numpy.int64)
		self._tile_offsets, self._n_tiles = _tile_offsets(self._lengths, tile)
		self._index = [None] * n_groups
		self._index_lock = threading.Lock()

	def _read_chroms(self, fd, offset, n, end, error):
		raw = _pread(fd, end - offset, offset)
		chroms, position, names = [], 0, set()
		for _ in range(n):
			if position + 2 > len(raw):
				raise ValueError(error + ": the file ends inside its chromosome "
					"table.")
			size = struct.unpack_from('<H', raw, position)[0]
			if position + 2 + size + 8 > len(raw):
				raise ValueError(error + ": the file ends inside its chromosome "
					"table.")
			try:
				name = raw[position + 2:position + 2 + size].decode()
			except UnicodeDecodeError:
				raise ValueError(error + ": a chromosome name is not UTF-8.") \
					from None
			length = struct.unpack_from('<Q', raw, position + 2 + size)[0]
			if len(name) == 0 or name in names or length == 0:
				raise ValueError(error + ": its chromosome table is not a list of "
					"distinct names and lengths.")
			names.add(name)
			chroms.append((name, int(length)))
			position += 2 + size + 8

		return chroms

	def __repr__(self):
		return "FigWigReader('{}', {} tracks, {} chromosomes, {} order)".format(
			self.path, len(self.tracks), len(self.chrom_sizes), self.order[:-1])

	def __getstate__(self):
		state = self.__dict__.copy()
		del state['_index_lock']
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		self._index_lock = threading.Lock()

	def select(self, **attributes) -> numpy.ndarray:
		"""The indices of the tracks whose records have these values.

		Each keyword is a key of the track records, such as a name or an
		attribute the tracks were written with, and a track is selected
		when every one of them equals the value given.

		Parameters
		----------
		``**attributes``: any
			The keys and the values they must have.

		Returns
		-------
		tracks: numpy.ndarray, dtype=int64
			The indices of the matching tracks, in the file's order, to pass
			to `read` as `tracks`.
		"""

		return numpy.array([i for i, track in enumerate(self.tracks) if all(
			key in track and track[key] == value for key, value in
			attributes.items())], dtype=numpy.int64)

	def _group_index(self, g):
		"""Group g's index, read on first use and kept: in position order the
		kind byte, compressed size, uncompressed size and offset of each
		tile's chunk; in track order the track's blocks."""

		with self._index_lock:
			if self._index[g] is None:
				self._index[g] = self._read_track_index(g) if self.order == \
					'tracks' else self._read_group_index(g)

		return self._index[g]

	def _read_group_index(self, g):
		error = "The index of {} cannot be read".format(self.path)
		n = self._n_tiles
		offset = int(self._directory[g])
		if not _HEADER.itemsize <= offset < self._file_size:
			raise ValueError(error + ": a group's record lies outside the file.")

		with open(self.path, 'rb') as handle:
			fd = handle.fileno()
			raw = _pread(fd, 4, offset)
			if len(raw) != 4:
				raise ValueError(error + ": the file ends inside it.")
			n_runs = struct.unpack('<I', raw)[0]
			size = n_runs * _RUN.itemsize + n + 8 * n
			raw = _pread(fd, size, offset + 4)
			if len(raw) != size:
				raise ValueError(error + ": the file ends inside it.")

		runs = numpy.frombuffer(raw[:n_runs * _RUN.itemsize], dtype=_RUN)
		at = n_runs * _RUN.itemsize
		kinds = numpy.frombuffer(raw[at:at + n], dtype=numpy.uint8).copy()
		sizes = numpy.frombuffer(raw[at + n:at + 5 * n], dtype='<u4').astype(
			numpy.int64)
		raws = numpy.frombuffer(raw[at + 5 * n:at + 9 * n], dtype='<u4').astype(
			numpy.int64)

		if ((kinds >> 2) > 7).any():
			raise ValueError(error + ": a chunk has a kind this figwig does not "
				"know.")
		if ((kinds == 0) != (sizes == 0)).any() or ((kinds == 0) != (raws == 0)
				).any():
			raise ValueError(error + ": a chunk's kind and size disagree.")

		offsets = numpy.zeros(n, dtype=numpy.int64)
		covered = numpy.zeros(n, dtype=numpy.bool_)
		limit = numpy.uint64(self._file_size)
		for run in runs:
			first, last = int(run['first']), int(run['last'])
			if first > last or last >= n or covered[first:last + 1].any() or \
					run['offset'] > limit or run['bytes'] > limit - run['offset']:
				raise ValueError(error + ": a run of chunks is out of order or "
					"lies outside the file.")

			chunk_sizes = sizes[first:last + 1]
			if int(chunk_sizes.sum()) != int(run['bytes']):
				raise ValueError(error + ": a run's chunks do not add up to its "
					"bytes.")

			covered[first:last + 1] = True
			offsets[first:last + 1] = int(run['offset']) + numpy.cumsum(
				chunk_sizes) - chunk_sizes

		if (sizes[~covered] != 0).any():
			raise ValueError(error + ": a chunk lies in no run.")

		return kinds, sizes, raws, offsets

	def _read_track_index(self, t):
		error = "The index of {} cannot be read".format(self.path)
		offset = int(self._directory[t])
		if not _HEADER.itemsize <= offset < self._file_size:
			raise ValueError(error + ": a track's record lies outside the file.")

		with open(self.path, 'rb') as handle:
			fd = handle.fileno()
			raw = _pread(fd, 4, offset)
			if len(raw) != 4:
				raise ValueError(error + ": the file ends inside it.")
			n_blocks = struct.unpack('<I', raw)[0]
			raw = _pread(fd, n_blocks * _BLOCK.itemsize, offset + 4)
			if len(raw) != n_blocks * _BLOCK.itemsize:
				raise ValueError(error + ": the file ends inside it.")

		blocks = numpy.frombuffer(raw, dtype=_BLOCK).copy()
		if (blocks['chrom'] >= len(self._lengths)).any() or (blocks['n'] == 0
				).any() or (blocks['first'] > blocks['last']).any() or (blocks[
				'last'] >= self._lengths[numpy.minimum(blocks['chrom'], len(
				self._lengths) - 1)]).any() or ((blocks['kind'] >> 2) > 7).any():
			raise ValueError(error + ": a block is out of range.")
		if (blocks['offset'].astype(numpy.int64) + blocks['size'] > self.
				_file_size).any():
			raise ValueError(error + ": a block lies outside the file.")
		keys = (blocks['chrom'].astype(numpy.int64) << 32) | blocks['first']
		if (keys[1:] <= keys[:-1]).any() or ((blocks['chrom'][1:] == blocks[
				'chrom'][:-1]) & (blocks['first'][1:] <= blocks['last'][:-1])).any():
			raise ValueError(error + ": a track's blocks are out of order.")
		return blocks

	def read(self, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
		numpy.ndarray, width: int, tracks: str | int | list | None = None,
		out: numpy.ndarray | None = None, dtype=None, n_jobs: int = 8,
		sparse: bool = False, labels: list | numpy.ndarray | None = None
		) -> numpy.ndarray | SparseWindows:
		"""Read the per-base values of many windows of one width.

		Window j is [starts[j], starts[j] + width) on chroms[j], and its
		values go to row j of the output whatever order the windows are
		given in. The output has one channel for each track in `tracks`, in
		that order, and shape (n, tracks, width), or (n, width) when `tracks`
		is one name or index.

		With `sparse=True` the read returns the windows' nonzero bases as a
		`SparseWindows` of entries instead, which is what a read of many
		tracks that are mostly zero, such as single cells, should ask for.
		With `labels`, one integer per track read, it returns the tracks
		summed per label, as (n, labels, width), which turns cells into
		pseudobulks in one read; a label of -1 leaves a track out.

		A base that was not written is 0. A base past the end of its
		chromosome is NaN in a float output and 0 in an integer one. A dense
		output of more than 8 GiB raises unless `out` is given, since a read
		of thousands of tracks wants `sparse=True` or `labels`.

		Parameters
		----------
		chroms: str, list of str, or numpy.ndarray of str
			The chromosome of each window, or one name for every window.
			Each must be one of the file's chromosomes.

		starts: list of int or numpy.ndarray of int, shape=(n,)
			The start of each window, inclusive and base-0.

		width: int
			The length of every window, in bases.

		tracks: str, int, list of str or int, or None, optional
			The tracks to read, by name or index, such as what `select`
			returns. None is every track, in the file's order. Default is
			None.

		out: numpy.ndarray or None, optional
			A writeable, C-contiguous array to write the values into, of the
			output's shape and dtype, such as a view of a pinned torch
			tensor. Default is None.

		dtype: numpy dtype or None, optional
			The type of the output. None is the tracks' own dtype when they
			share one, float32 when any of them is float32, and otherwise
			the smallest integer type that holds all of them, and float32 for
			sums per label. An integer type that cannot hold a track's values
			raises. Default is None.

		n_jobs: int, optional
			The most threads to use, or -1 for one per CPU. Default is 8.

		sparse: bool, optional
			Whether to return the nonzero bases as entries instead of a dense
			array. Default is False.

		labels: list or numpy.ndarray of int, or None, optional
			One label per track read, from 0, or -1 to leave the track out,
			to sum the tracks of each label into one channel. Default is None.

		Returns
		-------
		out: numpy.ndarray or SparseWindows
			The value of every base of every window in every track, dense;
			or the nonzero bases as entries with `sparse`; or the sums per
			label with `labels`.
		"""

		n_jobs = _check_n_jobs(n_jobs)
		ids, single = self._select(tracks)
		windows = _Windows(chroms, starts, width)
		n, width = windows.n, windows.width

		unknown = [name for name in windows.names if name not in self._tids]
		if len(unknown) > 0:
			raise ValueError("chromosome {!r} is not one of the file's "
				"chromosomes.".format(unknown[0]))
		windows.check_range()
		if sparse and labels is not None:
			raise ValueError("sparse and labels cannot both be given.")
		if sparse and out is not None:
			raise ValueError("out cannot be given with sparse=True.")

		if labels is not None:
			labels = numpy.asarray(labels)
			if labels.shape != (len(ids),) or labels.dtype.kind not in 'iu':
				raise ValueError("labels must hold one integer per track read, "
					"shape ({},), not {} of {}.".format(len(ids), labels.shape,
					labels.dtype))
			labels = labels.astype(numpy.int64)
			if len(ids) != len(set(ids.tolist())):
				raise ValueError("a track is named twice in tracks; with labels "
					"each track is read once.")
			if labels.size and labels.min() < -1:
				raise ValueError("labels must be -1 or from 0 upward.")
			n_labels = int(labels.max()) + 1 if labels.size else 0
			dtype = numpy.dtype(numpy.float32) if dtype is None else \
				self._dtype(ids, dtype, check=False)
			shape = (n, n_labels, width)
			single = False
		else:
			dtype = self._dtype(ids, dtype)
			shape = (n, width) if single else (n, len(ids), width)

		if not sparse:
			if out is None:
				n_bytes = int(numpy.prod(shape)) * dtype.itemsize
				if n_bytes > _MAX_DENSE_BYTES:
					raise ValueError("the dense output would take {:.1f} GiB; read "
						"fewer windows or tracks at a time, pass out=, or ask for "
						"sparse=True or labels=.".format(n_bytes / 2 ** 30))
				# Blocks and sums only add nonzero bases, so their output starts
				# from zero; chunks set every base of a window.
				zeroed = labels is not None or self.order == 'tracks'
				out = numpy.zeros(shape, dtype=dtype) if zeroed else numpy.empty(
					shape, dtype=dtype)
			elif not isinstance(out, numpy.ndarray):
				raise TypeError("out must be a numpy array, not {}.".format(
					type(out).__name__))
			elif out.shape != shape or out.dtype != dtype or not out.flags[
					'C_CONTIGUOUS'] or not out.flags['WRITEABLE']:
				raise ValueError("out must be a writeable, C-contiguous {} array "
					"of shape {}.".format(dtype, shape))
			elif labels is not None or self.order == 'tracks':
				out[...] = 0

		if n == 0:
			if sparse:
				empty = numpy.zeros(0, dtype=numpy.int64)
				return SparseWindows(empty, empty, empty, numpy.zeros(0,
					dtype=dtype), (0, len(ids), width))
			return out

		out3 = None if sparse else out.reshape(n, 1, width) if single else out
		tids = numpy.array([self._tids[name] for name in windows.names],
			dtype=numpy.int64)[windows.codes]
		starts = windows.starts
		lengths = self._lengths[tids]

		# A track asked for twice is read once and copied to its other
		# channels after the read; with labels every track is its own
		# channel's label.
		first_channel, copies = {}, []
		for channel, i in enumerate(ids.tolist()):
			if i in first_channel:
				copies.append((channel, first_channel[i]))
			else:
				first_channel[i] = channel
		if labels is not None:
			first_channel = {i: int(labels[c]) for i, c in first_channel.items()
				if labels[c] >= 0}

		mode = 'list' if sparse else 'sum' if labels is not None else 'dense'
		if self.order == 'tracks':
			tasks = self._plan_tracks(first_channel, tids, starts, lengths, width,
				out3, mode, n_jobs)
		else:
			tasks = []
			read_ids = numpy.array(list(first_channel), dtype=numpy.int64)
			channels = numpy.array(list(first_channel.values()), dtype=numpy.int64)
			for g in numpy.unique(self._group_of[read_ids]).tolist():
				mine = self._group_of[read_ids] == g
				slots = numpy.full(len(self._groups[g]), -1, dtype=numpy.int64)
				slots[self._slot_of[read_ids[mine]]] = channels[mine]
				tasks.extend(self._plan(g, slots, tids, starts, lengths, width,
					out3, mode, n_jobs))

		self._run(tasks, n_jobs, out3 if mode == 'sum' else None)

		if sparse:
			parts = [task.entries for task in tasks if task.entries is not None]
			if parts:
				rows = numpy.concatenate([p[0] for p in parts])
				channels = numpy.concatenate([p[1] for p in parts])
				offsets = numpy.concatenate([p[2] for p in parts])
				values = numpy.concatenate([p[3] for p in parts])
			else:
				rows = channels = offsets = numpy.zeros(0, dtype=numpy.int64)
				values = numpy.zeros(0, dtype=numpy.float64)
			if copies:
				extra = [(numpy.full(int((channels == source).sum()), channel,
					dtype=numpy.int64), source) for channel, source in copies]
				rows = numpy.concatenate([rows] + [rows[channels == s] for _, s in
					extra])
				offsets = numpy.concatenate([offsets] + [offsets[channels == s] for
					_, s in extra])
				values = numpy.concatenate([values] + [values[channels == s] for
					_, s in extra])
				channels = numpy.concatenate([channels] + [c for c, _ in extra])
			# One key per entry sorts several times faster than three keys.
			key = (rows * len(ids) + channels) * width + offsets
			order = numpy.argsort(key)
			return SparseWindows(rows[order], channels[order], offsets[order],
				values[order].astype(dtype), (n, len(ids), width))

		for channel, source in copies:
			out3[:, channel] = out3[:, source]

		# Bases past the end of a chromosome lie in no tile or block.
		over = numpy.flatnonzero(starts + width > lengths)
		if len(over) > 0:
			fill = numpy.nan if dtype.kind == 'f' else 0
			for j in over.tolist():
				out3[j, :, max(0, int(lengths[j] - starts[j])):] = fill

		return out

	def read_entries(self, tracks: str | int | list | None = None, chroms: str |
		list[str] | None = None, n_jobs: int = 8) -> SparseEntries:
		"""Every nonzero base of tracks over whole chromosomes, as entries.

		This is the read a file in track order is laid out for: one track's
		cuts over the genome, or a few tracks', as `SparseEntries` sorted by
		chromosome, track and position. In position order it decodes every
		chunk of the tracks' groups on those chromosomes.

		Parameters
		----------
		tracks: str, int, list of str or int, or None, optional
			The tracks, by name or index. None is every track. Default is
			None.

		chroms: str, list of str, or None, optional
			The chromosomes, by name. None is every chromosome, in the file's
			order. Default is None.

		n_jobs: int, optional
			The most threads to use, or -1 for one per CPU. Default is 8.

		Returns
		-------
		entries: SparseEntries
			The chromosome index, position, track index and value of every
			nonzero base.
		"""

		ids, _ = self._select(tracks)
		if chroms is None:
			names = list(self.chrom_sizes)
		else:
			names = [chroms] if isinstance(chroms, str) else list(chroms)
			for name in names:
				if name not in self._tids:
					raise ValueError("chromosome {!r} is not one of the file's "
						"chromosomes.".format(name))

		parts = []
		for name in names:
			length = self.chrom_sizes[name]
			found = self.read(name, [0], length, tracks=ids.tolist(), n_jobs=n_jobs,
				sparse=True)
			parts.append((numpy.full(len(found.offsets), self._tids[name],
				dtype=numpy.int64), found.offsets, ids[found.channels],
				found.values))

		if not parts:
			empty = numpy.zeros(0, dtype=numpy.int64)
			return SparseEntries(empty, empty, empty, numpy.zeros(0))

		return SparseEntries(*[numpy.concatenate([p[k] for p in parts]) for k
			in range(4)])

	def _select(self, tracks):
		if tracks is None:
			return numpy.arange(len(self.tracks)), False

		single = isinstance(tracks, (str, int, numpy.integer)) and \
			not isinstance(tracks, bool)
		items = [tracks] if single else tracks
		if isinstance(items, (str, dict)) or not hasattr(items, '__iter__'):
			raise TypeError("tracks must be a name, an index, or a list of "
				"them, not {}.".format(type(tracks).__name__))

		ids = []
		for item in items:
			if isinstance(item, str):
				if item not in self._track_ids:
					raise ValueError("{!r} is not one of the file's tracks."
						.format(item))
				ids.append(self._track_ids[item])
			elif isinstance(item, (int, numpy.integer)) and not isinstance(item,
					bool):
				if not 0 <= item < len(self.tracks):
					raise ValueError("track {} is out of range; the file has {} "
						"tracks.".format(item, len(self.tracks)))
				ids.append(int(item))
			else:
				raise TypeError("tracks must be names or indices, not {}.".format(
					type(item).__name__))

		if len(ids) == 0:
			raise ValueError("tracks must name at least one track.")

		return numpy.array(ids, dtype=numpy.int64), single

	def _dtype(self, ids, dtype, check=True):
		"""The output dtype for these tracks, checked against their values."""

		# The tracks' dtypes as codes into _DTYPES, so that a read of
		# thousands of tracks does not build a dtype for each.
		codes = numpy.unique(self._dtype_codes[ids])
		own = [numpy.dtype(_DTYPES[c]) for c in codes.tolist()]
		if dtype is None:
			if len(own) == 1:
				dtype = own[0]
			elif any(d.kind == 'f' for d in own):
				dtype = numpy.dtype(numpy.float32)
			else:
				dtype = numpy.result_type(*own)
		else:
			try:
				dtype = numpy.dtype(dtype)
			except TypeError:
				raise TypeError("dtype is not a dtype: {!r}.".format(dtype)) \
					from None
			if dtype.kind not in 'iuf':
				raise ValueError("dtype must be an integer or float type, not {}."
					.format(dtype))

		if check and dtype.kind in 'iu':
			info = numpy.iinfo(dtype)
			floats = numpy.flatnonzero(self._dtype_codes[ids] == _DTYPES.index(
				'float32'))
			if len(floats) > 0:
				raise ValueError("track {!r} is float32 and cannot be read as {}."
					.format(self.tracks[int(ids[floats[0]])]['name'], dtype))
			outside = numpy.flatnonzero((self.stats['min'][ids] < info.min) | (
				self.stats['max'][ids] > info.max))
			if len(outside) > 0:
				i = int(ids[outside[0]])
				raise ValueError("track {!r} has values from {} to {}, which {} "
					"does not hold.".format(self.tracks[i]['name'], self.stats[i][
					'min'], self.stats[i]['max'], dtype))

		return dtype

	def _plan(self, g, slots, tids, starts, lengths, width, out, mode, n_jobs):
		"""The tasks that read group g's part of the windows: each a batch
		of chunks and the (window, tile) pairs that fall in them."""

		tile = self.tile
		kinds, sizes, raws, offsets = self._group_index(g)

		# The tiles each window overlaps, clipped to its chromosome.
		stops = numpy.minimum(starts + width, lengths)
		k0 = starts // tile
		k1 = (stops - 1) // tile
		counts = numpy.maximum(k1 - k0 + 1, 0)
		counts[stops <= starts] = 0
		total = int(counts.sum())
		if total == 0:
			return []

		rows = numpy.repeat(numpy.arange(len(starts)), counts)
		local = numpy.repeat(k0, counts) + numpy.arange(total) - numpy.repeat(
			numpy.cumsum(counts) - counts, counts)
		gtiles = self._tile_offsets[tids[rows]] + local
		order = numpy.argsort(gtiles, kind='stable')
		rows, local, gtiles = rows[order], local[order], gtiles[order]

		unique, first = numpy.unique(gtiles, return_index=True)
		inverse = numpy.repeat(numpy.arange(len(unique)), numpy.diff(
			numpy.append(first, total)))
		chunk_kinds = kinds[unique]
		live = chunk_kinds != 0
		bounds = raws[unique] + 8

		# Batches of chunks, cut where a batch would hold too many bytes
		# before compression or too many chunks: at most _BATCH_CHUNKS, and
		# fewer when that would give the threads too few batches to share.
		most = _batch_chunks(int(live.sum()), n_jobs)
		tasks = []
		a = 0
		while a < len(unique):
			total_bytes, n_live, b = 0, 0, a
			while b < len(unique) and (b == a or (total_bytes + bounds[b] <=
					_BATCH_BYTES and n_live < most)):
				total_bytes += int(bounds[b])
				n_live += int(live[b])
				b += 1

			p0, p1 = int(first[a]), int(first[b]) if b < len(unique) else total
			chunk = numpy.where(live[inverse[p0:p1]], inverse[p0:p1] - a, -1)
			tasks.append(_Task(self, g, slots, unique[a:b][live[a:b]],
				chunk_kinds[a:b][live[a:b]], offsets[unique[a:b][live[a:b]]],
				sizes[unique[a:b][live[a:b]]], int(total_bytes),
				_compact(chunk, live[a:b]), local[p0:p1] * tile, starts[rows[p0:p1]],
				rows[p0:p1], out, width, mode))
			a = b

		return tasks

	def _plan_tracks(self, channels, tids, starts, lengths, width, out, mode,
		n_jobs):
		"""The tasks that read windows from a file in track order: each a
		batch of blocks and the (window, block, channel) pairs."""

		pairs = []
		for i, channel in channels.items():
			blocks = self._group_index(i)
			if len(blocks) == 0:
				continue
			for tid in numpy.unique(tids).tolist():
				mine = numpy.flatnonzero(blocks['chrom'] == tid)
				windows = numpy.flatnonzero(tids == tid)
				if len(mine) == 0 or len(windows) == 0:
					continue
				firsts = blocks['first'][mine].astype(numpy.int64)
				lasts = blocks['last'][mine].astype(numpy.int64)
				stops = numpy.minimum(starts[windows] + width, lengths[windows])
				lo = numpy.searchsorted(lasts, starts[windows], side='left')
				hi = numpy.searchsorted(firsts, stops, side='left')
				counts = numpy.maximum(hi - lo, 0)
				total = int(counts.sum())
				if total == 0:
					continue
				rows = numpy.repeat(windows, counts)
				which = numpy.repeat(lo, counts) + numpy.arange(total) - \
					numpy.repeat(numpy.cumsum(counts) - counts, counts)
				pairs.append((numpy.full(total, i, dtype=numpy.int64), mine[which],
					rows, numpy.full(total, channel, dtype=numpy.int64)))

		if not pairs:
			return []

		track_of = numpy.concatenate([p[0] for p in pairs])
		block_of = numpy.concatenate([p[1] for p in pairs])
		rows = numpy.concatenate([p[2] for p in pairs])
		slots = numpy.concatenate([p[3] for p in pairs])

		# Each block once, batched by its uncompressed bytes.
		keys = track_of * (1 << 40) + block_of
		unique, first, inverse = numpy.unique(keys, return_index=True,
			return_inverse=True)
		order = numpy.argsort(inverse, kind='stable')
		rows, slots, inverse = rows[order], slots[order], inverse[order]
		u_tracks, u_blocks = track_of[first], block_of[first]
		records = numpy.concatenate([self._group_index(int(t))[b:b + 1] for t, b
			in zip(u_tracks.tolist(), u_blocks.tolist())])
		bounds = records['raw'].astype(numpy.int64) + 8
		edges = numpy.concatenate([numpy.flatnonzero(numpy.diff(inverse,
			prepend=-1)), [len(inverse)]])

		most = _batch_chunks(len(unique), n_jobs)
		tasks = []
		a = 0
		while a < len(unique):
			total_bytes, b = 0, a
			while b < len(unique) and (b == a or (total_bytes + bounds[b] <=
					_BATCH_BYTES and b - a < most)):
				total_bytes += int(bounds[b])
				b += 1
			p0, p1 = int(edges[a]), int(edges[b])
			tasks.append(_BlockTask(self, records[a:b], int(total_bytes),
				inverse[p0:p1] - a, slots[p0:p1], starts[rows[p0:p1]], rows[p0:p1],
				out, width, mode))
			a = b

		return tasks

	def _run(self, tasks, n_jobs, summed=None):
		"""Run the tasks on up to `n_jobs` threads, the first on this thread
		when a kernel has not been compiled yet, so that each is compiled
		once before any thread starts.

		Tasks that sum into `summed`, the output of a read with labels, may
		add to the same cells from different groups or tracks, so on several
		threads each thread sums into its own copy, and the copies are added
		to `summed` at the end.
		"""

		if len(tasks) == 0:
			return

		kernels = (_read_tiles, _check_chunks, _list_tiles, _read_blocks,
			_check_blocks, _inflate_chunks)
		first = []
		if any(not getattr(kernel, 'signatures', True) for kernel in kernels):
			first.append(tasks[0])

		rest = tasks[len(first):]
		with open(self.path, 'rb') as handle:
			fd = handle.fileno()
			for task in first:
				task.run(fd)

			if n_jobs == 1 or len(rest) <= 1:
				for task in rest:
					task.run(fd)
			elif summed is None:
				with ThreadPoolExecutor(min(n_jobs, len(rest))) as pool:
					list(pool.map(lambda task: task.run(fd), rest))
			else:
				n_threads = min(n_jobs, len(rest))
				buffers = [numpy.zeros_like(summed) for _ in range(n_threads)]

				def run_share(k):
					for task in rest[k::n_threads]:
						task.out = buffers[k]
						task.run(fd)

				with ThreadPoolExecutor(n_threads) as pool:
					list(pool.map(run_share, range(n_threads)))
				for buffer in buffers:
					summed += buffer


def _batch_chunks(n, n_jobs):
	"""The most chunks or blocks in one batch of a read of n of them on
	n_jobs threads: _BATCH_CHUNKS, or fewer so that each thread gets about
	_BATCHES_PER_THREAD batches to balance the work, but at least
	_MIN_BATCH_CHUNKS."""

	return max(_MIN_BATCH_CHUNKS, min(_BATCH_CHUNKS, -(-n // (
		_BATCHES_PER_THREAD * n_jobs))))


def _compact(chunk, live):
	"""Chunk positions among a batch's live chunks, from positions among all
	of its chunks, with -1 kept for the zero ones."""

	mapping = numpy.cumsum(live) - 1
	return numpy.where(chunk >= 0, mapping[numpy.maximum(chunk, 0)], -1)


def _load_batch(reader, fd, offsets, sizes, bound):
	"""A batch's chunks or blocks, read with one pread per run of adjacent
	ones and inflated into one buffer, as (data, positions, lengths)."""

	order = numpy.argsort(offsets, kind='stable')
	o, s = offsets[order], sizes[order]
	breaks = numpy.flatnonzero(o[1:] != o[:-1] + s[:-1]) + 1
	breaks = [0] + breaks.tolist() + [len(o)]

	ends = numpy.cumsum(s)
	starts = ends - s
	raw = numpy.empty(int(s.sum()) + 8, dtype=numpy.uint8)
	for r0, r1 in zip(breaks[:-1], breaks[1:]):
		position, end = int(starts[r0]), int(ends[r1 - 1])
		got = _pread_into(fd, raw[position:end], int(o[r0]))
		if got < end - position:
			raise ValueError("{} ends inside a chunk; the file is truncated."
				.format(reader.path))

	# Back to the batch's own order.
	starts_in = numpy.empty_like(starts)
	starts_in[order] = starts
	sizes_in = sizes.astype(numpy.int64)

	if reader._codec == _CODEC_NONE:
		return raw, starts_in, sizes_in

	# libdeflate inflates about twice as fast as zlib, when the deflate
	# package is installed; zlib's uncompress() is next, and the zlib module
	# takes any chunk neither handles, or every chunk when neither loads.
	positions = numpy.zeros(len(sizes), dtype=numpy.int64)
	lengths = numpy.zeros(len(sizes), dtype=numpy.int64)
	inflate = _libdeflate_inflate()
	uncompress = _zlib_uncompress()
	used = 0
	buffer = numpy.zeros(0, dtype=numpy.uint8)
	if inflate is not None:
		alloc, decompress, free = inflate
		decompressor = alloc()
		if decompressor == 0:
			raise MemoryError("libdeflate could not allocate a decompressor.")

		buffer = numpy.empty(bound + 8 * len(sizes), dtype=numpy.uint8)
		try:
			used = _inflate_chunks_libdeflate(decompress, decompressor, raw,
				starts_in, sizes_in, buffer, positions, lengths, numpy.zeros(1,
				dtype=numpy.uintp))
		finally:
			free(decompressor)
	elif uncompress is not None:
		function, ulong = uncompress
		buffer = numpy.empty(bound + 8 * len(sizes), dtype=numpy.uint8)
		used = _inflate_chunks(function, raw, starts_in, sizes_in, buffer,
			positions, lengths, numpy.zeros(1, dtype=ulong))
	else:
		positions[:] = -1

	parts = [buffer[:used]]
	position = used
	for k in numpy.flatnonzero(positions < 0).tolist():
		stream = raw[starts_in[k]:starts_in[k] + sizes_in[k]].tobytes()
		try:
			block = zlib.decompressobj().decompress(stream, bound + 1)
		except zlib.error:
			block = b''

		padded = block + bytes(-len(block) % 8)
		parts.append(numpy.frombuffer(padded, dtype=numpy.uint8))
		positions[k], lengths[k] = position, len(block)
		position += len(padded)

	data = parts[0] if len(parts) == 1 else numpy.concatenate(parts)
	return data, positions, lengths


class _Task:
	"""One batch of a position-order read: a group's live chunks and the
	(window, tile) pairs that fall in them, including pairs on zero tiles."""

	def __init__(self, reader, g, slots, gtiles, kinds, offsets, sizes, bound,
		pair_chunks, pair_bases, pair_starts, pair_rows, out, width, mode):
		self.reader, self.g, self.slots = reader, g, slots
		self.gtiles, self.kinds = gtiles, kinds
		self.offsets, self.sizes, self.bound = offsets, sizes, bound
		self.pair_chunks = numpy.ascontiguousarray(pair_chunks, dtype=numpy.int64)
		self.pair_bases = numpy.ascontiguousarray(pair_bases, dtype=numpy.int64)
		self.pair_starts = numpy.ascontiguousarray(pair_starts, dtype=numpy.int64)
		self.pair_rows = numpy.ascontiguousarray(pair_rows, dtype=numpy.int64)
		self.out, self.width, self.mode = out, width, mode
		self.entries = None

	def run(self, fd):
		reader = self.reader
		n = len(self.gtiles)
		if n > 0:
			data, positions, lengths = _load_batch(reader, fd, self.offsets,
				self.sizes, self.bound)
			failed = numpy.zeros(n, dtype=numpy.int64)
			if _check_chunks(reader.tile, data, positions, lengths, self.kinds,
					len(self.slots), self.slots, failed) > 0:
				k = int(numpy.argmax(failed))
				raise ValueError("a chunk of {} cannot be read: tile {} of the "
					"group of {!r} is corrupt, or the file is truncated.".format(
					reader.path, int(self.gtiles[k]), reader.tracks[reader._groups[
					self.g][0]]['name']))
		else:
			data = numpy.zeros(8, dtype=numpy.uint8)
			positions = numpy.zeros(0, dtype=numpy.int64)

		if self.mode == 'list':
			empty = numpy.zeros(0, dtype=numpy.int64)
			count = _list_tiles(reader.tile, data, positions, self.kinds, len(
				self.slots), self.slots, self.pair_chunks, self.pair_bases,
				self.pair_starts, self.pair_rows, self.width, False, empty, empty,
				empty, numpy.zeros(0, dtype=numpy.float64))
			rows = numpy.empty(count, dtype=numpy.int64)
			channels = numpy.empty(count, dtype=numpy.int64)
			offsets = numpy.empty(count, dtype=numpy.int64)
			values = numpy.empty(count, dtype=numpy.float64)
			_list_tiles(reader.tile, data, positions, self.kinds, len(self.slots),
				self.slots, self.pair_chunks, self.pair_bases, self.pair_starts,
				self.pair_rows, self.width, True, rows, channels, offsets, values)
			self.entries = (rows, channels, offsets, values)
			return

		_read_tiles(reader.tile, data, positions, self.kinds, len(self.slots),
			self.slots, self.pair_chunks, self.pair_bases, self.pair_starts,
			self.pair_rows, self.out, self.mode == 'sum')


class _BlockTask:
	"""One batch of a track-order read: blocks of some tracks and the
	(window, block, channel) pairs that may fall in them."""

	def __init__(self, reader, records, bound, pair_blocks, pair_slots,
		pair_starts, pair_rows, out, width, mode):
		self.reader, self.records, self.bound = reader, records, bound
		self.pair_blocks = numpy.ascontiguousarray(pair_blocks, dtype=numpy.int64)
		self.pair_slots = numpy.ascontiguousarray(pair_slots, dtype=numpy.int64)
		self.pair_starts = numpy.ascontiguousarray(pair_starts, dtype=numpy.int64)
		self.pair_rows = numpy.ascontiguousarray(pair_rows, dtype=numpy.int64)
		self.out, self.width, self.mode = out, width, mode
		self.entries = None

	def run(self, fd):
		reader = self.reader
		records = self.records
		data, positions, lengths = _load_batch(reader, fd, records['offset']
			.astype(numpy.int64), records['size'].astype(numpy.int64), self.bound)
		kinds = records['kind'].astype(numpy.uint8)
		n_entries = records['n'].astype(numpy.int64)
		failed = numpy.zeros(len(records), dtype=numpy.int64)
		if _check_blocks(data, positions, lengths, kinds, n_entries, failed) > 0:
			raise ValueError("a block of {} cannot be read: it is corrupt, or the "
				"file is truncated.".format(reader.path))

		empty = numpy.zeros(0, dtype=numpy.int64)
		out = self.out if self.out is not None else numpy.zeros((1, 1, 1),
			dtype=numpy.float32)
		if self.mode == 'list':
			count = _read_blocks(data, positions, kinds, n_entries, self.pair_blocks,
				self.pair_slots, self.pair_starts, self.pair_rows, self.width, out,
				False, True, False, empty, empty, empty, numpy.zeros(0,
				dtype=numpy.float64))
			rows = numpy.empty(count, dtype=numpy.int64)
			channels = numpy.empty(count, dtype=numpy.int64)
			offsets = numpy.empty(count, dtype=numpy.int64)
			values = numpy.empty(count, dtype=numpy.float64)
			_read_blocks(data, positions, kinds, n_entries, self.pair_blocks,
				self.pair_slots, self.pair_starts, self.pair_rows, self.width, out,
				False, True, True, rows, channels, offsets, values)
			self.entries = (rows, channels, offsets, values)
			return

		_read_blocks(data, positions, kinds, n_entries, self.pair_blocks,
			self.pair_slots, self.pair_starts, self.pair_rows, self.width, out,
			self.mode == 'sum', False, False, empty, empty, empty, numpy.zeros(0,
			dtype=numpy.float64))


def read_figwig(figwig: str | os.PathLike | FigWigReader, chroms: str |
	list[str] | numpy.ndarray, starts: list[int] | numpy.ndarray, width: int,
	tracks: str | int | list | None = None, out: numpy.ndarray | None = None,
	dtype=None, n_jobs: int = 8, sparse: bool = False, labels: list |
	numpy.ndarray | None = None) -> numpy.ndarray | SparseWindows:
	"""Read the per-base values of many windows of one width from a figwig
	file.

	This reads the windows as `FigWigReader.read` does, from a path or an
	open reader, into an array of shape (n, tracks, width), or (n, width)
	when `tracks` is one name or index, or as entries with `sparse=True`, or
	summed per label with `labels`. A path is opened on every call; to read
	the same file more than once, as a data loader does, open it with
	`FigWigReader` and pass that, which keeps its index.


	Parameters
	----------
	figwig: str, os.PathLike, or FigWigReader
		The figwig file, as a path or an open reader.

	chroms: str, list of str, or numpy.ndarray of str
		The chromosome of each window, or one name for every window.

	starts: list of int or numpy.ndarray of int, shape=(n,)
		The start of each window, inclusive and base-0.

	width: int
		The length of every window, in bases.

	tracks: str, int, list of str or int, or None, optional
		The tracks to read; see `FigWigReader.read`. Default is None.

	out: numpy.ndarray or None, optional
		An array to write the values into; see `FigWigReader.read`. Default
		is None.

	dtype: numpy dtype or None, optional
		The type of the output; see `FigWigReader.read`. Default is None.

	n_jobs: int, optional
		The most threads to use, or -1 for one per CPU. Default is 8.

	sparse: bool, optional
		Whether to return the nonzero bases as entries; see
		`FigWigReader.read`. Default is False.

	labels: list or numpy.ndarray of int, or None, optional
		One label per track read, to sum tracks per label; see
		`FigWigReader.read`. Default is None.


	Returns
	-------
	out: numpy.ndarray or SparseWindows
		The value of every base of every window in every track, or the
		entries, or the sums per label.
	"""

	if not isinstance(figwig, (str, os.PathLike, FigWigReader)):
		raise TypeError("figwig must be a path or a FigWigReader, not {}.".format(
			type(figwig).__name__))

	reader = figwig if isinstance(figwig, FigWigReader) else FigWigReader(figwig)
	return reader.read(chroms, starts, width, tracks=tracks, out=out,
		dtype=dtype, n_jobs=n_jobs, sparse=sparse, labels=labels)


def write_figwig(path: str | os.PathLike, chrom_sizes: dict | list,
	tracks: list, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
	numpy.ndarray, values, ends: list[int] | numpy.ndarray | None = None,
	genome: str | None = None, attributes: dict | None = None,
	tile: int = _DEFAULT_TILE, groups: int | list = _DEFAULT_GROUPS,
	level: int | None = None, engine: str = 'auto', n_jobs: int = 8,
	order: str = 'positions'):
	"""Write the values of windows, or of intervals, for every track to one
	figwig file.

	This writes the windows or intervals as `FigWigWriter.write` does, in
	one call, from values of shape (n, len(tracks), width), or (n,
	len(tracks)) with `ends`. That is the (batch, channels, length) layout
	that `read_figwig` gives and that sequence models predict, so that

		write_figwig(path, chrom_sizes, tracks, chroms, starts, y)

	writes back what `read_figwig(path, chroms, starts, width)` read, or a
	model's predictions for the same windows. If writing fails part way, the
	file is left without a header, which no reader takes for a figwig.


	Parameters
	----------
	path: str or os.PathLike
		The file to write. It is created, or emptied.

	chrom_sizes: dict or list of (str, int)
		The chromosomes and their lengths; see `FigWigWriter`.

	tracks: list of str or of dict
		The tracks, one per channel of `values`; see `FigWigWriter`.

	chroms: str, list of str, or numpy.ndarray of str
		The chromosome of each window or interval, or one name for all.

	starts: list of int or numpy.ndarray of int, shape=(n,)
		The start of each window or interval.

	values: array-like of numbers
		The value of every base of each window in each track, of shape (n,
		len(tracks), width); or with `ends`, the value of each interval in
		each track, of shape (n, len(tracks)).

	ends: list of int or numpy.ndarray of int, shape=(n,), or None, optional
		The end of each interval. If None, the items are windows. Default is
		None.

	genome: str or None, optional
		The reference genome, as a name or URL; see `FigWigWriter`. Default
		is None.

	attributes: dict or None, optional
		Attributes of the file; see `FigWigWriter`. Default is None.

	tile: int, optional
		The bases per tile; see `FigWigWriter`. Default is 16384.

	groups: int or list of lists of str, optional
		The tracks stored together; see `FigWigWriter`. Default is 8.

	level: int or None, optional
		The compression level; see `FigWigWriter`. Default is None.

	engine: str, optional
		What compresses the chunks; see `FigWigWriter`. Default is 'auto'.

	n_jobs: int, optional
		The most threads that compress chunks at once, besides the calling
		thread, or -1 for one per CPU. Default is 8.

	order: str, optional
		'positions' or 'tracks'; see `FigWigWriter`. Default is 'positions'.
	"""

	with FigWigWriter(path, chrom_sizes, tracks, genome=genome,
			attributes=attributes, tile=tile, groups=groups, level=level,
			engine=engine, n_jobs=n_jobs, order=order) as writer:
		writer.write(chroms, starts, values, ends=ends)
