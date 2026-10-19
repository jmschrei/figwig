# figwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""figwig files: many per-base tracks in one file, read in windows.

A figwig file holds the per-base values of any number of tracks over one
genome, with a JSON record of attributes for each track and for the file,
and is read the way `BigWigReader` reads a bigWig: many windows of one
width at a time, into one array of shape (windows, tracks, width). Its
layout is built for that read. Every track is cut on one grid of tiles of
`tile` bases, so a window's tiles are found by division rather than by a
tree search; the tiles of a group of tracks at one position are one chunk,
so a window read decodes each chunk once for every track in it; a chunk
whose values are all zero takes no space; and the other chunks hold their
values densely or as nonzero bases, in the narrowest integer that holds
them, so that per-base counts come back as the integers they are.

The file is, in order: a 128-byte header; the chunks, as they were written;
the chromosome table; a table of each track's sum, nonzero bases, minimum
and maximum; the metadata, a JSON document holding the genome, the file's
attributes and every track's record; and the index, one record per group
holding the kind byte and compressed size of every tile, and the runs of
tiles that lie back to back in the file. `_figwig_kernels` describes the
chunks.
"""

from __future__ import annotations

import json
import os
import struct
import threading
import zlib

from concurrent.futures import ThreadPoolExecutor

import numpy

from ._figwig_kernels import _check_chunks
from ._figwig_kernels import _encode_dense_tiles
from ._figwig_kernels import _encode_tiles
from ._figwig_kernels import _inflate_chunks
from ._figwig_kernels import _read_tiles
from ._figwig_kernels import _WIDTHS
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

# Tiles are a power of two of bases from 16 to 65536, so that a sparse
# chunk's offsets fit uint16; a group's chunk must fit a uint32 size.
_MIN_TILE, _MAX_TILE = 16, 65536
_DEFAULT_TILE = 16384
_DEFAULT_GROUPS = 8
_MAX_GROUP_BASES = 1 << 29

# The most bytes of chunks before compression that one task encodes or
# decodes, and the most chunks one read task takes.
_BATCH_BYTES = 1 << 25
_BATCH_CHUNKS = 256

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


def _check_tile(tile):
	if isinstance(tile, bool) or not isinstance(tile, (int, numpy.integer)):
		raise TypeError("tile must be an integer.")
	if not _MIN_TILE <= tile <= _MAX_TILE or tile & (tile - 1) != 0:
		raise ValueError("tile must be a power of two from {} to {}, not {}."
			.format(_MIN_TILE, _MAX_TILE, tile))

	return int(tile)


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


class FigWigWriter:
	"""Write a figwig file, a batch of windows or intervals at a time, on
	several threads.

	`FigWigWriter` takes the values of several tracks at once with `write`,
	as the windows `FigWigReader.read` reads, of shape (n, tracks, width),
	or as intervals, and writes them as they come, so that tracks larger
	than memory, such as a model's predictions over a genome, can be written
	a batch at a time. Use it as a context manager, or call `close` when
	every value has been written; the tables and the index are written
	there.

	Tracks are written in groups, whose tiles are stored together: a read
	of any track in a group decodes the whole group's chunk, so the tracks
	of a group should be those read together, such as the two strands of
	an assay, and a file for hundreds of tracks read a few at a time wants
	small groups. Each `write` call covers whole groups, and within a group
	its items come in position order on each chromosome, after the items
	of earlier calls, and must not overlap; a group may move to a new
	chromosome in any order, but cannot return to one it has left. Writing
	several groups in turn on one chromosome is allowed, and costs nothing
	but a longer index.

	Each tile of a group is encoded as the writer reaches it: as nothing
	when every value is zero, and otherwise densely or as nonzero bases,
	whichever takes fewer bytes, in the narrowest integer type that holds
	the tile's values, or float32 where they are not integers. A tile that
	a call fills only part of waits for the next call. The chunks are
	compressed on up to `n_jobs` threads while the calling thread encodes
	the next batch, with zlib or libdeflate, as `BigWigWriter` compresses
	its blocks, and the file does not depend on the number of threads or on
	how the values were split between calls, only on the values and on
	which calls wrote which groups.

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
		The bases per tile, a power of two from 16 to 65536. Smaller tiles
		decode less for each window and compress a little less well. Default
		is 16384.

	groups: int or list of lists of str, optional
		The tracks whose tiles are stored together: a number of consecutive
		tracks per group, or a list of groups of track names that holds every
		track once. Default is 8.

	level: int, optional
		The compression level, as `BigWigWriter` takes it, or 0 to store the
		chunks without compressing them, which decodes fastest. Default is 6.

	engine: str, optional
		What compresses the chunks: 'zlib', 'libdeflate', 'isal' or 'auto';
		see `BigWigWriter`. Default is 'auto'.

	n_jobs: int, optional
		The most threads that compress chunks at once, besides the calling
		thread, or -1 for one per CPU. Default is 8.
	"""

	def __init__(self, path: str | os.PathLike, chrom_sizes: dict | list,
		tracks: list, genome: str | None = None, attributes: dict | None = None,
		tile: int = _DEFAULT_TILE, groups: int | list = _DEFAULT_GROUPS,
		level: int = 6, engine: str = 'auto', n_jobs: int = 8):
		self._file = None
		self._pool = None

		self.chrom_sizes = dict(_check_chrom_sizes(chrom_sizes))
		self._chrom_list = list(self.chrom_sizes.items())
		self._tids = {name: i for i, name in enumerate(self.chrom_sizes)}
		self._lengths = numpy.array(list(self.chrom_sizes.values()),
			dtype=numpy.int64)

		self.tracks = _check_tracks(tracks)
		self._groups = _check_groups(groups, self.tracks)
		self.tile = _check_tile(tile)
		for group in self._groups:
			if len(group) * self.tile > _MAX_GROUP_BASES:
				raise ValueError("a group of {} tracks at a tile of {} bases "
					"holds more than 2**29 values per chunk; use smaller groups "
					"or tiles.".format(len(group), self.tile))

		self._track_ids = {track['name']: i for i, track in enumerate(self.tracks)}
		self._group_of = numpy.zeros(len(self.tracks), dtype=numpy.int64)
		for g, group in enumerate(self._groups):
			self._group_of[group] = g
			for slot, i in enumerate(group):
				self.tracks[i]['group'] = g

		if genome is not None and not isinstance(genome, str):
			raise TypeError("genome must be a string or None, not {}.".format(
				type(genome).__name__))

		self.genome = genome
		self.attributes = _check_attributes(attributes)
		self.engine, self.level = _check_engine(engine, level)
		self.n_jobs = _check_n_jobs(n_jobs)
		self.path = path

		self._tile_offsets, self._n_tiles = _tile_offsets(self._lengths,
			self.tile)

		n_groups = len(self._groups)
		self._kinds = [numpy.zeros(self._n_tiles, dtype=numpy.uint8) for _ in
			range(n_groups)]
		self._sizes = [numpy.zeros(self._n_tiles, dtype=numpy.uint32) for _ in
			range(n_groups)]
		self._runs = [[] for _ in range(n_groups)]
		self._buffers = [None] * n_groups
		self._open = [None] * n_groups
		self._last = [None] * n_groups
		self._done = [set() for _ in range(n_groups)]
		self._stats = numpy.zeros((len(self.tracks), 5), dtype=numpy.float64)
		self._stats[:, 2] = numpy.inf
		self._stats[:, 3] = -numpy.inf
		self._inflight = []

		self._file = open(path, 'w+b')
		self._file.write(bytes(_HEADER.itemsize))

		if self.n_jobs > 1:
			self._pool = ThreadPoolExecutor(self.n_jobs)

	def __repr__(self):
		return "FigWigWriter({!r}, {} tracks)".format(os.fspath(self.path),
			len(self.tracks))

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
		past the end of its chromosome only where its values are NaN, as
		`FigWigReader.read` gives them there for float output.

		A base that is not written is 0. NaN is a value for a float32 track,
		and an error for an integer one.

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

		names, codes = _chrom_codes(chroms, n)
		unknown = [name for name in names if name not in self._tids]
		if len(unknown) > 0:
			raise ValueError("chromosome {!r} is not one of the writer's "
				"chromosomes.".format(unknown[0]))
		if n == 0:
			return

		tids = numpy.array([self._tids[name] for name in names],
			dtype=numpy.int64)[codes]
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
			last, done = self._last[g], self._done[g]
			for a, b in zip(bounds[:-1], bounds[1:]):
				tid = int(tids[a])
				if last is not None and tid != last[0]:
					done = done | {last[0]}
				if tid in done:
					raise ValueError("chromosome {!r} is written again after "
						"the group of {!r} moved on from it.".format(
						self._chrom_list[tid][0], self.tracks[self._groups[g][0]]
						['name']))
				if last is not None and tid == last[0] and starts[a] < last[1]:
					raise ValueError("{} starts at {}, before the end of what "
						"was written on it before, {}.".format(describe(a if
						order is None else int(order[a])), starts[a], last[1]))
				last = (tid, int(ends[b - 1]))

		inside = None
		if values.ndim == 3:
			inside = numpy.arange(values.shape[2])[None, :] < (ends - starts)[:,
				None]
		for g, members in groups.items():
			slots = numpy.array([members[int(i)] for i in self._groups[g]],
				dtype=numpy.int64)
			self._check_declared(g, slots, values, inside)

		# The encoders take float32 and integers as they are, and other types
		# as float64, from a copy, since encoding goes on after this call
		# returns and the caller may then change `values`.
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
				block = numpy.array(block, dtype=dtype, order='C')
				self._write_group(g, int(tids[a]), starts[a:b], ends[a:b], block,
					values.ndim == 3)

		# Batches are written as they finish compressing, keeping a few in
		# flight per thread, so that a long call does not hold every chunk
		# of its values in memory at once.
		self._write(max(0, len(self._inflight) - (0 if self._pool is None else
			2 * self.n_jobs)))

	def _check_declared(self, g, slots, values, inside):
		"""Raise if a value of a track with a declared dtype does not fit it.
		`inside` masks the bases of windows that lie within their
		chromosome."""

		for i, slot in zip(self._groups[g].tolist(), slots.tolist()):
			dtype = self.tracks[i]['dtype']
			if dtype is None or dtype == 'float32':
				continue

			block = values[:, slot].astype(numpy.float64, copy=False)
			if inside is not None:
				block = block[inside]
			if numpy.isnan(block).any():
				raise ValueError("track {!r} is {}, and a value is NaN.".format(
					self.tracks[i]['name'], dtype))
			if (block != numpy.floor(block)).any():
				raise ValueError("track {!r} is {}, and a value is not an "
					"integer.".format(self.tracks[i]['name'], dtype))

			info = numpy.iinfo(dtype)
			if block.size and (block.min() < info.min or block.max() > info.max):
				raise ValueError("track {!r} is {}, and a value is outside {} to "
					"{}.".format(self.tracks[i]['name'], dtype, info.min, info.max))

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
		bound = n_tracks * tile * 4 + 8
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

		last = self._last[g]
		if last is not None and last[0] != tid:
			self._done[g].add(last[0])
		self._last[g] = (tid, int(ends[-1]))

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
		bytes, sizes, kind bytes, file tiles, stats) of the live tiles."""

		n_tracks = len(self._groups[g])
		tile = self.tile
		n = len(tiles)
		out = numpy.empty(n * (n_tracks * tile * 4 + 8), dtype=numpy.uint8)
		bounds = numpy.zeros(n + 1, dtype=numpy.int64)
		kinds = numpy.zeros(n, dtype=numpy.uint8)
		stats = numpy.zeros((n_tracks, 5), dtype=numpy.float64)
		stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf
		_encode_dense_tiles(tile, values, items, columns, out, bounds, kinds,
			stats)
		return self._compress(g, tid, tiles, n, out, bounds, kinds) + (stats,)

	def _encode(self, g, tid, tiles, lo, hi, items, preset, keep_last):
		"""Run the encoder on a batch of tiles of one group and chromosome,
		on this thread, since it fills the group's buffer, and hand the
		chunks it made to the compressors."""

		n_tracks = len(self._groups[g])
		tile = self.tile
		if items is None:
			starts = numpy.zeros(0, dtype=numpy.int64)
			ends = numpy.zeros(0, dtype=numpy.int64)
			values = numpy.zeros((0, n_tracks, 1), dtype=numpy.float64)
			by_base = False
			lo = hi = numpy.zeros(len(tiles), dtype=numpy.int64)
		else:
			starts, ends, values, by_base = items

		n = len(tiles)
		out = numpy.empty(n * (n_tracks * tile * 4 + 8), dtype=numpy.uint8)
		bounds = numpy.zeros(n + 1, dtype=numpy.int64)
		kinds = numpy.zeros(n, dtype=numpy.uint8)
		stats = self._stats[self._groups[g]]
		n_done = _encode_tiles(tile, tiles, lo, hi, starts, ends, values, by_base,
			self._buffers[g], preset, keep_last, out, bounds, kinds, stats)
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
		(compressed bytes, sizes, kind bytes, file tiles) of the live
		tiles."""

		live = numpy.flatnonzero(kinds[:n_done])
		gtiles = self._tile_offsets[tid] + tiles[:n_done][live]
		if len(live) == 0:
			return numpy.zeros(0, dtype=numpy.uint8), numpy.zeros(0,
				dtype=numpy.int64), kinds[:0], gtiles

		cuts = numpy.append(bounds[live], bounds[n_done])
		data = out[:bounds[n_done]]
		if self.level == 0:
			return data, numpy.diff(cuts), kinds[:n_done][live], gtiles

		compressed, sizes = _compress_blocks(self.engine, self.level, data, cuts)
		return compressed, sizes, kinds[:n_done][live], gtiles

	def _write(self, n_batches):
		"""Write the first `n_batches` batches in flight to the file, in
		order, record their chunks in the index, and fold in the stats of
		those encoded on the pool."""

		for result, g, _ in self._inflight[:n_batches]:
			if not isinstance(result, tuple):
				result = result.result()
			if len(result) == 5:
				stats = result[4]
				rows = self._groups[g]
				self._stats[rows, 0] += stats[:, 0]
				self._stats[rows, 1] += stats[:, 1]
				self._stats[rows, 2] = numpy.minimum(self._stats[rows, 2],
					stats[:, 2])
				self._stats[rows, 3] = numpy.maximum(self._stats[rows, 3],
					stats[:, 3])
				self._stats[rows, 4] = numpy.maximum(self._stats[rows, 4],
					stats[:, 4])

			compressed, sizes, kinds, gtiles = result[:4]
			if len(gtiles) == 0:
				continue

			offset = self._file.tell()
			self._file.write(compressed[:sizes.sum()])
			self._kinds[g][gtiles] = kinds
			self._sizes[g][gtiles] = sizes
			total = int(sizes.sum())

			runs = self._runs[g]
			if len(runs) > 0 and runs[-1][2] + runs[-1][3] == offset and \
					runs[-1][1] < gtiles[0] and runs[-1][3] + total < 2 ** 40:
				runs[-1][1] = int(gtiles[-1])
				runs[-1][3] += total
			else:
				runs.append([int(gtiles[0]), int(gtiles[-1]), offset, total])

		del self._inflight[:n_batches]

	def close(self):
		"""Write the open tiles, the tables, the index and the header, and
		close the file. Calling it again does nothing."""

		if self._file is None:
			return

		try:
			for g, open_ in enumerate(self._open):
				if open_ is not None:
					self._encode(g, open_[0], numpy.array([open_[1]]), None, None,
						None, True, False)
					self._open[g] = None
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
				self.attributes, 'tracks': self.tracks}).encode()
			self._file.write(metadata)

			index_offset = self._file.tell()
			directory = numpy.zeros(len(self._groups), dtype='<u8')
			position = index_offset + directory.nbytes
			records = []
			for g in range(len(self._groups)):
				runs = numpy.zeros(len(self._runs[g]), dtype=_RUN)
				for r, run in enumerate(self._runs[g]):
					runs[r] = tuple(run)
				record = struct.pack('<I', len(runs)) + runs.tobytes() + \
					self._kinds[g].tobytes() + self._sizes[g].tobytes()
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


class FigWigReader:
	"""A figwig file, read many windows and tracks at a time on several
	threads.

	`FigWigReader` reads the per-base values of many windows of one width,
	from any of the file's tracks, in one call, into an array of shape
	(windows, tracks, width): the (batch, channels, length) layout that
	sequence models take. It is built for loading training data from many
	tracks at once, such as the counts of every assay of a cell type under
	a batch of windows.

	The file's tables, metadata and the directory of its index are read when
	it is opened, and each group's index on its first read, after which the
	index is kept for every later read. A read sorts its windows, finds the
	chunks they overlap by division on the tile grid, and works through those
	chunks in batches. Each batch is read from the file, decompressed and
	decoded straight into the windows' rows of the output, on up to `n_jobs`
	threads; zlib's `uncompress()` and the numba decoder run without the
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
		return "FigWigReader('{}', {} tracks, {} chromosomes)".format(self.path,
			len(self.tracks), len(self.chrom_sizes))

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
		"""Group g's index: the kind byte, compressed size and offset of each
		tile's chunk, read on first use and kept."""

		with self._index_lock:
			if self._index[g] is None:
				self._index[g] = self._read_group_index(g)

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
			size = n_runs * _RUN.itemsize + n + 4 * n
			raw = _pread(fd, size, offset + 4)
			if len(raw) != size:
				raise ValueError(error + ": the file ends inside it.")

		runs = numpy.frombuffer(raw[:n_runs * _RUN.itemsize], dtype=_RUN)
		kinds = numpy.frombuffer(raw[n_runs * _RUN.itemsize:n_runs *
			_RUN.itemsize + n], dtype=numpy.uint8).copy()
		sizes = numpy.frombuffer(raw[n_runs * _RUN.itemsize + n:],
			dtype='<u4').astype(numpy.int64)

		if ((kinds & 3) == 3).any() or ((kinds >> 2) > 6).any():
			raise ValueError(error + ": a chunk has a kind this figwig does not "
				"know.")
		if ((kinds == 0) != (sizes == 0)).any():
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

		return kinds, sizes, offsets

	def read(self, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
		numpy.ndarray, width: int, tracks: str | int | list | None = None,
		out: numpy.ndarray | None = None, dtype=None, n_jobs: int = 8
		) -> numpy.ndarray:
		"""Read the per-base values of many windows of one width.

		Window j is [starts[j], starts[j] + width) on chroms[j], and its
		values go to row j of the output whatever order the windows are
		given in. The output has one channel for each track in `tracks`, in
		that order, and shape (n, tracks, width), or (n, width) when `tracks`
		is one name or index.

		A base that was not written is 0. A base past the end of its
		chromosome is NaN in a float output and 0 in an integer one.

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
			the smallest integer type that holds all of them. An integer type
			that cannot hold a track's values raises. Default is None.

		n_jobs: int, optional
			The most threads to use, or -1 for one per CPU. Default is 8.

		Returns
		-------
		out: numpy.ndarray, shape=(n, tracks, width) or (n, width)
			The value of every base of every window in every track.
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

		dtype = self._dtype(ids, dtype)
		shape = (n, width) if single else (n, len(ids), width)
		if out is None:
			out = numpy.empty(shape, dtype=dtype)
		elif not isinstance(out, numpy.ndarray):
			raise TypeError("out must be a numpy array, not {}.".format(
				type(out).__name__))
		elif out.shape != shape or out.dtype != dtype or not out.flags[
				'C_CONTIGUOUS'] or not out.flags['WRITEABLE']:
			raise ValueError("out must be a writeable, C-contiguous {} array of "
				"shape {}.".format(dtype, shape))

		if n == 0:
			return out

		out3 = out.reshape(n, 1, width) if single else out
		tids = numpy.array([self._tids[name] for name in windows.names],
			dtype=numpy.int64)[windows.codes]
		starts = windows.starts
		lengths = self._lengths[tids]

		# A track asked for twice is read once and copied to its other
		# channels after the read.
		first_channel, copies = {}, []
		for channel, i in enumerate(ids.tolist()):
			if i in first_channel:
				copies.append((channel, first_channel[i]))
			else:
				first_channel[i] = channel

		tasks = []
		for g in sorted(set(self._group_of[ids].tolist())):
			slots = numpy.full(len(self._groups[g]), -1, dtype=numpy.int64)
			for i, channel in first_channel.items():
				if self._group_of[i] == g:
					slots[numpy.flatnonzero(self._groups[g] == i)[0]] = channel
			tasks.extend(self._plan(g, slots, tids, starts, lengths, width, out3))

		self._run(tasks, n_jobs)
		for channel, source in copies:
			out3[:, channel] = out3[:, source]

		# Bases past the end of a chromosome lie in no tile.
		over = numpy.flatnonzero(starts + width > lengths)
		if len(over) > 0:
			fill = numpy.nan if dtype.kind == 'f' else 0
			for j in over.tolist():
				out3[j, :, max(0, int(lengths[j] - starts[j])):] = fill

		return out

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

	def _dtype(self, ids, dtype):
		"""The output dtype for these tracks, checked against their values."""

		own = [numpy.dtype(self.tracks[i]['dtype']) for i in ids.tolist()]
		if dtype is None:
			if all(d == own[0] for d in own):
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

		if dtype.kind in 'iu':
			info = numpy.iinfo(dtype)
			for i, d in zip(ids.tolist(), own):
				if d.kind == 'f':
					raise ValueError("track {!r} is float32 and cannot be read as "
						"{}.".format(self.tracks[i]['name'], dtype))
				if self.stats[i]['min'] < info.min or self.stats[i]['max'] > \
						info.max:
					raise ValueError("track {!r} has values from {} to {}, which "
						"{} does not hold.".format(self.tracks[i]['name'],
						self.stats[i]['min'], self.stats[i]['max'], dtype))

		return dtype

	def _plan(self, g, slots, tids, starts, lengths, width, out):
		"""The tasks that read group g's part of the windows: each a batch
		of chunks and the (window, tile) pairs that fall in them."""

		tile = self.tile
		kinds, sizes, offsets = self._group_index(g)

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
		bounds = numpy.where(live, len(slots) * tile * _WIDTHS[chunk_kinds >> 2]
			+ 8, 0)

		# Batches of chunks, cut where a batch would hold too many bytes
		# before compression or too many chunks.
		tasks = []
		a = 0
		while a < len(unique):
			total_bytes, n_live, b = 0, 0, a
			while b < len(unique) and (b == a or (total_bytes + bounds[b] <=
					_BATCH_BYTES and n_live < _BATCH_CHUNKS)):
				total_bytes += int(bounds[b])
				n_live += int(live[b])
				b += 1

			p0, p1 = int(first[a]), int(first[b]) if b < len(unique) else total
			chunk = numpy.where(live[inverse[p0:p1]], inverse[p0:p1] - a, -1)
			tasks.append(_Task(self, g, slots, unique[a:b][live[a:b]],
				chunk_kinds[a:b][live[a:b]], offsets[unique[a:b][live[a:b]]],
				sizes[unique[a:b][live[a:b]]], int(total_bytes),
				_compact(chunk, live[a:b]), local[p0:p1] * tile, starts[rows[p0:p1]],
				rows[p0:p1], out))
			a = b

		return tasks

	def _run(self, tasks, n_jobs):
		"""Run the tasks on up to `n_jobs` threads, the first on this thread
		when a kernel has not been compiled yet, so that each is compiled
		once before any thread starts."""

		if len(tasks) == 0:
			return

		first = []
		if not getattr(_read_tiles, 'signatures', True) or not getattr(
				_check_chunks, 'signatures', True) or (self._codec == _CODEC_ZLIB
				and not getattr(_inflate_chunks, 'signatures', True)):
			first.append(tasks[0])

		rest = tasks[len(first):]
		with open(self.path, 'rb') as handle:
			fd = handle.fileno()
			for task in first:
				task.run(fd)

			if n_jobs == 1 or len(rest) <= 1:
				for task in rest:
					task.run(fd)
			else:
				with ThreadPoolExecutor(min(n_jobs, len(rest))) as pool:
					list(pool.map(lambda task: task.run(fd), rest))


def _compact(chunk, live):
	"""Chunk positions among a batch's live chunks, from positions among all
	of its chunks, with -1 kept for the zero ones."""

	mapping = numpy.cumsum(live) - 1
	return numpy.where(chunk >= 0, mapping[numpy.maximum(chunk, 0)], -1)


class _Task:
	"""One batch of a read: a group's live chunks and the (window, tile)
	pairs that fall in them, including pairs on zero tiles."""

	def __init__(self, reader, g, slots, gtiles, kinds, offsets, sizes, bound,
		pair_chunks, pair_bases, pair_starts, pair_rows, out):
		self.reader, self.g, self.slots = reader, g, slots
		self.gtiles, self.kinds = gtiles, kinds
		self.offsets, self.sizes, self.bound = offsets, sizes, bound
		self.pair_chunks = numpy.ascontiguousarray(pair_chunks, dtype=numpy.int64)
		self.pair_bases = numpy.ascontiguousarray(pair_bases, dtype=numpy.int64)
		self.pair_starts = numpy.ascontiguousarray(pair_starts, dtype=numpy.int64)
		self.pair_rows = numpy.ascontiguousarray(pair_rows, dtype=numpy.int64)
		self.out = out

	def run(self, fd):
		reader = self.reader
		n = len(self.gtiles)
		if n > 0:
			data, positions, lengths = self._load(fd)
			failed = numpy.zeros(n, dtype=numpy.int64)
			if _check_chunks(reader.tile, data, positions, lengths, self.kinds,
					len(self.slots), failed) > 0:
				k = int(numpy.argmax(failed))
				raise ValueError("a chunk of {} cannot be read: tile {} of the "
					"group of {!r} is corrupt, or the file is truncated.".format(
					reader.path, int(self.gtiles[k]), reader.tracks[reader._groups[
					self.g][0]]['name']))
		else:
			data = numpy.zeros(8, dtype=numpy.uint8)
			positions = numpy.zeros(0, dtype=numpy.int64)

		_read_tiles(reader.tile, data, positions, self.kinds, len(self.slots),
			self.slots, self.pair_chunks, self.pair_bases, self.pair_starts,
			self.pair_rows, self.out)

	def _load(self, fd):
		"""The batch's chunks, read with one pread per run of adjacent
		chunks and inflated into one buffer, as (data, positions, lengths)."""

		reader = self.reader
		offsets, sizes = self.offsets, self.sizes
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

		# Back to the batch's own order of chunks.
		starts_in = numpy.empty_like(starts)
		starts_in[order] = starts
		sizes_in = sizes.astype(numpy.int64)

		if reader._codec == _CODEC_NONE:
			return raw, starts_in, sizes_in

		positions = numpy.zeros(len(sizes), dtype=numpy.int64)
		lengths = numpy.zeros(len(sizes), dtype=numpy.int64)
		uncompress = _zlib_uncompress()
		used = 0
		buffer = numpy.zeros(0, dtype=numpy.uint8)
		if uncompress is not None:
			function, ulong = uncompress
			buffer = numpy.empty(self.bound + 8 * len(sizes), dtype=numpy.uint8)
			used = _inflate_chunks(function, raw, starts_in, sizes_in, buffer,
				positions, lengths, numpy.zeros(1, dtype=ulong))
		else:
			positions[:] = -1

		parts = [buffer[:used]]
		position = used
		for k in numpy.flatnonzero(positions < 0).tolist():
			stream = raw[starts_in[k]:starts_in[k] + sizes_in[k]].tobytes()
			try:
				block = zlib.decompressobj().decompress(stream, self.bound + 1)
			except zlib.error:
				block = b''

			padded = block + bytes(-len(block) % 8)
			parts.append(numpy.frombuffer(padded, dtype=numpy.uint8))
			positions[k], lengths[k] = position, len(block)
			position += len(padded)

		data = parts[0] if len(parts) == 1 else numpy.concatenate(parts)
		return data, positions, lengths


def read_figwig(figwig: str | os.PathLike | FigWigReader, chroms: str |
	list[str] | numpy.ndarray, starts: list[int] | numpy.ndarray, width: int,
	tracks: str | int | list | None = None, out: numpy.ndarray | None = None,
	dtype=None, n_jobs: int = 8) -> numpy.ndarray:
	"""Read the per-base values of many windows of one width from a figwig
	file.

	This reads the windows as `FigWigReader.read` does, from a path or an
	open reader, into an array of shape (n, tracks, width), or (n, width)
	when `tracks` is one name or index. A path is opened on every call; to
	read the same file more than once, as a data loader does, open it with
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


	Returns
	-------
	out: numpy.ndarray, shape=(n, tracks, width) or (n, width)
		The value of every base of every window in every track.
	"""

	if not isinstance(figwig, (str, os.PathLike, FigWigReader)):
		raise TypeError("figwig must be a path or a FigWigReader, not {}.".format(
			type(figwig).__name__))

	reader = figwig if isinstance(figwig, FigWigReader) else FigWigReader(figwig)
	return reader.read(chroms, starts, width, tracks=tracks, out=out,
		dtype=dtype, n_jobs=n_jobs)


def write_figwig(path: str | os.PathLike, chrom_sizes: dict | list,
	tracks: list, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
	numpy.ndarray, values, ends: list[int] | numpy.ndarray | None = None,
	genome: str | None = None, attributes: dict | None = None,
	tile: int = _DEFAULT_TILE, groups: int | list = _DEFAULT_GROUPS,
	level: int = 6, engine: str = 'auto', n_jobs: int = 8):
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

	level: int, optional
		The compression level; see `FigWigWriter`. Default is 6.

	engine: str, optional
		What compresses the chunks; see `FigWigWriter`. Default is 'auto'.

	n_jobs: int, optional
		The most threads that compress chunks at once, besides the calling
		thread, or -1 for one per CPU. Default is 8.
	"""

	with FigWigWriter(path, chrom_sizes, tracks, genome=genome,
			attributes=attributes, tile=tile, groups=groups, level=level,
			engine=engine, n_jobs=n_jobs) as writer:
		writer.write(chroms, starts, values, ends=ends)
