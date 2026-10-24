# figwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""figwig files: many per-base tracks in one file, read in windows.

A figwig file holds the per-base values of any number of tracks over one
genome, with a JSON record of attributes for each track and for the file,
and is read the way `BigWigReader` reads a bigWig: many windows of one
width at a time, into one array of shape (windows, tracks, width), or as a
sparse list of the nonzero bases, or summed per label of each track, or
summed per bin of each window.

A file is written in one of two orders. In position order, every track is
cut on one grid of tiles of `tile` bases, so a window's tiles are found by
division rather than by a tree search; the tiles of a group of tracks at
one position are one chunk, so a window read decodes each chunk once for
every track in it; a chunk whose values are all zero takes no space; and
the other chunks hold their values densely, as nonzero bases per track, as
runs of one value, or as a sparse matrix of the tracks present, in the
narrowest integer that holds them. That is the order for reading windows
across many tracks. In track order, each track's nonzero bases lie in
blocks per chromosome, so one track's values over the genome are a few
slices, and a window read searches each requested track's blocks. That is
the order for reading a few of many tracks, such as cells of a single-cell
experiment.

The file is, in order: a 128-byte header; the chunks or blocks, as they
were written; the chromosome table; a table of each track's sum, nonzero
bases, minimum and maximum; the metadata, a JSON document holding the
genome, the file's attributes and every track's record; and the index, one
record per group of tracks holding, in position order, the runs of tiles
that lie back to back in the file and, deflated, the kind byte and the
compressed and uncompressed sizes of every tile that holds values, and in
track order the blocks of each track. `_figwig_kernels` describes the
chunks and blocks; `docs/figwig.rst` gives every field.
"""

from __future__ import annotations

import json
import os
import stat
import struct
import tempfile
import threading
import warnings
import zlib

from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy

from ._figwig_kernels import _BLOCK_ENTRIES
from ._figwig_kernels import _CODE_MASK
from ._figwig_kernels import _CODE_SHIFT
from ._figwig_kernels import _F64
from ._figwig_kernels import _FLAG_FRACTION
from ._figwig_kernels import _FLAG_OVERFLOW
from ._figwig_kernels import _FLAG_WIDE
from ._figwig_kernels import _FLOAT32
from ._figwig_kernels import _FLOAT64
from ._figwig_kernels import _INTEGER
from ._figwig_kernels import _KIND_MASK
from ._figwig_kernels import _RUNS
from ._figwig_kernels import _STORED
from ._figwig_kernels import _check_blocks
from ._figwig_kernels import _check_chunks
from ._figwig_kernels import _encode_blocks
from ._figwig_kernels import _encode_dense_tiles
from ._figwig_kernels import _encode_entries
from ._figwig_kernels import _encode_tiles
from ._figwig_kernels import _inflate_chunks
from ._figwig_kernels import _bucket_sort
from ._figwig_kernels import _inflate_chunks_libdeflate
from ._figwig_kernels import _pread_runs
from ._figwig_kernels import _list_tiles
from ._figwig_kernels import _read_blocks
from ._figwig_kernels import _read_tiles
from ._kernels import _libc_pread
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
# chunk's offsets fit uint16; a group's chunk must fit a uint32 size; and
# a genome's tiles are capped so that an index stays a few hundred
# megabytes at most.
_MIN_TILE, _MAX_TILE = 16, 65536
_DEFAULT_TILE = 16384
_DEFAULT_GROUPS = 8
_MAX_GROUP_BASES = 1 << 29
_MAX_TILES = 1 << 26
_MAX_NAME_BYTES = 65535
_MAX_CHROM_LENGTH = 2 ** 32 - 1

# The most bytes of chunks before compression that one task encodes or
# decodes, the most and the fewest chunks one read task takes, how many
# tasks each thread of a write should get, the chunks a read needs per
# thread before it uses another, the most bytes a dense read allocates
# before asking for a sparse or summed one instead, and the most bytes of
# scratch a read task decodes a chunk into.
_BATCH_BYTES = 1 << 25
_BATCH_CHUNKS = 256
_MIN_BATCH_CHUNKS = 16
_BATCHES_PER_THREAD = 8
_CHUNKS_PER_THREAD = 64
_SPLIT_PAIRS = 1024
_MAX_DENSE_BYTES = 1 << 33
_SCRATCH_BYTES = 1 << 28

# A file keeps each track's sum over every tile, for summaries without a
# read of the chunks, when those sums take at most this many bytes.
_SUMMARY_BYTES = 1 << 28
_PAIRED_TRACKS = 16  # distinct tracks per read of paired sums
_COPY_ON_POOL = 1 << 26  # bytes of a write's block above which its copy is made on the pool

# Whether a tile whose few tracks have values may be stored as a matrix
# chunk, which a compressed file chooses only when the sparse chunk's
# per-track counts, mostly zero, would not deflate to less.
_MATRIX_CHUNKS = True

_DTYPES = ('uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'float32',
	'float64')
_RESERVED = ('name', 'dtype', 'group')
_CLASSES = {'float32': _FLOAT32, 'float64': _FLOAT64}

_HEADER = numpy.dtype([('magic', '<u4'), ('version', '<u2'), ('flags', '<u2'),
	('tile', '<u4'), ('codec', '<u2'), ('level', '<u2'), ('n_chroms', '<u4'),
	('n_tracks', '<u4'), ('n_groups', '<u4'), ('tables_crc', '<u4'),
	('chroms', '<u8'), ('tracks', '<u8'), ('index', '<u8'), ('data', '<u8'),
	('metadata', '<u8'), ('metadata_length', '<u8'), ('zooms', '<u8'),
	('length', '<u8'), ('required', '<u4'), ('optional', '<u4'),
	('header_crc', '<u4'), ('padding', 'V20')])
# Feature bits a reader must know to read the file (`required`) or may
# ignore (`optional`). No required bit is defined, so a reader refuses any
# set; the one optional bit says each group's index record ends with a zoom
# section (written by every file since the first release), and any other
# optional bit is ignored.
_KNOWN_REQUIRED = 0
_OPTIONAL_ZOOMS = 1
_MAX_ZOOMS = 12  # levels, each 4 times coarser than the last
# The metadata keys the format itself writes; any other key of a file's
# metadata is kept as it is through an append.
_METADATA_KEYS = ('figwig', 'genome', 'attributes', 'order', 'tracks')
_STATS = numpy.dtype([('sum', '<f8'), ('nonzero', '<u8'), ('min', '<f8'),
	('max', '<f8')])
_RUN = numpy.dtype([('first', '<u8'), ('last', '<u8'), ('offset', '<u8'),
	('bytes', '<u8')])
_BLOCK = numpy.dtype([('chrom', '<u4'), ('first', '<u4'), ('last', '<u4'),
	('n', '<u4'), ('offset', '<u8'), ('size', '<u4'), ('raw', '<u4'),
	('kind', '<u4'), ('crc', '<u4')])
_LIVE = numpy.dtype([('tile', '<u4'), ('kind', 'u1'), ('size', '<u4'),
	('raw', '<u4'), ('crc', '<u4')])
_LIVE_FIELDS = ('tile', 'kind', 'size', 'raw', 'crc')


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


def _json(document, what):
	"""`document` as JSON, or a TypeError or ValueError saying why not; NaN
	and infinities are refused, since they are not JSON."""

	try:
		return json.dumps(document, allow_nan=False)
	except TypeError as e:
		raise TypeError("{} must be JSON serializable: {}".format(what, e)) \
			from None
	except ValueError:
		raise ValueError("{} must be JSON serializable: NaN and infinity are not "
			"JSON.".format(what)) from None


def _check_tracks(tracks, empty_ok=False):
	"""The `tracks` argument as a list of dicts with a `name`, a `dtype` that
	may be None, and the attributes given."""

	if tracks is None and empty_ok:
		tracks = []
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
		_json(track, "the attributes of track {!r}".format(name))
		records.append(track)
		names.add(name)

	if len(records) == 0 and not empty_ok:
		raise ValueError("tracks must hold at least one track.")

	return records


def _check_groups(groups, tracks):
	"""The group of each track, as a list of arrays of track ids, each in
	the file's track order, which is the order a group's tracks take in
	its chunks."""

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
		out.append(numpy.sort(numpy.array([ids[name] for name in group],
			dtype=numpy.int64)))

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

	_json(attributes, "attributes")
	return dict(attributes)


def _tile_offsets(lengths, tile):
	"""The first tile of each chromosome in the file's tile numbering, and
	the number of tiles in all."""

	counts = (lengths + tile - 1) // tile
	offsets = numpy.concatenate([[0], numpy.cumsum(counts)])
	return offsets[:-1], int(offsets[-1])


def _zoom_grid(lengths, tile, factor):
	"""The first bin of each chromosome at a zoom level of `factor` tiles
	per bin, and the number of bins in all: a chromosome of n tiles has
	ceil(n / factor) bins, and a bin never crosses a chromosome."""

	counts = -(-((lengths + tile - 1) // tile) // factor)
	offsets = numpy.concatenate([[0], numpy.cumsum(counts)])
	return offsets[:-1], int(offsets[-1])


def _check_zooms(zooms):
	if isinstance(zooms, bool) or not isinstance(zooms, (int, numpy.integer)):
		raise TypeError("zooms must be an integer, the number of zoom levels, "
			"not {}.".format(type(zooms).__name__))
	if not 0 <= zooms <= _MAX_ZOOMS:
		raise ValueError("zooms must be from 0 to {}, not {}.".format(_MAX_ZOOMS,
			zooms))
	return int(zooms)


def _infer_dtype(nonzero, low, high, flags):
	"""The narrowest dtype that holds a track's values: float32 for a
	fraction, float64 for an integer beyond int32 and uint32, which float32
	would round, and otherwise the narrowest integer type."""

	flags = int(flags)
	if flags & _FLAG_FRACTION:
		return 'float32'
	if flags & _FLAG_WIDE:
		return 'float64'
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
	return 'float64'


def _positions(positions, name):
	"""`positions` as a 1-D int64 array."""

	positions = numpy.asarray(positions)
	if positions.ndim != 1:
		raise ValueError("{} must be one-dimensional, not of shape {}: give one "
			"per window, as a list or array.".format(name, positions.shape))
	if positions.size and positions.dtype.kind not in 'iu':
		raise TypeError("{} must be integers, not {}.".format(name,
			positions.dtype))

	return positions.astype(numpy.int64)


def _track_indices(tracks, track_ids, n_tracks, whose):
	"""The tracks `tracks` names, by name or index, one or a list, as (ids,
	whether one track was named); `whose` says whose tracks they are in a
	message."""

	single = isinstance(tracks, (str, int, numpy.integer)) and not isinstance(
		tracks, bool)
	items = [tracks] if single else tracks
	if isinstance(items, (str, dict)) or not hasattr(items, '__iter__'):
		raise TypeError("tracks must be a name, an index, or a list of them, "
			"not {}.".format(type(tracks).__name__))

	ids = []
	for item in items:
		if isinstance(item, str):
			if item not in track_ids:
				raise ValueError("{!r} is not one of {} tracks.".format(item, whose))
			ids.append(track_ids[item])
		elif isinstance(item, (int, numpy.integer)) and not isinstance(item,
				bool):
			if not 0 <= item < n_tracks:
				raise ValueError("track {} is out of range; there are {} tracks."
					.format(item, n_tracks))
			ids.append(int(item))
		else:
			raise TypeError("tracks must be names or indices, not {}.".format(
				type(item).__name__))
	return numpy.array(ids, dtype=numpy.int64), single


def _check_exact(values):
	"""Raise for a 64-bit integer array with a value beyond 2**53, which
	float64, and so the file, cannot hold exactly, and for an infinite
	value, which no figwig file holds: a base is a finite value or NaN."""

	values = numpy.asarray(values)
	if values.dtype.kind in 'iu' and values.dtype.itemsize == 8 and values.size:
		if int(values.max()) > 2 ** 53 or int(values.min()) < -2 ** 53:
			raise ValueError("an integer lies beyond 2**53, which a figwig file "
				"cannot hold exactly; scale or clip the values first.")
	if values.dtype.kind == 'f' and values.size and numpy.isinf(values).any():
		raise ValueError("values must be finite or NaN, but one is infinite; "
			"clip the values first.")


def _check_values_fit(name, dtype, values):
	"""Raise if the values of a track with a declared dtype do not fit it."""

	if dtype is None or dtype in ('float32', 'float64'):
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


def _header_crc(header_bytes):
	"""The CRC-32 of a header's 128 bytes with its own CRC field as 0."""

	at = _HEADER.fields['header_crc'][1]
	return zlib.crc32(header_bytes[:at] + bytes(4) + header_bytes[at + 4:])


def _chromosome_difference(theirs, ours):
	"""How two chromosome tables differ, in a few words."""

	only_theirs = [name for name in theirs if name not in ours]
	only_ours = [name for name in ours if name not in theirs]
	lengths = [name for name in theirs if name in ours and theirs[name] !=
		ours[name]]
	parts = []
	if only_theirs:
		parts.append("it has {} the writer lacks, such as {!r}".format(
			"{} chromosomes".format(len(only_theirs)) if len(only_theirs) > 1
			else "a chromosome", only_theirs[0]))
	if only_ours:
		parts.append("it lacks {} the writer has, such as {!r}".format(
			"{} chromosomes".format(len(only_ours)) if len(only_ours) > 1 else
			"a chromosome", only_ours[0]))
	if lengths:
		parts.append("{!r} is {} bases long in it and {} in the writer".format(
			lengths[0], theirs[lengths[0]], ours[lengths[0]]))
	return '; '.join(parts)


def _unknown_chromosome(name, chrom_sizes):
	"""The message for a chromosome name the file does not have."""

	names = list(chrom_sizes)
	shown = ', '.join(map(repr, names[:6]))
	if len(names) > 6:
		shown += ", ... ({} in all)".format(len(names))
	return "chromosome {!r} is not one of the file's chromosomes, {}.".format(
		name, shown)


def _lock_for_appending(handle, path):
	"""Take the file's lock while appending to it, so that a second writer
	appending at once, which would write its chunks over the first's, is
	refused; the lock goes with the handle when it closes."""

	try:
		import fcntl
	except ImportError:
		return
	try:
		fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
	except OSError:
		handle.close()
		raise ValueError("{} is being appended to by another writer.".format(
			path)) from None


def _temporary_path(path):
	"""A new file beside `path` to write into, renamed over `path` when the
	writer closes, so that a reader of `path` never sees a part of a file
	and a failed write leaves what was there."""

	directory, base = os.path.split(os.path.abspath(path))
	fd, name = tempfile.mkstemp(prefix=base + '.', suffix='.partial',
		dir=directory)
	os.close(fd)
	# mkstemp makes the file readable by its owner alone; the file is for
	# whoever the umask lets read a file, as open() would make it.
	mask = os.umask(0)
	os.umask(mask)
	try:
		os.chmod(name, 0o666 & ~mask)
	except OSError:
		pass
	return name


class FigWigWriter:
	"""Write a figwig file, a batch of windows, intervals or nonzero bases at
	a time, on several threads.

	`FigWigWriter` takes the values of several tracks at once with `write`,
	as the windows `FigWigReader.read` reads, of shape (n, tracks, width),
	or as intervals, and the nonzero bases of tracks with `write_sparse`,
	and writes them as they come, so that tracks larger than memory, such as
	a model's predictions over a genome, can be written a batch at a time.
	Use it as a context manager, or call `close` when every value has been
	written; the tables and the index are written there, and the file is
	then moved into place: until `close` returns, the path holds what it
	held before, and a writer that fails or is abandoned leaves it so.

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
	longer index and a slower first read of each group.

	Each tile of a group is encoded as the writer reaches it: as nothing
	when every value is zero, and otherwise densely, as nonzero bases per
	track, as runs of one value, or as a sparse matrix of the tracks
	present, whichever takes the fewest bytes, in the narrowest integer type
	that holds the tile's values, float32 where they are not integers, or
	float64 where float32 would change an integer of an integer track or a
	value of a float64 track. A tile that `write` fills only part of waits
	for the next call. The chunks are compressed on up to `n_jobs` threads
	while the calling thread encodes the next batch, with zlib or
	libdeflate, as `BigWigWriter` compresses its blocks, and a chunk that
	compression would grow is stored as it is. The chunks do not depend on
	the number of threads or on how the values were split between calls,
	only on the values and on which calls wrote which groups; the sum in the
	statistics table, a compensated float64, may differ in its last bits
	between splits.

	In track order, every track is its own group, and its nonzero bases on
	each chromosome are stored in blocks of up to 65,536, in position
	order. Windows and intervals are reduced to their nonzero bases, and
	each track's bases must come in position order on each chromosome
	across calls. The blocks are stored uncompressed unless `level` is
	given, since a window read over many tracks decodes a block of each.

	A track's `dtype` is what `FigWigReader.read` returns its values as.
	Given, it is checked against every value written; left out, it is set
	when the writer closes, to the narrowest type that holds the track's
	values: an integer type, float32 where a value is not an integer, and
	float64 where an integer lies beyond int32 and uint32. Either way each
	chunk is stored as narrowly as its own values allow. A float32 track's
	values are what float32 makes of them, as are fractions of a track left
	undeclared; a value beyond float32's range is an error for either.

	Tracks of another figwig file written with the same chromosomes, tile,
	compression level and order can be added whole with `add_file`, which
	copies its chunks rather than decoding them, and `append=True` adds
	tracks to the file at `path` itself, in place: its chunks stay where
	they are, the new ones go after its end, and new tables, index and
	header are written at `close`, so that a file grows by a track at a
	time at the cost of the new chunks alone. The file keeps each track's sum over every tile, for
	`FigWigReader.tile_sums`, as long as those sums take at most 256 MB,
	and a CRC-32 of every chunk, which `FigWigReader(path, verify=True)`
	checks.

	Parameters
	----------
	path: str or os.PathLike
		The file to write. It is written beside its final name and moved
		there when the writer closes, or, with `append=True`, extended in
		place.

	chrom_sizes: dict, list of (str, int), or None
		The chromosomes and their lengths: a dict such as
		`BigWigReader.chrom_sizes`, or a list of (name, length) pairs. None
		with `append=True` takes the file's.

	tracks: list of str or of dict, or None
		The tracks, in order: a name for each, or a dict with a 'name', an
		optional 'dtype' (one of uint8, uint16, uint32, int8, int16, int32,
		float32 or float64), and any JSON-serializable attributes, such as
		the assay, the biosample or the strand, which `FigWigReader.select`
		matches. None, or an empty list, is allowed when tracks come from
		`append` or `add_file`.

	genome: str or None, optional
		The reference genome the coordinates are on, as a name such as
		'hg38' or a URL to its sequence. Default is None.

	attributes: dict or None, optional
		JSON-serializable attributes of the file as a whole, such as how it
		was made. Default is None.

	tile: int, optional
		The bases per tile in position order, a power of two from 16 to
		65536. Smaller tiles decode less for each window and compress a
		little less well; the genome may hold at most 2**26 tiles. Default
		is 16384.

	groups: int or list of lists of str, optional
		In position order, the tracks whose tiles are stored together: a
		number of consecutive tracks per group, or a list of groups of track
		names that holds every track once; within a chunk a group's tracks
		lie in the file's track order whatever order the list gives them. In
		track order every track is its own group, whatever is given. Default
		is 8.

	level: int or None, optional
		The compression level, as `BigWigWriter` takes it, or 0 to store the
		chunks without compressing them, which decodes fastest and costs
		four bytes per base of a dense float track. None is 6 in position
		order and 0 in track order. Default is None.

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

	append: bool, optional
		Whether to add `tracks` to the figwig file at `path`, in place: its
		chunks stay where they are, and the new chunks, the tables, the
		index and last the header are written after them, so that until
		`close` returns the file reads as it did, a writer that fails cuts
		it back to what it was, and its old tables and index are left
		behind as dead bytes, which `figwig merge` into a new path drops.
		Its chromosomes, tile, level and order are taken, and `chrom_sizes`,
		`tile`, `level` and `order` must agree with them or be left at None.
		Default is False.

	zooms: int or None, optional
		The number of zoom levels to keep beside the tile sums: level k holds
		each track's sum over bins of 4**k tiles, in float64, which
		`FigWigReader.tile_sums(level=k)` returns without decoding a chunk,
		for coarse genome-wide profiles. Levels are kept while the tile sums
		are (tracks x tiles x 4 bytes at most 256 MB). None is the file's
		with `append=True` and 0 otherwise. Default is None.
	"""

	def __init__(self, path: str | os.PathLike, chrom_sizes: dict | list | None,
		tracks: list | None, genome: str | None = None, attributes: dict | None =
		None, tile: int | None = None, groups: int | list = _DEFAULT_GROUPS,
		level: int | None = None, engine: str = 'auto', n_jobs: int = 8,
		order: str | None = None, append: bool = False, zooms: int | None = None):
		self._file = None
		self._pool = None
		self._partial = None
		self._length = None
		self._times = None
		self._written = False
		self._adopted = set()
		self._metadata = {}

		source = None
		if append:
			source = FigWigReader(path)
			if zooms is None:
				zooms = len(source.zooms)
			for given, own, name in ((chrom_sizes, source.chrom_sizes,
					'chrom_sizes'), (tile, source.tile, 'tile'), (level, source._level,
					'level'), (order, source.order, 'order')):
				if given is not None and (dict(_check_chrom_sizes(given)) if name
						== 'chrom_sizes' else _ORDERS.get(given, given) if name == 'order'
						else given) != own:
					raise ValueError("{} must be the file's with append=True, {!r}, "
						"or None.".format(name, own))
			chrom_sizes, tile, level, order = source.chrom_sizes, source.tile, \
				source._level, source.order
			if genome is None:
				genome = source.genome
			self._metadata = {key: value for key, value in source._metadata.items()
				if key not in _METADATA_KEYS}
			if attributes is None:
				attributes = source.attributes
			elif isinstance(attributes, dict):
				attributes = {**source.attributes, **attributes}
		if tile is None:
			tile = _DEFAULT_TILE
		if order is None:
			order = 'positions'

		self.chrom_sizes = dict(_check_chrom_sizes(chrom_sizes))
		self._chrom_list = list(self.chrom_sizes.items())
		for name, length in self._chrom_list:
			if len(name.encode()) > _MAX_NAME_BYTES:
				raise ValueError("chromosome name {!r} is longer than {} bytes."
					.format(name[:40] + '...', _MAX_NAME_BYTES))
			if length > _MAX_CHROM_LENGTH:
				raise ValueError("chromosome {!r} is {} bases long; a figwig file "
					"holds chromosomes of up to {}.".format(name, length,
					_MAX_CHROM_LENGTH))
		self._tids = {name: i for i, name in enumerate(self.chrom_sizes)}
		self._lengths = numpy.array(list(self.chrom_sizes.values()),
			dtype=numpy.int64)

		self.order = _check_order(order)
		self.tracks = _check_tracks(tracks, empty_ok=True)
		if self.order == 'tracks':
			self._groups = [numpy.array([i]) for i in range(len(self.tracks))]
		else:
			self._groups = _check_groups(groups, self.tracks) if self.tracks else []
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
		self._classes = [numpy.array([_CLASSES.get(self.tracks[i]['dtype'],
			_INTEGER) for i in group], dtype=numpy.int64) for group in
			self._groups]

		if genome is not None and not isinstance(genome, str):
			raise TypeError("genome must be a string or None, not {}.".format(
				type(genome).__name__))

		self.genome = genome
		self.attributes = _check_attributes(attributes)
		if level is None:
			level = 6 if self.order == 'positions' else 0
		self.engine, self.level = _check_engine(engine, level)
		self._matrix = _MATRIX_CHUNKS
		self.n_jobs = _check_n_jobs(n_jobs)
		self.path = os.fspath(path)

		self._tile_offsets, self._n_tiles = _tile_offsets(self._lengths,
			self.tile)
		if self._n_tiles > _MAX_TILES:
			raise ValueError("the genome has {:,} tiles of {} bases; a figwig "
				"file holds at most {:,} tiles, so use a tile of {} or more."
				.format(self._n_tiles, self.tile, _MAX_TILES, _smallest_tile(
				self._lengths)))

		n_groups = len(self._groups)
		self._live = [[] for _ in range(n_groups)]
		self._runs = [[] for _ in range(n_groups)]
		self._blocks = [[] for _ in range(n_groups)]
		self._sums = [[] for _ in range(n_groups)]
		self._pending = [None] * n_groups
		self._buffers = [None] * n_groups
		self._open = [None] * n_groups
		self._sealed = [None] * n_groups
		self._last = [None] * n_groups
		self._done = [set() for _ in range(n_groups)]
		self._stats = numpy.zeros((len(self.tracks), 6), dtype=numpy.float64)
		self._stats[:, 2] = numpy.inf
		self._stats[:, 3] = -numpy.inf
		self._inflight = []
		self._copies = True
		self._summaries = len(self.tracks) * self._n_tiles * 4 <= _SUMMARY_BYTES
		self.zooms = _check_zooms(0 if zooms is None else zooms)
		self._zoom_grids = [_zoom_grid(self._lengths, self.tile, 4 ** (k + 1))
			for k in range(self.zooms)]

		if source is None:
			self._partial = _temporary_path(self.path)
			self._file = open(self._partial, 'w+b')
			self._file.write(bytes(_HEADER.itemsize))
		else:
			# The file itself: the new chunks go after what the header says
			# is its end, and the tables, the index and last the header are
			# written at close, so the file reads as it did until then.
			self._length = source._file_size
			self._times = os.stat(self.path)
			handle = open(self.path, 'r+b')
			_lock_for_appending(handle, self.path)
			self._file = handle
			self._file.seek(self._length)

		if self.n_jobs > 1:
			self._pool = ThreadPoolExecutor(self.n_jobs)

		if source is not None:
			self._adopt(source, first=True)

	def __repr__(self):
		return "FigWigWriter({!r}, {} tracks, {} order)".format(self.path,
			len(self.tracks), self.order[:-1])

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc, traceback):
		if exc_type is None:
			self.close()
		else:
			self._abandon()

	def __del__(self):
		if getattr(self, '_file', None) is not None:
			if self._written:
				warnings.warn("FigWigWriter({!r}) was never closed, and its values "
					"were not written.".format(self.path), ResourceWarning,
					stacklevel=2)
			self._abandon()

	def _abandon(self):
		"""Stop without finishing the file: the partial file is removed, or
		the file appended to is cut back to its length, and the path holds
		what it held before the writer was made."""

		if self._pool is not None:
			# Nothing is waited for: close has already collected what it
			# needs, and a writer dropped with work in flight may be
			# finalized on one of the pool's own threads.
			self._pool.shutdown(wait=False, cancel_futures=True)
			self._pool = None
		if self._file is not None:
			if self._length is not None:
				# Cut back to the file as it was, with its modification time,
				# since its bytes are what they were and a reader open on it
				# need not be told otherwise.
				try:
					self._file.truncate(self._length)
					self._file.close()
					os.utime(self.path, ns=(self._times.st_atime_ns,
						self._times.st_mtime_ns))
				except OSError:
					pass
			self._file.close()
			self._file = None
		if self._partial is not None:
			try:
				os.remove(self._partial)
			except OSError:
				pass
			self._partial = None

	def _select(self, tracks):
		"""The tracks a call names, as (track ids in channel order, whether a
		single track was named)."""

		if tracks is None:
			# Every track of this writer's own, leaving out those that came
			# from another file.
			return numpy.array([i for i in range(len(self.tracks)) if int(
				self._group_of[i]) not in self._adopted], dtype=numpy.int64), False
		ids, single = _track_indices(tracks, self._track_ids, len(self.tracks),
			"the writer's")
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
			if g in self._adopted:
				raise ValueError("track {!r} came from another file, whose tracks "
					"cannot be written to.".format(self.tracks[list(members)[0]][
					'name']))
			if len(members) != len(self._groups[g]):
				missing = [self.tracks[i]['name'] for i in self._groups[g] if i
					not in members]
				raise ValueError("tracks must make up whole groups: {!r} is in "
					"the group of {!r}, which is not among them.".format(
					missing[0], self.tracks[list(members)[0]]['name']))

		return groups

	def add_file(self, path: str | os.PathLike):
		"""Add every track of another figwig file, by copying its chunks.

		The file must have been written with this writer's chromosomes,
		tile, compression level and order, and its track names must not be
		this writer's. Its tracks come after those written so far, in their
		own groups, with their attributes, dtypes and statistics, and cannot
		be written to here. Only its chunks are copied, so a file grown by
		`append=True` loses the tables and indexes it left behind.

		Parameters
		----------
		path: str or os.PathLike
			The figwig file whose tracks to add.
		"""

		if self._file is None:
			raise ValueError("the writer is closed.")
		self._adopt(FigWigReader(path), first=False)

	def _adopt(self, source, first):
		"""Take over the tracks, groups, runs or blocks, live tiles,
		summaries and statistics of `source`: with `first`, the file being
		appended to in place, whose chunks stay where they are and whose
		tracks come before this writer's own; otherwise another file, whose
		chunks are copied in and whose tracks come after."""

		if source.chrom_sizes != self.chrom_sizes:
			raise ValueError("{} was written with other chromosomes or lengths: "
				"{}.".format(source.path, _chromosome_difference(
				source.chrom_sizes, self.chrom_sizes)))
		for name, theirs, ours in (('order', source.order, self.order), ('tile',
				source.tile, self.tile), ('level', source._level, self.level)):
			if theirs != ours:
				raise ValueError("{} was written with {} {} where this writer has "
					"{}; figwig reorder writes it again with another.".format(
					source.path, name, theirs, ours))
		taken = set(self._track_ids) & {track['name'] for track in source.tracks}
		if taken:
			raise ValueError("track {!r} of {} is already one of the writer's."
				.format(sorted(taken)[0], source.path))

		# A file appended to keeps its chunks where they are; another file's
		# are copied whole, and its runs and blocks keep their layout, moved
		# to where the copy landed.
		self._write(len(self._inflight))
		extents = None if first else self._copy_chunks(source)

		n_groups, n_tracks = len(source._groups), len(source.tracks)
		if first:
			base_group, base_track = 0, 0
			for track in self.tracks:
				track['group'] += n_groups
			self._groups = [group + n_tracks for group in self._groups]
			self._group_of = self._group_of + n_groups
			self._adopted = {g + n_groups for g in self._adopted}
		else:
			base_group, base_track = len(self._groups), len(self.tracks)

		records = []
		for track in source.tracks:
			record = dict(track)
			record['group'] += base_group
			records.append(record)
		stats = numpy.zeros((n_tracks, 6), dtype=numpy.float64)
		stats[:, 0] = source.stats['sum']
		stats[:, 1] = source.stats['nonzero']
		stats[:, 2] = numpy.where(source.stats['nonzero'] > 0, source.stats['min'],
			numpy.inf)
		stats[:, 3] = numpy.where(source.stats['nonzero'] > 0, source.stats['max'],
			-numpy.inf)
		groups, classes, live, runs, blocks, sums = [], [], [], [], [], []
		for g, group in enumerate(source._groups):
			members = group + base_track
			groups.append(members)
			classes.append(numpy.array([_CLASSES.get(records[i - base_track][
				'dtype'], _INTEGER) for i in members], dtype=numpy.int64))
			if self.order == 'tracks':
				found = source._group_index(g).copy()
				if extents is not None:
					found['offset'] = _moved(found['offset'], found['size'], extents)
				blocks.append([found])
				live.append([])
				runs.append([])
			else:
				kinds, sizes, raws, offsets, crcs = source._group_index(g)
				tiles = numpy.flatnonzero(kinds)
				record = numpy.zeros(len(tiles), dtype=_LIVE)
				record['tile'], record['kind'], record['size'] = tiles, kinds[tiles], \
					sizes[tiles]
				record['raw'], record['crc'] = raws[tiles], crcs[tiles]
				live.append([record])
				found = source._runs[g]
				at = found['offset'] if extents is None else _moved(found['offset'],
					found['bytes'], extents)
				runs.append([[int(r['first']), int(r['last']), int(at[k]), int(r[
					'bytes'])] for k, r in enumerate(found)])
				blocks.append([])
			summary = source._tile_sums_of(g)
			sums.append([] if summary is None else [summary])

		def splice(own, new):
			return new + own if first else own + new

		self.tracks = splice(self.tracks, records)
		self._track_ids = {track['name']: i for i, track in enumerate(self.tracks)}
		self._stats = numpy.concatenate([stats, self._stats] if first else [
			self._stats, stats])
		self._group_of = numpy.concatenate([source._group_of + base_group,
			self._group_of] if first else [self._group_of, source._group_of +
			base_group])
		self._slot_of = numpy.concatenate([source._slot_of, self._slot_of] if
			first else [self._slot_of, source._slot_of])
		self._groups = splice(self._groups, groups)
		self._classes = splice(self._classes, classes)
		self._live = splice(self._live, live)
		self._runs = splice(self._runs, runs)
		self._blocks = splice(self._blocks, blocks)
		self._sums = splice(self._sums, sums)
		for name in ('_pending', '_buffers', '_open', '_sealed', '_last'):
			setattr(self, name, splice(getattr(self, name), [None] * n_groups))
		self._done = splice(self._done, [set() for _ in range(n_groups)])
		self._adopted |= set(range(base_group, base_group + n_groups))
		if any(not sums[k] for k in range(n_groups)) or len(self.tracks) * \
				self._n_tiles * 4 > _SUMMARY_BYTES:
			self._summaries = False

	def _copy_chunks(self, source):
		"""Copy every chunk of `source` into the file, each stretch of
		adjacent chunks in one piece, and return where the stretches lay and
		landed, for `_moved`: a file appended to in place holds its old
		tables and index between its chunks, which the copy leaves out."""

		ranges = []
		for g in range(len(source._groups)):
			index = source._group_index(g)
			found = index if source.order == 'tracks' else source._runs[g]
			ranges.append(numpy.stack([found['offset'].astype(numpy.int64), found[
				'size' if source.order == 'tracks' else 'bytes'].astype(
				numpy.int64)], axis=1))
		ranges = numpy.concatenate(ranges) if ranges else numpy.zeros((0, 2),
			dtype=numpy.int64)
		ranges = ranges[ranges[:, 1] > 0]
		ranges = ranges[numpy.argsort(ranges[:, 0], kind='stable')]
		starts, ends = ranges[:, 0], ranges[:, 0] + ranges[:, 1]
		if (starts[1:] < ends[:-1]).any():
			raise ValueError("{} cannot be read as a figwig file: two of its "
				"chunks overlap.".format(source.path))
		new = numpy.concatenate([[True], starts[1:] > ends[:-1]]) if len(
			starts) else numpy.zeros(0, dtype=numpy.bool_)
		firsts = numpy.flatnonzero(new)
		starts = starts[firsts]
		if len(firsts):
			ends = ends[numpy.append(firsts[1:], len(ends)) - 1]
		dests = numpy.zeros(len(starts), dtype=numpy.int64)

		fd = _open_read(source.path)
		try:
			for k in range(len(starts)):
				dests[k] = self._file.tell()
				position = int(starts[k])
				while position < ends[k]:
					piece = _pread(fd, min(1 << 24, int(ends[k]) - position), position)
					if not piece:
						raise ValueError("{} ends inside its chunks.".format(
							source.path))
					self._file.write(piece)
					position += len(piece)
		finally:
			os.close(fd)

		return starts, dests

	def _check_chroms(self, chroms, n):
		names, codes = _chrom_codes(chroms, n)
		unknown = [name for name in names if name not in self._tids]
		if len(unknown) > 0:
			raise ValueError("chromosome {!r} is not one of the writer's "
				"chromosomes.".format(unknown[0]))

		tids = numpy.array([self._tids[name] for name in names],
			dtype=numpy.int64)[codes]
		return names, codes, tids

	def _check_order_of(self, g, tids, starts, ends, bounds, describe, order,
		entries=False):
		"""Raise if a group's items on the chromosomes of a call, sorted,
		come before what was written to it earlier, or, for entries, fall in
		a tile that `write` left open."""

		last, done, sealed = self._last[g], self._done[g], self._sealed[g]
		open_ = self._open[g]
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
			if entries and open_ is not None and tid == open_[0] and starts[a] \
					// self.tile == open_[1]:
				raise ValueError("tile {} of {!r} was begun by write, and "
					"write_sparse cannot add to it; the windows and nonzero bases "
					"of one tile must come in one call.".format(open_[1],
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

		A base that is not written is 0. NaN is a value for a float track,
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
			whole groups. None is every track given to this writer, in order,
			leaving out those that came from another file. Default is None.
		"""

		if self._file is None:
			raise ValueError("the writer is closed.")

		ids, single = self._select(tracks)
		if len(ids) == 0:
			raise ValueError("tracks must name at least one track.")
		groups = self._whole_groups(ids)

		values = numpy.asarray(values)
		_check_exact(values)
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
				not in (None, 'float32', 'float64') for members in groups.values()
				for i in members):
			inside = numpy.arange(values.shape[2])[None, :] < (ends - starts)[:,
				None]
		for g, members in groups.items():
			for i, channel in members.items():
				if self.tracks[i]['dtype'] in (None, 'float32', 'float64'):
					continue
				block = values[:, channel]
				if inside is not None:
					block = block[inside]
				_check_values_fit(self.tracks[i]['name'], self.tracks[i]['dtype'],
					block)

		self._written = True
		if self.order == 'tracks':
			for g, members in groups.items():
				(i, channel), = members.items()
				for a, b in zip(bounds[:-1], bounds[1:]):
					positions, entry_values = _items_to_entries(starts[a:b],
						ends[a:b], values[a:b, channel])
					self._write_track_entries(i, int(tids[a]), positions,
						entry_values, int(ends[b - 1]))
			self._write(max(0, len(self._inflight) - (0 if self._pool is None
				else 2 * self.n_jobs)))
			return

		# The encoders take float32 and integers as they are, and other types
		# as float64. The conversion to that type is made on the pool, by
		# each task for its own tiles, so that this thread does no pass over
		# the values at the kernels' width; what it does is copy the block
		# in its own type, since encoding goes on after this call returns
		# and the caller may then change `values`. A caller that never
		# touches its arrays again, as the converters are, turns the copy
		# off with `_copies`, and each call then costs no pass at all here.
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
					# Intervals: one value per track, converted here, since
					# the block is small.
					block = numpy.ascontiguousarray(block[:, :, None], dtype=dtype)
				elif self._copies and self._pool is not None:
					block = self._copy(block)
				else:
					# Without a pool every task runs inside this call, so the
					# caller's array is not read after it returns.
					block = numpy.ascontiguousarray(block)
				self._write_group(g, int(tids[a]), starts[a:b], ends[a:b], block,
					values.ndim == 3, dtype)

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
		may name one base of one track. A call that raises writes nothing.

		One call takes about 90 bytes of memory per entry besides the arrays
		given, so a genome's entries are best written a chromosome at a
		time.

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
		_check_exact(values)
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
			(None, 'float32', 'float64')]
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
		# group wrote before, all of them before any is encoded.
		bounds = numpy.concatenate([[0], numpy.flatnonzero((tids[1:] != tids[
			:-1]) | (group_ids[1:] != group_ids[:-1])) + 1, [n]])
		for a, b in zip(bounds[:-1], bounds[1:]):
			low = int(positions[a:b].min())
			high = int(positions[a:b].max())
			self._check_order_of(int(group_ids[a]), tids[a:a + 1], numpy.array([
				low]), numpy.array([high + 1]), numpy.array([0, 1]), lambda j, c=a:
				describe(int(order[c])), None, entries=self.order == 'positions')

		self._written = True
		for a, b in zip(bounds[:-1], bounds[1:]):
			g, tid = int(group_ids[a]), int(tids[a])
			if self.order == 'tracks':
				self._write_track_entries(int(self._groups[g][0]), tid,
					positions[a:b], values[a:b], int(positions[b - 1]) + 1)
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
		# One key of the two sorts in a third of a lexsort's time where the
		# chromosome and group fit above the position's bits.
		high_bits = int(high.max()).bit_length()
		if longest + 25 + high_bits <= 63:
			return numpy.argsort((high << (longest + 25)) | low, kind='stable')
		return numpy.lexsort((low, high))

	def _write_group(self, g, tid, starts, ends, values, by_base, dtype=None):
		"""Encode one group's items on one chromosome, in order, into chunks,
		and hand them to the compressors. `values` is the block in the
		caller's type, or the kernels' when `dtype` is None; `dtype` is the
		kernels' type for it, which the tasks convert their tiles to."""

		if dtype is None:
			dtype = values.dtype

		tile = self.tile
		n_tracks = len(self._groups[g])
		if self._buffers[g] is None:
			self._buffers[g] = numpy.zeros((n_tracks, tile), dtype=numpy.float64)

		first_tile = int(starts[0] // tile)
		open_ = self._open[g]
		if open_ is not None:
			# A tile the earlier calls left open is continued when it is the
			# first of this call's, and sealed otherwise.
			if open_[0] != tid or open_[1] != first_tile:
				self._encode(g, open_[0], numpy.array([open_[1]]), None, None,
					None, False)
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
		edges = None
		for a, b in zip(cuts[:-1], cuts[1:]):
			if is_full[a]:
				items = full_items[numpy.searchsorted(full_tiles, tiles[a:b])]
				columns = tiles[a:b] * tile - starts[items]
				for e in range(a, b, per_batch):
					f = min(b, e + per_batch)
					self._encode_dense(g, tid, values, items[e - a:f - a],
						columns[e - a:f - a], tiles[e:f], dtype)
			else:
				if edges is None:
					edges = self._edges(starts, ends, values, by_base, dtype)
				e_starts, e_ends = edges[0], edges[1]
				segment = tiles[a:b]
				lo = numpy.searchsorted(e_ends, segment * tile, side='right')
				hi = numpy.searchsorted(e_starts, (segment + 1) * tile,
					side='left')
				for e in range(0, len(segment), per_batch):
					f = min(len(segment), e + per_batch)
					keep_last = b == len(tiles) and f == len(segment) and \
						not last_full
					self._encode(g, tid, segment[e:f], lo[e:f], hi[e:f], edges,
						keep_last)

		if not last_full:
			self._open[g] = (tid, last_tile)

		self._note_written(g, tid, int(ends[-1]))

	def _copy(self, block):
		"""A C-contiguous copy of a block in its own type, made in slices on
		the pool when the block is large, since one thread's memcpy would
		otherwise be most of what a write of many tracks costs here."""

		if self._pool is None or block.nbytes < _COPY_ON_POOL or block.shape[
				0] * block.shape[1] < 2:
			return numpy.array(block, order='C')
		out = numpy.empty(block.shape, dtype=block.dtype)
		flat_in = block.reshape(-1, block.shape[2]) if block.flags.c_contiguous \
			else None
		rows = block.shape[0] * block.shape[1]
		parts = min(self.n_jobs, rows)
		cuts = numpy.linspace(0, rows, parts + 1).astype(numpy.int64)
		flat_out = out.reshape(-1, block.shape[2])

		def copy(k):
			a, b = int(cuts[k]), int(cuts[k + 1])
			if flat_in is not None:
				flat_out[a:b] = flat_in[a:b]
			else:
				for r in range(a, b):
					flat_out[r] = block[r // block.shape[1], r % block.shape[1]]

		list(self._pool.map(copy, range(parts)))
		return out

	def _edges(self, starts, ends, values, by_base, dtype):
		"""The items of a block that the buffer encoder sees, in the
		kernels' type: for windows, only the parts outside their whole tiles
		(a window's first and last partial tile, or the whole of a window
		that covers no tile), as items of at most two tiles' width, so that
		the thread converts the edges of the windows rather than the block;
		intervals and blocks already of the kernels' type as they are."""

		small = values.dtype == dtype and values.shape[2] <= 2 * self.tile
		if not by_base or small:
			return (starts, ends, numpy.ascontiguousarray(values, dtype=dtype),
				by_base)
		tile = self.tile
		width = values.shape[2]
		f0 = -(-starts // tile)
		f1 = ends // tile - 1
		whole = f1 >= f0
		if not whole.any():
			return (starts, ends, numpy.ascontiguousarray(values, dtype=dtype),
				by_base)
		e_starts, e_ends, e_values = [], [], []
		edge = min(width, 2 * tile)
		for j in numpy.flatnonzero(whole).tolist():
			s, e = int(starts[j]), int(ends[j])
			left, right = int(f0[j]) * tile, int(f1[j] + 1) * tile
			if s < left:
				e_starts.append(s)
				e_ends.append(left)
				e_values.append(values[j, :, :left - s])
			if right < e:
				e_starts.append(right)
				e_ends.append(e)
				e_values.append(values[j, :, right - s:e - s])
		for j in numpy.flatnonzero(~whole).tolist():
			e_starts.append(int(starts[j]))
			e_ends.append(int(ends[j]))
			e_values.append(values[j, :, :int(ends[j] - starts[j])])
		order = numpy.argsort(e_starts, kind='stable')
		out = numpy.zeros((len(order), values.shape[1], edge), dtype=dtype)
		for k, j in enumerate(order.tolist()):
			piece = e_values[j]
			out[k, :, :piece.shape[1]] = piece
		return (numpy.array(e_starts, dtype=numpy.int64)[order], numpy.array(
			e_ends, dtype=numpy.int64)[order], out, True)

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
				False)
			self._open[g] = None

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
		"""Encode and compress one batch of tiles from their entries. The
		encoder stops before a tile that might not fit its buffer, which
		holds at most about _BATCH_BYTES, and is called again from there."""

		n = len(tiles)
		n_entries = len(positions)
		largest = int(numpy.diff(first).max())
		need = max(4 + 4 * n_tracks + 11 * largest + 16, 24 + 18 * largest + 16)
		bound = n * (4 * n_tracks + 24) + 18 * n_entries + 16
		stats = numpy.zeros((n_tracks, 6), dtype=numpy.float64)
		stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf
		tile_sums = numpy.zeros((n, n_tracks), dtype=numpy.float32)
		parts = []
		t = 0
		while t < n:
			# A buffer per go: an uncompressed part is a view of its buffer,
			# which the next go must not write over.
			out = numpy.empty(min(bound, _BATCH_BYTES + need), dtype=numpy.uint8)
			bounds = numpy.zeros(n - t + 1, dtype=numpy.int64)
			kinds = numpy.zeros(n - t, dtype=numpy.uint8)
			done = _encode_entries(self.tile, n_tracks, tiles[t:], first[t:],
				slots, positions, values, self._classes[g], self._matrix,
				self.level > 0, out, bounds, kinds, stats, tile_sums[t:])
			parts.append(self._compress(tid, tiles[t:], done, out, bounds,
				kinds))
			t += done
		return _gather_parts(parts) + (stats, self._summary_of(tid, tiles,
			tile_sums))

	def _encode_dense(self, g, tid, values, items, columns, tiles, dtype):
		"""Encode whole tiles covered by windows, each tile from window
		items[t] at column columns[t], on the pool when there is one, and
		compress them in the same task."""

		task = (self._dense_task, g, tid, values, items, columns, tiles, dtype)
		if self._pool is None:
			self._inflight.append((task[0](*task[1:]), g, None))
		else:
			self._inflight.append((self._pool.submit(*task), g, None))

	def _dense_task(self, g, tid, values, items, columns, tiles, dtype):
		"""Encode and compress one batch of whole tiles. Returns (compressed
		bytes, sizes, raw sizes, kind bytes, file tiles, stats) of the live
		tiles. A block that is not of the kernels' type has the batch's own
		tiles converted here, on the pool; one that is, as the converters'
		float32 windows are, is read in place."""

		n_tracks = len(self._groups[g])
		tile = self.tile
		n = len(tiles)
		if values.dtype != dtype:
			rows, items = numpy.unique(items, return_inverse=True)
			c0, c1 = int(columns.min()), int(columns.max()) + tile
			if len(rows) == 1:
				# One window's tiles: a view, converted in one pass.
				values = numpy.ascontiguousarray(values[int(rows[0]), :, c0:c1],
					dtype=dtype)[None]
			else:
				converted = numpy.empty((len(rows), n_tracks, c1 - c0),
					dtype=dtype)
				for k, r in enumerate(rows.tolist()):
					converted[k] = values[r, :, c0:c1]
				values = converted
			columns = columns - c0
		items = numpy.ascontiguousarray(items, dtype=numpy.int64)
		stats = numpy.zeros((n_tracks, 6), dtype=numpy.float64)
		stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf
		tile_sums = numpy.zeros((n, n_tracks), dtype=numpy.float32)
		parts = []
		t = 0
		# The buffer holds every tile at four bytes per value and one more
		# at eight; a batch with float64 chunks is encoded in several goes.
		while t < n:
			out = numpy.empty((n - t) * (n_tracks * tile * 4 + 16) + n_tracks *
				tile * 4, dtype=numpy.uint8)
			bounds = numpy.zeros(n - t + 1, dtype=numpy.int64)
			kinds = numpy.zeros(n - t, dtype=numpy.uint8)
			done = _encode_dense_tiles(tile, values, items[t:], columns[t:],
				self._classes[g], self._matrix, self.level > 0, out, bounds, kinds,
				stats, tile_sums[t:])
			parts.append(self._compress(tid, tiles[t:], done, out, bounds,
				kinds))
			t += done
		return _gather_parts(parts) + (stats, self._summary_of(tid, tiles,
			tile_sums))

	def _encode(self, g, tid, tiles, lo, hi, items, keep_last):
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
		stats = self._stats[self._groups[g]]
		tile_sums = numpy.zeros((n, n_tracks), dtype=numpy.float32)
		expected = n - 1 if keep_last else n
		t = 0
		while True:
			# The encoder runs at least once, to fill the buffer with a tile
			# it keeps. The buffer holds every tile at four bytes per value
			# and one more at eight; the encoder stops before a tile that
			# might not fit, and is called again from there with the buffer
			# clear.
			out = numpy.empty((n - t) * (n_tracks * tile * 4 + 16) + n_tracks *
				tile * 4, dtype=numpy.uint8)
			bounds = numpy.zeros(n - t + 1, dtype=numpy.int64)
			kinds = numpy.zeros(n - t, dtype=numpy.uint8)
			n_done = _encode_tiles(tile, tiles[t:], lo[t:], hi[t:], starts, ends,
				values, by_base, self._buffers[g], keep_last,
				self._classes[g], self._matrix, self.level > 0, out, bounds, kinds,
				stats, tile_sums[t:])
			live = numpy.flatnonzero(kinds[:n_done])
			if len(live) > 0:
				args = (tid, tiles[t:], n_done, out, bounds, kinds)
				if self._pool is None:
					self._inflight.append((self._compress(*args), g, None))
				else:
					self._inflight.append((self._pool.submit(self._compress, *args),
						g, None))
			t += n_done
			if t >= expected:
				break
			if n_done == 0:
				raise RuntimeError("a tile did not fit an empty encode buffer.")
		self._stats[self._groups[g]] = stats
		self._check_overflow(self._groups[g])
		summary = self._summary_of(tid, tiles[:expected], tile_sums[:expected])
		if summary is not None:
			self._sums[g].append(summary)

	def _summary_of(self, tid, tiles, tile_sums):
		"""The file tiles of these tiles and their tracks' sums, for the
		summaries, or None when the file keeps none."""

		if not self._summaries:
			return None
		live = numpy.flatnonzero(tile_sums.any(axis=1))
		return self._tile_offsets[tid] + tiles[live], tile_sums[live]

	def _summary_record(self, g):
		"""The summary of group g for the index: the tiles with a nonzero
		sum and each track's sum over them, deflated, or an empty record."""

		if not self._summaries:
			return struct.pack('<II', 0, 0) + struct.pack('<I', 0)
		n_tracks = len(self._groups[g])
		tiles = numpy.concatenate([t for t, _ in self._sums[g]] + [numpy.zeros(0,
			dtype=numpy.int64)])
		sums = numpy.concatenate([v for _, v in self._sums[g]] + [numpy.zeros((0,
			n_tracks), dtype=numpy.float32)]).astype(numpy.float64)
		if len(tiles) == 0:
			stream = zlib.compress(b'', 6)
			return struct.pack('<II', 0, len(stream)) + stream + \
				self._zoom_section(tiles, sums)
		order = numpy.argsort(tiles, kind='stable')
		tiles, sums = tiles[order], sums[order]
		first = numpy.concatenate([[True], tiles[1:] != tiles[:-1]])
		starts = numpy.flatnonzero(first)
		sums = numpy.add.reduceat(sums, starts, axis=0)
		tiles = tiles[starts]
		stream = zlib.compress(tiles.astype('<u4').tobytes() + sums.astype(
			'<f4').tobytes(), 6)
		return struct.pack('<II', len(tiles), len(stream)) + stream + \
			self._zoom_section(tiles, sums)

	def _zoom_section(self, tiles, sums):
		"""The zoom levels of a group, from its tile sums (file tiles
		ascending, sums (tiles, tracks) float64): the level count, then per
		level its factor, bin count, stream length and CRC-32, and a zlib
		stream of bin u4 + f8 sums per track over the bins with a sum."""

		parts = [struct.pack('<I', self.zooms)]
		if self.zooms and len(tiles):
			tids = numpy.searchsorted(self._tile_offsets, tiles, side='right') - 1
			local = tiles - self._tile_offsets[tids]
		for k in range(self.zooms):
			factor = 4 ** (k + 1)
			if len(tiles):
				bins = self._zoom_grids[k][0][tids] + local // factor
				first = numpy.flatnonzero(numpy.concatenate([[True], bins[1:] !=
					bins[:-1]]))
				zsums = numpy.add.reduceat(sums, first, axis=0)
				bins = bins[first]
			else:
				bins, zsums = tiles, sums
			stream = zlib.compress(bins.astype('<u4').tobytes() + zsums.astype(
				'<f8').tobytes(), 6)
			parts.append(struct.pack('<IIII', factor, len(bins), len(stream),
				zlib.crc32(stream)) + stream)
		return b''.join(parts)

	def _check_overflow(self, rows):
		flagged = numpy.flatnonzero(self._stats[rows, 4].astype(numpy.int64) &
			_FLAG_OVERFLOW)
		if len(flagged) > 0:
			i = int(rows[flagged[0]])
			raise ValueError("a value of track {!r} lies beyond float32's range; "
				"declare the track float64 to keep it.".format(self.tracks[i][
				'name']))

	def _compress(self, tid, tiles, n_done, out, bounds, kinds):
		"""Compress the first `n_done` encoded tiles of a batch. Returns
		(compressed bytes, sizes, raw sizes, kind bytes, file tiles) of the
		live tiles. A chunk that compression would grow is stored as it is,
		and its kind byte says so."""

		live = numpy.flatnonzero(kinds[:n_done])
		gtiles = self._tile_offsets[tid] + tiles[:n_done][live]
		if len(live) == 0:
			empty = numpy.zeros(0, dtype=numpy.int64)
			return numpy.zeros(0, dtype=numpy.uint8), empty, empty, kinds[:0], \
				gtiles, empty

		cuts = numpy.append(bounds[live], bounds[n_done])
		raw = numpy.diff(cuts)
		data = out[:bounds[n_done]]
		kinds = kinds[:n_done][live]
		if self.level == 0:
			return data, raw, raw, kinds, gtiles, _crcs(data, cuts)

		compressed, sizes = _compress_blocks(self.engine, self.level, data, cuts)
		# The streams lie at the front of a larger buffer; a batch encoded
		# in several goes is joined from its parts, so each is cut to its
		# streams here.
		compressed = compressed[:int(sizes.sum())]
		grown = numpy.flatnonzero(sizes >= raw)
		if len(grown) == 0:
			return compressed, sizes, raw, kinds, gtiles, _crcs(compressed,
				numpy.concatenate([[0], numpy.cumsum(sizes)]))

		# The streams are laid out back to back; those that grew are
		# replaced by their chunks' own bytes.
		kinds = kinds.copy()
		kinds[grown] |= _STORED
		ends = numpy.cumsum(sizes)
		pieces = []
		for k in range(len(sizes)):
			if sizes[k] >= raw[k]:
				pieces.append(data[cuts[k]:cuts[k + 1]])
				sizes[k] = raw[k]
			else:
				pieces.append(compressed[ends[k] - sizes[k]:ends[k]])
		stream = numpy.concatenate(pieces)
		return stream, sizes, raw, kinds, gtiles, _crcs(stream,
			numpy.concatenate([[0], numpy.cumsum(sizes)]))

	def _write_track_entries(self, t, tid, positions, values, end):
		"""Add one track's nonzero bases on one chromosome, in position
		order, to its pending block, and encode the blocks that are full.
		`end` is where the items the entries came from end, which later
		items on the chromosome must come after."""

		pending = self._pending[t]
		if pending is not None and pending[0] != tid:
			self._flush_track(t)
			pending = None
		if pending is None:
			pending = self._pending[t] = [tid, [], [], 0]

		keep = values != 0
		if not keep.all():
			positions, values = positions[keep], values[keep]
		if len(positions) > 0:
			pending[1].append(positions)
			pending[2].append(values.astype(numpy.float64, copy=False))
			pending[3] += len(positions)
			if pending[3] >= _BLOCK_ENTRIES:
				self._flush_track(t, final=False)

		self._note_written(t, tid, end)

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
		stats = numpy.zeros((1, 6), dtype=numpy.float64)
		stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf
		_encode_blocks(positions, values, first, int(self._classes[t][0]), out,
			bounds, kinds, stats)
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
		records['crc'] = _crcs(compressed, numpy.concatenate([[0], numpy.cumsum(
			sizes)]))
		summary = None
		if self._summaries:
			tiles = positions // self.tile
			unique, inverse = numpy.unique(tiles, return_inverse=True)
			# NaN bases are no part of the sums, as in position order.
			sums = numpy.bincount(inverse, weights=numpy.where(values == values,
				values, 0.0), minlength=len(unique))
			summary = (self._tile_offsets[tid] + unique, sums.astype(
				numpy.float32)[:, None])
		return compressed, sizes, raw, kinds, records, None, stats, summary

	def _write(self, n_batches):
		"""Write the first `n_batches` batches in flight to the file, in
		order, record their chunks or blocks in the index, and fold in the
		stats of those encoded on the pool."""

		for result, g, kind in self._inflight[:n_batches]:
			if not isinstance(result, tuple):
				result = result.result()
			if len(result) == 8:
				stats, summary = result[6], result[7]
				rows = self._groups[g]
				_fold_sums(self._stats, rows, stats[:, 0] + stats[:, 5])
				self._stats[rows, 1] += stats[:, 1]
				self._stats[rows, 2] = numpy.minimum(self._stats[rows, 2],
					stats[:, 2])
				self._stats[rows, 3] = numpy.maximum(self._stats[rows, 3],
					stats[:, 3])
				self._stats[rows, 4] = (self._stats[rows, 4].astype(numpy.int64) |
					stats[:, 4].astype(numpy.int64)).astype(numpy.float64)
				self._check_overflow(rows)
				if summary is not None:
					self._sums[g].append(summary)

			compressed, sizes, raw, kinds, where, crcs = result[:6]
			if len(where) == 0:
				continue

			offset = self._file.tell()
			self._file.write(compressed[:sizes.sum()])
			total = int(sizes.sum())

			if kind == 'blocks':
				where['offset'] = offset + numpy.cumsum(sizes) - sizes
				self._blocks[g].append(where)
				continue

			live = numpy.zeros(len(where), dtype=_LIVE)
			live['tile'] = where
			live['kind'] = kinds
			live['size'] = sizes
			live['raw'] = raw
			live['crc'] = crcs
			self._live[g].append(live)

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
		the header, close the file and move it to its path. Calling it again
		does nothing."""

		if self._file is None:
			return

		try:
			if len(self.tracks) == 0:
				raise ValueError("the writer has no tracks; give some, or add a "
					"file's.")
			for g, open_ in enumerate(self._open):
				if open_ is not None:
					self._encode(g, open_[0], numpy.array([open_[1]]), None, None,
						None, False)
					self._open[g] = None
			for t in range(len(self._pending)):
				if self._pending[t] is not None:
					self._flush_track(t)
			self._write(len(self._inflight))

			for i, track in enumerate(self.tracks):
				if track['dtype'] is None:
					sum_, nonzero, low, high, flags, _ = self._stats[i]
					track['dtype'] = _infer_dtype(nonzero, low, high, flags)

			chroms_offset = self._file.tell()
			chrom_table = b''.join(struct.pack('<H', len(name.encode())) +
				name.encode() + struct.pack('<Q', length) for name, length in
				self._chrom_list)

			tracks_offset = chroms_offset + len(chrom_table)
			stats = numpy.zeros(len(self.tracks), dtype=_STATS)
			stats['sum'] = self._stats[:, 0] + self._stats[:, 5]
			stats['nonzero'] = self._stats[:, 1]
			stats['min'] = numpy.where(self._stats[:, 1] > 0, self._stats[:, 2], 0)
			stats['max'] = numpy.where(self._stats[:, 1] > 0, self._stats[:, 3], 0)

			metadata_offset = tracks_offset + stats.nbytes
			from . import __version__
			metadata = _json({**self._metadata, 'figwig': __version__, 'genome':
				self.genome, 'attributes': self.attributes, 'order': self.order,
				'tracks': self.tracks}, "the metadata").encode()

			# Each group's record carries the CRC-32 of its index bytes, and
			# the header that of the tables and the directory, so that a
			# reader refuses a damaged table rather than reading by it.
			index_offset = metadata_offset + len(metadata)

			def record_of(g):
				if self.order == 'tracks':
					blocks = numpy.concatenate(self._blocks[g]) if self._blocks[g] \
						else numpy.zeros(0, dtype=_BLOCK)
					body = blocks.tobytes()
					record = struct.pack('<II', len(blocks), zlib.crc32(body)) + body
				else:
					runs = numpy.zeros(len(self._runs[g]), dtype=_RUN)
					for r, run in enumerate(self._runs[g]):
						runs[r] = tuple(run)
					live = numpy.concatenate(self._live[g]) if self._live[g] else \
						numpy.zeros(0, dtype=_LIVE)
					live = live[numpy.argsort(live['tile'], kind='stable')]
					stream = zlib.compress(b''.join(live[field].tobytes() for field
						in _LIVE_FIELDS), 6)
					body = runs.tobytes() + struct.pack('<II', len(live), len(stream)) \
						+ stream
					record = struct.pack('<II', len(runs), zlib.crc32(body)) + body
				return record + self._summary_record(g)

			# Each group's record deflates its live tiles and summary, which
			# for thousands of groups is seconds on one thread.
			if self._pool is None:
				records = [record_of(g) for g in range(len(self._groups))]
			else:
				records = list(self._pool.map(record_of, range(len(self._groups))))
			directory = numpy.zeros(len(self._groups), dtype='<u8')
			position = index_offset + directory.nbytes
			for g, record in enumerate(records):
				directory[g] = position
				position += len(record)

			tables = chrom_table + stats.tobytes() + metadata + directory.tobytes()
			self._file.write(tables)
			for record in records:
				self._file.write(record)
			length = self._file.tell()

			header = numpy.zeros(1, dtype=_HEADER)
			header['magic'] = _FIGWIG_MAGIC
			header['version'] = _VERSION
			header['flags'] = _FLAG_TRACKS if self.order == 'tracks' else 0
			header['optional'] = _OPTIONAL_ZOOMS
			header['tile'] = self.tile
			header['codec'] = _CODEC_NONE if self.level == 0 else _CODEC_ZLIB
			header['level'] = self.level
			header['n_chroms'] = len(self._chrom_list)
			header['n_tracks'] = len(self.tracks)
			header['n_groups'] = len(self._groups)
			header['tables_crc'] = zlib.crc32(tables)
			header['chroms'] = chroms_offset
			header['tracks'] = tracks_offset
			header['index'] = index_offset
			header['data'] = _HEADER.itemsize
			header['metadata'] = metadata_offset
			header['metadata_length'] = len(metadata)
			header['length'] = length
			header['header_crc'] = _header_crc(header.tobytes())
			if self._length is not None:
				# The header is the one write over the file as it was, so
				# everything it points to is on disk before it.
				self._file.flush()
				os.fsync(self._file.fileno())
			self._file.seek(0)
			self._file.write(header.tobytes())
			self._file.flush()
			self._file.close()
			self._file = None
			self._length = None
			if self._partial is not None:
				os.replace(self._partial, self.path)
				self._partial = None
		finally:
			self._abandon()


def _moved(offsets, sizes, extents):
	"""Where chunks at `offsets` of `sizes` bytes lie after `_copy_chunks`
	moved the stretches `extents`; an empty chunk's offset is the header's
	end."""

	starts, dests = extents
	offsets = numpy.asarray(offsets, dtype=numpy.int64)
	if len(starts) == 0:
		return numpy.full(len(offsets), _HEADER.itemsize, dtype=numpy.int64)
	k = numpy.searchsorted(starts, offsets, side='right') - 1
	moved = dests[numpy.maximum(k, 0)] + offsets - starts[numpy.maximum(k, 0)]
	return numpy.where(numpy.asarray(sizes) > 0, moved, _HEADER.itemsize)


def _crcs(data, cuts):
	"""The CRC-32 of each piece data[cuts[k]:cuts[k + 1]], as uint32."""

	view = memoryview(data)
	return numpy.array([zlib.crc32(view[int(cuts[k]):int(cuts[k + 1])]) for k
		in range(len(cuts) - 1)], dtype=numpy.int64)


def _gather_parts(parts):
	"""The results of one batch's encodings, each (bytes, sizes, raws,
	kinds, tiles, crcs), as one."""

	if len(parts) == 1:
		return parts[0]
	return tuple(numpy.concatenate([part[k] for part in parts]) for k in
		range(6))


def _smallest_tile(lengths):
	"""The smallest tile at which a genome of these lengths has at most
	_MAX_TILES tiles."""

	tile = _MIN_TILE
	while tile < _MAX_TILE and _tile_offsets(lengths, tile)[1] > _MAX_TILES:
		tile *= 2
	return tile


def _fold_sums(stats, rows, values):
	"""Add `values` into the sums of `rows` of a (tracks, 6) statistics
	array with Neumaier's compensation, kept in column 5."""

	total = stats[rows, 0]
	t = total + values
	stats[rows, 5] += numpy.where(numpy.abs(total) >= numpy.abs(values),
		(total - t) + values, (values - t) + total)
	stats[rows, 0] = t


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


def _open_read(path):
	"""A descriptor on `path` for reading, which must be a regular file:
	a directory, a FIFO or a device would hang or fail inside a read."""

	try:
		if not stat.S_ISREG(os.stat(path).st_mode):
			raise ValueError("{} is not a regular file.".format(path))
		return os.open(path, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
	except OSError as e:
		raise ValueError("{} cannot be read: {}".format(path, e.strerror or
			e)) from None


class FigWigReader:
	"""A figwig file, read many windows and tracks at a time on several
	threads.

	`FigWigReader` reads the per-base values of many windows of one width,
	from any of the file's tracks, in one call, into an array of shape
	(windows, tracks, width): the (batch, channels, length) layout that
	sequence models take. It is built for loading training data from many
	tracks at once, such as the counts of every assay of a cell type under
	a batch of windows. The same read gives the nonzero bases as a sparse
	list with `sparse=True`, the tracks summed per label with `labels=`,
	which is what a file of thousands of cells is read as, or each window
	summed per bin of `bins` bases, which is what a model of binned
	coverage trains on.

	The file's tables, metadata and the directory of its index are read when
	it is opened, and each group's index on its first read, after which the
	index is kept. In position order a read sorts its windows, finds the
	chunks they overlap by division on the tile grid, and works through
	those chunks in batches; in track order it finds each requested track's
	blocks over the windows' chromosomes and searches them. Each batch is
	read from the file, decompressed and decoded straight into the windows'
	rows of the output, on up to `n_jobs` threads, as many as the batch
	count warrants; zlib's `uncompress()` and the numba decoders run without
	the GIL. Values come back in each track's dtype, or in the one asked for.

	A `FigWigReader` holds no open file between reads, checks on every read
	that the file is the one it opened, can be read from several Python
	threads at once, and can be pickled, so it can be part of a PyTorch
	Dataset read by DataLoader workers; a pickled reader carries no index,
	and reads it again in the process that unpickles it.

	Parameters
	----------
	path: str or os.PathLike
		The path to a figwig file.

	verify: bool, optional
		Whether to check every chunk read against the CRC-32 the file keeps
		for it, which a compressed chunk's zlib checksum covers already and
		an uncompressed chunk does not. Default is False.

	Attributes
	----------
	path: str
		The absolute path of the file.

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
		and 'max', over the bases that are not NaN: a NaN base is stored
		and read back but is no part of the statistics, so a track whose
		uncovered bases are NaN keeps those of its covered ones.
	"""

	def __init__(self, path: str | os.PathLike, verify: bool = False):
		self.path = os.path.abspath(os.fspath(path))
		self.verify = bool(verify)
		error = "{} cannot be read as a figwig file".format(self.path)

		fd = _open_read(self.path)
		try:
			stat = os.fstat(fd)
			file_size = stat.st_size
			raw = _pread(fd, _HEADER.itemsize, 0)
			if len(raw) < _HEADER.itemsize:
				raise ValueError(error + ": it is too short to hold a header.")

			header_bytes = bytes(raw)
			header = numpy.frombuffer(raw, dtype=_HEADER)[0]
			if int(header['magic']) != _FIGWIG_MAGIC:
				raise ValueError("{} is not a figwig file.".format(self.path))
			if _header_crc(header_bytes) != int(header['header_crc']) and int(
					header['version']) == _VERSION:
				raise ValueError(error + ": its header does not match its "
					"checksum; the file is corrupt.")
			if int(header['version']) != _VERSION:
				raise ValueError(error + ": it is version {} of the format, and "
					"this figwig reads version {}; {}".format(int(header[
					'version']), _VERSION, "a newer figwig wrote it." if int(header[
					'version']) > _VERSION else "the file is corrupt."))

			if int(header['required']) & ~_KNOWN_REQUIRED:
				raise ValueError(error + ": it needs a feature this figwig does not "
					"know (bits {:#x}); a newer figwig wrote it.".format(int(header[
					'required']) & ~_KNOWN_REQUIRED))
			if int(header['flags']) & ~_FLAG_TRACKS:
				raise ValueError(error + ": its flags hold a bit this figwig does "
					"not know; a newer figwig wrote it.")
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
			if any(not _HEADER.itemsize <= o <= length for o in offsets) or any(
					a > b for a, b in zip(offsets[:-1], offsets[1:])):
				raise ValueError(error + ": a table lies outside the file.")

			n_chroms, n_tracks, n_groups = (int(header[key]) for key in (
				'n_chroms', 'n_tracks', 'n_groups'))
			if n_chroms == 0 or n_tracks == 0 or n_groups == 0:
				raise ValueError(error + ": it has no chromosomes, tracks or "
					"groups.")
			if 8 * n_groups > length - offsets[3]:
				raise ValueError(error + ": the file ends inside its index.")
			# The tables and the index's directory are checked against the
			# header's CRC-32 before anything is taken from them, so a flipped
			# chromosome length or track record is refused rather than read.
			tables = _pread(fd, offsets[3] + 8 * n_groups - offsets[0], offsets[0])
			if len(tables) != offsets[3] + 8 * n_groups - offsets[0]:
				raise ValueError(error + ": the file ends inside its tables.")
			if zlib.crc32(tables) != int(header['tables_crc']):
				raise ValueError(error + ": its tables do not match their "
					"checksum; the file is corrupt.")
			order = 'tracks' if int(header['flags']) & _FLAG_TRACKS else 'positions'
			if order == 'tracks' and n_groups != n_tracks:
				raise ValueError(error + ": a file in track order needs a group "
					"per track.")

			# Every count is checked against the file before its bytes are
			# asked for, so that a corrupt count cannot ask for memory the
			# file could not fill.
			chroms = self._read_chroms(fd, offsets[0], n_chroms, offsets[1],
				error)
			if n_tracks * _STATS.itemsize > length - offsets[1]:
				raise ValueError(error + ": the file ends inside its track table.")
			stats = _pread(fd, n_tracks * _STATS.itemsize, offsets[1])
			if len(stats) != n_tracks * _STATS.itemsize:
				raise ValueError(error + ": the file ends inside its track "
					"table.")
			stats = numpy.frombuffer(stats, dtype=_STATS).copy()

			metadata_length = int(header['metadata_length'])
			if metadata_length > length - offsets[2]:
				raise ValueError(error + ": its metadata lies outside the file.")
			raw = _pread(fd, metadata_length, offsets[2])
			try:
				metadata = json.loads(raw.decode())
			except (UnicodeDecodeError, ValueError, RecursionError):
				raise ValueError(error + ": its metadata is not a JSON "
					"document.") from None
			if not isinstance(metadata, dict) or not isinstance(metadata.get(
					'attributes', {}), dict) or metadata.get('attributes') is None:
				raise ValueError(error + ": its metadata is not a JSON object with "
					"attributes.")

			directory = tables[offsets[3] - offsets[0]:]
		finally:
			os.close(fd)

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
		self.written_by = metadata.get('figwig')
		self._metadata = metadata
		self.order = order
		self.stats = stats
		self.tile = tile
		self.chrom_sizes = dict(chroms)
		self._codec = codec
		self._level = int(header['level'])
		self._optional = int(header['optional'])
		self._file_size = length
		self._tables_at = offsets[0]
		self._identity = (stat.st_ino, stat.st_dev, header_bytes)
		self._directory = numpy.frombuffer(directory, dtype='<u8').astype(
			numpy.int64)
		self._track_ids = {track['name']: i for i, track in enumerate(tracks)}
		self._group_of = numpy.array([track['group'] for track in tracks],
			dtype=numpy.int64)
		by_group = numpy.argsort(self._group_of, kind='stable')
		counts = numpy.bincount(self._group_of, minlength=n_groups)
		if (counts == 0).any():
			raise ValueError(error + ": a group has no tracks.")
		self._groups = numpy.split(by_group, numpy.cumsum(counts)[:-1])
		self._slot_of = numpy.zeros(n_tracks, dtype=numpy.int64)
		for group in self._groups:
			self._slot_of[group] = numpy.arange(len(group))
		self._dtype_codes = numpy.array([_DTYPES.index(track['dtype']) for
			track in tracks], dtype=numpy.int64)

		self._tids = {name: i for i, (name, _) in enumerate(chroms)}
		self._lengths = numpy.array([length for _, length in chroms],
			dtype=numpy.int64)
		self._tile_offsets, self._n_tiles = _tile_offsets(self._lengths, tile)
		if self._n_tiles > _MAX_TILES:
			raise ValueError(error + ": its chromosomes make {:,} tiles of {} "
				"bases, more than a figwig file holds.".format(self._n_tiles, tile))
		self._index = [None] * n_groups
		self._summary_at = [None] * n_groups
		self._zoom_levels = [None] * n_groups
		self._zoom_factors = None
		self._runs = [None] * n_groups
		self._index_lock = threading.Lock()
		self._pool = None

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
			if len(name) == 0 or name in names or not 1 <= length <= \
					_MAX_CHROM_LENGTH:
				raise ValueError(error + ": its chromosome table is not a list of "
					"distinct names and lengths of up to 2**32 - 1.")
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
		state['_index'] = [None] * len(self._index)
		state['_summary_at'] = [None] * len(self._index)
		state['_zoom_levels'] = [None] * len(self._index)
		state['_runs'] = [None] * len(self._index)
		state['_pool'] = None
		state.pop('_pinned', None)
		return state

	def __setstate__(self, state):
		self.__dict__.update(state)
		self._index_lock = threading.Lock()

	def _check_file(self):
		"""Raise if the file at the path is no longer the one that was
		opened, which its inode and its header tell: a file replaced by a
		new write has another inode, one rewritten or appended to in place
		has another header once the writer closes, and until then the
		chunks this reader's index names are where they were."""

		try:
			stat = os.stat(self.path)
			fd = _open_read(self.path)
			try:
				raw = _pread(fd, _HEADER.itemsize, 0)
			finally:
				os.close(fd)
		except OSError as e:
			raise ValueError("{} cannot be read: {}".format(self.path,
				e.strerror)) from None
		if (stat.st_ino, stat.st_dev, raw) != self._identity:
			raise ValueError("{} has been replaced or rewritten since this reader "
				"opened it; open it again.".format(self.path))
		if stat.st_size < self._file_size:
			raise ValueError("{} has been truncated since this reader opened it."
				.format(self.path))

	def _threads(self, n_jobs, n_chunks):
		"""The threads a read of n_chunks chunks or blocks uses: one per
		_CHUNKS_PER_THREAD of them, up to n_jobs, since a small read gains
		nothing from threads and pays for waking them."""

		return max(1, min(n_jobs, n_chunks // _CHUNKS_PER_THREAD))

	def _pool_for(self, n_threads):
		"""A thread pool of at least n_threads threads, kept between reads,
		and made again in a process that forked with one."""

		pid = os.getpid()
		if self._pool is None or self._pool[0] != pid or self._pool[1] < \
				n_threads:
			self._pool = (pid, n_threads, ThreadPoolExecutor(n_threads))
		return self._pool[2]

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

		known = set()
		for track in self.tracks:
			known.update(track)
		for key in attributes:
			if key not in known:
				raise ValueError("no track has an attribute {!r}; the tracks' keys "
					"are {}.".format(key, ', '.join(sorted(known))))
		return numpy.array([i for i, track in enumerate(self.tracks) if all(
			key in track and track[key] == value for key, value in
			attributes.items())], dtype=numpy.int64)

	def _group_index(self, g):
		"""Group g's index, read on first use and kept: in position order the
		kind byte, compressed size, uncompressed size and offset of each
		tile's chunk; in track order the track's blocks."""

		if self._index[g] is None:
			index = self._read_track_index(g) if self.order == 'tracks' else \
				self._read_group_index(g)
			with self._index_lock:
				if self._index[g] is None:
					self._index[g] = index

		return self._index[g]

	def _load_indexes(self, groups, n_threads):
		"""Read the indexes of these groups that have not been read yet, on
		up to n_threads threads, since a file of hundreds of groups costs a
		few milliseconds per group to inflate and lay out."""

		missing = [g for g in groups if self._index[g] is None]
		if len(missing) > 1 and n_threads > 1:
			pool = self._pool_for(n_threads)
			list(pool.map(self._group_index, missing))
		else:
			for g in missing:
				self._group_index(g)

	def _read_group_index(self, g):
		error = "The index of {} cannot be read".format(self.path)
		n = self._n_tiles
		offset = int(self._directory[g])
		if not _HEADER.itemsize <= offset < self._file_size:
			raise ValueError(error + ": a group's record lies outside the file.")
		room = self._file_size - offset

		fd = _open_read(self.path)
		try:
			raw = _pread(fd, 8, offset)
			if len(raw) != 8:
				raise ValueError(error + ": the file ends inside it.")
			n_runs, crc = struct.unpack('<II', raw)
			if 8 + n_runs * _RUN.itemsize + 8 > room:
				raise ValueError(error + ": the file ends inside it.")
			raw = _pread(fd, n_runs * _RUN.itemsize + 8, offset + 8)
			if len(raw) != n_runs * _RUN.itemsize + 8:
				raise ValueError(error + ": the file ends inside it.")
			runs = numpy.frombuffer(raw[:n_runs * _RUN.itemsize], dtype=_RUN)
			n_live, n_bytes = struct.unpack('<II', raw[n_runs * _RUN.itemsize:])
			at = offset + 8 + n_runs * _RUN.itemsize + 8
			if n_live > n or n_bytes > self._file_size - at:
				raise ValueError(error + ": the file ends inside it.")
			stream = _pread(fd, n_bytes, at)
			if len(stream) != n_bytes:
				raise ValueError(error + ": the file ends inside it.")
		finally:
			os.close(fd)

		if zlib.crc32(stream, zlib.crc32(raw)) != crc:
			raise ValueError(error + ": it does not match its checksum; the file "
				"is corrupt.")
		raw = _inflate_exact(stream, 17 * n_live, error)
		tiles = numpy.frombuffer(raw[:4 * n_live], dtype='<u4').astype(numpy.int64)
		if n_live > 0 and (tiles[-1] >= n or (tiles[1:] <= tiles[:-1]).any()):
			raise ValueError(error + ": its tiles are out of order.")
		kinds = numpy.zeros(n, dtype=numpy.uint8)
		sizes = numpy.zeros(n, dtype=numpy.uint32)
		raws = numpy.zeros(n, dtype=numpy.uint32)
		crcs = numpy.zeros(n, dtype=numpy.uint32)
		kinds[tiles] = numpy.frombuffer(raw[4 * n_live:5 * n_live],
			dtype=numpy.uint8)
		sizes[tiles] = numpy.frombuffer(raw[5 * n_live:9 * n_live], dtype='<u4')
		raws[tiles] = numpy.frombuffer(raw[9 * n_live:13 * n_live], dtype='<u4')
		crcs[tiles] = numpy.frombuffer(raw[13 * n_live:17 * n_live], dtype='<u4')
		self._summary_at[g] = self._summary_header(at + n_bytes, error)

		if (((kinds >> _CODE_SHIFT) & _CODE_MASK) > _F64).any() or ((kinds &
				_KIND_MASK) > _RUNS).any():
			raise ValueError(error + ": a chunk has a kind this figwig does not "
				"know.")
		if ((kinds == 0) != (sizes == 0)).any() or ((kinds == 0) != (raws == 0)
				).any() or ((kinds & _STORED > 0) & (sizes != raws)).any() or (
				self._codec == _CODEC_NONE and (sizes != raws).any()):
			raise ValueError(error + ": a chunk's kind and size disagree.")
		# No chunk of the group is larger than its dense float64 form plus
		# its headers, so a declared size past that asks for memory the
		# chunk cannot fill.
		n_tracks = len(self._groups[g])
		largest = n_tracks * self.tile * 8 + 16 * n_tracks + 64
		if (raws > largest).any() or (sizes > largest + 64).any():
			raise ValueError(error + ": a chunk's size is larger than any chunk "
				"of {} tracks can be.".format(n_tracks))

		offsets = numpy.zeros(n, dtype=numpy.int64)
		covered = numpy.zeros(n, dtype=numpy.bool_)
		limit = numpy.uint64(self._file_size)
		for run in runs:
			first, last = int(run['first']), int(run['last'])
			if first > last or last >= n or covered[first:last + 1].any() or \
					run['offset'] > limit or run['bytes'] > limit - run['offset']:
				raise ValueError(error + ": a run of chunks is out of order or "
					"lies outside the file.")

			chunk_sizes = sizes[first:last + 1].astype(numpy.int64)
			if int(chunk_sizes.sum()) != int(run['bytes']):
				raise ValueError(error + ": a run's chunks do not add up to its "
					"bytes.")

			covered[first:last + 1] = True
			offsets[first:last + 1] = int(run['offset']) + numpy.cumsum(
				chunk_sizes) - chunk_sizes

		if (sizes[~covered] != 0).any():
			raise ValueError(error + ": a chunk lies in no run.")
		if len(runs) > 1:
			order = numpy.argsort(runs['offset'], kind='stable')
			ends = runs['offset'][order] + runs['bytes'][order]
			if (runs['offset'][order][1:] < ends[:-1]).any():
				raise ValueError(error + ": two runs of chunks overlap in the "
					"file.")

		self._runs[g] = runs
		return kinds, sizes, raws, offsets, crcs

	def _summary_header(self, at, error):
		"""Where a group's summary stream lies and what it holds: (offset,
		n_sum, n_bytes), read from the record at `at`."""

		if at + 8 > self._file_size:
			raise ValueError(error + ": the file ends inside it.")
		fd = _open_read(self.path)
		try:
			raw = _pread(fd, 8, at)
		finally:
			os.close(fd)
		if len(raw) != 8:
			raise ValueError(error + ": the file ends inside it.")
		n_sum, n_bytes = struct.unpack('<II', raw)
		if n_sum > self._n_tiles or n_bytes > self._file_size - at - 8:
			raise ValueError(error + ": the file ends inside it.")
		return at + 8, n_sum, n_bytes

	def _tile_sums_of(self, g):
		"""Group g's summary, as (file tiles, sums of shape (tiles, tracks)
		float32), or None when the file keeps none for it."""

		self._group_index(g)
		at, n_sum, n_bytes = self._summary_at[g]
		if n_bytes == 0:
			return None
		n_tracks = len(self._groups[g])
		fd = _open_read(self.path)
		try:
			stream = _pread(fd, n_bytes, at)
		finally:
			os.close(fd)
		error = "The index of {} cannot be read".format(self.path)
		if len(stream) != n_bytes:
			raise ValueError(error + ": the file ends inside it.")
		raw = _inflate_exact(stream, n_sum * (4 + 4 * n_tracks), error)
		tiles = numpy.frombuffer(raw[:4 * n_sum], dtype='<u4').astype(numpy.int64)
		if n_sum > 0 and (tiles.max() >= self._n_tiles or (tiles[1:] <= tiles[
				:-1]).any()):
			raise ValueError(error + ": its summary is out of order.")
		sums = numpy.frombuffer(raw[4 * n_sum:], dtype='<f4').reshape(n_sum,
			n_tracks)
		return tiles, sums.astype(numpy.float32)

	@property
	def zooms(self) -> tuple:
		"""The zoom levels the file keeps, as the tiles per bin of each:
		(4, 16, 64) for three levels, () for none."""

		if self._zoom_factors is None:
			levels = self._zoom_levels_of(0) if len(self._groups) else []
			self._zoom_factors = tuple(factor for factor, _, _ in levels)
		return self._zoom_factors

	def _zoom_levels_of(self, g):
		"""Group g's zoom levels, as a list of (factor, bins, sums of shape
		(bins, tracks) float64), empty when the file keeps none."""

		if self._zoom_levels[g] is not None:
			return self._zoom_levels[g]
		self._group_index(g)
		at, n_sum, n_bytes = self._summary_at[g]
		levels = []
		if self._optional & _OPTIONAL_ZOOMS:
			error = "The index of {} cannot be read".format(self.path)
			n_tracks = len(self._groups[g])
			fd = _open_read(self.path)
			try:
				at += n_bytes
				raw = _pread(fd, 4, at)
				if len(raw) != 4:
					raise ValueError(error + ": the file ends inside it.")
				n_zooms = struct.unpack('<I', raw)[0]
				if n_zooms > _MAX_ZOOMS:
					raise ValueError(error + ": it names more zoom levels than a "
						"figwig file holds.")
				at += 4
				for k in range(n_zooms):
					raw = _pread(fd, 16, at)
					if len(raw) != 16:
						raise ValueError(error + ": the file ends inside it.")
					factor, n, length, crc = struct.unpack('<IIII', raw)
					offsets, n_bins = _zoom_grid(self._lengths, self.tile, 4 ** (k + 1))
					if factor != 4 ** (k + 1) or n > n_bins or length > \
							self._file_size - at - 16:
						raise ValueError(error + ": a zoom level is not one this figwig "
							"knows, or the file ends inside it.")
					stream = _pread(fd, length, at + 16)
					if len(stream) != length or zlib.crc32(stream) != crc:
						raise ValueError(error + ": a zoom level does not match its "
							"checksum; the file is corrupt.")
					decoded = _inflate_exact(stream, n * (4 + 8 * n_tracks), error)
					bins = numpy.frombuffer(decoded[:4 * n], dtype='<u4').astype(
						numpy.int64)
					if n > 0 and (bins[-1] >= n_bins or (bins[1:] <= bins[:-1]).any()):
						raise ValueError(error + ": a zoom level is out of order.")
					sums = numpy.frombuffer(decoded[4 * n:], dtype='<f8').reshape(n,
						n_tracks).astype(numpy.float64)
					levels.append((factor, bins, sums))
					at += 16 + length
			finally:
				os.close(fd)
		self._zoom_levels[g] = levels
		return levels

	def tile_sums(self, tracks: str | int | list | None = None, chroms: str |
		list[str] | None = None, level: int = 0) -> numpy.ndarray:
		"""Each track's sum over every tile of the genome, or over every bin
		of a zoom level, from the file's summaries, without decoding a chunk.

		This is the file's coarse profile, at the resolution of its tiles
		(16,384 bases unless the file was written with another tile), for a
		genome-wide plot, a per-chromosome total or a coarser binning by a
		reshape; it stands in for a bigWig's zoom levels. The file keeps the
		sums when every track's take at most 256 MB together.

		Parameters
		----------
		tracks: str, int, list of str or int, or None, optional
			The tracks, by name or index. None is every track. Default is
			None.

		chroms: str, list of str, or None, optional
			The chromosomes, by name, whose tiles to give, one after another
			in the order given. None is every chromosome, in the file's
			order. Default is None.

		level: int, optional
			0 for the tile sums; k for the file's zoom level k, whose bins
			are 4**k tiles, kept in float64 when the file was written with
			`zooms` of k or more (`FigWigReader.zooms` lists the levels).
			Default is 0.

		Returns
		-------
		sums: numpy.ndarray, dtype=float32 (level 0) or float64
			The sums, of shape (tracks, bins), or (bins,) when `tracks` is
			one name or index, where bin k of a chromosome of length L holds
			bases [k * b, (k + 1) * b) for b the tile or the level's bin, and
			the chromosome has ceil(L / b) bins.
		"""

		if isinstance(level, bool) or not isinstance(level, (int, numpy.integer)):
			raise TypeError("level must be an integer, not {}.".format(type(
				level).__name__))
		if level < 0:
			raise ValueError("level must be 0 or a zoom level, not {}.".format(
				level))
		ids, single = self._select(tracks)
		if chroms is None:
			names = list(self.chrom_sizes)
		else:
			names = [chroms] if isinstance(chroms, str) else list(chroms)
			if len(names) == 0:
				raise ValueError("chroms must name at least one chromosome, or be "
					"None for all of them.")
			for name in names:
				if name not in self._tids:
					raise ValueError(_unknown_chromosome(name, self.chrom_sizes))
		self._check_file()

		if level == 0:
			offsets, n_bins = self._tile_offsets, self._n_tiles
			out = numpy.zeros((len(ids), n_bins), dtype=numpy.float32)
		else:
			offsets, n_bins = _zoom_grid(self._lengths, self.tile, 4 ** level)
			out = numpy.zeros((len(ids), n_bins), dtype=numpy.float64)
		for g in numpy.unique(self._group_of[ids]).tolist():
			if level == 0:
				summary = self._tile_sums_of(g)
			else:
				levels = self._zoom_levels_of(g)
				summary = None if len(levels) < level else levels[level - 1][1:]
			if summary is None:
				raise ValueError("{} keeps no {} of the group of {!r}; it was "
					"written with more tracks than the summaries hold{}.".format(
					self.path, "summary" if level == 0 else "zoom level {}".format(
					level), self.tracks[int(self._groups[g][0])]['name'], "" if level
					== 0 else ", or with fewer zoom levels"))
			bins, sums = summary
			channels = numpy.flatnonzero(self._group_of[ids] == g)
			out[numpy.ix_(channels, bins)] = sums[:, self._slot_of[ids[
				channels]]].T

		pieces = []
		span = self.tile * 4 ** level
		for name in names:
			tid = self._tids[name]
			first = int(offsets[tid])
			pieces.append(out[:, first:first + -(-int(self._lengths[tid]) // span)])
		out = numpy.concatenate(pieces, axis=1)
		return out[0] if single else out

	def _read_track_index(self, t):
		error = "The index of {} cannot be read".format(self.path)
		offset = int(self._directory[t])
		if not _HEADER.itemsize <= offset < self._file_size:
			raise ValueError(error + ": a track's record lies outside the file.")

		fd = _open_read(self.path)
		try:
			raw = _pread(fd, 8, offset)
			if len(raw) != 8:
				raise ValueError(error + ": the file ends inside it.")
			n_blocks, crc = struct.unpack('<II', raw)
			if n_blocks * _BLOCK.itemsize > self._file_size - offset - 8:
				raise ValueError(error + ": the file ends inside it.")
			raw = _pread(fd, n_blocks * _BLOCK.itemsize, offset + 8)
			if len(raw) != n_blocks * _BLOCK.itemsize:
				raise ValueError(error + ": the file ends inside it.")
		finally:
			os.close(fd)

		if zlib.crc32(raw) != crc:
			raise ValueError(error + ": it does not match its checksum; the file "
				"is corrupt.")
		blocks = numpy.frombuffer(raw, dtype=_BLOCK).copy()
		self._summary_at[t] = self._summary_header(offset + 8 + n_blocks *
			_BLOCK.itemsize, error)
		if (blocks['chrom'] >= len(self._lengths)).any() or (blocks['n'] == 0
				).any() or (blocks['first'] > blocks['last']).any() or (blocks[
				'last'] >= self._lengths[numpy.minimum(blocks['chrom'], len(
				self._lengths) - 1)]).any() or (((blocks['kind'] >> _CODE_SHIFT) &
				_CODE_MASK) > _F64).any() or ((blocks['kind'] & _KIND_MASK) != 1
				).any() or (blocks['kind'] > 255).any():
			raise ValueError(error + ": a block is out of range.")
		limit = numpy.uint64(self._file_size)
		if (blocks['offset'] > limit).any() or (blocks['size'].astype(
				numpy.uint64) > limit - blocks['offset']).any():
			raise ValueError(error + ": a block lies outside the file.")
		if (blocks['raw'].astype(numpy.int64) > 12 * blocks['n'].astype(
				numpy.int64) + 64).any():
			raise ValueError(error + ": a block's size is larger than its entries "
				"can be.")
		keys = (blocks['chrom'].astype(numpy.int64) << 32) | blocks['first']
		if (keys[1:] <= keys[:-1]).any() or ((blocks['chrom'][1:] == blocks[
				'chrom'][:-1]) & (blocks['first'][1:] <= blocks['last'][:-1])).any():
			raise ValueError(error + ": a track's blocks are out of order.")
		return blocks

	def read(self, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
		numpy.ndarray, width: int, tracks: str | int | list | None = None,
		out: numpy.ndarray | None = None, dtype=None, n_jobs: int = 8,
		sparse: bool = False, labels: list | numpy.ndarray | None = None,
		bins: int | None = None, nan: str = 'propagate', device: str | int |
		None = None, channels: str | None = None, items: str | None = None) -> \
		numpy.ndarray | SparseWindows:
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
		pseudobulks in one read; a label of -1 leaves a track out. With
		`bins`, it returns each window summed per bin of `bins` bases, as
		(n, tracks, width // bins), which is the target of a model of binned
		coverage and, with `bins=width`, each window's total per track.

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
			share one, and otherwise the narrowest type that holds every one
			of them exactly, which is float64 for a float32 track read with
			an int32 or uint32 one; sums per label or bin are float32, or
			float64 where that would round a track. An integer type that
			cannot hold a track's values, or a sum, raises. Default is None.

		n_jobs: int, optional
			The most threads to use, or -1 for one per CPU; a read uses one
			thread per 64 chunks it touches, up to this many. Default is 8.

		sparse: bool, optional
			Whether to return the nonzero bases as entries instead of a dense
			array. Default is False.

		labels: list or numpy.ndarray of int, or None, optional
			One label per track read, from 0, or -1 to leave the track out,
			to sum the tracks of each label into one channel. Default is None.

		bins: int or None, optional
			The bases per bin, a divisor of `width`, to sum each window over;
			None reads every base. Default is None.

		nan: str, optional
			With `bins`, what a NaN base does to its bin's sum: 'propagate'
			makes the sum NaN, 'skip' leaves the base out, and 'count' makes
			each bin the number of its bases that are not NaN. Default is
			'propagate'.

		device: str, int or None, optional
			'cuda', 'cuda:N' or a device index to land the output in that
			GPU's memory as a `cupy.ndarray`, which PyTorch, JAX, TensorFlow
			and CuPy take without a copy through DLPack
			(`torch.from_dlpack(y)`); needs the `cuda` extra. Sparse tracks
			are scattered on the device from their entries, dense ones are
			decoded into a pinned buffer and copied, and the device's stream
			is synchronized before the array is returned. `out` must then
			be a cupy array. Default is None, a numpy array in host memory.

		channels: str or None, optional
			With `items`, the name of a track attribute whose values are the
			channels of a dimension, such as "channel" for the m6A and 5mC
			calls of Fiber-seq reads: the tracks read are arranged by the
			item they belong to and their channel, and the output is
			(n, items, channels, width), items and channels in the order of
			their first appearance among the tracks read. Every item must
			have every channel once. Not with `sparse`, `labels` or `out`.
			Default is None.

		items: str or None, optional
			With `channels`, the name of the track attribute that names the
			item each track belongs to, such as "read". Default is None.

		Returns
		-------
		out: numpy.ndarray, cupy.ndarray or SparseWindows
			The value of every base of every window in every track, dense;
			or the nonzero bases as entries with `sparse`; or the sums per
			label with `labels`, or per bin with `bins`; or, with `channels`,
			(n, items, channels, width).
		"""

		if channels is not None or items is not None:
			ids, shape = self._arrange(tracks, channels, items)
			if sparse or labels is not None or out is not None:
				raise ValueError("channels cannot be given with sparse, labels or "
					"out.")
			y = self.read(chroms, starts, width, tracks=ids, dtype=dtype,
				n_jobs=n_jobs, bins=bins, nan=nan, device=device)
			return y.reshape(y.shape[:1] + shape + y.shape[2:])
		if device is not None:
			from ._cuda import _read_on_device
			return _read_on_device(self, device, chroms, starts, width, tracks, out,
				dtype, n_jobs, sparse, labels, bins, None, nan)
		return self._read(chroms, starts, width, tracks, out, dtype, n_jobs,
			sparse, labels, bins, None, nan)

	def _arrange(self, tracks, channels, items):
		"""The tracks `tracks` names arranged by item and channel: their ids
		in item-major order, and the (items, channels) shape."""

		if channels is None or items is None:
			raise ValueError("channels and items are given together: the "
				"attribute that names each track's channel and the one that names "
				"the item it belongs to.")
		for key in (channels, items):
			if not isinstance(key, str):
				raise TypeError("channels and items are attribute names, not {}."
					.format(type(key).__name__))
		ids, _ = self._select(tracks)
		item_of, channel_of = [], []
		for i in ids.tolist():
			record = self.tracks[i]
			for key, into in ((items, item_of), (channels, channel_of)):
				if key not in record:
					raise ValueError("track {!r} has no attribute {!r}.".format(record[
						'name'], key))
				into.append(record[key])
		item_names = list(dict.fromkeys(item_of))
		channel_names = list(dict.fromkeys(channel_of))
		grid = {}
		for i, item, channel in zip(ids.tolist(), item_of, channel_of):
			if (item, channel) in grid:
				raise ValueError("tracks {!r} and {!r} are both {}={!r} of {}={!r}."
					.format(self.tracks[grid[(item, channel)]]['name'], self.tracks[i][
					'name'], channels, channel, items, item))
			grid[(item, channel)] = i
		arranged = []
		for item in item_names:
			for channel in channel_names:
				if (item, channel) not in grid:
					raise ValueError("{}={!r} has no track with {}={!r}; every item "
						"needs every channel.".format(items, item, channels, channel))
				arranged.append(grid[(item, channel)])
		return numpy.array(arranged, dtype=numpy.int64), (len(item_names), len(
			channel_names))

	def sums(self, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
		numpy.ndarray, ends: list[int] | numpy.ndarray, tracks: str | int | list
		| None = None, dtype=None, n_jobs: int = 8, nan: str = 'propagate',
		paired: bool = False, device: str | int | None = None, channels: str |
		None = None, items: str | None = None) -> numpy.ndarray:
		"""The sum of each track over intervals of any lengths.

		Each interval is `chroms[i]`, from `starts[i]` to `ends[i]`, and
		its sum is taken as `read` takes it with `bins`, from the tiles it
		overlaps and without laying its bases out; the intervals may differ
		in length, which `read`, with one `width` for every window, cannot
		take. Bases past the end of a chromosome count as 0.

		Parameters
		----------
		chroms: str or list of str
			The chromosome of each interval, or one for all of them.

		starts: list of int or numpy.ndarray
			Each interval's first base, 0-based.

		ends: list of int or numpy.ndarray
			Each interval's end, exclusive, at or after its start.

		tracks: str, int, list, or None, optional
			The tracks, as `read` takes them. Default is None, every track.

		dtype: optional
			The type of the sums, as `read` takes it with `bins`: the tracks'
			own type, or one that holds their sums. Default is None.

		n_jobs: int, optional
			The threads, as `read` takes it. Default is 8.

		nan: str, optional
			What a NaN base does to its interval's sum: 'propagate' makes
			the sum NaN, 'skip' leaves the base out, and 'count' makes the
			result the number of bases that are not NaN, so that a mean over
			the bases a track covers is the sums with 'skip' over those with
			'count', for a file written with its uncovered bases NaN. Default
			is 'propagate'.

		paired: bool, optional
			Whether `tracks` names one track per interval, interval i being
			summed over track tracks[i] alone, as a read's span over the read
			that is its track, in reads of a few tracks of one group at a
			time; the result is then (n,). Default is False.

		device: str, int or None, optional
			'cuda', 'cuda:N' or a device index to land the sums in that
			GPU's memory as a `cupy.ndarray`, as `read` does. Default is
			None.

		channels, items: str or None, optional
			The attributes that arrange the tracks by item and channel, as
			in `read`; the sums are then (n, items, channels). Not with
			`paired`. Default is None.

		Returns
		-------
		sums: numpy.ndarray, shape=(n, tracks) or (n,)
			Each interval's sum in each track, or in the one track named, or
			in its own track with `paired`.
		"""

		starts = _positions(starts, 'starts')
		ends = _positions(ends, 'ends')
		if ends.shape != starts.shape:
			raise ValueError("ends must hold one end per start, {}, not {}."
				.format(starts.shape, ends.shape))
		if (ends < starts).any():
			raise ValueError("an interval's end is before its start.")
		if channels is not None or items is not None:
			ids, shape = self._arrange(tracks, channels, items)
			if paired:
				raise ValueError("channels cannot be given with paired=True.")
			y = self.sums(chroms, starts, ends, tracks=ids, dtype=dtype,
				n_jobs=n_jobs, nan=nan, device=device)
			return y.reshape(y.shape[:1] + shape)
		if paired:
			sums = self._paired_sums(chroms, starts, ends, tracks, dtype, n_jobs,
				nan)
			if device is not None:
				from ._cuda import _cupy
				return _cupy().asarray(sums)
			return sums
		width = int((ends - starts).max()) if len(starts) else 1
		if device is not None:
			from ._cuda import _read_on_device
			return _read_on_device(self, device, chroms, starts, max(width, 1),
				tracks, None, dtype, n_jobs, False, None, max(width, 1), ends,
				nan)[..., 0]
		out = self._read(chroms, starts, max(width, 1), tracks, None, dtype,
			n_jobs, False, None, max(width, 1), ends, nan)
		return out[..., 0]

	def _paired_sums(self, chroms, starts, ends, tracks, dtype, n_jobs, nan):
		"""`sums` with a track per interval: the intervals are sorted by
		track, cut into batches of at most `_PAIRED_TRACKS` distinct tracks
		of one group, each batch is summed over its own tracks in one read,
		and each interval keeps its own track's sum. A read over k tracks
		costs about k times per interval what a read over one does, so the
		batches are small; they still save most of the per-read overhead of
		a read per track."""

		if tracks is None:
			raise ValueError("paired=True needs one track per interval in tracks.")
		ids, _ = _track_indices(tracks, self._track_ids, len(self.tracks),
			"the file's")
		if ids.shape != starts.shape:
			raise ValueError("paired=True needs one track per interval, {}, not "
				"{}.".format(starts.shape, ids.shape))
		names, codes = _chrom_codes(chroms, len(starts))
		chroms = numpy.array(names)[codes]
		out = None
		order = numpy.lexsort((ids, self._group_of[ids]))
		sorted_ids = ids[order]
		# A batch starts at every _PAIRED_TRACKS-th distinct track, and at
		# a change of group.
		first = numpy.flatnonzero(numpy.diff(sorted_ids)) + 1
		cuts = set(first[_PAIRED_TRACKS - 1::_PAIRED_TRACKS].tolist())
		cuts.update((numpy.flatnonzero(numpy.diff(self._group_of[sorted_ids]))
			+ 1).tolist())
		for batch in numpy.split(order, sorted(cuts)):
			if len(batch) == 0:
				continue
			mine = numpy.unique(ids[batch])
			found = self.sums(chroms[batch], starts[batch], ends[batch],
				tracks=mine.tolist(), dtype=dtype, n_jobs=n_jobs, nan=nan)
			if out is None:
				out = numpy.zeros(len(starts), dtype=found.dtype)
			out[batch] = found[numpy.arange(len(batch)), numpy.searchsorted(mine,
				ids[batch])]
		return out if out is not None else numpy.zeros(0, dtype=self._dtype(ids,
			dtype, True))

	def _read(self, chroms, starts, width, tracks, out, dtype, n_jobs, sparse,
		labels, bins, stops, nan='propagate'):
		"""`read`, with `stops` the windows' ends when they differ in
		length (`sums`), and then `width` the longest and `bins` equal to
		it, so that each window's sum is its one bin; `nan` is what a NaN
		base does to a binned sum."""

		n_jobs = _check_n_jobs(n_jobs)
		if nan not in _NAN_MODES:
			raise ValueError("nan must be 'propagate', 'skip' or 'count', not {!r}."
				.format(nan))
		nan_mode = _NAN_MODES[nan]
		if nan_mode and bins is None:
			raise ValueError("nan= applies to sums: give bins too.")
		if nan_mode and labels is not None:
			raise ValueError("nan= cannot be given with labels, whose sums run "
				"over tracks.")
		ids, single = self._select(tracks)
		windows = _Windows(chroms, starts, width)
		n, width = windows.n, windows.width

		unknown = [name for name in windows.names if name not in self._tids]
		if len(unknown) > 0:
			raise ValueError(_unknown_chromosome(unknown[0], self.chrom_sizes))
		windows.check_range()
		if sparse and (labels is not None or bins is not None):
			raise ValueError("sparse cannot be given with labels or bins.")
		if sparse and out is not None:
			raise ValueError("out cannot be given with sparse=True.")

		if bins is not None:
			if isinstance(bins, bool) or not isinstance(bins, (int,
					numpy.integer)):
				raise TypeError("bins must be an integer or None.")
			if bins < 1 or width % bins != 0:
				raise ValueError("bins must be a divisor of the width, {}, not {}."
					.format(width, bins))
			bins = int(bins)
		summing = labels is not None or bins is not None
		out_width = width // bins if bins is not None else width

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
			n_channels = int(labels.max()) + 1 if labels.size else 0
			single = False
		else:
			n_channels = len(ids)

		dtype = self._dtype(ids, dtype, summing)
		shape = (n, out_width) if single else (n, n_channels, out_width)
		self._check_file()

		# Threads, from the chunks or blocks the read touches, before the
		# output is allocated, since a summed read in track order holds a
		# copy of the output per thread.
		tids = numpy.array([self._tids[name] for name in windows.names],
			dtype=numpy.int64)[windows.codes]
		starts = windows.starts
		lengths = self._lengths[tids]
		first_channel, copies = {}, []
		for channel, i in enumerate(ids.tolist()):
			if i in first_channel:
				copies.append((channel, first_channel[i]))
			else:
				first_channel[i] = channel
		if labels is not None:
			first_channel = {i: int(labels[c]) for i, c in first_channel.items()
				if labels[c] >= 0}

		# The tasks come first, since the threads and the copies follow
		# them; they get their output once it is allocated.
		mode = 'list' if sparse else 'sum' if summing else 'dense'
		accumulate = labels is not None
		cut_inside = False
		if self.order == 'positions':
			read_ids = numpy.array(list(first_channel), dtype=numpy.int64)
			groups_read = numpy.unique(self._group_of[read_ids]).tolist()
			tiles = _window_tiles(self.tile, self._tile_offsets, tids, starts,
				lengths, width, stops)
			n_threads = self._threads(n_jobs, len(tiles[2]) * len(groups_read))
			self._load_indexes(groups_read, n_threads)
			tasks = []
			channels = numpy.array(list(first_channel.values()), dtype=numpy.int64)
			most = _batch_chunks(len(tiles[2]) * len(groups_read), n_threads)
			# The tracks read, grouped once: a scan per group would cost
			# tracks times groups on a read of thousands of each.
			by_group = numpy.argsort(self._group_of[read_ids], kind='stable')
			edges = numpy.searchsorted(self._group_of[read_ids][by_group],
				numpy.array(groups_read + [len(self._groups)]))
			for k, g in enumerate(groups_read):
				mine = by_group[edges[k]:edges[k + 1]]
				slots = numpy.full(len(self._groups[g]), -1, dtype=numpy.int64)
				slots[self._slot_of[read_ids[mine]]] = channels[mine]
				planned, cut = self._plan(g, slots, tiles, starts, width, None, mode,
					accumulate, bins or 1, most, stops, nan_mode)
				tasks.extend(planned)
				cut_inside |= cut
			if len(tasks) < n_threads:
				tasks, cut = _split_tasks(tasks, n_threads, bins is not None and
					bins > 1)
				cut_inside |= cut
		else:
			self._load_indexes(list(first_channel), n_jobs)
			tasks = self._plan_tracks(first_channel, tids, starts, lengths, width,
				None, mode, accumulate, bins or 1, n_jobs, stops, nan_mode)
			n_threads = max(1, min(n_jobs, len(tasks)))
		# In position order a group's tasks sum straight into the output when
		# they write disjoint cells: a label's cells are one per window base,
		# a window's two tiles fill different bases, and a binned read's
		# batches are not cut inside a window unless a window is longer than
		# a batch may grow; then, and in track order, where a track's blocks
		# are spread over the tasks, each thread sums into its own copy.
		buffers = n_threads if summing and n_threads > 1 and (self.order ==
			'tracks' or cut_inside) else 0

		inner = None
		if not sparse:
			n_bytes = int(numpy.prod(shape)) * dtype.itemsize
			accumulator = int(numpy.prod(shape)) * 8 if summing and dtype.itemsize \
				< 8 else 0
			if out is None:
				if n_bytes * (1 + buffers) + accumulator > _MAX_DENSE_BYTES:
					raise ValueError("the dense output would take {:.1f} GiB{}; read "
						"fewer windows or tracks at a time, pass out=, or, for tracks "
						"that are mostly zero, ask for sparse=True or labels=.".format(
						n_bytes / 2 ** 30, " with its {} copies for the threads"
						.format(buffers) if buffers else ""))
				# Blocks and sums only add nonzero bases, so their output starts
				# from zero; chunks set every base of a window.
				zeroed = summing or self.order == 'tracks'
				out = numpy.zeros(shape, dtype=dtype) if zeroed else numpy.empty(
					shape, dtype=dtype)
			elif not isinstance(out, numpy.ndarray):
				raise TypeError("out must be a numpy array, not {}.".format(
					type(out).__name__))
			elif out.shape != shape or out.dtype != dtype or not out.flags[
					'C_CONTIGUOUS'] or not out.flags['WRITEABLE']:
				raise ValueError("out must be a writeable, C-contiguous {} array "
					"of shape {}; pass dtype= to read as another type.".format(
					dtype, shape))
			elif summing or self.order == 'tracks':
				out[...] = 0

			# Sums into an integer output are taken in int64 and checked
			# against the output's type, since a sum can overflow a type
			# that each track's values fit; sums into a float32 output are
			# taken in float64, since a bin is added to base by base.
			inner = out
			if summing and dtype.kind in 'iu':
				inner = numpy.zeros(shape, dtype=numpy.int64)
			elif summing and dtype != numpy.float64:
				inner = numpy.zeros(shape, dtype=numpy.float64)

		if n == 0:
			if sparse:
				empty = numpy.zeros(0, dtype=numpy.int64)
				return SparseWindows(empty, empty, empty, numpy.zeros(0,
					dtype=dtype), (0, len(ids), width))
			return out

		out3 = None if sparse else inner.reshape(n, 1, out_width) if single \
			else inner
		for task in tasks:
			task.out = out3

		self._run(tasks, n_threads, out3 if mode == 'sum' else None, buffers > 0)

		if sparse:
			return self._gather(tasks, copies, n, len(ids), width, dtype)

		if nan_mode == 2:
			# The kernels count a bin's NaN bases, since the bases that are
			# not stored are the zeros; the bases that are not NaN are the
			# rest of the bin, clipped to the chromosome.
			span = numpy.clip(numpy.minimum(starts + width if stops is None else
				stops, lengths) - starts, 0, None)
			per_bin = numpy.clip(span[:, None] - numpy.arange(out_width)[None, :]
				* bins, 0, bins)
			counted = inner.reshape(n, -1, out_width)
			counted[...] = per_bin[:, None, :] - counted

		if inner is not out:
			if dtype.kind in 'iu':
				info = numpy.iinfo(dtype)
				if (inner < info.min).any() or (inner > info.max).any():
					raise ValueError("a sum lies outside what {} holds; read the "
						"sums as a wider type.".format(dtype))
			out[...] = inner.reshape(out.shape)

		for channel, source in copies:
			out3[:, channel] = out3[:, source]

		# Bases past the end of a chromosome lie in no tile or block.
		over = numpy.flatnonzero(starts + width > lengths)
		if len(over) > 0 and bins is None:
			fill = numpy.nan if dtype.kind == 'f' else 0
			for j in over.tolist():
				out[j, ..., max(0, int(lengths[j] - starts[j])):] = fill

		return out

	def _gather(self, tasks, copies, n, n_channels, width, dtype):
		"""The entries of a sparse read's tasks, as one `SparseWindows`
		sorted by row, channel and offset, with each array built and freed
		as late as it can be."""

		parts = [task.entries for task in tasks if task.entries is not None]
		if parts:
			rows = numpy.concatenate([p[0] for p in parts])
			channels = numpy.concatenate([p[1] for p in parts])
			offsets = numpy.concatenate([p[2] for p in parts])
			values = numpy.concatenate([p[3] for p in parts])
		else:
			rows = channels = offsets = numpy.zeros(0, dtype=numpy.int64)
			values = numpy.zeros(0, dtype=dtype)
		for task in tasks:
			task.entries = None
		del parts

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

		# The tasks list each window's offsets in order, so a stable
		# counting sort on (row, channel) is enough, in linear time; a key
		# per entry sorts the rest.
		if n * n_channels <= 4 * len(rows) + 1024:
			rows_, channels_, offsets_, values_, ok = _bucket_sort(rows, channels,
				offsets, values, n_channels, n * n_channels)
			if ok:
				return SparseWindows(rows_, channels_, offsets_, values_.astype(
					dtype, copy=False), (n, n_channels, width))
		key = (rows * n_channels + channels) * width + offsets
		order = numpy.argsort(key)
		del key
		rows = rows[order]
		channels = channels[order]
		offsets = offsets[order]
		values = values[order].astype(dtype, copy=False)
		return SparseWindows(rows, channels, offsets, values, (n, n_channels,
			width))

	def read_entries(self, tracks: str | int | list | None = None, chroms: str |
		list[str] | None = None, n_jobs: int = 8) -> SparseEntries:
		"""Every nonzero base of tracks over whole chromosomes, as entries.

		This is the read a file in track order is laid out for: one track's
		cuts over the genome, or a few tracks', as `SparseEntries` sorted by
		chromosome, track and position. In position order it decodes every
		chunk of the tracks' groups on those chromosomes. The values come in
		the narrowest type that holds every track's exactly.

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
					raise ValueError(_unknown_chromosome(name, self.chrom_sizes))

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
			return SparseEntries(empty, empty, empty, numpy.zeros(0,
				dtype=self._dtype(ids, None)))

		return SparseEntries(*[numpy.concatenate([p[k] for p in parts]) for k
			in range(4)])

	def _select(self, tracks):
		"""The tracks a call names, as (track ids in channel order, whether a
		single track was named); None is every track."""

		if tracks is None:
			return numpy.arange(len(self.tracks)), False
		ids, single = _track_indices(tracks, self._track_ids, len(self.tracks),
			"the file's")
		if len(ids) == 0:
			raise ValueError("tracks must name at least one track.")
		return ids, single

	def _dtype(self, ids, dtype, summing=False):
		"""The output dtype for these tracks, checked against their values:
		the narrowest type that holds every track exactly, or, for sums,
		float32 unless that would round a track."""

		# The tracks' dtypes as codes into _DTYPES, so that a read of
		# thousands of tracks does not build a dtype for each.
		codes = numpy.unique(self._dtype_codes[ids])
		own = [numpy.dtype(_DTYPES[c]) for c in codes.tolist()]
		if dtype is None:
			if summing:
				# Integer counts sum past float32's 2**24 in a few windows.
				dtype = numpy.dtype(numpy.float64) if any(d.kind in 'iu' for d in
					own) else numpy.result_type(numpy.float32, *own)
			elif len(own) == 1:
				dtype = own[0]
			else:
				dtype = numpy.result_type(*own)
		else:
			if type(dtype).__module__ == 'torch':
				dtype = str(dtype).split('.')[-1]
			if dtype in ('bfloat16', 'float16'):
				raise ValueError("dtype must be an integer or float type of 32 or "
					"64 bits, not {}; read float32 and cast.".format(dtype))
			try:
				dtype = numpy.dtype(dtype)
			except TypeError:
				raise TypeError("dtype is not a dtype: {!r}.".format(dtype)) \
					from None
			if dtype.kind not in 'iuf' or (dtype.kind == 'f' and dtype.itemsize
					< 4):
				raise ValueError("dtype must be an integer or float type of 32 "
					"or 64 bits, not {}.".format(dtype))

		if dtype.kind in 'iu':
			info = numpy.iinfo(dtype)
			floats = numpy.flatnonzero(self._dtype_codes[ids] >= _DTYPES.index(
				'float32'))
			if len(floats) > 0:
				raise ValueError("track {!r} is {} and cannot be read as {}."
					.format(self.tracks[int(ids[floats[0]])]['name'], self.tracks[
					int(ids[floats[0]])]['dtype'], dtype))
			outside = numpy.flatnonzero((self.stats['min'][ids] < info.min) | (
				self.stats['max'][ids] > info.max))
			if len(outside) > 0 and not summing:
				i = int(ids[outside[0]])
				raise ValueError("track {!r} has values from {} to {}, which {} "
					"does not hold.".format(self.tracks[i]['name'], self.stats[i][
					'min'], self.stats[i]['max'], dtype))

		return dtype

	def _plan(self, g, slots, tiles, starts, width, out, mode, accumulate, bins,
		most, stops=None, nan_mode=0):
		"""The tasks that read group g's part of the windows: each a batch
		of up to `most` chunks and the (window, tile) pairs that fall in
		them, from the windows' tiles as `_window_tiles` gives them, as
		(tasks, whether a batch was cut inside a window of a binned read,
		which happens when a window is longer than a batch may grow)."""

		tile = self.tile
		kinds, sizes, raws, offsets, crcs = self._group_index(g)
		rows, local, unique, first, inverse, total = tiles
		if total == 0:
			return [], False

		chunk_kinds = kinds[unique]
		live = chunk_kinds != 0
		bounds = raws[unique].astype(numpy.int64) + 8

		# Batches of chunks, cut where a batch would hold too many bytes
		# before compression or too many chunks; a binned read's batches
		# are not cut inside a window, whose bins would otherwise be summed
		# by two tasks at once, unless the window runs past twice a batch.
		tasks = []
		cut_inside = False
		continues = _continuing_windows(rows, inverse, len(unique)) if bins > 1 \
			else None
		a = 0
		while a < len(unique):
			total_bytes, n_live, b = 0, 0, a
			while b < len(unique) and (b == a or (total_bytes + bounds[b] <=
					_BATCH_BYTES and n_live < most)):
				total_bytes += int(bounds[b])
				n_live += int(live[b])
				b += 1
			if bins > 1:
				while b < len(unique) and continues[b - 1]:
					if total_bytes + bounds[b] > 2 * _BATCH_BYTES or n_live >= 2 * most:
						cut_inside = True
						break
					total_bytes += int(bounds[b])
					n_live += int(live[b])
					b += 1

			p0, p1 = int(first[a]), int(first[b]) if b < len(unique) else total
			chunk = numpy.where(live[inverse[p0:p1]], inverse[p0:p1] - a, -1)
			tasks.append(_Task(self, g, slots, unique[a:b][live[a:b]],
				chunk_kinds[a:b][live[a:b]], offsets[unique[a:b][live[a:b]]],
				sizes[unique[a:b][live[a:b]]].astype(numpy.int64), int(total_bytes),
				_compact(chunk, live[a:b]), local[p0:p1] * tile, starts[rows[p0:p1]],
				rows[p0:p1], out, width, mode, accumulate, bins, crcs[unique[a:b][
				live[a:b]]], _empty_pairs if stops is None else stops[rows[p0:p1]],
				nan_mode))
			a = b

		return tasks, cut_inside

	def _plan_tracks(self, channels, tids, starts, lengths, width, out, mode,
		accumulate, bins, n_jobs, stops=None, nan_mode=0):
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
				ends = numpy.minimum(starts[windows] + width if stops is None else
					stops[windows], lengths[windows])
				lo = numpy.searchsorted(lasts, starts[windows], side='left')
				hi = numpy.searchsorted(firsts, ends, side='left')
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
				out, width, mode, accumulate, bins, _empty_pairs if stops is None
				else stops[rows[p0:p1]], nan_mode))
			a = b

		return tasks

	def _run(self, tasks, n_threads, summed=None, copies=False):
		"""Run the tasks on up to `n_threads` threads, the first on this
		thread when a kernel has not been compiled yet, so that each is
		compiled once before any thread starts.

		Tasks that sum into `summed`, the output of a read with labels or
		bins, run a group at a time when the tasks of one group write
		disjoint cells (labels, and bins on the tile grid, in position
		order), since those of two groups do not; with `copies`, where a
		track's blocks are spread over the tasks in track order, or a bin
		straddles two tiles, each thread sums into its own copy, and the
		copies are added to `summed` at the end.
		"""

		if len(tasks) == 0:
			return

		# The first task runs alone only while a kernel this read calls is
		# not compiled yet; a kernel another kind of read uses does not
		# matter, or every read would start with a serial task.
		if isinstance(tasks[0], _BlockTask):
			kernels = (_read_blocks, _check_blocks)
		elif tasks[0].mode == 'list':
			kernels = (_list_tiles, _check_chunks)
		else:
			kernels = (_read_tiles, _check_chunks)
		first = []
		if any(not getattr(kernel, 'signatures', True) for kernel in kernels):
			first.append(tasks[0])

		rest = tasks[len(first):]
		fd = _open_read(self.path)
		pool = self._pool_for(n_threads) if n_threads > 1 and len(rest) > 1 \
			else None

		def run_all(tasks_):
			# The tasks in n_threads shares, so that a read on fewer threads
			# than the pool has takes no more than it asked for.
			if pool is None or len(tasks_) <= 1:
				for task in tasks_:
					task.run(fd)
				return

			def run_share(k):
				for task in tasks_[k::n_threads]:
					task.run(fd)

			list(pool.map(run_share, range(min(n_threads, len(tasks_)))))

		try:
			for task in first:
				task.run(fd)

			if summed is None or not copies:
				if summed is None or not rest or isinstance(rest[0], _BlockTask):
					run_all(rest)
				else:
					a = 0
					while a < len(rest):
						b = a + 1
						while b < len(rest) and rest[b].g == rest[a].g:
							b += 1
						run_all(rest[a:b])
						a = b
			elif pool is None:
				for task in rest:
					task.run(fd)
			else:
				# Each thread adds its copy as it finishes, under a lock, so
				# the adds overlap the threads still reading.
				lock = threading.Lock()

				def run_share(k):
					buffer = numpy.zeros_like(summed)
					for task in rest[k::n_threads]:
						task.out = buffer
						task.run(fd)
					with lock:
						numpy.add(summed, buffer, out=summed)

				list(pool.map(run_share, range(n_threads)))
		finally:
			os.close(fd)


_empty_pairs = numpy.zeros(0, dtype=numpy.int64)
_NAN_MODES = {'propagate': 0, 'skip': 1, 'count': 2}


def _continuing_windows(rows, inverse, n_unique):
	"""For each distinct tile i of a read, whether a window has pairs on
	both tile i and tile i + 1, from the pairs' window rows and distinct
	tiles: a window's tiles are consecutive, so it spans every boundary
	from its first to its last."""

	if len(rows) == 0:
		return numpy.zeros(n_unique, dtype=numpy.bool_)
	n = int(rows.max()) + 1
	lo = numpy.full(n, n_unique, dtype=numpy.int64)
	hi = numpy.full(n, -1, dtype=numpy.int64)
	numpy.minimum.at(lo, rows, inverse)
	numpy.maximum.at(hi, rows, inverse)
	spans = lo < hi
	edges = numpy.zeros(n_unique + 1, dtype=numpy.int64)
	numpy.add.at(edges, lo[spans], 1)
	numpy.add.at(edges, hi[spans], -1)
	return numpy.cumsum(edges)[:n_unique] > 0


def _window_tiles(tile, tile_offsets, tids, starts, lengths, width, stops=None):
	"""The tiles the windows overlap, clipped to their chromosomes, as
	(window rows, tiles on the chromosome, distinct file tiles, where each
	distinct tile's pairs start, each pair's distinct tile, pairs), the
	pairs sorted by file tile. The windows are `width` wide, or end at
	`stops`."""

	stops = numpy.minimum(starts + width if stops is None else stops, lengths)
	k0 = starts // tile
	k1 = (stops - 1) // tile
	counts = numpy.maximum(k1 - k0 + 1, 0)
	counts[stops <= starts] = 0
	total = int(counts.sum())

	rows = numpy.repeat(numpy.arange(len(starts)), counts)
	local = numpy.repeat(k0, counts) + numpy.arange(total) - numpy.repeat(
		numpy.cumsum(counts) - counts, counts)
	gtiles = tile_offsets[tids[rows]] + local
	order = numpy.argsort(gtiles, kind='stable')
	rows, local, gtiles = rows[order], local[order], gtiles[order]
	unique, first = numpy.unique(gtiles, return_index=True)
	inverse = numpy.repeat(numpy.arange(len(unique)), numpy.diff(
		numpy.append(first, total)))
	return rows, local, unique, first, inverse, total


def _batch_chunks(n, n_jobs):
	"""The most chunks or blocks in one batch of a read of n of them on
	n_jobs threads: _BATCH_CHUNKS, or fewer so that a small read still has
	a batch for every thread, but at least _MIN_BATCH_CHUNKS. Smaller
	batches than that balanced the threads no better and cost each about
	25 microseconds of Python."""

	return min(_BATCH_CHUNKS, max(_MIN_BATCH_CHUNKS, -(-n // n_jobs)))


def _split_tasks(tasks, n_threads, by_window=False):
	"""Tasks of a read with fewer batches than threads, each with many
	pairs, cut into tasks of fewer pairs on the same chunks, so that the
	windows packed into a few tiles are copied on every thread; a binned
	read's are cut between windows. Returns (tasks, whether a cut fell
	inside a window)."""

	total = sum(len(task.pair_rows) for task in tasks)
	if total < _SPLIT_PAIRS:
		return tasks, False
	parts = -(-n_threads // len(tasks))
	out = []
	cut_inside = False
	for task in tasks:
		pieces, cut = task.split(parts, by_window)
		out.extend(pieces)
		cut_inside |= cut
	return out, cut_inside


def _compact(chunk, live):
	"""Chunk positions among a batch's live chunks, from positions among all
	of its chunks, with -1 kept for the zero ones."""

	mapping = numpy.cumsum(live) - 1
	return numpy.where(chunk >= 0, mapping[numpy.maximum(chunk, 0)], -1)


def _inflate_exact(stream, n, error):
	"""The n bytes a zlib stream holds, or a ValueError."""

	try:
		inflater = zlib.decompressobj()
		raw = inflater.decompress(stream, n + 1)
	except zlib.error:
		raise ValueError(error + ": it is corrupt.") from None
	if len(raw) != n or not inflater.eof:
		raise ValueError(error + ": it is corrupt.")
	return raw


def _load_batch(reader, fd, offsets, sizes, kinds, bound, crcs=None):
	"""A batch's chunks or blocks, read with one pread per run of adjacent
	ones and inflated into one buffer, as (data, positions, lengths); a
	chunk whose kind byte says it is stored as it is is copied. With the
	reader's `verify`, every chunk's bytes are checked against `crcs`."""

	order = numpy.argsort(offsets, kind='stable')
	o, s = offsets[order], sizes[order]
	breaks = numpy.flatnonzero(o[1:] != o[:-1] + s[:-1]) + 1
	breaks = [0] + breaks.tolist() + [len(o)]

	ends = numpy.cumsum(s)
	starts = ends - s
	raw = numpy.empty(int(s.sum()) + 8, dtype=numpy.uint8)
	run_starts = starts[breaks[:-1]]
	run_ends = ends[numpy.asarray(breaks[1:], dtype=numpy.int64) - 1]
	run_offsets = o[breaks[:-1]].astype(numpy.int64)
	got = numpy.zeros(len(run_starts), dtype=numpy.int64)
	pread = _libc_pread()
	if pread is not None:
		# Without the GIL: the threads of a read would otherwise take turns
		# holding it through every pread.
		_pread_runs(pread, fd, raw, run_offsets, run_starts, run_ends, got)
	for k in range(len(run_starts)):
		position, end = int(run_starts[k]), int(run_ends[k])
		if got[k] != end - position:
			if _pread_into(fd, raw[position:end], int(run_offsets[k])) < end - \
					position:
				raise ValueError("{} ends inside a chunk; the file is truncated."
					.format(reader.path))

	# Back to the batch's own order.
	starts_in = numpy.empty_like(starts)
	starts_in[order] = starts
	sizes_in = sizes.astype(numpy.int64)

	stored = (kinds & _STORED) > 0
	if reader.verify and crcs is not None:
		found = _crcs(raw, numpy.concatenate([[0], ends]))
		found_in = numpy.empty_like(found)
		found_in[order] = found
		if (found_in != crcs.astype(numpy.int64)).any():
			raise ValueError("a chunk of {} does not match its checksum; the "
				"file is corrupt.".format(reader.path))
	elif crcs is not None and reader._codec != _CODEC_NONE and stored.any():
		# A chunk stored as it is inside a compressed file has no checksum
		# of its own, so its CRC is checked as a compressed chunk's
		# Adler-32 is on every read.
		view = memoryview(raw)
		for j in numpy.flatnonzero(stored[order]).tolist():
			if zlib.crc32(view[int(starts[j]):int(ends[j])]) != int(crcs[order[j]]):
				raise ValueError("a chunk of {} does not match its checksum; the "
					"file is corrupt.".format(reader.path))

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
				starts_in, sizes_in, stored, buffer, positions, lengths,
				numpy.zeros(1, dtype=numpy.uintp))
		finally:
			free(decompressor)
	elif uncompress is not None:
		function, ulong = uncompress
		buffer = numpy.empty(bound + 8 * len(sizes), dtype=numpy.uint8)
		used = _inflate_chunks(function, raw, starts_in, sizes_in, stored, buffer,
			positions, lengths, numpy.zeros(1, dtype=ulong))
	else:
		positions[:] = -1

	parts = [buffer[:used]]
	position = used
	for k in numpy.flatnonzero(positions < 0).tolist():
		stream = raw[starts_in[k]:starts_in[k] + sizes_in[k]].tobytes()
		if stored[k]:
			block = stream
		else:
			# The stream's Adler-32 is only checked once the inflater reaches
			# its end, so a stream that stops short is as corrupt as one
			# that fails.
			try:
				inflater = zlib.decompressobj()
				block = inflater.decompress(stream, bound + 1)
				if not inflater.eof:
					block = b''
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
		pair_chunks, pair_bases, pair_starts, pair_rows, out, width, mode,
		accumulate, bins, crcs, pair_stops, nan_mode=0):
		self.reader, self.g, self.slots = reader, g, slots
		self.gtiles, self.kinds, self.crcs = gtiles, kinds, crcs
		self.offsets, self.sizes, self.bound = offsets, sizes, bound
		self.pair_chunks = numpy.ascontiguousarray(pair_chunks, dtype=numpy.int64)
		self.pair_bases = numpy.ascontiguousarray(pair_bases, dtype=numpy.int64)
		self.pair_starts = numpy.ascontiguousarray(pair_starts, dtype=numpy.int64)
		self.pair_rows = numpy.ascontiguousarray(pair_rows, dtype=numpy.int64)
		self.pair_stops = numpy.ascontiguousarray(pair_stops, dtype=numpy.int64)
		self.out, self.width, self.mode = out, width, mode
		self.accumulate, self.bins, self.nan_mode = accumulate, bins, nan_mode
		self.entries = None

	def split(self, parts, by_window=False):
		"""This task as `parts` tasks over the same chunks, each with a
		share of the pairs; with `by_window` the shares are cut between
		windows where there are enough windows, since a binned read sums a
		window's pairs into one row. Returns (tasks, whether a cut fell
		inside a window)."""

		n = len(self.pair_rows)
		cuts = numpy.linspace(0, n, parts + 1).astype(numpy.int64)
		cut_inside = False
		if by_window:
			# The pairs lie in tile order, so a window that spans two tiles
			# has pairs apart; a cut is safe only at a tile boundary that no
			# window spans.
			# Consecutive pairs lie on one tile while their chunk and base
			# agree (two chromosomes' first tiles share a base).
			change = (numpy.diff(self.pair_chunks) != 0) | (numpy.diff(
				self.pair_bases) != 0)
			inverse = numpy.concatenate([[0], numpy.cumsum(change)])
			spans = _continuing_windows(self.pair_rows, inverse, int(inverse[-1])
				+ 1 if n else 0)
			firsts = numpy.flatnonzero(change) + 1
			safe = firsts[~spans[inverse[firsts] - 1]]
			if len(safe) + 1 >= parts:
				cuts[1:-1] = safe[numpy.minimum(numpy.searchsorted(safe,
					cuts[1:-1]), len(safe) - 1)]
				cuts = numpy.unique(cuts)
			else:
				cut_inside = parts > 1
		out = []
		for a, b in zip(cuts[:-1], cuts[1:]):
			if b > a:
				out.append(_Task(self.reader, self.g, self.slots, self.gtiles,
					self.kinds, self.offsets, self.sizes, self.bound,
					self.pair_chunks[a:b], self.pair_bases[a:b], self.pair_starts[a:b],
					self.pair_rows[a:b], self.out, self.width, self.mode,
					self.accumulate, self.bins, self.crcs, self.pair_stops[a:b] if
					len(self.pair_stops) else self.pair_stops, self.nan_mode))
		return out, cut_inside

	def run(self, fd):
		reader = self.reader
		n = len(self.gtiles)
		if n > 0:
			data, positions, lengths = _load_batch(reader, fd, self.offsets,
				self.sizes, self.kinds, self.bound, self.crcs)
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

		# A scratch tile for chunks that many windows fall on, when it fits
		# _SCRATCH_BYTES; a read of more channels than that decodes each
		# pair on its own.
		n_channels = self.out.shape[1]
		scratch = numpy.zeros((0, 0), dtype=numpy.float64)
		if n_channels * reader.tile * 8 <= _SCRATCH_BYTES:
			scratch = numpy.empty((n_channels, reader.tile), dtype=numpy.float64)

		_read_tiles(reader.tile, data, positions, self.kinds, len(self.slots),
			self.slots, self.pair_chunks, self.pair_bases, self.pair_starts,
			self.pair_rows, self.out, self.accumulate, scratch, self.bins,
			self.pair_stops, self.nan_mode)


class _BlockTask:
	"""One batch of a track-order read: blocks of some tracks and the
	(window, block, channel) pairs that may fall in them."""

	def __init__(self, reader, records, bound, pair_blocks, pair_slots,
		pair_starts, pair_rows, out, width, mode, accumulate, bins, pair_stops,
		nan_mode=0):
		self.reader, self.records, self.bound = reader, records, bound
		self.pair_blocks = numpy.ascontiguousarray(pair_blocks, dtype=numpy.int64)
		self.pair_slots = numpy.ascontiguousarray(pair_slots, dtype=numpy.int64)
		self.pair_starts = numpy.ascontiguousarray(pair_starts, dtype=numpy.int64)
		self.pair_rows = numpy.ascontiguousarray(pair_rows, dtype=numpy.int64)
		self.pair_stops = numpy.ascontiguousarray(pair_stops, dtype=numpy.int64)
		self.out, self.width, self.mode = out, width, mode
		self.accumulate, self.bins, self.nan_mode = accumulate, bins, nan_mode
		self.entries = None

	def run(self, fd):
		reader = self.reader
		records = self.records
		kinds = records['kind'].astype(numpy.uint8)
		data, positions, lengths = _load_batch(reader, fd, records['offset']
			.astype(numpy.int64), records['size'].astype(numpy.int64), kinds,
			self.bound, records['crc'])
		n_entries = records['n'].astype(numpy.int64)
		failed = numpy.zeros(len(records), dtype=numpy.int64)
		if _check_blocks(data, positions, lengths, kinds, n_entries, records[
				'first'].astype(numpy.int64), records['last'].astype(numpy.int64),
				failed) > 0:
			raise ValueError("a block of {} cannot be read: it is corrupt, or the "
				"file is truncated.".format(reader.path))

		empty = numpy.zeros(0, dtype=numpy.int64)
		out = self.out if self.out is not None else numpy.zeros((1, 1, 1),
			dtype=numpy.float32)
		scratch = numpy.zeros(self.width if self.bins > 1 else 0,
			dtype=numpy.float64)
		if self.mode == 'list':
			count = _read_blocks(data, positions, kinds, n_entries, self.pair_blocks,
				self.pair_slots, self.pair_starts, self.pair_rows, self.width, out,
				False, True, False, empty, empty, empty, numpy.zeros(0,
				dtype=numpy.float64), scratch, 1, empty, 0)
			rows = numpy.empty(count, dtype=numpy.int64)
			channels = numpy.empty(count, dtype=numpy.int64)
			offsets = numpy.empty(count, dtype=numpy.int64)
			values = numpy.empty(count, dtype=numpy.float64)
			_read_blocks(data, positions, kinds, n_entries, self.pair_blocks,
				self.pair_slots, self.pair_starts, self.pair_rows, self.width, out,
				False, True, True, rows, channels, offsets, values, scratch, 1, empty,
				0)
			self.entries = (rows, channels, offsets, values)
			return

		_read_blocks(data, positions, kinds, n_entries, self.pair_blocks,
			self.pair_slots, self.pair_starts, self.pair_rows, self.width, out,
			self.accumulate, False, False, empty, empty, empty, numpy.zeros(0,
			dtype=numpy.float64), scratch, self.bins, self.pair_stops, self.nan_mode)


def read_figwig(figwig: str | os.PathLike | FigWigReader, chroms: str |
	list[str] | numpy.ndarray, starts: list[int] | numpy.ndarray, width: int,
	tracks: str | int | list | None = None, out: numpy.ndarray | None = None,
	dtype=None, n_jobs: int = 8, sparse: bool = False, labels: list |
	numpy.ndarray | None = None, bins: int | None = None, nan: str =
	'propagate', device: str | int | None = None, channels: str | None = None,
	items: str | None = None) -> numpy.ndarray | SparseWindows:
	"""Read the per-base values of many windows of one width from a figwig
	file.

	This reads the windows as `FigWigReader.read` does, from a path or an
	open reader, into an array of shape (n, tracks, width), or (n, width)
	when `tracks` is one name or index, or as entries with `sparse=True`, or
	summed per label with `labels`, or per bin with `bins`. A path is opened
	on every call; to read the same file more than once, as a data loader
	does, open it with `FigWigReader` and pass that, which keeps its index.


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

	bins: int or None, optional
		The bases per bin to sum each window over; see `FigWigReader.read`.
		Default is None.

	nan: str, optional
		What a NaN base does to a binned sum; see `FigWigReader.read`.
		Default is 'propagate'.

	device: str, int or None, optional
		A GPU to land the output on, as a cupy array; see
		`FigWigReader.read`. Default is None.

	channels, items: str or None, optional
		The attributes that arrange the tracks by item and channel into
		(n, items, channels, width); see `FigWigReader.read`. Default is
		None.

	Returns
	-------
	out: numpy.ndarray, cupy.ndarray or SparseWindows
		The value of every base of every window in every track, or the
		entries, or the sums per label or bin.
	"""

	if not isinstance(figwig, (str, os.PathLike, FigWigReader)):
		raise TypeError("figwig must be a path or a FigWigReader, not {}.".format(
			type(figwig).__name__))

	reader = figwig if isinstance(figwig, FigWigReader) else FigWigReader(figwig)
	return reader.read(chroms, starts, width, tracks=tracks, out=out,
		dtype=dtype, n_jobs=n_jobs, sparse=sparse, labels=labels, bins=bins,
		nan=nan, device=device, channels=channels, items=items)


def write_figwig(path: str | os.PathLike, chrom_sizes: dict | list,
	tracks: list, chroms: str | list[str] | numpy.ndarray, starts: list[int] |
	numpy.ndarray, values, ends: list[int] | numpy.ndarray | None = None,
	genome: str | None = None, attributes: dict | None = None,
	tile: int = _DEFAULT_TILE, groups: int | list = _DEFAULT_GROUPS,
	level: int | None = None, engine: str = 'auto', n_jobs: int = 8,
	order: str = 'positions', zooms: int = 0):
	"""Write the values of windows, or of intervals, for every track to one
	figwig file.

	This writes the windows or intervals as `FigWigWriter.write` does, in
	one call, from values of shape (n, len(tracks), width), or (n,
	len(tracks)) with `ends`. That is the (batch, channels, length) layout
	that `read_figwig` gives and that sequence models predict, so that

		write_figwig(path, chrom_sizes, tracks, chroms, starts, y)

	writes back what `read_figwig(path, chroms, starts, width)` read, or a
	model's predictions for the same windows. If writing fails part way, the
	path holds what it held before.


	Parameters
	----------
	path: str or os.PathLike
		The file to write.

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
			engine=engine, n_jobs=n_jobs, order=order, zooms=zooms) as writer:
		writer.write(chroms, starts, values, ends=ends)
