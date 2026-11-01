# summary_files.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The files and helpers of the regional summary index tests.

`write_signal` and `write_counts` write one file each, with or without the
index, from values this module builds densely, so that a test compares a
read with the dense values and not with another read. Within a tile every
base of a track has the same value, so a tile's sum is a product of a power
of two with one value, exact in binary64 whatever order a writer adds it in.

`expected_tile_sums` is the specification's tile sum, rounded to f4, and its
zoom levels, the f4 sums added in f8. `patch`, `repair` and `replace` change
the bytes of a file as the corruption tests need, repairing the checksums
over what they change.
"""

import struct
import zlib

import numpy

from figwig import FigWigReader
from figwig import FigWigWriter

from .summary_reference import read_index


TILE = 16

# One chromosome long enough for its summary to need windows, one shorter
# than a window, one without signal; chrA has a gap of tiles with no signal.
SIGNAL_CHROMS = {'chrA': 400_000, 'chrB': 603, 'empty': 267}
GAP = (4096 * TILE, 6144 * TILE)
SIGNAL_TRACKS = [{'name': 'signal', 'dtype': 'float32'}, {'name': 'counts',
	'dtype': 'int32'}, {'name': 'wide', 'dtype': 'float64'}, {'name': 'nan',
	'dtype': 'float32'}, {'name': 'silent', 'dtype': 'int32'}]
SIGNAL_GROUPS = [['signal', 'counts'], ['wide', 'nan'], ['silent']]

COUNTS_CHROMS = {'chrA': 16 * 32768 + 7, 'chrB': 643, 'empty': 263}
COUNTS_GAP = (16384 * TILE, 18432 * TILE)
COUNTS_TRACKS = [{'name': 'c{}'.format(i), 'dtype': 'int32'} for i in range(4)]


def signal_values(chrom):
	"""The values of SIGNAL_TRACKS over `chrom`, (5, length): a float32 and an
	integer track, a float64 track whose tile sums do not fit f4, a track of
	NaN and a track of zeros."""

	length = SIGNAL_CHROMS[chrom]
	tiles = -(-length // TILE)
	rng = numpy.random.default_rng(len(chrom))
	rows = numpy.zeros((5, tiles))
	rows[0] = rng.normal(2, 1, tiles).astype(numpy.float32)
	rows[1] = numpy.arange(tiles) % 10 + 1
	rows[2] = 2 ** 24 + 1 + numpy.arange(tiles) % 3
	rows[3] = numpy.nan
	values = numpy.repeat(rows, TILE, axis=1)[:, :length]
	if chrom == 'chrA':
		values[:, GAP[0]:GAP[1]] = 0
	elif chrom == 'empty':
		values[:] = 0
	return values


def counts_values(chrom):
	"""The values of COUNTS_TRACKS over `chrom`, (4, length), with a gap and,
	in the last track, a value whose tile sums are 2**24 or more."""

	length = COUNTS_CHROMS[chrom]
	x = numpy.arange(length, dtype=numpy.int64)
	values = numpy.stack([x % 11, x % 7 - 3, x % 17, numpy.full(length,
		2 ** 24 + 1)])
	if chrom == 'chrA':
		values[:, COUNTS_GAP[0]:COUNTS_GAP[1]] = 0
	elif chrom == 'empty':
		values[:] = 0
	return values


def _write(path, chroms, tracks, groups, values_of, order, indexed, zooms,
	n_jobs, level, **options):
	if indexed is not None:
		options['summary_index'] = indexed
	with FigWigWriter(path, chroms, tracks, tile=TILE, groups=groups,
			order=order, level=level, engine='zlib', zooms=zooms, n_jobs=n_jobs,
			**options) as writer:
		for chrom in chroms:
			writer.write(chrom, [0], values_of(chrom)[None].astype(numpy.float64))
	return path


def write_signal(path, order='positions', indexed=True, n_jobs=2, level=6,
	**options):
	"""SIGNAL_TRACKS over SIGNAL_CHROMS, in three groups (one track per group
	in track order), three zoom levels. `indexed` None leaves the writer's
	default."""

	return _write(path, SIGNAL_CHROMS, SIGNAL_TRACKS, SIGNAL_GROUPS,
		signal_values, order, indexed, 3, n_jobs, level, **options)


def write_counts(path, order='positions', indexed=True, n_jobs=1, prefix='c'):
	"""COUNTS_TRACKS over COUNTS_CHROMS, in one group (four in track order),
	two zoom levels; the tracks are named `prefix` and 0 to 3, so that two
	files can be merged."""

	tracks = [{'name': '{}{}'.format(prefix, i), 'dtype': 'int32'} for i in
		range(4)]
	return _write(path, COUNTS_CHROMS, tracks, 4, counts_values, order, indexed,
		2, n_jobs, 6)


def expected_tile_sums(values_of, chroms, names, ids, level):
	"""The tile sums of the dense values: tracks `ids` over chromosomes
	`names`, as `FigWigReader.tile_sums(ids, names, level)` returns them,
	float32 for level 0 and float64 above."""

	pieces = []
	for name in names:
		values = numpy.nan_to_num(values_of(name)[ids], nan=0.0)
		tiles = -(-chroms[name] // TILE)
		sums = numpy.add.reduceat(values.astype(numpy.float64), numpy.arange(
			tiles) * TILE, axis=1).astype(numpy.float32)
		if level:
			factor = 4 ** level
			sums = numpy.add.reduceat(sums.astype(numpy.float64), numpy.arange(
				-(-tiles // factor)) * factor, axis=1)
		pieces.append(sums)
	return numpy.concatenate(pieces, axis=1)


class Watch:
	"""The preads and inflates a read made, as the `watched` fixture records
	them: (offset, bytes) in `preads`, (decoded bytes, raw) in `inflated`,
	where raw means a restart fragment and not a whole stream."""

	def __init__(self):
		self.preads, self.inflated = [], []

	def whole_streams(self, path):
		"""(offset, bytes) of the complete stream of every restarted level."""

		return {(level.source_offset, level.source_bytes) for group in
			read_index(path).groups for level in group.levels if level.window_bins}


def read_regions(path, watch, tracks=None):
	"""The tile sums of each chromosome of `path` read alone, at every level the
	file has, as bytes, and whether those reads inflated a restart fragment: a
	read of a chromosome that is a small part of the genome goes through the
	index, where there is one, and a read of the whole genome never does."""

	watch.inflated.clear()
	reader = FigWigReader(path)
	out = [reader.tile_sums(tracks=tracks, chroms=chrom, level=level).tobytes()
		for chrom in reader.chrom_sizes for level in range(1 + len(reader.zooms))]
	return out, any(raw for _, raw in watch.inflated)


def patch(data, at, fmt, *values):
	"""Pack `values` into the bytearray `data` at `at`."""

	struct.pack_into(fmt, data, at, *values)


def repair(data, at, size, field):
	"""Set the CRC-32 u4 at `at + field` to the CRC-32 of data[at:at + size]
	with that field as zero."""

	patch(data, at + field, '<I', 0)
	patch(data, at + field, '<I', zlib.crc32(data[at:at + size]))


def with_locator(source, out, make):
	"""Write `out` as `source` with the metadata's locator object replaced by
	`make(old)`, bytes no longer than the old, padded with spaces to its length
	(which JSON allows, and which keeps every offset), and the tables' CRC-32,
	the header's and the root's copy of the header repaired, so that only the
	locator differs. The header's offsets are those of the header table of
	docs/format.rst."""

	data = bytearray(source.read_bytes())
	n_groups = struct.unpack_from('<I', data, 24)[0]
	chroms, index, metadata, length = (struct.unpack_from('<Q', data, at)[0] for
		at in (32, 48, 64, 72))
	text = bytes(data[metadata:metadata + length])
	key = b'"org.figwig.summary-restarts": '
	start = text.index(key) + len(key)
	end = text.index(b'}', start) + 1
	new = make(text[start:end]).ljust(end - start)
	assert len(new) == end - start
	data[metadata + start:metadata + end] = new
	patch(data, 28, '<I', zlib.crc32(data[chroms:index + 8 * n_groups]))
	repair(data, 0, 128, 104)
	root = read_index(source).root
	data[root + 64:root + 192] = data[:128]
	repair(data, root, 256, 12)
	out.write_bytes(data)
	return out


def repair_page(data, level, at):
	"""Repair the CRC-32 of the page of `level` that holds the byte at `at`."""

	page = max(p for p in level.pages if p <= at)
	repair(data, page, 16 + 40 * struct.unpack_from('<I', data, page + 8)[0], 12)


def put_fragment(data, level, window, which, raw=b'', body=None):
	"""Write the `which` ('ids' or 'sums') fragment of `window` in place, with
	its length and both CRC-32 values in the descriptor and the page's CRC-32
	repaired, so that only a rule of the specification can reject it. The
	fragment is `raw` deflated and flushed as a fragment is, or `body` as
	given. `level` and `window` come from `summary_reference.read_index`, and
	the fragment has to be no longer than the one it replaces."""

	if body is None:
		compressor = zlib.compressobj(wbits=-15)
		body = compressor.compress(raw) + compressor.flush(zlib.Z_FULL_FLUSH)
	if which == 'ids':
		at, size, field = window.ids_offset, window.ids_bytes, 3
	else:
		at, size, field = window.sums_offset, window.sums_bytes, 7
	assert len(body) <= size
	data[level.source_offset + at:level.source_offset + at + len(body)] = body
	patch(data, window.at + 4 * field, '<III', len(body), zlib.crc32(body),
		zlib.crc32(raw))
	repair_page(data, level, window.at)
