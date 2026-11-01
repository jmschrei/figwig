"""The regional summary index of docs/format.rst.

A group's summary and each of its zoom levels is one zlib stream. The index
restarts the compressor inside those streams and lists, per window of bins, where
each piece starts, so that a reader inflates only the pieces for a region. It
adds nothing to the version 1 layout but a root, a directory, level catalogs and
descriptor pages in the data region, an optional bit and a metadata key.

`_SummaryIndex` is the writer's side: it compresses one level at a time, choosing
which levels to restart, and serializes the directory when the file's tables are
placed. `_SummaryIndexReader` is the reader's: it opens the root, loads and checks
the catalog of a group, and yields the rows of a level for ranges of bins from
whole streams or from restart fragments. The structures are the `struct` formats
and the descriptor dtype below; each is checked by a `parse_` or `check_`
function that takes bytes and does no reading, so that a rule can be tested
without a file.
"""

import zlib
from collections import namedtuple
from struct import Struct

import numpy


METADATA_KEY = 'org.figwig.summary-restarts'
OPTIONAL_BIT = 2
VERSION = 2
MAGIC = b'FWRSUM1\0'
PAGE_MAGIC = b'FWRP'
MAX_LEVELS = 13
PAGE_ENTRIES = 102
PAGE_STRIDE = 4096
TARGET_BYTES = 1 << 15
MAX_WINDOW_BINS = 2048
TINY_BYTES = 1 << 16
INDEX_BYTES = 1 << 24
WHOLE_LEVEL_SHARE = 0.5
MAX_LEVEL_BYTES = 1 << 29

# Where a version 1 group record keeps each stream: the summary's header (n_sum,
# n_sum_bytes, crc) and stream, then n_zooms, then each zoom level's header
# (factor, n, n_bytes, crc) and stream.
SUMMARY_HEADER = 12
ZOOM_COUNT = 4
ZOOM_HEADER = 16

# The structures. A record's CRC-32 is of its bytes with the crc field as zero.
Root = namedtuple('Root', 'magic version len_root crc directory_offset n_groups '
	'descriptors_per_page window_target reserved_a header_snapshot reserved_b')
ROOT = Struct('<8sHHIQIII28s128s64s')
ROOT_CRC_AT = 12
Locator = namedtuple('Locator', 'levels_offset n_levels levels_crc group '
	'n_tracks crc reserved')
LOCATOR = Struct('<QIIIIII')
LOCATOR_CRC_AT = 24
Entry = namedtuple('Entry', 'source_offset index_offset source_rows source_bytes '
	'source_crc window_bins')
ENTRY = Struct('<QQIIII')
PageHeader = namedtuple('PageHeader', 'magic first_window count crc')
PAGE = Struct('<4sIII')
PAGE_CRC_AT = 12
DESCRIPTOR = numpy.dtype([(name, '<u4') for name in ('n_rows',
	'first_source_row', 'ids_offset', 'ids_bytes', 'ids_crc', 'ids_raw_crc',
	'sums_offset', 'sums_bytes', 'sums_crc', 'sums_raw_crc')])
Descriptor = namedtuple('Descriptor', DESCRIPTOR.names)

ROOT_BYTES = ROOT.size

_Level = namedtuple('_Level', Entry._fields + ('total',))

# What the reader knows of the file around the index: the file's final header,
# the offset of its chromosome table (where the data region ends), its length,
# the offset of each group's version 1 record, the tracks of each group, the tiles
# of each chromosome, its order, the bytes of a run or block item in a record,
# and whether optional bit 0 (zoom sections) is set.
_Layout = namedtuple('_Layout', 'header bound length records tracks tiles order '
	'item_bytes zoomed')


class _SummaryIndexError(ValueError):
	"""A regional summary index that a reader recognizes, and could use, is
	corrupt. The message is the reason; the reader adds the file and the
	feature."""


def _require(condition, message):
	if not condition:
		raise _SummaryIndexError(message)


def _crc(data, expected, field=None):
	if field is not None:
		data = data[:field] + bytes(4) + data[field + 4:]
	_require(zlib.crc32(data) == expected, 'checksum mismatch.')


def _sealed(struct, record, body=b''):
	"""`record` packed with its CRC-32 filled in, then `body`; the CRC covers both."""

	crc = zlib.crc32(struct.pack(*record._replace(crc=0)) + body)
	return struct.pack(*record._replace(crc=crc)) + body


def _ceil_div(a, b):
	return -(-a // b)


def _inflate(data, size, raw=False):
	try:
		decoder = zlib.decompressobj(-15 if raw else 15)
		decoded = decoder.decompress(data, size + 1)
	except zlib.error:
		raise _SummaryIndexError('corrupt DEFLATE stream.') from None
	_require(len(decoded) == size and decoder.eof and not decoder.unused_data
		and not decoder.unconsumed_tail, 'wrong-sized or incomplete DEFLATE stream.')
	return decoded


def _merge_ranges(ranges):
	"""The sorted union of nonempty, half-open bin ranges, as a list of
	(first, end) tuples of ints. `ranges` is an (n, 2) array or an iterable of
	pairs; ranges that touch are one."""

	bounds = numpy.asarray(ranges if isinstance(ranges, numpy.ndarray) else
		list(ranges), dtype=numpy.int64).reshape(-1, 2)
	bounds = bounds[bounds[:, 0] < bounds[:, 1]]
	if len(bounds) == 0:
		return []
	bounds = bounds[numpy.argsort(bounds[:, 0], kind='stable')]
	reach = numpy.maximum.accumulate(bounds[:, 1])
	firsts = numpy.flatnonzero(numpy.concatenate([[True], bounds[1:, 0] >
		reach[:-1]]))
	ends = reach[numpy.concatenate([firsts[1:] - 1, [len(bounds) - 1]])]
	return list(zip(bounds[firsts, 0].tolist(), ends.tolist()))


##
# Sizes and offsets both sides compute
##

def _stream_offsets(summary_at, sizes, zoomed):
	"""Where the streams of a group's levels start in its version 1 record, and
	where the record ends, given where its summary's header starts and the
	stream sizes (the summary's first). `zoomed` is optional bit 0, which puts
	n_zooms after the summary even when it is 0."""

	offsets, at = [], summary_at + SUMMARY_HEADER
	for k, size in enumerate(sizes):
		offsets.append(at)
		at += size
		if k == 0 and zoomed:
			at += ZOOM_COUNT
		if k + 1 < len(sizes):
			at += ZOOM_HEADER
	return offsets, at


def _pages_bytes(windows):
	"""The bytes of the descriptor pages of `windows` windows."""

	return windows * DESCRIPTOR.itemsize + _ceil_div(windows, PAGE_ENTRIES) * \
		PAGE.size


def _index_bytes(total, bins):
	"""The bytes of the pages of a level of `total` bins in windows of `bins`."""

	return _pages_bytes(_ceil_div(total, bins))


def _window_bins(row_bytes):
	"""The largest power of two of at most MAX_WINDOW_BINS bins whose rows of
	`row_bytes` bytes fit the decoded target; 1 when even one row does not."""

	bins = 1
	while bins < MAX_WINDOW_BINS and bins * 2 * row_bytes <= TARGET_BYTES:
		bins *= 2
	return bins


def _within(first, bound, offset, size):
	_require(size > 0 and first <= offset <= bound - size,
		'directory section outside data region.')


##
# Parsing and checking, from bytes
##

def _parse_root(raw, header):
	"""The root in `raw`, or None when its copy of the header is not `header`."""

	root = Root._make(ROOT.unpack(raw))
	_require(root.magic == MAGIC and (root.version, root.len_root) == (VERSION,
		ROOT_BYTES), 'invalid root magic, version or size.')
	_crc(raw, root.crc, ROOT_CRC_AT)
	_require(not any(root.reserved_a) and not any(root.reserved_b),
		'nonzero reserved bytes.')
	return root if root.header_snapshot == header else None


def _parse_locator(raw, group, tracks, zoomed):
	locator = Locator._make(LOCATOR.unpack(raw))
	_crc(raw, locator.crc, LOCATOR_CRC_AT)
	_require(locator.group == group and locator.n_tracks == tracks and
		1 <= locator.n_levels <= MAX_LEVELS and (locator.n_levels == 1 or zoomed)
		and locator.reserved == 0, 'invalid group locator.')
	return locator


def _check_levels(entries, summary_at, end, tiles, tracks, zoomed, zoom_count,
	first, bound):
	"""The levels of a catalog, as `_Level`s, with the (start, end) of the pages
	each restarted level takes; refuses a level that is not where the version 1
	record has it, or is out of order, or has windows or pages that break a rule.
	`summary_at` and `end` are the version 1 record's, `tiles` those of each
	chromosome, `zoom_count` the record's n_zooms, `first` and `bound` the data
	region's."""

	offsets, last = _stream_offsets(summary_at, [e.source_bytes for e in entries], zoomed)
	_require(last == end and (not zoomed or zoom_count == len(entries) - 1),
		'catalog does not end with its version 1 record.')
	levels, sections = [], []
	for k, (entry, source) in enumerate(zip(entries, offsets)):
		total = sum(_ceil_div(int(n), 4 ** k) for n in tiles)
		level = _Level(*entry, total)
		_require(level.source_offset == source and level.source_rows <= total and
			(level.source_bytes > 0 or level.source_rows == level.source_crc == 0) and
			(k == 0 or level.source_bytes > 0), 'invalid source extent, count or order.')
		if level.window_bins:
			_require(level.window_bins <= MAX_WINDOW_BINS and level.window_bins & (level.window_bins - 1)
				== 0 and level.window_bins * (4 + (8 if k else 4) * tracks) <= TARGET_BYTES
				and level.source_rows > 0 and level.source_bytes >= 6, 'invalid window dimensions.')
			size = _index_bytes(total, level.window_bins)
			_within(first, bound, level.index_offset, size)
			sections.append((level.index_offset, level.index_offset + size))
		else:
			_require(level.index_offset == 0, 'ordinary level has a descriptor index.')
		levels.append(level)
	return levels, sections


def _parse_page(data, ref, page, tracks, wide):
	"""The descriptors of page `page` of level `ref`, in `data`, each checked
	against its window and its neighbours; `wide` is a zoom level's f8 sums."""

	first = page * PAGE_ENTRIES
	count = min(PAGE_ENTRIES, _ceil_div(ref.total, ref.window_bins) - first)
	header = PageHeader._make(PAGE.unpack_from(data))
	_require(header.magic == PAGE_MAGIC and (header.first_window, header.count) ==
		(first, count), 'invalid descriptor page header.')
	_crc(data, header.crc, PAGE_CRC_AT)
	rows = [Descriptor(*row) for row in numpy.frombuffer(data, DESCRIPTOR, count,
		PAGE.size).tolist()]
	previous, first_sums = None, None
	for i, row in enumerate(rows):
		_require(row.n_rows <= min(ref.window_bins, ref.total - (first + i) * ref.window_bins) and
			row.first_source_row <= ref.source_rows - row.n_rows, 'invalid row count or rank.')
		if not row.n_rows:
			_require(not any(row), 'nonzero empty-window descriptor.')
			continue
		for at, size, raw in [(row.ids_offset, row.ids_bytes, 4 * row.n_rows),
				(row.sums_offset, row.sums_bytes, (8 if wide else 4) * tracks * row.n_rows)]:
			_require(2 <= at <= ref.source_bytes - 4 - size and 0 < size <= 2 * raw + 64,
				'invalid fragment extent.')
		_require(row.ids_offset + row.ids_bytes <= row.sums_offset,
			'overlapping ID/sum fragments.')
		if previous is None:
			first_sums = row.sums_offset
		else:
			_require(row.first_source_row == previous.first_source_row + previous.n_rows and
				row.ids_offset >= previous.ids_offset + previous.ids_bytes and
				row.sums_offset >= previous.sums_offset + previous.sums_bytes,
				'rows or fragments overlap or are out of order.')
		previous = row
	_require(previous is None or previous.ids_offset + previous.ids_bytes <=
		first_sums, 'overlapping page fragments.')
	return rows


def _check_wrapper(header, ref, level):
	"""The version 1 header of a level's stream and its zlib header, in `header`
	(the stream's header, then the stream's first two bytes), against the
	catalog; a restarted stream has the 32 KiB window its fragments rely on."""

	size = ZOOM_HEADER if level else SUMMARY_HEADER
	expected = ((4 ** level,) if level else ()) + (ref.source_rows, ref.source_bytes, ref.source_crc)
	_require(Struct('<' + 'I' * len(expected)).unpack_from(header) == expected,
		'version 1 stream header mismatch.')
	cmf, flg = header[size:size + 2]
	_require(cmf & 15 == 8 and (cmf >> 4 == 7 if ref.window_bins else cmf >> 4 <= 7)
		and (cmf * 256 + flg) % 31 == 0 and not flg & 32, 'invalid zlib wrapper.')


##
# The reader
##

class _SummaryIndexReader:
	"""Lazy, bounded reads of a file's index. No file descriptor or decoded
	window is kept.

	``read(offset, size)`` is supplied by the caller for each operation. Only
	checked group catalogs are retained, their total size bounded by the
	directory's 16 MiB limit; pages and fragments live for one iteration.
	"""

	@classmethod
	def open(cls, read, locator, layout):
		"""The reader of the index `locator` (the metadata's) names, or None when
		there is none to use: a locator that is not an object, another extension
		version, or a root whose copy of the header is not the file's. `layout`
		is the file's `_Layout`."""

		if not isinstance(locator, dict) or locator.get('version') != VERSION:
			return None
		offset = locator.get('offset')
		_require(type(offset) is int and locator.get('length') == ROOT_BYTES and
			128 <= offset <= layout.bound - ROOT_BYTES, 'invalid root locator.')
		root = _parse_root(read(offset, ROOT_BYTES), layout.header)
		if root is None:
			return None
		first = root.directory_offset + root.n_groups * LOCATOR.size
		_require(root.directory_offset == offset + ROOT_BYTES and root.n_groups ==
			len(layout.records) and root.descriptors_per_page == PAGE_ENTRIES and
			root.window_target == TARGET_BYTES and first <= layout.bound and
			layout.bound - offset <= INDEX_BYTES, 'invalid root dimensions.')
		out = cls()
		out.layout = layout
		out.first = first
		out.directory = root.directory_offset
		out.catalogs = [None] * root.n_groups
		out.wrapped = set()
		return out

	def levels(self, read, group):
		"""The checked levels of `group`, loaded once."""

		if self.catalogs[group] is None:
			self.catalogs[group] = self._catalog(read, group)
		return self.catalogs[group]

	def _record_extent(self, group):
		records = self.layout.records
		start = int(records[group])
		end = int(records[group + 1]) if group + 1 < len(records) else \
			self.layout.length
		_require(128 <= start <= end - 8, 'invalid version 1 group extent.')
		return start, end

	def _summary_at(self, read, start, end):
		"""Where the summary's header starts in the version 1 record at `start`:
		after its runs and live tiles, or its blocks."""

		n = Struct('<I').unpack_from(read(start, 8))[0]
		at = start + 8 + n * self.layout.item_bytes
		if self.layout.order == 'positions':
			_require(at <= end - 8, 'raw index extends outside its group.')
			at += 8 + Struct('<II').unpack(read(at, 8))[1]
		_require(at <= end - SUMMARY_HEADER, 'summary starts outside its group.')
		return at

	def _catalog(self, read, group):
		layout = self.layout
		locator = _parse_locator(read(self.directory + group * LOCATOR.size,
			LOCATOR.size), group, layout.tracks[group], layout.zoomed)
		size = locator.n_levels * ENTRY.size
		_within(self.first, layout.bound, locator.levels_offset, size)
		raw = read(locator.levels_offset, size)
		_crc(raw, locator.levels_crc)
		entries = [Entry._make(row) for row in ENTRY.iter_unpack(raw)]
		start, end = self._record_extent(group)
		summary_at = self._summary_at(read, start, end)
		zoom_count = 0
		if layout.zoomed:
			# n_zooms follows the summary's stream, wherever the catalog says that
			# ends; _check_levels then refuses a catalog whose streams do not lie
			# where the record has them, so a guess that was wrong costs nothing.
			after = entries[0].source_offset + entries[0].source_bytes
			_require(after + ZOOM_COUNT <= end, 'version 1 zoom count differs from '
				'catalog.')
			zoom_count = Struct('<I').unpack(read(after, ZOOM_COUNT))[0]
		levels, sections = _check_levels(entries, summary_at, end, layout.tiles,
			layout.tracks[group], layout.zoomed, zoom_count, self.first,
			layout.bound)
		sections = sorted(sections + [(locator.levels_offset, locator.levels_offset + size)])
		_require(all(a[1] <= b[0] for a, b in zip(sections, sections[1:])),
			'overlapping directory sections.')
		return levels

	def _page(self, read, ref, page, tracks, wide):
		count = min(PAGE_ENTRIES, _ceil_div(ref.total, ref.window_bins) - page *
			PAGE_ENTRIES)
		return _parse_page(read(ref.index_offset + page * PAGE_STRIDE, PAGE.size + count *
			DESCRIPTOR.itemsize), ref, page, tracks, wide)

	def _fragment(self, read, ref, at, size, crc, raw_crc, length):
		"""The decoded bytes of one restart fragment, after its checks."""

		data = read(ref.source_offset + at, size)
		_crc(data, crc)
		_require(data.endswith(b'\x00\x00\xff\xff'), 'missing full-flush marker.')
		decoded = _inflate(data + b'\x03\x00', length, raw=True)
		_crc(decoded, raw_crc)
		return decoded

	def _whole(self, read, ref, tracks, level):
		"""The rows of a level read whole, with the stream's CRC-32 and Adler-32."""

		size = ref.source_rows * (4 + (8 if level else 4) * tracks)
		_require(size <= MAX_LEVEL_BYTES and ref.source_bytes <= MAX_LEVEL_BYTES,
			'complete summary exceeds allocation limit.')
		data = read(ref.source_offset, ref.source_bytes)
		_crc(data, ref.source_crc)
		decoded = _inflate(data, size)
		ids = numpy.frombuffer(decoded[:4 * ref.source_rows], dtype='<u4')
		_require(not ref.source_rows or ids[-1] < ref.total and (ids[1:] > ids[:-1]).all(),
			'source IDs out of order or bounds.')
		yield ids, numpy.frombuffer(decoded[4 * ref.source_rows:], dtype='<f8' if level
			else '<f4').reshape(-1, tracks)

	def _windows(self, read, ref, tracks, level, windows):
		"""The rows of the windows in the merged `windows` ranges, one window
		decoded at a time, each from its two fragments."""

		page_id, page = None, None
		for window in (j for a, b in windows for j in range(a, b)):
			if window // PAGE_ENTRIES != page_id:
				page_id = window // PAGE_ENTRIES
				page = self._page(read, ref, page_id, tracks, level > 0)
			row = page[window % PAGE_ENTRIES]
			if not row.n_rows:
				continue
			ids = numpy.frombuffer(self._fragment(read, ref, row.ids_offset,
				row.ids_bytes, row.ids_crc, row.ids_raw_crc, 4 * row.n_rows),
				dtype='<u4')
			sums = self._fragment(read, ref, row.sums_offset, row.sums_bytes,
				row.sums_crc, row.sums_raw_crc, (8 if level else 4) * tracks *
				row.n_rows)
			_require(ids[0] >= window * ref.window_bins and ids[-1] < min(ref.total,
				(window + 1) * ref.window_bins) and (ids[1:] > ids[:-1]).all(),
				'window IDs out of order or bounds.')
			yield ids, numpy.frombuffer(sums, dtype='<f8' if level else
				'<f4').reshape(-1, tracks)

	def rows(self, read, group, level, ranges):
		"""Yield sparse bin IDs and sums for `ranges` of bins of `level`, keeping
		at most one window decoded.

		A query that wants at least WHOLE_LEVEL_SHARE of a level's windows, and a
		level that is not restarted, read the complete zlib stream, with its
		CRC-32 and Adler-32; otherwise the requested pages and both fragment
		CRC-32 values are checked. `level` is one of the group's `levels` and has
		a stream (``source_bytes`` is not 0).
		"""

		ref, tracks = self.levels(read, group)[level], self.layout.tracks[group]
		if (group, level) not in self.wrapped:
			size = ZOOM_HEADER if level else SUMMARY_HEADER
			_check_wrapper(read(ref.source_offset - size, size + 2), ref, level)
			self.wrapped.add((group, level))
		if ref.window_bins:
			bounds = numpy.asarray(ranges, dtype=numpy.int64).reshape(-1, 2)
			windows = _merge_ranges(numpy.column_stack([bounds[:, 0] //
				ref.window_bins, -(-bounds[:, 1] // ref.window_bins)]))
			covered = sum(b - a for a, b in windows)
		if not ref.window_bins or covered >= WHOLE_LEVEL_SHARE * _ceil_div(ref.total,
				ref.window_bins):
			yield from self._whole(read, ref, tracks, level)
		else:
			yield from self._windows(read, ref, tracks, level, windows)


##
# The writer
##

def _restart(ids, sums, bins, windows, dtype):
	"""One level compressed with a restart after the ids and the sums of each
	window that has rows: the stream, and the descriptors of its `windows`
	windows (a DESCRIPTOR array, all zero for a window with no rows). The
	stream inflates to the ids then the sums, as the ordinary stream does."""

	ranks = numpy.searchsorted(ids, numpy.arange(windows + 1, dtype=numpy.int64)
		* bins)
	occupied = numpy.flatnonzero(ranks[1:] != ranks[:-1])
	entries = numpy.zeros(windows, DESCRIPTOR)
	entries['n_rows'][occupied] = (ranks[1:] - ranks[:-1])[occupied]
	entries['first_source_row'][occupied] = ranks[:-1][occupied]
	compressor = zlib.compressobj(6)
	stream = bytearray(compressor.compress(b''))
	for values, name, raw_dtype in ((ids, 'ids', '<u4'), (sums, 'sums', dtype)):
		for window in occupied:
			raw = values[ranks[window]:ranks[window + 1]].astype(raw_dtype).tobytes()
			at = len(stream)
			fragment = compressor.compress(raw) + compressor.flush(zlib.Z_FULL_FLUSH)
			for field, value in (('_offset', at), ('_bytes', len(fragment)), ('_crc',
					zlib.crc32(fragment)), ('_raw_crc', zlib.crc32(raw))):
				entries[name + field][window] = value
			stream.extend(fragment)
	stream.extend(compressor.flush(zlib.Z_FINISH))
	return bytes(stream), entries


def _pages_of(entries):
	"""The descriptor pages of an array of descriptors, one `bytes` each."""

	pages = []
	for first in range(0, len(entries), PAGE_ENTRIES):
		part = entries[first:first + PAGE_ENTRIES]
		pages.append(_sealed(PAGE, PageHeader(PAGE_MAGIC, first, len(part), 0),
			part.tobytes()))
	return pages


class _SummaryIndex:
	"""Compress one level at a time without retaining another decoded pyramid.

	`tracks` is the number of tracks of each group and `bins` the bins of each
	level over the genome. Each group has a deterministic share of the
	directory budget, so the choice of indexed levels does not depend on
	compression thread order.
	"""

	def __init__(self, tracks, bins):
		self.tracks = tracks
		self.bins = bins
		self.levels = [[] for _ in tracks]
		self.raw_sizes = [0] * len(tracks)
		base = ROOT_BYTES + len(tracks) * (LOCATOR.size + len(bins) * ENTRY.size)
		self.remaining = [max(0, (INDEX_BYTES - base) // len(tracks))] * len(tracks)

	def compress(self, group, level, ids, sums):
		"""The stream of one level of `group`, restarted if the level is large
		enough and the group's share of the budget holds its pages, and the
		ordinary stream otherwise; the level's record is kept for `serialize`."""

		dtype = '<f8' if level else '<f4'
		row_bytes = 4 + (8 if level else 4) * self.tracks[group]
		raw_bytes = len(ids) * row_bytes
		bins = _window_bins(row_bytes)
		needed = _index_bytes(self.bins[level], bins)
		eligible = len(ids) > 0 and row_bytes <= TARGET_BYTES and \
			needed <= self.remaining[group]
		stream = None
		if not eligible or raw_bytes <= TINY_BYTES:
			stream = zlib.compress(ids.astype('<u4').tobytes() +
				sums.astype(dtype).tobytes(), 6)
			eligible = eligible and (raw_bytes > TINY_BYTES or
				len(stream) > TINY_BYTES)
		entries = None
		if eligible:
			# The share is spent before the descriptor table exists.
			self.remaining[group] -= needed
			stream, entries = _restart(ids, sums, bins, _ceil_div(self.bins[level],
				bins), dtype)
		else:
			bins = 0
		self.levels[group].append((len(ids), len(stream), zlib.crc32(stream), bins,
			entries))
		return stream

	def size(self):
		"""The bytes `serialize` returns, 0 when no level is restarted."""

		restarted = [entries for levels in self.levels for _, _, _, _, entries in
			levels if entries is not None]
		if not restarted:
			return 0
		return ROOT_BYTES + len(self.tracks) * LOCATOR.size + sum(len(levels) for
			levels in self.levels) * ENTRY.size + sum(_pages_bytes(len(entries)) for
			entries in restarted)

	def serialize(self, offset, directory, header):
		"""The index as `size()` bytes to write at `offset`, once the file's tables
		are placed: `directory` holds each group's version 1 record offset, and
		`header` the file's final 128 header bytes."""

		locators, body = [], []
		at = ROOT_BYTES + len(self.tracks) * LOCATOR.size
		for group, levels in enumerate(self.levels):
			sources, _ = _stream_offsets(int(directory[group]) + self.raw_sizes[group],
				[size for _, size, _, _, _ in levels], True)
			catalog, pages = [], []
			catalog_at = at
			at += len(levels) * ENTRY.size
			for (rows, size, crc, bins, entries), source in zip(levels, sources):
				index_at = 0
				if entries is not None:
					index_at = offset + at
					level_pages = _pages_of(entries)
					pages.extend(level_pages)
					at += sum(len(page) for page in level_pages)
				catalog.append(ENTRY.pack(source, index_at, rows, size, crc, bins))
			catalog = b''.join(catalog)
			locators.append(_sealed(LOCATOR, Locator(offset + catalog_at,
				len(levels), zlib.crc32(catalog), group, self.tracks[group], 0, 0)))
			body.append(catalog + b''.join(pages))
		root = _sealed(ROOT, Root(MAGIC, VERSION, ROOT_BYTES, 0, offset +
			ROOT_BYTES, len(self.tracks), PAGE_ENTRIES, TARGET_BYTES, bytes(28),
			bytes(header), bytes(64)))
		return root + b''.join(locators) + b''.join(body)
