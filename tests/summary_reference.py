# summary_reference.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""A reader of the regional summary index, written from "The regional
summary index" of docs/format.rst with struct, zlib and numpy, on top of
the specification's reader of version 1 (format_reference.py). It imports
nothing from figwig.

`read_index(path)` returns None for a file that has no current index: the
bit or the key missing, a version it does not know, or a root whose copy of
the header is not the file's. It raises FormatError for an index that breaks
a rule of the page, and otherwise an `Index` whose windows hold their
decoded rows, each checked against the rows the version 1 reader finds in
the ordinary stream.
"""

import struct
import zlib
from collections import namedtuple

import numpy

from . import format_reference


KEY = 'org.figwig.summary-restarts'
ROOT_BYTES = 256
PAGE_BYTES = 4096
PER_PAGE = 102
TARGET = 32768
INDEX_LIMIT = 1 << 24

Index = namedtuple('Index', 'root header groups sections')
Group = namedtuple('Group', 'locator catalog levels')
Level = namedtuple('Level', 'k factor source_offset source_rows source_bytes '
	'source_crc window_bins index_offset total pages windows')
Window = namedtuple('Window', 'number at n_rows first_source_row ids_offset '
	'ids_bytes sums_offset sums_bytes ids sums')


def fail(message):
	raise format_reference.FormatError('summary index: ' + message)


def require(condition, message):
	if not condition:
		fail(message)


def crc_of(data, field=None):
	data = bytearray(data)
	if field is not None:
		data[field:field + 4] = bytes(4)
	return zlib.crc32(data)


def fragment(stream, at, size, crc, raw_crc, length):
	"""The decoded bytes of a fragment of `stream`, after every check the page
	lists for one."""

	require(2 <= at and size > 0 and at + size <= len(stream) - 4,
		'a fragment lies outside its stream')
	require(size <= 2 * length + 64, 'a fragment is too long for its rows')
	data = stream[at:at + size]
	require(zlib.crc32(data) == crc, 'a fragment fails its CRC-32')
	require(data.endswith(b'\x00\x00\xff\xff'), 'a fragment lacks its marker')
	decoder = zlib.decompressobj(-15)
	try:
		raw = decoder.decompress(data + b'\x03\x00', length + 1)
	except zlib.error:
		fail('a fragment is not DEFLATE')
	require(decoder.eof and not decoder.unused_data and len(raw) == length,
		'a fragment does not inflate to its length')
	require(zlib.crc32(raw) == raw_crc, 'a decoded fragment fails its CRC-32')
	return raw


def streams(file, g):
	"""(first byte, rows, bytes, crc, ids, sums) of each level of group `g` in
	the version 1 record, the summary first."""

	data, at = file.bytes, int(file.directory[g])
	n = struct.unpack_from('<I', data, at)[0]
	if file.header.order == 'tracks':
		at += 8 + n * 40
	else:
		at += 8 + n * 32
		at += 8 + struct.unpack_from('<I', data, at + 4)[0]
	rows, size, crc = struct.unpack_from('<III', data, at)
	levels = [(at + 12, rows, size, crc) + tuple(file.groups[g].summary or (None, None))]
	at += 12 + size
	if file.header.optional & 1:
		count = struct.unpack_from('<I', data, at)[0]
		at += 4
		for k in range(count):
			_, rows, size, crc = struct.unpack_from('<IIII', data, at)
			ids_sums = file.groups[g].zooms[k][1:]
			levels.append((at + 16, rows, size, crc) + tuple(ids_sums))
			at += 16 + size
	return levels


def check_wrapper(stream, restarted):
	"""The zlib header of a level's stream: deflate, no preset dictionary, a
	valid check value, and for a restarted level the 32 KiB window."""

	cmf, flg = stream[0], stream[1]
	require(cmf & 15 == 8 and (cmf >> 4 == 7 if restarted else cmf >> 4 <= 7) and
		(cmf * 256 + flg) % 31 == 0 and not flg & 32, 'a stream has an unusable zlib header')


def read_level(file, k, source, window_bins, index, tracks, sections):
	rows, size, crc, ids, sums = source[1:]
	factor = 4 ** k
	if size:
		check_wrapper(file.bytes[source[0]:source[0] + 2], bool(window_bins))
	total = sum(-(-int(t) // factor) for t in numpy.diff(file.tile_offsets))
	width = 8 if k else 4
	level = dict(k=k, factor=factor, source_offset=source[0], source_rows=rows,
		source_bytes=size, source_crc=crc, window_bins=window_bins,
		index_offset=index, total=total, pages=[], windows={})
	if not window_bins:
		require(index == 0, 'an ordinary level has pages')
		return Level(**level)
	require(window_bins & (window_bins - 1) == 0 and 0 < window_bins <= 2048,
		'bins is not a power of two of at most 2,048')
	require(window_bins * (4 + width * tracks) <= TARGET and rows > 0,
		'a restarted level is wider than the target or has no rows')
	windows = -(-total // window_bins)
	pages = -(-windows // PER_PAGE)
	sections.append((index, index + windows * 40 + pages * 16))
	stream = file.bytes[source[0]:source[0] + size]
	extents = []
	for p in range(pages):
		at = index + p * PAGE_BYTES
		count = min(PER_PAGE, windows - p * PER_PAGE)
		page = file.bytes[at:at + 16 + 40 * count]
		require(page[:4] == b'FWRP', 'a page lacks its magic')
		first, n, crc_ = struct.unpack_from('<III', page, 4)
		require((first, n) == (p * PER_PAGE, count), 'a page has the wrong range')
		require(crc_of(page, 12) == crc_, 'a page fails its CRC-32')
		level['pages'].append(at)
		previous = None
		for slot in range(count):
			fields = struct.unpack_from('<10I', page, 16 + 40 * slot)
			number = p * PER_PAGE + slot
			low, high = number * window_bins, min(total, (number + 1) * window_bins)
			lo, hi = numpy.searchsorted(ids, [low, high])
			if fields[0] == 0:
				require(not any(fields) and hi == lo, 'a window with no rows is not empty')
				continue
			n_rows, rank, ids_at, ids_size, ids_crc, ids_raw, sums_at, sums_size, \
				sums_crc, sums_raw = fields
			require((n_rows, rank) == (hi - lo, lo), 'a window has the wrong rows')
			require(ids_at + ids_size <= sums_at, 'ids and sums fragments overlap')
			if previous is not None:
				require(rank == previous[1] + previous[0] and ids_at >= previous[2] +
					previous[3] and sums_at >= previous[6] + previous[7],
					'windows are out of order')
			previous = fields
			extents += [(ids_at, ids_at + ids_size), (sums_at, sums_at + sums_size)]
			got_ids = numpy.frombuffer(fragment(stream, ids_at, ids_size, ids_crc,
				ids_raw, 4 * n_rows), dtype='<u4')
			got_sums = numpy.frombuffer(fragment(stream, sums_at, sums_size,
				sums_crc, sums_raw, width * tracks * n_rows), dtype='<f8' if k else
				'<f4').reshape(n_rows, tracks)
			require(numpy.array_equal(got_ids, ids[lo:hi]), 'a fragment holds other bin numbers')
			require(got_sums.tobytes() == sums[lo:hi].astype(got_sums.dtype).tobytes(),
				'a fragment holds other sums')
			level['windows'][number] = Window(number, at + 16 + 40 * slot, n_rows,
				rank, ids_at, ids_size, sums_at, sums_size, got_ids, got_sums)
	extents.sort()
	require(all(a[1] <= b[0] for a, b in zip(extents, extents[1:])), 'two fragments overlap')
	return Level(**level)


def read_index(path):
	file = format_reference.File(path)
	locator = file.metadata.get(KEY)
	if not file.header.optional & 2 or locator is None:
		return None
	if not isinstance(locator, dict) or locator.get('version') != 2:
		return None
	root = locator.get('offset')
	require(type(root) is int and locator.get('length') == ROOT_BYTES and
		128 <= root <= file.header.chroms - ROOT_BYTES, 'the root lies outside the data region')
	require(file.header.chroms - root <= INDEX_LIMIT, 'the index is longer than 16 MiB')
	data = file.bytes
	raw = data[root:root + ROOT_BYTES]
	require(raw[:8] == b'FWRSUM1\0', 'the root lacks its magic')
	version, size, crc, directory, groups, per_page, target = \
		struct.unpack_from('<HHIQIII', raw, 8)
	require((version, size) == (2, ROOT_BYTES), 'the root has the wrong version or size')
	require(crc_of(raw, 12) == crc, 'the root fails its CRC-32')
	require(not any(raw[36:64]) and not any(raw[192:]), 'the root has nonzero reserved bytes')
	if raw[64:192] != data[:128]:
		return None
	require((directory, groups, per_page, target) == (root + ROOT_BYTES,
		file.header.n_groups, PER_PAGE, TARGET), 'the root has wrong dimensions')
	sections = [(root, directory + 32 * groups)]
	out = []
	for g in range(groups):
		entry = data[directory + 32 * g:directory + 32 * g + 32]
		catalog, n_levels, levels_crc, group, tracks, crc, reserved = \
			struct.unpack('<QIIIIII', entry)
		require(crc_of(entry, 24) == crc, 'a locator fails its CRC-32')
		require((group, tracks, reserved) == (g, len(file.group_tracks[g]), 0) and
			1 <= n_levels <= 13, 'a locator holds the wrong group, tracks or levels')
		require(n_levels == 1 or file.header.optional & 1, 'zoom levels need optional bit 0')
		require(128 <= catalog and catalog + 32 * n_levels <= file.header.chroms,
			'a catalog lies outside the data region')
		sections.append((catalog, catalog + 32 * n_levels))
		raw_catalog = data[catalog:catalog + 32 * n_levels]
		require(zlib.crc32(raw_catalog) == levels_crc, 'a catalog fails its CRC-32')
		sources = streams(file, g)
		require(len(sources) == n_levels, 'the catalog has another number of levels')
		levels = []
		for k in range(n_levels):
			source, index, rows, size, source_crc, bins = struct.unpack_from(
				'<QQIIII', raw_catalog, 32 * k)
			require((source, rows, size, source_crc) == sources[k][:4],
				'a catalog entry is not the version 1 stream')
			levels.append(read_level(file, k, sources[k], bins, index, tracks,
				sections))
		out.append(Group(directory + 32 * g, catalog, levels))
	sections.sort()
	require(all(a[1] <= b[0] and b[1] <= file.header.chroms for a, b in
		zip(sections, sections[1:] + [(file.header.chroms, file.header.chroms)])),
		'two structures overlap or pass the chromosome table')
	return Index(root, data[:128], out, sections)
