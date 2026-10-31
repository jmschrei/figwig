"""Optional v1 summary restart directory; see docs/summary_restarts.rst."""

import struct
import zlib
from collections import namedtuple

import numpy


METADATA_KEY = 'org.figwig.summary-restarts'
OPTIONAL_BIT = 2
VERSION = 2
ROOT_BYTES = 256
PAGE_ENTRIES = 102
TARGET_BYTES = 1 << 15
MAX_WINDOW_BINS = 2048
TINY_BYTES = 1 << 16
INDEX_BYTES = 1 << 24

_Level = namedtuple('_Level', 'source index rows size crc span total')


class SummaryIndexError(ValueError):
	"""A recognized, current-generation summary accelerator is corrupt."""


def _require(condition, message):
	if not condition:
		raise SummaryIndexError('The summary restart index cannot be read: ' + message)


def _crc(data, expected, field=None):
	if field is not None:
		data = data[:field] + bytes(4) + data[field + 4:]
	_require(zlib.crc32(data) == expected, 'checksum mismatch.')


def _inflate(data, size, raw=False):
	try:
		decoder = zlib.decompressobj(-15 if raw else 15)
		decoded = decoder.decompress(data, size + 1)
	except zlib.error:
		raise SummaryIndexError('The summary restart index cannot be read: '
			'corrupt DEFLATE stream.') from None
	_require(len(decoded) == size and decoder.eof and not decoder.unused_data
		and not decoder.unconsumed_tail, 'wrong-sized or incomplete DEFLATE stream.')
	return decoded


def merge_ranges(ranges):
	"""The sorted union of nonempty, half-open bin ranges."""
	out = []
	for first, end in sorted(ranges):
		if first >= end:
			continue
		if out and first <= out[-1][1]:
			out[-1] = (out[-1][0], max(end, out[-1][1]))
		else:
			out.append((int(first), int(end)))
	return out


class SummaryIndexReader:
	"""Lazy, bounded directory reads. No file descriptor or decoded windows are kept.

	``read(offset, size)`` is supplied by the caller for each operation. Only
	checked group catalogs are retained; their total size is bounded by the
	directory's 16 MiB limit. Pages and fragments live for one iteration.
	"""

	@classmethod
	def open(cls, read, locator, header, bound, directory, tracks, tiles, order):
		if not isinstance(locator, dict) or locator.get('version') != VERSION:
			return None
		offset = locator.get('offset')
		_require(type(offset) is int and locator.get('length') == ROOT_BYTES and
			128 <= offset <= bound - ROOT_BYTES, 'invalid root locator.')
		root = read(offset, ROOT_BYTES)
		_require(root[:8] == b'FWRSUM1\0' and struct.unpack_from('<HH', root, 8)
			== (VERSION, ROOT_BYTES), 'invalid root magic, version or size.')
		_crc(root, struct.unpack_from('<I', root, 12)[0], 12)
		_require(not any(root[36:64]) and not any(root[192:]), 'nonzero reserved bytes.')
		if root[64:192] != header:
			return None
		at, groups, entries, target = struct.unpack_from('<QIII', root, 16)
		_require(at == offset + ROOT_BYTES and groups == len(directory) and
			entries == PAGE_ENTRIES and target == TARGET_BYTES and
			at + groups * 32 <= bound and bound - offset <= INDEX_BYTES,
			'invalid root dimensions.')
		out = cls()
		out.first = at + groups * 32
		out.bound = bound
		out.directory = at
		out.groups = directory
		out.tracks = tracks
		out.tiles = tiles
		out.order = order
		out.optional = struct.unpack_from('<I', header, 100)[0]
		out.length = struct.unpack_from('<Q', header, 88)[0]
		out.catalogs = [None] * groups
		return out

	def _bounds(self, offset, size):
		_require(size > 0 and self.first <= offset <= self.bound - size,
			'directory section outside data region.')

	def levels(self, read, group):
		if self.catalogs[group] is not None:
			return self.catalogs[group]
		entry = read(self.directory + group * 32, 32)
		at, count, crc, identity, tracks, checksum, reserved = struct.unpack('<Q6I', entry)
		_crc(entry, checksum, 24)
		_require(identity == group and tracks == self.tracks[group] and
			1 <= count <= 13 and (count == 1 or self.optional & 1) and reserved == 0,
			'invalid group locator.')
		self._bounds(at, count * 32)
		data = read(at, count * 32)
		_crc(data, crc)
		start = int(self.groups[group])
		end = int(self.groups[group + 1]) if group + 1 < len(self.groups) else self.length
		_require(128 <= start <= end - 8, 'invalid legacy group extent.')
		n = struct.unpack_from('<I', read(start, 8))[0]
		base = start + 8 + n * (40 if self.order == 'tracks' else 32)
		if self.order == 'positions':
			_require(base <= end - 8, 'raw index extends outside its group.')
			_, size = struct.unpack('<II', read(base, 8))
			base += 8 + size
		_require(base <= end - 12, 'summary starts outside its group.')
		sections = [(at, at + count * 32)]
		levels = []
		for k in range(count):
			total = sum(-(-int(n) // 4 ** k) for n in self.tiles)
			ref = _Level(*struct.unpack_from('<QQ4I', data, k * 32), total)
			_require(ref.source == base + (16 if k else 12) and
				ref.source <= end - ref.size and ref.rows <= total and
				(ref.size > 0 or ref.rows == ref.crc == 0) and (k == 0 or ref.size > 0),
				'invalid source extent, count or order.')
			if ref.span:
				_require(ref.span <= MAX_WINDOW_BINS and ref.span & (ref.span - 1) == 0 and
					ref.span * (4 + (8 if k else 4) * tracks) <= TARGET_BYTES and
					ref.rows > 0 and ref.size >= 6, 'invalid window dimensions.')
				windows = -(-total // ref.span)
				size = windows * 40 + -(-windows // PAGE_ENTRIES) * 16
				self._bounds(ref.index, size)
				sections.append((ref.index, ref.index + size))
			else:
				_require(ref.index == 0, 'ordinary level has a descriptor index.')
			levels.append(ref)
			base = ref.source + ref.size
			if k == 0 and self.optional & 1:
				_require(base <= end - 4 and struct.unpack('<I', read(base, 4))[0]
					== count - 1, 'legacy zoom count differs from catalog.')
				base += 4
		_require(base == end, 'catalog does not end with its legacy group.')
		sections.sort()
		_require(all(a[1] <= b[0] for a, b in zip(sections, sections[1:])),
			'overlapping directory sections.')
		self.catalogs[group] = levels
		return levels

	def _page(self, read, ref, page, tracks, wide):
		first = page * PAGE_ENTRIES
		count = min(PAGE_ENTRIES, -(-ref.total // ref.span) - first)
		data = read(ref.index + page * 4096, 16 + count * 40)
		_require(data[:4] == b'FWRP' and struct.unpack_from('<II', data, 4) ==
			(first, count), 'invalid descriptor page header.')
		_crc(data, struct.unpack_from('<I', data, 12)[0], 12)
		previous, first_sum = None, None
		rows = []
		for i in range(count):
			row = struct.unpack_from('<10I', data, 16 + i * 40)
			n, rank = row[:2]
			_require(n <= min(ref.span, ref.total - (first + i) * ref.span) and
				rank <= ref.rows - n, 'invalid row count or rank.')
			if not n:
				_require(not any(row), 'nonzero empty-window descriptor.')
			else:
				for offset, size, raw in [(row[2], row[3], 4 * n),
						(row[6], row[7], (8 if wide else 4) * tracks * n)]:
					_require(2 <= offset <= ref.size - 4 - size and
						0 < size <= 2 * raw + 64, 'invalid fragment extent.')
				_require(row[2] + row[3] <= row[6], 'overlapping ID/sum fragments.')
				if previous is not None:
					_require(rank == previous[1] + previous[0] and
						row[2] >= previous[2] + previous[3] and
						row[6] >= previous[6] + previous[7],
						'rows or fragments overlap or are out of order.')
				else:
					first_sum = row[6]
				previous = row
			rows.append(row)
		_require(previous is None or previous[2] + previous[3] <= first_sum,
			'overlapping page fragments.')
		return rows

	def rows(self, read, group, level, ranges):
		"""Yield sparse bin IDs and sums, keeping at most one window decoded.

		Broad queries use the complete zlib stream, with its CRC and Adler-32;
		a regional query validates the requested pages and both fragment CRCs.
		"""
		levels = self.levels(read, group)
		if level >= len(levels) or levels[level].size == 0:
			return
		ref = levels[level]
		tracks, width = self.tracks[group], 8 if level else 4
		header_size = 16 if level else 12
		header = read(ref.source - header_size, header_size + 2)
		expected = ((4 ** level,) if level else ()) + (ref.rows, ref.size, ref.crc)
		_require(struct.unpack_from('<' + 'I' * len(expected), header) == expected,
			'legacy source header mismatch.')
		cmf, flg = header[-2:]
		_require(cmf & 15 == 8 and (cmf >> 4 == 7 if ref.span else cmf >> 4 <= 7)
			and (cmf * 256 + flg) % 31 == 0 and not flg & 32, 'invalid zlib wrapper.')
		dtype = '<f8' if level else '<f4'
		windows = merge_ranges((a // ref.span, -(-b // ref.span)) for a, b
			in ranges) if ref.span else []
		if not ref.span or sum(b - a for a, b in windows) * 2 >= -(-ref.total // ref.span):
			size = ref.rows * (4 + width * tracks)
			_require(size <= 1 << 29 and ref.size <= 1 << 29,
				'complete summary exceeds allocation limit.')
			data = read(ref.source, ref.size)
			_crc(data, ref.crc)
			decoded = _inflate(data, size)
			ids = numpy.frombuffer(decoded[:4 * ref.rows], dtype='<u4')
			_require(not ref.rows or ids[-1] < ref.total and (ids[1:] > ids[:-1]).all(),
				'source IDs out of order or bounds.')
			yield ids, numpy.frombuffer(decoded[4 * ref.rows:], dtype=dtype).reshape(-1, tracks)
			return
		page_id, entries = None, None
		for window in (j for a, b in windows for j in range(a, b)):
			if window // PAGE_ENTRIES != page_id:
				page_id = window // PAGE_ENTRIES
				entries = self._page(read, ref, page_id, tracks, level > 0)
			row = entries[window % PAGE_ENTRIES]
			if not row[0]:
				continue
			parts = []
			for start, raw_size in [(2, 4 * row[0]), (6, width * tracks * row[0])]:
				offset, size, crc, raw_crc = row[start:start + 4]
				data = read(ref.source + offset, size)
				_crc(data, crc)
				_require(data.endswith(b'\x00\x00\xff\xff'), 'missing full-flush marker.')
				decoded = _inflate(data + b'\x03\x00', raw_size, raw=True)
				_crc(decoded, raw_crc)
				parts.append(decoded)
			ids = numpy.frombuffer(parts[0], dtype='<u4')
			_require(ids[0] >= window * ref.span and ids[-1] < min(ref.total,
				(window + 1) * ref.span) and (ids[1:] > ids[:-1]).all(),
				'window IDs out of order or bounds.')
			yield ids, numpy.frombuffer(parts[1], dtype=dtype).reshape(-1, tracks)


class SummaryIndex:
	"""Compress one level at a time without retaining another decoded pyramid.

	Each group has a deterministic share of the directory budget, so the
	choice of indexed levels does not depend on compression thread order.
	"""

	def __init__(self, tracks, bins):
		self.tracks = tracks
		self.bins = bins
		self.levels = [[] for _ in tracks]
		self.raw_sizes = [0] * len(tracks)
		base = ROOT_BYTES + len(tracks) * (32 + len(bins) * 32)
		self.remaining = [max(0, (INDEX_BYTES - base) // len(tracks))] * len(tracks)

	def compress(self, group, level, ids, sums):
		width = 4 if level == 0 else 8
		dtype = '<f4' if level == 0 else '<f8'
		row_bytes = 4 + width * self.tracks[group]
		raw_bytes = len(ids) * row_bytes
		window_bins = 1
		while window_bins < MAX_WINDOW_BINS and window_bins * 2 * row_bytes <= TARGET_BYTES:
			window_bins *= 2
		n_windows = -(-self.bins[level] // window_bins)
		index_bytes = n_windows * 40 + -(-n_windows // PAGE_ENTRIES) * 16
		eligible = len(ids) > 0 and row_bytes <= TARGET_BYTES and \
			index_bytes <= self.remaining[group]
		stream = None
		if not eligible or raw_bytes <= TINY_BYTES:
			stream = zlib.compress(ids.astype('<u4').tobytes() +
				sums.astype(dtype).tobytes(), 6)
			eligible = eligible and (raw_bytes > TINY_BYTES or
				len(stream) > TINY_BYTES)
		entries = None
		if eligible:
			# The allocation is budgeted before the dense descriptor table exists.
			self.remaining[group] -= index_bytes
			entries = bytearray(n_windows * 40)
			# Ranks scale with directory windows, not every listed source row.
			ranks = numpy.searchsorted(ids, numpy.arange(n_windows + 1,
				dtype=numpy.int64) * window_bins)
			occupied = numpy.flatnonzero(ranks[1:] != ranks[:-1])
			compressor = zlib.compressobj(6)
			stream = bytearray(compressor.compress(b''))
			for values, field in ((ids, 8), (sums, 24)):
				for window in occupied:
					first, end = ranks[window], ranks[window + 1]
					if field == 8:
						struct.pack_into('<II', entries, window * 40,
							int(end - first), int(first))
					raw = values[first:end].astype('<u4' if field == 8 else
						dtype).tobytes()
					offset = len(stream)
					fragment = compressor.compress(raw) + compressor.flush(
						zlib.Z_FULL_FLUSH)
					if len(fragment) > 2 * len(raw) + 64 or not fragment.endswith(
							b'\x00\x00\xff\xff'):
						raise ValueError('a summary restart fragment exceeds its bound.')
					struct.pack_into('<IIII', entries, window * 40 + field,
						offset, len(fragment), zlib.crc32(fragment), zlib.crc32(raw))
					stream.extend(fragment)
			stream.extend(compressor.flush(zlib.Z_FINISH))
			stream = bytes(stream)
		else:
			window_bins = 0
		self.levels[group].append((len(ids), len(stream), zlib.crc32(stream),
			window_bins, entries))
		return stream

	def size(self):
		if not any(level[4] is not None for levels in self.levels for level in levels):
			return 0
		size = ROOT_BYTES + len(self.tracks) * 32
		for levels in self.levels:
			size += len(levels) * 32
			for _, _, _, _, entries in levels:
				if entries is not None:
					n_windows = len(entries) // 40
					size += len(entries) + -(-n_windows // PAGE_ENTRIES) * 16
		return size

	def serialize(self, offset, directory, header):
		"""Resolve legacy payload offsets after core table locations are known."""
		out = bytearray(self.size())
		out[:8] = b'FWRSUM1\0'
		struct.pack_into('<HHIQIII', out, 8, VERSION, ROOT_BYTES, 0,
			offset + ROOT_BYTES, len(self.tracks), PAGE_ENTRIES, TARGET_BYTES)
		out[64:192] = header
		at = ROOT_BYTES + len(self.tracks) * 32
		for group, levels in enumerate(self.levels):
			catalog_at = at
			at += len(levels) * 32
			source = int(directory[group]) + self.raw_sizes[group] + 12
			for level, (n, size, crc, window_bins, entries) in enumerate(levels):
				if level:
					source += 20 if level == 1 else 16
				index_at = offset + at if entries is not None else 0
				struct.pack_into('<QQIIII', out, catalog_at + level * 32,
					source, index_at, n, size, crc, window_bins)
				if entries is not None:
					n_windows = len(entries) // 40
					for first in range(0, n_windows, PAGE_ENTRIES):
						count = min(PAGE_ENTRIES, n_windows - first)
						length = 16 + count * 40
						out[at:at + 4] = b'FWRP'
						struct.pack_into('<III', out, at + 4, first, count, 0)
						out[at + 16:at + length] = entries[first * 40:(first + count) * 40]
						struct.pack_into('<I', out, at + 12, zlib.crc32(
							memoryview(out)[at:at + length]))
						at += length
				source += size
			locator = ROOT_BYTES + group * 32
			struct.pack_into('<QIIIIII', out, locator, offset + catalog_at,
				len(levels), zlib.crc32(memoryview(out)[catalog_at:catalog_at +
					len(levels) * 32]), group, self.tracks[group], 0, 0)
			struct.pack_into('<I', out, locator + 24, zlib.crc32(
				memoryview(out)[locator:locator + 32]))
		struct.pack_into('<I', out, 12, zlib.crc32(memoryview(out)[:ROOT_BYTES]))
		return out
