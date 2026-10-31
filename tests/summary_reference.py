"""Independent, whole-file verifier for docs/summary_restarts.rst.

This intentionally imports no production summary-index encoder or parser.
``read_index(path)`` returns None for an unadvertised/unknown/stale extension,
otherwise root metadata and groups/levels, with decoded occupied windows.
The unchanged v1 specification reader supplies the original summary oracle.
"""

import struct
import zlib

import numpy

from . import format_reference as v1


KEY = 'org.figwig.summary-restarts'


def require(condition, message):
	if not condition:
		raise v1.FormatError('summary restart index: ' + message)


def check_crc(data, expected, message, field=None):
	if field is not None:
		data = bytearray(data)
		struct.pack_into('<I', data, field, 0)
	require(zlib.crc32(data) == expected, message + ' CRC mismatch')


def fragment(data, relative, size, crc, raw_crc, expected):
	require(2 <= relative and size > 0 and relative <= len(data) - 4 - size,
		'fragment outside source stream')
	require(size <= 2 * len(expected) + 64, 'oversized compressed fragment')
	encoded = data[relative:relative + size]
	check_crc(encoded, crc, 'fragment')
	require(encoded.endswith(b'\x00\x00\xff\xff'), 'missing full-flush marker')
	decoder = zlib.decompressobj(-15)
	try:
		raw = decoder.decompress(encoded + b'\x03\x00', len(expected) + 1)
	except zlib.error as error:
		raise v1.FormatError('invalid restart DEFLATE: ' + str(error)) from None
	require(len(raw) == len(expected) and decoder.eof and not decoder.unused_data
		and not decoder.unconsumed_tail, 'wrong-sized or incomplete fragment')
	check_crc(raw, raw_crc, 'decoded fragment')
	require(raw == expected, 'fragment differs from legacy summary')
	return raw


def source_levels(file, group):
	data = file.bytes
	at = int(file.directory[group])
	n = struct.unpack_from('<I', data, at)[0]
	if file.header.order == 'tracks':
		at += 8 + n * 40
	else:
		at += 8 + n * 32
		at += 8 + struct.unpack_from('<I', data, at + 4)[0]
	n, size, crc = struct.unpack_from('<III', data, at)
	at += 12
	levels = [(at, n, size, crc, file.groups[group].summary)]
	at += size
	if file.header.optional & 1:
		count = struct.unpack_from('<I', data, at)[0]
		at += 4
		for k in range(count):
			factor, n, size, crc = struct.unpack_from('<IIII', data, at)
			at += 16
			require(factor == 4 ** (k + 1), 'invalid legacy zoom factor')
			levels.append((at, n, size, crc, file.groups[group].zooms[k][1:]))
			at += size
	require(at == (int(file.directory[group + 1]) if group + 1 < len(file.groups)
		else file.header.length), 'legacy group extent mismatch')
	return levels


def read_index(path):
	file = v1.File(path)
	locator = file.metadata.get(KEY)
	if not file.header.optional & 2 or locator is None:
		return None
	require(isinstance(locator, dict), 'invalid locator')
	if locator.get('version') != 2:
		return None
	offset = locator.get('offset')
	require(type(offset) is int and locator.get('length') == 256 and
		128 <= offset <= file.header.chroms - 256, 'root outside data region')
	data = file.bytes
	root = data[offset:offset + 256]
	require(root[:8] == b'FWRSUM1\0', 'invalid root magic')
	version, length, crc, directory, groups, per_page, target = \
		struct.unpack_from('<HHIQIII', root, 8)
	require(version == 2 and length == 256, 'invalid root version/size')
	check_crc(root, crc, 'root', 12)
	require(not any(root[36:64]) and not any(root[192:]), 'nonzero root reserved bytes')
	if root[64:192] != data[:128]:
		return None
	require(directory == offset + 256 and groups == file.header.n_groups and
		per_page == 102 and target == 32768, 'invalid root fields')
	minimum = directory + groups * 32
	require(minimum <= file.header.chroms, 'truncated group directory')
	regions = [(offset, minimum)]

	def inside(at, size):
		require(minimum <= at <= file.header.chroms - size and size > 0,
			'directory section outside data region')
		regions.append((at, at + size))
		return data[at:at + size]

	out = {'offset': offset, 'header': data[:128], 'groups': []}
	for g in range(groups):
		entry = data[directory + g * 32:directory + (g + 1) * 32]
		catalog_at, count, catalog_crc, group, tracks, entry_crc, reserved = \
			struct.unpack('<QIIIIII', entry)
		check_crc(entry, entry_crc, 'group locator', 24)
		require(group == g and tracks == len(file.group_tracks[g]) and
			1 <= count <= 13 and reserved == 0, 'invalid group locator')
		catalog = inside(catalog_at, count * 32)
		check_crc(catalog, catalog_crc, 'level catalog')
		old_levels = source_levels(file, g)
		require(count == len(old_levels), 'level count differs from legacy record')
		group_out = {'group': g, 'tracks': tracks, 'levels': []}
		for k in range(count):
			source, index_at, n, size, source_crc, span = struct.unpack_from(
				'<QQIIII', catalog, k * 32)
			old_at, old_n, old_size, old_crc, summary = old_levels[k]
			require((source, n, size, source_crc) == (old_at, old_n, old_size,
				old_crc), 'source descriptor differs from legacy record')
			level_out = {'level': k, 'factor': 4 ** k, 'source_offset': source,
				'source_rows': n, 'source_bytes': size, 'source_crc': source_crc,
				'window_bins': span, 'index_offset': index_at, 'windows': None}
			group_out['levels'].append(level_out)
			if index_at == 0:
				require(span == 0, 'legacy level has a window size')
				continue
			width = 4 if k == 0 else 8
			require(span > 0 and span <= 2048 and span & (span - 1) == 0 and
				n > 0 and summary is not None and (4 + width * tracks) <= target,
				'invalid indexed level')
			require(span * (4 + width * tracks) <= target, 'window exceeds decoded target')
			total = sum(-(-int(tiles) // (4 ** k)) for tiles in numpy.diff(file.tile_offsets))
			windows = -(-total // span)
			n_pages = -(-windows // 102)
			encoded = inside(index_at, windows * 40 + n_pages * 16)
			ids, sums = summary
			stream = data[source:source + size]
			level_out['windows'] = []
			level_out['window_count'] = windows
			level_out['page_offsets'] = []
			fragments = []
			at = 0
			for first_window in range(0, windows, 102):
				page_count = min(102, windows - first_window)
				page_size = 16 + page_count * 40
				page = encoded[at:at + page_size]
				require(page[:4] == b'FWRP', 'invalid descriptor page magic')
				first, page_n, page_crc = struct.unpack_from('<III', page, 4)
				require((first, page_n) == (first_window, page_count), 'invalid page range')
				check_crc(page, page_crc, 'descriptor page', 12)
				level_out['page_offsets'].append(index_at + at)
				for slot in range(page_count):
					window = first_window + slot
					fields = struct.unpack_from('<10I', page, 16 + slot * 40)
					n_rows, first_row, id_at, id_size, id_crc, id_raw_crc, \
						value_at, value_size, value_crc, value_raw_crc = fields
					lo, hi = numpy.searchsorted(ids, [window * span, min(total,
						(window + 1) * span)])
					require(n_rows == hi - lo, 'window row count differs from legacy')
					if n_rows == 0:
						require(not any(fields), 'nonzero empty-window descriptor')
						continue
					require(first_row == lo and first_row + n_rows <= n,
						'invalid source row rank')
					require(id_at + id_size <= value_at, 'ID and sum fragments overlap')
					raw_ids = fragment(stream, id_at, id_size, id_crc, id_raw_crc,
						ids[lo:hi].astype('<u4').tobytes())
					raw_sums = fragment(stream, value_at, value_size, value_crc,
						value_raw_crc, sums[lo:hi].astype('<f4' if k == 0 else '<f8').tobytes())
					fragments.extend([(id_at, id_at + id_size), (value_at, value_at + value_size)])
					level_out['windows'].append({'window': window, 'n': n_rows,
						'first_source_row': first_row, 'ids_relative_offset': id_at,
						'ids_compressed_bytes': id_size, 'ids_crc': id_crc,
						'ids_raw_crc': id_raw_crc, 'sums_relative_offset': value_at,
						'sums_compressed_bytes': value_size, 'sums_crc': value_crc,
						'sums_raw_crc': value_raw_crc,
						'ids': numpy.frombuffer(raw_ids, dtype='<u4').copy(),
						'sums': numpy.frombuffer(raw_sums, dtype='<f4' if k == 0 else
							'<f8').reshape(n_rows, tracks).copy()})
				at += page_size
			fragments.sort()
			require(all(a[1] <= b[0] for a, b in zip(fragments, fragments[1:])),
				'overlapping fragments')
		out['groups'].append(group_out)
	regions.sort()
	require(all(a[1] <= b[0] for a, b in zip(regions, regions[1:])),
		'overlapping directory sections')
	return out
