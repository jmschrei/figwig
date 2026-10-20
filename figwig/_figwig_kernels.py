# _figwig_kernels.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The compiled half of figwig's own file format: encoding tiles of values
into chunks, inflating chunks, and copying windows out of them.

A figwig file in position order holds every track's values on a fixed grid
of tiles of `tile` bases. The tiles of the tracks of one group, at one
position, are one chunk, which is stored in one of four ways. A chunk whose
every value is 0 is stored as nothing. A dense chunk holds every value,
track after track, in the narrowest element type that holds the chunk's
values: uint8, uint16 or uint32 for integers from 0, int8, int16 or int32
for integers below 0, and float32 for anything else. A sparse chunk holds,
for each track, the number of nonzero bases, then every track's nonzero
bases' offsets within the tile as uint16 differences from the one before,
padded to a multiple of 4 bytes, then every track's nonzero values in that
element type. A matrix chunk, for a group of many tracks few of which have
a value in the tile, holds the number of tracks present, their positions in
the group (uint32, ascending), a uint32 pointer per present track into the
entries plus one at the end, the entries' offsets within the tile (uint16,
ascending within each track), padding to 4 bytes, and the values. The
writer picks the kind that takes the fewest bytes before compression. Each
chunk is padded to a multiple of 8 bytes, so that a chunk that starts on an
8-byte boundary can be read through typed views.

A file in track order holds, for each track and chromosome, blocks of up
to `_BLOCK_ENTRIES` nonzero bases: their positions on the chromosome
(uint32, ascending), padding to 8 bytes, and their values in the block's
element type.

The kind byte of a chunk or block, kept in the file's index, is the kind
(0 zero, 1 dense, 2 sparse, 3 matrix; 1 for a block) in its low two bits
and the element code in the next three.

Every kernel here is compiled by numba with `nogil=True`, so that several
threads can encode, inflate and decode at once.
"""

import numba
import numpy


_ZERO, _DENSE, _SPARSE, _MATRIX = 0, 1, 2, 3
_U8, _U16, _U32, _I8, _I16, _I32, _F32 = 0, 1, 2, 3, 4, 5, 6
_WIDTHS = numpy.array([1, 2, 4, 1, 2, 4, 4], dtype=numpy.int64)
_BLOCK_ENTRIES = 1 << 16


@numba.njit(nogil=True, cache=True, inline='always')
def _pad8(n):
	return (n + 7) // 8 * 8


@numba.njit(nogil=True, cache=True, inline='always')
def _pad4(n):
	return (n + 3) // 4 * 4


@numba.njit(nogil=True, cache=True)
def _store(out, offset, code, value):
	"""Write one value of element type `code` at byte `offset` of `out`."""

	if code == _U8:
		out[offset] = numpy.uint8(value)
	elif code == _U16:
		out[offset:offset + 2].view(numpy.uint16)[0] = numpy.uint16(value)
	elif code == _U32:
		out[offset:offset + 4].view(numpy.uint32)[0] = numpy.uint32(value)
	elif code == _I8:
		out[offset:offset + 1].view(numpy.int8)[0] = numpy.int8(value)
	elif code == _I16:
		out[offset:offset + 2].view(numpy.int16)[0] = numpy.int16(value)
	elif code == _I32:
		out[offset:offset + 4].view(numpy.int32)[0] = numpy.int32(value)
	else:
		out[offset:offset + 4].view(numpy.float32)[0] = numpy.float32(value)


@numba.njit(nogil=True, cache=True)
def _load(data, offset, code):
	"""One value of element type `code` at byte `offset` of `data`, as a
	float64."""

	if code == _U8:
		return numpy.float64(data[offset])
	elif code == _U16:
		return numpy.float64(data[offset:offset + 2].view(numpy.uint16)[0])
	elif code == _U32:
		return numpy.float64(data[offset:offset + 4].view(numpy.uint32)[0])
	elif code == _I8:
		return numpy.float64(data[offset:offset + 1].view(numpy.int8)[0])
	elif code == _I16:
		return numpy.float64(data[offset:offset + 2].view(numpy.int16)[0])
	elif code == _I32:
		return numpy.float64(data[offset:offset + 4].view(numpy.int32)[0])
	else:
		return numpy.float64(data[offset:offset + 4].view(numpy.float32)[0])


@numba.njit(nogil=True, cache=True)
def _note(v, g, nnz, mins, maxs, sums, nonint):
	"""Fold one nonzero value of track g into the tile's statistics."""

	nnz[g] += 1
	sums[g] += v
	if v < mins[g]:
		mins[g] = v
	if v > maxs[g]:
		maxs[g] = v
	if v != numpy.floor(v) or v < -2147483648.0 or v >= 4294967296.0:
		nonint[g] = True


@numba.njit(nogil=True, cache=True)
def _code_of(integral, low, high):
	"""The element code for values from `low` to `high`."""

	if not integral:
		return _F32
	elif low >= 0.0:
		return _U8 if high <= 255.0 else _U16 if high <= 65535.0 else _U32
	elif low >= -128.0 and high <= 127.0:
		return _I8
	elif low >= -32768.0 and high <= 32767.0:
		return _I16
	elif high <= 2147483647.0:
		return _I32
	return _F32


@numba.njit(nogil=True, cache=True)
def _pick_kind(n_tracks, n_present, total, w, width, allow_dense, allow_matrix):
	"""The kind of chunk that takes the fewest bytes, and those bytes."""

	dense_bytes = n_tracks * width * w
	sparse_bytes = 4 * n_tracks + _pad4(2 * total) + total * w
	matrix_bytes = 8 + 8 * n_present + _pad4(2 * total) + total * w
	kind, best = _SPARSE, sparse_bytes
	if allow_matrix and matrix_bytes < best:
		kind, best = _MATRIX, matrix_bytes
	if allow_dense and dense_bytes <= best:
		kind, best = _DENSE, dense_bytes
	return kind, best


@numba.njit(nogil=True, cache=True)
def _fold_stats(stats, nnz, mins, maxs, sums, nonint):
	for g in range(stats.shape[0]):
		stats[g, 0] += sums[g]
		stats[g, 1] += nnz[g]
		if mins[g] < stats[g, 2]:
			stats[g, 2] = mins[g]
		if maxs[g] > stats[g, 3]:
			stats[g, 3] = maxs[g]
		if nonint[g]:
			stats[g, 4] = 1.0


@numba.njit(nogil=True, cache=True)
def _dense_payload(out, position, block, c0, width, code):
	"""Write block[:, c0:c0 + width] densely at `position` of `out`, and
	return where it ends."""

	n_tracks = block.shape[0]
	end = position + n_tracks * width * _WIDTHS[code]
	if code == _U8:
		for g in range(n_tracks):
			for p in range(width):
				out[position + g * width + p] = numpy.uint8(block[g, c0 + p])
	elif code == _U16:
		view = out[position:end].view(numpy.uint16)
		for g in range(n_tracks):
			for p in range(width):
				view[g * width + p] = numpy.uint16(block[g, c0 + p])
	elif code == _U32:
		view = out[position:end].view(numpy.uint32)
		for g in range(n_tracks):
			for p in range(width):
				view[g * width + p] = numpy.uint32(block[g, c0 + p])
	elif code == _I8:
		view = out[position:end].view(numpy.int8)
		for g in range(n_tracks):
			for p in range(width):
				view[g * width + p] = numpy.int8(block[g, c0 + p])
	elif code == _I16:
		view = out[position:end].view(numpy.int16)
		for g in range(n_tracks):
			for p in range(width):
				view[g * width + p] = numpy.int16(block[g, c0 + p])
	elif code == _I32:
		view = out[position:end].view(numpy.int32)
		for g in range(n_tracks):
			for p in range(width):
				view[g * width + p] = numpy.int32(block[g, c0 + p])
	else:
		view = out[position:end].view(numpy.float32)
		for g in range(n_tracks):
			for p in range(width):
				view[g * width + p] = numpy.float32(block[g, c0 + p])
	return end


@numba.njit(nogil=True, cache=True)
def _sparse_header(out, position, kind, n_tracks, nnz):
	"""Write a sparse or matrix chunk's header. Returns where the offsets
	start, where the values start, and the entries in all."""

	total = 0
	for g in range(n_tracks):
		total += nnz[g]

	if kind == _SPARSE:
		counts = out[position:position + 4 * n_tracks].view(numpy.uint32)
		for g in range(n_tracks):
			counts[g] = nnz[g]
		offsets_at = position + 4 * n_tracks
	else:
		n_present = 0
		for g in range(n_tracks):
			if nnz[g] > 0:
				n_present += 1
		out[position:position + 4].view(numpy.uint32)[0] = n_present
		tracks = out[position + 4:position + 4 + 4 * n_present].view(numpy.uint32)
		ptr = out[position + 4 + 4 * n_present:position + 8 + 8 * n_present].view(
			numpy.uint32)
		k, running = 0, 0
		for g in range(n_tracks):
			if nnz[g] > 0:
				tracks[k] = g
				ptr[k] = running
				running += nnz[g]
				k += 1
		ptr[n_present] = running
		offsets_at = position + 8 + 8 * n_present

	values_at = offsets_at + _pad4(2 * total)
	for p in range(offsets_at + 2 * total, values_at):
		out[p] = 0
	return offsets_at, values_at, total


@numba.njit(nogil=True, cache=True)
def _finish_chunk(out, end):
	"""Zero the padding after a chunk; returns where the next one starts."""

	for p in range(end, _pad8(end)):
		out[p] = 0
	return _pad8(end)


@numba.njit(nogil=True, cache=True)
def _encode_tiles(tile, tiles, lo, hi, starts, ends, values, by_base, buffer,
	preset, keep_last, allow_matrix, out, bounds, kinds, stats):
	"""Encode the tiles of one group on one chromosome into chunks.

	Tile t of this call is tile tiles[t] of the chromosome, the bases
	[tiles[t] * tile, (tiles[t] + 1) * tile). The items lo[t] to hi[t] - 1
	overlap it: item i covers [starts[i], ends[i]) and gives base p of track
	g the value values[i, g, p - starts[i]], or values[i, g, 0] for every
	base when `by_base` is false. Items are sorted and do not overlap, and
	a tile's items are the only ones that touch it.

	`buffer`, (tracks, tile) float64, is zero on entry except when `preset`,
	in which case it already holds values of tile 0 written by earlier calls,
	and is zero again on return except when `keep_last`, in which case the
	last tile is filled and left there, unencoded, for a later call to add
	to. Every other tile is encoded: its chunk's bytes are written to
	out[bounds[t]:bounds[t + 1]], padded to 8 bytes, with bounds[0] given,
	and kinds[t] is its kind byte. A tile whose values are all zero takes no
	bytes. stats[g] accumulates the sum, nonzero bases, minimum and maximum
	of the encoded values of track g, and stats[g, 4] becomes 1 once a value
	of track g is not an integer, or is an integer outside what int32 and
	uint32 hold. With `allow_matrix`, a tile may be stored as a matrix
	chunk.

	Returns the number of tiles encoded, which is len(tiles) or one fewer.
	"""

	n_tracks, width = buffer.shape
	n_tiles = tiles.shape[0]
	nnz = numpy.zeros(n_tracks, dtype=numpy.int64)
	mins = numpy.zeros(n_tracks, dtype=numpy.float64)
	maxs = numpy.zeros(n_tracks, dtype=numpy.float64)
	sums = numpy.zeros(n_tracks, dtype=numpy.float64)
	nonint = numpy.zeros(n_tracks, dtype=numpy.bool_)
	position = bounds[0]

	for t in range(n_tiles):
		base = tiles[t] * tile
		scan = t == 0 and preset

		for i in range(lo[t], hi[t]):
			a = max(starts[i], base)
			b = min(ends[i], base + width)
			for g in range(n_tracks):
				if by_base:
					for p in range(a, b):
						buffer[g, p - base] = values[i, g, p - starts[i]]
				else:
					v = values[i, g, 0]
					for p in range(a, b):
						buffer[g, p - base] = v

		if t == n_tiles - 1 and keep_last:
			break

		# The nonzero bases, their range and whether they are all integers
		# that fit 32 bits, from the whole buffer when earlier calls filled
		# some of it, and from this call's items otherwise.
		for g in range(n_tracks):
			nnz[g] = 0
			mins[g] = 0.0
			maxs[g] = 0.0
			sums[g] = 0.0
			nonint[g] = False

		integral = True
		total = 0
		n_present = 0
		for g in range(n_tracks):
			if scan:
				for p in range(width):
					v = buffer[g, p]
					if v != 0.0:
						_note(v, g, nnz, mins, maxs, sums, nonint)
			else:
				for i in range(lo[t], hi[t]):
					a = max(starts[i], base)
					b = min(ends[i], base + width)
					for p in range(a, b):
						v = buffer[g, p - base]
						if v != 0.0:
							_note(v, g, nnz, mins, maxs, sums, nonint)

			if nonint[g]:
				integral = False
			if nnz[g] > 0:
				n_present += 1
			total += nnz[g]

		if total == 0:
			kinds[t] = _ZERO
			bounds[t + 1] = position
			continue

		code = _code_of(integral, mins.min(), maxs.max())
		w = _WIDTHS[code]
		kind, n_bytes = _pick_kind(n_tracks, n_present, total, w, width, True,
			allow_matrix)
		kinds[t] = kind | (code << 2)

		if kind == _DENSE:
			end = _dense_payload(out, position, buffer, 0, width, code)
		else:
			offsets_at, values_at, _ = _sparse_header(out, position, kind,
				n_tracks, nnz)
			end = values_at + total * w
			offsets = out[offsets_at:offsets_at + 2 * total].view(numpy.uint16)

			# Each track's nonzero bases in order, as the kernel finds them
			# by walking the buffer or the sorted items.
			d = 0
			o = values_at
			for g in range(n_tracks):
				last = 0
				if scan:
					for p in range(width):
						v = buffer[g, p]
						if v != 0.0:
							offsets[d] = numpy.uint16(p - last if kind == _SPARSE
								else p)
							_store(out, o, code, v)
							last = p
							d += 1
							o += w
				else:
					for i in range(lo[t], hi[t]):
						a = max(starts[i], base)
						b = min(ends[i], base + width)
						for p in range(a - base, b - base):
							v = buffer[g, p]
							if v != 0.0:
								offsets[d] = numpy.uint16(p - last if kind == _SPARSE
									else p)
								_store(out, o, code, v)
								last = p
								d += 1
								o += w

		position = _finish_chunk(out, end)
		bounds[t + 1] = position
		_fold_stats(stats, nnz, mins, maxs, sums, nonint)

		# Back to zero for the next tile: the whole buffer when it was
		# scanned, and otherwise only the bases this tile's items filled.
		if scan:
			buffer[:, :] = 0.0
		else:
			for i in range(lo[t], hi[t]):
				a = max(starts[i], base)
				b = min(ends[i], base + width)
				for g in range(n_tracks):
					for p in range(a, b):
						buffer[g, p - base] = 0.0

	return n_tiles - 1 if keep_last and n_tiles > 0 else n_tiles


@numba.njit(nogil=True, cache=True)
def _encode_dense_tiles(tile, values, items, columns, allow_matrix, out, bounds,
	kinds, stats):
	"""Encode whole tiles that windows cover, straight from their values.

	Tile t is values[items[t], :, columns[t]:columns[t] + tile], where
	`values` holds the windows' values, (windows, tracks, width), so no
	scratch buffer is filled or cleared. The chunks, `bounds`, `kinds` and
	`stats` are as in `_encode_tiles`, and every tile is encoded.
	"""

	n_tracks = values.shape[1]
	n_tiles = items.shape[0]
	nnz = numpy.zeros(n_tracks, dtype=numpy.int64)
	mins = numpy.zeros(n_tracks, dtype=numpy.float64)
	maxs = numpy.zeros(n_tracks, dtype=numpy.float64)
	sums = numpy.zeros(n_tracks, dtype=numpy.float64)
	nonint = numpy.zeros(n_tracks, dtype=numpy.bool_)
	position = bounds[0]

	for t in range(n_tiles):
		block = values[items[t]]
		c0 = columns[t]
		integral = True
		total = 0
		n_present = 0
		for g in range(n_tracks):
			nnz[g] = 0
			mins[g] = 0.0
			maxs[g] = 0.0
			sums[g] = 0.0
			nonint[g] = False
			for p in range(c0, c0 + tile):
				v = numpy.float64(block[g, p])
				if v != 0.0:
					_note(v, g, nnz, mins, maxs, sums, nonint)
			if nonint[g]:
				integral = False
			if nnz[g] > 0:
				n_present += 1
			total += nnz[g]

		if total == 0:
			kinds[t] = _ZERO
			bounds[t + 1] = position
			continue

		code = _code_of(integral, mins.min(), maxs.max())
		w = _WIDTHS[code]
		kind, n_bytes = _pick_kind(n_tracks, n_present, total, w, tile, True,
			allow_matrix)
		kinds[t] = kind | (code << 2)

		if kind == _DENSE:
			end = _dense_payload(out, position, block, c0, tile, code)
		else:
			offsets_at, values_at, _ = _sparse_header(out, position, kind,
				n_tracks, nnz)
			end = values_at + total * w
			offsets = out[offsets_at:offsets_at + 2 * total].view(numpy.uint16)
			d = 0
			o = values_at
			for g in range(n_tracks):
				last = 0
				for p in range(tile):
					v = numpy.float64(block[g, c0 + p])
					if v != 0.0:
						offsets[d] = numpy.uint16(p - last if kind == _SPARSE else p)
						_store(out, o, code, v)
						last = p
						d += 1
						o += w

		position = _finish_chunk(out, end)
		bounds[t + 1] = position
		_fold_stats(stats, nnz, mins, maxs, sums, nonint)

	return n_tiles


@numba.njit(nogil=True, cache=True)
def _encode_entries(tile, n_tracks, tiles, first, tracks, positions, values,
	allow_matrix, out, bounds, kinds, stats):
	"""Encode tiles from their nonzero bases, given as entries.

	The entries of tile t are first[t] to first[t + 1] - 1, sorted by track
	then position, with no two on one (track, position): entry e is base
	positions[e] of the chromosome in track tracks[e], a position in the
	group, with the value values[e]. Tiles are stored sparsely or as a
	matrix, never densely. The chunks, `bounds`, `kinds` and `stats` are as
	in `_encode_tiles`.
	"""

	nnz = numpy.zeros(n_tracks, dtype=numpy.int64)
	mins = numpy.zeros(n_tracks, dtype=numpy.float64)
	maxs = numpy.zeros(n_tracks, dtype=numpy.float64)
	sums = numpy.zeros(n_tracks, dtype=numpy.float64)
	nonint = numpy.zeros(n_tracks, dtype=numpy.bool_)
	position = bounds[0]

	for t in range(tiles.shape[0]):
		base = tiles[t] * tile
		e0, e1 = first[t], first[t + 1]
		for g in range(n_tracks):
			nnz[g] = 0
			mins[g] = 0.0
			maxs[g] = 0.0
			sums[g] = 0.0
			nonint[g] = False

		total = 0
		for e in range(e0, e1):
			v = values[e]
			if v != 0.0:
				_note(v, tracks[e], nnz, mins, maxs, sums, nonint)
				total += 1

		if total == 0:
			kinds[t] = _ZERO
			bounds[t + 1] = position
			continue

		integral = True
		n_present = 0
		for g in range(n_tracks):
			if nonint[g]:
				integral = False
			if nnz[g] > 0:
				n_present += 1

		code = _code_of(integral, mins.min(), maxs.max())
		w = _WIDTHS[code]
		kind, n_bytes = _pick_kind(n_tracks, n_present, total, w, tile, False,
			allow_matrix)
		kinds[t] = kind | (code << 2)
		offsets_at, values_at, _ = _sparse_header(out, position, kind, n_tracks,
			nnz)
		end = values_at + total * w
		offsets = out[offsets_at:offsets_at + 2 * total].view(numpy.uint16)

		d = 0
		o = values_at
		last_track = -1
		last = 0
		for e in range(e0, e1):
			v = values[e]
			if v == 0.0:
				continue
			if tracks[e] != last_track:
				last_track = tracks[e]
				last = 0
			p = positions[e] - base
			offsets[d] = numpy.uint16(p - last if kind == _SPARSE else p)
			_store(out, o, code, v)
			last = p
			d += 1
			o += w

		position = _finish_chunk(out, end)
		bounds[t + 1] = position
		_fold_stats(stats, nnz, mins, maxs, sums, nonint)

	return tiles.shape[0]


@numba.njit(nogil=True, cache=True)
def _encode_blocks(positions, values, first, out, bounds, kinds, stats):
	"""Encode blocks of one track's nonzero bases on one chromosome, for a
	file in track order.

	Block t holds entries first[t] to first[t + 1] - 1: positions[e] on the
	chromosome, ascending, with values[e], none of them 0. Its bytes are
	out[bounds[t]:bounds[t + 1]]: the positions as uint32, padding to 8
	bytes, the values in the block's element type, padding to 8 bytes.
	`stats` is the (1, 5) statistics row of the track.
	"""

	nnz = numpy.zeros(1, dtype=numpy.int64)
	mins = numpy.zeros(1, dtype=numpy.float64)
	maxs = numpy.zeros(1, dtype=numpy.float64)
	sums = numpy.zeros(1, dtype=numpy.float64)
	nonint = numpy.zeros(1, dtype=numpy.bool_)
	position = bounds[0]
	for t in range(first.shape[0] - 1):
		e0, e1 = first[t], first[t + 1]
		n = e1 - e0
		nnz[0] = 0
		mins[0] = 0.0
		maxs[0] = 0.0
		sums[0] = 0.0
		nonint[0] = False
		for e in range(e0, e1):
			_note(values[e], 0, nnz, mins, maxs, sums, nonint)

		code = _code_of(not nonint[0], mins[0], maxs[0])
		w = _WIDTHS[code]
		kinds[t] = _DENSE | (code << 2)
		view = out[position:position + 4 * n].view(numpy.uint32)
		for e in range(e0, e1):
			view[e - e0] = numpy.uint32(positions[e])
		values_at = _pad8(position + 4 * n)
		for p in range(position + 4 * n, values_at):
			out[p] = 0
		o = values_at
		for e in range(e0, e1):
			_store(out, o, code, values[e])
			o += w
		position = _finish_chunk(out, o)
		bounds[t + 1] = position
		_fold_stats(stats, nnz, mins, maxs, sums, nonint)

	return first.shape[0] - 1


@numba.njit(nogil=True, cache=True)
def _inflate_chunks(uncompress, data, starts, sizes, buffer, positions, lengths,
	length):
	"""Inflate chunks one after another into `buffer`, each starting on an
	8-byte boundary, without the GIL.

	Chunk k is the zlib stream data[starts[k]:starts[k] + sizes[k]]. On
	return positions[k] is where its bytes start in `buffer` and lengths[k]
	how many there are, or positions[k] is -1 if uncompress() could not
	inflate it into the rest of `buffer`, in which case zlib.decompress gets
	it. `length` is a one-element array of C unsigned longs, which are 32
	bits on Windows. Returns the number of bytes of `buffer` used.
	"""

	position = 0
	for k in range(starts.shape[0]):
		length[0] = min(buffer.shape[0] - position, 2 ** 32 - 1)
		source = data[starts[k]:starts[k] + sizes[k]]
		status = uncompress(buffer[position:].ctypes, length.ctypes,
			source.ctypes, sizes[k])

		if status != 0:
			positions[k] = -1
			lengths[k] = 0
		else:
			positions[k] = position
			lengths[k] = numpy.int64(length[0])
			position = _pad8(position + lengths[k])

	return position


@numba.njit(nogil=True, cache=True)
def _matrix_header(data, position):
	"""A matrix chunk's (n_present, tracks, ptr, offsets_at, values_at,
	total)."""

	n_present = numpy.int64(data[position:position + 4].view(numpy.uint32)[0])
	tracks = data[position + 4:position + 4 + 4 * n_present].view(numpy.uint32)
	ptr = data[position + 4 + 4 * n_present:position + 8 + 8 * n_present].view(
		numpy.uint32)
	total = numpy.int64(ptr[n_present])
	offsets_at = position + 8 + 8 * n_present
	values_at = offsets_at + _pad4(2 * total)
	return n_present, tracks, ptr, offsets_at, values_at, total


@numba.njit(nogil=True, cache=True)
def _check_chunks(tile, data, positions, lengths, kinds, n_tracks, failed):
	"""Whether each chunk of a batch can be decoded by `_read_tiles`.

	Chunk k's bytes are data[positions[k]:positions[k] + lengths[k]], and
	kinds[k] its kind byte. A chunk fails, and gets failed[k] = 1, when its
	kind or element code is not one the format defines, when it is shorter
	than its kind says, or when its offsets or track positions are out of
	range or order. Returns the number of chunks that failed.
	"""

	n_failed = 0
	for k in range(positions.shape[0]):
		kind = kinds[k] & 3
		code = kinds[k] >> 2
		position = positions[k]
		length = lengths[k]
		bad = False

		if position < 0 or code > _F32 or kind == _ZERO:
			bad = True
		elif kind == _DENSE:
			bad = length < n_tracks * tile * _WIDTHS[code]
		elif kind == _SPARSE:
			if length < 4 * n_tracks:
				bad = True
			else:
				counts = data[position:position + 4 * n_tracks].view(numpy.uint32)
				total = 0
				for g in range(n_tracks):
					total += numpy.int64(counts[g])

				deltas_at = position + 4 * n_tracks
				values_at = deltas_at + _pad4(2 * total)
				if length < values_at - position + total * _WIDTHS[code]:
					bad = True
				else:
					deltas = data[deltas_at:deltas_at + 2 * total].view(numpy.uint16)
					d = 0
					for g in range(n_tracks):
						p = 0
						for i in range(numpy.int64(counts[g])):
							p += numpy.int64(deltas[d + i])
							if p >= tile:
								bad = True
						d += numpy.int64(counts[g])
		else:
			if length < 4:
				bad = True
			else:
				n_present = numpy.int64(data[position:position + 4].view(
					numpy.uint32)[0])
				if n_present > n_tracks or length < 8 + 8 * n_present:
					bad = True
				else:
					_, tracks, ptr, offsets_at, values_at, total = _matrix_header(
						data, position)
					if length < values_at - position + total * _WIDTHS[code]:
						bad = True
					else:
						offsets = data[offsets_at:offsets_at + 2 * total].view(
							numpy.uint16)
						for j in range(n_present):
							if tracks[j] >= n_tracks or (j > 0 and tracks[j] <=
									tracks[j - 1]) or ptr[j + 1] < ptr[j]:
								bad = True
								break
							for i in range(numpy.int64(ptr[j]), numpy.int64(ptr[j +
									1])):
								if offsets[i] >= tile or (i > numpy.int64(ptr[j]) and
										offsets[i] <= offsets[i - 1]):
									bad = True

		if bad:
			failed[k] = 1
			n_failed += 1

	return n_failed


@numba.njit(nogil=True, cache=True)
def _check_blocks(data, positions, lengths, kinds, n_entries, failed):
	"""Whether each block of a track-order batch can be decoded: its kind
	is 1, its code known, it is long enough, and its positions ascend."""

	n_failed = 0
	for k in range(positions.shape[0]):
		kind = kinds[k] & 3
		code = kinds[k] >> 2
		n = n_entries[k]
		position = positions[k]
		bad = position < 0 or kind != _DENSE or code > _F32 or lengths[k] < \
			_pad8(4 * n) + n * _WIDTHS[code]
		if not bad:
			view = data[position:position + 4 * n].view(numpy.uint32)
			for i in range(1, n):
				if view[i] <= view[i - 1]:
					bad = True
					break
		if bad:
			failed[k] = 1
			n_failed += 1

	return n_failed


@numba.njit(nogil=True, cache=True)
def _read_tiles(tile, data, positions, kinds, n_tracks, slots, pair_chunks,
	pair_bases, pair_starts, pair_rows, out, accumulate):
	"""Copy the parts of windows that fall in a batch of chunks into `out`.

	Pair q is a window and one tile it overlaps: the tile starts at base
	pair_bases[q] of its chromosome, the window at pair_starts[q] and is
	out.shape[2] bases wide, and the window's values go to row pair_rows[q]
	of `out`, of shape (windows, channels, width). The tile's chunk is chunk
	pair_chunks[q] of the batch, whose bytes start at positions[chunk] of
	`data` and whose kind byte is kinds[chunk], or -1 for a tile whose
	values are all zero. The chunk holds `n_tracks` tracks, and track g of
	it goes to channel slots[g], or nowhere when slots[g] is -1. Every chunk
	has passed `_check_chunks`.

	Without `accumulate`, the window's bases are set, zeros included, and
	every value is cast to the type of `out`. With it, values are added to
	`out`, which is zero on entry, so that several tracks with one channel
	in `slots` are summed. A window's bases past its chromosome's end are
	not in any tile, and are set by the caller.
	"""

	width = out.shape[2]
	for q in range(pair_chunks.shape[0]):
		base = pair_bases[q]
		start = pair_starts[q]
		row = pair_rows[q]
		a = max(start, base)
		b = min(start + width, base + tile)
		if b <= a:
			continue

		chunk = pair_chunks[q]
		if chunk < 0:
			if not accumulate:
				for g in range(n_tracks):
					slot = slots[g]
					if slot >= 0:
						for p in range(a - start, b - start):
							out[row, slot, p] = 0
			continue

		kind = kinds[chunk] & 3
		code = kinds[chunk] >> 2
		w = _WIDTHS[code]
		position = positions[chunk]

		if kind == _DENSE:
			for g in range(n_tracks):
				slot = slots[g]
				if slot < 0:
					continue

				at = position + (g * tile + (a - base)) * w
				n = b - a
				if accumulate:
					for p in range(n):
						out[row, slot, a - start + p] += _load(data, at + p * w, code)
				elif code == _U8:
					for p in range(n):
						out[row, slot, a - start + p] = data[at + p]
				elif code == _U16:
					view = data[at:at + 2 * n].view(numpy.uint16)
					for p in range(n):
						out[row, slot, a - start + p] = view[p]
				elif code == _U32:
					view = data[at:at + 4 * n].view(numpy.uint32)
					for p in range(n):
						out[row, slot, a - start + p] = view[p]
				elif code == _I8:
					view = data[at:at + n].view(numpy.int8)
					for p in range(n):
						out[row, slot, a - start + p] = view[p]
				elif code == _I16:
					view = data[at:at + 2 * n].view(numpy.int16)
					for p in range(n):
						out[row, slot, a - start + p] = view[p]
				elif code == _I32:
					view = data[at:at + 4 * n].view(numpy.int32)
					for p in range(n):
						out[row, slot, a - start + p] = view[p]
				else:
					view = data[at:at + 4 * n].view(numpy.float32)
					for p in range(n):
						out[row, slot, a - start + p] = view[p]
		elif kind == _SPARSE:
			counts = data[position:position + 4 * n_tracks].view(numpy.uint32)
			total = 0
			for g in range(n_tracks):
				total += numpy.int64(counts[g])

			deltas_at = position + 4 * n_tracks
			deltas = data[deltas_at:deltas_at + 2 * total].view(numpy.uint16)
			values_at = deltas_at + _pad4(2 * total)

			d = 0
			for g in range(n_tracks):
				slot = slots[g]
				count = numpy.int64(counts[g])
				if slot < 0:
					d += count
					continue

				if not accumulate:
					for p in range(a - start, b - start):
						out[row, slot, p] = 0

				p = 0
				for i in range(count):
					p += numpy.int64(deltas[d + i])
					at = base + p
					if at >= b:
						break
					if at >= a:
						v = _load(data, values_at + (d + i) * w, code)
						if accumulate:
							out[row, slot, at - start] += v
						else:
							out[row, slot, at - start] = v

				d += count
		else:
			n_present, tracks, ptr, offsets_at, values_at, total = _matrix_header(
				data, position)
			offsets = data[offsets_at:offsets_at + 2 * total].view(numpy.uint16)
			if not accumulate:
				for g in range(n_tracks):
					slot = slots[g]
					if slot >= 0:
						for p in range(a - start, b - start):
							out[row, slot, p] = 0

			for k in range(n_present):
				slot = slots[tracks[k]]
				if slot < 0:
					continue
				# The first offset in the window, by binary search.
				lo, hi = numpy.int64(ptr[k]), numpy.int64(ptr[k + 1])
				while lo < hi:
					mid = (lo + hi) // 2
					if base + numpy.int64(offsets[mid]) < a:
						lo = mid + 1
					else:
						hi = mid
				for i in range(lo, numpy.int64(ptr[k + 1])):
					at = base + numpy.int64(offsets[i])
					if at >= b:
						break
					v = _load(data, values_at + i * w, code)
					if accumulate:
						out[row, slot, at - start] += v
					else:
						out[row, slot, at - start] = v

	return 0


@numba.njit(nogil=True, cache=True)
def _list_tiles(tile, data, positions, kinds, n_tracks, slots, pair_chunks,
	pair_bases, pair_starts, pair_rows, width, fill, out_rows, out_channels,
	out_offsets, out_values):
	"""The nonzero bases of windows in a batch of chunks, as entries.

	The pairs are as in `_read_tiles`. Each nonzero base of a selected
	track in a window gives one entry: the window's row, the track's
	channel, the base's offset in the window and its value. Without `fill`
	only the count is returned; with it the entries are written to the
	output arrays, which must hold them. Returns the number of entries.
	"""

	k_out = 0
	for q in range(pair_chunks.shape[0]):
		base = pair_bases[q]
		start = pair_starts[q]
		row = pair_rows[q]
		a = max(start, base)
		b = min(start + width, base + tile)
		chunk = pair_chunks[q]
		if b <= a or chunk < 0:
			continue

		kind = kinds[chunk] & 3
		code = kinds[chunk] >> 2
		w = _WIDTHS[code]
		position = positions[chunk]

		if kind == _DENSE:
			for g in range(n_tracks):
				slot = slots[g]
				if slot < 0:
					continue
				at = position + (g * tile + (a - base)) * w
				for p in range(b - a):
					v = _load(data, at + p * w, code)
					if v != 0.0:
						if fill:
							out_rows[k_out] = row
							out_channels[k_out] = slot
							out_offsets[k_out] = a - start + p
							out_values[k_out] = v
						k_out += 1
		elif kind == _SPARSE:
			counts = data[position:position + 4 * n_tracks].view(numpy.uint32)
			total = 0
			for g in range(n_tracks):
				total += numpy.int64(counts[g])
			deltas_at = position + 4 * n_tracks
			deltas = data[deltas_at:deltas_at + 2 * total].view(numpy.uint16)
			values_at = deltas_at + _pad4(2 * total)
			d = 0
			for g in range(n_tracks):
				slot = slots[g]
				count = numpy.int64(counts[g])
				if slot < 0:
					d += count
					continue
				p = 0
				for i in range(count):
					p += numpy.int64(deltas[d + i])
					at = base + p
					if at >= b:
						break
					if at >= a:
						if fill:
							out_rows[k_out] = row
							out_channels[k_out] = slot
							out_offsets[k_out] = at - start
							out_values[k_out] = _load(data, values_at + (d + i) * w,
								code)
						k_out += 1
				d += count
		else:
			n_present, tracks, ptr, offsets_at, values_at, total = _matrix_header(
				data, position)
			offsets = data[offsets_at:offsets_at + 2 * total].view(numpy.uint16)
			for k in range(n_present):
				slot = slots[tracks[k]]
				if slot < 0:
					continue
				lo, hi = numpy.int64(ptr[k]), numpy.int64(ptr[k + 1])
				while lo < hi:
					mid = (lo + hi) // 2
					if base + numpy.int64(offsets[mid]) < a:
						lo = mid + 1
					else:
						hi = mid
				for i in range(lo, numpy.int64(ptr[k + 1])):
					at = base + numpy.int64(offsets[i])
					if at >= b:
						break
					if fill:
						out_rows[k_out] = row
						out_channels[k_out] = slot
						out_offsets[k_out] = at - start
						out_values[k_out] = _load(data, values_at + i * w, code)
					k_out += 1

	return k_out


@numba.njit(nogil=True, cache=True)
def _read_blocks(data, positions, kinds, n_entries, pair_blocks, pair_slots,
	pair_starts, pair_rows, width, out, accumulate, list_entries, fill,
	out_rows, out_channels, out_offsets, out_values):
	"""Windows out of a batch of track-order blocks.

	Pair q is a window and one block of one track that may hold its bases:
	block pair_blocks[q] of the batch, whose bytes start at
	positions[block] of `data` with n_entries[block] entries, for the window
	at pair_starts[q] of `width` bases, going to row pair_rows[q] and channel
	pair_slots[q] of `out`. The block's positions ascend, so the window's
	entries are found by binary search. Without `list_entries` the values
	are set into `out`, or added with `accumulate`; with it they are counted
	as entries, and written to the output arrays when `fill`. `out` is zero
	on entry in every mode. Returns the number of entries.
	"""

	k_out = 0
	for q in range(pair_blocks.shape[0]):
		block = pair_blocks[q]
		n = n_entries[block]
		code = kinds[block] >> 2
		w = _WIDTHS[code]
		position = positions[block]
		start = pair_starts[q]
		end = start + width
		row, slot = pair_rows[q], pair_slots[q]
		view = data[position:position + 4 * n].view(numpy.uint32)
		values_at = _pad8(position + 4 * n)
		lo, hi = 0, n
		while lo < hi:
			mid = (lo + hi) // 2
			if numpy.int64(view[mid]) < start:
				lo = mid + 1
			else:
				hi = mid
		for i in range(lo, n):
			at = numpy.int64(view[i])
			if at >= end:
				break
			v = _load(data, values_at + i * w, code)
			if list_entries:
				if fill:
					out_rows[k_out] = row
					out_channels[k_out] = slot
					out_offsets[k_out] = at - start
					out_values[k_out] = v
				k_out += 1
			elif accumulate:
				out[row, slot, at - start] += v
			else:
				out[row, slot, at - start] = v

	return k_out
