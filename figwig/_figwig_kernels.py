# _figwig_kernels.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The compiled half of figwig's own file format: encoding tiles of values
into chunks, inflating chunks, and copying windows out of them.

A figwig file holds every track's values on a fixed grid of tiles of `tile`
bases. The tiles of the tracks of one group, at one position, are one
chunk, which is stored in one of three ways. A chunk whose every value is 0
is stored as nothing. A dense chunk holds every value, track after track,
in the narrowest element type that holds the chunk's values: uint8, uint16
or uint32 for integers from 0, int8, int16 or int32 for integers below 0,
and float32 for anything else. A sparse chunk holds, for each track, the
number of nonzero bases, then every track's nonzero bases' offsets within
the tile as uint16 differences from the one before, padded to a multiple
of 4 bytes, then every track's nonzero values in that element type. The
writer picks dense or sparse by the bytes each takes before compression.
Each chunk is padded to a multiple of 8 bytes, so that a chunk that starts
on an 8-byte boundary can be read through typed views.

The kind byte of a chunk, kept in the file's index, is the kind (0 zero, 1
dense, 2 sparse) in its low two bits and the element code in the next three.

Every kernel here is compiled by numba with `nogil=True`, so that several
threads can encode, inflate and decode at once.
"""

import numba
import numpy


_ZERO, _DENSE, _SPARSE = 0, 1, 2
_U8, _U16, _U32, _I8, _I16, _I32, _F32 = 0, 1, 2, 3, 4, 5, 6
_WIDTHS = numpy.array([1, 2, 4, 1, 2, 4, 4], dtype=numpy.int64)


@numba.njit(nogil=True, cache=True, inline='always')
def _pad8(n):
	return (n + 7) // 8 * 8


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
def _encode_tiles(tile, tiles, lo, hi, starts, ends, values, by_base, buffer,
	preset, keep_last, out, bounds, kinds, stats):
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
	uint32 hold.

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
		integral = True
		total = 0
		for g in range(n_tracks):
			nnz[g] = 0
			mins[g] = 0.0
			maxs[g] = 0.0
			sums[g] = 0.0
			nonint[g] = False

		for g in range(n_tracks):
			if scan:
				for p in range(width):
					v = buffer[g, p]
					if v != 0.0:
						nnz[g] += 1
						sums[g] += v
						if v < mins[g]:
							mins[g] = v
						if v > maxs[g]:
							maxs[g] = v
						if v != numpy.floor(v) or v < -2147483648.0 or \
								v >= 4294967296.0:
							nonint[g] = True
			else:
				for i in range(lo[t], hi[t]):
					a = max(starts[i], base)
					b = min(ends[i], base + width)
					for p in range(a, b):
						v = buffer[g, p - base]
						if v != 0.0:
							nnz[g] += 1
							sums[g] += v
							if v < mins[g]:
								mins[g] = v
							if v > maxs[g]:
								maxs[g] = v
							if v != numpy.floor(v) or v < -2147483648.0 or \
									v >= 4294967296.0:
								nonint[g] = True

			if nonint[g]:
				integral = False
			total += nnz[g]

		if total == 0:
			kinds[t] = _ZERO
			bounds[t + 1] = position
			continue

		low = mins.min()
		high = maxs.max()
		if not integral:
			code = _F32
		elif low >= 0.0:
			code = _U8 if high <= 255.0 else _U16 if high <= 65535.0 else _U32
		elif low >= -128.0 and high <= 127.0:
			code = _I8
		elif low >= -32768.0 and high <= 32767.0:
			code = _I16
		elif high <= 2147483647.0:
			code = _I32
		else:
			code = _F32

		w = _WIDTHS[code]
		dense_bytes = n_tracks * width * w
		sparse_bytes = 4 * n_tracks + (2 * total + 3) // 4 * 4 + total * w

		if dense_bytes <= sparse_bytes:
			kinds[t] = _DENSE | (code << 2)
			end = position + dense_bytes
			if code == _U8:
				for g in range(n_tracks):
					for p in range(width):
						out[position + g * width + p] = numpy.uint8(buffer[g, p])
			elif code == _U16:
				view = out[position:end].view(numpy.uint16)
				for g in range(n_tracks):
					for p in range(width):
						view[g * width + p] = numpy.uint16(buffer[g, p])
			elif code == _U32:
				view = out[position:end].view(numpy.uint32)
				for g in range(n_tracks):
					for p in range(width):
						view[g * width + p] = numpy.uint32(buffer[g, p])
			elif code == _I8:
				view = out[position:end].view(numpy.int8)
				for g in range(n_tracks):
					for p in range(width):
						view[g * width + p] = numpy.int8(buffer[g, p])
			elif code == _I16:
				view = out[position:end].view(numpy.int16)
				for g in range(n_tracks):
					for p in range(width):
						view[g * width + p] = numpy.int16(buffer[g, p])
			elif code == _I32:
				view = out[position:end].view(numpy.int32)
				for g in range(n_tracks):
					for p in range(width):
						view[g * width + p] = numpy.int32(buffer[g, p])
			else:
				view = out[position:end].view(numpy.float32)
				for g in range(n_tracks):
					for p in range(width):
						view[g * width + p] = numpy.float32(buffer[g, p])
		else:
			kinds[t] = _SPARSE | (code << 2)
			counts = out[position:position + 4 * n_tracks].view(numpy.uint32)
			deltas_at = position + 4 * n_tracks
			values_at = deltas_at + (2 * total + 3) // 4 * 4
			end = values_at + total * w
			deltas = out[deltas_at:deltas_at + 2 * total].view(numpy.uint16)
			for p in range(deltas_at + 2 * total, values_at):
				out[p] = 0

			# Each track's nonzero bases in order, as the kernel finds them
			# by walking the buffer or the sorted items.
			d = 0
			o = values_at
			for g in range(n_tracks):
				counts[g] = nnz[g]
				last = 0
				if scan:
					for p in range(width):
						v = buffer[g, p]
						if v != 0.0:
							deltas[d] = numpy.uint16(p - last)
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
								deltas[d] = numpy.uint16(p - last)
								_store(out, o, code, v)
								last = p
								d += 1
								o += w

		for p in range(end, _pad8(end)):
			out[p] = 0
		position = _pad8(end)
		bounds[t + 1] = position

		for g in range(n_tracks):
			stats[g, 0] += sums[g]
			stats[g, 1] += nnz[g]
			if mins[g] < stats[g, 2]:
				stats[g, 2] = mins[g]
			if maxs[g] > stats[g, 3]:
				stats[g, 3] = maxs[g]
			if nonint[g]:
				stats[g, 4] = 1.0

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
def _encode_dense_tiles(tile, values, items, columns, out, bounds, kinds, stats):
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
		for g in range(n_tracks):
			nnz[g] = 0
			mins[g] = 0.0
			maxs[g] = 0.0
			sums[g] = 0.0
			nonint[g] = False
			for p in range(c0, c0 + tile):
				v = numpy.float64(block[g, p])
				if v != 0.0:
					nnz[g] += 1
					sums[g] += v
					if v < mins[g]:
						mins[g] = v
					if v > maxs[g]:
						maxs[g] = v
					if v != numpy.floor(v) or v < -2147483648.0 or \
							v >= 4294967296.0:
						nonint[g] = True
			if nonint[g]:
				integral = False
			total += nnz[g]

		if total == 0:
			kinds[t] = _ZERO
			bounds[t + 1] = position
			continue

		low = mins.min()
		high = maxs.max()
		if not integral:
			code = _F32
		elif low >= 0.0:
			code = _U8 if high <= 255.0 else _U16 if high <= 65535.0 else _U32
		elif low >= -128.0 and high <= 127.0:
			code = _I8
		elif low >= -32768.0 and high <= 32767.0:
			code = _I16
		elif high <= 2147483647.0:
			code = _I32
		else:
			code = _F32

		w = _WIDTHS[code]
		dense_bytes = n_tracks * tile * w
		sparse_bytes = 4 * n_tracks + (2 * total + 3) // 4 * 4 + total * w

		if dense_bytes <= sparse_bytes:
			kinds[t] = _DENSE | (code << 2)
			end = position + dense_bytes
			if code == _U8:
				for g in range(n_tracks):
					for p in range(tile):
						out[position + g * tile + p] = numpy.uint8(block[g, c0 + p])
			elif code == _U16:
				view = out[position:end].view(numpy.uint16)
				for g in range(n_tracks):
					for p in range(tile):
						view[g * tile + p] = numpy.uint16(block[g, c0 + p])
			elif code == _U32:
				view = out[position:end].view(numpy.uint32)
				for g in range(n_tracks):
					for p in range(tile):
						view[g * tile + p] = numpy.uint32(block[g, c0 + p])
			elif code == _I8:
				view = out[position:end].view(numpy.int8)
				for g in range(n_tracks):
					for p in range(tile):
						view[g * tile + p] = numpy.int8(block[g, c0 + p])
			elif code == _I16:
				view = out[position:end].view(numpy.int16)
				for g in range(n_tracks):
					for p in range(tile):
						view[g * tile + p] = numpy.int16(block[g, c0 + p])
			elif code == _I32:
				view = out[position:end].view(numpy.int32)
				for g in range(n_tracks):
					for p in range(tile):
						view[g * tile + p] = numpy.int32(block[g, c0 + p])
			else:
				view = out[position:end].view(numpy.float32)
				for g in range(n_tracks):
					for p in range(tile):
						view[g * tile + p] = numpy.float32(block[g, c0 + p])
		else:
			kinds[t] = _SPARSE | (code << 2)
			counts = out[position:position + 4 * n_tracks].view(numpy.uint32)
			deltas_at = position + 4 * n_tracks
			values_at = deltas_at + (2 * total + 3) // 4 * 4
			end = values_at + total * w
			deltas = out[deltas_at:deltas_at + 2 * total].view(numpy.uint16)
			for p in range(deltas_at + 2 * total, values_at):
				out[p] = 0

			d = 0
			o = values_at
			for g in range(n_tracks):
				counts[g] = nnz[g]
				last = 0
				for p in range(tile):
					v = numpy.float64(block[g, c0 + p])
					if v != 0.0:
						deltas[d] = numpy.uint16(p - last)
						_store(out, o, code, v)
						last = p
						d += 1
						o += w

		for p in range(end, _pad8(end)):
			out[p] = 0
		position = _pad8(end)
		bounds[t + 1] = position

		for g in range(n_tracks):
			stats[g, 0] += sums[g]
			stats[g, 1] += nnz[g]
			if mins[g] < stats[g, 2]:
				stats[g, 2] = mins[g]
			if maxs[g] > stats[g, 3]:
				stats[g, 3] = maxs[g]
			if nonint[g]:
				stats[g, 4] = 1.0

	return n_tiles


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
def _check_chunks(tile, data, positions, lengths, kinds, n_tracks, failed):
	"""Whether each chunk of a batch can be decoded by `_read_tiles`.

	Chunk k's bytes are data[positions[k]:positions[k] + lengths[k]], and
	kinds[k] its kind byte. A chunk fails, and gets failed[k] = 1, when its
	kind or element code is not one the format defines, when it is shorter
	than its kind says, or when a sparse chunk's offsets run past the tile.
	Returns the number of chunks that failed.
	"""

	n_failed = 0
	for k in range(positions.shape[0]):
		kind = kinds[k] & 3
		code = kinds[k] >> 2
		position = positions[k]
		length = lengths[k]
		bad = False

		if position < 0 or code > _F32 or kind == _ZERO or kind == 3:
			bad = True
		elif kind == _DENSE:
			bad = length < n_tracks * tile * _WIDTHS[code]
		elif length < 4 * n_tracks:
			bad = True
		else:
			counts = data[position:position + 4 * n_tracks].view(numpy.uint32)
			total = 0
			for g in range(n_tracks):
				total += numpy.int64(counts[g])

			deltas_at = position + 4 * n_tracks
			values_at = deltas_at + (2 * total + 3) // 4 * 4
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

		if bad:
			failed[k] = 1
			n_failed += 1

	return n_failed


@numba.njit(nogil=True, cache=True)
def _read_tiles(tile, data, positions, kinds, n_tracks, slots, pair_chunks,
	pair_bases, pair_starts, pair_rows, out):
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

	A sparse chunk's nonzero bases are scattered over zeros, and every value
	is cast to the type of `out`. A window's bases past its chromosome's end
	are not in any tile, and are set by the caller.
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
				if code == _U8:
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
		else:
			counts = data[position:position + 4 * n_tracks].view(numpy.uint32)
			total = 0
			for g in range(n_tracks):
				total += numpy.int64(counts[g])

			deltas_at = position + 4 * n_tracks
			deltas = data[deltas_at:deltas_at + 2 * total].view(numpy.uint16)
			values_at = deltas_at + (2 * total + 3) // 4 * 4

			d = 0
			for g in range(n_tracks):
				slot = slots[g]
				count = numpy.int64(counts[g])
				if slot < 0:
					d += count
					continue

				for p in range(a - start, b - start):
					out[row, slot, p] = 0

				p = 0
				for i in range(count):
					p += numpy.int64(deltas[d + i])
					at = base + p
					if at >= b:
						break
					if at >= a:
						out[row, slot, at - start] = _load(data, values_at + (d +
							i) * w, code)

				d += count

	return 0
