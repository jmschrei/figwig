# _figwig_kernels.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The compiled half of figwig's own file format: encoding tiles of values
into chunks, inflating chunks, and copying windows out of them.

A figwig file in position order holds every track's values on a fixed grid
of tiles of `tile` bases. The tiles of the tracks of one group, at one
position, are one chunk, which is stored in one of five ways. A chunk whose
every value is 0 is stored as nothing. A dense chunk holds every value,
track after track, in the chunk's element type. A sparse chunk holds the
number of nonzero bases in all (uint32), then for each track the bytes its
offsets take (uint32), then every track's nonzero bases' offsets within
the tile, the first as it is and the rest as differences from the one
before, each as a varint (LEB128: seven bits per byte, low bits first, the
high bit set on every byte but the last), then padding to a multiple of 4
bytes, then every track's nonzero values in the element type. A track's
number of bases is the number of its bytes without the high bit, so the
bytes per track let a read of some of a group's tracks step over the
others' offsets, counting them in one pass, and cost no more than a count
per track would. A runs chunk has the same layout, but its offsets are the
bases where a track's value changes, and its values the value from each
of those bases up to the next, or to the end of the tile; a track's value
is 0 before its first change. That is the layout for binned signal and
fold change, whose values hold for hundreds of bases. A matrix chunk, for
a group of many tracks few of which have a value in the tile, holds the
number of tracks present, their positions in the group (uint32,
ascending), a uint32 pointer per present track into the entries plus one
at the end, the entries' offsets within the tile (uint16, ascending within
each track, so that a track's entries in a window are found by binary
search), padding to 4 bytes, and the values. The writer picks the kind that
takes the fewest bytes before compression, counting a sparse or runs
chunk's per-track bytes at a sixty-fourth of their size in a compressed
file, where runs of zero counts deflate to almost nothing. Each chunk is
padded to a multiple of 8 bytes, so that a chunk that starts on an 8-byte
boundary can be read through typed views.

The element type is the narrowest that holds the chunk's values: uint8,
uint16 or uint32 for integers from 0, int8, int16 or int32 for integers
below 0, "one" when every nonzero value is 1, as in a tile of single-cell
cut sites, in which case no values are stored at all, and otherwise float32,
or float64 when float32 would change a value that must be kept: an integer
of a track declared or inferred as an integer type that lies beyond 2**24,
or any value of a float64 track. A runs chunk never uses "one", since its
values include the zeros a track returns to.

A file in track order holds, for each track and chromosome, blocks of up
to `_BLOCK_ENTRIES` nonzero bases: their positions on the chromosome
(uint32, ascending), padding to 8 bytes, and their values in the block's
element type.

The kind byte of a chunk or block, kept in the file's index, is the kind
(0 zero, 1 dense, 2 sparse, 3 matrix, 4 runs; 1 for a block) in its low
three bits, the element code (0 uint8, 1 uint16, 2 uint32, 3 int8, 4 int16,
5 int32, 6 float32, 7 one, 8 float64) in the next four, and in its high bit
whether the chunk is stored as it is in a compressed file, because
deflating it would have made it larger.

Every kernel here is compiled by numba with `nogil=True`, so that several
threads can encode, inflate and decode at once. The inner loops run over
one-dimensional views of the rows they touch: numba vectorizes a loop over
a row, and not the same loop written with two- or three-dimensional
indexing, which runs several times slower.
"""

import numba
import numpy


_ZERO, _DENSE, _SPARSE, _MATRIX, _RUNS = 0, 1, 2, 3, 4
_U8, _U16, _U32, _I8, _I16, _I32, _F32, _ONE, _F64 = 0, 1, 2, 3, 4, 5, 6, 7, 8
_WIDTHS = numpy.array([1, 2, 4, 1, 2, 4, 4, 0, 8], dtype=numpy.int64)
_KIND_MASK, _CODE_SHIFT, _CODE_MASK, _STORED = 7, 3, 15, 128
_BLOCK_ENTRIES = 1 << 16

# The classes of track a chunk's encoder tells apart: an integer or
# undeclared track, whose integers are kept exact; a float32 track, whose
# values are what float32 makes of them; and a float64 track.
_INTEGER, _FLOAT32, _FLOAT64 = 0, 1, 2

# The flags of a track's statistics: a value that is not an integer, an
# integer beyond what int32 and uint32 hold, and a value beyond float32.
_FLAG_FRACTION, _FLAG_WIDE, _FLAG_OVERFLOW = 1, 2, 4

# A track whose nonzero bases are fewer than this fraction of a tile is
# scanned with a branch per base, which is fast when the branch is rarely
# taken; a denser track is scanned without branches.
_SPARSE_SCAN = 0.25

# What a sparse or runs chunk's per-track byte counts cost once deflated,
# as a fraction of their size, when most of them are zero.
_DEFLATED_COUNTS = 64

# A chunk with at least this many windows on it is decoded once into a
# dense scratch tile, from which the windows are copied, rather than once
# per window.
_SCRATCH_PAIRS = 4

_EXACT = 16777216.0


@numba.njit(nogil=True, cache=True, inline='always')
def _pad8(n):
	return (n + 7) // 8 * 8


@numba.njit(nogil=True, cache=True, inline='always')
def _pad4(n):
	return (n + 3) // 4 * 4


@numba.njit(nogil=True, cache=True, inline='always')
def _kind_of(byte):
	return byte & _KIND_MASK


@numba.njit(nogil=True, cache=True, inline='always')
def _code_in(byte):
	return (byte >> _CODE_SHIFT) & _CODE_MASK


@numba.njit(nogil=True, cache=True, inline='always')
def _varint_bytes(v):
	"""The bytes the varint of v, from 0 to 2**21 - 1, takes."""

	return 1 if v < 128 else 2 if v < 16384 else 3


@numba.njit(nogil=True, cache=True, inline='always')
def _put_varint(out, at, v):
	"""Write the varint of v at `at` of `out`; returns where it ends."""

	while v >= 128:
		out[at] = numpy.uint8((v & 127) | 128)
		v >>= 7
		at += 1
	out[at] = numpy.uint8(v)
	return at + 1


@numba.njit(nogil=True, cache=True, inline='always')
def _get_varint(data, at):
	"""The varint at `at` of `data`, of up to three bytes, and where it
	ends. The second byte is read whether or not it belongs to the varint,
	so one byte past the last varint must lie within `data`; it does, since
	values or padding follow a chunk's offsets."""

	b = numpy.int64(data[at])
	if b < 128:
		return b, at + 1
	v = b & 127
	b = numpy.int64(data[at + 1])
	if b < 128:
		return v | (b << 7), at + 2
	return v | ((b & 127) << 7) | (numpy.int64(data[at + 2]) << 14), at + 3


@numba.njit(nogil=True, cache=True, inline='always')
def _as_stored(v, cls):
	"""The value a track of class `cls` keeps of v: float32's rounding of a
	float32 track's value and of any fraction, since a track with a fraction
	reads as float32; integers and a float64 track's values as they are.
	-0.0 becomes 0.0."""

	v = v + 0.0
	if cls == _FLOAT64:
		return v
	r = numpy.float64(numpy.float32(v))
	return r if (cls == _FLOAT32 or v != numpy.floor(v)) else v


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
	elif code == _F32:
		return numpy.float64(data[offset:offset + 4].view(numpy.float32)[0])
	elif code == _F64:
		return data[offset:offset + 8].view(numpy.float64)[0]
	return 1.0


@numba.njit(nogil=True, cache=True)
def _store_values(out, at, code, values, d0, d1):
	"""Write values[d0:d1] at byte `at` of `out` in element type `code`;
	returns where they end."""

	n = d1 - d0
	if code == _U8:
		for d in range(n):
			out[at + d] = numpy.uint8(values[d0 + d])
	elif code == _U16:
		view = out[at:at + 2 * n].view(numpy.uint16)
		for d in range(n):
			view[d] = numpy.uint16(values[d0 + d])
	elif code == _U32:
		view = out[at:at + 4 * n].view(numpy.uint32)
		for d in range(n):
			view[d] = numpy.uint32(values[d0 + d])
	elif code == _I8:
		view = out[at:at + n].view(numpy.int8)
		for d in range(n):
			view[d] = numpy.int8(values[d0 + d])
	elif code == _I16:
		view = out[at:at + 2 * n].view(numpy.int16)
		for d in range(n):
			view[d] = numpy.int16(values[d0 + d])
	elif code == _I32:
		view = out[at:at + 4 * n].view(numpy.int32)
		for d in range(n):
			view[d] = numpy.int32(values[d0 + d])
	elif code == _F32:
		view = out[at:at + 4 * n].view(numpy.float32)
		for d in range(n):
			view[d] = numpy.float32(values[d0 + d])
	elif code == _F64:
		view = out[at:at + 8 * n].view(numpy.float64)
		for d in range(n):
			view[d] = values[d0 + d]
	return at + n * _WIDTHS[code]


@numba.njit(nogil=True, cache=True)
def _code_of(integral, low, high, total, sum_, wide):
	"""The element code for `total` nonzero values from `low` to `high`,
	where `low` and `high` are at most and at least 0, that add up to
	`sum_`: an integer type when `integral`, float64 when `wide`, float32
	otherwise."""

	if not integral:
		return _F64 if wide else _F32
	elif low >= 0.0:
		if high == 1.0 and sum_ == total:
			return _ONE
		return _U8 if high <= 255.0 else _U16 if high <= 65535.0 else _U32
	elif low >= -128.0 and high <= 127.0:
		return _I8
	elif low >= -32768.0 and high <= 32767.0:
		return _I16
	elif high <= 2147483647.0:
		return _I32
	return _F64 if wide else _F32


@numba.njit(nogil=True, cache=True)
def _pick_kind(n_tracks, n_present, total, delta_bytes, w, width, n_points,
	run_bytes, allow_dense, allow_matrix, allow_runs, compressed):
	"""The kind of chunk that takes the fewest bytes, and those bytes. A
	dense or runs chunk of values that are all 1, whose width `w` is 0, holds
	them as uint8. In a compressed file the per-track counts of a sparse or
	runs chunk, mostly zero when the tracks are many, count at a fraction of
	their size."""

	counts = 4 * n_tracks
	if compressed:
		counts = counts // _DEFLATED_COUNTS + 4
	dense_bytes = n_tracks * width * max(w, 1)
	sparse_bytes = 4 + counts + _pad4(delta_bytes) + total * w
	matrix_bytes = 8 + 8 * n_present + _pad4(2 * total) + total * w
	runs_bytes = 4 + counts + _pad4(run_bytes) + n_points * max(w, 1)
	kind, best = _SPARSE, sparse_bytes
	if allow_matrix and matrix_bytes < best:
		kind, best = _MATRIX, matrix_bytes
	if allow_runs and runs_bytes < best:
		kind, best = _RUNS, runs_bytes
	if allow_dense and dense_bytes <= best:
		kind, best = _DENSE, dense_bytes
	return kind, best


@numba.njit(nogil=True, cache=True)
def _fold_stats(stats, nnz, mins, maxs, sums, flags):
	"""Add a tile's per-track statistics into `stats`, (tracks, 5): the sum
	with Neumaier's compensation, kept in stats[g, 5] when there is a sixth
	column, the nonzero bases, the minimum, the maximum and the flags."""

	compensated = stats.shape[1] > 5
	for g in range(stats.shape[0]):
		if compensated:
			total = stats[g, 0]
			v = sums[g]
			t = total + v
			if abs(total) >= abs(v):
				stats[g, 5] += (total - t) + v
			else:
				stats[g, 5] += (v - t) + total
			stats[g, 0] = t
		else:
			stats[g, 0] += sums[g]
		stats[g, 1] += nnz[g]
		if mins[g] < stats[g, 2]:
			stats[g, 2] = mins[g]
		if maxs[g] > stats[g, 3]:
			stats[g, 3] = maxs[g]
		stats[g, 4] = numpy.float64(numpy.int64(stats[g, 4]) | flags[g])


@numba.njit(nogil=True, cache=True)
def _dense_payload(out, position, block, c0, width, code, classes):
	"""Write block[:, c0:c0 + width] densely at `position` of `out`, and
	return where it ends."""

	n_tracks = block.shape[0]
	w = _WIDTHS[code]
	for g in range(n_tracks):
		row = block[g, c0:c0 + width]
		cls = classes[g]
		at = position + g * width * w
		if code == _U8:
			view = out[at:at + width]
			for p in range(width):
				view[p] = numpy.uint8(row[p])
		elif code == _U16:
			view = out[at:at + 2 * width].view(numpy.uint16)
			for p in range(width):
				view[p] = numpy.uint16(row[p])
		elif code == _U32:
			view = out[at:at + 4 * width].view(numpy.uint32)
			for p in range(width):
				view[p] = numpy.uint32(row[p])
		elif code == _I8:
			view = out[at:at + width].view(numpy.int8)
			for p in range(width):
				view[p] = numpy.int8(row[p])
		elif code == _I16:
			view = out[at:at + 2 * width].view(numpy.int16)
			for p in range(width):
				view[p] = numpy.int16(row[p])
		elif code == _I32:
			view = out[at:at + 4 * width].view(numpy.int32)
			for p in range(width):
				view[p] = numpy.int32(row[p])
		elif code == _F32:
			view = out[at:at + 4 * width].view(numpy.float32)
			for p in range(width):
				view[p] = numpy.float32(row[p] + 0.0)
		else:
			view = out[at:at + 8 * width].view(numpy.float64)
			for p in range(width):
				view[p] = _as_stored(numpy.float64(row[p]), cls)
	return position + n_tracks * width * w


@numba.njit(nogil=True, cache=True)
def _collect(row, width, cls, offs, vals, d, g, nnz, mins, maxs, sums, flags,
	dbytes, npts, rbytes):
	"""Scan one track's row for its nonzero bases, with a branch per base:
	their offsets go to offs[d:] and their values to vals[d:], and the
	track's statistics, offset bytes, changes of value and the bytes of
	their offsets are set. Returns the entries found."""

	lo = 0.0
	hi = 0.0
	s = 0.0
	frac = False
	over = False
	last = 0
	db = 0
	d0 = d
	prev = 0.0
	last_change = 0
	rb = 0
	points = 0
	for p in range(width):
		v = numpy.float64(row[p])
		over |= cls != _FLOAT64 and numpy.isinf(numpy.float32(v)) and not \
			numpy.isinf(v)
		v = _as_stored(v, cls)
		changed = v != prev and not (numpy.isnan(v) and numpy.isnan(prev))
		points += changed
		rb += _varint_bytes(p - last_change) if changed else 0
		last_change = p if changed else last_change
		prev = v
		if v != 0.0:
			offs[d] = p
			vals[d] = v
			d += 1
			s += v
			if v < lo:
				lo = v
			if v > hi:
				hi = v
			frac |= v != numpy.floor(v)
			db += _varint_bytes(p - last)
			last = p
	nnz[g] = d - d0
	mins[g] = lo
	maxs[g] = hi
	sums[g] = s
	flags[g] = (_FLAG_FRACTION if frac else 0) | (_FLAG_WIDE if (lo <
		-2147483648.0 or hi >= 4294967296.0) else 0) | (_FLAG_OVERFLOW if over
		else 0)
	dbytes[g] = db
	npts[g] = points
	rbytes[g] = rb
	return d - d0


@numba.njit(nogil=True, cache=True)
def _survey(row, width, cls, g, nnz, mins, maxs, sums, flags, npts, rbytes):
	"""One track's statistics from its row, without a branch per base, for
	a row with many nonzero bases. Zeros do not move a minimum or maximum
	that starts at zero, nor a sum."""

	lo = 0.0
	hi = 0.0
	s = 0.0
	frac = False
	over = False
	prev = 0.0
	last_change = 0
	rb = 0
	points = 0
	count = 0
	for p in range(width):
		v = numpy.float64(row[p])
		over |= cls != _FLOAT64 and numpy.isinf(numpy.float32(v)) and not \
			numpy.isinf(v)
		v = _as_stored(v, cls)
		changed = v != prev and not (numpy.isnan(v) and numpy.isnan(prev))
		points += changed
		rb += _varint_bytes(p - last_change) if changed else 0
		last_change = p if changed else last_change
		prev = v
		s += v
		count += v != 0.0
		lo = v if v < lo else lo
		hi = v if v > hi else hi
		frac |= v != numpy.floor(v)
	nnz[g] = count
	mins[g] = lo
	maxs[g] = hi
	sums[g] = s
	flags[g] = (_FLAG_FRACTION if frac else 0) | (_FLAG_WIDE if (lo <
		-2147483648.0 or hi >= 4294967296.0) else 0) | (_FLAG_OVERFLOW if over
		else 0)
	npts[g] = points
	rbytes[g] = rb


@numba.njit(nogil=True, cache=True)
def _collect_runs(row, width, cls, offs, vals, d):
	"""One track's changes of value, from an implicit 0 before the row: the
	offsets where the value changes go to offs[d:] and the new values to
	vals[d:]. Returns the changes found."""

	prev = 0.0
	d0 = d
	for p in range(width):
		v = _as_stored(numpy.float64(row[p]), cls)
		if v != prev and not (numpy.isnan(v) and numpy.isnan(prev)):
			offs[d] = p
			vals[d] = v
			d += 1
		prev = v
	return d - d0


@numba.njit(nogil=True, cache=True)
def _write_entries(out, position, kind, code, n_tracks, nnz, offs, vals):
	"""Write a sparse, runs or matrix chunk at `position` of `out` from the
	entries of each track, offs and vals in track order, and return where
	it ends."""

	total = 0
	n_present = 0
	for g in range(n_tracks):
		total += nnz[g]
		if nnz[g] > 0:
			n_present += 1

	if kind == _SPARSE or kind == _RUNS:
		out[position:position + 4].view(numpy.uint32)[0] = total
		lengths = out[position + 4:position + 4 + 4 * n_tracks].view(
			numpy.uint32)
		at = position + 4 + 4 * n_tracks
		d = 0
		for g in range(n_tracks):
			begin = at
			last = 0
			for i in range(nnz[g]):
				at = _put_varint(out, at, offs[d] - last)
				last = offs[d]
				d += 1
			lengths[g] = at - begin
	else:
		out[position:position + 4].view(numpy.uint32)[0] = n_present
		tracks = out[position + 4:position + 4 + 4 * n_present].view(numpy.uint32)
		ptr = out[position + 4 + 4 * n_present:position + 8 + 8 * n_present].view(
			numpy.uint32)
		k = 0
		running = 0
		for g in range(n_tracks):
			if nnz[g] > 0:
				tracks[k] = g
				ptr[k] = running
				running += nnz[g]
				k += 1
		ptr[n_present] = running
		at = position + 8 + 8 * n_present
		offsets = out[at:at + 2 * total].view(numpy.uint16)
		for d in range(total):
			offsets[d] = numpy.uint16(offs[d])
		at += 2 * total

	values_at = _pad4(at)
	for p in range(at, values_at):
		out[p] = 0
	return _store_values(out, values_at, code, vals, 0, total)


@numba.njit(nogil=True, cache=True)
def _finish_chunk(out, end):
	"""Zero the padding after a chunk; returns where the next one starts."""

	for p in range(end, _pad8(end)):
		out[p] = 0
	return _pad8(end)


@numba.njit(nogil=True, cache=True)
def _chunk_code(n_tracks, classes, nnz, mins, maxs, sums, flags):
	"""The element code of a tile from its tracks' statistics, and the
	tile's total, low, high and sum. A track of the integer class whose
	integers lie beyond 2**24, or a float64 track, makes a chunk that cannot
	be an integer type float64 rather than float32."""

	integral = True
	wide = False
	total = 0
	low, high, sum_ = 0.0, 0.0, 0.0
	for g in range(n_tracks):
		total += nnz[g]
		if flags[g] & (_FLAG_FRACTION | _FLAG_WIDE):
			integral = False
		if nnz[g] > 0:
			cls = classes[g]
			if cls == _FLOAT64 or (cls == _INTEGER and not flags[g] &
					_FLAG_FRACTION and (maxs[g] > _EXACT or mins[g] < -_EXACT)):
				wide = True
		if mins[g] < low:
			low = mins[g]
		if maxs[g] > high:
			high = maxs[g]
		sum_ += sums[g]
	return _code_of(integral, low, high, total, sum_, wide), total


@numba.njit(nogil=True, cache=True)
def _encode_block(block, c0, width, classes, allow_matrix, compressed, clear,
	out, position, offs, vals, nnz, mins, maxs, sums, flags, dbytes, npts,
	rbytes, stats):
	"""Encode block[:, c0:c0 + width], one tile of a group, as one chunk at
	`position` of `out`. Returns its kind byte, 0 for a tile whose values
	are all zero, and where the chunk ends.

	Each track's row is scanned for its nonzero bases, which are collected
	in `offs` and `vals`, scratch of a tile of every track, when the row
	has few, and surveyed for its statistics when it has many; the
	statistics are folded into `stats`. `classes` gives each track's class.
	With `clear`, the rows are zero on return.
	"""

	n_tracks = block.shape[0]
	total = 0
	n_present = 0
	collected = True
	for g in range(n_tracks):
		row = block[g, c0:c0 + width]
		nz = 0
		for p in range(width):
			nz += row[p] != 0
		if nz == 0:
			nnz[g] = 0
			mins[g] = 0.0
			maxs[g] = 0.0
			sums[g] = 0.0
			flags[g] = 0
			dbytes[g] = 0
			npts[g] = 0
			rbytes[g] = 0
			continue

		n_present += 1
		if nz < _SPARSE_SCAN * width:
			total += _collect(row, width, classes[g], offs, vals, total, g, nnz,
				mins, maxs, sums, flags, dbytes, npts, rbytes)
		else:
			_survey(row, width, classes[g], g, nnz, mins, maxs, sums, flags,
				npts, rbytes)
			dbytes[g] = -1
			total += nnz[g]
			collected = False

	if total == 0:
		return 0, position

	code, total = _chunk_code(n_tracks, classes, nnz, mins, maxs, sums, flags)
	w = _WIDTHS[code]
	n_points = 0
	run_bytes = 0
	for g in range(n_tracks):
		n_points += npts[g]
		run_bytes += rbytes[g]

	# A surveyed track's offsets have not been counted; unless the chunk is
	# dense even if every offset took one byte, its entries are collected
	# now, after the tracks before it.
	kind = _DENSE
	if not collected:
		kind, _ = _pick_kind(n_tracks, n_present, total, total, w, width,
			n_points, run_bytes, True, allow_matrix, True, compressed)
		if kind != _DENSE:
			d = 0
			for g in range(n_tracks):
				if dbytes[g] < 0:
					_collect(block[g, c0:c0 + width], width, classes[g], offs, vals,
						d, g, nnz, mins, maxs, sums, flags, dbytes, npts, rbytes)
				d += nnz[g]
			collected = True
	if collected:
		delta_bytes = 0
		for g in range(n_tracks):
			delta_bytes += dbytes[g]
		kind, _ = _pick_kind(n_tracks, n_present, total, delta_bytes, w, width,
			n_points, run_bytes, True, allow_matrix, True, compressed)

	if kind == _DENSE:
		if code == _ONE:
			code = _U8
		end = _dense_payload(out, position, block, c0, width, code, classes)
	elif kind == _RUNS:
		# The changes of value take the place of the nonzero bases in the
		# scratch, so the rows are cleared whole below.
		if code == _ONE:
			code = _U8
		d = 0
		for g in range(n_tracks):
			npts[g] = _collect_runs(block[g, c0:c0 + width], width, classes[g],
				offs, vals, d)
			d += npts[g]
			dbytes[g] = -1
		end = _write_entries(out, position, kind, code, n_tracks, npts, offs,
			vals)
	else:
		end = _write_entries(out, position, kind, code, n_tracks, nnz, offs,
			vals)
	_fold_stats(stats, nnz, mins, maxs, sums, flags)

	if clear:
		d = 0
		for g in range(n_tracks):
			row = block[g, c0:c0 + width]
			if dbytes[g] < 0:
				row[:] = 0
			else:
				for i in range(nnz[g]):
					row[offs[d + i]] = 0
			d += nnz[g]

	return kind | (code << _CODE_SHIFT), end


@numba.njit(nogil=True, cache=True)
def _encode_tiles(tile, tiles, lo, hi, starts, ends, values, by_base, buffer,
	preset, keep_last, classes, allow_matrix, compressed, out, bounds, kinds,
	stats):
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
	bytes. stats[g] accumulates the sum, nonzero bases, minimum, maximum and
	flags of the encoded values of track g, whose class is classes[g]. With
	`allow_matrix`, a tile may be stored as a matrix chunk; `compressed`
	says whether the chunks will be deflated.

	Returns the number of tiles encoded, which is len(tiles) or one fewer.
	"""

	n_tracks, width = buffer.shape
	n_tiles = tiles.shape[0]
	nnz = numpy.zeros(n_tracks, dtype=numpy.int64)
	mins = numpy.zeros(n_tracks, dtype=numpy.float64)
	maxs = numpy.zeros(n_tracks, dtype=numpy.float64)
	sums = numpy.zeros(n_tracks, dtype=numpy.float64)
	flags = numpy.zeros(n_tracks, dtype=numpy.int64)
	dbytes = numpy.zeros(n_tracks, dtype=numpy.int64)
	npts = numpy.zeros(n_tracks, dtype=numpy.int64)
	rbytes = numpy.zeros(n_tracks, dtype=numpy.int64)
	offs = numpy.empty(n_tracks * width, dtype=numpy.int64)
	vals = numpy.empty(n_tracks * width, dtype=numpy.float64)
	position = bounds[0]

	for t in range(n_tiles):
		base = tiles[t] * tile
		for i in range(lo[t], hi[t]):
			a = max(starts[i], base)
			b = min(ends[i], base + width)
			n = b - a
			for g in range(n_tracks):
				# A short item, such as one base's counts, is copied by
				# index; a long one through row views, which vectorize.
				if n < 16:
					for p in range(n):
						buffer[g, a - base + p] = values[i, g, (a - starts[i] + p) if
							by_base else 0]
				elif by_base:
					row = buffer[g, a - base:b - base]
					source = values[i, g, a - starts[i]:b - starts[i]]
					for p in range(n):
						row[p] = source[p]
				else:
					buffer[g, a - base:b - base] = values[i, g, 0]

		if t == n_tiles - 1 and keep_last:
			break

		kinds[t], end = _encode_block(buffer, 0, width, classes, allow_matrix,
			compressed, True, out, position, offs, vals, nnz, mins, maxs, sums,
			flags, dbytes, npts, rbytes, stats)
		position = _finish_chunk(out, end)
		bounds[t + 1] = position

	return n_tiles - 1 if keep_last and n_tiles > 0 else n_tiles


@numba.njit(nogil=True, cache=True)
def _encode_dense_tiles(tile, values, items, columns, classes, allow_matrix,
	compressed, out, bounds, kinds, stats):
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
	flags = numpy.zeros(n_tracks, dtype=numpy.int64)
	dbytes = numpy.zeros(n_tracks, dtype=numpy.int64)
	npts = numpy.zeros(n_tracks, dtype=numpy.int64)
	rbytes = numpy.zeros(n_tracks, dtype=numpy.int64)
	offs = numpy.empty(n_tracks * tile, dtype=numpy.int64)
	vals = numpy.empty(n_tracks * tile, dtype=numpy.float64)
	position = bounds[0]

	for t in range(n_tiles):
		kinds[t], end = _encode_block(values[items[t]], columns[t], tile,
			classes, allow_matrix, compressed, False, out, position, offs, vals,
			nnz, mins, maxs, sums, flags, dbytes, npts, rbytes, stats)
		position = _finish_chunk(out, end)
		bounds[t + 1] = position

	return n_tiles


@numba.njit(nogil=True, cache=True)
def _encode_entries(tile, n_tracks, tiles, first, tracks, positions, values,
	classes, allow_matrix, compressed, out, bounds, kinds, stats):
	"""Encode tiles from their nonzero bases, given as entries.

	The entries of tile t are first[t] to first[t + 1] - 1, sorted by track
	then position, with no two on one (track, position): entry e is base
	positions[e] of the chromosome in track tracks[e], a position in the
	group, with the value values[e]. Tiles are stored sparsely or as a
	matrix, never densely. The chunks, `bounds`, `kinds` and `stats` are as
	in `_encode_tiles`. Encoding stops before a tile whose chunk might not
	fit the rest of `out`; returns the number of tiles encoded.
	"""

	nnz = numpy.zeros(n_tracks, dtype=numpy.int64)
	mins = numpy.zeros(n_tracks, dtype=numpy.float64)
	maxs = numpy.zeros(n_tracks, dtype=numpy.float64)
	sums = numpy.zeros(n_tracks, dtype=numpy.float64)
	flags = numpy.zeros(n_tracks, dtype=numpy.int64)
	most = 0
	for t in range(tiles.shape[0]):
		most = max(most, first[t + 1] - first[t])
	offs = numpy.empty(most, dtype=numpy.int64)
	vals = numpy.empty(most, dtype=numpy.float64)
	position = bounds[0]

	for t in range(tiles.shape[0]):
		base = tiles[t] * tile
		e0, e1 = first[t], first[t + 1]
		n = e1 - e0
		need = max(4 + 4 * n_tracks + 3 * n + 8 * n + 16, 24 + 18 * n + 16)
		if position + need > out.shape[0]:
			return t
		for g in range(n_tracks):
			nnz[g] = 0
			mins[g] = 0.0
			maxs[g] = 0.0
			sums[g] = 0.0
			flags[g] = 0

		# The entries of each track, in order, with their statistics and
		# the bytes of their offsets.
		total = 0
		delta_bytes = 0
		n_present = 0
		last_track = -1
		last = 0
		for e in range(e0, e1):
			g = tracks[e]
			v = values[e]
			if classes[g] != _FLOAT64 and numpy.isinf(numpy.float32(v)) and not \
					numpy.isinf(v):
				flags[g] |= _FLAG_OVERFLOW
			v = _as_stored(v, classes[g])
			if v == 0.0:
				continue
			if g != last_track:
				last_track = g
				last = 0
				n_present += 1
			p = positions[e] - base
			offs[total] = p
			vals[total] = v
			total += 1
			nnz[g] += 1
			sums[g] += v
			if v < mins[g]:
				mins[g] = v
			if v > maxs[g]:
				maxs[g] = v
			if v != numpy.floor(v):
				flags[g] |= _FLAG_FRACTION
			delta_bytes += _varint_bytes(p - last)
			last = p

		if total == 0:
			kinds[t] = _ZERO
			bounds[t + 1] = position
			continue

		for g in range(n_tracks):
			if mins[g] < -2147483648.0 or maxs[g] >= 4294967296.0:
				flags[g] |= _FLAG_WIDE
		code, total = _chunk_code(n_tracks, classes, nnz, mins, maxs, sums, flags)
		w = _WIDTHS[code]
		kind, _ = _pick_kind(n_tracks, n_present, total, delta_bytes, w, tile,
			0, 0, False, allow_matrix, False, compressed)
		kinds[t] = kind | (code << _CODE_SHIFT)
		end = _write_entries(out, position, kind, code, n_tracks, nnz, offs,
			vals)
		position = _finish_chunk(out, end)
		bounds[t + 1] = position
		_fold_stats(stats, nnz, mins, maxs, sums, flags)

	return tiles.shape[0]


@numba.njit(nogil=True, cache=True)
def _encode_blocks(positions, values, first, cls, out, bounds, kinds, stats):
	"""Encode blocks of one track's nonzero bases on one chromosome, for a
	file in track order.

	Block t holds entries first[t] to first[t + 1] - 1: positions[e] on the
	chromosome, ascending, with values[e], none of them 0. Its bytes are
	out[bounds[t]:bounds[t + 1]]: the positions as uint32, padding to 8
	bytes, the values in the block's element type, padding to 8 bytes.
	`stats` is the (1, 5) statistics row of the track, of class `cls`.
	"""

	nnz = numpy.zeros(1, dtype=numpy.int64)
	mins = numpy.zeros(1, dtype=numpy.float64)
	maxs = numpy.zeros(1, dtype=numpy.float64)
	sums = numpy.zeros(1, dtype=numpy.float64)
	flags = numpy.zeros(1, dtype=numpy.int64)
	classes = numpy.full(1, cls, dtype=numpy.int64)
	position = bounds[0]
	for t in range(first.shape[0] - 1):
		e0, e1 = first[t], first[t + 1]
		n = e1 - e0
		lo = 0.0
		hi = 0.0
		s = 0.0
		frac = False
		over = False
		for e in range(e0, e1):
			v = values[e]
			over |= cls != _FLOAT64 and numpy.isinf(numpy.float32(v)) and not \
				numpy.isinf(v)
			v = _as_stored(v, cls)
			values[e] = v
			s += v
			lo = v if v < lo else lo
			hi = v if v > hi else hi
			frac |= v != numpy.floor(v)
		nnz[0] = n
		mins[0] = lo
		maxs[0] = hi
		sums[0] = s
		flags[0] = (_FLAG_FRACTION if frac else 0) | (_FLAG_WIDE if (lo <
			-2147483648.0 or hi >= 4294967296.0) else 0) | (_FLAG_OVERFLOW if over
			else 0)

		code, _ = _chunk_code(1, classes, nnz, mins, maxs, sums, flags)
		kinds[t] = _DENSE | (code << _CODE_SHIFT)
		view = out[position:position + 4 * n].view(numpy.uint32)
		for e in range(n):
			view[e] = numpy.uint32(positions[e0 + e])
		values_at = _pad8(position + 4 * n)
		for p in range(position + 4 * n, values_at):
			out[p] = 0
		end = _store_values(out, values_at, code, values, e0, e1)
		position = _finish_chunk(out, end)
		bounds[t + 1] = position
		_fold_stats(stats, nnz, mins, maxs, sums, flags)

	return first.shape[0] - 1


@numba.njit(nogil=True, cache=True)
def _inflate_chunks(uncompress, data, starts, sizes, stored, buffer, positions,
	lengths, length):
	"""Inflate chunks one after another into `buffer`, each starting on an
	8-byte boundary, without the GIL.

	Chunk k is the zlib stream data[starts[k]:starts[k] + sizes[k]], or its
	bytes as they are when stored[k]. On return positions[k] is where its
	bytes start in `buffer` and lengths[k] how many there are, or
	positions[k] is -1 if uncompress() could not inflate it into the rest of
	`buffer`, in which case zlib.decompress gets it. `length` is a
	one-element array of C unsigned longs, which are 32 bits on Windows.
	Returns the number of bytes of `buffer` used.
	"""

	position = 0
	for k in range(starts.shape[0]):
		source = data[starts[k]:starts[k] + sizes[k]]
		if stored[k]:
			if position + sizes[k] > buffer.shape[0]:
				positions[k] = -1
				lengths[k] = 0
				continue
			buffer[position:position + sizes[k]] = source
			positions[k] = position
			lengths[k] = sizes[k]
			position = _pad8(position + sizes[k])
			continue

		length[0] = min(buffer.shape[0] - position, 2 ** 32 - 1)
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
def _inflate_chunks_libdeflate(decompress, decompressor, data, starts, sizes,
	stored, buffer, positions, lengths, length):
	"""`_inflate_chunks` with libdeflate's zlib_decompress(), which is about
	twice as fast as zlib's uncompress().

	`decompressor` is a libdeflate decompressor, used by one thread at a
	time, and `length` a one-element array of size_t. A chunk that
	zlib_decompress() rejects, or that does not fit the rest of `buffer`,
	gets positions[k] = -1 and is left to zlib.decompress.
	"""

	position = 0
	for k in range(starts.shape[0]):
		source = data[starts[k]:starts[k] + sizes[k]]
		if stored[k]:
			if position + sizes[k] > buffer.shape[0]:
				positions[k] = -1
				lengths[k] = 0
				continue
			buffer[position:position + sizes[k]] = source
			positions[k] = position
			lengths[k] = sizes[k]
			position = _pad8(position + sizes[k])
			continue

		status = decompress(decompressor, source.ctypes, sizes[k],
			buffer[position:].ctypes, buffer.shape[0] - position, length.ctypes)

		if status != 0:
			positions[k] = -1
			lengths[k] = 0
		else:
			positions[k] = position
			lengths[k] = numpy.int64(length[0])
			position = _pad8(position + lengths[k])

	return position


@numba.njit(nogil=True, cache=True)
def _sparse_header(data, position, n_tracks):
	"""A sparse or runs chunk's (lengths, total, deltas_at, delta_bytes,
	values_at): `lengths` holds the bytes each track's offsets take, from
	`deltas_at` on, and `delta_bytes` their sum."""

	total = numpy.int64(data[position:position + 4].view(numpy.uint32)[0])
	lengths = data[position + 4:position + 4 + 4 * n_tracks].view(numpy.uint32)
	delta_bytes = 0
	for g in range(n_tracks):
		delta_bytes += numpy.int64(lengths[g])
	deltas_at = position + 4 + 4 * n_tracks
	values_at = deltas_at + _pad4(delta_bytes)
	return lengths, total, deltas_at, delta_bytes, values_at


@numba.njit(nogil=True, cache=True)
def _count_varints(data, at, end):
	"""The varints in data[at:end], by their last bytes."""

	n = 0
	for q in range(at, end):
		n += data[q] < 128
	return n


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
def _check_chunks(tile, data, positions, lengths, kinds, n_tracks, slots,
	failed):
	"""Whether each chunk of a batch can be decoded by `_read_tiles` for the
	tracks with a slot.

	Chunk k's bytes are data[positions[k]:positions[k] + lengths[k]], and
	kinds[k] its kind byte. A chunk fails, and gets failed[k] = 1, when its
	kind or element code is not one the format defines, when it is shorter
	than its kind says, when its track positions are out of range or order,
	when a sparse or runs chunk's varints do not add up to its total, or
	when the offsets of a track that is read, one with slots[g] >= 0, are
	out of range or order or its varints do not fill their bytes exactly;
	the offsets of a track that is not read are only counted, which is all
	the decoders do with them. Returns the number of chunks that failed.
	"""

	n_failed = 0
	for k in range(positions.shape[0]):
		kind = _kind_of(kinds[k])
		code = _code_in(kinds[k])
		position = positions[k]
		length = lengths[k]
		bad = False

		if position < 0 or code > _F64 or kind == _ZERO or kind > _RUNS:
			bad = True
		elif kind == _DENSE:
			bad = code == _ONE or length < n_tracks * tile * _WIDTHS[code]
		elif kind == _SPARSE or kind == _RUNS:
			if length < 4 + 4 * n_tracks or (kind == _RUNS and code == _ONE):
				bad = True
			else:
				track_bytes, total, deltas_at, delta_bytes, values_at = \
					_sparse_header(data, position, n_tracks)
				if length < values_at - position + total * _WIDTHS[code] or \
						_count_varints(data, deltas_at, deltas_at + delta_bytes) != \
						total:
					bad = True
				else:
					at = deltas_at
					for g in range(n_tracks):
						end = at + numpy.int64(track_bytes[g])
						if slots[g] < 0:
							at = end
							continue
						p = 0
						while at < end:
							# A varint of up to three bytes, whole within the
							# track's bytes: checked byte by byte near their end.
							if at + 3 > end and data[at] >= 128 and (at + 1 >= end or (
									data[at + 1] >= 128 and (at + 2 >= end or data[at +
									2] >= 128))):
								bad = True
								break
							delta, at = _get_varint(data, at)
							p += delta
							if p >= tile:
								bad = True
								break
						if bad or at != end:
							bad = True
							break
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
							if slots[tracks[j]] < 0:
								continue
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
def _check_blocks(data, positions, lengths, kinds, n_entries, firsts, lasts,
	failed):
	"""Whether each block of a track-order batch can be decoded: its kind
	is 1, its code known, it is long enough, its positions ascend, and its
	first and last positions are those its index record gives."""

	n_failed = 0
	for k in range(positions.shape[0]):
		kind = _kind_of(kinds[k])
		code = _code_in(kinds[k])
		n = n_entries[k]
		position = positions[k]
		bad = position < 0 or kind != _DENSE or code > _F64 or n == 0 or \
			lengths[k] < _pad8(4 * n) + n * _WIDTHS[code]
		if not bad:
			view = data[position:position + 4 * n].view(numpy.uint32)
			bad = numpy.int64(view[0]) != firsts[k] or numpy.int64(view[n - 1]) \
				!= lasts[k]
			for i in range(1, n):
				if view[i] <= view[i - 1]:
					bad = True
					break
		if bad:
			failed[k] = 1
			n_failed += 1

	return n_failed


@numba.njit(nogil=True, cache=True, inline='always')
def _copy_row(dst, view, n, accumulate):
	if accumulate:
		for p in range(n):
			dst[p] += view[p]
	else:
		for p in range(n):
			dst[p] = view[p]


@numba.njit(nogil=True, cache=True)
def _copy_dense(dst, data, at, code, n, accumulate):
	"""dst[:n] = the n values of element type `code` at `at` of `data`, or
	dst[:n] += them with `accumulate`."""

	if code == _U8:
		_copy_row(dst, data[at:at + n], n, accumulate)
	elif code == _U16:
		_copy_row(dst, data[at:at + 2 * n].view(numpy.uint16), n, accumulate)
	elif code == _U32:
		_copy_row(dst, data[at:at + 4 * n].view(numpy.uint32), n, accumulate)
	elif code == _I8:
		_copy_row(dst, data[at:at + n].view(numpy.int8), n, accumulate)
	elif code == _I16:
		_copy_row(dst, data[at:at + 2 * n].view(numpy.int16), n, accumulate)
	elif code == _I32:
		_copy_row(dst, data[at:at + 4 * n].view(numpy.int32), n, accumulate)
	elif code == _F64:
		_copy_row(dst, data[at:at + 8 * n].view(numpy.float64), n, accumulate)
	else:
		_copy_row(dst, data[at:at + 4 * n].view(numpy.float32), n, accumulate)


@numba.njit(nogil=True, cache=True, inline='always')
def _value(values, i, ones):
	"""Value i of a typed view of a chunk's values, or 1 for a chunk whose
	values are all 1 and stores none."""

	return 1.0 if ones else numpy.float64(values[i])


@numba.njit(nogil=True, cache=True)
def _put_run(value, s, e, start, channels, slot, accumulate, listing, fill,
	row, k_out, out_rows, out_channels, out_offsets, out_values):
	"""Bases s to e - 1 of a window that starts at `start` take `value`:
	set or added into row `slot` of `channels`, or listed as entries when
	the value is not 0. Returns the entries listed."""

	if listing:
		if value != 0.0:
			for p in range(s, e):
				if fill:
					out_rows[k_out] = row
					out_channels[k_out] = slot
					out_offsets[k_out] = p - start
					out_values[k_out] = value
				k_out += 1
		return k_out

	dst = channels[slot]
	if accumulate:
		for p in range(s - start, e - start):
			dst[p] += value
	else:
		for p in range(s - start, e - start):
			dst[p] = value
	return k_out


# The decoders below take a chunk's values as a typed view, so that numba
# compiles one copy of each for every element type and the type is chosen
# once per chunk rather than once per value; `ones` marks a chunk whose
# values are all 1 and whose view is empty.

@numba.njit(nogil=True, cache=True)
def _window_entries(kind, data, at, counts, tracks, ptr, offsets, slots,
	selected, values, ones, base, a, b, start, channels, accumulate, fill, row,
	k_out, out_rows, out_channels, out_offsets, out_values):
	"""One window's bases of one sparse, runs or matrix chunk: set or added
	into the window's `channels`, (channels, width) of the output, when
	`channels` has a row, or listed as entries from k_out on when `fill`,
	counted either way. Returns the entries in the window.

	A sparse or runs chunk's varints start at `at` of `data`, `counts[g]`
	bytes of them for track g; a matrix chunk's present `tracks`, `ptr` and
	uint16 `offsets` are as its header gives them, and each present track's
	first base in the window is found by binary search. `selected` lists
	the group's tracks with a slot, ascending. `values` is the typed view of
	the chunk's values.
	"""

	listing = channels.shape[0] == 0
	if kind == _SPARSE or kind == _RUNS:
		# Entry d of the values is the d-th varint; the varints of tracks
		# not read, and those after the window, are counted by their last
		# bytes, in one pass over every run of them, from `pending` on.
		d = 0
		pending = at
		for g in range(counts.shape[0]):
			slot = slots[g]
			end = at + numpy.int64(counts[g])
			if slot < 0:
				at = end
				continue

			if pending < at:
				d += _count_varints(data, pending, at)
			if not listing and not accumulate and kind == _SPARSE:
				channels[slot, a - start:b - start] = 0

			p = 0
			if kind == _RUNS:
				# The value from each change up to the next, clipped to the
				# window; 0 before the first change.
				current = 0.0
				since = base
				while at < end:
					delta, at = _get_varint(data, at)
					p += delta
					here = base + p
					if here >= b:
						d += 1
						break
					if here > since:
						k_out = _put_run(current, max(since, a), here, start, channels,
							slot, accumulate, listing, fill, row, k_out, out_rows,
							out_channels, out_offsets, out_values)
					current = _value(values, d, ones)
					since = here
					d += 1
				if since < b:
					k_out = _put_run(current, max(since, a), b, start, channels, slot,
						accumulate, listing, fill, row, k_out, out_rows,
						out_channels, out_offsets, out_values)
			else:
				while at < end:
					delta, at = _get_varint(data, at)
					p += delta
					here = base + p
					if here >= b:
						d += 1
						break
					if here >= a:
						if listing:
							if fill:
								out_rows[k_out] = row
								out_channels[k_out] = slot
								out_offsets[k_out] = here - start
								out_values[k_out] = _value(values, d, ones)
							k_out += 1
						elif accumulate:
							channels[slot, here - start] += _value(values, d, ones)
						else:
							channels[slot, here - start] = _value(values, d, ones)
					d += 1
			pending = at
			at = end
		return k_out

	# The present tracks that are read: all of them in turn, or, when few
	# of the group's tracks are read, each of those looked up among the
	# present ones by binary search.
	n_present = tracks.shape[0]
	few = selected.shape[0] * 16 < n_present
	for j in range(selected.shape[0] if few else n_present):
		if few:
			g = selected[j]
			lo, hi = 0, n_present
			while lo < hi:
				mid = (lo + hi) // 2
				if numpy.int64(tracks[mid]) < g:
					lo = mid + 1
				else:
					hi = mid
			if lo == n_present or numpy.int64(tracks[lo]) != g:
				continue
			k = lo
		else:
			k = j
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
			here = base + numpy.int64(offsets[i])
			if here >= b:
				break
			if listing:
				if fill:
					out_rows[k_out] = row
					out_channels[k_out] = slot
					out_offsets[k_out] = here - start
					out_values[k_out] = _value(values, i, ones)
				k_out += 1
			elif accumulate:
				channels[slot, here - start] += _value(values, i, ones)
			else:
				channels[slot, here - start] = _value(values, i, ones)

	return k_out


@numba.njit(nogil=True, cache=True)
def _dense_entries(data, at, code, n, row, slot, a, start, fill, k_out,
	out_rows, out_channels, out_offsets, out_values):
	"""The nonzero bases among n values of a dense chunk, as entries."""

	for p in range(n):
		v = _load(data, at + p * _WIDTHS[code], code)
		if v != 0.0:
			if fill:
				out_rows[k_out] = row
				out_channels[k_out] = slot
				out_offsets[k_out] = a - start + p
				out_values[k_out] = v
			k_out += 1
	return k_out


@numba.njit(nogil=True, cache=True)
def _window_of_chunk(tile, data, position, kind, code, n_tracks, slots,
	selected, base, start, a, b, channels, accumulate, fill, row, k_out,
	out_rows, out_channels, out_offsets, out_values):
	"""The bases of one window in one chunk, set or added into `channels`
	or listed as entries, by the chunk's kind and element type. Returns the
	entries in the window, for a listing."""

	w = _WIDTHS[code]
	listing = channels.shape[0] == 0
	if kind == _DENSE:
		for g in range(n_tracks):
			slot = slots[g]
			if slot < 0:
				continue
			at = position + (g * tile + (a - base)) * w
			if listing:
				k_out = _dense_entries(data, at, code, b - a, row, slot, a, start,
					fill, k_out, out_rows, out_channels, out_offsets, out_values)
			else:
				_copy_dense(channels[slot, a - start:b - start], data, at, code,
					b - a, accumulate)
		return k_out

	if kind == _SPARSE or kind == _RUNS:
		counts, total, at, delta_bytes, values_at = _sparse_header(data,
			position, n_tracks)
		tracks = ptr = counts[:0]
		offsets = data[at:at].view(numpy.uint16)
	else:
		n_present, tracks, ptr, at, values_at, total = _matrix_header(data,
			position)
		counts = tracks[:0]
		offsets = data[at:at + 2 * total].view(numpy.uint16)
		if not listing and not accumulate:
			for g in range(n_tracks):
				if slots[g] >= 0:
					channels[slots[g], a - start:b - start] = 0

	v0 = values_at
	if code == _U8:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + total], False, base, a, b, start, channels,
			accumulate, fill, row, k_out, out_rows, out_channels, out_offsets,
			out_values)
	elif code == _U16:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + 2 * total].view(numpy.uint16), False, base, a, b,
			start, channels, accumulate, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)
	elif code == _U32:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + 4 * total].view(numpy.uint32), False, base, a, b,
			start, channels, accumulate, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)
	elif code == _I8:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + total].view(numpy.int8), False, base, a, b,
			start, channels, accumulate, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)
	elif code == _I16:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + 2 * total].view(numpy.int16), False, base, a, b,
			start, channels, accumulate, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)
	elif code == _I32:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + 4 * total].view(numpy.int32), False, base, a, b,
			start, channels, accumulate, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)
	elif code == _F32:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + 4 * total].view(numpy.float32), False, base, a,
			b, start, channels, accumulate, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)
	elif code == _F64:
		return _window_entries(kind, data, at, counts, tracks, ptr, offsets,
			slots, selected, data[v0:v0 + 8 * total].view(numpy.float64), False, base, a,
			b, start, channels, accumulate, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)
	return _window_entries(kind, data, at, counts, tracks, ptr, offsets, slots,
		selected, data[v0:v0], True, base, a, b, start, channels, accumulate,
		fill, row, k_out, out_rows, out_channels, out_offsets, out_values)


@numba.njit(nogil=True, cache=True)
def _bin_into(dst, src, src0, s0, s1, bins):
	"""dst[k] += the sum over each bin k of `bins` bases of a window that
	the window's bases s0 to s1 - 1 touch, base s0 being src[src0]."""

	k0 = s0 // bins
	k1 = (s1 - 1) // bins
	for k in range(k0, k1 + 1):
		e0 = max(k * bins, s0)
		e1 = min((k + 1) * bins, s1)
		total = 0.0
		for p in range(e0, e1):
			total += src[src0 + p - s0]
		dst[k] += total


@numba.njit(nogil=True, cache=True)
def _read_tiles(tile, data, positions, kinds, n_tracks, slots, pair_chunks,
	pair_bases, pair_starts, pair_rows, out, accumulate, scratch, bins):
	"""Copy the parts of windows that fall in a batch of chunks into `out`.

	Pair q is a window and one tile it overlaps: the tile starts at base
	pair_bases[q] of its chromosome, the window at pair_starts[q] and is
	out.shape[2] * bins bases wide, and the window's values go to row
	pair_rows[q] of `out`, of shape (windows, channels, width). The tile's
	chunk is chunk pair_chunks[q] of the batch, whose bytes start at
	positions[chunk] of `data` and whose kind byte is kinds[chunk], or -1
	for a tile whose values are all zero. The chunk holds `n_tracks` tracks,
	and track g of it goes to channel slots[g], or nowhere when slots[g] is
	-1. Every chunk has passed `_check_chunks`. Pairs on one chunk are
	consecutive.

	Without `accumulate`, the window's bases are set, zeros included, and
	every value is cast to the type of `out`. With it, values are added to
	`out`, which is zero on entry, so that several tracks with one channel
	in `slots` are summed. With `bins` above 1, each cell of `out` is the
	sum of `bins` consecutive bases of the window, added to `out`, which is
	zero on entry. A window's bases past its chromosome's end are not in any
	tile, and are set by the caller.

	`scratch`, (channels, tile) float64 or empty, holds a whole chunk's
	decoded tracks when many pairs fall on one chunk, or whenever `bins`
	is above 1, so that the chunk is decoded once rather than once per
	window.
	"""

	width = out.shape[2] * bins
	empty = numpy.zeros(0, dtype=numpy.int64)
	empty_values = numpy.zeros(0, dtype=numpy.float64)
	selected = numpy.flatnonzero(slots >= 0)
	targets = numpy.unique(slots[selected])
	n_pairs = pair_chunks.shape[0]
	decoded = -1
	q = 0
	while q < n_pairs:
		chunk = pair_chunks[q]
		q1 = q + 1
		while q1 < n_pairs and pair_chunks[q1] == chunk:
			q1 += 1

		if chunk < 0:
			if not accumulate and bins == 1:
				for r in range(q, q1):
					base = pair_bases[r]
					start = pair_starts[r]
					a = max(start, base)
					b = min(start + width, base + tile)
					if b <= a:
						continue
					for g in range(n_tracks):
						slot = slots[g]
						if slot >= 0:
							out[pair_rows[r], slot, a - start:b - start] = 0
			q = q1
			continue

		kind = _kind_of(kinds[chunk])
		code = _code_in(kinds[chunk])
		via_scratch = scratch.shape[0] > 0 and (bins > 1 or (q1 - q >=
			_SCRATCH_PAIRS and kind != _DENSE))
		if via_scratch and chunk != decoded:
			base = pair_bases[q]
			if accumulate:
				for j in range(targets.shape[0]):
					scratch[targets[j], :] = 0
			_window_of_chunk(tile, data, positions[chunk], kind, code, n_tracks,
				slots, selected, base, base, base, base + tile, scratch, accumulate,
				False, 0, 0, empty, empty, empty, empty_values)
			decoded = chunk

		for r in range(q, q1):
			base = pair_bases[r]
			start = pair_starts[r]
			row = pair_rows[r]
			a = max(start, base)
			b = min(start + width, base + tile)
			if b <= a:
				continue
			if not via_scratch:
				_window_of_chunk(tile, data, positions[chunk], kind, code, n_tracks,
					slots, selected, base, start, a, b, out[row], accumulate, False,
					row, 0, empty, empty, empty, empty_values)
				continue
			for j in range(targets.shape[0]):
				slot = targets[j]
				if bins > 1:
					_bin_into(out[row, slot], scratch[slot], a - base, a - start,
						b - start, bins)
				else:
					_copy_row(out[row, slot, a - start:b - start], scratch[slot,
						a - base:b - base], b - a, accumulate)
		q = q1

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
	no_channels = numpy.zeros((0, 0), dtype=numpy.float64)
	selected = numpy.flatnonzero(slots >= 0)
	for q in range(pair_chunks.shape[0]):
		base = pair_bases[q]
		start = pair_starts[q]
		row = pair_rows[q]
		a = max(start, base)
		b = min(start + width, base + tile)
		chunk = pair_chunks[q]
		if b <= a or chunk < 0:
			continue

		k_out = _window_of_chunk(tile, data, positions[chunk], _kind_of(kinds[
			chunk]), _code_in(kinds[chunk]), n_tracks, slots, selected, base,
			start, a, b, no_channels, False, fill, row, k_out, out_rows,
			out_channels, out_offsets, out_values)

	return k_out


@numba.njit(nogil=True, cache=True)
def _block_window(view, values, ones, n, start, end, row, slot, dst,
	accumulate, list_entries, fill, k_out, out_rows, out_channels, out_offsets,
	out_values):
	"""One window's entries of one block, whose positions are `view` and
	values the typed view `values`; see `_read_blocks`."""

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
		v = _value(values, i, ones)
		if list_entries:
			if fill:
				out_rows[k_out] = row
				out_channels[k_out] = slot
				out_offsets[k_out] = at - start
				out_values[k_out] = v
			k_out += 1
		elif accumulate:
			dst[at - start] += v
		else:
			dst[at - start] = v
	return k_out


@numba.njit(nogil=True, cache=True)
def _block_into(code, view, data, v0, n, start, end, row, slot, dst, adding,
	list_entries, fill, k_out, out_rows, out_channels, out_offsets, out_values):
	"""`_block_window` with the block's values as the typed view its element
	code says."""

	if code == _U8:
		return _block_window(view, data[v0:v0 + n], False, n, start, end, row,
			slot, dst, adding, list_entries, fill, k_out, out_rows, out_channels,
			out_offsets, out_values)
	elif code == _U16:
		return _block_window(view, data[v0:v0 + 2 * n].view(numpy.uint16), False,
			n, start, end, row, slot, dst, adding, list_entries, fill, k_out,
			out_rows, out_channels, out_offsets, out_values)
	elif code == _U32:
		return _block_window(view, data[v0:v0 + 4 * n].view(numpy.uint32), False,
			n, start, end, row, slot, dst, adding, list_entries, fill, k_out,
			out_rows, out_channels, out_offsets, out_values)
	elif code == _I8:
		return _block_window(view, data[v0:v0 + n].view(numpy.int8), False, n,
			start, end, row, slot, dst, adding, list_entries, fill, k_out,
			out_rows, out_channels, out_offsets, out_values)
	elif code == _I16:
		return _block_window(view, data[v0:v0 + 2 * n].view(numpy.int16), False,
			n, start, end, row, slot, dst, adding, list_entries, fill, k_out,
			out_rows, out_channels, out_offsets, out_values)
	elif code == _I32:
		return _block_window(view, data[v0:v0 + 4 * n].view(numpy.int32), False,
			n, start, end, row, slot, dst, adding, list_entries, fill, k_out,
			out_rows, out_channels, out_offsets, out_values)
	elif code == _F32:
		return _block_window(view, data[v0:v0 + 4 * n].view(numpy.float32),
			False, n, start, end, row, slot, dst, adding, list_entries, fill,
			k_out, out_rows, out_channels, out_offsets, out_values)
	elif code == _F64:
		return _block_window(view, data[v0:v0 + 8 * n].view(numpy.float64),
			False, n, start, end, row, slot, dst, adding, list_entries, fill,
			k_out, out_rows, out_channels, out_offsets, out_values)
	return _block_window(view, data[v0:v0], True, n, start, end, row, slot, dst,
		adding, list_entries, fill, k_out, out_rows, out_channels, out_offsets,
		out_values)


@numba.njit(nogil=True, cache=True)
def _read_blocks(data, positions, kinds, n_entries, pair_blocks, pair_slots,
	pair_starts, pair_rows, width, out, accumulate, list_entries, fill,
	out_rows, out_channels, out_offsets, out_values, scratch, bins):
	"""Windows out of a batch of track-order blocks.

	Pair q is a window and one block of one track that may hold its bases:
	block pair_blocks[q] of the batch, whose bytes start at
	positions[block] of `data` with n_entries[block] entries, for the window
	at pair_starts[q] of `width` bases, going to row pair_rows[q] and channel
	pair_slots[q] of `out`. The block's positions ascend, so the window's
	entries are found by binary search. Without `list_entries` the values
	are set into `out`, or added with `accumulate`; with it they are counted
	as entries, and written to the output arrays when `fill`. With `bins`
	above 1 the window's bases are summed per bin of `bins` bases into
	`out`, through `scratch`, a float64 row of `width`. `out` is zero on
	entry in every mode. Returns the number of entries.
	"""

	k_out = 0
	for q in range(pair_blocks.shape[0]):
		block = pair_blocks[q]
		n = n_entries[block]
		code = _code_in(kinds[block])
		position = positions[block]
		start = pair_starts[q]
		end = start + width
		row, slot = pair_rows[q], pair_slots[q]
		view = data[position:position + 4 * n].view(numpy.uint32)
		v0 = _pad8(position + 4 * n)
		if list_entries:
			k_out = _block_into(code, view, data, v0, n, start, end, row, slot,
				out[0, 0], False, True, fill, k_out, out_rows, out_channels,
				out_offsets, out_values)
		elif bins > 1:
			scratch[:] = 0
			_block_into(code, view, data, v0, n, start, end, row, slot, scratch,
				True, False, fill, k_out, out_rows, out_channels, out_offsets,
				out_values)
			_bin_into(out[row, slot], scratch, 0, 0, width, bins)
		else:
			_block_into(code, view, data, v0, n, start, end, row, slot, out[row,
				slot], accumulate, False, fill, k_out, out_rows, out_channels,
				out_offsets, out_values)

	return k_out
