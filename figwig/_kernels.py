# _kernels.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The compiled half of figwig: inflating data blocks and decoding windows.

Every kernel here is compiled by numba with `nogil=True`, so that several
threads can inflate and decode at once. zlib's own `uncompress()` is loaded
through ctypes and called from inside a kernel, because `zlib.decompress`
holds the GIL between blocks.
"""

import os
import threading

import numba
import numpy


# zlib's uncompress() through ctypes, loaded on the first read.
_ZLIB_UNCOMPRESS = []

# zlib's compress2() through ctypes, loaded on the first write.
_ZLIB_COMPRESS = []

# libdeflate's compressor functions through ctypes, from the `deflate`
# package, loaded on the first write that uses them.
_LIBDEFLATE = []

# The names zlib's shared library goes by on Linux, macOS and Windows, tried
# in this order before ctypes.util.find_library is asked for 'z' and 'zlib'.
_ZLIB_NAMES = ('libz.so.1', 'libz.1.dylib', 'libz.dylib', 'zlib1.dll',
	'zlib.dll')

# Where there is no pread, as on Windows, a read seeks the shared file
# descriptor and then reads from it, and this lock keeps another thread from
# moving the file position in between.
_SEEK_LOCK = threading.Lock()


def _zlib_uncompress():
	"""zlib's uncompress() as a ctypes function, or None if it cannot be loaded.

	`_inflate_blocks` calls it without the GIL. The zlib module links the
	same library on Linux, where it is already loaded under this name. A
	Python whose zlib is built into the interpreter, as on Windows, may have
	no zlib library to load. Without it, every block is inflated by
	`zlib.decompress`, with the same result.
	"""

	if len(_ZLIB_UNCOMPRESS) == 0:
		_ZLIB_UNCOMPRESS.append(_load_zlib_uncompress())

	return _ZLIB_UNCOMPRESS[0]


def _load_zlib_uncompress():
	"""(uncompress, the dtype of a C unsigned long), or None."""

	try:
		import ctypes
	except ImportError:
		return None

	function = _zlib_function('uncompress')
	if function is None:
		return None

	function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
		ctypes.c_ulong]
	function.restype = ctypes.c_int
	return function, numpy.dtype(ctypes.c_ulong)


def _zlib_function(symbol):
	"""The function `symbol` from zlib's shared library, through ctypes, from
	the first of its names that loads and has it, or None."""

	import ctypes
	import ctypes.util

	for name in _ZLIB_NAMES + ('z', 'zlib'):
		try:
			if name in ('z', 'zlib'):
				name = ctypes.util.find_library(name)
				if name is None:
					continue

			return getattr(ctypes.CDLL(name), symbol)
		except (OSError, AttributeError):
			continue

	return None


def _zlib_compress():
	"""zlib's compress2() as a ctypes function, or None if it cannot be loaded.

	`_deflate_blocks_zlib` calls it without the GIL. Where it cannot be
	loaded, blocks are compressed by `zlib.compress`, with the same bytes,
	since both are zlib.
	"""

	if len(_ZLIB_COMPRESS) == 0:
		_ZLIB_COMPRESS.append(_load_zlib_compress())

	return _ZLIB_COMPRESS[0]


def _load_zlib_compress():
	"""(compress2, the dtype of a C unsigned long), or None."""

	try:
		import ctypes
	except ImportError:
		return None

	function = _zlib_function('compress2')
	if function is None:
		return None

	function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
		ctypes.c_ulong, ctypes.c_int]
	function.restype = ctypes.c_int
	return function, numpy.dtype(ctypes.c_ulong)


def _libdeflate():
	"""libdeflate's (alloc_compressor, zlib_compress, free_compressor) as
	ctypes functions, or None if the `deflate` package is not installed or
	its library does not export them.

	The `deflate` package links libdeflate into its extension module, and
	exports libdeflate's functions from it. `_deflate_blocks_libdeflate`
	calls `zlib_compress` without the GIL. A compressor is passed to it as an
	integer, which is why its first argument is a size_t rather than a
	pointer.
	"""

	if len(_LIBDEFLATE) == 0:
		_LIBDEFLATE.append(_load_libdeflate())

	return _LIBDEFLATE[0]


def _load_libdeflate():
	try:
		import ctypes
		import deflate._deflate
	except ImportError:
		return None

	try:
		library = ctypes.CDLL(deflate._deflate.__file__)
		alloc = library.libdeflate_alloc_compressor
		compress = library.libdeflate_zlib_compress
		free = library.libdeflate_free_compressor
	except (OSError, AttributeError):
		return None

	alloc.argtypes = [ctypes.c_int]
	alloc.restype = ctypes.c_size_t
	compress.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
		ctypes.c_void_p, ctypes.c_size_t]
	compress.restype = ctypes.c_size_t
	free.argtypes = [ctypes.c_size_t]
	free.restype = None
	return alloc, compress, free


@numba.njit(nogil=True, cache=True)
def _inflate_blocks(uncompress, data, starts, sizes, buffer, blocks, length):
	"""Inflate data blocks one after another into `buffer`, without the GIL.

	Block k is the zlib stream data[starts[k]:starts[k] + sizes[k]], and is
	inflated only when blocks[k, 5] is 1. Its words are then
	buffer[4 * blocks[k, 0]:4 * blocks[k, 1]]. A block that inflates to no
	bytes or to bytes that are not whole 32-bit words gets blocks[k, 5] = 0,
	as it would from zlib.decompress. A block that uncompress() cannot
	inflate into the rest of `buffer` gets 2, and is left to zlib.decompress.
	`length` is a one-element array of C unsigned longs, which are 32 bits on
	Windows, so the room offered for a block is at most 2**32 - 1 bytes.
	Returns the number of bytes of `buffer` that were used.
	"""

	position = 0
	for k in range(starts.shape[0]):
		if blocks[k, 5] != 1:
			continue

		length[0] = min(buffer.shape[0] - position, 2 ** 32 - 1)
		source = data[starts[k]:starts[k] + sizes[k]]
		status = uncompress(buffer[position:].ctypes, length.ctypes,
			source.ctypes, sizes[k])
		n = numpy.int64(length[0])

		if status != 0:
			blocks[k, 5] = 2
		elif n == 0 or n % 4 != 0:
			blocks[k, 5] = 0
		else:
			blocks[k, 0] = position // 4
			blocks[k, 1] = (position + n) // 4
			position += n

	return position


@numba.njit(nogil=True, cache=True)
def _deflate_blocks_zlib(compress2, level, data, bounds, out, sizes, length):
	"""Compress blocks one after another into `out` with zlib's compress2(),
	without the GIL.

	Block k is data[bounds[k]:bounds[k + 1]]. Each block's zlib stream is
	written straight after the one before it, so the streams of a run of
	blocks end up back to back in `out` and can be written to the file as
	they are; sizes[k] is the length of block k's. compress2() is offered all
	of `out` that is left, which is enough as long as `out` holds a bound on
	every block's stream. `length` is a one-element array of C unsigned
	longs, which are 32 bits on Windows. Returns -1, or the first block that
	compress2() failed on.
	"""

	position = 0
	for k in range(sizes.shape[0]):
		length[0] = min(out.shape[0] - position, 2 ** 32 - 1)
		source = data[bounds[k]:bounds[k + 1]]
		status = compress2(out[position:].ctypes, length.ctypes, source.ctypes,
			bounds[k + 1] - bounds[k], level)
		if status != 0:
			return k

		sizes[k] = length[0]
		position += sizes[k]

	return -1


@numba.njit(nogil=True, cache=True)
def _deflate_blocks_libdeflate(compress, compressor, data, bounds, out, sizes):
	"""Compress blocks one after another into `out` with libdeflate's
	zlib_compress(), without the GIL.

	The layout of `data`, `bounds`, `out` and `sizes` is that of
	`_deflate_blocks_zlib`. `compressor` is a libdeflate compressor, which
	holds the compression level and can be used by one thread at a time.
	Returns -1, or the first block that did not fit in what was left of
	`out`.
	"""

	position = 0
	for k in range(sizes.shape[0]):
		source = data[bounds[k]:bounds[k + 1]]
		n = compress(compressor, source.ctypes, bounds[k + 1] - bounds[k],
			out[position:].ctypes, out.shape[0] - position)
		if n == 0:
			return k

		sizes[k] = n
		position += n

	return -1


@numba.njit(nogil=True, cache=True)
def _update_summary(spans, values, counts, stats):
	"""Fold items, in file order, into a bigWig's total summary.

	Item i covers spans[i] bases with the float32 values[i]. `counts` holds
	the bases covered and the number of items, `stats` the minimum, maximum,
	sum and sum of squares, as float64. The sums are added the way libBigWig
	(pyBigWig's writer) adds them, so that they agree with pyBigWig's to the
	last bit: the product of a span and a value is rounded to float32 before
	it is added to the sum, and the sum of squares adds span * value**2 in
	float64. The minimum and maximum are those of the values. libBigWig
	compares a value with its maximum only when it is not a new minimum, and
	starts its maximum at the smallest positive double, so its maximum misses
	a first value that is the largest, and stays at that double when no
	value is positive.
	"""

	for i in range(values.shape[0]):
		value = values[i]
		span = spans[i]
		as_double = numpy.float64(value)

		if as_double < stats[0]:
			stats[0] = as_double
		if as_double > stats[1]:
			stats[1] = as_double

		counts[0] += span
		counts[1] += 1
		stats[2] += numpy.float64(numpy.float32(span) * value)
		stats[3] += numpy.float64(span) * (as_double * as_double)


@numba.njit(nogil=True, cache=True)
def _decode_items(words, bounds, tids, starts, ends, values):
	"""The items of decompressed data blocks, in file order.

	Block b is the 32-bit words words[bounds[b]:bounds[b + 1]], one or more
	sections of a bigWig written by BigWigWriter. Item i is written to
	tids[i], starts[i], ends[i] and values[i]. Returns the number of items.
	"""

	floats = words.view(numpy.float32)
	n = 0
	for b in range(bounds.shape[0] - 1):
		offset = bounds[b]
		while offset < bounds[b + 1]:
			tid = numpy.int64(words[offset])
			section_start = numpy.int64(words[offset + 1])
			step = numpy.int64(words[offset + 3])
			span = numpy.int64(words[offset + 4])
			kind = words[offset + 5] & 0xFF
			count = numpy.int64(words[offset + 5] >> 16)
			item = offset + 6

			for i in range(count):
				if kind == 1:
					start = numpy.int64(words[item + 3 * i])
					end = numpy.int64(words[item + 3 * i + 1])
					value = floats[item + 3 * i + 2]
				elif kind == 2:
					start = numpy.int64(words[item + 2 * i])
					end = start + span
					value = floats[item + 2 * i + 1]
				else:
					start = section_start + i * step
					end = start + span
					value = floats[item + i]

				tids[n] = tid
				starts[n] = start
				ends[n] = end
				values[n] = value
				n += 1

			offset = item + count * (3 if kind == 1 else 2 if kind == 2 else 1)

	return n


@numba.njit(nogil=True, cache=True)
def _window_items(block, bits, firsts, tids, missing, by_row, segment_bases,
	block_bytes, capacity, starts, ends, values, parts):
	"""Windows' values as the items of data sections, each segment of them
	as the section type that takes it in the fewest bytes.

	Row r of `block` holds the values of a window from position firsts[r] on
	chromosome tids[r], and `bits` holds the same values as uint32. With
	`by_row`, each row is a segment; otherwise there is one row, and each
	`segment_bases` columns of it are one. A base is written when it is
	neither NaN nor `missing`.

	A segment's written bases form runs of consecutive bases and, within
	runs, stretches of one value, bit for bit. A data block holds
	capacity[kind] items, of 4 bytes a word, and takes `block_bytes` more:
	fixedStep (3) takes a word a base and a block a run, varStep (2) two
	words a base, and bedGraph (1) three words a stretch. A segment is laid
	out as the type with the fewest bytes, fixedStep first in a tie, then
	varStep.

	Item i is written to starts[i], ends[i] and values[i]: a base, or a
	stretch in bedGraph. Row p of `parts` is (first item, section type,
	chromosome), and a part starts where the type or chromosome changes, or
	at a gap in fixedStep, so that each part is one run of
	`BigWigWriter._append`. Returns the number of items and of parts.
	"""

	missing_is_nan = missing != missing
	width = block.shape[1]
	step = width if by_row else segment_bases
	n_items, n_parts = 0, 0
	part_kind, part_tid, part_end = -1, -1, -1

	for r in range(block.shape[0]):
		tid = tids[r]
		for c0 in range(0, width, step):
			c1 = min(width, c0 + step)

			n_bases, n_runs, n_stretches = 0, 0, 0
			kept, last = False, numpy.uint32(0)
			for c in range(c0, c1):
				value = block[r, c]
				if value != value or (not missing_is_nan and value == missing):
					kept = False
					continue

				n_bases += 1
				if not kept:
					n_runs += 1
					n_stretches += 1
				elif bits[r, c] != last:
					n_stretches += 1

				kept, last = True, bits[r, c]

			if n_bases == 0:
				continue

			fixed = 4 * n_bases + block_bytes * (n_runs + n_bases / capacity[3])
			var = 8 * n_bases + block_bytes * n_bases / capacity[2]
			bed = 12 * n_stretches + block_bytes * n_stretches / capacity[1]
			kind = 3 if fixed <= var and fixed <= bed else 2 if var <= bed else 1

			kept = False
			for c in range(c0, c1):
				value = block[r, c]
				if value != value or (not missing_is_nan and value == missing):
					kept = False
					continue

				position = firsts[r] + c
				if kind == 1 and kept and bits[r, c] == last:
					ends[n_items - 1] = position + 1
					continue

				if kind != part_kind or tid != part_tid or (kind == 3 and
						position != part_end):
					parts[n_parts, 0] = n_items
					parts[n_parts, 1] = kind
					parts[n_parts, 2] = tid
					n_parts += 1
					part_kind, part_tid = kind, tid

				starts[n_items] = position
				ends[n_items] = position + 1
				values[n_items] = value
				n_items += 1
				part_end = position + 1
				kept, last = True, bits[r, c]

	return n_items, n_parts


@numba.njit(nogil=True, cache=True)
def _zoom_records(size, tids, starts, ends, values, progress, open_ints,
	open_floats, out_ints, out_floats):
	"""Summarize items into the records of one zoom level, as libBigWig bins
	them.

	A record summarizes the bases from its start to at most `size` bases
	later. Items are taken in file order. The part of an item that falls
	within the open record's `size` bases, on its chromosome, is added to it;
	anything else starts a new record at the first base not yet summarized.
	A zoom block holds 1023 records, and once the open record is the 1023rd
	of its block, the next part of an item starts a new record, in a new
	block, even where it could have been added. That is how libBigWig
	(pyBigWig's writer) lays zoom levels out. Unlike libBigWig, every record's
	sum and sum of squares are kept, including those of the last record of
	a block, which libBigWig leaves at 0.

	The open record carries over between calls, in `open_ints`: whether there
	is one, its place in its block, chromosome, start, end and number of
	bases; and in `open_floats`: its minimum, maximum, sum and sum of squares
	in float64. The sum adds each part's float32 product of length and value,
	as libBigWig does. Each record that is closed is written to the next row
	of `out_ints` (chromosome, start, end, bases) and `out_floats` (minimum,
	maximum, sum, sum of squares, as float32). Returns the number of records
	written.

	When `out_ints` is full and another record closes, the kernel stops before
	changing anything, so that the output buffer can be any size. `progress`
	says where to start and, on return, where to go on: the item, and the
	first base of it not yet summarized, or -1 for its start. progress[0] is
	the number of items once every item is summarized.
	"""

	n = 0
	i = progress[0]
	while i < values.shape[0]:
		tid = tids[i]
		start = starts[i] if progress[1] < 0 else progress[1]
		end = ends[i]
		value = values[i]
		as_double = numpy.float64(value)

		while start < end:
			reach = size
			if start + reach > 0xFFFFFFFF:
				reach = 0xFFFFFFFF - start

			if (open_ints[0] == 1 and open_ints[1] < 1023 and open_ints[2] == tid
					and start < open_ints[3] + reach):
				length = min(open_ints[3] + reach, end) - start
				open_ints[4] = start + length
				open_ints[5] += length
				if open_floats[0] > as_double:
					open_floats[0] = as_double
				if open_floats[1] < as_double:
					open_floats[1] = as_double

				open_floats[2] += numpy.float64(numpy.float32(length) * value)
				open_floats[3] += numpy.float64(length) * (as_double * as_double)
				start += length
				continue

			if open_ints[0] == 1:
				if n == out_ints.shape[0]:
					progress[0] = i
					progress[1] = start
					return n

				out_ints[n, 0] = open_ints[2]
				out_ints[n, 1] = open_ints[3]
				out_ints[n, 2] = open_ints[4]
				out_ints[n, 3] = open_ints[5]
				out_floats[n, 0] = open_floats[0]
				out_floats[n, 1] = open_floats[1]
				out_floats[n, 2] = open_floats[2]
				out_floats[n, 3] = open_floats[3]
				n += 1

			if open_ints[1] == 1023:
				open_ints[1] = 0

			length = min(reach, end - start)
			open_ints[0] = 1
			open_ints[1] += 1
			open_ints[2] = tid
			open_ints[3] = start
			open_ints[4] = start + length
			open_ints[5] = length
			open_floats[0] = as_double
			open_floats[1] = as_double
			open_floats[2] = numpy.float64(numpy.float32(length) * value)
			open_floats[3] = numpy.float64(length) * (as_double * as_double)
			start += length

		progress[1] = -1
		i += 1

	progress[0] = i
	return n


def _pread_into(fd, buffer, offset):
	"""Read into the uint8 array `buffer` from `offset`; return the bytes read.

	A read may return fewer bytes than asked for, and Linux returns at most
	about 2 GiB from one, so this reads until `buffer` is full or the file
	ends. preadv reads straight into `buffer`, pread into a copy, and where
	neither exists, as on Windows, the file is seeked and read under
	`_SEEK_LOCK`, since its position is shared by every thread.
	"""

	total = 0
	while total < len(buffer):
		if hasattr(os, 'preadv'):
			n = os.preadv(fd, [buffer[total:]], offset + total)
		else:
			if hasattr(os, 'pread'):
				data = os.pread(fd, len(buffer) - total, offset + total)
			else:
				with _SEEK_LOCK:
					os.lseek(fd, offset + total, os.SEEK_SET)
					data = os.read(fd, len(buffer) - total)

			n = len(data)
			buffer[total:total + n] = numpy.frombuffer(data, dtype=numpy.uint8)

		if n == 0:
			break

		total += n

	return total


def _pread(fd, n, offset):
	"""Up to `n` bytes from `offset`, fewer only where the file ends."""

	buffer = numpy.empty(n, dtype=numpy.uint8)
	return buffer[:_pread_into(fd, buffer, offset)].tobytes()


@numba.njit(nogil=True, cache=True)
def _check_block(words, begin, end, chrom, base_start, base_end):
	"""Whether one decompressed data block can be read by `_read_windows`.

	The block is the 32-bit words `words[begin:end]`. Each section is a
	24-byte header (chromosome id, start, end, step, span, then the type and
	the item count) followed by its items. The block must hold whole
	bedGraph, varStep or fixedStep sections on the chromosome `chrom` of its
	index entry, whose items are sorted, do not overlap one another, and lie
	inside the entry's range [base_start, base_end). Overlapping items have
	no single value per base, and readers disagree on them: pybigtools sums
	them.

	Returns whether the block passes, the start of its first item and the
	end of its last, which `_read_windows` uses to check that items in
	neighbouring blocks do not overlap either. A block without items gives
	the largest int64 as its first start and `base_start` as its last end.
	"""

	offset = begin
	previous_end = base_start
	first_start = numpy.iinfo(numpy.int64).max
	while offset < end:
		if offset + 6 > end or words[offset] != chrom:
			return False, first_start, previous_end

		section_start = numpy.int64(words[offset + 1])
		step = numpy.int64(words[offset + 3])
		span = numpy.int64(words[offset + 4])
		kind = words[offset + 5] & 0xFF
		count = numpy.int64(words[offset + 5] >> 16)

		if kind == 1:
			size = 3
		elif kind == 2:
			size = 2
		elif kind == 3:
			size = 1
		else:
			return False, first_start, previous_end

		item = offset + 6
		next_offset = item + count * size
		if next_offset > end:
			return False, first_start, previous_end

		for i in range(count):
			if kind == 1:
				item_start = numpy.int64(words[item + 3 * i])
				item_end = numpy.int64(words[item + 3 * i + 1])
			elif kind == 2:
				item_start = numpy.int64(words[item + 2 * i])
				item_end = item_start + span
			else:
				item_start = section_start + i * step
				item_end = item_start + span

			if item_start < previous_end or item_end < item_start:
				return False, first_start, previous_end

			first_start = min(first_start, item_start)
			previous_end = item_end

		if previous_end > base_end:
			return False, first_start, previous_end

		offset = next_offset

	return True, first_start, previous_end


@numba.njit(nogil=True, cache=True)
def _read_windows(words, blocks, windows, out, signal, failed, missing):
	"""Write the per-base values of bigWig windows into rows of `out`.

	`words` holds decompressed data blocks, one after another, as 32-bit
	words. Row b of `blocks` describes block b: its first and last word, the
	chromosome, start and end of its index entry, and whether it was
	decompressed. Row j of `windows` describes window j: the blocks [lo, hi)
	that overlap it, its start, the length of its chromosome and the row of
	`out` it is written to, in channel `signal`. The width of every window
	is out.shape[2].

	Each base of a window is given the value of the item that covers it,
	`missing`, a float32, when no item does, and NaN past the end of the
	chromosome. Items with a NaN value are skipped, so their bases are
	`missing`. A window is not written, and is marked in `failed`, when one
	of its blocks fails `_check_block` or has an item that overlaps an item
	of an earlier block. A window's blocks are consecutive in the index, so
	comparing each block's first item with the furthest end of the blocks
	before it finds every overlap.
	"""

	values = words.view(numpy.float32)
	width = out.shape[2]
	nan = numpy.float32(numpy.nan)

	n_blocks = blocks.shape[0]
	good = numpy.zeros(n_blocks, dtype=numpy.bool_)
	first = numpy.zeros(n_blocks, dtype=numpy.int64)
	last = numpy.zeros(n_blocks, dtype=numpy.int64)
	for b in range(n_blocks):
		if blocks[b, 5] != 0:
			good[b], first[b], last[b] = _check_block(words, blocks[b, 0],
				blocks[b, 1], blocks[b, 2], blocks[b, 3], blocks[b, 4])

	for j in range(windows.shape[0]):
		lo, hi = windows[j, 0], windows[j, 1]
		start, row = windows[j, 2], windows[j, 4]

		usable = True
		reach = numpy.iinfo(numpy.int64).min
		for b in range(lo, hi):
			if not good[b] or first[b] < reach:
				usable = False

			reach = max(reach, last[b])

		if not usable:
			failed[j] = True
			continue

		end = min(start + width, max(start, windows[j, 3]))
		for p in range(end - start):
			out[row, signal, p] = missing
		for p in range(end - start, width):
			out[row, signal, p] = nan

		for b in range(lo, hi):
			offset = blocks[b, 0]
			while offset < blocks[b, 1]:
				section_start = numpy.int64(words[offset + 1])
				step = numpy.int64(words[offset + 3])
				span = numpy.int64(words[offset + 4])
				kind = words[offset + 5] & 0xFF
				count = numpy.int64(words[offset + 5] >> 16)
				item = offset + 6

				if kind == 1:
					size = 3
				elif kind == 2:
					size = 2
				else:
					size = 1

				# The first item that ends after the window starts. Items are
				# sorted and do not overlap, so their ends are sorted too.
				if kind == 3:
					i = 0
					if step > 0 and start > section_start + span:
						i = (start - section_start - span) // step + 1
				else:
					i, k = 0, count
					while i < k:
						m = (i + k) // 2
						if kind == 1:
							item_end = numpy.int64(words[item + 3 * m + 1])
						else:
							item_end = numpy.int64(words[item + 2 * m]) + span

						if item_end <= start:
							i = m + 1
						else:
							k = m

				while i < count:
					if kind == 1:
						item_start = numpy.int64(words[item + 3 * i])
						item_end = numpy.int64(words[item + 3 * i + 1])
						value = values[item + 3 * i + 2]
					elif kind == 2:
						item_start = numpy.int64(words[item + 2 * i])
						item_end = item_start + span
						value = values[item + 2 * i + 1]
					else:
						item_start = section_start + i * step
						item_end = item_start + span
						value = values[item + i]

					if item_start >= end:
						break

					if not numpy.isnan(value):
						for p in range(max(item_start, start) - start,
								min(item_end, end) - start):
							out[row, signal, p] = value

					i += 1

				offset = item + count * size
