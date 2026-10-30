# _indexed.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Coverage (`-c`) of coordinate-sorted, indexed BAM files, region by
region on a pool of threads, for `figwig bam2bw` and `bam2fw`.

The whole-file reader in bam2bw.py keeps two keys per aligned block of
every read until the file ends, which is tens of GB for a 30x genome, and
walks the file on one thread. Here each `_REGION` bases of a chromosome are
read on their own: from the offset the BAM index gives for the region's
first base, through bam2bw's own BGZF and record kernels, until the reads
start past the region. The depth over the region is exact, since every
read that covers a base of it is read, and the runs it gives are those of
`_coverage_runs` cut at the region's edges, which are joined again. A file
the record kernel would send to pysam raises `_Fallback`, and the caller
reads it as before.
"""

import os
import struct
import threading
import collections
import concurrent.futures

import numpy

from .bam2bw import _BGZF_SPARE
from .bam2bw import _BadBGZFBlock
from .bam2bw import _bam_kernel
from .bam2bw import _bgzf_kernel
from .bam2bw import _KEY_SHIFT
from .bam2bw import _inflate_bgzf
from .bam2bw import _inflate_engine


# A region's most bases, a multiple of the 16 kb index windows, and the
# BAM bytes at which a region is cut sooner. A region in flight holds an
# int32 of depth per base; one read as runs holds its keys, which the cut
# by bytes bounds, and may be longer.
_REGION = 1 << 22
_RUNS_REGION = 1 << 25
_REGION_BYTES = 16 << 20

# A region's depth with fewer changes of value than one per this many bases
# goes to the writer as runs, which cost it less than every base.
_RUN_SPACING = 256

# The bytes a region reads at a time, and the BGZF blocks its records are
# walked by. Each walk holds the GIL for its calls, and each reading thread
# keeps buffers of these sizes.
_REGION_READ = 1 << 23
_REGION_BLOCKS = 128

# The regions in flight per reading thread.
_AHEAD = 1

# The writer's share of the cores while regions are read: one in this many,
# at least one, the regions taking the rest.
_WRITER_SHARE = 4

# The linear index of a .bai has one offset per 2**14 bases.
_WINDOW_SHIFT = 14

# The pseudo-bin of a .bai that holds a reference's span of the file and its
# mapped and unmapped reads, and the first bin of 16 kb.
_PSEUDO_BIN = 37450
_LEAF_BIN = 4681


# Each pool thread's buffers, kept from one region to the next: allocating
# them per region cost the kernel's page faults as much time as the reads.
_BUFFERS = threading.local()


def _buffer(name, size, dtype='uint8'):
	"""This thread's buffer `name`, of at least `size` elements."""

	array = getattr(_BUFFERS, name, None)
	if array is None or len(array) < size:
		array = numpy.empty(max(size, 2 * (len(array) if array is not None else
			0)), dtype=dtype)
		setattr(_BUFFERS, name, array)
	return array


class _Fallback(Exception):
	"""A file this reader does not take as the whole-file reader would."""


def _index_path(filename):
	"""The .bai of a BAM file, when one exists and is not older than it."""

	candidates = [filename + '.bai']
	if filename.endswith('.bam'):
		candidates.append(filename[:-4] + '.bai')
	for path in candidates:
		if os.path.isfile(path) and os.path.getmtime(path) >= os.path.getmtime(
				filename):
			return path
	return None


def _read_index(path, n_ref):
	"""Each reference's linear index (the virtual offset of the first read
	that covers each 16 kb window, 0 for none), the virtual offset its
	reads start at, its mapped reads, per bin the virtual offset its last
	chunk ends at (0 for an empty bin, and for every bin past the array's
	end) and, per 16 kb window, the offset of the first read of a 16 kb bin
	at or after it (int64's maximum for none), from a .bai; None for a file
	this parser does not take, which includes one cut short."""

	with open(path, 'rb') as handle:
		data = handle.read()
	try:
		return _parse_index(data, n_ref)
	except (struct.error, ValueError):
		return None


def _parse_index(data, n_ref):
	"""`_read_index` of the bytes of a .bai."""

	if data[:4] != b'BAI\x01' or struct.unpack_from('<i', data, 4)[0] != n_ref:
		return None

	at = 8
	refs = []
	for _ in range(n_ref):
		n_bin, = struct.unpack_from('<i', data, at)
		at += 4
		first, mapped = None, None
		bins, counts, pieces = [], [], []
		for _ in range(n_bin):
			bin_id, n_chunk = struct.unpack_from('<Ii', data, at)
			at += 8
			if n_chunk < 0:
				return None
			if bin_id == _PSEUDO_BIN:
				if n_chunk != 2:
					return None
				first, _, mapped = struct.unpack_from('<QQQ', data, at)
			elif n_chunk > 0:
				if bin_id >= _PSEUDO_BIN:
					return None
				bins.append(bin_id)
				counts.append(n_chunk)
				pieces.append(data[at:at + 16 * n_chunk])
			at += 16 * n_chunk
		n_intv, = struct.unpack_from('<i', data, at)
		at += 4
		if n_intv < 0:
			return None
		# htslib writes no pseudo-bin for a reference without reads.
		if n_bin == 0:
			mapped = 0
		linear = numpy.frombuffer(data, dtype='<u8', count=n_intv, offset=at
			).astype(numpy.uint64)
		at += 8 * n_intv

		ids = numpy.array(bins, dtype=numpy.int64)
		chunks = numpy.frombuffer(b''.join(pieces), dtype='<u8').reshape(-1, 2)
		if len(chunks) and int(chunks.max()) >> 63:
			return None
		cuts = numpy.zeros(len(bins), dtype=numpy.int64)
		numpy.cumsum(counts[:-1], out=cuts[1:])
		if len(bins):
			begins = numpy.minimum.reduceat(chunks[:, 0], cuts).astype(numpy.int64)
			ends = numpy.maximum.reduceat(chunks[:, 1], cuts)
		else:
			begins = ends = numpy.empty(0, dtype=numpy.int64)
		if first is None and len(bins):
			first = int(begins.min())
		last = numpy.zeros(int(ids.max()) + 1 if len(bins) else 1,
			dtype=numpy.uint64)
		last[ids] = ends
		# The first read in a 16 kb bin at or after each window: every read
		# that starts before the window lies before it in a sorted file.
		leaves = ids >= _LEAF_BIN
		windows = max(n_intv, int(ids[leaves].max()) - _LEAF_BIN + 1 if
			leaves.any() else 0)
		after = numpy.full(windows + 1, numpy.iinfo(numpy.int64).max,
			dtype=numpy.int64)
		after[ids[leaves] - _LEAF_BIN] = begins[leaves]
		after = numpy.minimum.accumulate(after[::-1])[::-1]
		refs.append((linear, first, mapped, last, after))
	return refs


def _regions(files, chrom, size, longest=None):
	"""The [beg, end) regions of a chromosome: cut where the files' BAM
	bytes since the region's start, from their linear indexes, pass
	`_REGION_BYTES`, or its length `longest` (`_REGION` by default), on 16 kb
	windows, so that a stretch of many reads is cut finer than one of few."""

	n_windows = max((size + (1 << _WINDOW_SHIFT) - 1) >> _WINDOW_SHIFT, 1)
	offsets = numpy.zeros(n_windows + 1, dtype=numpy.float64)
	for f in files:
		tid = f.tids.get(chrom)
		if tid is None:
			continue
		linear = f.index[tid][0].astype(numpy.int64) >> 16
		if len(linear) == 0:
			continue
		# Empty windows take the offset of the window before them.
		filled = numpy.maximum.accumulate(linear[:n_windows])
		at = numpy.zeros(n_windows, dtype=numpy.int64)
		at[:len(filled)] = filled
		if len(filled) < n_windows:
			at[len(filled):] = filled[-1]
		offsets[:n_windows] += at - at[0]
		offsets[n_windows] += at[-1] - at[0]
	bounds = [0]
	start = 0
	window = (longest or _REGION) >> _WINDOW_SHIFT
	for w in range(1, n_windows):
		if w - start >= window or offsets[w] - offsets[start] >= _REGION_BYTES:
			bounds.append(w << _WINDOW_SHIFT)
			start = w
	bounds.append(max(size, 1))
	return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _start_offset(linear, first, beg):
	"""The virtual offset to read a region from: the linear index's offset
	for the window of `beg`, or the nearest earlier one where it holds 0,
	or the reference's first read."""

	w = max(beg, 0) >> _WINDOW_SHIFT
	if w >= len(linear):
		w = len(linear) - 1
	nonzero = numpy.flatnonzero(linear[:w + 1])
	if len(nonzero) == 0:
		return first
	return int(linear[nonzero[-1]])


def _end_offset(last, beg, end):
	"""The virtual offset after the last read in a bin that overlaps [beg,
	end), the reads that cover a base of it being in those bins (reg2bins of
	the SAM specification); 0 when none holds a read."""

	beg, end = max(beg, 0), max(end, beg + 1) - 1
	ends = [last[0]]
	for offset, shift in ((1, 26), (9, 23), (73, 20), (585, 17), (4681, 14)):
		lo = offset + (beg >> shift)
		hi = min(offset + (end >> shift) + 1, len(last))
		if hi > lo:
			ends.append(last[lo:hi].max())
	return int(max(ends))


class _File:
	"""One BAM file's references, index and the arrays the record kernel
	takes, with the chromosome read mapped to bucket 0."""

	def __init__(self, filename, chrom_sizes, args):
		import pysam

		# Before pysam opens the BAM, which warns of a stale index itself.
		index = _index_path(filename)
		if index is None:
			raise LookupError(filename)
		with pysam.AlignmentFile(filename, 'rb') as bam:
			self.references = [bam.get_reference_name(tid) for tid in range(
				bam.nreferences)]
			sort_order = bam.header.to_dict().get('HD', {}).get('SO')
		if sort_order != 'coordinate':
			raise LookupError(filename)
		self.index = _read_index(index, len(self.references))
		if self.index is None:
			raise LookupError(filename)
		# An index of another BAM can point past this one's end.
		size = os.path.getsize(filename)
		for linear, first, _, last, _ in self.index:
			furthest = max([first or 0, int(last.max()), int(linear.max()) if
				len(linear) else 0])
			if furthest >> 16 >= size:
				raise LookupError(filename)
		self.filename = filename
		self.tids = {name: tid for tid, name in enumerate(self.references)}
		self.options = numpy.array([args.pos_shift, args.neg_shift,
			args.three_prime, args.fragments, args.unstranded, len(
			self.references), args.coverage], dtype='int64')


def _region_keys(files, chrom, beg, end, stop):
	"""The keys of every read of `files` on `chrom` from the first that may
	cover base `beg` to the first that starts at `stop` or later, from
	`_bam_kernel` with chrom as bucket 0. The bytes read are those from the
	linear index's offset for `beg` to the end of the last chunk of a bin
	that overlaps the region, in reads of at most `_REGION_READ`, walked
	`_REGION_BLOCKS` BGZF blocks at a time."""

	kernel = _bam_kernel()
	scan = _bgzf_kernel()
	inflate = _inflate_engine()
	last_record = _last_record_kernel()
	keys = _buffer('keys', 1 << 16, 'int64')
	n_keys = 0
	capacity = (_REGION_READ + (1 << 16)) // 26 + 1
	offsets = _buffer('offsets', capacity, 'int64')
	sizes = _buffer('sizes', capacity, 'int64')
	lengths = _buffer('lengths', capacity, 'int64')
	scanned = numpy.zeros(3, dtype='int64')
	out = _buffer('out', _BGZF_SPARE + _REGION_BLOCKS * (1 << 16))

	for f in files:
		tid = f.tids.get(chrom)
		if tid is None:
			continue
		linear, first, _, last, after = f.index[tid]
		if first is None or len(linear) == 0:
			continue
		if beg > 0 and beg >> _WINDOW_SHIFT >= len(linear):
			continue
		voff = _start_offset(linear, first, beg)
		stop_offset = _end_offset(last, beg, stop)
		if stop_offset <= voff:
			continue
		# A tighter end: the first read of a 16 kb bin past `stop`.
		w = -(-min(stop, 1 << 31) >> _WINDOW_SHIFT)
		if w < len(after) - 1:
			stop_offset = min(stop_offset, int(after[w]))
		# The compressed bytes up to the end of the block that holds the end.
		remaining = (stop_offset >> 16) + (1 << 16) + 18 - (voff >> 16)

		tid_bucket = numpy.full(len(f.references) + 1, -1, dtype='int64')
		tid_bucket[tid + 1] = 0
		missing_seen = numpy.zeros(len(tid_bucket), dtype='uint8')
		missing_order = numpy.zeros(len(tid_bucket), dtype='int64')
		state = numpy.zeros(7, dtype='int64')
		state[4] = -1
		run_starts = numpy.zeros(2, dtype='int64')
		run_buckets = numpy.zeros(2, dtype='int64')
		finished = numpy.zeros(1, dtype='uint8')

		with open(f.filename, 'rb', buffering=0) as handle:
			handle.seek(voff >> 16)
			skip = voff & 0xFFFF
			rest = numpy.empty(0, dtype='uint8')
			carry = numpy.empty(0, dtype='uint8')
			stopped = False
			got = 0
			while not stopped:
				size = max(0, min(_REGION_READ, remaining))
				data = _buffer('data', len(carry) + size)[:len(carry) + size]
				data[:len(carry)] = carry
				n_read = handle.readinto(memoryview(data)[len(carry):])
				got += n_read
				remaining -= n_read
				n = len(carry) + n_read
				scan(data, n, offsets, sizes, lengths, scanned)
				count, used, status = (int(x) for x in scanned)
				if status != 0:
					raise _Fallback()
				carry = data[used:n].copy()

				for i in range(0, count, _REGION_BLOCKS):
					j = min(i + _REGION_BLOCKS, count)
					try:
						run = _inflate_bgzf(inflate, out, data, offsets[i:j].copy(),
							sizes[i:j].copy(), lengths[i:j].copy())
					except _BadBGZFBlock:
						raise _Fallback()
					a = _BGZF_SPARE + skip
					skip = 0
					if len(rest) <= a:
						buf = run[a - len(rest):]
						buf[:len(rest)] = rest
					else:
						buf = numpy.concatenate([rest, run[a:]])

					# A batch whose first record starts past the region, or on
					# a later reference, ends the scan: the file is sorted.
					if len(buf) >= 12:
						ref, pos = struct.unpack_from('<ii', buf, 4)
						if ref != tid or pos >= stop:
							stopped = True
							break

					needed = n_keys + len(buf) // 2 + 2
					if needed > len(keys):
						grown = _buffer('grown', max(needed, 2 * len(keys)), 'int64')
						grown[:n_keys] = keys[:n_keys]
						_BUFFERS.grown, _BUFFERS.keys = keys, grown
						keys = grown
					state[0] = n_keys
					consumed = kernel(buf, len(buf), tid_bucket, missing_seen,
						missing_order, f.options, keys, state, run_starts, run_buckets,
						finished)
					if state[3] != 0:
						raise _Fallback()
					n_keys = int(state[0])
					rest = buf[consumed:].copy()

					# Likewise a batch whose last whole record does, which
					# spares inflating the next batch.
					ref, pos = last_record(buf, consumed)
					if ref != -2 and (ref != tid or pos >= stop):
						stopped = True
						break

				if not stopped and (n_read == 0 or remaining <= 0):
					# The end of the file, or of the bytes the index gives:
					# whatever is left is records past the region.
					if n_read == 0 and remaining > 0 and (len(carry) > 0 or len(rest)
							> 0 or not got):
						raise _Fallback()
					break

	# A copy, since the next region on this thread writes over the buffer.
	return keys[:n_keys].copy()


def _last_record(buf, n):
	"""The reference and position of the last whole BAM record in buf[:n],
	or (-2, 0) when it holds none."""

	at = 0
	ref = -2
	pos = 0
	while at + 12 <= n:
		size = 0
		for k in range(4):
			size |= numpy.int64(buf[at + k]) << (8 * k)
		if at + 4 + size > n:
			break
		ref = 0
		pos = 0
		for k in range(4):
			ref |= numpy.int64(buf[at + 4 + k]) << (8 * k)
			pos |= numpy.int64(buf[at + 8 + k]) << (8 * k)
		if ref >= 1 << 31:
			ref -= 1 << 32
		if pos >= 1 << 31:
			pos -= 1 << 32
		at += 4 + size
	return ref, pos


def _last_record_kernel():
	"""`_last_record` compiled, on first use."""

	global _last_compiled
	if _last_compiled is None:
		import numba
		_last_compiled = numba.njit(cache=True, nogil=True)(_last_record)
	return _last_compiled


_last_compiled = None


def _runs_kernel():
	"""`_keys_to_runs` compiled, on first use."""

	global _runs_compiled
	if _runs_compiled is None:
		import numba
		_runs_compiled = numba.njit(cache=True, nogil=True)(_keys_to_runs)
	return _runs_compiled


_runs_compiled = None


def _keys_to_runs(keys, lo, hi, low, high, size, first, last, count_lo,
	count_hi, starts, ends, values, cut_before, cut_after):
	"""The runs of coverage of keys[lo:hi], sorted keys of one strand whose
	low _KEY_SHIFT bits are a position on the doubled line plus _KEY_BIAS:
	the stretches between consecutive event places whose depth is not 0,
	as `_coverage_runs` finds them, cut to [low, high) with the cut ends
	flagged and, in a chromosome's first or last region, clipped to [0,
	size). Writes them from 0 on and returns their number, the start events
	at places in [count_lo, count_hi), and the coverage clipped away."""

	mask = (numpy.int64(1) << 35) - 1
	bias = numpy.int64(1) << 33
	n = 0
	counted = 0
	lost = 0
	depth = 0
	i = lo
	while i < hi:
		position = (keys[i] & mask) - bias
		place = position >> 1
		# Every event at this place: equal keys, then the other parity.
		while i < hi and ((keys[i] & mask) - bias) >> 1 == place:
			p = (keys[i] & mask) - bias
			if p & 1:
				depth -= 1
			else:
				depth += 1
				if count_lo <= place and place < count_hi:
					counted += 1
			i += 1
		if i >= hi or depth == 0:
			continue
		next_place = ((keys[i] & mask) - bias) >> 1
		s, e = place, next_place
		before, after = s < low, e > high
		if before:
			s = low
		if after:
			e = high
		if first or last:
			cs = max(s, 0)
			ce = min(e, size)
			if s < 0 or e > size:
				lost += ((e - s) - max(ce - cs, 0)) * depth
			s, e = cs, ce
		if e <= s:
			continue
		starts[n] = s
		ends[n] = e
		values[n] = depth
		cut_before[n] = before
		cut_after[n] = after
		n += 1
	return n, counted, lost


def _dense_kernel():
	"""`_keys_to_dense` compiled, on first use."""

	global _dense_compiled
	if _dense_compiled is None:
		import numba
		_dense_compiled = numba.njit(cache=True, nogil=True)(_keys_to_dense)
	return _dense_compiled


_dense_compiled = None


def _keys_to_dense(keys, lo, hi, beg, size, first, last, count_lo, count_hi,
	out):
	"""The depth of keys[lo:hi], sorted keys of one strand as in
	`_keys_to_runs`, at each base of `out` from `beg` on, `out` being the
	region's bases inside the chromosome; the start events at places in
	[count_lo, count_hi); and, in a chromosome's first or last region, the
	coverage of the bases outside [0, size), as `_coverage_runs` counts it."""

	mask = (numpy.int64(1) << 35) - 1
	bias = numpy.int64(1) << 33
	end = beg + len(out)
	for q in range(len(out)):
		out[q] = 0
	counted = 0
	lost = 0
	depth = 0
	i = lo
	while i < hi:
		place = ((keys[i] & mask) - bias) >> 1
		while i < hi and ((keys[i] & mask) - bias) >> 1 == place:
			p = (keys[i] & mask) - bias
			if p & 1:
				depth -= 1
			else:
				depth += 1
				if count_lo <= place and place < count_hi:
					counted += 1
			i += 1
		if i >= hi or depth == 0:
			continue
		next_place = ((keys[i] & mask) - bias) >> 1
		if first and place < 0:
			lost += (min(next_place, 0) - place) * depth
		if last and next_place > size:
			lost += (next_place - max(place, size)) * depth
		a = max(place, beg)
		b = min(next_place, end)
		for q in range(a - beg, b - beg):
			out[q] = depth
	return counted, lost


def _region_dense(files, chrom, size, beg, end, first, last, margin,
	unstranded):
	"""The depth of each strand at each base of [beg, end) inside chrom, as
	int32 arrays, with the start events and the coverage clipped away that
	`_region_runs` gives."""

	stop = (1 << 62) if last else end + margin[1]
	keys = _region_keys(files, chrom, beg - margin[0], end, stop)
	n_outs = 1 if unstranded else 2
	width = max(min(end, size) - beg, 0)
	if len(keys) == 0:
		return [numpy.zeros(width, dtype=numpy.int32) for _ in range(n_outs)], 0, 0
	keys.sort(kind='stable')
	bounds = numpy.searchsorted(keys, numpy.arange(n_outs + 1, dtype='int64') <<
		_KEY_SHIFT)
	kernel = _dense_kernel()
	result, counted, lost = [], 0, 0
	for k in range(n_outs):
		out = numpy.empty(width, dtype=numpy.int32)
		c, gone = kernel(keys, int(bounds[k]), int(bounds[k + 1]), beg, size,
			first, last, 0 if first else beg, size if last else min(end, size), out)
		changes = int(numpy.count_nonzero(out[1:] != out[:-1])) + 1
		result.append(_depth_runs(out, beg) if changes * _RUN_SPACING < width
			else out)
		counted += c
		lost += gone
	return result, counted, lost


def _depth_runs(depth, beg):
	"""The runs of nonzero depth, (starts, ends, values), with depth[0] at
	base `beg`."""

	bounds = numpy.concatenate([[0], numpy.flatnonzero(depth[1:] != depth[:-1])
		+ 1, [len(depth)]])
	values = depth[bounds[:-1]]
	kept = values != 0
	return beg + bounds[:-1][kept], beg + bounds[1:][kept], values[kept]


def _region_runs(files, chrom, size, beg, end, first, last, margin, unstranded):
	"""The runs of each strand over [beg, end) of chrom, (starts, ends,
	values, cut_before, cut_after), as `_coverage_runs` finds them, cut at
	the region's edges and, in the first and last regions, clipped to the
	chromosome; the start events the region holds inside the chromosome,
	which `_read_count` counts; and the coverage clipped away."""

	low = -(1 << 62) if first else beg
	high = (1 << 62) if last else end
	stop = high if last else end + margin[1]
	keys = _region_keys(files, chrom, beg - margin[0], end, stop)

	outs = numpy.arange(1 if unstranded else 2, dtype='int64')
	result = []
	starts_counted = 0
	lost = 0
	if len(keys) == 0:
		empty = numpy.empty(0, dtype='int64')
		flag = numpy.empty(0, dtype=bool)
		return [(empty, empty, empty, flag, flag) for _ in outs], 0, 0

	# A stable sort, quick on keys that come mostly in order, as the start
	# events of a sorted file do.
	keys.sort(kind='stable')
	bounds = numpy.searchsorted(keys, numpy.append(outs, len(outs)) << _KEY_SHIFT)
	kernel = _runs_kernel()
	count_lo = 0 if first else beg
	count_hi = size if last else min(end, size)
	for k in range(len(outs)):
		a, b = int(bounds[k]), int(bounds[k + 1])
		m = b - a
		starts = numpy.empty(m, dtype='int64')
		ends = numpy.empty(m, dtype='int64')
		values = numpy.empty(m, dtype='int64')
		cut_before = numpy.empty(m, dtype=bool)
		cut_after = numpy.empty(m, dtype=bool)
		n, counted, gone = kernel(keys, a, b, low, high, size, first, last,
			count_lo, count_hi, starts, ends, values, cut_before, cut_after)
		result.append((starts[:n], ends[:n], values[:n], cut_before[:n],
			cut_after[:n]))
		starts_counted += counted
		lost += gone
	return result, starts_counted, lost


def _open_files(filenames, chrom_sizes, args):
	"""The files as `_File`s, or None when one is not a coordinate-sorted
	BAM with a .bai at least as new as it, or the options or sizes are ones
	this reader does not take. With -v, every reference's mapped reads must
	be in its index, for the messages of the chromosomes not in the sizes."""

	if not args.coverage or args.mate_pairs or args.fragments:
		return None
	if max(abs(args.pos_shift), abs(args.neg_shift)) >= 1 << 30:
		return None
	chroms = [chrom for chrom, _ in chrom_sizes]
	if len(set(chroms)) != len(chroms):
		return None
	try:
		files = [_File(filename, chrom_sizes, args) for filename in filenames]
	except (LookupError, OSError, ValueError):
		return None
	if args.verbose and any(mapped is None for f in files for _, _, mapped, _, _
			in f.index):
		return None
	return files


def _report_missing(files, chrom_sizes, args):
	"""With -v, the line the whole-file reader prints for each chromosome
	with mapped reads that the sizes do not name, file by file."""

	if not args.verbose:
		return
	from .bam2bw import _tqdm
	tqdm = _tqdm(args.verbose)
	named = set(chrom for chrom, _ in chrom_sizes)
	for f in files:
		for name, (_, _, mapped, _, _) in zip(f.references, f.index):
			if mapped and name not in named:
				tqdm.write("{} encountered in input but not in FASTA/chrom sizes."
					.format(name))
	print("\n" * len(files))


def _discarded_line(discarded):
	return ("{} entries fell outside the provided chromosome sizes and were "
		"discarded ({}).".format(sum(discarded.values()), ", ".join("{}: {}"
		.format(chrom, count) for chrom, count in discarded.items())))


def _shares(cores):
	"""The threads that read regions and the writer's n_jobs while both
	work at once, which together take `cores`; with one, the main thread
	does both in turn, which 0 reading threads means."""

	if cores <= 1:
		return 0, 1
	writer = max(1, cores // _WRITER_SHARE)
	return cores - writer, writer


def _bam2bw_coverage(args, chrom_sizes, cores):
	"""`figwig bam2bw -c` through this reader: True once the bigWigs are
	written, False when the files are not ones it takes or the record
	kernel sends one to pysam. Each chromosome is written as it is done
	unless -r needs the whole file's reads first. The bigWigs are written
	beside their names and moved there once complete, so that a run that
	fails or falls back leaves the paths as they were."""

	from .bam2bw import _LEVEL
	from .bam2bw import write_counts
	from .figwig import _temporary_path
	from .writer import BigWigWriter

	files = _open_files(args.filename, chrom_sizes, args)
	if files is None:
		return False
	sizes = dict(chrom_sizes)
	names = dict(zip(['+', '-'], [args.name + ".bw"] if args.unstranded else
		[args.name + ".+.bw", args.name + ".-.bw"]))
	partials, writers = {}, {}

	def open_bigwigs(n_jobs):
		for key, name in names.items():
			partials[key] = _temporary_path(name)
			writers[key] = BigWigWriter(partials[key], chrom_sizes, zooms=args.zooms,
				level=_LEVEL, n_jobs=n_jobs)

	def emit(chrom, key, starts, ends, values):
		write_counts(writers[key], [(chrom, sizes[chrom])], {chrom: (starts, ends,
			values)}, args.scale_factor, None)

	try:
		if args.read_depth:
			strands, discarded, read_depth = _indexed_coverage(files, chrom_sizes,
				args, cores if cores > 1 else 0)
			_report_missing(files, chrom_sizes, args)
			print(_discarded_line(discarded))
			open_bigwigs(cores)
			if args.verbose:
				print("Dividing through by a read depth of {}.".format(read_depth))
			for key in strands:
				write_counts(writers[key], chrom_sizes, strands[key], args.scale_factor,
					read_depth)
		else:
			readers, writing = _shares(cores)
			open_bigwigs(writing)
			_, discarded, _ = _indexed_coverage(files, chrom_sizes, args, readers,
				emit)
		for writer in writers.values():
			writer.close()
		for key, name in names.items():
			os.replace(partials[key], name)
	except BaseException as error:
		for writer in writers.values():
			writer.__exit__(type(error), error, error.__traceback__)
		for partial in partials.values():
			if os.path.exists(partial):
				os.remove(partial)
		if isinstance(error, _Fallback):
			return False
		raise
	if not args.read_depth:
		_report_missing(files, chrom_sizes, args)
		print(_discarded_line(discarded))
	return True


def _bam2fw_coverage(args, chrom_sizes, cores, tracks, groups):
	"""`figwig bam2fw -c` through this reader, file by file: True once the
	file is written, False when the files are not ones it takes or the
	record kernel sends one to pysam, with nothing written then."""

	from .bam2fw import _write_coverage
	from .bam2fw import _write_runs
	from .figwig import FigWigWriter

	files = _open_files(args.filename, chrom_sizes, args)
	if files is None:
		return False
	per_file = len(groups) // len(files)

	def writer_for(n_jobs=cores):
		return FigWigWriter(args.name + '.fw', chrom_sizes, tracks,
			genome=args.genome, tile=args.tile, groups=groups, level=args.level,
			n_jobs=n_jobs, zooms=args.zooms)

	if args.read_depth:
		try:
			results = [_indexed_coverage([f], chrom_sizes, args, cores if cores > 1
				else 0) for f in files]
		except _Fallback:
			return False
		discarded = {chrom: sum(r[1][chrom] for r in results) for chrom, _ in
			chrom_sizes}
		_report_missing(files, chrom_sizes, args)
		print(_discarded_line(discarded))
		with writer_for() as writer:
			writer._copies = False
			for k, (strands, _, depth) in enumerate(results):
				group = [name for names in groups[k * per_file:(k + 1) * per_file]
					for name in names]
				if args.verbose:
					print("Dividing {} through by a read depth of {}.".format(group[0],
						depth))
				_write_coverage(writer, group, strands, chrom_sizes,
					args.scale_factor, depth)
		return True

	# A writer with threads takes each region's depth as one dense window,
	# whose tiles it encodes on them; one without takes the runs, which cost
	# it less than every base of the window.
	readers, writing = _shares(cores)
	sizes = dict(chrom_sizes)
	discarded = {chrom: 0 for chrom, _ in chrom_sizes}
	try:
		with writer_for(writing) as writer:
			writer._copies = False
			for k, f in enumerate(files):
				group = [name for names in groups[k * per_file:(k + 1) * per_file]
					for name in names]
				track_of = dict(zip(['+', '-'], group))

				def emit_depth(chrom, key, beg, depth):
					if isinstance(depth, tuple):
						starts, ends, values = depth
						if len(starts):
							values = values.astype(numpy.float64)
							values *= args.scale_factor
							writer.write(chrom, starts, values[:, None], ends=ends, tracks=[
								track_of[key]])
						return
					if not depth.any():
						return
					values = depth
					if args.scale_factor != 1:
						# Only the covered bases, as the runs are scaled.
						values = depth.astype(numpy.float64)
						numpy.multiply(values, args.scale_factor, out=values, where=depth !=
							0)
					writer.write(chrom, [beg], values[None, None], tracks=[track_of[key]])

				def emit_runs(chrom, key, starts, ends, values):
					_write_runs(writer, track_of[key], chrom, sizes[chrom], starts, ends,
						values, args.scale_factor, None)

				dense = writing > 1
				_, lost, _ = _indexed_coverage([f], chrom_sizes, args, readers,
					emit_depth if dense else emit_runs, dense=dense)
				for chrom in lost:
					discarded[chrom] += lost[chrom]
	except _Fallback:
		return False
	_report_missing(files, chrom_sizes, args)
	print(_discarded_line(discarded))
	return True


def _indexed_coverage(files, chrom_sizes, args, threads, emit=None,
	dense=False):
	"""The coverage of `files` (from `_open_files`), pooled, as
	`_coverage_runs` gives it for each chromosome and strand, read region by
	region on `threads` threads, or on this one when `threads` is 0. Each chromosome's runs, in the order of
	`chrom_sizes`, go to `emit(chrom, strand, starts, ends, values)` a region
	at a time, a run that crosses a region's edge whole in the first call
	that holds it, or into the returned strands when `emit` is None. With
	`dense`, `emit(chrom, strand, beg, depth)` instead takes each region's
	depth at every base from `beg` on, as int32; regions start on multiples
	of `_REGION`. Returns (strands, discarded, read
	count) with strands['+'][chrom] = (starts, ends, values) and '-' too
	when stranded. Raises `_Fallback` when the record kernel or the BGZF
	reader meets something the whole-file reader hands to pysam."""

	# A read that starts before a region may reach into it by -ns more
	# bases than its span, and one that starts after it may reach back by -ps.
	margin = (max(args.neg_shift, 0), max(-args.pos_shift, 0))
	keys = ['+'] if args.unstranded else ['+', '-']
	strands = {key: {} for key in keys}
	parts = {key: {} for key in keys}
	discarded = {}
	read_count = 0

	# Regions in genome order, at most _AHEAD x threads of them read ahead of
	# the region being put together, so that memory follows that window.
	regions = []
	for chrom, size in chrom_sizes:
		pieces = _regions(files, chrom, size, _REGION if dense else _RUNS_REGION)
		for i, (beg, end) in enumerate(pieces):
			regions.append((chrom, size, beg, end, i == 0, i == len(pieces) - 1))

	_bam_kernel()
	_bgzf_kernel()
	_inflate_engine()
	# With no threads, each region is read in turn on this one, between writes.
	pool = concurrent.futures.ThreadPoolExecutor(threads) if threads > 0 else None
	try:
		pending = collections.deque()
		todo = iter(regions)

		def submit():
			region = next(todo, None)
			if region is not None:
				chrom, size, beg, end, first, last = region
				task = _region_dense if dense else _region_runs
				work = (files, chrom, size, beg, end, first, last, margin,
					args.unstranded)
				if pool is None:
					future = concurrent.futures.Future()
					future.set_result(task(*work))
				else:
					future = pool.submit(task, *work)
				pending.append((region, future))

		for _ in range(_AHEAD * threads if pool is not None else 1):
			submit()

		# A region's last run that its edge cut is held until the next
		# region's first shows whether the two are one run.
		held = {key: None for key in keys}
		lost = 0
		while pending:
			(chrom, size, beg, _, first, last), future = pending.popleft()
			pieces, count, gone = future.result()
			submit()
			read_count += count
			if first:
				lost = 0
			lost += gone
			if dense:
				for key, depth in zip(keys, pieces):
					emit(chrom, key, beg, depth)
				if last:
					discarded[chrom] = lost
				continue
			for k, key in enumerate(keys):
				s, e, v, cut_before, cut_after = pieces[k]
				out = []
				if held[key] is not None:
					hs, he, hv = held[key]
					held[key] = None
					if len(s) and cut_before[0] and he == s[0] and hv == v[0]:
						s = s.copy()
						s[0] = hs
					else:
						out.append((numpy.array([hs]), numpy.array([he]), numpy.array([hv])))
				if len(s) and cut_after[-1] and not last:
					held[key] = s[-1], e[-1], v[-1]
					s, e, v = s[:-1], e[:-1], v[:-1]
				out.append((s, e, v))
				for s, e, v in out:
					if len(s) == 0:
						continue
					if emit is not None:
						emit(chrom, key, s, e, v)
					else:
						parts[key].setdefault(chrom, []).append((s, e, v))
			if last:
				discarded[chrom] = lost
				if emit is None:
					for key in keys:
						chunks = parts[key].pop(chrom, [])
						strands[key][chrom] = tuple(numpy.concatenate([c[i] for c in
							chunks]) if chunks else numpy.empty(0, dtype='int64') for i in
							range(3))
	finally:
		if pool is not None:
			pool.shutdown(wait=True, cancel_futures=True)

	return strands, discarded, read_count
