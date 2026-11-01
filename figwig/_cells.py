# _cells.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Per-cell counts from a fragments file, for `figwig bam2fw
--barcode-column`.

A fragments file, as Cell Ranger ATAC writes it, holds one fragment per
line: chromosome, start, end, the droplet barcode and a count. `figwig
bam2bw` reads it as a pooled track; this module counts each barcode's
fragment ends apart, so that a figwig file holds one track per cell. The
scanner is a numba kernel like bam2bw's `_scan_intervals`, over the same
BGZF units and the same chromosome-name table, with a second table of the
same shape for barcodes. Every recorded end becomes one int64 key,
`((chromosome << cell_bits) | cell) << 32 | position`, so that one sort of
the keys orders the entries as `FigWigWriter.write_sparse` takes them, a
chromosome at a time, and the runs of equal keys are the counts. The
position must lie inside its chromosome, which bounds it to 32 bits; an
end outside is counted as dropped, as `figwig bam2fw` reports it.

The key's position part is the tile's number above the cell and the
offset within the tile below it, so that the sorted entries lie in the
order the writer's encoders take, tile by tile and track by track within
one, and the writer is told not to sort them again.

Barcodes are either listed up front (`--cells`), in which case every
thread shares one table and the kernel skips an unlisted barcode itself
(returning to Python for each of the hundreds of thousands of barcodes a
raw 10x file holds beside its cells, on every thread, was the serial cost
that stopped the scan gaining past four threads), or discovered
as they appear, in which case each unit of the file is scanned against a
copy of the table and the barcodes it finds are numbered in file order
afterwards, so that the tracks come out in order of first appearance
whatever the number of threads.
"""

import collections
import zlib

import numpy

from .bam2bw import _FNV_OFFSET
from .bam2bw import _FNV_PRIME
from .bam2bw import _IV_BGZF_EXTRA
from .bam2bw import _IV_BGZF_HEADER
from .bam2bw import _NameTable
from .bam2bw import _NotBlockGzip
from .bam2bw import _bgzf_units
from .bam2bw import _inflate_engine
from .bam2bw import _jit
from .bam2bw import _text_units
from .bam2bw import _tqdm

_POSITION_BITS = 32
_POSITION_MASK = (1 << _POSITION_BITS) - 1
_KEY_BITS = 63
_STREAM_BLOCK = 1 << 24
_BGZF_EOF = (b'\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00'
	b'\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00')


def _scan_cells(buf, n, i, n_lines, keys, n_keys, table_hash, table_name,
	name_data, name_offsets, name_slots, cell_hash, cell_name, cell_data,
	cell_offsets, cell_slots, field, cell_bits, tile_bits, sizes, dropped,
	skipped, keep_all, mode, pos_shift, neg_shift):
	"""Scan the lines of buf[i:n], which ends at a newline or at the end of
	the file, appending one key per recorded end of each fragment whose
	barcode, in column `field` (from 0), has a cell number in the barcode
	table. Compiled with numba by _cell_scanner. A key is the chromosome,
	the tile, the cell and the offset in the tile, from the high bits down,
	the tile and the offset together taking the position's 32 bits.

	Returns (status, i, n_lines, n_keys, a, b). Status 0: every line was
	scanned. Status 1: the chromosome name buf[a:b] is not in the name
	table; status 4: the barcode buf[a:b] is not in the barcode table, which
	is returned only with `keep_all`, since otherwise an unknown barcode is
	one not kept; status 2: keys is full. For those three, i is the start of
	the line to resume from once the caller has added the name or grown keys. Status 3: the line
	buf[a:b] cannot be read.

	mode is 0 to record the start, 1 to record end - 1, and 2 to record both.
	An end that falls outside its chromosome is counted in dropped[slot]; a
	fragment of a barcode whose cell number is -1 is counted in skipped[0].
	"""

	mask = table_hash.shape[0] - 1
	cell_mask = cell_hash.shape[0] - 1
	capacity = keys.shape[0]
	prev_a = 0
	prev_b = -1
	prev_slot = -1
	cell_a = 0
	cell_b = -1
	cell_prev = -1

	while i < n:
		line_start = i
		if n_keys + 2 > capacity:
			return 2, line_start, n_lines, n_keys, 0, 0

		# Split the line on spaces and tabs, keeping the bounds of the first
		# three fields and of the barcode field. Every byte must be printable
		# ASCII, a space, a tab, or a carriage return that ends the line.
		n_fields = 0
		a0 = 0
		b0 = 0
		a1 = 0
		b1 = 0
		a2 = 0
		b2 = 0
		a3 = 0
		b3 = 0
		bad = False
		while True:
			while i < n:
				c = buf[i]
				if c == 9 or c == 32:
					i += 1
				elif c == 13:
					if i + 1 < n and buf[i + 1] == 10:
						i += 1
					else:
						bad = True
						break
				elif c == 10 or (c >= 33 and c <= 126):
					break
				else:
					bad = True
					break

			if bad or i == n or buf[i] == 10:
				break

			a = i
			while i < n and buf[i] >= 33 and buf[i] <= 126:
				i += 1

			if n_fields == 0:
				a0 = a
				b0 = i
			elif n_fields == 1:
				a1 = a
				b1 = i
			elif n_fields == 2:
				a2 = a
				b2 = i
			if n_fields == field:
				a3 = a
				b3 = i
			n_fields += 1

		if bad:
			while i < n and buf[i] != 10:
				i += 1
			return 3, line_start, n_lines, n_keys, line_start, i

		next_i = i + 1 if i < n else n

		# Blank lines and # comments are skipped.
		if n_fields == 0 or buf[a0] == 35:
			i = next_i
			n_lines += 1
			continue

		if n_fields < 3 or n_fields <= field:
			return 3, line_start, n_lines, n_keys, line_start, i

		# Look the chromosome up, reusing the previous line's answer when the
		# name is the same, which it nearly always is in a sorted file.
		length = b0 - a0
		same = length == prev_b - prev_a
		if same:
			for k in range(length):
				if buf[a0 + k] != buf[prev_a + k]:
					same = False
					break

		if same:
			slot = prev_slot
		else:
			h = _FNV_OFFSET
			for k in range(a0, b0):
				h = (h ^ buf[k]) * _FNV_PRIME

			j = h & mask
			found = -1
			while True:
				name = table_name[j]
				if name < 0:
					break

				if table_hash[j] == h:
					s = name_offsets[name]
					if name_offsets[name + 1] - s == length:
						match = True
						for k in range(length):
							if name_data[s + k] != buf[a0 + k]:
								match = False
								break

						if match:
							found = name
							break

				j = (j + 1) & mask

			if found < 0:
				return 1, line_start, n_lines, n_keys, a0, b0

			slot = name_slots[found]
			prev_a = a0
			prev_b = b0
			prev_slot = slot

		if slot < 0:
			i = next_i
			n_lines += 1
			continue

		# The barcode, the same way; consecutive lines often share it too.
		length = b3 - a3
		same = length == cell_b - cell_a
		if same:
			for k in range(length):
				if buf[a3 + k] != buf[cell_a + k]:
					same = False
					break

		if same:
			cell = cell_prev
		else:
			h = _FNV_OFFSET
			for k in range(a3, b3):
				h = (h ^ buf[k]) * _FNV_PRIME

			j = h & cell_mask
			found = -1
			while True:
				name = cell_name[j]
				if name < 0:
					break

				if cell_hash[j] == h:
					s = cell_offsets[name]
					if cell_offsets[name + 1] - s == length:
						match = True
						for k in range(length):
							if cell_data[s + k] != buf[a3 + k]:
								match = False
								break

						if match:
							found = name
							break

				j = (j + 1) & cell_mask

			if found < 0:
				if keep_all:
					return 4, line_start, n_lines, n_keys, a3, b3
				cell = -1
			else:
				cell = cell_slots[found]
			cell_a = a3
			cell_b = b3
			cell_prev = cell

		if cell < 0:
			skipped[0] += 1
			i = next_i
			n_lines += 1
			continue

		k = a1
		negative = False
		if buf[k] == 45:
			negative = True
			k += 1
		elif buf[k] == 43:
			k += 1

		if k == b1 or b1 - k > 15:
			return 3, line_start, n_lines, n_keys, line_start, i

		start = 0
		while k < b1:
			c = buf[k]
			if c < 48 or c > 57:
				return 3, line_start, n_lines, n_keys, line_start, i

			start = start * 10 + (c - 48)
			k += 1

		if negative:
			start = -start

		k = a2
		negative = False
		if buf[k] == 45:
			negative = True
			k += 1
		elif buf[k] == 43:
			k += 1

		if k == b2 or b2 - k > 15:
			return 3, line_start, n_lines, n_keys, line_start, i

		end = 0
		while k < b2:
			c = buf[k]
			if c < 48 or c > 57:
				return 3, line_start, n_lines, n_keys, line_start, i

			end = end * 10 + (c - 48)
			k += 1

		if negative:
			end = -end

		start += pos_shift
		end += neg_shift
		size = sizes[slot]
		high = slot << (cell_bits + _POSITION_BITS)
		low = cell << tile_bits
		offset_mask = (1 << tile_bits) - 1

		if mode != 1:
			if start < 0 or start >= size:
				dropped[slot] += 1
			else:
				keys[n_keys] = high | ((start >> tile_bits) << (cell_bits +
					tile_bits)) | low | (start & offset_mask)
				n_keys += 1

		if mode != 0:
			if end - 1 < 0 or end - 1 >= size:
				dropped[slot] += 1
			else:
				keys[n_keys] = high | (((end - 1) >> tile_bits) << (cell_bits +
					tile_bits)) | low | ((end - 1) & offset_mask)
				n_keys += 1

		i = next_i
		n_lines += 1

	return 0, i, n_lines, n_keys, 0, 0


_cell_scanner_cache = []

def _cell_scanner():
	"""_scan_cells compiled with numba."""

	if not _cell_scanner_cache:
		_cell_scanner_cache.append(_jit(_scan_cells))

	return _cell_scanner_cache[0]


class _Barcodes(_NameTable):
	"""A name table of barcodes, each with its cell number or -1 for a
	barcode that is not kept. `n_kept` is the next cell number. A copy made
	for a scanning thread keeps the arrays and hashes but not the dict of
	names, which only the base table needs."""

	def __init__(self, n):
		super().__init__(n)
		self.n_kept = 0

	def add(self, data, slot):
		count = len(self.hashes)
		if self.names is None:
			self.names = {}
			super().add(data, slot)
			self.names = None
		else:
			super().add(data, slot)
		if slot >= 0:
			self.n_kept += 1
		return count

	def copy(self):
		other = _Barcodes.__new__(_Barcodes)
		for attr in ('table_hash', 'table_name', 'name_data', 'name_offsets',
			'name_slots'):
			setattr(other, attr, getattr(self, attr).copy())

		other.names = None
		other.hashes = list(self.hashes)
		other.n_kept = self.n_kept
		return other


class _CellScan:
	"""The state one scan of a fragments file shares between its units: the
	chromosome table, the barcode table, the sizes, and the counters."""

	def __init__(self, chroms, sizes, barcodes, field, cell_bits, tile_bits,
			mode, args, filename):
		self.table = _NameTable(len(chroms))
		for slot, chrom in enumerate(chroms):
			data = chrom.encode('utf-8')
			if len(data) > 0 and min(data) >= 33 and max(data) <= 126:
				self.table.add(data, slot)

		self.keep_all = barcodes is None
		self.cells = _Barcodes(1024 if barcodes is None else len(barcodes))
		for barcode in barcodes or ():
			self.cells.add(barcode.encode('utf-8'), self.cells.n_kept)

		self.sizes = numpy.asarray(sizes, dtype=numpy.int64)
		self.field = field
		self.cell_bits = cell_bits
		self.tile_bits = tile_bits
		self.mode = mode
		self.args = args
		self.filename = filename
		self.dropped = numpy.zeros(len(chroms), dtype=numpy.int64)
		self.skipped = numpy.zeros(1, dtype=numpy.int64)
		self.missing = []
		self.scan = _cell_scanner()

	def scan_lines(self, buf, i, end, keys, n_keys, table, cells, missing,
			new_cells, dropped, skipped):
		"""Run the scanner over buf[i:end] with the given tables and
		counters, adding each chromosome name it does not know to `table`
		(as not in the sizes) and `missing`, each barcode it does not know
		to `cells` (numbered, and listed in `new_cells`, when every barcode
		is kept; as not kept otherwise), and growing `keys` when it fills.
		Returns (keys, n_keys, n_lines)."""

		n_lines = 0
		args = self.args
		while True:
			status, i, n_lines, n_keys, a, b = self.scan(buf, end, i, n_lines,
				keys, n_keys, table.table_hash, table.table_name, table.name_data,
				table.name_offsets, table.name_slots, cells.table_hash,
				cells.table_name, cells.name_data, cells.name_offsets,
				cells.name_slots, self.field, self.cell_bits, self.tile_bits,
				self.sizes, dropped, skipped, self.keep_all, self.mode,
				args.pos_shift, args.neg_shift)

			if status == 0:
				return keys, n_keys, n_lines
			elif status == 1:
				data = bytes(buf[a:b])
				table.add(data, -1)
				missing.append(data.decode('ascii'))
			elif status == 2:
				grown = numpy.empty(2 * len(keys), dtype=numpy.int64)
				grown[:n_keys] = keys[:n_keys]
				keys = grown
			elif status == 4:
				data = bytes(buf[a:b])
				if cells.n_kept >= 1 << self.cell_bits:
					raise ValueError(("{} holds more than {:,} barcodes, which is "
						"the most a file with {:,} chromosomes can keep apart; "
						"give --cells the barcodes to keep.").format(self.filename,
						1 << self.cell_bits, len(self.sizes)))
				cells.add(data, cells.n_kept)
				new_cells.append(data)
			else:
				line = bytes(buf[a:b]).decode('ascii', errors='replace')
				raise ValueError(("{}: the line {!r} does not hold a chromosome, "
					"a start, an end and a barcode in column {}.").format(
					self.filename, line[:120], self.field + 1))


def _renumber(keys, first_new, new_cells, base, cell_bits, tile_bits):
	"""Give the keys of a unit scanned against a copy of `base` the cell
	numbers `base` assigns, adding the unit's new barcodes to it in order."""

	if len(new_cells) == 0:
		return keys

	remap = numpy.arange(first_new + len(new_cells), dtype=numpy.int64)
	for j, data in enumerate(new_cells):
		count = base.names.get(data)
		if count is None:
			count = base.add(data, base.n_kept)
		remap[first_new + j] = base.name_slots[count]

	cell = (keys >> tile_bits) & ((1 << cell_bits) - 1)
	keys += (remap[cell] - cell) << tile_bits
	return keys


def _inflate_blocks(inflate, data, spans):
	"""The text of a run of BGZF blocks, inflated by bam2bw's kernel in one
	call without the GIL, each block checked to be one gzip member of the
	length and CRC-32 its trailer gives; a block that is not raises
	_NotBlockGzip. Inflating the 128 blocks of a unit one by one through
	Python objects held the GIL for a few microseconds per block, which
	with 177,000 blocks in a file was the scan's floor past four threads."""

	offsets = numpy.array([at for at, _ in spans], dtype=numpy.int64)
	sizes = numpy.array([size for _, size in spans], dtype=numpy.int64)
	ends = offsets + sizes
	view = numpy.frombuffer(data, dtype=numpy.uint8)
	lengths = (view[ends - 4].astype(numpy.int64) | (view[ends - 3].astype(
		numpy.int64) << 8) | (view[ends - 2].astype(numpy.int64) << 16) | (
		view[ends - 1].astype(numpy.int64) << 24))
	if (lengths > 65536).any():
		raise _NotBlockGzip()
	text = bytearray(int(lengths.sum()))
	end = inflate(view, offsets, sizes, lengths, numpy.frombuffer(text,
		dtype=numpy.uint8), 0)
	if end != len(text):
		raise _NotBlockGzip()
	return text


def _scan_file_units(state, compressed, bar, threads):
	"""Scan a BGZF or uncompressed file in units on `threads` threads, as
	bam2bw's _scan_units does, returning the keys. Raises _NotBlockGzip
	when a compressed file is not a series of BGZF blocks."""

	from concurrent.futures import ThreadPoolExecutor

	import threading

	local = threading.local()
	inflate = _inflate_engine() if compressed else None

	def work(unit, first, cells):
		text = _inflate_blocks(inflate, *unit) if compressed else unit

		start = 0
		if not first:
			start = text.find(b'\n') + 1
			if start == 0:
				return False, bytes(text), b'', None, 0, None, None, None

		end = text.rfind(b'\n') + 1

		table = getattr(local, 'table', None)
		if table is None:
			table = local.table = state.table.copy()
		if cells is None:
			# Listed barcodes: one table, read by every thread and added to
			# by none.
			cells = state.cells

		names, new_cells = [], []
		first_new = cells.n_kept
		dropped = numpy.zeros(len(state.dropped), dtype=numpy.int64)
		skipped = numpy.zeros(1, dtype=numpy.int64)
		keys = numpy.empty((end - start) // 12 + 16, dtype=numpy.int64)
		keys, n_keys, n_lines = state.scan_lines(numpy.frombuffer(text,
			dtype=numpy.uint8), start, end, keys, 0, table, cells, names,
			new_cells, dropped, skipped)
		keys = keys[:n_keys]
		if not state.keep_all:
			# Sorted here, on the thread, so that each chromosome's keys are
			# runs the final sort merges; with barcodes still being numbered
			# the keys are sorted once they have their numbers.
			keys.sort()

		return (True, bytes(text[:start]), bytes(text[end:]), keys, n_lines,
			names, (first_new, new_cells), (dropped, skipped))

	parts = []
	carry = bytearray()
	carry_table = state.table.copy()

	def scan_carry():
		keys = numpy.empty(16, dtype=numpy.int64)
		keys, n_keys, n_lines = state.scan_lines(numpy.frombuffer(carry,
			dtype=numpy.uint8), 0, len(carry), keys, 0, carry_table, state.cells,
			state.missing, [], state.dropped, state.skipped)
		parts.append(keys[:n_keys])
		bar.update(n_lines)

	def collect(result):
		nonlocal carry

		split, head, tail, keys, n_lines, names, found, counters = result
		carry += head
		if split:
			if len(carry) > 0:
				scan_carry()

			carry = bytearray(tail)
			if state.keep_all:
				keys = _renumber(keys, found[0], found[1], state.cells,
					state.cell_bits, state.tile_bits)
			parts.append(keys)
			state.missing.extend(names)
			state.dropped += counters[0]
			state.skipped += counters[1]
			bar.update(n_lines)

	pool = ThreadPoolExecutor(threads)
	try:
		with open(state.filename, "rb", buffering=0) as handle:
			units = _bgzf_units(handle) if compressed else _text_units(handle)

			pending = collections.deque()
			for k, unit in enumerate(units):
				# When barcodes are discovered as they come, each unit scans
				# a copy of the table as it stands when the unit is handed
				# out, taken here on the thread that also adds to it.
				cells = state.cells.copy() if state.keep_all else None
				pending.append(pool.submit(work, unit, k == 0, cells))
				if len(pending) > 2 * threads:
					collect(pending.popleft().result())

			while pending:
				collect(pending.popleft().result())

		if len(carry) > 0:
			scan_carry()
	finally:
		pool.shutdown(wait=True, cancel_futures=True)

	state.missing = list(dict.fromkeys(state.missing))
	return parts, not state.keep_all


def _scan_file_stream(state, compressed, bar):
	"""Scan a file as a stream on the calling thread, which any gzip file
	allows, returning the keys."""

	if compressed:
		from isal import igzip
		handle = igzip.open(state.filename, "rb")
	else:
		handle = open(state.filename, "rb")

	keys = numpy.empty(1 << 22, dtype=numpy.int64)
	n_keys = 0
	with handle:
		pending = bytearray()
		while True:
			block = handle.read(_STREAM_BLOCK)
			pending += block

			if len(block) > 0:
				end = pending.rfind(b'\n') + 1
				if end == 0:
					continue
			else:
				end = len(pending)

			buf = numpy.frombuffer(pending, dtype=numpy.uint8)
			keys, n_keys, n_lines = state.scan_lines(buf, 0, end, keys, n_keys,
				state.table, state.cells, state.missing, [], state.dropped,
				state.skipped)

			bar.update(n_lines)
			del buf
			del pending[:end]

			if len(block) == 0:
				break

	return [keys[:n_keys]], False


def _finish_slot(keys, cell_bits, tile_bits):
	"""One chromosome's keys, sorted and counted: (positions, cells,
	counts) as int64, int64 and float64, the form the writer takes without
	a copy. The keys come as sorted runs, one per unit of the file, which
	the stable sort merges in about one pass."""

	keys.sort(kind='stable')
	n = len(keys)
	first = numpy.empty(n, dtype=bool)
	first[0] = True
	numpy.not_equal(keys[1:], keys[:-1], out=first[1:])
	starts = numpy.flatnonzero(first)
	del first
	distinct = keys[starts]
	counts = numpy.diff(starts, append=n).astype(numpy.float64)
	del starts
	cells = (distinct >> tile_bits) & ((1 << cell_bits) - 1)
	positions = (distinct >> (tile_bits + cell_bits)) & (_POSITION_MASK >>
		tile_bits)
	positions <<= tile_bits
	positions |= distinct & ((1 << tile_bits) - 1)
	return positions, cells, counts


def _join_and_finish(pieces, cell_bits, tile_bits):
	return _finish_slot(numpy.concatenate(pieces), cell_bits, tile_bits)


def _finish(parts, presorted, chroms, cell_bits, tile_bits, threads):
	"""The counts of every chromosome from the units' keys: each unit's
	keys are cut at the chromosomes' edges (sorted first where they are
	not), and each chromosome's pieces are joined, merged with a stable
	sort, counted and decoded on `threads` threads, largest first, numpy's
	sorts, copies and ufuncs running without the GIL. A unit's array is
	dropped once every chromosome it holds a piece of is done, so the keys
	are not held whole beside the decoded arrays."""

	from concurrent.futures import ThreadPoolExecutor
	from concurrent.futures import as_completed

	shift = _POSITION_BITS + cell_bits
	edges = numpy.arange(1, len(chroms), dtype=numpy.int64) << shift
	pool = ThreadPoolExecutor(threads)
	try:
		def cut(keys):
			if not presorted:
				keys.sort()
			# The last chromosome's edge, len(chroms) << shift, is 2**63 when
			# len(chroms) is a power of two and wraps in int64, so it ends at
			# len(keys): every key's slot is below len(chroms).
			return numpy.concatenate(([0], numpy.searchsorted(keys, edges),
				[len(keys)]))

		bounds = list(pool.map(cut, parts))
		pieces = [[] for _ in chroms]
		owners = [[] for _ in chroms]
		users = [0] * len(parts)
		for i, (keys, b) in enumerate(zip(parts, bounds)):
			for slot in numpy.flatnonzero(b[1:] > b[:-1]).tolist():
				pieces[slot].append(keys[b[slot]:b[slot + 1]])
				owners[slot].append(i)
				users[i] += 1
		del bounds

		order = sorted((slot for slot in range(len(chroms)) if pieces[slot]),
			key=lambda slot: -sum(len(piece) for piece in pieces[slot]))
		futures = {}
		for slot in order:
			futures[pool.submit(_join_and_finish, pieces[slot], cell_bits,
				tile_bits)] = slot
			pieces[slot] = None
		counts = {}
		for future in as_completed(futures):
			slot = futures[future]
			counts[chroms[slot]] = future.result()
			for i in owners[slot]:
				users[i] -= 1
				if users[i] == 0:
					parts[i] = None
		return {chroms[slot]: counts[chroms[slot]] for slot in order}
	finally:
		pool.shutdown(wait=True)


def _is_bgzf(filename):
	with open(filename, 'rb') as handle:
		header = handle.read(16)
	return header[:4] == _IV_BGZF_HEADER and header[10:16] == _IV_BGZF_EXTRA


def _check_bgzf_end(filename):
	"""Raise for a BGZF file that does not end with the empty block that
	marks its end, which a file cut at a block boundary lacks, since every
	block before the cut would otherwise read as the whole file."""

	with open(filename, 'rb') as handle:
		handle.seek(0, 2)
		size = handle.tell()
		handle.seek(max(size - len(_BGZF_EOF), 0))
		tail = handle.read()
	if tail != _BGZF_EOF:
		raise ValueError(("{} ends without the empty block that ends a BGZF file, "
			"so it was cut short.").format(filename))


def _read_cells(filename, args, chrom_sizes, field, barcodes, threads, idx=0,
		tile=16384):
	"""Count each barcode's fragment ends in a fragments file.

	Parameters
	----------
	filename: str
		A BED/tsv file, BGZF, gzip or uncompressed.

	args: argparse.Namespace
		`figwig bam2fw`'s arguments: fragments, three_prime, pos_shift,
		neg_shift and verbose decide what is recorded.

	chrom_sizes: list of (str, int)
		The chromosomes to keep, in order.

	field: int
		The column that holds the barcode, from 0.

	barcodes: list of str or None
		The barcodes to keep, in the order their tracks take, or None to
		keep every barcode in order of first appearance.

	threads: int
		Threads to inflate and scan on; a gzip file that is not BGZF is
		read on one.

	idx: int
		The input's index, for the progress bar's row.

	tile: int
		The figwig file's tile, a power of two, which orders the entries.

	Returns
	-------
	counts: dict
		Chromosome name -> (positions, cells, counts), the distinct
		(position, cell) pairs with fragment ends and how many each has, as
		int64, int64 and float64 arrays, sorted by tile, then cell, then
		position: the order `FigWigWriter.write_sparse` encodes in when the
		cells are one group, in the types it takes without a copy.

	barcodes: list of str
		The barcodes kept, cell number i being barcodes[i].

	missing: list of str
		Chromosome names not in `chrom_sizes`, in order of first appearance.

	dropped: dict
		Chromosome name -> ends that fell outside it after the shifts.

	skipped: int
		Fragments of barcodes not in `barcodes`.
	"""

	chroms = [chrom for chrom, _ in chrom_sizes]
	sizes = [size for _, size in chrom_sizes]
	if len(chroms) >= 1 << 22:
		raise ValueError("--barcode-column takes at most {:,} chromosomes.".format(
			1 << 22))
	if max(abs(args.pos_shift), abs(args.neg_shift)) >= 1 << 38:
		raise ValueError("--barcode-column takes shifts below 2**38.")
	if max(sizes, default=0) > _POSITION_MASK:
		raise ValueError("--barcode-column takes chromosomes of at most {:,} "
			"bases.".format(_POSITION_MASK))

	slot_bits = max(1, int(max(len(chroms), 1) - 1).bit_length())
	cell_bits = _KEY_BITS - _POSITION_BITS - slot_bits
	if barcodes is not None and len(barcodes) > 1 << cell_bits:
		raise ValueError(("--cells lists {:,} barcodes, more than the {:,} a file "
			"with {:,} chromosomes can keep apart.").format(len(barcodes),
			1 << cell_bits, len(chroms)))

	mode = 2 if args.fragments else (1 if args.three_prime else 0)
	tile_bits = int(tile).bit_length() - 1
	state = _CellScan(chroms, sizes, barcodes, field, cell_bits, tile_bits, mode,
		args, filename)
	compressed = not filename.endswith(('.tsv', '.bed'))

	from isal.isal_zlib import IsalError

	bgzf = compressed and _is_bgzf(filename)
	if bgzf:
		_check_bgzf_end(filename)

	bar = _tqdm(args.verbose)(disable=not args.verbose, position=idx,
		desc=filename.split('/')[-1])
	try:
		parts = None
		if threads > 1 and (not compressed or bgzf):
			try:
				parts, presorted = _scan_file_units(state, compressed, bar, threads)
			except _NotBlockGzip:
				bar.reset()
		if parts is None:
			parts, presorted = _scan_file_stream(state, compressed, bar)
	except (EOFError, IsalError, zlib.error) as e:
		raise ValueError("{} is cut short, or is not a gzip file: {}".format(
			filename, e))
	finally:
		bar.close()

	parts = [keys for keys in parts if len(keys) > 0]
	counts = _finish(parts, presorted, chroms, cell_bits, tile_bits, threads)
	del parts

	if barcodes is None:
		table = state.cells
		barcodes = [bytes(table.name_data[table.name_offsets[k]:table
			.name_offsets[k + 1]]).decode('utf-8') for k in range(len(
			table.hashes))]

	dropped = {chroms[slot]: int(state.dropped[slot]) for slot in
		numpy.flatnonzero(state.dropped)}
	return counts, list(barcodes), state.missing, dropped, int(state.skipped[0])
