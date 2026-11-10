# fiberbam2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig fiberbam2fw` command: a Fiber-seq BAM into one figwig file
with a track per read and channel.

A Fiber-seq read is one chromatin fiber. Its signal is per read and per
base: the probability that each A or T carries an m6A (the `A+a` and `T-a`
blocks of the MM and ML tags), the probability that each CpG carries a 5mC
(`C+m`), the nucleosomes fibertools calls from the m6A pattern (`ns` and
`nl`), the stretches between them that the methyltransferase reached, or
MSPs (`as` and `al`), and the FIRE quality of each MSP (`aq`). This writes
each as a channel, five tracks per read named `<read>.m6a`, `<read>.cpg`,
`<read>.nuc`, `<read>.msp` and `<read>.fire`, on reference coordinates, with
the read's alignment and tags as the tracks' attributes, in groups of
consecutive reads per channel, the layout `FigWigReader.read(...,
channels="channel", items="read")` arranges into (windows, reads, channels,
width).
"""

import argparse
import concurrent.futures
import contextlib
import importlib.util
import math
import os
import sys
import threading

import numba
import numpy

from ._tables import _exit
from ._tables import _match_rows
from ._tables import _read_table
from ._tables import _refuse_existing
from .figwig import _DEFAULT_TILE
from .figwig import FigWigWriter

CHANNELS = ('m6a', 'cpg', 'nuc', 'msp', 'fire')

# The tags a track carries when its read has them, as `--tags` defaults.
_DEFAULT_TAGS = ('HP', 'RG', 'rq', 'ec')

_CHUNK_VALUES = 1 << 29

# The most threads that read and parse blocks of reads ahead of the writer.
# Each holds a block's entries, about 300 MB for 1,024 long reads, until it
# is written; more than 8 took more memory than they saved time.
_PARSE_THREADS = 8

_CHANNEL_TAGS = {
	'm6a': ('MM', 'Mm'), 'cpg': ('MM', 'Mm'),
	'nuc': ('ns', 'Ma', 'MA'), 'msp': ('as', 'Ma', 'MA'),
	'fire': ('as', 'Ma', 'MA')}

# Bits of the status `_mod_calls` returns.
_PAST_SEQUENCE = 1
_SHORT_ML = 2
_BAD_MM = 4
_DUPLICATE = 8
_NO_SEQUENCE = 16
_ODD_6MA = 32

_COMPLEMENT = numpy.zeros(256, dtype=numpy.uint8)
for _a, _b in zip(b'ACGTNacgtn', b'TGCANtgcan'):
	_COMPLEMENT[_a] = _b


def _parser(prog):
	parser = argparse.ArgumentParser(prog=prog, description="Convert a "
		"Fiber-seq BAM into one figwig file with a track per read and "
		"channel: the m6A and 5mC probabilities of each base, the "
		"nucleosomes, the methyltransferase-sensitive patches (MSPs) between "
		"them and the FIRE quality of each patch, on reference coordinates.")
	parser.add_argument('filename', nargs='+', help="Coordinate-sorted "
		"Fiber-seq BAM files with MM and ML tags, and ns, nl, as, al and aq "
		"tags or an MA and AQ tag. The chromosomes are those of the header, "
		"which must agree between files.")
	parser.add_argument('-n', '--name', required=True, help="The figwig file "
		"to write, without its .fw extension.")
	parser.add_argument('-p', '--parallel', default=1, type=int, help="The "
		"number of cores to use, or a negative number to count back from the "
		"number of CPUs: -1 is all of them.")
	parser.add_argument('-m', '--metadata', default=None, help="A tab- or "
		"comma-separated table with a header and one row per input file, "
		"matched on its first column by the path given here or its base name. "
		"A 'name' column puts a prefix before the file's read names, and every "
		"other column is an attribute of the file's tracks.")
	parser.add_argument('-g', '--genome', default=None, help="The reference "
		"genome, as a name or a URL, recorded in the file.")
	parser.add_argument('-c', '--channels', default=','.join(CHANNELS),
		help="The channels to write, a comma-separated subset of {}. Every "
		"read has each channel written once. Default is all of them.".format(
		', '.join(CHANNELS)))
	parser.add_argument('--threshold', default=None, type=int, help="Write "
		"each m6A and 5mC call whose probability (the ML tag, 0 to 255) is at "
		"least this as 1 and leave the others out, where the default writes "
		"every call's probability, which a reader can threshold at any value "
		"(fibertools uses 125 for m6A). A call of probability 0 is not "
		"written either way.")
	parser.add_argument('--tags', default=','.join(_DEFAULT_TAGS), help="The "
		"SAM tags that each track carries as an attribute, comma-separated, "
		"when its read has them; 'none' for none. Default is {}.".format(
		','.join(_DEFAULT_TAGS)))
	parser.add_argument('--primary-only', action='store_true', help="Leave "
		"out supplementary alignments, which carry the whole read's calls and "
		"are placed by their own alignment, so that a read split across two "
		"loci has a track on each. Secondary alignments are always left out.")
	parser.add_argument('--min-mapq', default=0, type=int, help="Leave out "
		"alignments of lower mapping quality. Default is 0.")
	parser.add_argument('--group-size', default=1024, type=int, help="The "
		"consecutive reads whose tiles of a channel are stored together. "
		"Default is 1024.")
	parser.add_argument('-t', '--tile', default=_DEFAULT_TILE, type=int,
		help="The bases per tile, a power of two from 16 to 65536. Default is "
		"{}.".format(_DEFAULT_TILE))
	parser.add_argument('-z', '--zooms', default=0, type=int, help="The number "
		"of zoom levels to keep beside the tile sums. Default is 0.")
	parser.add_argument('-l', '--level', default=1, type=int, help="The "
		"compression level, or 0 to store the chunks uncompressed. Default is "
		"1.")
	parser.add_argument('--force', action='store_true', help="Write over a "
		"figwig file at <-n>.fw, which is refused without it.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


@numba.njit(nogil=True, cache=True)
def _reference_positions(ops, lengths, start, n_bases, out):
	"""Fill `out` with the reference position of each base of SEQ, or -1
	where it has none, and return the hard-clipped bases at its two ends."""

	out[:n_bases] = -1
	left = 0
	right = 0
	q = 0
	r = start
	seen_aligned = False
	for k in range(len(ops)):
		op = ops[k]
		n = lengths[k]
		if op == 0 or op == 7 or op == 8:
			for _ in range(n):
				if q < n_bases:
					out[q] = r
				q += 1
				r += 1
			seen_aligned = True
		elif op == 1 or op == 4:
			q += n
		elif op == 2 or op == 3:
			r += n
			seen_aligned = True
		elif op == 5:
			if seen_aligned or q > 0:
				right += n
			else:
				left += n
	return left, right


@numba.njit(nogil=True, cache=True)
def _read_int(text, i):
	"""The unsigned integer at `text[i:]`, the index after it, and whether
	there were digits."""

	value = 0
	digits = 0
	while i < len(text) and text[i] >= 48 and text[i] <= 57:
		value = value * 10 + (text[i] - 48)
		i += 1
		digits += 1
	return value, i, digits > 0


@numba.njit(nogil=True, cache=True)
def _mod_calls(seq, reverse, refpos, mm, ml, threshold, complement,
		m6a_pos, m6a_val, cpg_pos, cpg_val, seen, value_of, by_letter, counts):
	"""Place the m6A and 5mC calls of a read's MM and ML tags on the
	reference.

	MM counts bases along the original read, which for a reverse-strand
	record is the reverse complement of SEQ: each block names a base, a
	strand and the modification codes, then how many of that base to skip
	before each call. `A+a` and `T-a` are 6mA on the read's A's and on the
	opposite strand's A's (T's of the read), `C+m` is 5mC. ML holds one
	probability per call and code, block after block. Calls of other codes
	are passed over. A call on a base without a reference position is
	counted, not placed. The calls placed come in order of position.

	`counts` receives: m6a calls, written, unplaced, cpg calls, written,
	unplaced, A and T bases of the read, and the status bits. `seen`,
	`value_of` and `by_letter` are room for a value per base.
	"""

	n_seq = len(seq)
	i = 0
	n_mm = len(mm)
	ml_at = 0
	n_m6a = 0
	n_cpg = 0
	status = 0
	counts[:] = 0
	seen[:n_seq] = 0

	# The bases of the original read by letter, each letter's in the read's
	# order, so that a skip count is an index rather than a walk.
	start = numpy.zeros(257, dtype=numpy.int64)
	for s in range(n_seq):
		start[(complement[seq[s]] if reverse else seq[s]) + 1] += 1
	counts[6] = start[66] + start[85]
	for x in range(256):
		start[x + 1] += start[x]
	fill = start[:256].copy()
	for cursor in range(n_seq):
		s = n_seq - 1 - cursor if reverse else cursor
		b = complement[seq[s]] if reverse else seq[s]
		by_letter[fill[b]] = s
		fill[b] += 1

	codes = numpy.zeros(8, dtype=numpy.int64)
	while i < n_mm:
		while i < n_mm and (mm[i] == 59 or mm[i] == 32):
			i += 1
		if i >= n_mm:
			break
		base = mm[i]
		if i + 1 >= n_mm or (mm[i + 1] != 43 and mm[i + 1] != 45):
			status |= 4
			break
		strand = mm[i + 1]
		i += 2
		n_codes = 0
		while i < n_mm and mm[i] != 44 and mm[i] != 59 and mm[i] != 63 and mm[i] != 46:
			if n_codes < 8:
				codes[n_codes] = mm[i]
				n_codes += 1
			i += 1
		if n_codes == 0:
			status |= 4
			break
		if i < n_mm and (mm[i] == 63 or mm[i] == 46):
			i += 1

		is_m6a = (base == 65 and strand == 43) or (base == 84 and strand == 45)
		for c in range(n_codes):
			if codes[c] == 97 and not is_m6a:
				status |= 32
		is_cpg = (base == 67 and strand == 43) or (base == 71 and strand == 45)
		# The next base of this kind of the original read, counting that
		# kind's bases, or every base for N.
		cursor = 0
		n_kind = n_seq if base == 78 else start[base + 1] - start[base]
		while i < n_mm and mm[i] == 44:
			skip, i, ok = _read_int(mm, i + 1)
			if not ok:
				status |= 4
				i = n_mm
				break
			# A count of too many digits wraps negative.
			if skip < 0 or skip >= n_kind - cursor:
				status |= 1
				i = n_mm
				break
			cursor += skip
			if base != 78:
				found = by_letter[start[base] + cursor]
			else:
				found = n_seq - 1 - cursor if reverse else cursor
			cursor += 1

			for c in range(n_codes):
				if ml_at >= len(ml):
					status |= 2
					i = n_mm
					break
				p = ml[ml_at]
				ml_at += 1
				code = codes[c]
				if code == 97 and is_m6a:
					counts[0] += 1
					if refpos[found] < 0:
						counts[2] += 1
					elif seen[found] != 0:
						status |= 8
					elif (threshold > 0 and p >= threshold) or (threshold <= 0 and p > 0):
						seen[found] = 1
						value_of[found] = 1 if threshold > 0 else p
				elif code == 109 and is_cpg:
					counts[3] += 1
					if refpos[found] < 0:
						counts[5] += 1
					elif seen[found] != 0:
						status |= 8
					elif (threshold > 0 and p >= threshold) or (threshold <= 0 and p > 0):
						seen[found] = 2
						value_of[found] = 1 if threshold > 0 else p
			if status & 2:
				break
		if i < n_mm and mm[i] != 59:
			# Anything but the end of a block's list is malformed.
			if not (status & 3):
				status |= 4
			break

	# The calls in the order of SEQ, which is the order of their places.
	for s in range(n_seq):
		if seen[s] == 1:
			m6a_pos[n_m6a] = refpos[s]
			m6a_val[n_m6a] = value_of[s]
			n_m6a += 1
		elif seen[s] == 2:
			cpg_pos[n_cpg] = refpos[s]
			cpg_val[n_cpg] = value_of[s]
			n_cpg += 1
	counts[1] = n_m6a
	counts[4] = n_cpg
	counts[7] = status
	return n_m6a, n_cpg


@numba.njit(nogil=True, cache=True)
def _footprints(starts, lengths, reverse, full, left, n_seq, refpos, lo, hi):
	"""The reference span of each interval of the original molecule, from its
	first to its last base with a reference position, or -1 for an interval
	with none."""

	for k in range(len(starts)):
		a = full - (starts[k] + lengths[k]) if reverse else starts[k]
		s0 = max(a - left, 0)
		s1 = min(a + lengths[k] - left, n_seq)
		lo[k] = -1
		hi[k] = -1
		for s in range(s0, s1):
			if refpos[s] >= 0:
				lo[k] = refpos[s]
				break
		for s in range(s1 - 1, s0 - 1, -1):
			if refpos[s] >= 0:
				hi[k] = refpos[s]
				break


# The tags the raw reader looks for, as the columns of its table.
_AUX_TAGS = numpy.frombuffer(b'MMMmMLMlnsnlasalaqMaMA', dtype=numpy.uint8).reshape(-1, 2)
_MM, _Mm, _ML, _Ml, _NS, _NL, _AS, _AL, _AQ, _Ma, _MA = range(11)

# BAM's 4-bit bases as letters.
_LETTERS = numpy.frombuffer(b'=ACMGRSVTWYHKDBN', dtype=numpy.uint8)


@numba.njit(nogil=True, cache=True)
def _le(buf, p, width):
	"""The unsigned little-endian integer of `width` bytes at buf[p]."""

	value = numpy.int64(0)
	for k in range(width):
		value |= numpy.int64(buf[p + k]) << (8 * k)
	return value


@numba.njit(nogil=True, cache=True)
def _int32(buf, p):
	value = _le(buf, p, 4)
	return value - (numpy.int64(1) << 32) if value >= (numpy.int64(1) << 31) else value


@numba.njit(nogil=True, cache=True)
def _array_width(subtype):
	"""The bytes of a B-type array's values, or -1 for an unknown type."""

	if subtype == 99 or subtype == 67:
		return 1
	if subtype == 115 or subtype == 83:
		return 2
	if subtype == 105 or subtype == 73 or subtype == 102:
		return 4
	return -1


@numba.njit(nogil=True, cache=True)
def _aux_size(buf, p, end):
	"""The bytes of the aux field at buf[p], its tag and type included, or
	-1 when it does not parse."""

	if p + 3 > end:
		return -1
	kind = buf[p + 2]
	if kind == 65 or kind == 99 or kind == 67:
		size = 4
	elif kind == 115 or kind == 83:
		size = 5
	elif kind == 105 or kind == 73 or kind == 102:
		size = 7
	elif kind == 100:
		size = 11
	elif kind == 90 or kind == 72:
		q = p + 3
		while q < end and buf[q] != 0:
			q += 1
		if q >= end:
			return -1
		size = q + 1 - p
	elif kind == 66:
		if p + 8 > end:
			return -1
		width = _array_width(buf[p + 3])
		if width < 0:
			return -1
		size = 8 + width * _le(buf, p + 4, 4)
	else:
		return -1
	return size if p + size <= end else -1


@numba.njit(nogil=True, cache=True)
def _integers(buf, p, out):
	"""The values of the integer B-type field at buf[p] into out, as int64;
	returns how many, or -1 for an array of another type."""

	subtype = buf[p + 3]
	width = _array_width(subtype)
	if width < 0 or subtype == 102:
		return -1
	signed = subtype == 99 or subtype == 115 or subtype == 105
	count = _le(buf, p + 4, 4)
	for k in range(count):
		value = _le(buf, p + 8 + k * width, width)
		if signed and value >= (numpy.int64(1) << (8 * width - 1)):
			value -= numpy.int64(1) << (8 * width)
		out[k] = value
	return count


@numba.njit(nogil=True, cache=True)
def _is_integers(buf, p):
	return buf[p + 2] == 66 and _array_width(buf[p + 3]) > 0 and buf[p + 3] != 102


@numba.njit(nogil=True, cache=True)
def _raw_block(buf, places, tids, starts, which, threshold, complement,
		letters, names):
	"""Parse BAM records, record r at buf[places[r]], into spans as
	`_Parser.read` makes them, for the channels m6a, cpg, nuc, msp and fire,
	each a run of spans per read: per channel the first and last bases and
	values, and the spans of each read, (5, n + 1), as offsets. With them,
	each read's statistics, (n, 10): the counts of `_mod_calls`' m6a calls,
	unplaced, cpg calls, unplaced, A and T bases and status, the nuc, msp and
	fire calls, and 0 where the read has no MM tag, 1 where its calls were
	placed and 2 where they could not be; and, for each read, whether it is
	one this does not parse, as a record of an unexpected shape, which the
	caller reads through pysam instead; and the first read whose chromosome
	and start are not `tids` and `starts`, or -1. `which` says which of the
	five channels to make."""

	n = places.shape[0]
	mods = which[0] or which[1]
	footprints = which[2] or which[3] or which[4]
	tags = numpy.full((n, names.shape[0]), -1, dtype=numpy.int64)
	fields = numpy.zeros((n, 5), dtype=numpy.int64)
	odd = numpy.zeros(n, dtype=numpy.bool_)
	bound = numpy.zeros(5, dtype=numpy.int64)
	most_seq = 0
	most_cigar = 0
	most_calls = 1
	most_features = 0
	changed = -1
	for r in range(n):
		q = places[r]
		end = q + 4 + _int32(buf, q)
		rec = q + 4
		if _int32(buf, rec) != tids[r] or _int32(buf, rec + 4) != starts[r]:
			changed = r
			break
		n_name = buf[rec + 8]
		n_cigar = _le(buf, rec + 12, 2)
		flag = _le(buf, rec + 14, 2)
		n_seq = _int32(buf, rec + 16)
		cigar = rec + 32 + n_name
		seq = cigar + 4 * n_cigar
		aux = seq + (n_seq + 1) // 2 + n_seq
		fields[r, 0] = flag
		fields[r, 1] = n_cigar
		fields[r, 2] = cigar
		fields[r, 3] = n_seq
		fields[r, 4] = seq
		# htslib swaps in the CG tag's CIGAR for one too long for the record.
		if n_cigar == 2 and _le(buf, cigar, 4) == (n_seq << 4 | 4) and \
				_le(buf, cigar + 4, 4) & 15 == 3:
			odd[r] = True
		p = aux
		while p < end:
			size = _aux_size(buf, p, end)
			if size < 0:
				odd[r] = True
				break
			for k in range(names.shape[0]):
				if tags[r, k] < 0 and buf[p] == names[k, 0] and buf[p + 1] == names[k, 1]:
					tags[r, k] = p
			p += size
		if odd[r]:
			continue
		most_seq = max(most_seq, n_seq)
		most_cigar = max(most_cigar, n_cigar)

		if mods:
			mm = tags[r, _MM] if tags[r, _MM] >= 0 else tags[r, _Mm]
			ml = tags[r, _ML] if tags[r, _ML] >= 0 else tags[r, _Ml]
			if mm >= 0:
				if buf[mm + 2] != 90:
					odd[r] = True
				else:
					p = mm + 3
					while buf[p] != 0:
						if buf[p] >= 128:
							odd[r] = True
						p += 1
			if ml >= 0:
				if buf[ml + 2] != 66 or buf[ml + 3] != 67:
					odd[r] = True
				else:
					calls = _le(buf, ml + 4, 4)
					bound[0] += calls
					bound[1] += calls
					most_calls = max(most_calls, calls)
		if footprints:
			ns, nl, as_, al, aq = tags[r, _NS], tags[r, _NL], tags[r, _AS], tags[r, _AL], \
				tags[r, _AQ]
			if ns >= 0 and nl >= 0:
				if not (_is_integers(buf, ns) and _is_integers(buf, nl)) or _le(buf, ns +
						4, 4) != _le(buf, nl + 4, 4):
					odd[r] = True
				else:
					bound[2] += _le(buf, ns + 4, 4)
					most_features = max(most_features, _le(buf, ns + 4, 4))
			if as_ >= 0 and al >= 0:
				count = _le(buf, as_ + 4, 4)
				if not (_is_integers(buf, as_) and _is_integers(buf, al)) or _le(buf,
						al + 4, 4) != count:
					odd[r] = True
				elif aq >= 0 and (not _is_integers(buf, aq) or _le(buf, aq + 4, 4) !=
						count):
					odd[r] = True
				else:
					bound[3] += count
					bound[4] += count
					most_features = max(most_features, count)
			elif tags[r, _Ma] >= 0 or tags[r, _MA] >= 0:
				odd[r] = True

	lo = numpy.empty((5, bound.max()), dtype=numpy.int64)
	hi = numpy.empty((5, bound.max()), dtype=numpy.int64)
	values = numpy.empty((5, bound.max()), dtype=numpy.uint8)
	at = numpy.zeros((5, n + 1), dtype=numpy.int64)
	stats = numpy.zeros((n, 10), dtype=numpy.int64)

	sequence = numpy.empty(most_seq, dtype=numpy.uint8)
	ops = numpy.empty(most_cigar, dtype=numpy.int64)
	lengths = numpy.empty(most_cigar, dtype=numpy.int64)
	refpos = numpy.empty(most_seq, dtype=numpy.int64)
	seen = numpy.empty(most_seq, dtype=numpy.uint8)
	value_of = numpy.empty(most_seq, dtype=numpy.uint8)
	by_letter = numpy.empty(most_seq, dtype=numpy.int64)
	m6a_pos = numpy.empty(most_calls, dtype=numpy.int64)
	m6a_val = numpy.empty(most_calls, dtype=numpy.uint8)
	cpg_pos = numpy.empty(most_calls, dtype=numpy.int64)
	cpg_val = numpy.empty(most_calls, dtype=numpy.uint8)
	counts = numpy.zeros(8, dtype=numpy.int64)
	f_starts = numpy.empty(most_features, dtype=numpy.int64)
	f_lengths = numpy.empty(most_features, dtype=numpy.int64)
	quality = numpy.empty(most_features, dtype=numpy.int64)
	f_lo = numpy.empty(most_features, dtype=numpy.int64)
	f_hi = numpy.empty(most_features, dtype=numpy.int64)
	empty = numpy.zeros(0, dtype=numpy.uint8)
	used = numpy.zeros(5, dtype=numpy.int64)

	for r in range(n if changed < 0 else 0):
		at[:, r + 1] = used
		if odd[r]:
			continue
		flag, n_cigar, cigar, n_seq, seq = fields[r, 0], fields[r, 1], fields[r, 2], \
			fields[r, 3], fields[r, 4]
		reverse = flag & 16 != 0
		for k in range(n_seq):
			code = numpy.int64(buf[seq + k // 2])
			sequence[k] = letters[(code >> 4) if k % 2 == 0 else (code & 15)]
		for k in range(n_cigar):
			op = _le(buf, cigar + 4 * k, 4)
			ops[k] = op & 15
			lengths[k] = op >> 4
		left, right = _reference_positions(ops[:n_cigar], lengths[:n_cigar],
			starts[r], n_seq, refpos)
		full = n_seq + left + right

		if mods:
			mm = tags[r, _MM] if tags[r, _MM] >= 0 else tags[r, _Mm]
			ml = tags[r, _ML] if tags[r, _ML] >= 0 else tags[r, _Ml]
			if mm >= 0 and n_seq > 0 and left + right == 0:
				p = mm + 3
				while buf[p] != 0:
					p += 1
				calls = empty if ml < 0 else buf[ml + 8:ml + 8 + _le(buf, ml + 4, 4)]
				a, c = _mod_calls(sequence[:n_seq], reverse, refpos, buf[mm + 3:p], calls,
					threshold, complement, m6a_pos, m6a_val, cpg_pos, cpg_val, seen,
					value_of, by_letter, counts)
				for k in range(a):
					lo[0, used[0] + k] = m6a_pos[k]
					hi[0, used[0] + k] = m6a_pos[k]
					values[0, used[0] + k] = m6a_val[k]
				for k in range(c):
					lo[1, used[1] + k] = cpg_pos[k]
					hi[1, used[1] + k] = cpg_pos[k]
					values[1, used[1] + k] = cpg_val[k]
				used[0] += a
				used[1] += c
				stats[r, 0] = counts[0]
				stats[r, 1] = counts[2]
				stats[r, 2] = counts[3]
				stats[r, 3] = counts[5]
				stats[r, 4] = counts[6]
				stats[r, 5] = counts[7]
				stats[r, 9] = 1
			elif mm >= 0:
				stats[r, 5] = _NO_SEQUENCE
				stats[r, 9] = 2

		if footprints:
			ns, nl, as_, al, aq = tags[r, _NS], tags[r, _NL], tags[r, _AS], tags[r, _AL], \
				tags[r, _AQ]
			for c in range(2, 5):
				if not which[c]:
					continue
				if c == 2:
					if ns < 0 or nl < 0:
						continue
					m = _integers(buf, ns, f_starts)
					_integers(buf, nl, f_lengths)
					quality[:m] = 1
				else:
					if as_ < 0 or al < 0:
						continue
					m = _integers(buf, as_, f_starts)
					_integers(buf, al, f_lengths)
					if c == 3:
						quality[:m] = 1
					elif aq >= 0:
						_integers(buf, aq, quality)
					else:
						quality[:m] = 0
					if c == 4:
						# FIREs are the MSPs of a quality above 0.
						kept = 0
						for k in range(m):
							if quality[k] > 0:
								f_starts[kept] = f_starts[k]
								f_lengths[kept] = f_lengths[k]
								quality[kept] = quality[k]
								kept += 1
						m = kept
				stats[r, 4 + c] = m
				if m == 0:
					continue
				_footprints(f_starts[:m], f_lengths[:m], reverse, full, left, n_seq,
					refpos, f_lo, f_hi)
				for k in range(m):
					if f_lo[k] >= 0:
						lo[c, used[c]] = f_lo[k]
						hi[c, used[c]] = f_hi[k]
						values[c, used[c]] = min(quality[k], 255)
						used[c] += 1
		at[:, r + 1] = used
	return lo, hi, values, at, stats, odd, changed


def _fill(lo, hi):
	"""The bases of each span lo..hi (inclusive), span after span."""

	n = hi - lo + 1
	total = int(n.sum())
	if total == 0:
		return numpy.zeros(0, dtype=numpy.int64)
	ends = numpy.cumsum(n)
	return numpy.arange(total, dtype=numpy.int64) - numpy.repeat(ends - n - lo, n)


@numba.njit(nogil=True, cache=True)
def _sort_spans(lo, hi, values, at):
	"""Each read's spans (at[r] to at[r + 1]) sorted by start, and whether
	two spans of a read share a base. The arrays given are returned when
	every read's spans are in order already."""

	ordered = True
	for r in range(at.shape[0] - 1):
		for k in range(at[r] + 1, at[r + 1]):
			if lo[k] <= hi[k - 1]:
				ordered = False
	if ordered:
		return lo, hi, values, False

	lo_out = numpy.empty_like(lo)
	hi_out = numpy.empty_like(hi)
	values_out = numpy.empty_like(values)
	overlap = False
	for r in range(at.shape[0] - 1):
		a, b = at[r], at[r + 1]
		descending = True
		for k in range(a + 1, b):
			if lo[k] >= lo[k - 1]:
				descending = False
				break
		if descending:
			order = numpy.arange(b - a - 1, -1, -1)
		else:
			order = numpy.argsort(lo[a:b], kind='mergesort')
		for k in range(b - a):
			lo_out[a + k] = lo[a + order[k]]
			hi_out[a + k] = hi[a + order[k]]
			values_out[a + k] = values[a + order[k]]
			if k > 0 and lo_out[a + k] <= hi_out[a + k - 1]:
				overlap = True
	return lo_out, hi_out, values_out, overlap


@numba.njit(nogil=True, cache=True)
def _emit(lo, hi, values, at, r0, r1, first, shift, positions, tracks, out):
	"""The bases of the sorted, disjoint spans of reads r0 to r1, read r's
	in track first + r, in the order the writer encodes them: by tile, then
	track, then position. Returns how many there are."""

	cursor = at[r0:r1].copy()
	size = numpy.int64(1) << shift
	none = numpy.iinfo(numpy.int64).max
	t = none
	for r in range(r0, r1):
		if at[r] < at[r + 1]:
			t = min(t, lo[at[r]] >> shift)
	e = 0
	while t != none:
		a = t << shift
		b = a + size
		following = none
		for r in range(r0, r1):
			k = cursor[r - r0]
			end = at[r + 1]
			while k < end and lo[k] < b:
				for p in range(max(lo[k], a), min(hi[k], b - 1) + 1):
					positions[e] = p
					tracks[e] = first + r
					out[e] = values[k]
					e += 1
				if hi[k] >= b:
					break
				k += 1
			cursor[r - r0] = k
			if k < end:
				following = min(following, max(lo[k], b) >> shift)
		t = following
	return e


def _array(read, *keys, dtype=numpy.int64):
	for key in keys:
		if read.has_tag(key):
			return numpy.asarray(read.get_tag(key), dtype=dtype)
	return None


def _tag(read, *keys):
	for key in keys:
		if read.has_tag(key):
			return read.get_tag(key)
	return None


def _molecule_features(read):
	"""The nucleosomes, MSPs and FIREs of a read as (starts, lengths) pairs
	on the original molecule, and the FIRE qualities: from the ns, nl, as, al
	and aq tags, or from an MA tag with its AQ."""

	empty = numpy.zeros(0, dtype=numpy.int64)
	nuc = msp = fire = (empty, empty)
	quality = empty
	if read.has_tag('ns') and read.has_tag('nl'):
		nuc = (_array(read, 'ns'), _array(read, 'nl'))
	if read.has_tag('as') and read.has_tag('al'):
		msp = (_array(read, 'as'), _array(read, 'al'))
		quality = _array(read, 'aq')
		if quality is None:
			quality = numpy.zeros(len(msp[0]), dtype=numpy.int64)
		if len(quality) != len(msp[0]):
			raise ValueError("read {} has {} MSPs and {} qualities in aq.".format(
				read.query_name, len(msp[0]), len(quality)))
		fire = (msp[0][quality > 0], msp[1][quality > 0])
		quality = quality[quality > 0]
	elif read.has_tag('Ma') or read.has_tag('MA'):
		text = _tag(read, 'MA', 'Ma')
		qualities = _array(read, 'AQ', 'Aq')
		qualities = empty if qualities is None else qualities
		used = 0
		for group in text.split(';')[1:]:
			if ':' not in group:
				continue
			head, body = group.split(':', 1)
			pairs = [item.split('-') for item in body.split(',') if item]
			# An MA tag counts from 1.
			starts = numpy.array([int(a) for a, _ in pairs], dtype=numpy.int64) - 1
			lengths = numpy.array([int(b) for _, b in pairs], dtype=numpy.int64)
			has_quality = head.endswith('Q')
			scores = None
			if has_quality:
				scores = qualities[used:used + len(starts)]
				used += len(starts)
			name = head.rstrip('Q').rstrip('.+-')
			if name == 'nuc':
				nuc = (starts, lengths)
			elif name == 'msp':
				msp = (starts, lengths)
			elif name == 'fire':
				fire = (starts, lengths)
				quality = scores if scores is not None else numpy.ones(
					len(starts), dtype=numpy.int64)
	return nuc, msp, fire, quality


class _Parser:
	"""Parses reads into entries for one group, on any thread."""

	def __init__(self, channels, threshold):
		self.channels = channels
		self.threshold = 0 if threshold is None else threshold
		self._local = threading.local()

	def _scratch(self, n_seq, n_ml):
		"""This thread's arrays for a read of n_seq bases and n_ml calls,
		kept from read to read, since an allocation per read costs page
		faults that do not run in parallel."""

		local = self._local
		if getattr(local, 'bases', -1) < n_seq:
			local.bases = max(n_seq, 2 * getattr(local, 'bases', 0))
			local.refpos = numpy.empty(local.bases, dtype=numpy.int64)
			local.seen = numpy.empty(local.bases, dtype=numpy.uint8)
			local.value_of = numpy.empty(local.bases, dtype=numpy.uint8)
			local.by_letter = numpy.empty(local.bases, dtype=numpy.int64)
		if getattr(local, 'calls', -1) < n_ml:
			local.calls = max(n_ml, 2 * getattr(local, 'calls', 0))
			local.m6a_pos = numpy.empty(local.calls, dtype=numpy.int64)
			local.m6a_val = numpy.empty(local.calls, dtype=numpy.uint8)
			local.cpg_pos = numpy.empty(local.calls, dtype=numpy.int64)
			local.cpg_val = numpy.empty(local.calls, dtype=numpy.uint8)
			local.counts = numpy.zeros(8, dtype=numpy.int64)
		return local

	def read(self, read):
		"""One alignment's spans and statistics: {channel: (first bases, last
		bases, values)}, a call being a span of one base, and a dict of the
		numbers that become attributes."""

		n_seq = 0 if read.query_sequence is None else len(read.query_sequence)
		reverse = bool(read.is_reverse)
		ops = lengths = None
		cigar = read.cigartuples
		if cigar is not None:
			table = numpy.array(cigar, dtype=numpy.int64).reshape(-1, 2)
			ops, lengths = table[:, 0].copy(), table[:, 1].copy()
		else:
			ops = lengths = numpy.zeros(0, dtype=numpy.int64)
		ml = None
		if 'm6a' in self.channels or 'cpg' in self.channels:
			ml = _array(read, 'ML', 'Ml', dtype=numpy.uint8)
		local = self._scratch(n_seq, 1 if ml is None else max(len(ml), 1))
		refpos = local.refpos
		left, right = _reference_positions(ops, lengths, read.reference_start,
			n_seq, refpos)
		full = n_seq + left + right

		entries = {}
		stats = {}
		nothing = numpy.zeros(0, dtype=numpy.int64)
		empty = (nothing, nothing, numpy.zeros(0, dtype=numpy.uint8))

		if 'm6a' in self.channels or 'cpg' in self.channels:
			mm = _tag(read, 'MM', 'Mm')
			m6a = cpg = empty
			if mm is not None and n_seq > 0 and left + right == 0:
				seq = numpy.frombuffer(read.query_sequence.encode('ascii'),
					dtype=numpy.uint8)
				ml = numpy.zeros(0, dtype=numpy.uint8) if ml is None else ml
				counts = local.counts
				a, c = _mod_calls(seq, reverse, refpos, numpy.frombuffer(
					mm.encode('ascii'), dtype=numpy.uint8), ml, self.threshold,
					_COMPLEMENT, local.m6a_pos, local.m6a_val, local.cpg_pos,
					local.cpg_val, local.seen, local.value_of, local.by_letter, counts)
				positions = local.m6a_pos[:a].copy()
				m6a = (positions, positions, local.m6a_val[:a].copy())
				positions = local.cpg_pos[:c].copy()
				cpg = (positions, positions, local.cpg_val[:c].copy())
				stats['m6a_calls'] = int(counts[0])
				stats['m6a_unplaced'] = int(counts[2])
				stats['cpg_calls'] = int(counts[3])
				stats['cpg_unplaced'] = int(counts[5])
				stats['at_bases'] = int(counts[6])
				stats['status'] = int(counts[7])
			elif mm is not None:
				stats['status'] = _NO_SEQUENCE
			entries['m6a'] = m6a
			entries['cpg'] = cpg

		if any(c in self.channels for c in ('nuc', 'msp', 'fire')):
			nuc, msp, fire, quality = _molecule_features(read)
			for channel, (starts, spans) in (('nuc', nuc), ('msp', msp), ('fire', fire)):
				if channel not in self.channels or len(starts) == 0:
					entries[channel] = empty
					stats[channel + '_calls'] = 0
					continue
				lo = numpy.empty(len(starts), dtype=numpy.int64)
				hi = numpy.empty(len(starts), dtype=numpy.int64)
				_footprints(starts, spans, reverse, full, left, n_seq, refpos, lo, hi)
				values = quality if channel == 'fire' else numpy.ones(
					len(starts), dtype=numpy.int64)
				keep = lo >= 0
				entries[channel] = (lo[keep], hi[keep], numpy.minimum(values,
					255)[keep].astype(numpy.uint8))
				stats[channel + '_calls'] = int(len(starts))
		return entries, stats


def _keep(read, primary_only, min_mapq, wanted):
	if read.is_unmapped or read.reference_id < 0 or read.is_secondary:
		return False
	if primary_only and read.is_supplementary:
		return False
	if read.mapping_quality < min_mapq:
		return False
	return any(read.has_tag(tag) for tag in wanted)


def _tag_values(read, tags):
	out = {}
	for tag in tags:
		if read.has_tag(tag):
			value = read.get_tag(tag)
			if isinstance(value, float):
				if not math.isfinite(value):
					continue
			elif not isinstance(value, (int, str)):
				value = [x for x in value]
			out[tag] = value
	return out


def _scan(filename, args, wanted, tags):
	"""The kept alignments of a file in order, as dicts of what their tracks
	carry, with the file's chromosomes, how many alignments each filter
	left out and, for a BGZF-compressed BAM, the virtual offsets where
	each kept alignment starts and ends and its chromosome's id, (reads,
	3), or None for other files."""

	import pysam

	reads = []
	skipped = dict(unmapped=0, secondary=0, supplementary=0, low_mapq=0, no_tags=0)
	previous = (-1, -1)
	with pysam.AlignmentFile(filename, check_sq=False, threads=args.cores) as handle:
		header = list(zip(handle.references, handle.lengths))
		sample = {}
		for group in handle.header.to_dict().get('RG', []):
			sample[group['ID']] = group.get('SM')
		if len(header) == 0:
			raise ValueError("{} has no chromosomes in its header.".format(filename))
		seekable = handle.format == 'BAM' and handle.compression == 'BGZF'
		places = []
		alignments = handle.fetch(until_eof=True)
		while True:
			at = handle.tell() if seekable else None
			if seekable and places and places[-1][1] < 0:
				places[-1][1] = at
			read = next(alignments, None)
			if read is None:
				break
			if read.is_unmapped or read.reference_id < 0:
				skipped['unmapped'] += 1
				continue
			if read.is_secondary:
				skipped['secondary'] += 1
				continue
			if args.primary_only and read.is_supplementary:
				skipped['supplementary'] += 1
				continue
			if read.mapping_quality < args.min_mapq:
				skipped['low_mapq'] += 1
				continue
			if not any(read.has_tag(tag) for tag in wanted):
				skipped['no_tags'] += 1
				continue
			if read.reference_end > handle.lengths[read.reference_id]:
				raise ValueError("{} has a read, {}, that ends at {:,}, past the end of {} "
					"({:,} bases) in its header.".format(filename, read.query_name,
					read.reference_end, read.reference_name, handle.lengths[
					read.reference_id]))
			key = (read.reference_id, read.reference_start)
			if key < previous:
				raise ValueError("{} is not coordinate sorted: {} at {:,} follows a "
					"read at a later position; sort it with samtools sort.".format(
					filename, read.reference_name, read.reference_start))
			previous = key
			record = dict(read=read.query_name, chrom=read.reference_name,
				start=int(read.reference_start), end=int(read.reference_end),
				strand='-' if read.is_reverse else '+', mapq=int(read.mapping_quality),
				flag=int(read.flag))
			if read.query_length:
				record['length'] = int(read.query_length)
			record.update(_tag_values(read, tags))
			rg = record.get('RG')
			if rg is not None and sample.get(rg):
				record['sample'] = sample[rg]
			reads.append(record)
			if seekable:
				places.append([at, -1, read.reference_id])
	offsets = numpy.array(places, dtype=numpy.int64).reshape(-1, 3) if seekable else None
	return header, reads, skipped, offsets


def _tracks(filenames, reads_per_file, channels, group_size, metadata):
	"""The track records and groups: for each block of `group_size`
	consecutive reads, the group of each channel's tracks, read after read."""

	rows = [None] * len(filenames)
	if metadata is not None:
		header, records = _read_table(metadata)
		rows = _match_rows(metadata, records, header, filenames)

	tracks, groups, layout = [], [], []
	for filename, row, reads in zip(filenames, rows, reads_per_file):
		prefix = _base_name(filename) if len(filenames) > 1 else ''
		attributes = {}
		if row is not None:
			first = list(row)[0]
			for column, value in row.items():
				if column == 'name' and len(value) > 0:
					prefix = value
				elif column not in ('name', 'dtype', first):
					attributes[column] = value

		seen = {}
		ids = []
		for record in reads:
			name = prefix + '.' + record['read'] if prefix else record['read']
			if name in seen:
				seen[name] += 1
				name = '{}@{}:{}'.format(name, record['chrom'], record['start'])
				if name in seen:
					seen[name] += 1
					name = '{}#{}'.format(name, seen[name])
			seen.setdefault(name, 0)
			ids.append(name)

		for b in range(0, len(reads), group_size):
			block = list(range(b, min(b + group_size, len(reads))))
			layout.append((len(tracks), len(block)))
			for channel in channels:
				names = []
				for k in block:
					record = reads[k]
					track = {'name': '{}.{}'.format(ids[k], channel), 'dtype': 'uint8',
						'read': ids[k], 'channel': channel}
					track.update(attributes)
					for key, value in record.items():
						track['read_name' if key == 'read' else key] = value
					tracks.append(track)
					names.append(track['name'])
				groups.append(names)
	return tracks, groups, layout


def _base_name(filename):
	name = os.path.basename(filename)
	for extension in ('.bam', '.sam', '.cram'):
		if name.endswith(extension):
			return name[:-len(extension)]
	return name


def _inflate(filename, begin, end, offsets, scan, inflate):
	"""The bytes of a BGZF file from virtual offset `begin` to `end`,
	uncompressed, and where each of `offsets`, virtual offsets in that
	range, falls in them, by bam2bw's `scan` for blocks and its `inflate`,
	which checks each block as htslib does, without the GIL; or None where
	a block is not of the shape they read."""

	first, last = begin >> 16, end >> 16
	with open(filename, 'rb') as handle:
		handle.seek(first)
		data = numpy.frombuffer(handle.read(last - first + 65536), dtype=numpy.uint8)
	capacity = len(data) // 26 + 1
	starts, sizes, lengths = (numpy.empty(capacity, dtype=numpy.int64) for _ in
		range(3))
	scanned = numpy.zeros(3, dtype=numpy.int64)
	scan(data, len(data), starts, sizes, lengths, scanned)
	# The blocks up to the one `end` falls in, which must all be whole.
	n = int(numpy.searchsorted(starts[:scanned[0]], last - first, side='right'))
	if n == 0 or (starts[n - 1] != last - first if end & 0xffff else starts[n - 1] +
			sizes[n - 1] < last - first):
		return None
	starts, sizes, lengths = starts[:n], sizes[:n], lengths[:n]
	at = numpy.concatenate([[0], numpy.cumsum(lengths)])
	out = numpy.empty(int(at[-1]), dtype=numpy.uint8)
	if inflate(data, starts, sizes, lengths, out, 0) != len(out):
		return None
	k = numpy.searchsorted(starts, (offsets >> 16) - first)
	if (k >= n).any() or (starts[numpy.minimum(k, n - 1)] != (offsets >> 16) -
			first).any():
		return None
	return out, at[k] + (offsets & 0xffff)


class _Shared:
	"""A BAM's pysam handle, which the main thread opens when it begins
	writing the file's reads, as it would to read them, and the threads
	that parse its blocks share to read a record now and then."""

	def __init__(self):
		self.handle = None
		self.lock = threading.Lock()
		self.ready = threading.Event()

	@contextlib.contextmanager
	def open(self, filename):
		import pysam

		with pysam.AlignmentFile(filename, check_sq=False) as handle:
			self.handle = handle
			self.ready.set()
			try:
				yield
			finally:
				self.handle = None

	def abandon(self):
		self.ready.set()

	def read_at(self, offset):
		"""The record at virtual offset `offset`."""

		self.ready.wait()
		with self.lock:
			if self.handle is None:
				raise ValueError("the file was closed before its reads were parsed.")
			self.handle.seek(offset)
			return next(self.handle)


def _load_block(parser, filename, shared, bgzf, offsets, end, records, tids,
		first, shift):
	"""Parse the alignments of `records`, from the scan, which start at
	virtual offsets `offsets` of a BGZF-compressed BAM, the last ending at
	`end`, from their bytes; those of an unexpected shape, and every one
	when the file's blocks are, are read through `shared`, a `_Shared`, and
	parsed by `_Parser.read`. `bgzf` is bam2bw's kernels that find and
	inflate BGZF blocks."""

	n = len(records)
	starts = numpy.array([record['start'] for record in records], dtype=numpy.int64)
	inflated = _inflate(filename, offsets[0], end, offsets, *bgzf)
	odd = numpy.ones(n, dtype=bool)
	if inflated is not None:
		buf, places = inflated
		which = numpy.array([c in parser.channels for c in CHANNELS])
		lo, hi, values, at, table, odd, changed = _raw_block(buf, places, tids,
			starts, which, parser.threshold, _COMPLEMENT, _LETTERS, _AUX_TAGS)
		if changed >= 0:
			raise ValueError("{} changed while it was read.".format(filename))
		spans = {channel: (lo[c, :at[c, -1]], hi[c, :at[c, -1]], values[c, :at[c,
			-1]], at[c]) for c, channel in enumerate(CHANNELS) if channel in
			parser.channels}
		stats = [_stats_of(row, parser.channels) for row in table.tolist()]
	else:
		nothing = numpy.zeros(0, dtype=numpy.int64)
		spans = {channel: (nothing, nothing, numpy.zeros(0, dtype=numpy.uint8),
			numpy.zeros(n + 1, dtype=numpy.int64)) for channel in parser.channels}
		stats = [None] * n

	if odd.any():
		reads = {}
		for r in numpy.flatnonzero(odd).tolist():
			read = shared.read_at(int(offsets[r]))
			if read.reference_id != tids[r] or read.reference_start != starts[r]:
				raise ValueError("{} changed while it was read.".format(filename))
			reads[r], stats[r] = parser.read(read)
		spans = {channel: _splice(*spans[channel], {r: entries[channel] for r,
			entries in reads.items()}) for channel in parser.channels}
	return _entries(parser.channels, spans, tids, first, shift), stats


def _stats_of(row, channels):
	"""A read's statistics, as `_Parser.read` gives them, from its row of
	`_raw_block`'s table."""

	stats = {}
	if 'm6a' in channels or 'cpg' in channels:
		if row[9] == 1:
			stats.update(m6a_calls=row[0], m6a_unplaced=row[1], cpg_calls=row[2],
				cpg_unplaced=row[3], at_bases=row[4], status=row[5])
		elif row[9] == 2:
			stats['status'] = _NO_SEQUENCE
	if any(c in channels for c in ('nuc', 'msp', 'fire')):
		stats.update(nuc_calls=row[6], msp_calls=row[7], fire_calls=row[8])
	return stats


def _splice(lo, hi, values, at, replace):
	"""A channel's spans with those of the reads of `replace`, {read: (lo,
	hi, values)}, in place of theirs."""

	parts = ([], [], [])
	counts = numpy.diff(at)
	previous = 0
	for r in sorted(replace):
		for part, array, new in zip(parts, (lo, hi, values), replace[r]):
			part.append(array[at[previous]:at[r]])
			part.append(new)
		counts[r] = len(replace[r][0])
		previous = r + 1
	for part, array in zip(parts, (lo, hi, values)):
		part.append(array[at[previous]:])
	return tuple(numpy.concatenate(part) for part in parts) + (numpy.concatenate(
		[[0], numpy.cumsum(counts)]),)


def _parse_block(parser, reads, first, shift):
	"""The entries of a block of reads, as `_entries` gives them, and the
	reads' statistics."""

	n = len(reads)
	spans = {channel: ([], [], []) for channel in parser.channels}
	counts = {channel: numpy.zeros(n + 1, dtype=numpy.int64) for channel in
		parser.channels}
	stats = []
	tids = numpy.empty(n, dtype=numpy.int64)
	for local, read in enumerate(reads):
		entries, read_stats = parser.read(read)
		stats.append(read_stats)
		tids[local] = read.reference_id
		for channel in parser.channels:
			lo, hi, values = entries[channel]
			counts[channel][local + 1] = len(lo)
			spans[channel][0].append(lo)
			spans[channel][1].append(hi)
			spans[channel][2].append(values)
	spans = {channel: tuple(numpy.concatenate(part) for part in spans[channel]) + (
		numpy.cumsum(counts[channel]),) for channel in parser.channels}
	return _entries(parser.channels, spans, tids, first, shift), stats


def _entries(channels, spans, tids, first, shift):
	"""The entries of a block of reads whose tracks of channel c start at
	first + c * len(tids), from each channel's spans (first bases, last
	bases, values, and where each read's start). Each channel's entries are
	a list of (chromosome id, positions, tracks, values), a call of
	`write_sparse` each, sorted in the writer's order, with the entries of
	each track; or, where two spans of a read share a base, which the
	writer refuses, the list of one such call with the entries unsorted."""

	n = len(tids)
	# Each chromosome's reads, which in a sorted file are consecutive.
	cuts = numpy.concatenate([[0], numpy.flatnonzero(tids[1:] != tids[:-1]) + 1,
		[n]])
	out = {}
	for c, channel in enumerate(channels):
		track0 = first + c * n
		lo, hi, values, at = spans[channel]
		bases = numpy.concatenate([[0], numpy.cumsum(hi - lo + 1)])
		calls = []
		lo_sorted, hi_sorted, values_sorted, overlap = _sort_spans(lo, hi, values, at)
		if overlap:
			reads_of = numpy.repeat(numpy.arange(n), numpy.diff(at))
			lengths = hi - lo + 1
			calls.append((None, tids[reads_of].repeat(lengths), _fill(lo, hi),
				reads_of.repeat(lengths) + track0, values.repeat(lengths)))
		else:
			for r0, r1 in zip(cuts[:-1].tolist(), cuts[1:].tolist()):
				total = int(bases[at[r1]] - bases[at[r0]])
				if total == 0:
					continue
				positions = numpy.empty(total, dtype=numpy.int64)
				tracks = numpy.empty(total, dtype=numpy.int64)
				out_values = numpy.empty(total, dtype=numpy.uint8)
				_emit(lo_sorted, hi_sorted, values_sorted, at, r0, r1, track0, shift,
					positions, tracks, out_values)
				calls.append((int(tids[r0]), positions, tracks, out_values))
		out[channel] = (calls, numpy.diff(bases[at]))
	return out


_STAT_KEYS = {
	'm6a': ('m6a_calls', 'm6a_unplaced', 'at_bases'),
	'cpg': ('cpg_calls', 'cpg_unplaced'),
	'nuc': ('nuc_calls',), 'msp': ('msp_calls',), 'fire': ('fire_calls',)}

_STATUS_TEXT = (
	(_PAST_SEQUENCE, "a skip count in MM runs past the end of the read"),
	(_SHORT_ML, "ML has fewer values than MM has calls"),
	(_BAD_MM, "MM is malformed"),
	(_DUPLICATE, "MM names a base twice"),
	(_NO_SEQUENCE, "the alignment has no sequence or is hard-clipped, so its MM cannot be placed"),
	(_ODD_6MA, "MM has 6mA on a base and strand other than A+ or T-, which is left out"))


def main(argv=None, prog='figwig fiberbam2fw'):
	args = _parser(prog).parse_args(argv)

	missing = [name for name in ('pysam', 'isal', 'deflate') if
		importlib.util.find_spec(name) is None]
	if len(missing) > 0:
		raise SystemExit(("{} needs {}, from figwig's bam2bw extra: pip install "
			"'figwig[bam2bw]'").format(prog, ', '.join(missing)))

	from .bam2bw import _bgzf_kernel
	from .bam2bw import _cores
	from .bam2bw import _inflate_engine

	try:
		if not args.force:
			_refuse_existing(args.name + '.fw', '--force')
		channels = tuple(c for c in args.channels.split(',') if c)
		for channel in channels:
			if channel not in CHANNELS:
				raise ValueError("--channels takes {}, not {!r}.".format(
					', '.join(CHANNELS), channel))
		if len(set(channels)) != len(channels) or len(channels) == 0:
			raise ValueError("--channels needs each channel once.")
		if args.threshold is not None and not 1 <= args.threshold <= 255:
			raise ValueError("--threshold is a probability from 1 to 255.")
		if args.group_size < 1 or args.group_size * args.tile > _CHUNK_VALUES:
			raise ValueError("--group-size is from 1 to {:,} reads at tile {:,}, "
				"since a chunk holds at most 2**29 values.".format(
				_CHUNK_VALUES // args.tile, args.tile))
		tags = [] if args.tags == 'none' else [t for t in args.tags.split(',') if t]
		for filename in args.filename:
			if not os.path.isfile(filename):
				raise ValueError("{} is not a file.".format(filename))
		args.cores = _cores(args.parallel)
		wanted = sorted({tag for channel in channels for tag in _CHANNEL_TAGS[channel]})

		header = None
		reads_per_file, skipped_per_file, offsets_per_file = [], [], []
		for filename in args.filename:
			file_header, reads, skipped, offsets = _scan(filename, args, wanted, tags)
			if header is None:
				header = file_header
			elif header != file_header:
				raise ValueError("{} has other chromosomes than {}: the files "
					"must be aligned to one assembly.".format(filename, args.filename[0]))
			if len(reads) == 0:
				raise ValueError("{} has no alignment with a {} tag{}.".format(
					filename, ' or '.join(wanted), "" if args.min_mapq == 0 else
					" at --min-mapq {}".format(args.min_mapq)))
			reads_per_file.append(reads)
			skipped_per_file.append(skipped)
			offsets_per_file.append(offsets)
		tracks, groups, layout = _tracks(args.filename, reads_per_file, channels,
				args.group_size, args.metadata)
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e)

	attributes = {'fiberbam2fw': {'files': [os.path.basename(f) for f in
		args.filename], 'channels': list(channels), 'threshold': args.threshold,
		'primary_only': args.primary_only, 'min_mapq': args.min_mapq,
		'reads': [len(r) for r in reads_per_file],
		'skipped': skipped_per_file}}

	import pysam

	problems = {}
	parser = _Parser(channels, args.threshold)
	shift = args.tile.bit_length() - 1
	chroms = [name for name, _ in header]
	workers = min(args.cores, _PARSE_THREADS)
	shared = [_Shared() for _ in args.filename]
	# bam2bw's kernels compile, or load, and bind libdeflate here, before the
	# threads use them.
	bgzf = (_bgzf_kernel(), _inflate_engine())
	try:
		with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
			pending = []
			block_at = 0
			writer = None

			def drain(limit):
				while len(pending) > limit:
					future, base, n = pending.pop(0)
					entries, stats = future.result()
					_write_block(writer, entries, stats, base, n, channels, chroms,
						problems)

			def submit(f, b):
				nonlocal block_at
				base, n = layout[block_at]
				block_at += 1
				if not isinstance(b, int):
					task = pool.submit(_parse_block, parser, b, base, shift)
				else:
					block = offsets_per_file[f][b:b + args.group_size]
					task = pool.submit(_load_block, parser, args.filename[f], shared[f],
						bgzf, block[:, 0], int(block[-1, 1]), reads_per_file[f][b:b +
						args.group_size], block[:, 2], base, shift)
				pending.append((task, base, n))
				if writer is not None:
					drain(workers + 1)

			# The first blocks of a BAM are parsed while the writer checks the
			# tracks.
			early = 0
			if offsets_per_file[0] is not None:
				early = min(len(reads_per_file[0]), (workers + 1) * args.group_size)
				for b in range(0, early, args.group_size):
					submit(0, b)

			with FigWigWriter(args.name + '.fw', header, tracks, genome=args.genome,
					attributes=attributes, tile=args.tile, groups=groups, level=args.level,
					n_jobs=args.cores, zooms=args.zooms) as opened:
				writer = opened
				writer._copies = False
				for f, (filename, reads, offsets) in enumerate(zip(args.filename,
						reads_per_file, offsets_per_file)):
					if offsets is not None:
						# Each block is read by the thread that parses it, from its
						# bytes, and only a read of an unexpected shape through pysam.
						with shared[f].open(filename):
							for b in range(early if f == 0 else 0, len(reads), args.group_size):
								submit(f, b)
							drain(0)
						continue

					with pysam.AlignmentFile(filename, check_sq=False,
							threads=min(args.cores, 4)) as handle:
						block = []
						count = 0
						for read in handle.fetch(until_eof=True):
							if not _keep(read, args.primary_only, args.min_mapq, wanted):
								continue
							block.append(read)
							count += 1
							if len(block) == args.group_size:
								submit(f, block)
								block = []
						if block:
							submit(f, block)
						drain(0)
						if count != len(reads):
							raise ValueError("{} changed while it was read: {} alignments "
								"the first time and {} the second.".format(filename,
								len(reads), count))
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e)
	finally:
		for one in shared:
			one.abandon()

	for text, names in problems.items():
		print("{}: warning: {:,} alignments, such as {}: {}; their m6A and 5mC calls "
			"are partly or wholly left out.".format(prog, len(names), names[0], text),
			file=sys.stderr)
	if args.verbose:
		for filename, reads, skipped in zip(args.filename, reads_per_file,
				skipped_per_file):
			print("{}: {:,} alignments written{}.".format(filename, len(reads),
				"".join(", {:,} {} left out".format(v, k.replace('_', ' ')) for k, v
				in skipped.items() if v)), file=sys.stderr)


def _write_block(writer, entries, stats, base, n, channels, chroms, problems):
	"""Write one block's entries, a call per channel and chromosome, and
	record its reads' statistics in their tracks."""

	for c, channel in enumerate(channels):
		first = base + c * n
		for local, read_stats in enumerate(stats):
			track = writer.tracks[first + local]
			for key in _STAT_KEYS[channel]:
				if key in read_stats:
					track[key] = read_stats[key]
			if channel == 'm6a':
				status = read_stats.get('status', 0)
				for bit, text in _STATUS_TEXT:
					if status & bit:
						problems.setdefault(text, []).append(track['read_name'])
		calls, counts = entries[channel]
		for call in calls:
			if call[0] is not None:
				writer._sorted_entries = True
				writer.write_sparse(chroms[call[0]], *call[1:])
				continue
			writer._sorted_entries = False
			chrom, positions, tracks, values = call[1:]
			tids = numpy.unique(chrom)
			for tid in tids:
				take = slice(None) if len(tids) == 1 else chrom == tid
				writer.write_sparse(chroms[int(tid)], positions[take], tracks[take],
					values[take])
		if counts.sum() == 0:
			continue
		for local in range(n):
			writer.tracks[first + local]['n_values'] = int(counts[local])
