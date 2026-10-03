# bam2bw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""
`figwig bam2bw`: convert one or more SAM/BAM files, or BED/tsv files of
fragments, into a pair of stranded bigWig files of per-base counts of the
reads' 5' ends, or into one unstranded file. When several files are given,
their reads are pooled. Only mapped reads are counted.

This is the bam2bw command line tool, with the same arguments, outputs and
messages, reading its inputs the way the winner of a speed search over
bam2bw's code reads them, and writing its bigWigs with `BigWigWriter`:

- A BAM file is not read through pysam's per-read objects. Its BGZF blocks
  are inflated by libdeflate on the -p threads and its records are walked by
  a numba kernel, which writes one integer key per recorded position; sorting
  the keys and counting runs of equal keys gives each chromosome's counts.
- A BED/tsv file, gzipped or not, is scanned by a numba kernel, on the -p
  threads when it is BGZF or uncompressed.
- Anything those readers do not take exactly as htslib or the original
  per-line loop would -- a malformed record or line, --mate_pairs, a SAM
  file, a remote file -- is read by the original pysam or per-line loop, so
  that its result, or its error, is what bam2bw 0.5.1 gives.

The counts are written by `BigWigWriter` rather than pyBigWig, so the bigWigs
hold the same entries as bam2bw's but not the same bytes, and -z writes the
zoom levels figwig writes.

The readers need the packages in figwig's `bam2bw` extra: pysam, pyfaidx and
biopython (a FASTA as the sizes), tqdm (-v), joblib (several files), isal
and deflate. They are imported where they are used.
"""

import io
import os
import sys
import gzip
import zlib
import queue
import codecs
import numpy
import argparse
import itertools
import threading
import collections
import importlib.util

from .writer import BigWigWriter


# The packages of figwig's bam2bw extra that every run may need. pyfaidx and
# biopython, tqdm and joblib are needed only for a FASTA as the sizes, -v and
# several files, and are imported only then.
_REQUIRED = 'pysam', 'isal', 'deflate'


def _parser(prog):
	"""bam2bw's command line, as `prog`."""

	parser = argparse.ArgumentParser(
		prog=prog,
		description='This tool will convert BAM files to bigwig files without an intermediate.')

	parser.add_argument('filename', nargs='+',
		help="""The SAM/BAM or tsv/tsv.gz file to be processed.""")
	parser.add_argument('-s', '--sizes', required=True,
		help="""A chrom_sizes, .fai, or FASTA file. Only the first two columns of
		a chrom_sizes/.fai file are read. A compressed FASTA must be BGZF, not
		gzip.""")

	# Data properties

	parser.add_argument('-u', '--unstranded', action='store_true',
		help="Have only one, unstranded, output.")

	end_group = parser.add_mutually_exclusive_group()
	end_group.add_argument('-f', '--fragments', action='store_true', default=False,
		help="The data is fragments and so both ends should be recorded.")
	end_group.add_argument('-3p', '--three_prime', action='store_true', default=False,
		help="Record the 3' end of each read instead of the 5' end.")

	parser.add_argument('-ps', '--pos_shift', default=0, type=int,
		help="""A shift to apply to positive strand reads.""")
	parser.add_argument('-ns', '--neg_shift', default=0, type=int,
		help="""A shift to apply to negative strand reads.""")

	parser.add_argument('-mp', '--mate_pairs', action='store_true', default=False,
		help="""Treat paired-end reads as a single RNA/fragment tag instead of
		counting each mate independently: buffer reads by name, and once both
		mates of a pair are seen, record one jointly-determined position (see
		--rna5 and --opposite_strand). BAM/SAM input only.""")
	parser.add_argument('--rna5', choices=('read1', 'read2'), default='read1',
		help="""Which mate carries the 5' end of the RNA/fragment. The other
		mate's own 5' end is used as the RNA's 3' end. Only used with
		--mate_pairs.""")
	parser.add_argument('--opposite_strand', action='store_true', default=False,
		help="""Report the strand of the mate opposite the one chosen by
		--rna5, instead of that mate's own strand. Only used with
		--mate_pairs.""")

	parser.add_argument('-sf', '--scale_factor', default=1, type=float,
		help="""A scaling factor to multiply each position by.""")
	parser.add_argument('-r', '--read_depth', default=False, action='store_true',
		help="""Whether to divide through by total (pre-scaled) read depth.""")

	# Misc arguments

	parser.add_argument('-p', '--parallel', default=1, type=int,
		help="The number of jobs to use, max of one per input file.")
	parser.add_argument('-n', '--name', required=True)
	parser.add_argument('-z', '--zooms', default=0, type=int,
		help="""The number of zooms to store in the bigwig.""")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


# Without -v every progress bar is disabled, and nothing is written through
# tqdm, so tqdm is not imported for a bar that wraps no iterable and would
# only be updated and closed. A bar that wraps an iterable is still tqdm's own,
# so that an error raised while iterating has the traceback it always had.

class _QuietBar(object):
	"""tqdm, as bam2bw uses it without -v."""

	def __new__(cls, iterable=None, **kwargs):
		if iterable is None:
			return object.__new__(cls)

		from tqdm import tqdm
		return tqdm(iterable, **kwargs)

	def __init__(self, iterable=None, **kwargs):
		pass

	def update(self, n=1):
		pass

	def reset(self, total=None):
		pass

	def close(self):
		pass

	@staticmethod
	def write(*args, **kwargs):
		from tqdm import tqdm
		return tqdm.write(*args, **kwargs)


def _tqdm(verbose):
	"""tqdm with -v, and _QuietBar without it."""

	if verbose:
		from tqdm import tqdm
		return tqdm

	return _QuietBar


def _check_filenames(args):
	"""Raise for an input that bam2bw does not read."""

	acceptable_formats = '.bam', '.sam', '.tsv', '.tsv.gz', '.bed', '.bed.gz'

	for filename in args.filename:
		if not filename.endswith(acceptable_formats):
			raise ValueError("Filenames must end in one of {}.".format(', '.join(acceptable_formats)))

	if args.mate_pairs:
		for filename in args.filename:
			if not filename.endswith(('.bam', '.sam')):
				raise ValueError("--mate_pairs only supports BAM/SAM input files.")


# Here, we are determining the chromosomes and their sizes. This is necessary
# for creating the bigWig header(s) and for figuring out which reads to filter
# out (those that do not map to the provided chromosomes).
#
# Because we allow you to provide either a two-column chrom_sizes file or a
# FASTA file, we need code to handle the two situations. We use pyfaidx to
# quickly process the FASTA file so that we do not have to scan through the
# entire thing just to get the sizes.

def is_fasta(path):
	base = path.rstrip(".gz")
	return base.endswith((".fa", ".fasta", ".fna", ".fas"))

def read_chrom_sizes(path):
	"""The (name, length) pairs of a chrom_sizes, .fai or FASTA file, in
	order."""

	chrom_sizes = []

	# If provided a FASTA file, read the lengths of the sequences. pyfaidx reads
	# BGZF but not plain gzip, and its own error does not say what to do about it.
	if is_fasta(path):
		import pyfaidx

		try:
			fa = pyfaidx.Fasta(path)
		except pyfaidx.UnsupportedCompressionFormat:
			raise ValueError(("{} is compressed with gzip. Only BGZF is supported "
				"for compressed FASTA files -- recompress it with `bgzip`, or pass "
				"a chrom_sizes file instead.").format(path))

		for chrom, seq in fa.items():
			chrom_sizes.append((chrom, len(seq)))

	# If provided a chrom_sizes file, just use the provided lengths. Only the first
	# two columns are read, so a samtools .fai index works here as well as a
	# two-column chrom_sizes file, and blank lines and # comments are skipped.
	else:
		with open(path, "r") as size_file:
			for i, line in enumerate(size_file):
				line = line.strip()
				if len(line) == 0 or line.startswith("#"):
					continue

				fields = line.split()
				if len(fields) < 2:
					raise ValueError(("{}, line {}: expected a chromosome name and "
						"a length, got {}.").format(path, i + 1, repr(line)))

				try:
					size = int(fields[1])
				except ValueError:
					raise ValueError(("{}, line {}: expected an integer length for "
						"{}, got {}.").format(path, i + 1, fields[0],
							repr(fields[1])))

				chrom_sizes.append((fields[0], size))

	return chrom_sizes

###

# This is the main loop that goes through the reads and records them in
# one or two dictionaries (depending on if the data is stranded). The
# processing of BAM/SAM files relies on pysam, whereas tsv/tsv.gz files use
# basic file iteration. The processing of reads in both cases is largely the
# same except that, usually, the -f flag will be passed in for .tsv/.tsv.gz
# files because those come from fragments from ATAC-seq-like experiments,
# whereas BAM/SAM files are usually just reads.

# First, a function that reads a single file and returns one or two
# dictionaries of reads. This can be parallelized across files.

MateInfo = collections.namedtuple('MateInfo', ['start', 'end', 'is_forward'])

def record_positions(args, pos_reads, neg_reads, chrom, five_prime, three_prime, is_forward):
	reads = pos_reads if is_forward else neg_reads

	reads[chrom][three_prime if args.three_prime else five_prime] += 1

	if args.fragments:
		reads[chrom][five_prime if args.three_prime else three_prime] += 1

def record_mate_pair(args, pos_reads, neg_reads, chrom, anchor, other):
	# The RNA's 5' end is the anchor mate's own 5' end (per --rna5); the RNA's
	# 3' end is the other mate's own 5' end, since that mate started
	# sequencing from the opposite end of the fragment.
	five_prime = anchor.start if anchor.is_forward else anchor.end - 1
	three_prime = other.start if other.is_forward else other.end - 1
	is_forward = other.is_forward if args.opposite_strand else anchor.is_forward

	record_positions(args, pos_reads, neg_reads, chrom, five_prime, three_prime, is_forward)

# A BAM file (without --mate_pairs) is not read through pysam's per-read
# objects, which cost more than a microsecond per read. Instead its BGZF
# blocks are inflated in runs and the binary records are walked by a numba
# kernel, which writes one integer key per recorded position. Sorting
# the keys and counting runs of equal keys then gives, for every chromosome
# and strand, the sorted positions and their counts.
#
# A key packs the output (a chromosome, times two plus the strand when the
# data is stranded) into the bits above _KEY_SHIFT and the position plus
# _KEY_BIAS into the bits below, so that negative positions from a shift sort
# correctly. A position outside [-_KEY_BIAS, 2 ** _KEY_SHIFT - _KEY_BIAS) is
# handed to the pysam loop instead.

_KEY_SHIFT = 35
_KEY_BIAS = 1 << 33

def _parse_bam_records(buf, n, tid_bucket, missing_seen, missing_order,
	options, keys, state, run_starts, run_buckets, finished):
	"""Walk the complete BAM records in buf[:n] and append their keys.

	This is compiled with numba on first use (see `_bam_kernel`). It returns
	the number of bytes consumed: every whole record in the buffer, so that
	the caller carries the partial record at the end into the next chunk.
	`state` holds the number of keys written, the number of chromosomes not
	in the sizes file seen so far, the number of records read, and a status
	that is set to 1 when the file needs the pysam loop instead: a record
	htslib would reject, a mapped read with no CIGAR (whose error the pysam
	loop raises), a CIGAR that may be stored in a CG tag, or a position that
	does not fit in a key.

	It also logs the runs of consecutive keys that share a chromosome (see
	`_KeyCounter`): state[4] is the chromosome of the last key written,
	state[5] the number of runs, whose first key and chromosome are in
	run_starts and run_buckets, and state[6] is set to 1, and the logging
	stops, when a chromosome whose run has ended comes back (`finished`
	marks those), so that the runs no longer hold each chromosome's keys.
	"""

	# Every index into buf is cast to uint64: a signed index costs numba a
	# negative-index fix-up per byte, and stops the four byte loads of an
	# integer from being merged into one.
	def u16(i):
		p = numpy.uint64(i)
		return numpy.int64(buf[p]) | (numpy.int64(buf[p + numpy.uint64(1)]) << 8)

	def u32(i):
		p = numpy.uint64(i)
		return (numpy.int64(buf[p]) | (numpy.int64(buf[p + numpy.uint64(1)]) << 8) |
			(numpy.int64(buf[p + numpy.uint64(2)]) << 16) |
			(numpy.int64(buf[p + numpy.uint64(3)]) << 24))

	def i32(i):
		x = u32(i)
		return x - ((x >> 31) << 32)

	pos_shift, neg_shift = options[0], options[1]
	use_three_prime, fragments = options[2] != 0, options[3] != 0
	unstranded, n_ref = options[4] != 0, options[5]
	key_limit = numpy.int64(1) << _KEY_SHIFT

	n_keys, n_missing, n_records = state[0], state[1], state[2]
	current, n_runs, mixed = state[4], state[5], state[6]
	status = 0
	off = 0

	while off + 4 <= n:
		block_size = i32(off)
		if block_size < 32:
			status = 1
			break

		end = off + 4 + block_size
		if end > n:
			break

		r = off + 4
		tid = i32(r)
		pos = i32(r + 4)
		l_read_name = numpy.int64(buf[numpy.uint64(r + 8)])
		n_cigar = u16(r + 12)
		flag = u16(r + 14)
		l_seq = i32(r + 16)
		next_tid = i32(r + 20)

		# The checks htslib's bam_read1 and sam_read1 make before handing a
		# record to pysam, which raises on any of them.
		l_extranul = (4 - l_read_name % 4) % 4
		if l_seq < 0 or l_read_name < 1:
			status = 1
			break
		if (4 * n_cigar + l_read_name + l_extranul + (l_seq + 1) // 2 + l_seq
			> block_size - 32 + l_extranul):
			status = 1
			break
		if tid < -1 or tid >= n_ref or next_tid < -1 or next_tid >= n_ref:
			status = 1
			break

		# The reference end is pos plus the CIGAR operations that consume the
		# reference (M, D, N, =, X), and pos + 1 when there are none, as
		# htslib's bam_endpos computes it. htslib also rejects a mapped read
		# whose CIGAR consumes a different number of bases than its sequence
		# holds, and moves a CIGAR of more than 65535 operations out of a CG
		# tag when the record's own CIGAR is the placeholder kSmN; that
		# placeholder sends the file to the pysam loop.
		cigar = r + 32 + l_read_name
		mapped = (flag & 4) == 0
		ref_len = numpy.int64(0)
		if n_cigar > 0:
			if (tid >= 0 and pos >= 0 and
				u32(cigar) == (((l_seq << 4) | 4) & 0xFFFFFFFF)):
				status = 1
				break

			if mapped:
				query_len = numpy.int64(0)
				for j in range(n_cigar):
					word = u32(cigar + 4 * j)
					kind = (0x3C1A7 >> ((word & 15) << 1)) & 3
					if kind & 1:
						query_len += word >> 4
					if kind & 2:
						ref_len += word >> 4

				if l_seq > 0 and query_len != l_seq:
					status = 1
					break

		if not mapped:
			n_records += 1
			off = end
			continue

		bucket = tid_bucket[numpy.uint64(tid + 1)]
		if bucket < 0:
			if missing_seen[numpy.uint64(tid + 1)] == 0:
				missing_seen[numpy.uint64(tid + 1)] = 1
				missing_order[n_missing] = tid
				n_missing += 1

			n_records += 1
			off = end
			continue

		if n_cigar == 0:
			status = 1
			break

		if ref_len == 0:
			ref_len = 1

		start = pos + pos_shift
		stop = pos + ref_len + neg_shift

		if flag & 16:
			five_prime, three_prime = stop - 1, start
			strand = 1
		else:
			five_prime, three_prime = start, stop - 1
			strand = 0

		if unstranded:
			out = bucket << _KEY_SHIFT
		else:
			out = (2 * bucket + strand) << _KEY_SHIFT

		first, second = five_prime, three_prime
		if use_three_prime:
			first, second = three_prime, five_prime

		first += _KEY_BIAS
		second += _KEY_BIAS
		if first < 0 or first >= key_limit:
			status = 1
			break
		if fragments and (second < 0 or second >= key_limit):
			status = 1
			break

		if bucket != current:
			if mixed == 0:
				if current >= 0:
					finished[current] = 1
				if finished[bucket] != 0:
					mixed = 1
				else:
					run_starts[n_runs] = n_keys
					run_buckets[n_runs] = bucket
					n_runs += 1
			current = bucket

		keys[numpy.uint64(n_keys)] = out | first
		n_keys += 1
		if fragments:
			keys[numpy.uint64(n_keys)] = out | second
			n_keys += 1

		n_records += 1
		off = end

	state[0] = n_keys
	state[1] = n_missing
	state[2] = n_records
	state[3] = status
	state[4] = current
	state[5] = n_runs
	state[6] = mixed
	return off

def _jit(func):
	"""func compiled with numba.njit(cache=True, nogil=True) on its first
	call, or loaded from numba's cache."""

	import numba
	return numba.njit(cache=True, nogil=True)(func)

_bam_kernel_compiled = None

def _bam_kernel():
	global _bam_kernel_compiled
	if _bam_kernel_compiled is None:
		_bam_kernel_compiled = _jit(_parse_bam_records)
	return _bam_kernel_compiled

# Sorting the keys and counting their runs is done a group of chromosomes at a
# time while the file is still being read, as soon as the kernel has moved past
# them, on the threads that inflate the file's blocks when it has more than
# one core (see _read_bam_keys_threaded). In a
# coordinate-sorted file each chromosome's keys are one run, so a group of runs
# holds every key of its chromosomes and is counted on its own. The kernel
# flags a file where a chromosome comes back after its run has ended; its keys
# are then sorted and counted all at once after the reading, as they always
# were. A group is never smaller than _GROUP_KEYS keys, so that a file of many
# small contigs costs a few numpy calls rather than one per contig.

_GROUP_KEYS = 1 << 18

def _count_keys(keys, outs):
	"""Sort keys in place and count its runs of equal keys. Returns the sorted
	distinct positions, their counts, and for each output in outs (sorted)
	the bounds lo, hi of its slice of the two."""

	keys.sort()

	starts = numpy.empty(len(keys), dtype=bool)
	starts[0] = True
	numpy.not_equal(keys[1:], keys[:-1], out=starts[1:])
	starts = numpy.flatnonzero(starts)

	counts = numpy.diff(starts, append=len(keys))
	positions = keys[starts]
	del starts

	lo = numpy.searchsorted(positions, outs << _KEY_SHIFT)
	hi = numpy.searchsorted(positions, (outs + 1) << _KEY_SHIFT)

	positions &= (1 << _KEY_SHIFT) - 1
	positions -= _KEY_BIAS
	return positions, counts, outs, lo, hi

class _KeyCounter:
	"""Counts the keys of the chromosomes the kernel has finished with, in
	groups, on a pool of `threads` threads when `threads` > 1 and inline
	otherwise. The threaded BAM reader inflates on the same pool, so that the
	two share the file's cores. `advance` is called after every kernel call and
	`wait` before the keys array is replaced; `finish` returns the counts of
	every key, and `close` shuts the pool down."""

	def __init__(self, arrays, stranded, threads):
		self.state, self.run_starts, self.run_buckets = arrays[4], arrays[5], arrays[6]
		self.stranded = stranded
		self.pool = None
		if threads > 1:
			from concurrent.futures import ThreadPoolExecutor
			self.pool = ThreadPoolExecutor(threads)

		self.pending = []
		self.start = 0
		self.first_run = 0

	def _outs(self, buckets):
		buckets = numpy.sort(buckets)
		if not self.stranded:
			return buckets
		return (2 * buckets[:, None] + numpy.arange(2)).ravel()

	def _submit(self, keys, buckets):
		outs = self._outs(buckets)
		if self.pool is None:
			self.pending.append(_count_keys(keys, outs))
		else:
			self.pending.append(self.pool.submit(_count_keys, keys, outs))

	def advance(self, keys):
		state = self.state
		if state[6] != 0:
			return

		# Every run but the last has ended.
		last = int(state[5]) - 1
		if last <= self.first_run:
			return

		end = int(self.run_starts[last])
		if end - self.start < _GROUP_KEYS:
			return

		self._submit(keys[self.start:end], self.run_buckets[self.first_run:last].copy())
		self.start = end
		self.first_run = last

	def wait(self):
		self.pending = [part if isinstance(part, tuple) else part.result()
			for part in self.pending]

	def close(self):
		if self.pool is not None:
			self.pool.shutdown(wait=True, cancel_futures=True)
			self.pool = None

	def finish(self, keys, n_outs):
		"""The counts of keys, every key the kernel wrote: a list of
		(positions, counts, outs, lo, hi) from _count_keys."""

		try:
			n_keys = len(keys)
			if self.state[6] != 0:
				self.wait()
				self.pending = []
				if n_keys > 0:
					self.pending.append(_count_keys(keys, numpy.arange(n_outs,
						dtype='int64')))
			elif n_keys > self.start:
				last = int(self.state[5])
				self._submit(keys[self.start:], self.run_buckets[self.first_run:last].copy())

			self.wait()
			return self.pending
		finally:
			self.close()

# A BGZF block is a whole gzip member of at most 64 KiB inflated, whose header
# gives its compressed size, so the blocks can be found without inflating them
# and inflated in any order. With more than one core for a file, they are
# inflated by a pool of threads while the calling thread walks the records;
# with one, the calling thread inflates each run of blocks before walking it.
# A run of _BGZF_BATCH blocks is inflated by one call of a numba kernel that
# holds no GIL and calls libdeflate, which the `deflate` package bundles and
# which takes about 16% less CPU than isal on these blocks, on each block in
# turn. Each block is inflated into a small buffer of the thread's own and
# copied into the run's buffer: libdeflate writing into the run's buffer
# directly, whose memory the walking thread read last, took about 15% more CPU
# here. The runs' buffers are used again rather than allocated per run, which
# cost 1.5 M page faults. The pool is the one the keys are counted on (see
# _KeyCounter), so that a file's -p threads inflate and count, and the calling
# thread reads, finds the blocks and walks the records.
#
# A file that is not a series of such blocks from its first byte to its last
# goes to the pysam loop, as a bad block does (below). Reading it as one gzip
# stream instead, as Python's gzip module does, would skip zero bytes between
# members and ignore the block sizes, where htslib stops with an error.
#
# A block is taken only when it is exactly one gzip member: libdeflate inflates
# it to the length its trailer gives, the CRC32 matches, and the member ends at
# the end of the block. htslib takes the CRC32 from the block's last 8 bytes
# and ignores any bytes between the deflate data and them, and it does not
# check the length, so a block that fails here is one htslib may reject or may
# read differently. So is a block that declares an impossible size, and bytes
# that are not a gzip header where a block should start, which htslib rejects.
# The file is then read by pysam, which decides as htslib does.
#
# Where the kernel cannot be built, the same check is made through ctypes, one
# call per block; where the package's library does not export libdeflate's
# functions at all, each block's output is checked against the CRC32 and length
# in its last 8 bytes instead (see _inflate_engine).

_BGZF_READ = 1 << 22
_BGZF_BATCH = 64
_BGZF_SPARE = 1 << 16

class _NotBGZF(Exception):
	"""The file is not a series of BGZF blocks as htslib writes them; it is
	then read by the pysam loop."""

class _BadBGZFBlock(Exception):
	"""A block with a BGZF header is not exactly one gzip member that inflates
	to the length and CRC32 its trailer gives, declares an impossible size, or
	is followed by bytes that are not a gzip header, all of which htslib
	rejects or may read differently; pysam then reads the file."""

def _scan_bgzf_blocks(data, n, offsets, sizes, lengths, scanned):
	"""Find the BGZF blocks that lie whole in data[:n].

	This is compiled with numba on first use (see `_bgzf_kernel`). It writes
	the offset, compressed size and inflated size of each block, and sets
	scanned to the number of blocks, the offset of the first byte after them,
	and a status: 1 when a gzip header is not a BGZF header with htslib's
	single BC field, which htslib may read as plain gzip, and 2 when the bytes
	are not a gzip header at all or a BGZF header declares an impossible size,
	which htslib rejects.
	"""

	count = 0
	at = 0
	status = 0

	while count < len(offsets) and at + 18 <= n:
		if data[at] != 31 or data[at + 1] != 139 or data[at + 2] != 8:
			status = 2
			break
		if (data[at + 3] != 4 or data[at + 10] != 6 or data[at + 11] != 0 or
			data[at + 12] != 66 or data[at + 13] != 67 or data[at + 14] != 2 or
			data[at + 15] != 0):
			status = 1
			break

		size = (numpy.int64(data[at + 16]) | (numpy.int64(data[at + 17]) << 8)) + 1
		if size < 26:
			status = 2
			break
		if at + size > n:
			break

		end = at + size
		length = (numpy.int64(data[end - 4]) | (numpy.int64(data[end - 3]) << 8) |
			(numpy.int64(data[end - 2]) << 16) | (numpy.int64(data[end - 1]) << 24))
		if length > 65536:
			status = 2
			break

		offsets[count] = at
		sizes[count] = size
		lengths[count] = length
		count += 1
		at = end

	scanned[0] = count
	scanned[1] = at
	scanned[2] = status

_bgzf_kernel_compiled = None

def _bgzf_kernel():
	global _bgzf_kernel_compiled
	if _bgzf_kernel_compiled is None:
		_bgzf_kernel_compiled = _jit(_scan_bgzf_blocks)
	return _bgzf_kernel_compiled

def _inflate_bgzf_blocks(data, offsets, sizes, lengths, out, start):
	"""Inflate the BGZF blocks data[offsets[i]:offsets[i] + sizes[i]] one after
	another into out, from out[start] on, and return where the last one ends.
	-1 means that a block is not exactly one gzip member that libdeflate
	inflates to lengths[i] bytes with the CRC32 of its trailer, or does not
	fit.

	This is compiled with numba on first use (see `_inflate_kernel`), which
	also registers the libdeflate functions it calls.
	"""

	decompressor = _libdeflate_alloc_decompressor()
	if decompressor == 0:
		return numpy.int64(-1)

	used = numpy.zeros(2, dtype=numpy.uint64)
	used_in = used.ctypes.data
	used_out = used_in + 8
	scratch = numpy.empty(1 << 16, dtype=numpy.uint8)
	tmp = scratch.ctypes.data
	src = data.ctypes.data
	dst = out.ctypes.data
	n_data = numpy.uint64(len(data))
	n_out = numpy.uint64(len(out))
	at_out = numpy.uint64(start)
	good = at_out <= n_out

	for i in range(len(offsets) if good else 0):
		at = numpy.uint64(offsets[i])
		size = numpy.uint64(sizes[i])
		length = numpy.uint64(lengths[i])
		if at + size > n_data or length > n_out - at_out:
			good = False
			break

		result = _libdeflate_gzip_decompress_ex(decompressor, src + at, size,
			tmp, length, used_in, used_out)
		if result != 0 or used[0] != size or used[1] != length:
			good = False
			break
		_memcpy(dst + at_out, tmp, length)
		at_out += length

	_libdeflate_free_decompressor(decompressor)
	return numpy.int64(at_out) if good else numpy.int64(-1)

_inflate_kernel_compiled = None

def _inflate_kernel():
	"""`_inflate_bgzf_blocks` compiled, or None when libdeflate's functions
	cannot be found in the `deflate` package, or numba's external functions
	(not public API) are not there to call them by.

	numba will not cache a function that calls a ctypes function pointer, so
	the kernel calls the functions by name instead, as external functions, and
	the names are bound here to the addresses in the package's library, before
	the kernel is compiled or loaded from the cache.
	"""

	global _inflate_kernel_compiled
	global _libdeflate_alloc_decompressor, _libdeflate_free_decompressor
	global _libdeflate_gzip_decompress_ex, _memcpy

	if _inflate_kernel_compiled is None:
		import ctypes
		import deflate._deflate

		library = ctypes.CDLL(deflate._deflate.__file__)
		names = ('alloc_decompressor', 'free_decompressor', 'gzip_decompress_ex')
		try:
			import llvmlite.binding
			from numba.core import types

			external = types.ExternalFunction
			add_symbol = llvmlite.binding.add_symbol
			addresses = [ctypes.cast(getattr(library, 'libdeflate_' + name),
				ctypes.c_void_p).value for name in names]
		except (ImportError, AttributeError):
			_inflate_kernel_compiled = False
			return None

		for name, address in zip(names, addresses):
			add_symbol('bam2bw_libdeflate_' + name, address)
		add_symbol('bam2bw_memcpy',
			ctypes.cast(ctypes.CDLL(None).memcpy, ctypes.c_void_p).value)

		u = types.uintp
		_libdeflate_alloc_decompressor = external(
			'bam2bw_libdeflate_alloc_decompressor', u())
		_libdeflate_free_decompressor = external(
			'bam2bw_libdeflate_free_decompressor', types.void(u))
		_libdeflate_gzip_decompress_ex = external(
			'bam2bw_libdeflate_gzip_decompress_ex', types.int32(u, u, u, u, u, u, u))
		_memcpy = external('bam2bw_memcpy', u(u, u, u))

		_inflate_kernel_compiled = _jit(_inflate_bgzf_blocks)

	return _inflate_kernel_compiled or None

_libdeflate_functions = []

def _libdeflate():
	"""(ctypes, alloc, free, gzip_decompress_ex): libdeflate's decompressor
	allocator, its free and its gzip_decompress_ex, from the library inside
	the deflate package, or None where that library does not export them."""

	if not _libdeflate_functions:
		functions = None
		try:
			import ctypes
			import deflate._deflate

			lib = ctypes.CDLL(deflate._deflate.__file__)
			alloc = lib.libdeflate_alloc_decompressor
			alloc.argtypes = []
			alloc.restype = ctypes.c_void_p
			free = lib.libdeflate_free_decompressor
			free.argtypes = [ctypes.c_void_p]
			free.restype = None
			inflate = lib.libdeflate_gzip_decompress_ex
			inflate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
				ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
				ctypes.POINTER(ctypes.c_size_t)]
			inflate.restype = ctypes.c_int
			functions = ctypes, alloc, free, inflate
		except (ImportError, OSError, AttributeError):
			pass

		_libdeflate_functions.append(functions)

	return _libdeflate_functions[0]

def _inflate_bgzf_blocks_ctypes(data, offsets, sizes, lengths, out, start):
	"""`_inflate_bgzf_blocks` through ctypes, one call per block, each of
	which releases the GIL: the same check, for where the kernel cannot be
	built."""

	ctypes, alloc, free, inflate = _libdeflate()
	used = ctypes.c_size_t()
	made = ctypes.c_size_t()
	used_ref = ctypes.byref(used)
	made_ref = ctypes.byref(made)
	source = data.ctypes.data
	target = out.ctypes.data
	end = start

	decompressor = alloc()
	if not decompressor:
		return -1
	try:
		for at, size, length in zip(offsets.tolist(), sizes.tolist(),
			lengths.tolist()):
			if (at + size > len(data) or end + length > len(out) or
				inflate(decompressor, source + at, size, target + end, length,
					used_ref, made_ref) != 0 or
				used.value != size or made.value != length):
				return -1
			end += length
	finally:
		free(decompressor)

	return end

def _inflate_bgzf_blocks_checked(data, offsets, sizes, lengths, out, start):
	"""`_inflate_bgzf_blocks` through the deflate package's own functions,
	for where its library does not export libdeflate's. They do not say where
	a member ends, so each block's output is checked against the CRC32 and
	length in its last 8 bytes, as htslib checks it."""

	import deflate
	from isal import isal_zlib

	view = memoryview(data)
	end = start
	try:
		for at, size, length in zip(offsets.tolist(), sizes.tolist(),
			lengths.tolist()):
			stop = at + size

			# The deflate package reads a capacity of 0 as "return nothing"
			# without inflating, so an empty block is given room for one byte.
			part = deflate.deflate_decompress(view[at + 18:stop - 8], length or 1)
			if (len(part) != length or end + length > len(out) or
				isal_zlib.crc32(part) != int.from_bytes(view[stop - 8:stop - 4],
					"little")):
				return -1
			out[end:end + length] = numpy.frombuffer(part, dtype='uint8')
			end += length
	except deflate.DeflateError:
		return -1

	return end

def _inflate_engine():
	"""What inflates a run of blocks for `_inflate_bgzf`: the numba kernel; or
	where it cannot be built, the same check through ctypes; or where the
	deflate package does not export libdeflate's functions, the check of each
	block's trailer. None of the three takes a block that htslib would reject
	or read differently. Called by the reading thread, before the pool's
	threads use what it imports."""

	kernel = _inflate_kernel()
	if kernel is not None:
		return kernel
	if _libdeflate() is not None:
		return _inflate_bgzf_blocks_ctypes
	return _inflate_bgzf_blocks_checked

def _inflate_bgzf(kernel, out, data, offsets, sizes, lengths):
	"""Inflate a run of BGZF blocks into out after its first _BGZF_SPARE bytes,
	where the caller puts the partial record left over from the run before,
	and return the part of out they fill."""

	end = kernel(data, offsets, sizes, lengths, out, _BGZF_SPARE)
	if end < 0:
		raise _BadBGZFBlock()
	return out[:end]

def _bam_header_end(header, n_ref):
	"""Where the records start after the BAM header at the front of `header`:
	-1 when more bytes are needed to tell, None when it is not a BAM header
	with n_ref references. pysam has already parsed and checked the header,
	so this only finds where the records start."""

	def int32(at):
		return int.from_bytes(header[at:at + 4], "little", signed=True)

	if len(header) < 8:
		return -1
	if header[:4] != b"BAM\x01":
		return None

	l_text = int32(4)
	if l_text < 0:
		return None
	if len(header) < 12 + l_text:
		return -1
	if int32(8 + l_text) != n_ref:
		return None

	end = 12 + l_text
	for _ in range(n_ref):
		if len(header) < end + 4:
			return -1
		if int32(end) < 0:
			return None
		end += 8 + int32(end)

	return end if len(header) >= end else -1

def _read_bam_keys_threaded(filename, n_ref, kernel, arrays, progress, threads,
	counter):
	"""Inflate a BAM file's BGZF blocks, on the `threads` threads of counter's
	pool when there is more than one, which also counts the keys, and run the
	kernel over its records. Returns the array of keys, or None when the file
	has to be read by pysam instead, or raises _NotBGZF, _BadBGZFBlock or the
	OSError that stopped the reading, when it also has to be read by pysam. The
	caller closes counter, which cancels or waits for the runs still being
	inflated, when this does not return keys."""

	pool = counter.pool
	inflate = _inflate_engine()

	tid_bucket, missing_seen, missing_order, options, state = arrays[:5]
	scan = _bgzf_kernel()
	keys = numpy.empty(1 << 20, dtype='int64')
	spare = _BGZF_SPARE

	def batches(f):
		# Runs of _BGZF_BATCH blocks in file order, from reads of _BGZF_READ
		# bytes; a block cut by the end of a read starts the next one.
		capacity = (_BGZF_READ + (1 << 16)) // 26 + 1
		offsets = numpy.empty(capacity, dtype='int64')
		sizes = numpy.empty(capacity, dtype='int64')
		lengths = numpy.empty(capacity, dtype='int64')
		scanned = numpy.zeros(3, dtype='int64')
		rest = numpy.empty(0, dtype='uint8')

		while True:
			data = numpy.empty(len(rest) + _BGZF_READ, dtype='uint8')
			data[:len(rest)] = rest
			n_read = f.readinto(memoryview(data)[len(rest):])
			n = len(rest) + n_read

			scan(data, n, offsets, sizes, lengths, scanned)
			count, end, status = (int(x) for x in scanned)

			# Copies, since the next scan writes over these.
			offsets_ = offsets[:count].copy()
			sizes_ = sizes[:count].copy()
			lengths_ = lengths[:count].copy()
			for i in range(0, count, _BGZF_BATCH):
				j = i + _BGZF_BATCH
				yield data, offsets_[i:j], sizes_[i:j], lengths_[i:j]

			# Either sends the file to pysam, as a bad block among the ones
			# before it would.
			if status == 1:
				raise _NotBGZF()
			if status == 2:
				raise _BadBGZFBlock()

			rest = data[end:n].copy()
			if n_read == 0:
				# Bytes after the last whole block: a cut block or trailing
				# data, which pysam judges.
				if len(rest) > 0:
					raise _NotBGZF()
				return

	def inflated(f, pool):
		# The inflated runs in file order: inflated here without a pool, and
		# with one keeping 2 x threads in flight.
		#
		# Each run is inflated into a buffer of its own, and a buffer is used
		# again once the run it holds has been walked, which is when the
		# caller asks for the next one.
		if pool is None:
			out = numpy.empty(_BGZF_SPARE + _BGZF_BATCH * (1 << 16), dtype='uint8')
			for batch in batches(f):
				yield _inflate_bgzf(inflate, out, *batch)
			return

		pending = collections.deque()
		free = []

		def submit(batch):
			out = free.pop() if free else numpy.empty(
				_BGZF_SPARE + _BGZF_BATCH * (1 << 16), dtype='uint8')
			pending.append((out, pool.submit(_inflate_bgzf, inflate, out, *batch)))

		def take():
			out, future = pending.popleft()
			return out, future.result()

		for batch in batches(f):
			submit(batch)
			if len(pending) > 2 * threads:
				out, run = take()
				yield run
				free.append(out)

		while pending:
			yield take()[1]

	def walk(buf):
		# Run the kernel over buf and return its unconsumed tail, or None when
		# the file needs the pysam loop.
		nonlocal keys

		n = len(buf)
		needed = state[0] + 2 * (n // 36) + 2
		if needed > len(keys):
			counter.wait()
			grown = numpy.empty(max(needed, 2 * len(keys)), dtype='int64')
			grown[:state[0]] = keys[:state[0]]
			keys = grown

		n_records = state[2]
		consumed = kernel(buf, n, tid_bucket, missing_seen, missing_order,
			options, keys, state, *arrays[5:])
		if state[3] != 0:
			return None

		progress.update(state[2] - n_records)
		counter.advance(keys)
		return buf[consumed:].copy()

	try:
		with open(filename, "rb", buffering=0) as f:
			runs = inflated(f, pool)

			header = bytearray()
			start = -1
			for run in runs:
				header += memoryview(run)[spare:]
				start = _bam_header_end(header, n_ref)
				if start != -1:
					break

			if start is None or start == -1:
				return None

			rest = walk(numpy.frombuffer(header, dtype='uint8')[start:])
			if rest is None:
				return None

			for buf in runs:
				if len(rest) <= spare:
					buf[spare - len(rest):spare] = rest
					rest = walk(buf[spare - len(rest):])
				else:
					rest = walk(numpy.concatenate([rest, buf[spare:]]))

				if rest is None:
					return None
	except _BadBGZFBlock:
		return None

	# Bytes left over at the end of the stream are a truncated record.
	if len(rest) > 0:
		return None

	return keys[:state[0]]

def count_bam_reads(bam, filename, chrom_sizes, pos_reads, neg_reads, args,
	idx, name):
	"""Count the reads of an open BAM file without going through pysam's
	per-read objects.

	On success, the entry of every chromosome and strand that has reads is
	replaced by a tuple (positions, counts, start, end): its sorted positions
	and their counts are positions[start:end] and counts[start:end], two int64
	arrays shared by every entry, and True is returned. False means that the
	file holds something this reader does not handle exactly as the pysam loop
	would -- an unusual record, an error, a remote file -- and that nothing
	has been recorded or printed, so the caller reads it with pysam instead.
	`bam` is the file as opened by pysam, positioned after the header, which
	this reader does not move.
	"""

	if not os.path.isfile(filename):
		return False
	if max(abs(args.pos_shift), abs(args.neg_shift)) >= 1 << 30:
		return False

	# Names are looked up the way the pysam loop's read.reference_name does,
	# and a record on tid -1 has the name None.
	references = [bam.get_reference_name(tid) for tid in range(bam.nreferences)]
	chroms = list(dict.fromkeys(chrom for chrom, _ in chrom_sizes))
	if 2 * len(chroms) >= 1 << (63 - _KEY_SHIFT):
		return False

	bucket_of = {chrom: i for i, chrom in enumerate(chroms)}
	tid_bucket = numpy.array([-1] + [bucket_of.get(chrom, -1) for chrom in references],
		dtype='int64')
	missing_seen = numpy.zeros(len(tid_bucket), dtype='uint8')
	missing_order = numpy.zeros(len(tid_bucket), dtype='int64')
	options = numpy.array([args.pos_shift, args.neg_shift, args.three_prime,
		args.fragments, args.unstranded, len(references)], dtype='int64')
	state = numpy.zeros(7, dtype='int64')
	state[4] = -1
	run_starts = numpy.zeros(len(chroms) + 1, dtype='int64')
	run_buckets = numpy.zeros(len(chroms) + 1, dtype='int64')
	finished = numpy.zeros(len(chroms), dtype='uint8')
	arrays = (tid_bucket, missing_seen, missing_order, options, state, run_starts,
		run_buckets, finished)

	tqdm = _tqdm(args.verbose)
	kernel = _bam_kernel()
	progress = tqdm(disable=not args.verbose, position=idx, desc=name)

	# The cores -p gives each of the files being read at the same time.
	n_files = len(args.filename)
	threads = args.parallel // min(args.parallel, n_files) if args.parallel > 1 else 1

	# A file the block reader cannot take exactly as htslib would read it goes
	# to the pysam loop.
	counter = _KeyCounter(arrays, not args.unstranded, threads)
	keys = None
	try:
		keys = _read_bam_keys_threaded(filename, len(references), kernel,
			arrays, progress, threads, counter)
	except (_NotBGZF, _BadBGZFBlock, OSError):
		pass
	finally:
		if keys is None:
			counter.close()

	if keys is None:
		progress.leave = False
		progress.close()
		return False

	progress.close()

	# Runs of equal keys are one position's count.
	n_outs = len(chroms) if args.unstranded else 2 * len(chroms)
	for positions, counts, outs, lo, hi in counter.finish(keys, n_outs):
		# Each entry names its group's arrays and its slice of them rather
		# than holding its own views, so that pickling the result back from a
		# worker process writes the arrays once instead of once per entry.
		used = numpy.flatnonzero(hi > lo)
		for out, a, b in zip(outs[used].tolist(), lo[used].tolist(),
			hi[used].tolist()):
			entry = positions, counts, a, b
			if args.unstranded:
				pos_reads[chroms[out]] = entry
			elif out % 2 == 0:
				pos_reads[chroms[out // 2]] = entry
			else:
				neg_reads[chroms[out // 2]] = entry

	del keys

	if args.verbose:
		reported = set()
		for tid in missing_order[:state[1]]:
			chrom = None if tid < 0 else references[tid]
			if chrom not in reported:
				reported.add(chrom)
				tqdm.write("{} encountered in input but not in FASTA/chrom sizes.".format(
					chrom))

	return True

# The tsv/bed reader below computes what the per-line loop in extract_reads
# computes, several times faster. It handles the common shape of these files --
# ASCII text whose lines hold a chromosome name and two integer coordinates in
# their first three whitespace-separated fields -- and gives up on anything
# else: a non-ASCII byte or control character, a lone carriage return, a line
# with fewer than three fields, a coordinate that is not a plain integer of at
# most 15 digits (a float, an exponent, inf or nan), or a read error. Giving up
# raises _UnsupportedIntervals, before anything is printed or recorded, and the
# per-line loop is then run on the whole file, so that it produces the result,
# or the error with its line number, that it always has.
#
# Each position is recorded as one int64 key, the chromosome's index times
# 2**40 plus the position offset by 2**39, so that a single sort of the keys
# groups them by chromosome and position and their run lengths are the counts.

class _UnsupportedIntervals(Exception):
	pass


_INTERVAL_BLOCK = 1 << 24
_IV_KEY_SHIFT = 40
_IV_KEY_STEP = 1 << _IV_KEY_SHIFT
_IV_KEY_OFFSET = 1 << 39
_FNV_OFFSET = 0xcbf29ce484222325 - (1 << 64)
_FNV_PRIME = 0x100000001b3


def _name_hash(data):
	"""64-bit FNV-1a of a chromosome name's bytes, as a signed integer, so that
	it matches the wrapping int64 arithmetic of _scan_intervals."""

	h = _FNV_OFFSET
	for c in data:
		h = ((h ^ c) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
		h = h - (1 << 64) if h >= (1 << 63) else h

	return h


def _scan_intervals(buf, n, i, n_lines, keys, n_keys, table_hash, table_name,
	name_data, name_offsets, name_slots, mode, pos_shift, neg_shift):
	"""Scan the lines of buf[i:n], which ends at a newline or at the end of the
	file, appending one key per recorded position to keys. Compiled with numba
	by _interval_scanner.

	Returns (status, i, n_lines, n_keys, a, b). Status 0: every line was
	scanned. Status 1: the chromosome name buf[a:b] is not in the name table,
	and status 2: keys is full; for both, i is the start of the line to resume
	from once the caller has added the name or grown keys. Status 3: the line
	at i holds something only the per-line loop handles.

	mode is 0 to record the start, 1 to record end - 1, and 2 to record both.
	"""

	mask = table_hash.shape[0] - 1
	capacity = keys.shape[0]
	prev_a = 0
	prev_b = -1
	prev_slot = -1

	while i < n:
		line_start = i
		if n_keys + 2 > capacity:
			return 2, line_start, n_lines, n_keys, 0, 0

		# Split the line on spaces and tabs, as str.split() would, keeping the
		# bounds of the first three fields. Every byte must be printable ASCII,
		# a space, a tab, or a carriage return that ends the line.
		n_fields = 0
		a0 = 0
		b0 = 0
		a1 = 0
		b1 = 0
		a2 = 0
		b2 = 0
		while True:
			while i < n:
				c = buf[i]
				if c == 9 or c == 32:
					i += 1
				elif c == 13:
					if i + 1 < n and buf[i + 1] == 10:
						i += 1
					else:
						return 3, line_start, n_lines, n_keys, 0, 0
				elif c == 10 or (c >= 33 and c <= 126):
					break
				else:
					return 3, line_start, n_lines, n_keys, 0, 0

			if i == n or buf[i] == 10:
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
			n_fields += 1

		next_i = i + 1 if i < n else n

		# Blank lines and # comments are skipped.
		if n_fields == 0 or buf[a0] == 35:
			i = next_i
			n_lines += 1
			continue

		if n_fields < 3:
			return 3, line_start, n_lines, n_keys, 0, 0

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

		# A chromosome missing from the sizes is skipped before its
		# coordinates are read, as in the per-line loop.
		if slot >= 0:
			k = a1
			negative = False
			if buf[k] == 45:
				negative = True
				k += 1
			elif buf[k] == 43:
				k += 1

			if k == b1 or b1 - k > 15:
				return 3, line_start, n_lines, n_keys, 0, 0

			start = 0
			while k < b1:
				c = buf[k]
				if c < 48 or c > 57:
					return 3, line_start, n_lines, n_keys, 0, 0

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
				return 3, line_start, n_lines, n_keys, 0, 0

			end = 0
			while k < b2:
				c = buf[k]
				if c < 48 or c > 57:
					return 3, line_start, n_lines, n_keys, 0, 0

				end = end * 10 + (c - 48)
				k += 1

			if negative:
				end = -end

			start += pos_shift
			end += neg_shift

			if mode != 1:
				if start < -_IV_KEY_OFFSET or start >= _IV_KEY_OFFSET:
					return 3, line_start, n_lines, n_keys, 0, 0

				keys[n_keys] = slot * _IV_KEY_STEP + start + _IV_KEY_OFFSET
				n_keys += 1

			if mode != 0:
				if end - 1 < -_IV_KEY_OFFSET or end - 1 >= _IV_KEY_OFFSET:
					return 3, line_start, n_lines, n_keys, 0, 0

				keys[n_keys] = slot * _IV_KEY_STEP + end - 1 + _IV_KEY_OFFSET
				n_keys += 1

		i = next_i
		n_lines += 1

	return 0, i, n_lines, n_keys, 0, 0


_interval_scanner_cache = []

def _interval_scanner():
	"""_scan_intervals compiled with numba."""

	if not _interval_scanner_cache:
		_interval_scanner_cache.append(_jit(_scan_intervals))

	return _interval_scanner_cache[0]


class _NameTable:
	"""An open-addressing table from chromosome-name bytes to a slot, the
	chromosome's index in the sizes, or -1 for a name not in the sizes. It is
	held in arrays that _scan_intervals reads; only Python adds to it."""

	def __init__(self, n):
		size = 64
		while size < 4 * n:
			size *= 2

		self.table_hash = numpy.zeros(size, dtype=numpy.int64)
		self.table_name = numpy.full(size, -1, dtype=numpy.int64)
		self.name_data = numpy.zeros(max(64, 16 * n), dtype=numpy.uint8)
		self.name_offsets = numpy.zeros(n + 17, dtype=numpy.int64)
		self.name_slots = numpy.zeros(n + 16, dtype=numpy.int64)
		self.names = {}
		self.hashes = []

	def add(self, data, slot):
		count = len(self.hashes)
		if 2 * (count + 1) > len(self.table_hash):
			size = 2 * len(self.table_hash)
			self.table_hash = numpy.zeros(size, dtype=numpy.int64)
			self.table_name = numpy.full(size, -1, dtype=numpy.int64)
			for name, h in enumerate(self.hashes):
				self._place(h, name)

		if count + 1 >= len(self.name_slots):
			self.name_slots = numpy.concatenate([self.name_slots,
				numpy.zeros(len(self.name_slots), dtype=numpy.int64)])
			self.name_offsets = numpy.concatenate([self.name_offsets,
				numpy.zeros(len(self.name_offsets), dtype=numpy.int64)])

		start = self.name_offsets[count]
		end = start + len(data)
		if end > len(self.name_data):
			self.name_data = numpy.concatenate([self.name_data,
				numpy.zeros(max(end, len(self.name_data)), dtype=numpy.uint8)])

		self.name_data[start:end] = numpy.frombuffer(data, dtype=numpy.uint8)
		self.name_offsets[count + 1] = end
		self.name_slots[count] = slot

		h = _name_hash(data)
		self.hashes.append(h)
		self.names[data] = count
		self._place(h, count)

	def _place(self, h, name):
		mask = len(self.table_hash) - 1
		j = h & mask
		while self.table_name[j] >= 0:
			j = (j + 1) & mask

		self.table_hash[j] = h
		self.table_name[j] = name

	def copy(self):
		other = _NameTable.__new__(_NameTable)
		for attr in ('table_hash', 'table_name', 'name_data', 'name_offsets',
			'name_slots'):
			setattr(other, attr, getattr(self, attr).copy())

		other.names = dict(self.names)
		other.hashes = list(self.hashes)
		return other


def _scan_lines(scan, buf, i, end, keys, n_keys, table, missing, mode, args):
	"""Run the scanner over the lines of buf[i:end], adding each name it does
	not know to `table` (as not in the sizes) and to the list `missing`, and
	growing `keys` when it fills. Returns (keys, n_keys, n_lines), or raises
	_UnsupportedIntervals at a line the per-line loop has to handle."""

	n_lines = 0
	while True:
		status, i, n_lines, n_keys, a, b = scan(buf, end, i, n_lines,
			keys, n_keys, table.table_hash, table.table_name,
			table.name_data, table.name_offsets, table.name_slots,
			mode, args.pos_shift, args.neg_shift)

		if status == 0:
			return keys, n_keys, n_lines
		elif status == 1:
			data = bytes(buf[a:b])
			if data in table.names:
				raise _UnsupportedIntervals()

			table.add(data, -1)
			missing.append(data.decode('ascii'))
		elif status == 2:
			grown = numpy.empty(2 * len(keys), dtype=numpy.int64)
			grown[:n_keys] = keys[:n_keys]
			keys = grown
		else:
			raise _UnsupportedIntervals()


# With more than one core for a file, the file is cut into units of about 8 MB
# of text, which a pool of threads scans at the same time: both isal and the
# scanner release the GIL. A BGZF file (the usual form of a fragments file) is
# a series of gzip members of at most 64 KB inflated, each with a header that
# gives its compressed size, so the members can be found without inflating
# them, and a unit of them is inflated by the thread that scans it. Each
# thread scans the lines that lie wholly inside its unit; the line cut by the
# end of one unit and the start of the next is put back together and scanned
# in order by the calling thread. Every thread has its own copy of the name
# table, and names not in the sizes are listed per unit, so that the units'
# lists, in file order and with repeats dropped, give the order in which the
# names are first encountered. A line any thread declines sends the whole file
# to the per-line loop, as in the single-threaded reader. A file that is not a
# series of BGZF blocks that each inflate whole and check out, exactly as the
# stream reader would read them, raises _NotBlockGzip, and is read by the
# single-threaded reader instead, which decides between itself and the
# per-line loop as it always has.

_IV_UNIT_BLOCKS = 128
_IV_UNIT_BYTES = 1 << 23
_IV_READ = 1 << 23
_IV_BGZF_HEADER = b'\x1f\x8b\x08\x04'
_IV_BGZF_EXTRA = b'\x06\x00BC\x02\x00'


class _NotBlockGzip(Exception):
	pass


def _bgzf_units(handle):
	"""Yield (data, spans) for runs of up to _IV_UNIT_BLOCKS whole BGZF blocks
	read from the binary file `handle`: each block is data[at:at + size] for
	(at, size) in spans. Raises _NotBlockGzip where the file is not a series
	of blocks with htslib's header, or ends inside one."""

	rest = b''
	while True:
		data = bytearray(len(rest) + _IV_READ)
		data[:len(rest)] = rest
		n_read = handle.readinto(memoryview(data)[len(rest):])
		n = len(rest) + n_read

		at = 0
		spans = []
		while at + 18 <= n:
			if (data[at:at + 4] != _IV_BGZF_HEADER or
				data[at + 10:at + 16] != _IV_BGZF_EXTRA):
				raise _NotBlockGzip()

			size = (data[at + 16] | (data[at + 17] << 8)) + 1
			if size < 26:
				raise _NotBlockGzip()
			if at + size > n:
				break

			spans.append((at, size))
			at += size
			if len(spans) == _IV_UNIT_BLOCKS:
				yield data, spans
				spans = []

		if spans:
			yield data, spans

		rest = bytes(data[at:n])
		if n_read == 0:
			if len(rest) > 0:
				raise _NotBlockGzip()
			return


def _inflate_unit(data, spans):
	"""The text of a run of BGZF blocks. Every block must be one gzip member
	that ends exactly where its header says, with a matching CRC and length;
	anything else raises _NotBlockGzip."""

	from isal import isal_zlib

	view = memoryview(data)
	parts = []
	try:
		for at, size in spans:
			inflater = isal_zlib.decompressobj(31)
			parts.append(inflater.decompress(view[at:at + size]))
			if not inflater.eof or len(inflater.unused_data) > 0:
				raise _NotBlockGzip()
	except _NotBlockGzip:
		raise
	except Exception:
		raise _NotBlockGzip()

	return bytearray().join(parts)


def _text_units(handle):
	"""Yield the bytes of an uncompressed file in pieces of at most
	_IV_UNIT_BYTES."""

	while True:
		text = bytearray(_IV_UNIT_BYTES)
		n = handle.readinto(text)
		if n == 0:
			return

		del text[n:]
		yield text


def _scan_units(filename, compressed, scan, table, mode, args, bar, threads):
	"""The keys and the missing names of a tsv/bed file, as the
	single-threaded loop in _read_intervals finds them, scanned by `threads`
	threads. Leaves `table` unchanged. Raises _UnsupportedIntervals, or
	_NotBlockGzip when the file has to be read as a stream instead."""

	from concurrent.futures import ThreadPoolExecutor

	local = threading.local()

	def work(unit, first):
		text = _inflate_unit(*unit) if compressed else unit

		# The text before the first newline ends the line the unit before
		# began; a unit without one lies inside a single line.
		start = 0
		if not first:
			start = text.find(b'\n') + 1
			if start == 0:
				return False, bytes(text), b'', None, 0, None

		end = text.rfind(b'\n') + 1

		own = getattr(local, 'table', None)
		if own is None:
			own = local.table = table.copy()

		names = []
		keys = numpy.empty((end - start) // 12 + 16, dtype=numpy.int64)
		keys, n_keys, n_lines = _scan_lines(scan, numpy.frombuffer(text,
			dtype=numpy.uint8), start, end, keys, 0, own, names, mode, args)

		return True, bytes(text[:start]), bytes(text[end:]), keys[:n_keys], n_lines, names

	parts = []
	missing = []
	carry = bytearray()
	carry_table = table.copy()

	def scan_carry():
		# A line put back together from the ends of two or more units.
		keys = numpy.empty(16, dtype=numpy.int64)
		keys, n_keys, n_lines = _scan_lines(scan, numpy.frombuffer(carry,
			dtype=numpy.uint8), 0, len(carry), keys, 0, carry_table, missing,
			mode, args)
		parts.append(keys[:n_keys])
		bar.update(n_lines)

	def collect(result):
		nonlocal carry

		split, head, tail, keys, n_lines, names = result
		carry += head
		if split:
			if len(carry) > 0:
				scan_carry()

			carry = bytearray(tail)
			parts.append(keys)
			missing.extend(names)
			bar.update(n_lines)

	pool = ThreadPoolExecutor(threads)
	try:
		with open(filename, "rb", buffering=0) as handle:
			units = _bgzf_units(handle) if compressed else _text_units(handle)

			pending = collections.deque()
			for k, unit in enumerate(units):
				pending.append(pool.submit(work, unit, k == 0))
				if len(pending) > 2 * threads:
					collect(pending.popleft().result())

			while pending:
				collect(pending.popleft().result())

		if len(carry) > 0:
			scan_carry()
	finally:
		pool.shutdown(wait=True, cancel_futures=True)

	keys = numpy.concatenate(parts) if parts else numpy.empty(0, dtype=numpy.int64)
	return keys, list(dict.fromkeys(missing))


def _read_intervals(filename, args, chroms, idx, name, threads=1):
	"""Count the positions that a tsv/bed file records, as the per-line loop in
	extract_reads would, for the chromosomes in `chroms` (the keys of
	pos_reads, in order), scanning on up to `threads` threads.

	Returns (counts, missing, bar): counts maps each chromosome with any
	position to a tuple (positions, counts, start, end), the form
	count_bam_reads gives: its sorted distinct positions and how often each
	was recorded are positions[start:end] and counts[start:end], two arrays
	shared by every chromosome; missing lists the names not in `chroms` in the order
	they were first encountered; bar is the open -v progress bar. Raises
	_UnsupportedIntervals, or whatever opening or decompressing the file
	raised, when the per-line loop must be used instead.
	"""

	scan = _interval_scanner()

	if max(abs(args.pos_shift), abs(args.neg_shift)) >= 1 << 38:
		raise _UnsupportedIntervals()
	if len(chroms) >= 1 << 22:
		raise _UnsupportedIntervals()

	table = _NameTable(len(chroms))
	for slot, chrom in enumerate(chroms):
		data = chrom.encode('utf-8')
		if len(data) > 0 and min(data) >= 33 and max(data) <= 126:
			table.add(data, slot)

	mode = 2 if args.fragments else (1 if args.three_prime else 0)
	compressed = filename[-4:] not in ('.tsv', '.bed')

	if compressed:
		from isal import igzip
		handle = igzip.open(filename, "rb")
	else:
		handle = open(filename, "rb")

	bar = _tqdm(args.verbose)(disable=not args.verbose, position=idx, desc=name)
	try:
		with handle:
			found = None
			if threads > 1:
				try:
					found = _scan_units(filename, compressed, scan, table, mode,
						args, bar, threads)
				except _NotBlockGzip:
					bar.reset()

			if found is not None:
				keys, missing = found
				n_keys = len(keys)
			else:
				missing = []
				keys = numpy.empty(1 << 22, dtype=numpy.int64)
				n_keys = 0

				pending = bytearray()
				while True:
					block = handle.read(_INTERVAL_BLOCK)
					pending += block

					if len(block) > 0:
						end = pending.rfind(b'\n') + 1
						if end == 0:
							continue
					else:
						end = len(pending)

					buf = numpy.frombuffer(pending, dtype=numpy.uint8)
					keys, n_keys, n_lines = _scan_lines(scan, buf, 0, end, keys,
						n_keys, table, missing, mode, args)

					bar.update(n_lines)
					del buf
					del pending[:end]

					if len(block) == 0:
						break

		keys = keys[:n_keys]
		keys.sort()

		counts = {}
		if n_keys > 0:
			first = numpy.empty(n_keys, dtype=bool)
			first[0] = True
			numpy.not_equal(keys[1:], keys[:-1], out=first[1:])
			starts = numpy.flatnonzero(first)
			del first

			distinct = keys[starts]
			del keys
			run_lengths = numpy.diff(starts, append=n_keys)
			del starts

			slots = distinct >> _IV_KEY_SHIFT
			positions = (distinct & (_IV_KEY_STEP - 1)) - _IV_KEY_OFFSET
			del distinct
			bounds = numpy.searchsorted(slots, numpy.arange(len(chroms) + 1))

			for slot in numpy.flatnonzero(bounds[1:] > bounds[:-1]):
				counts[chroms[slot]] = (positions, run_lengths,
					int(bounds[slot]), int(bounds[slot + 1]))
	except BaseException:
		# Nothing has been printed or recorded, so the per-line loop can
		# start over; only the progress bar has to be taken down.
		bar.leave = False
		bar.close()
		raise

	return counts, missing, bar


# When a file has a core to itself beyond the one parsing it, a gzipped text
# file is decompressed in a background thread, which overlaps the parsing
# because zlib releases the GIL while it inflates. The lines must come out
# exactly as iterating over gzip.open(filename, "rt") gives them, errors
# included, so the background reader only hands on data that the stdlib reader
# would also return before any error of its own: it is at least as strict
# (zlib checks every member's header, CRC and length; a member may be followed
# only by zero padding or another member), every piece it decompresses is
# checked to decode, and the last 64 KB it has decompressed are held back. A
# TextIOWrapper reading gzip.open decompresses and decodes in pieces of at most
# 8 KB, so when the background reader fails for any reason, every line it has
# handed on is one the stdlib reader also yields before its own failure. The
# file is then re-read with gzip.open, those lines are skipped, and the rest
# (with the stdlib's own exception, if there is one) come from it.

class _GunzipFailed(Exception):
	pass

class _QueueReader(io.BufferedIOBase):
	"""The read side of _gunzip: hands on, in order, the pieces it queues."""

	def __init__(self, pieces):
		self._pieces = pieces
		self._piece = memoryview(b'')
		self._offset = 0
		self._eof = False

	def readable(self):
		return True

	def read1(self, size=-1):
		while self._offset >= len(self._piece):
			if self._eof:
				return b''

			piece = self._pieces.get()
			if piece is None:
				raise _GunzipFailed()
			if len(piece) == 0:
				self._eof = True
				return b''

			self._piece = memoryview(piece)
			self._offset = 0

		if size is None or size < 0:
			size = len(self._piece)

		chunk = self._piece[self._offset:self._offset + size]
		self._offset += len(chunk)
		return chunk

	read = read1

def _gunzip(handle, pieces, encoding, stop):
	"""Decompress a gzip file into the queue `pieces`, ending with b'' after
	the whole file or with None on any failure."""

	def put(item):
		while not stop.is_set():
			try:
				pieces.put(item, timeout=0.1)
				return
			except queue.Full:
				pass

		raise _GunzipFailed()

	try:
		decoder = codecs.getincrementaldecoder(encoding)()
		clean = True
		held = collections.deque()
		held_bytes = 0
		decompressor = zlib.decompressobj(wbits=31)
		member_done = False

		while True:
			data = handle.read(1 << 20)
			if not data:
				break

			while data:
				if member_done:
					data = data.lstrip(b'\x00')
					if not data:
						break

					decompressor = zlib.decompressobj(wbits=31)
					member_done = False

				out = decompressor.decompress(data, 1 << 23)
				if decompressor.eof:
					data = decompressor.unused_data
					member_done = True
				else:
					data = decompressor.unconsumed_tail

				if len(out) == 0:
					continue

				if not (clean and out.isascii()):
					decoder.decode(out)
					clean = not decoder.getstate()[0]

				held.append(out)
				held_bytes += len(out)
				while held_bytes - len(held[0]) >= 1 << 16:
					held_bytes -= len(held[0])
					put(held.popleft())

		# A file that ends inside a member, or holds none, goes to gzip.open.
		if not member_done:
			raise _GunzipFailed()

		decoder.decode(b'', True)
		for piece in held:
			put(piece)
		put(b'')

	except BaseException:
		try:
			put(None)
		except _GunzipFailed:
			pass

	finally:
		handle.close()

def _gzip_lines(filename, handle):
	pieces = queue.Queue(8)
	stop = threading.Event()

	text = io.TextIOWrapper(_QueueReader(pieces))
	text._CHUNK_SIZE = 1 << 20

	thread = threading.Thread(target=_gunzip, args=(handle, pieces,
		text.encoding, stop), daemon=True)
	thread.start()

	n = 0
	try:
		for line in text:
			yield line
			n += 1

		return
	except Exception:
		pass
	finally:
		stop.set()

	replay = gzip.open(filename, "rt")
	for _ in itertools.islice(replay, n):
		pass

	yield from replay

def open_gzip_text(filename, threads):
	"""Returns the lines of a gzipped text file, as gzip.open(filename, "rt")
	does, decompressing in a background thread when threads > 1."""

	if threads < 2:
		return gzip.open(filename, "rt")

	# The background reader checks that each piece decodes, which is only
	# known to be cheap and exact for UTF-8 (ASCII data needs no decoding).
	encoding = io.TextIOWrapper(io.BytesIO()).encoding
	if codecs.lookup(encoding).name != 'utf-8':
		return gzip.open(filename, "rt")

	# Opened here rather than in the generator so that a missing file fails
	# at the same point, and with the same error, as gzip.open.
	return _gzip_lines(filename, open(filename, "rb"))

def extract_reads(args, chrom_sizes, idx, threads=1):
	tqdm = _tqdm(args.verbose)
	missing_chroms = set()
	pos_reads = {}
	neg_reads = pos_reads if args.unstranded else {}

	# Create dictionaries for each chrom, regardless
	for chrom, _ in chrom_sizes:
		pos_reads[chrom] = collections.defaultdict(int)
		neg_reads[chrom] = collections.defaultdict(int) # Redundant if unstranded

	# Extract the file being considered	
	filename = args.filename[idx]
	name = filename.split("/")[-1]
	if filename.endswith((".bam", ".sam")):
		# pysam has to be told which of the two it is -- "rb" is binary BAM
		# and "r" is text SAM. Neither needs an index because the loop below
		# fetches until_eof.
		import pysam

		mode = "rb" if filename.endswith(".bam") else "r"
		bam = pysam.AlignmentFile(filename, mode)

		if mode == "rb" and not args.mate_pairs:
			if count_bam_reads(bam, filename, chrom_sizes, pos_reads, neg_reads,
				args, idx, name):
				bam.close()
				return pos_reads, neg_reads

		pending_mates = {}

		# These are read once per read in the loop below, which is hot enough
		# that the attribute lookups show up in a profile.
		mate_pairs = args.mate_pairs
		fragments = args.fragments
		use_three_prime = args.three_prime

		for read in tqdm(bam.fetch(until_eof=True), disable=not args.verbose, position=idx, desc=name):
			if read.is_unmapped:
				continue

			# Check whether the chrom is in the allowable chroms. Otherwise, discard.

			chrom = read.reference_name
			if chrom not in pos_reads:
				if chrom not in missing_chroms:
					missing_chroms.add(chrom)
					if args.verbose:
						tqdm.write("{} encountered in input but not in FASTA/chrom sizes.".format(
							chrom))

				continue

			# A mapped record whose CIGAR is '*' has no reference end, so
			# there is no position to record for the far end of the read.
			# reference_end is computed from the CIGAR on every access, so it
			# is read once here and reused below.
			reference_end = read.reference_end
			if reference_end is None:
				raise ValueError(("{}: read {} is mapped to {} but has no "
					"CIGAR, so its reference end is unknown.").format(name,
						read.query_name, chrom))

			start = read.reference_start + args.pos_shift
			end = reference_end + args.neg_shift

			if mate_pairs:
				# Only consider one alignment per mate so that the two halves
				# of a pair can be matched up unambiguously by read name.
				if not read.is_proper_pair or read.is_secondary or read.is_supplementary:
					continue

				partner = pending_mates.pop(read.query_name, None)
				mate_info = MateInfo(start, end, read.is_forward)

				if partner is None or partner[0] == read.is_read1:
					# Either the first mate seen for this name, or an
					# unexpected duplicate record for the same mate -- in the
					# latter case the stale entry is dropped in favor of this one.
					pending_mates[read.query_name] = (read.is_read1, mate_info)
					continue

				_, partner_info = partner
				read1_info = mate_info if read.is_read1 else partner_info
				read2_info = partner_info if read.is_read1 else mate_info

				if args.rna5 == 'read1':
					anchor, other = read1_info, read2_info
				else:
					anchor, other = read2_info, read1_info

				record_mate_pair(args, pos_reads, neg_reads, chrom, anchor, other)
			else:
				# Here, we need to deal with two related issues.
				#
				#    (1) Does the read map to the fwd or rev strand?
				#    (2) Are we mapping the start or the strand and the end (fragments)?
				#
				# Accordingly, we first check to see the strand the read is on and take the
				# start of the read (start for fwd, end-1 for bwd reads). Then, we need to
				# check whether we want both starts and ends and record both if so. This
				# strategy works even if the underlying data is not stranded because
				# pos_reads and neg_reads are the same dictionary in that case.

				if read.is_forward:
					five_prime, three_prime = start, end - 1
					reads = pos_reads
				else:
					five_prime, three_prime = end - 1, start
					reads = neg_reads

				reads[chrom][three_prime if use_three_prime else five_prime] += 1
				if fragments:
					reads[chrom][five_prime if use_three_prime else three_prime] += 1

		bam.close()
	
	elif filename[-4:] in ('.tsv', '.bed') or filename[-7:] in ('.tsv.gz', '.bed.gz'):
		# The fast reader above handles the common case. Anything it does not
		# handle, including every malformed file, falls through to the
		# per-line loop below, which then runs on the whole file.
		try:
			intervals = _read_intervals(filename, args, list(pos_reads), idx, name,
				threads)
		except Exception:
			intervals = None

		if intervals is not None:
			counts, missing, bar = intervals
			pos_reads.update(counts)

			if args.verbose:
				for chrom in missing:
					tqdm.write("{} encountered in input but not in FASTA/chrom sizes.".format(
						chrom))

			bar.close()
			return pos_reads, neg_reads

		# Open the file using the correct opener -- the standard one if the file is
		# not compressed, otherwise the gzip opener if gzipped.
		
		if filename[-4:] in ('.tsv', '.bed'):
			f = open(filename, "r")
		elif filename[-7:] in ('.tsv.gz', '.bed.gz'):
			f = open_gzip_text(filename, threads)
		
		# Here, we process the entries in a similar manner to using pysam except that
		# we assume the coordinates are all fwd strand. We do not explicitly assume
		# that we want both the start and the end of the entry, which is controlled
		# using the -f flag, but we do not handle strandedness here.

		for i, line in enumerate(tqdm(f, disable=not args.verbose, position=idx,
			desc=name)):
			fields = line.split()

			# Blank lines and # comments are skipped, matching the chrom_sizes
			# reader above. A 10x CellRanger fragments file opens with a #
			# header, so rejecting it would fail on the most common input.
			# Testing fields[0] rather than the raw line costs no allocation on
			# this per-read path and still catches a # behind leading
			# whitespace; split() never yields an empty field, so [0][0] is safe.

			if len(fields) == 0 or fields[0][0] == '#':
				continue

			if len(fields) < 3:
				raise ValueError(("{}, line {}: expected at least three columns "
					"holding a chromosome, a start and an end, got {}.").format(
						filename, i + 1, repr(line.strip())))

			# Check whether the chrom is in the allowable chroms. Otherwise, discard.

			chrom, start, end = fields[:3]
			if chrom not in pos_reads:
				if chrom not in missing_chroms:
					missing_chroms.add(chrom)
					if args.verbose:
						tqdm.write("{} encountered in input but not in FASTA/chrom sizes.".format(
							chrom))

				continue

			try:
				start = int(float(start)) + args.pos_shift
				end = int(float(end)) + args.neg_shift
			except ValueError:
				raise ValueError(("{}, line {}: expected numeric coordinates, "
					"got {} and {}.").format(filename, i + 1, repr(start),
						repr(end)))

			pos_reads[chrom][end-1 if args.three_prime else start] += 1
			if args.fragments:
				pos_reads[chrom][start if args.three_prime else end-1] += 1

	return pos_reads, neg_reads

### Collect the reads into a single object.

# Each file's counts for one chromosome and strand are either a dictionary of
# position -> count or, from count_bam_reads and _read_intervals, a slice of
# arrays holding sorted positions and their counts. Both are brought to a tuple of sorted positions
# and counts, and the files are summed. Each dictionary is dropped as soon as
# it has been converted.

def as_arrays(counts):
	if isinstance(counts, tuple):
		positions, values, start, end = counts
		return positions[start:end], values[start:end]

	positions = numpy.fromiter(counts.keys(), dtype='int64', count=len(counts))
	values = numpy.fromiter(counts.values(), dtype='int64', count=len(counts))
	idxs = numpy.argsort(positions)
	return positions[idxs], values[idxs]

def merge_counts(parts):
	parts = [part for part in parts if len(part[0]) > 0]
	if len(parts) == 0:
		return numpy.empty(0, dtype='int64'), numpy.empty(0, dtype='int64')
	elif len(parts) == 1:
		return parts[0]

	positions = numpy.concatenate([positions for positions, _ in parts])
	values = numpy.concatenate([values for _, values in parts])
	idxs = numpy.argsort(positions, kind='stable')
	positions, values = positions[idxs], values[idxs]

	starts = numpy.flatnonzero(numpy.concatenate([[True],
		positions[1:] != positions[:-1]]))
	return positions[starts], numpy.add.reduceat(values, starts)

###

# The counts are written by figwig's BigWigWriter, as single bases, which it
# writes as varStep sections: two numbers per position rather than the three
# of a bedGraph item. A base whose value is NaN is all BigWigWriter leaves
# out, so that a scale factor of 0 writes zeros, as pyBigWig does. Chromosomes
# go to the writer in batches of at least _WRITE_ITEMS positions, with their
# ids rather than a name per position, so that 50,000 small contigs take a few
# calls rather than one each.

_WRITE_ITEMS = 1 << 20
_MISSING = numpy.float32(numpy.nan)

# The compression level of the blocks. libdeflate's level 1 took about 0.4 s
# less than its level 6 to write the counts of a 2.4 GB ATAC BAM at -p 2, for
# files 3.5% larger, and unlike ISA-L's level 1, which the speed search used,
# gives the same bytes on every run.
_LEVEL = 1

def write_counts(writer, chrom_sizes, reads, scale_factor, read_depth):
	"""Write each chromosome's counts, reads[chrom] = (sorted positions,
	counts), times scale_factor and divided by read_depth unless it is None."""

	names, starts, counts = [], [], []
	n = 0

	def flush():
		values = numpy.concatenate(counts).astype('float64')
		values *= scale_factor

		if read_depth is not None:
			values /= read_depth

		codes = numpy.repeat(numpy.arange(len(names)), [len(s) for s in starts])
		writer._write_items(names, codes, numpy.concatenate(starts),
			values[:, None], None, _MISSING)

	for chrom in dict.fromkeys(chrom for chrom, _ in chrom_sizes):
		positions, chrom_counts = reads[chrom]
		if len(positions) == 0:
			continue

		names.append(chrom)
		starts.append(positions)
		counts.append(chrom_counts)
		n += len(positions)

		if n >= _WRITE_ITEMS:
			flush()
			names, starts, counts = [], [], []
			n = 0

	if n > 0:
		flush()


def main(argv=None, prog='figwig bam2bw', fast_exit=False):
	"""Run bam2bw with the arguments `argv`, by default sys.argv[1:].

	With fast_exit=True, the process ends once the files are written and
	stdout and stderr are flushed, as `figwig bam2bw` does (see
	`_exit_now`).
	"""

	args = _parser(prog).parse_args(argv)

	missing = [name for name in _REQUIRED if importlib.util.find_spec(name) is None]
	if len(missing) > 0:
		raise SystemExit(("{} needs {}, from figwig's bam2bw extra: pip install "
			"'figwig[bam2bw]'").format(prog, ', '.join(missing)))

	_check_filenames(args)
	chrom_sizes = read_chrom_sizes(args.sizes)
	tqdm = _tqdm(args.verbose)

	# Share a single lock across the worker processes so that each file's tqdm
	# progress bar renders on its own line (via position=idx) instead of the
	# processes clobbering each other's cursor movements on stdout. The lock is
	# created before Parallel so it is inherited by the forked workers. Without
	# -v there is no bar to render.
	if args.verbose:
		import multiprocessing
		tqdm.set_lock(multiprocessing.RLock())

	# One process per file, never more processes than files, and a single job runs
	# in this process: starting a pool for it would only add the cost of pickling
	# its result back. The cores -p leaves over go to each file's decompression.
	# A -p below 1 keeps joblib's own meaning.

	n_files = len(args.filename)

	if args.parallel < 1:
		from joblib import Parallel, delayed

		f = delayed(extract_reads)
		reads = Parallel(n_jobs=args.parallel, backend='multiprocessing')(
			f(args, chrom_sizes, i) for i in range(n_files)
		)
	else:
		n_jobs = min(args.parallel, n_files)
		threads = args.parallel // n_jobs

		if n_jobs == 1:
			reads = [extract_reads(args, chrom_sizes, i, threads) for i in range(n_files)]
		else:
			from joblib import Parallel, delayed

			f = delayed(extract_reads)
			reads = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
				f(args, chrom_sizes, i, threads) for i in range(n_files)
			)

	# The persistent per-file bars leave the cursor part-way up the screen, so
	# drop below them before printing anything else.
	if args.verbose:
		print("\n" * len(args.filename))

	pos_reads, neg_reads = {}, {}
	for chrom in dict.fromkeys(chrom for chrom, _ in chrom_sizes):
		pos_reads[chrom] = merge_counts([as_arrays(pos_reads_.pop(chrom))
			for pos_reads_, _ in reads])

		if not args.unstranded:
			neg_reads[chrom] = merge_counts([as_arrays(neg_reads_.pop(chrom))
				for _, neg_reads_ in reads])

	del reads
	if args.unstranded:
		neg_reads = pos_reads

	###

	# A bigWig can only hold positions that fall inside the chromosome it declares,
	# and pyBigWig discards anything else without raising or returning an error.
	# That silently removes reads from the output -- either because the sizes file
	# disagrees with the BAM header, or because a shift pushed an end off the edge
	# of a chromosome -- and, worse, read depth would be summed over reads that
	# never reach the file, leaving the track normalized to less than it claims.
	# Discarding them here instead means it can be counted, reported, and excluded
	# from the read depth below.

	discarded = collections.defaultdict(int)

	for chrom, size in chrom_sizes:
		strands = [pos_reads]
		if not args.unstranded:
			strands.append(neg_reads)

		for reads_ in strands:
			idxs, counts = reads_[chrom]
			lo, hi = numpy.searchsorted(idxs, [0, size])

			if lo > 0 or hi < len(idxs):
				discarded[chrom] += int(counts[:lo].sum()) + int(counts[hi:].sum())
				reads_[chrom] = idxs[lo:hi], counts[lo:hi]

	if len(discarded) > 0:
		print("{} entries fell outside the provided chromosome sizes and were discarded ({}).".format(
			sum(discarded.values()), ", ".join("{}: {}".format(chrom, count)
				for chrom, count in discarded.items())))

	###

	# Here, we open the bigWigs that we will be saving data into. If the data is
	# stranded, we are saving two bigWigs. If the data is not stranded, we are only
	# saving one bigWig. Each compresses its blocks on up to -p threads, and the
	# second is written after the first is closed, so that the two do not
	# compress at the same time.

	def open_bigwig(path):
		return BigWigWriter(path, chrom_sizes, zooms=args.zooms, level=_LEVEL,
			n_jobs=max(1, args.parallel))

	if args.unstranded:
		outputs = [(open_bigwig(args.name + ".bw"), pos_reads)]
	else:
		outputs = [(open_bigwig(args.name + ".+.bw"), pos_reads)]
		outputs.append((open_bigwig(args.name + ".-.bw"), neg_reads))

	try:
		read_depth = None
		if args.read_depth:
			read_depth = sum([int(pos_reads[chrom][1].sum()) for chrom, _ in chrom_sizes])
			if not args.unstranded:
				read_depth += sum([int(neg_reads[chrom][1].sum()) for chrom, _ in chrom_sizes])

			if args.verbose:
				print("Dividing through by a read depth of {}.".format(read_depth))

		for writer, reads_ in outputs:
			write_counts(writer, chrom_sizes, reads_, args.scale_factor, read_depth)
			writer.close()
	except BaseException as error:
		for writer, _ in outputs:
			writer.__exit__(type(error), error, error.__traceback__)
		raise

	if fast_exit:
		_exit_now(args)


# After a numba kernel has been loaded, Python's own shutdown takes 40-70 ms
# longer than ending the process at once. Once the files are closed and
# stdout and stderr are flushed it has nothing visible left to do, so the
# process ends here. Python shuts down as usual instead
# with -v, which may leave a bar for shutdown to finish; after worker
# processes, whose clean-up runs at exit; while a non-daemon thread is alive;
# or when a flush fails.

def _exit_now(args):
	if args.verbose or 'joblib' in sys.modules or 'multiprocessing' in sys.modules:
		return

	for thread in threading.enumerate():
		if thread is not threading.main_thread() and not thread.daemon:
			return

	try:
		sys.stdout.flush()
		sys.stderr.flush()
	except BaseException:
		return

	os._exit(0)
