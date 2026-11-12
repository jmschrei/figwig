# fiber.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The Fiber-seq row of the README's table: one K562 Fiber-seq BAM, written,
read and sized.

Write: `figwig fiberbam2fw` with its defaults (all five channels, uint8
tracks, groups of 1,024 reads) on `--threads` threads against a single-thread
pysam loop that places every m6A and 5mC probability on the reference with
`modified_bases` and `get_reference_positions`, writes nothing, and collects
the m6A calls of every read into a scipy CSR matrix of positions x reads per
chromosome (the loop's time includes sorting the calls into the matrix, not
saving them).

Read: the m6A probabilities summed over all reads in 8,192 windows of 2,048 bp
on random reads (`labels=`, one label for every read), from the m6A tracks of
the file `fiberbam2fw` wrote, against the CSR held in memory, given its
fastest of 1 and `--threads` threads. The file's m6A tracks hold the 0 to 255
probabilities, so the matrix holds them too and the two sums are equal; the
windows' arrays are compared by a hash, and every read's total over its span
and every chromosome's total are compared between the file and the pysam pass.

Size: the figwig file (five channels) against the BAM; the matrix's size in
memory goes in a note.

The test uses the `large` BAM; `--quick` uses the `small` one.
"""

import os
import json
import collections
import time
import hashlib
import statistics
import subprocess
import concurrent.futures

import numpy

import common
import render

LARGE = 'fiber-seq/large/fiber-seq_large_K562_hg38_long.bam'
SMALL = 'fiber-seq/small/fiber-seq_small_K562_hg38_long.bam'
N_WINDOWS = 8192
WIDTH = 2048


def paths(ctx):
	root = os.path.join(ctx['scratch'], 'fiber')
	return {'root': root, 'fw': os.path.join(root, 'fibers.fw'), 'coo': os.path.join(root, 'coo'),
		'reads': os.path.join(root, 'reads.npz')}


def figwig_write(ctx, bam):
	"""Timed: the fiberbam2fw command with its defaults, interpreter start
	included."""

	p = paths(ctx)
	os.makedirs(p['root'], exist_ok=True)
	common.fresh(p['fw'])
	command = [ctx['python'], '-m', 'figwig', 'fiberbam2fw', '-n', p['fw'][:-3], '-g', 'hg38', '-p',
		str(ctx['threads']), '--force', bam]
	start = time.perf_counter()
	result = subprocess.run(command, capture_output=True, text=True)
	seconds = time.perf_counter() - start
	if result.returncode != 0:
		raise RuntimeError(result.stderr[-2000:])
	return {'seconds': seconds, 'mb': common.size_mb(p['fw']), 'command': ' '.join(command[2:])}


def pysam_pass(ctx, bam):
	"""Timed: place every m6A and 5mC probability on the reference, and sort
	the m6A calls of each chromosome into CSR arrays (positions x reads).
	The arrays are saved afterwards for the read workers."""

	import pysam
	import scipy.sparse

	p = paths(ctx)
	os.makedirs(p['coo'], exist_ok=True)
	start = time.perf_counter()
	calls = {}
	spans = []
	totals = []
	m6a = cpg = reads = 0
	with pysam.AlignmentFile(bam) as handle:
		lengths = dict(zip(handle.references, handle.lengths))
		for read in handle.fetch(until_eof=True):
			if read.is_unmapped or read.is_secondary or not read.has_tag('MM'):
				continue
			refs = numpy.array(read.get_reference_positions(full_length=True), dtype=numpy.float64)
			chrom = read.reference_name
			number = len(spans)
			total = 0
			for (_, _, code), items in read.modified_bases.items():
				if code not in ('a', 'm') or not items:
					continue
				items = numpy.array(items)
				at = refs[items[:, 0]]
				keep = ~numpy.isnan(at) & (items[:, 1] > 0)
				if code == 'm':
					cpg += int(keep.sum())
					continue
				m6a += int(keep.sum())
				total += int(items[keep, 1].sum())
				calls.setdefault(chrom, []).append((at[keep].astype(numpy.int32),
					numpy.full(int(keep.sum()), number, dtype=numpy.int32),
					items[keep, 1].astype(numpy.uint8)))
			spans.append((read.query_name, chrom, read.reference_start, read.reference_end))
			totals.append(total)
			reads += 1
	matrices = {}
	for chrom, parts in calls.items():
		pos = numpy.concatenate([a for a, _, _ in parts])
		read = numpy.concatenate([b for _, b, _ in parts])
		value = numpy.concatenate([c for _, _, c in parts])
		order = numpy.argsort(pos, kind='stable')
		indptr = numpy.zeros(lengths[chrom] + 1, dtype=numpy.int32)
		numpy.cumsum(numpy.bincount(pos, minlength=lengths[chrom]), out=indptr[1:])
		matrices[chrom] = scipy.sparse.csr_matrix((value[order], read[order], indptr),
			shape=(lengths[chrom], len(spans)))
	seconds = time.perf_counter() - start

	for chrom, m in matrices.items():
		numpy.savez(os.path.join(p['coo'], chrom + '.npz'), data=m.data, indices=m.indices, indptr=m.indptr)
	names, chroms, starts, ends = zip(*spans)
	numpy.savez(p['reads'], names=numpy.array(names), chroms=numpy.array(chroms),
		starts=numpy.array(starts), ends=numpy.array(ends), totals=numpy.array(totals),
		lengths=json.dumps(lengths))
	rng = numpy.random.default_rng(0)
	# reads on chromosomes at least one window long
	eligible = numpy.flatnonzero([lengths[c] >= WIDTH for c in chroms])
	picked = rng.choice(eligible, N_WINDOWS, replace=len(eligible) < N_WINDOWS)
	# a window past the end of its chromosome is NaN in figwig's float sums
	win_starts = [int(min(rng.integers(starts[i], max(ends[i] - WIDTH, starts[i]) + 1),
		lengths[chroms[i]] - WIDTH)) for i in picked]
	numpy.savez(os.path.join(p['root'], 'windows.npz'), names=numpy.array([chroms[i] for i in picked]),
		starts=numpy.array(win_starts, dtype=numpy.int64))
	return {'seconds': seconds, 'alignments': reads, 'm6a_calls_placed': m6a, 'cpg_calls_placed': cpg,
		'matrix_mb': sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in matrices.values()) / 1e6}


def check(ctx):
	"""Untimed: every read's m6A total over its span in the file against the
	pysam pass's, and the chromosomes' totals."""

	import figwig

	p = paths(ctx)
	z = numpy.load(p['reads'])
	reader = figwig.FigWigReader(p['fw'])
	selected = [i for i, t in enumerate(reader.tracks) if t['channel'] == 'm6a']
	tracks = [reader.tracks[i] for i in selected]
	sums = reader.sums([t['chrom'] for t in tracks], [t['start'] for t in tracks],
		[t['end'] for t in tracks], tracks=selected, paired=True, n_jobs=ctx['threads'])
	theirs = sorted((t['read_name'], t['chrom'], t['start'], float(v)) for t, v in zip(tracks, sums))
	mine = sorted((str(n), str(c), int(st), float(tt)) for n, c, st, tt in zip(z['names'], z['chroms'],
		z['starts'], z['totals']))
	count_theirs, count_mine = collections.Counter(theirs), collections.Counter(mine)
	only_figwig = sum((count_theirs - count_mine).values())
	only_pysam = sum((count_mine - count_theirs).values())
	return {'figwig_reads': len(theirs), 'pysam_reads': len(mine), 'only_figwig': only_figwig,
		'only_pysam': only_pysam, 'figwig_total': sum(r[3] for r in theirs),
		'pysam_total': sum(r[3] for r in mine),
		'first_differing': [list(r) for r in list((count_mine - count_theirs).elements())[:3]]}


def digest(y):
	return hashlib.sha256(numpy.ascontiguousarray(y, dtype=numpy.float64).tobytes()).hexdigest()[:16]


def figwig_read(ctx, windows_threads):
	"""Timed: the sum over all reads of 8,192 windows from the m6A tracks of
	the open file, after a warm-up read of 1,000 windows. Opening the file and
	selecting the tracks (`open_seconds`) is timed apart, as the matrix's
	loading is left out of the baseline's time."""

	import figwig

	p = paths(ctx)
	z = numpy.load(os.path.join(p['root'], 'windows.npz'))
	names, starts = z['names'], z['starts']
	start = time.perf_counter()
	reader = figwig.FigWigReader(p['fw'])
	tracks = [i for i, t in enumerate(reader.tracks) if t['channel'] == 'm6a']
	labels = numpy.zeros(len(tracks), dtype=numpy.int64)
	open_seconds = time.perf_counter() - start

	def read(count):
		return reader.read(names[:count], starts[:count], WIDTH, tracks=tracks, labels=labels,
			n_jobs=windows_threads)

	read(1000)
	start = time.perf_counter()
	y = read(len(names))
	seconds = time.perf_counter() - start
	return {'seconds': seconds, 'open_seconds': open_seconds, 'digest': digest(y), 'sum': float(y.sum()),
		'shape': list(y.shape)}


def scipy_read(ctx, windows_threads):
	"""Timed: the same sums from the matrices held in memory, the windows
	split over a thread pool when `windows_threads` is above 1."""

	import scipy.sparse

	p = paths(ctx)
	z = numpy.load(p['reads'])
	lengths = json.loads(str(z['lengths']))
	n_reads = len(z['names'])
	matrices = {}
	for chrom in lengths:
		path = os.path.join(p['coo'], chrom + '.npz')
		if os.path.exists(path):
			m = numpy.load(path)
			matrices[chrom] = scipy.sparse.csr_matrix((m['data'], m['indices'], m['indptr']),
				shape=(lengths[chrom], n_reads))
	w = numpy.load(os.path.join(p['root'], 'windows.npz'))
	names, starts = w['names'], w['starts']

	def window(out, row, start):
		matrix = matrices.get(names[row])
		a, b = max(start, 0), min(start + WIDTH, lengths[names[row]])
		if matrix is None or a >= b:
			return
		sub = matrix[a:b]
		rows = numpy.repeat(numpy.arange(b - a), numpy.diff(sub.indptr))
		out[row, 0, a - start:b - start] = numpy.bincount(rows, weights=sub.data, minlength=b - a)

	def read(count):
		out = numpy.zeros((count, 1, WIDTH), dtype=numpy.float64)
		pieces = numpy.array_split(numpy.arange(count), windows_threads)
		with concurrent.futures.ThreadPoolExecutor(windows_threads) as pool:
			list(pool.map(lambda rows: [window(out, int(r), int(starts[r])) for r in rows], pieces))
		return out

	read(min(1000, len(names)))
	start = time.perf_counter()
	y = read(len(names))
	seconds = time.perf_counter() - start
	return {'seconds': seconds, 'digest': digest(y), 'sum': float(y.sum()), 'shape': list(y.shape)}


def fiber(ctx):
	quick = ctx['quick']
	bam = common.testdata(SMALL if quick else LARGE)
	threads = ctx['threads']
	raw = {}

	fw_seconds, fw_runs = common.repeated(ctx, 'fiber:figwig_write', threads=threads, bam=bam)
	raw['figwig_write'] = fw_runs
	fw_mb = fw_runs[-1]['mb']
	py_seconds, py_runs = common.repeated(ctx, 'fiber:pysam_pass', threads=1, bam=bam)
	raw['pysam_write'] = py_runs
	matrix_mb = py_runs[-1]['matrix_mb']
	same = common.worker(ctx, 'fiber:check', threads=threads)
	raw['same_values'] = same
	if same['only_figwig'] or same['only_pysam']:
		raise AssertionError('figwig and the pysam pass disagree: {}'.format(same))

	fr_seconds, fr_runs = common.repeated(ctx, 'fiber:figwig_read', threads=threads,
		windows_threads=threads)
	raw['figwig_read'] = fr_runs
	scipy_seconds, scipy_runs = {}, {}
	for t in sorted({1, threads}):
		scipy_seconds[t], scipy_runs[t] = common.repeated(ctx, 'fiber:scipy_read', threads=t,
			windows_threads=t)
		raw['scipy_read_{}_threads'.format(t)] = scipy_runs[t]
	best = min(scipy_seconds, key=scipy_seconds.get)
	digests = {r['digest'] for r in fr_runs} | {r['digest'] for rs in scipy_runs.values() for r in rs}
	if len(digests) != 1:
		raise AssertionError('figwig and the scipy matrix gave different sums: {}'.format(digests))
	raw['scipy_seconds_by_threads'] = scipy_seconds

	open_seconds = statistics.median(r['open_seconds'] for r in fr_runs)
	alignments = py_runs[-1]['alignments']
	bam_mb = common.size_mb(bam)
	notes = [
		'Fiber-seq: figwig and the pysam pass agree on the m6A total of each of the {:,} alignments over '
		'its span, and on the sums over the windows (the file and the matrix both hold the 0 to 255 '
		'probabilities).'.format(same['figwig_reads']),
		'The pysam loop places {:,} m6A and {:,} 5mC calls and keeps only the m6A calls, in a scipy matrix '
		'of {} that the read time leaves out building; the figwig file also holds the 5mC, nucleosome, '
		'MSP and FIRE channels.'.format(py_runs[-1]['m6a_calls_placed'], py_runs[-1]['cpg_calls_placed'],
		render.megabytes(matrix_mb))]
	notes.append('Reading times leave out opening the figwig file, which takes {} here, and loading '
		'the scipy matrix.'.format(render.seconds(open_seconds)))
	return {
		'setting': 'K562 Fiber-seq BAM, {:,} alignments, {}; reads: m6A summed over all reads in {:,} '
			'windows of {:,} bp{}'.format(alignments, render.megabytes(bam_mb), N_WINDOWS, WIDTH,
			'; quick tier' if quick else ''),
		'baseline': 'a pysam loop with `modified_bases` (write), the BAM (size) and a scipy CSR matrix '
			'in memory (read)',
		'read': [common.item(fr_seconds, scipy_seconds[best], 's',
			baseline_name='scipy CSR, {} thread{}'.format(best, '' if best == 1 else 's'))],
		'write': [common.item(fw_seconds, py_seconds, 's',
			baseline_name='pysam, 1 thread, m6A and 5mC placed, nothing written')],
		'size': [common.item(fw_mb, bam_mb, 'MB', baseline_name='the BAM')],
		'inputs': {'bam': os.path.relpath(bam, common.TESTDATA), 'alignments': alignments,
			'windows': N_WINDOWS, 'width': WIDTH, 'figwig_threads': threads, 'pysam_threads': 1,
			'repeats': ctx['repeats']},
		'raw': raw, 'notes': notes}


ROWS = {'fiber': fiber}
