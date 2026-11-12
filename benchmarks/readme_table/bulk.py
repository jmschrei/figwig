# bulk.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The bulk rows of the README table: figwig against the bigWig tools most
people use, on one, 48 and 1,024 count tracks.

	bulk-1     one ATAC-seq bigWig, 1,000 bp windows on peaks, read by
	           pybigtools, written by pyBigWig
	bulk-48    48 count tracks over chr1 (six real, the rest rolled copies as
	           in compare_formats.py), 2,048 bp windows; the figwig file against
	           48 bigWigs read by pybigtools and written by pyBigWig
	bulk-1024  the same with 1,024 tracks; pyBigWig is not run, so the baseline
	           write and size are the 1,024 bigWigs figwig writes

The measurements are `compare_readers.py`, `compare_writers.py` and
`compare_formats.py`'s own functions, called in a worker process each, so
every timing has its own interpreter. Reads of several bigWigs by pybigtools
are `compare_readers.read_pybigtools` once per track, on a thread pool of
window chunks, timed with the files' opening and hashed as `compare_formats`
hashes what it reads.
"""

import io
import os
import sys
import gzip
import json
import time
import shutil
import hashlib
import argparse
import contextlib
import subprocess

import numpy

import common

BENCH = os.path.dirname(common.HERE)
PREP_THREADS = 16
ZOOMS = 10
CHR1 = 248956422

# bulk-1's full-size signal: the cut counts of this BAM, made by bam2bw.
ATAC_BAM = 'atac/full/atac_full_HCT116_hg38_PE101.bam'
HG38_SIZES = 'reference/hg38/hg38.primary.chrom.sizes'

# Quick runs use the medium tier and fewer tracks.
ATAC = {'full': ('atac/large/atac_large_K562_hg38_PE_cut-counts.bigWig',
		'atac/large/atac_large_K562_hg38_PE_IDR-conservative_ENCFF855PCP.narrowPeak.gz'),
	'quick': ('atac/medium/atac_medium_K562_hg38_PE_cut-counts.bigWig',
		'atac/medium/atac_medium_K562_hg38_PE_pseudorep.narrowPeak.gz')}
COUNT_BAMS = {'full': ['dnase/full/dnase_full_K562_hg38_SE36_ENCFF425WDA.bam',
		'chip-seq/full/chip-seq_full_K562-CTCF_hg38_SE36_ENCFF172KOJ.bam',
		'cage/full/cage_full_K562_hg38_SE_ENCFF754FAU.bam',
		'pro-cap/full/pro-cap_full_K562_hg38_PE_ENCFF877SQU.bam',
		'chip-seq/full/chip-seq_full_K562-H3K4me3_hg38_SE76crop36.bam',
		'chip-seq/full/chip-seq_full_K562-MAX_hg38_SE101crop100.bam'],
	'quick': ['atac/large/atac_large_K562_hg38_PE.bam', 'dnase/large/dnase_large_K562_hg38_SE36.bam',
		'pro-cap/large/pro-cap_large_K562_hg38_PE.bam', 'cage/large/cage_large_K562_hg38_SE.bam']}


def sys_path():
	if BENCH not in sys.path:
		sys.path.insert(0, BENCH)


def capture(function, *args):
	"""The JSON line a compare_* function prints."""

	buffer = io.StringIO()
	with contextlib.redirect_stdout(buffer):
		function(*args)
	return json.loads(buffer.getvalue().strip().splitlines()[-1])


def digest(y):
	"""compare_formats' hash of what a read returned."""

	y = numpy.ascontiguousarray(numpy.nan_to_num(y.astype(numpy.float32)))
	return hashlib.sha256(y.tobytes()).hexdigest()[:12]


def only(values, what):
	values = set(values)
	if len(values) != 1:
		raise RuntimeError('{}: {} distinct outputs: {}'.format(what, len(values), sorted(values)))
	return values.pop()


###
# Windows

def draw_negatives(by, sizes, width, total, rng):
	"""`total` (chromosome, position) pairs drawn uniformly over the
	chromosomes of `by` (sorted positions per chromosome), at least `width`
	bases from every position in `by`."""

	rows = []
	names = sorted(by)
	weights = numpy.array([sizes[c] for c in names], dtype=float)
	picks = rng.choice(len(names), size=total, p=weights / weights.sum())
	for i, name in enumerate(names):
		near = numpy.asarray(by[name])
		count = int((picks == i).sum())
		while count:
			draw = rng.integers(width, sizes[name] - width, size=count * 2)
			at = numpy.searchsorted(near, draw)
			gap = numpy.minimum(abs(draw - near[numpy.minimum(at, len(near) - 1)]),
				abs(draw - near[numpy.maximum(at - 1, 0)]))
			draw = draw[gap >= width][:count]
			rows += [(name, int(m)) for m in draw]
			count -= len(draw)
	return rows


def signal_windows_bed(entries, width, n, out, seed=0):
	"""A BED of `n` one-base regions at randomly chosen bases with signal (a
	window of `width` bases centred on one lies inside its chromosome), then
	`n` drawn at random at least `width` bases from every base with signal.
	`entries` is compare_writers' cache of the bigWig's intervals. Returns
	how many."""

	import compare_writers
	chroms, ents, _ = compare_writers.load_entries(entries)
	rng = numpy.random.default_rng(seed)
	pool = {c: numpy.sort(s[(s >= width // 2) & (s + width // 2 <= chroms[c])])
		for c, (s, _, _) in ents.items()}
	pool = {c: s for c, s in pool.items() if len(s)}
	names = sorted(pool)
	counts = numpy.array([len(pool[c]) for c in names])
	flat = rng.choice(counts.sum(), size=n, replace=False)
	bounds = numpy.cumsum(counts)
	rows = []
	for j in numpy.sort(flat):
		i = numpy.searchsorted(bounds, j, side='right')
		rows.append((names[i], int(pool[names[i]][j - (bounds[i - 1] if i else 0)])))
	rng.shuffle(rows)
	rows += draw_negatives({c: s for c, (s, _, _) in ents.items() if len(s)}, chroms, width, n, rng)
	with open(out, 'w') as handle:
		for name, mid in rows:
			handle.write('{}\t{}\t{}\n'.format(name, mid, mid + 1))
	return len(rows)


def windows_bed(peaks, sizes, width, n, out, chrom=None, seed=0, negatives=False):
	"""A BED of one-base regions at the peaks' centres, so that a window of
	`width` bases centred on one lies inside its chromosome. `n` of them (all
	if fewer) in random order, or with `negatives` every peak's centre, in
	order, and then as many centres drawn uniformly over the peaks'
	chromosomes at least `width` bases from any peak. Returns how many."""

	mids = set()
	with gzip.open(peaks, 'rt') as handle:
		for line in handle:
			if not line.strip() or line.startswith(('#', 'track', 'browser')):
				continue
			name, start, end = line.split('\t')[:3]
			if name not in sizes or (chrom is not None and name != chrom):
				continue
			mid = int(start) + (int(end) - int(start)) // 2
			if mid - width // 2 >= 0 and mid - width // 2 + width <= sizes[name]:
				mids.add((name, mid))

	mids = sorted(mids)
	rng = numpy.random.default_rng(seed)
	if negatives:
		by = {}
		for name, mid in mids:
			by.setdefault(name, []).append(mid)
		rows = list(mids) + draw_negatives(by, sizes, width, len(mids), rng)
	else:
		rows = [mids[i] for i in rng.permutation(len(mids))[:n]]

	with open(out, 'w') as handle:
		for name, mid in rows:
			handle.write('{}\t{}\t{}\n'.format(name, mid, mid + 1))
	return len(rows)


###
# Workers: each is one process and returns a JSON-able dict. `jobs` is the
# threads a worker's own tool is given; common.worker's `threads` pins the
# process.

def read_one(ctx, lib, jobs, order, bigwig, bed, width):
	"""compare_readers' time_one: one library, threads, window order."""

	sys_path()
	import compare_readers
	args = argparse.Namespace(lib=lib, threads=jobs, order=order, bigwig=[bigwig],
		bed=[bed], width=width)
	return capture(compare_readers.time_one, args)


def write_one(ctx, lib, engine, jobs, zooms, bigwig, scratch):
	"""compare_writers' time_one: one library writing one bigWig's values."""

	sys_path()
	import compare_writers
	args = argparse.Namespace(lib=lib, engine=engine, threads=jobs, zooms=zooms,
		bigwig=bigwig, scratch=scratch)
	return capture(compare_writers.time_one, args)


def prepare_real(ctx, bam, out, sizes, jobs):
	"""One count track of a BAM over the chromosomes of `sizes`: the 5' end
	of every read, unstranded, by figwig bam2bw."""

	command = [sys.executable, '-m', 'figwig', 'bam2bw', bam, '-s', sizes, '-u', '-n',
		out, '-p', str(jobs), '-z', '0']
	result = subprocess.run(command, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError('bam2bw failed: ' + result.stderr[-2000:])
	return {'path': out + '.bw'}


def prepare_derived(ctx, workdir, sources, n, jobs):
	"""compare_formats.prepare_tracks: bigWigs 0 to n - 1 in workdir/tracks,
	the sources first and then rolled copies."""

	sys_path()
	import compare_formats
	args = argparse.Namespace(workdir=workdir, bigwig=sources, prepare_threads=jobs)
	start = time.perf_counter()
	with contextlib.redirect_stdout(sys.stderr):
		compare_formats.prepare_tracks(args, n)
	return {'seconds': time.perf_counter() - start}


def cache_entries(ctx, bigwigs, scratch, jobs):
	"""compare_writers' cache of every interval of each bigWig, on `jobs`
	processes."""

	sys_path()
	import compare_writers
	from concurrent.futures import ProcessPoolExecutor
	os.makedirs(scratch, exist_ok=True)
	with ProcessPoolExecutor(jobs) as pool:
		list(pool.map(compare_writers.cache_entries, [scratch] * len(bigwigs), bigwigs))
	return {'n': len(bigwigs)}


def bw2fw(ctx, workdir, n, jobs):
	"""figwig bw2fw of n bigWigs into one group, tile 16,384, libdeflate 6."""

	sys_path()
	import compare_formats
	path = compare_formats.figwig_path(workdir, n)
	bigwigs = [compare_formats.track_path(workdir, i) for i in range(n)]
	seconds = compare_formats.shell([sys.executable, '-m', 'figwig', 'bw2fw', '-o', path, '-f',
		'-p', str(jobs), '--groups', str(n), '-t', '16384', '-l', '6'] + bigwigs, dict(os.environ))
	return {'seconds': seconds, 'mb': common.size_mb(path),
		'bigwig_mb': sum(common.size_mb(b) for b in bigwigs)}


def fw2bw(ctx, workdir, n, jobs, zooms):
	"""figwig fw2bw: n bigWigs from the figwig file, with zoom levels."""

	sys_path()
	import compare_formats
	out = os.path.join(workdir, 'back_{}'.format(n))
	common.fresh(out)
	# fw2bw reads a group's tracks together, and a group of every track would be
	# a dense output over the 8 GiB limit, so the bigWigs come from groups of 8
	source = os.path.join(workdir, 'tracks_groups8_{}.fw'.format(n))
	if not os.path.exists(source):
		compare_formats.shell([sys.executable, '-m', 'figwig', 'bw2fw', '-o', source, '-f',
			'-p', str(jobs), '--groups', '8', '-t', '16384', '-l', '6'] + [
			compare_formats.track_path(workdir, i) for i in range(n)], dict(os.environ))
	seconds = compare_formats.shell([sys.executable, '-m', 'figwig', 'fw2bw',
		source, '-o', out, '-p', str(jobs), '-z', str(zooms), '-l', '6'], dict(os.environ))
	mb = common.size_mb(out)
	common.fresh(out)
	return {'seconds': seconds, 'mb': mb, 'files': n}


def figwig_read(ctx, workdir, n, jobs, width, windows, bed):
	"""compare_formats' time_one on the figwig file, into float32."""

	sys_path()
	import compare_formats
	args = argparse.Namespace(workdir=workdir, n=n, threads=jobs, mode='figwig_float32',
		batches=False, width=width, windows=windows, bed=[bed])
	return capture(compare_formats.time_one, args)


def pybigtools_read(ctx, workdir, n, jobs, width, windows, bed):
	"""The n bigWigs read by pybigtools into float32 (windows, n, width): one
	compare_readers.read_pybigtools per track, files opened inside the
	timing."""

	sys_path()
	import compare_formats
	import compare_readers
	names, starts = compare_formats.load_windows([bed], width, windows)
	paths = [compare_formats.track_path(workdir, i) for i in range(n)]

	def read(sel):
		out = numpy.empty((len(sel), n, width), dtype=numpy.float32)
		part, idx = names[sel], numpy.arange(len(sel))
		for t, path in enumerate(paths):
			compare_readers.read_pybigtools(path, part, starts[sel], idx, out[:, t, :],
				jobs, width)
		return out

	read(numpy.arange(min(64, len(starts))))
	start = time.perf_counter()
	y = read(numpy.arange(len(starts)))
	seconds = time.perf_counter() - start
	return {'seconds': seconds, 'digest': digest(y), 'n': n, 'windows': len(starts)}


def pybigwig_write_all(ctx, workdir, n, entries, zooms, jobs=1):
	"""compare_writers' write_pybigwig over the n tracks' cached intervals, on
	one thread, timed as a whole. The bigWigs are then read back (2,000
	windows, as compare_writers does) and matched to the source tracks'."""

	sys_path()
	import compare_formats
	import compare_writers
	tracks = []
	for i in range(n):
		source = compare_formats.track_path(workdir, i)
		chroms, ents, single = compare_writers.load_entries(
			compare_writers.entries_path(entries, source))
		tracks.append((source, chroms, ents, single))

	out = os.path.join(workdir, 'pybigwig_{}'.format(n))
	common.fresh(out)
	os.makedirs(out)
	start = time.perf_counter()
	for i, (source, chroms, ents, single) in enumerate(tracks):
		compare_writers.write_pybigwig(os.path.join(out, 'track_{:04d}.bw'.format(i)), chroms,
			ents, single, 'zlib', 1, zooms)
	seconds = time.perf_counter() - start

	mismatched = []
	for i, (source, chroms, ents, single) in enumerate(tracks):
		path = os.path.join(out, 'track_{:04d}.bw'.format(i))
		if compare_writers.windows_hash(path, chroms, ents) != compare_writers.windows_hash(
				source, chroms, ents):
			mismatched.append(i)
	mb = common.size_mb(out)
	common.fresh(out)
	if mismatched:
		raise RuntimeError('pyBigWig wrote different values for tracks {}'.format(mismatched))
	return {'seconds': seconds, 'mb': mb, 'files': n}


###
# Rows

def measure(ctx, target, configs, **kwargs):
	"""Each config (a dict of worker kwargs with `jobs`, which also pins the
	process) repeated; returns {name: (median seconds, every result)}."""

	return {name: common.repeated(ctx, target, threads=extra['jobs'], **dict(kwargs, **extra))
		for name, extra in configs.items()}


def fastest(found):
	return min(found, key=lambda name: found[name][0])


def thread_name(label, jobs, extra=''):
	return '{}, {} thread{}{}'.format(label, jobs, 's' if jobs > 1 else '', extra)


def bulk_1(ctx):
	T, quick = ctx['threads'], ctx['quick']
	scratch = os.path.join(ctx['scratch'], 'bulk-1')
	os.makedirs(scratch, exist_ok=True)
	sys_path()
	import pyBigWig
	import compare_writers
	width = 1000
	bed = os.path.join(scratch, 'windows.bed')

	if quick:
		bigwig_rel, peaks_rel = ATAC['quick']
		bigwig = common.testdata(bigwig_rel)
		with contextlib.closing(pyBigWig.open(bigwig)) as handle:
			sizes = dict(handle.chroms())
		windows = windows_bed(common.testdata(peaks_rel), sizes, width, 0, bed, negatives=True)
		compare_writers.cache_entries(scratch, bigwig)
		windows_from = 'peak centres of {} and as many random negatives'.format(peaks_rel)
		setting = 'peaks and negatives'
	else:
		bigwig_rel, peaks_rel = ATAC_BAM + ' (figwig bam2bw -u, 5\' ends)', None
		bigwig = os.path.join(scratch, 'atac-counts')
		common.worker(ctx, 'bulk:prepare_real', threads=PREP_THREADS, bam=common.testdata(ATAC_BAM),
			out=bigwig, sizes=common.testdata(HG38_SIZES), jobs=PREP_THREADS)
		bigwig += '.bw'
		compare_writers.cache_entries(scratch, bigwig)
		windows = signal_windows_bed(compare_writers.entries_path(scratch, bigwig), width,
			83875, bed)
		windows_from = ('random bases with signal and as many random negatives at least '
			'1,000 bp from any base with signal (no peak file for this sample)')
		setting = 'random signal bases and negatives'

	bigwig_mb = common.size_mb(bigwig)
	try:
		ours = measure(ctx, 'bulk:read_one', {'figwig': dict(lib='figwig', jobs=T,
			order='given')}, bigwig=bigwig, bed=bed, width=width)
		others = {thread_name('pybigtools', t, ', {} windows'.format(o)): dict(lib='pybigtools',
			jobs=t, order=o) for t in sorted({1, T}) for o in ('given', 'sorted')}
		theirs = measure(ctx, 'bulk:read_one', others, bigwig=bigwig, bed=bed, width=width)
		best = fastest(theirs)
		sha = only([r['sha256'] for d in (ours, theirs) for _, v in d.values() for r in v],
			'bulk-1 reads')
		read = common.item(ours['figwig'][0], theirs[best][0], baseline_name=best)

		writes = {'figwig': dict(lib='figwig', engine='libdeflate', jobs=T),
			'pyBigWig': dict(lib='pybigwig', engine='zlib', jobs=1)}
		wrote = measure(ctx, 'bulk:write_one', writes, zooms=0, bigwig=bigwig, scratch=scratch)
		whash = only([r['hash'] for _, v in wrote.values() for r in v], 'bulk-1 writes')
		write = common.item(wrote['figwig'][0], wrote['pyBigWig'][0], baseline_name='pyBigWig, 1 thread')
		size = common.item(wrote['figwig'][1][-1]['bytes'] / 1e6,
			wrote['pyBigWig'][1][-1]['bytes'] / 1e6, unit='MB', baseline_name='pyBigWig')
		last = wrote['figwig'][1][-1]
	finally:
		shutil.rmtree(scratch, ignore_errors=True)

	return {'setting': '1 ATAC-seq track, {:,} windows of 1,000 bp on {}'.format(windows, setting),
		'baseline': 'one bigWig; read by pybigtools, written by pyBigWig',
		'read': [read], 'write': [write], 'size': [size],
		'inputs': {'bigwig': bigwig_rel, 'peaks': peaks_rel, 'windows_from': windows_from,
			'bigwig_mb': bigwig_mb, 'windows': windows, 'width': width,
			'written_items': last['items'], 'written_as_single_bases': last['single'],
			'zoom_levels_written': 0},
		'raw': {'read': {k: {'median': v[0], 'results': v[1]} for d in (ours, theirs)
			for k, v in d.items()}, 'read_sha256': sha,
			'write': {k: {'median': v[0], 'results': v[1]} for k, v in wrote.items()},
			'write_hash': whash},
		'notes': []}


def tracks_row(ctx, key, n, windows):
	"""bulk-48 and bulk-1024: n tracks over chr1, `windows` windows."""

	T, quick = ctx['threads'], ctx['quick']
	tier = 'quick' if quick else 'full'
	if quick:
		n, windows = {'bulk-48': (8, 512), 'bulk-1024': (24, 128)}[key]
	width = 2048
	bams = COUNT_BAMS[tier]
	workdir = os.path.join(ctx['scratch'], 'counts')
	os.makedirs(workdir, exist_ok=True)

	sizes = os.path.join(workdir, 'chr1.sizes')
	with open(sizes, 'w') as handle:
		handle.write('chr1\t{}\n'.format(CHR1))
	reals = []
	for bam in bams:
		out = os.path.join(workdir, 'real_' + os.path.basename(bam)[:-4])
		if not os.path.exists(out + '.bw'):
			common.worker(ctx, 'bulk:prepare_real', threads=PREP_THREADS,
				bam=common.testdata(bam), out=out, sizes=sizes, jobs=PREP_THREADS)
		reals.append(out + '.bw')
	prepared = common.worker(ctx, 'bulk:prepare_derived', threads=PREP_THREADS,
		workdir=workdir, sources=reals, n=n, jobs=PREP_THREADS)

	sys_path()
	import pyBigWig
	with contextlib.closing(pyBigWig.open(reals[0])) as handle:
		chroms = dict(handle.chroms())
	peaks_rel = ATAC['quick' if quick else 'full'][1]
	bed = os.path.join(workdir, 'windows-{}.bed'.format(key))
	found = windows_bed(common.testdata(peaks_rel), chroms, width, windows, bed, chrom='chr1')

	fw = os.path.join(workdir, 'tracks_{}.fw'.format(n))
	entries = os.path.join(workdir, 'entries')
	try:
		fw_write = measure(ctx, 'bulk:bw2fw', {'figwig': dict(jobs=T)}, workdir=workdir, n=n)['figwig']
		if key == 'bulk-1024':
			bw_write = measure(ctx, 'bulk:fw2bw', {'figwig': dict(jobs=T)}, workdir=workdir, n=n,
				zooms=ZOOMS)['figwig']
			write_name = 'bigWigs written by figwig from a file of groups of 8; pyBigWig not run'
			size_name = 'the bigWigs written by figwig; pyBigWig not run'
		else:
			tracks = [os.path.join(workdir, 'tracks', 'track_{:03d}.bw'.format(i)) for i in range(n)]
			common.worker(ctx, 'bulk:cache_entries', threads=PREP_THREADS, bigwigs=tracks,
				scratch=entries, jobs=PREP_THREADS)
			bw_write = measure(ctx, 'bulk:pybigwig_write_all', {'pyBigWig': dict(jobs=1)},
				workdir=workdir, n=n, entries=entries, zooms=ZOOMS)['pyBigWig']
			write_name = 'pyBigWig, 1 thread'
			size_name = 'pyBigWig'
		common.fresh(entries)

		reading = {'figwig file': dict(jobs=T)}
		ours = measure(ctx, 'bulk:figwig_read', reading, workdir=workdir, n=n, width=width,
			windows=windows, bed=bed)
		others = {thread_name('pybigtools', t): dict(jobs=t) for t in sorted({1, T})}
		theirs = measure(ctx, 'bulk:pybigtools_read', others, workdir=workdir, n=n, width=width,
			windows=windows, bed=bed)
		best = fastest(theirs)
		sha = only([r['digest'] for d in (ours, theirs) for _, v in d.values() for r in v],
			key + ' reads')
	finally:
		common.fresh(fw)

	read = common.item(ours['figwig file'][0], theirs[best][0], baseline_name=best)
	write = common.item(fw_write[0], bw_write[0], baseline_name=write_name)
	size = common.item(fw_write[1][-1]['mb'], bw_write[1][-1]['mb'], unit='MB',
		baseline_name=size_name)
	notes = []
	if key == 'bulk-1024':
		notes.append('The 1,024-track baseline write and size are the 1,024 bigWigs figwig wrote '
			'with {} zoom levels ({:.0f} s, {:.1f} GB); pyBigWig was not run at that size.'.format(
			ZOOMS, bw_write[0], bw_write[1][-1]['mb'] / 1000))
	return {'setting': '{:,} count tracks over chr1, {:,} windows of 2,048 bp'.format(n, found),
		'baseline': 'one bigWig per track; read by pybigtools, written by {}'.format(
			'figwig' if key == 'bulk-1024' else 'pyBigWig'),
		'read': [read], 'write': [write], 'size': [size],
		'inputs': {'real_tracks': [b for b in bams], 'tracks': n, 'real_count': len(bams),
			'derived': 'rolled copies of the real tracks (compare_formats.prepare_tracks)',
			'peaks': peaks_rel, 'windows': found, 'width': width,
			'figwig_file': 'one group of all tracks, tile 16,384, libdeflate 6, bw2fw',
			'baseline_zoom_levels': ZOOMS, 'prepare_derived_seconds': prepared['seconds']},
		'raw': {'read': {k: {'median': v[0], 'results': v[1]} for d in (ours, theirs)
			for k, v in d.items()}, 'read_digest': sha,
			'write_figwig_file': {'median': fw_write[0], 'results': fw_write[1]},
			'write_baseline': {'median': bw_write[0], 'results': bw_write[1]}},
		'notes': notes}


def bulk_48(ctx):
	return tracks_row(ctx, 'bulk-48', 48, 8192)


def bulk_1024(ctx):
	return tracks_row(ctx, 'bulk-1024', 1024, 1024)


ROWS = {'bulk-1': bulk_1, 'bulk-48': bulk_48, 'bulk-1024': bulk_1024}
