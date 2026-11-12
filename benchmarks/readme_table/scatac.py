# scatac.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The scATAC-seq row of the README's table: the 10x PBMC fragments file as one
track per called cell.

Write: `figwig bam2fw --barcode-column 4 --cells cells.tsv -f` on `--threads`
threads against SnapATAC2's `pp.import_fragments` with the same cells as its
whitelist on 8 threads. Both outputs hold the same fragment ends of the called
cells, which `check` verifies from the outputs themselves: every cell's figwig
total equals twice its SnapATAC2 fragment count.

Read: 10 pseudobulks (the cells in 10 random labels) summed over 8,192 windows
of 2,048 bp centred on K562 ATAC-seq peaks, from the figwig file `bam2fw`
wrote, against a scipy CSR matrix of positions x cells per chromosome, built
from the fragments and held in memory before the timer starts. The baseline is
given its fastest of 1 and `--threads` threads, and both return the same
(windows, labels, width) float64 array, which is checked by a hash.

Size: the figwig file against SnapATAC2's output, which stores fragments
rather than per-base counts. The CSR's size in memory goes in a note.

The test uses the full fragments file; `--quick` uses the `large` tier.
"""

import os
import sys
import json
import time
import hashlib
import statistics
import subprocess
import concurrent.futures

import numpy

import common
import render

BENCHMARKS = os.path.join(common.REPO, 'benchmarks')
sys.path.insert(0, BENCHMARKS)
FULL = 'scatac-seq/full/scatac-seq_full_PBMC_hg38_PE_fragments.tsv.gz'
LARGE = 'scatac-seq/large/scatac-seq_large_PBMC_hg38_PE_fragments.tsv.gz'
CELLS = 'scatac-seq/aux/scatac-seq_aux_PBMC_hg38_PE_singlecell.csv'
PEAKS = 'peaks/large/atac-seq_large_K562_hg38_PE_replicated_ENCFF827JWX.narrowPeak.gz'
SIZES = 'reference/hg38/hg38.ucsc.chrom.sizes'
N_WINDOWS = 8192
WIDTH = 2048
N_LABELS = 10
SNAP_THREADS = 8


def paths(ctx):
	"""The files of the row inside its scratch directory."""

	root = os.path.join(ctx['scratch'], 'scatac')
	return {'root': root, 'cells': os.path.join(root, 'cells.tsv'),
		'fw': os.path.join(root, 'cells.fw'), 'snap': os.path.join(root, 'snap.h5ad'),
		'coo': os.path.join(root, 'coo'), 'tmp': os.path.join(root, 'tmp'),
		'windows': os.path.join(root, 'windows.npz'), 'prep': os.path.join(root, 'prep.json')}


def chrom_sizes(ctx):
	with open(common.testdata(SIZES)) as handle:
		return {line.split()[0]: int(line.split()[1]) for line in handle if line.strip()}


def labels_for(n_cells):
	"""The label of every cell, in the order of the cells table."""

	return numpy.random.default_rng(0).integers(0, N_LABELS, n_cells)


def prep(ctx, fragments):
	"""Untimed: the cells table, the windows, and every fragment end of the
	called cells as sorted (position, cell, count) arrays per chromosome, the
	input of the scipy matrix."""

	import pandas
	from compare_formats import load_windows

	p = paths(ctx)
	for key in ('coo', 'tmp'):
		os.makedirs(p[key], exist_ok=True)
	table = pandas.read_csv(common.testdata(CELLS))
	barcodes = table.loc[table['is__cell_barcode'] == 1, 'barcode']
	barcodes.to_frame().to_csv(p['cells'], sep='\t', index=False)
	index = pandas.Index(barcodes)
	n_cells = len(index)

	sizes = chrom_sizes(ctx)
	keys = {}
	reader = pandas.read_csv(fragments, sep='\t', comment='#', header=None, usecols=[0, 1, 2, 3],
		names=['chrom', 'start', 'end', 'barcode'], dtype={'chrom': 'category', 'start': 'int64',
		'end': 'int64', 'barcode': 'str'}, chunksize=10_000_000)
	for chunk in reader:
		cell = index.get_indexer(chunk['barcode'])
		keep = (cell >= 0) & chunk['chrom'].isin(sizes).to_numpy()
		chunk, cell = chunk[keep], cell[keep].astype(numpy.int64)
		for chrom, rows in chunk.groupby('chrom', observed=True).indices.items():
			sub = chunk.iloc[rows]
			c = cell[rows]
			# both ends of a fragment: the first base and the last base
			keys.setdefault(chrom, []).append(numpy.concatenate([
				(sub['start'].to_numpy() << 14) | c, ((sub['end'].to_numpy() - 1) << 14) | c]))
	ends = 0
	for chrom, parts in keys.items():
		k = numpy.concatenate(parts)
		k.sort()
		unique, count = numpy.unique(k, return_counts=True)
		if count.max() > 255:
			raise ValueError('more than 255 fragment ends of one cell on one base')
		ends += int(count.sum())
		numpy.savez(os.path.join(p['coo'], chrom + '.npz'), pos=(unique >> 14).astype(numpy.int32),
			cell=(unique & 16383).astype(numpy.int16), count=count.astype(numpy.uint8))

	# every peak in random order; the first windows that end inside their chromosome (past the end,
	# figwig's float sums are NaN)
	names, starts = load_windows([common.testdata(PEAKS)], WIDTH, 10 ** 9)
	inside = numpy.array([s + WIDTH <= sizes[c] for c, s in zip(names, starts)])
	names, starts = names[inside][:N_WINDOWS], starts[inside][:N_WINDOWS]
	numpy.savez(p['windows'], names=names, starts=starts)
	info = {'n_cells': n_cells, 'fragment_ends': ends}
	with open(p['prep'], 'w') as handle:
		json.dump(info, handle)
	return info


def figwig_write(ctx, fragments):
	"""Timed: the bam2fw command, interpreter start included."""

	p = paths(ctx)
	common.fresh(p['fw'])
	command = [ctx['python'], '-m', 'figwig', 'bam2fw', '-s', common.testdata(SIZES), '-n',
		p['fw'][:-3], '-f', '-p', str(ctx['threads']), '--barcode-column', '4', '--cells',
		p['cells'], '--force', fragments]
	start = time.perf_counter()
	result = subprocess.run(command, capture_output=True, text=True)
	seconds = time.perf_counter() - start
	if result.returncode != 0:
		raise RuntimeError(result.stderr[-2000:])
	return {'seconds': seconds, 'mb': common.size_mb(p['fw']), 'command': ' '.join(command[2:])}


def snap_write(ctx, fragments):
	"""Timed: SnapATAC2 import_fragments with the called cells as whitelist,
	writing its h5ad."""

	import snapatac2 as snap

	p = paths(ctx)
	common.fresh(p['snap'])
	common.fresh(p['tmp'])
	os.makedirs(p['tmp'])
	start = time.perf_counter()
	data = snap.pp.import_fragments(fragments, chrom_sizes(ctx), file=p['snap'],
		whitelist=p['cells'], min_num_fragments=0, sorted_by_barcode=False, tempdir=p['tmp'],
		n_jobs=SNAP_THREADS)
	seconds = time.perf_counter() - start
	data.close()
	common.fresh(p['tmp'])
	return {'seconds': seconds, 'mb': common.size_mb(p['snap']), 'version': snap.__version__}


def check(ctx):
	"""Untimed: every cell's figwig total against twice its SnapATAC2 fragment
	count."""

	import snapatac2 as snap
	import figwig

	p = paths(ctx)
	data = snap.read(p['snap'])
	counts = dict(zip(data.obs_names, data.obs['n_fragment'].to_numpy().tolist()))
	data.close()
	reader = figwig.FigWigReader(p['fw'])
	total = numpy.zeros(len(reader.tracks))
	for chrom, length in reader.chrom_sizes.items():
		total += reader.sums(chrom, [0], [length], n_jobs=ctx['threads'])[0]
	names = [t['name'] for t in reader.tracks]
	if set(names) != set(counts):
		raise AssertionError('the two outputs hold different cells')
	expected = numpy.array([2 * counts[n] for n in names], dtype=numpy.float64)
	return {'cells': len(names), 'figwig_ends': int(total.sum()), 'snapatac2_ends': int(expected.sum()),
		'cells_differing': int((total != expected).sum())}


def digest(y):
	return hashlib.sha256(numpy.ascontiguousarray(y, dtype=numpy.float64).tobytes()).hexdigest()[:16]


def figwig_read(ctx, windows_threads):
	"""Timed: sum 8,192 windows per label from the open file, after a warm-up
	read of 1,000 windows. Opening the file (`open_seconds`) is timed apart,
	as the matrix's loading is left out of the baseline's time."""

	import figwig

	p = paths(ctx)
	z = numpy.load(p['windows'])
	names, starts = z['names'], z['starts']
	labels = labels_for(json.load(open(p['prep']))['n_cells'])
	start = time.perf_counter()
	reader = figwig.FigWigReader(p['fw'])
	open_seconds = time.perf_counter() - start
	reader.read(names[:1000], starts[:1000], WIDTH, labels=labels, n_jobs=windows_threads)
	start = time.perf_counter()
	y = reader.read(names, starts, WIDTH, labels=labels, n_jobs=windows_threads)
	seconds = time.perf_counter() - start
	return {'seconds': seconds, 'open_seconds': open_seconds, 'digest': digest(y), 'sum': float(y.sum()),
		'shape': list(y.shape)}


def build_matrices(ctx):
	"""Untimed: the CSR of positions x cells for every chromosome."""

	import scipy.sparse

	p = paths(ctx)
	matrices = {}
	for chrom, length in chrom_sizes(ctx).items():
		path = os.path.join(p['coo'], chrom + '.npz')
		if not os.path.exists(path):
			continue
		z = numpy.load(path)
		indptr = numpy.zeros(length + 1, dtype=numpy.int32)
		numpy.cumsum(numpy.bincount(z['pos'], minlength=length), out=indptr[1:])
		n_cells = json.load(open(p['prep']))['n_cells']
		matrices[chrom] = scipy.sparse.csr_matrix((z['count'], z['cell'].astype(numpy.int32), indptr),
			shape=(length, n_cells))
	return matrices


def window_sums(matrix, labels, out, row, start):
	"""One window's sums per label into out[row], through scipy's row slice."""

	a, b = max(start, 0), min(start + WIDTH, matrix.shape[0])
	if a >= b:
		return
	sub = matrix[a:b]
	rows = numpy.repeat(numpy.arange(b - a), numpy.diff(sub.indptr))
	sums = numpy.bincount(labels[sub.indices] * (b - a) + rows, weights=sub.data,
		minlength=N_LABELS * (b - a))
	out[row, :, a - start:b - start] = sums.reshape(N_LABELS, b - a)


def scipy_read(ctx, windows_threads):
	"""Timed: the same sums from the matrices held in memory, windows split
	over a thread pool when `windows_threads` is above 1."""

	p = paths(ctx)
	matrices = build_matrices(ctx)
	size = sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in matrices.values()) / 1e6
	z = numpy.load(p['windows'])
	names, starts = z['names'], z['starts']
	n_cells = json.load(open(p['prep']))['n_cells']
	labels = labels_for(n_cells)

	def read(selection):
		out = numpy.zeros((len(selection), N_LABELS, WIDTH), dtype=numpy.float64)
		pieces = numpy.array_split(numpy.arange(len(selection)), windows_threads)
		with concurrent.futures.ThreadPoolExecutor(windows_threads) as pool:
			list(pool.map(lambda rows: [window_sums(matrices[names[selection[r]]], labels, out, int(r),
				int(starts[selection[r]])) for r in rows], pieces))
		return out

	read(numpy.arange(1000))
	start = time.perf_counter()
	y = read(numpy.arange(len(names)))
	seconds = time.perf_counter() - start
	return {'seconds': seconds, 'digest': digest(y), 'sum': float(y.sum()), 'shape': list(y.shape),
		'matrix_mb': size}


def scatac(ctx):
	quick = ctx['quick']
	fragments = common.testdata(LARGE if quick else FULL)
	repeats = ctx['repeats']
	threads = ctx['threads']
	p = paths(ctx)
	os.makedirs(p['root'], exist_ok=True)
	raw = {}

	info = common.worker(ctx, 'scatac:prep', threads=1, fragments=fragments)
	n_cells, ends = info['n_cells'], info['fragment_ends']

	fw_seconds, fw_runs = common.repeated(ctx, 'scatac:figwig_write', threads=threads, fragments=fragments)
	sn_seconds, sn_runs = common.repeated(ctx, 'scatac:snap_write', threads=SNAP_THREADS, fragments=fragments)
	raw['figwig_write'] = fw_runs
	raw['snapatac2_write'] = sn_runs
	same = common.worker(ctx, 'scatac:check', threads=threads)
	raw['same_fragment_ends'] = same
	if same['cells_differing'] or same['figwig_ends'] != same['snapatac2_ends']:
		raise AssertionError('figwig and SnapATAC2 do not hold the same fragment ends: {}'.format(same))
	fw_mb, sn_mb = fw_runs[-1]['mb'], sn_runs[-1]['mb']
	os.remove(p['snap'])

	fr_seconds, fr_runs = common.repeated(ctx, 'scatac:figwig_read', threads=threads,
		windows_threads=threads)
	raw['figwig_read'] = fr_runs
	scipy_seconds = {}
	scipy_runs = {}
	for t in sorted({1, threads}):
		scipy_seconds[t], scipy_runs[t] = common.repeated(ctx, 'scatac:scipy_read', threads=t,
			windows_threads=t)
		raw['scipy_read_{}_threads'.format(t)] = scipy_runs[t]
	best = min(scipy_seconds, key=scipy_seconds.get)
	digests = {r['digest'] for r in fr_runs} | {r['digest'] for rs in scipy_runs.values() for r in rs}
	if len(digests) != 1:
		raise AssertionError('figwig and the scipy matrix gave different sums: {}'.format(digests))
	matrix_mb = scipy_runs[best][-1]['matrix_mb']

	notes = [
		'scATAC-seq: figwig and SnapATAC2 hold the same {:,} fragment ends of the {:,} called cells '
		'(every cell\'s figwig total is twice its SnapATAC2 fragment count), and figwig and the scipy '
		'matrix gave identical sums over the windows.'.format(same['figwig_ends'], n_cells),
		'The scipy matrix is {} in memory (the read times leave out building it and opening the figwig file, '
		'which takes {}), where the figwig file is {}.'.format(render.megabytes(matrix_mb),
		render.seconds(statistics.median(r['open_seconds'] for r in fr_runs)), render.megabytes(fw_mb)),
		'SnapATAC2 stores each cell\'s fragments, so counts over windows or peaks are still to be '
		'computed from its output.']
	raw['scipy_seconds_by_threads'] = scipy_seconds
	return {
		'setting': '10x PBMC scATAC-seq, {:,} cells, {:.0f} million cut sites; reads: 10 pseudobulks '
			'over {:,} windows of {:,} bp on K562 ATAC peaks{}'.format(n_cells, ends / 1e6, N_WINDOWS,
			WIDTH, '; quick tier' if quick else ''),
		'baseline': 'SnapATAC2 (write, size) and a scipy CSR matrix in memory (read)',
		'read': [common.item(fr_seconds, scipy_seconds[best], 's',
			baseline_name='scipy CSR, {} thread{}'.format(best, '' if best == 1 else 's'))],
		'write': [common.item(fw_seconds, sn_seconds, 's',
			baseline_name='SnapATAC2, {} threads'.format(SNAP_THREADS))],
		'size': [common.item(fw_mb, sn_mb, 'MB',
			baseline_name='SnapATAC2, which stores fragments rather than per-base counts')],
		'inputs': {'fragments': os.path.relpath(fragments, common.TESTDATA), 'cells': CELLS,
			'peaks': PEAKS, 'genome': SIZES, 'windows': N_WINDOWS, 'width': WIDTH, 'labels': N_LABELS,
			'cells_called': n_cells, 'fragment_ends': ends, 'figwig_threads': threads,
			'snapatac2_threads': SNAP_THREADS, 'repeats': repeats},
		'raw': raw, 'notes': notes}


ROWS = {'scatac': scatac}
