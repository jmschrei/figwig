# hic.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The Hi-C row of the README table: a figwig file against cooler.

Input: the raw counts of K562 in situ Hi-C (4DN 4DNFI18UHVRO, hg38) at 5 kb,
907M contacts, taken from the test collection's `.mcool`. They are rebuilt
into a pixel table on disk (`hic_common.py`), the upper triangle sorted by
(bin1, bin2), with the pixels the 4DN file repeats summed (figwig's
`write_sparse` refuses repeats). Both writers start from that table.

The figwig file has one track per bin, so track i is row i of the matrix, in
bin coordinates (each chromosome is as long as its number of bins) and the
upper triangle only, as cooler stores it. Reads are a 2 Mb window on the
diagonal as a dense matrix (median over random windows) with one thread on
both sides. The write includes the work of turning the pixel table into
`write_sparse` calls (a spill by column chromosome, a sort by position), which
is how the figwig file is built from a table. A second figwig file with both
triangles and a 'diagonal' track is built for the note: each track is a whole
row, so it reads a bin's contacts across the genome (virtual 4C).

Every read is hashed against cooler's on the same queries and every file's
counts are summed per row against the pixel table, and a mismatch raises.
"""

import os
import sys
import json
import time
import shutil

import numpy
import pandas

import common
import render
from hic_common import load_meta
from hic_common import chunks
from hic_common import bin_starts
from hic_common import tile_calls
from hic_common import digest
from hic_common import mirror
from hic_common import time_queries
from hic_common import same_hashes

SOURCES = {
	'full': 'hi-c/full/hi-c_full_K562_hg38_PE_4DNFI18UHVRO.mcool',
	'quick': 'hi-c/large/hi-c_large_K562_hg38_PE.mcool',
}
RESOLUTION = 5000
SPAN = 2_000_000
# Layouts from the layout sweep: bin coordinates, tiles of 256 bins.
UPPER = {'triangle': 'upper', 'tile': 256, 'group_size': 1024}
BOTH = {'triangle': 'both', 'tile': 256, 'group_size': 4096}
READ_THREADS = 1


def extract(ctx, mcool, resolution, out):
	"""One resolution of an .mcool as a pixel table: the raw counts sorted by
	(bin1, bin2), repeated pixels summed. Untimed."""

	import h5py

	os.makedirs(out, exist_ok=True)
	with h5py.File(mcool, 'r') as handle:
		group = handle['resolutions/{}'.format(resolution)]
		names = [name.decode() if isinstance(name, bytes) else str(name)
			for name in group['chroms/name'][:]]
		lengths = group['chroms/length'][:].astype(int).tolist()
		chrom_offsets = group['indexes/chrom_offset'][:].astype(int).tolist()
		bin1 = group['pixels/bin1_id'][:].astype(numpy.int64)
		bin2 = group['pixels/bin2_id'][:].astype(numpy.int64)
		count = group['pixels/count'][:]
		attrs = {key: int(group.attrs[key]) for key in ('nnz', 'sum', 'nbins')}

	nbins = attrs['nbins']
	assert chrom_offsets[-1] == nbins
	assert len(bin1) == attrs['nnz'] and int(count.sum()) == attrs['sum']
	assert (bin1 <= bin2).all() and (count > 0).all()

	# The 4DN .mcool repeats some pixels (a row split in two); they are summed.
	key = bin1 * nbins + bin2
	del bin1, bin2
	order = numpy.argsort(key, kind='stable')
	key = key[order]
	starts = numpy.flatnonzero(numpy.concatenate([[True], key[1:] != key[:-1]]))
	repeats = int(len(key) - len(starts))
	count = numpy.add.reduceat(count[order].astype(numpy.int64), starts)
	del order
	key = key[starts]
	bin1, bin2 = key // nbins, key % nbins
	assert int(count.sum()) == attrs['sum']
	on = bin1 == bin2
	for name, column in (('bin1', bin1), ('bin2', bin2), ('count', count)):
		column.astype(numpy.int32).tofile(os.path.join(out, name + '.i32'))

	meta = {'source': mcool, 'resolution': resolution, 'chroms': names,
		'lengths': lengths, 'chrom_offsets': chrom_offsets, 'nbins': nbins,
		'nnz': len(key), 'sum': attrs['sum'], 'nnz_source': attrs['nnz'],
		'repeated_pixels_summed': repeats, 'diagonal_pixels': int(on.sum()),
		'diagonal_sum': int(count[on].sum())}
	with open(os.path.join(out, 'meta.json'), 'w') as handle:
		json.dump(meta, handle)
	return {key: meta[key] for key in ('nnz', 'sum', 'nnz_source',
		'repeated_pixels_summed', 'diagonal_pixels', 'diagonal_sum')}


def write_cooler(ctx, table, out):
	import cooler

	meta = load_meta(table)
	common.fresh(out)
	starts, chrom = bin_starts(meta)
	names = numpy.array(meta['chroms'])
	lengths = numpy.array(meta['lengths'])
	ends = numpy.minimum(starts + meta['resolution'], lengths[chrom])
	bins = pandas.DataFrame({'chrom': names[chrom], 'start': starts, 'end': ends})

	def pixels():
		for bin1, bin2, count in chunks(table, 10_000_000):
			yield pandas.DataFrame({'bin1_id': bin1.astype(numpy.int64),
				'bin2_id': bin2.astype(numpy.int64), 'count': count})

	start = time.perf_counter()
	cooler.create_cooler(out, bins, pixels(), ordered=True, symmetric_upper=True,
		dtypes={'count': 'int32'})
	seconds = time.perf_counter() - start
	return {'seconds': seconds, 'mb': common.size_mb(out), 'rss_mb': common.peak_rss_mb()}


def spill(meta, table, spill_dir, triangle):
	"""Split the entries (row, column, value) by the column's chromosome into
	flat int32 files, each in the table's order. With both triangles each
	contact is in both rows and the diagonal once more in the last track."""

	offsets = meta['chrom_offsets']
	n_chroms = len(meta['chroms'])
	handles = [[open(os.path.join(spill_dir, '{}.{}'.format(c, name)), 'wb')
		for name in ('row', 'col', 'val')] for c in range(n_chroms)]
	for bin1, bin2, count in chunks(table, 10_000_000):
		if triangle == 'both':
			off = bin1 != bin2
			on = ~off
			rows = numpy.concatenate([bin1, bin2[off], numpy.full(on.sum(),
				meta['nbins'], dtype=numpy.int32)])
			cols = numpy.concatenate([bin2, bin1[off], bin1[on]])
			vals = numpy.concatenate([count, count[off], count[on]])
		else:
			rows, cols, vals = bin1, bin2, count
		chrom = numpy.searchsorted(offsets, cols, side='right') - 1
		order = numpy.argsort(chrom, kind='stable')
		bounds = numpy.searchsorted(chrom[order], numpy.arange(n_chroms + 1))
		for c in range(n_chroms):
			a, b = bounds[c], bounds[c + 1]
			if b > a:
				take = order[a:b]
				for handle, array in zip(handles[c], (rows, cols, vals)):
					array[take].tofile(handle)
	for trio in handles:
		for handle in trio:
			handle.close()


def write_figwig(ctx, table, out, spill_dir, triangle, tile, group_size):
	"""One track per bin in bin coordinates. figwig takes a group's entries
	in position order one chromosome at a time, so the entries are spilled to
	one file per column chromosome, each is sorted by position and written
	in calls that end on a tile."""

	from figwig import FigWigWriter

	meta = load_meta(table)
	common.fresh(out)
	offsets = meta['chrom_offsets']
	names = meta['chroms']
	starts, chrom = bin_starts(meta)
	tracks = ['{}:{}'.format(names[c], s) for c, s in zip(chrom.tolist(), starts.tolist())]
	groups = [tracks[i:i + group_size] for i in range(0, len(tracks), group_size)]
	if triangle == 'both':
		# A contact between two fine bins of one coarse bin is in both rows,
		# so the diagonal as its own track lets a reader count it once.
		tracks.append('diagonal')
		groups.append(['diagonal'])
	chrom_sizes = dict(zip(names, numpy.diff(offsets).tolist()))

	start = time.perf_counter()
	shutil.rmtree(spill_dir, ignore_errors=True)
	os.makedirs(spill_dir)
	spill(meta, table, spill_dir, triangle)
	with FigWigWriter(out, chrom_sizes, tracks, genome='hg38', tile=tile, groups=groups,
			n_jobs=ctx['threads'], attributes={'resolution': meta['resolution'],
			'coords': 'bin', 'triangle': triangle}) as writer:
		for c, name in enumerate(names):
			rows, cols, vals = [numpy.fromfile(os.path.join(spill_dir, '{}.{}'.format(c, k)),
				dtype=numpy.int32) for k in ('row', 'col', 'val')]
			if len(rows) == 0:
				continue
			positions = cols.astype(numpy.int64) - offsets[c]
			order = numpy.argsort(positions, kind='stable')
			rows, positions, vals = rows[order], positions[order], vals[order]
			del cols, order
			cuts = tile_calls(positions, tile, 10_000_000)
			for a, b in zip(cuts[:-1], cuts[1:]):
				writer.write_sparse(name, positions[a:b], rows[a:b], vals[a:b])
	seconds = time.perf_counter() - start
	shutil.rmtree(spill_dir, ignore_errors=True)
	return {'seconds': seconds, 'mb': common.size_mb(out), 'rss_mb': common.peak_rss_mb()}


def diag_windows(meta, span, n, seed):
	"""(chromosome index, start bp) of n windows of span bp inside
	chromosomes, starting on a bin, chromosomes drawn by length."""

	resolution = meta['resolution']
	lengths = numpy.array(meta['lengths'])
	eligible = numpy.flatnonzero(lengths >= span + resolution)
	p = lengths[eligible] / lengths[eligible].sum()
	random_state = numpy.random.default_rng(seed)
	chroms = random_state.choice(eligible, size=n, p=p)
	starts = [random_state.integers(0, (lengths[c] - span) // resolution + 1) * resolution
		for c in chroms]
	return list(zip(chroms.tolist(), [int(s) for s in starts]))


class Cooler:
	def __init__(self, path, meta):
		import cooler
		self.c = cooler.Cooler(path)
		self.meta = meta

	def diag(self, chrom, start, span):
		return self.c.matrix(balance=False).fetch((self.meta['chroms'][chrom], start, start + span))

	def row(self, i):
		return self.c.matrix(balance=False, sparse=True)[i:i + 1, :].toarray()[0]


class FigWig:
	"""A file in bin coordinates; with the upper triangle the window is
	mirrored, with both triangles a row is one track."""

	def __init__(self, path, meta):
		from figwig import FigWigReader
		self.fw = FigWigReader(path)
		self.meta = meta
		self.upper = self.fw.attributes['triangle'] == 'upper'
		self.names = numpy.array(meta['chroms'])
		res = meta['resolution']

		# Every chromosome tiled by windows of 10 Mb, for a row.
		width = 10_000_000 // res
		self.row_chroms, self.row_starts, self.row_bins = [], [], []
		for c, length in enumerate(meta['lengths']):
			n_bins = -(-length // res)
			for s in range(0, n_bins, width):
				self.row_chroms.append(meta['chroms'][c])
				self.row_starts.append(s)
			self.row_bins.append(n_bins)
		self.row_chroms = numpy.array(self.row_chroms)
		self.row_starts = numpy.array(self.row_starts)
		self.row_width = width

	def diag(self, chrom, start, span):
		res = self.meta['resolution']
		i0 = self.meta['chrom_offsets'][chrom] + start // res
		k = span // res
		y = self.fw.read(self.names[chrom:chrom + 1], [start // res], k,
			tracks=numpy.arange(i0, i0 + k), n_jobs=READ_THREADS)[0]
		return mirror(y) if self.upper else y

	def row(self, i):
		y = self.fw.read(self.row_chroms, self.row_starts, self.row_width, tracks=[i],
			n_jobs=READ_THREADS)[:, 0, :]
		out, w = [], 0
		for c, n_bins in enumerate(self.row_bins):
			n_windows = int(numpy.sum(self.row_chroms == self.meta['chroms'][c]))
			out.append(y[w:w + n_windows].ravel()[:n_bins])
			w += n_windows
		return numpy.concatenate(out)


def read_window(ctx, kind, path, table, n, budget, verify):
	"""Median seconds per 2 Mb diagonal window (a dense matrix) of one file."""

	meta = load_meta(table)
	reader = Cooler(path, meta) if kind == 'cooler' else FigWig(path, meta)

	def run(item):
		return reader.diag(item[0], item[1], SPAN)

	out = time_queries(run, diag_windows(meta, SPAN, n, 0), diag_windows(meta, SPAN, 3, 1000),
		budget, verify, lambda m: digest(numpy.rint(m)))
	out['rss_mb'] = common.peak_rss_mb()
	return out


def read_row(ctx, kind, path, table, n, budget, verify):
	"""Median seconds per row (a bin's contacts over the genome) of one file."""

	meta = load_meta(table)
	reader = Cooler(path, meta) if kind == 'cooler' else FigWig(path, meta)
	rows = lambda seed, k: numpy.random.default_rng(seed).integers(0, meta['nbins'], k).tolist()  # noqa: E731
	out = time_queries(reader.row, rows(0, n), rows(1000, 1), budget, verify,
		lambda v: digest(numpy.rint(v)))
	out['rss_mb'] = common.peak_rss_mb()
	return out


def expected_rows(table, triangle, meta):
	"""Per-row sums the pixel table implies for a figwig file, and its entry
	counts (the last track of a both-triangle file is the diagonal)."""

	n = meta['nbins'] + (1 if triangle == 'both' else 0)
	sums = numpy.zeros(n, dtype=numpy.int64)
	counts = numpy.zeros(n, dtype=numpy.int64)
	for bin1, bin2, count in chunks(table, 20_000_000):
		sums += numpy.bincount(bin1, weights=count, minlength=n).astype(numpy.int64)
		counts += numpy.bincount(bin1, minlength=n)
		if triangle == 'both':
			off = bin1 != bin2
			sums += numpy.bincount(bin2[off], weights=count[off], minlength=n).astype(numpy.int64)
			counts += numpy.bincount(bin2[off], minlength=n)
			on = ~off
			sums[-1] += int(count[on].sum())
			counts[-1] += int(on.sum())
	return sums, counts


def check_totals(ctx, kind, path, table, triangle='upper'):
	"""The file's counts against the pixel table: the total, and for figwig
	each row's sum and number of pixels. Raises on a mismatch."""

	meta = load_meta(table)
	if kind == 'cooler':
		import h5py
		with h5py.File(path, 'r') as handle:
			count = handle['pixels/count']
			total = sum(int(count[a:a + 50_000_000].sum(dtype=numpy.int64))
				for a in range(0, len(count), 50_000_000))
			n = len(count)
		if total != meta['sum'] or n != meta['nnz']:
			raise RuntimeError('cooler file: sum {} and {} pixels, table {} and {}'.format(
				total, n, meta['sum'], meta['nnz']))
		return {'sum': total, 'pixels': n}

	from figwig import FigWigReader
	fw = FigWigReader(path)
	sums, counts = expected_rows(table, triangle, meta)
	got_sums = numpy.zeros(len(sums), dtype=numpy.int64)
	got_counts = numpy.zeros(len(sums), dtype=numpy.int64)
	step = 20_000
	for a in range(0, len(sums), step):
		tracks = numpy.arange(a, min(a + step, len(sums)))
		e = fw.read_entries(tracks=tracks, n_jobs=ctx['threads'])
		got_sums += numpy.bincount(e.tracks, weights=e.values, minlength=len(sums)).astype(numpy.int64)
		got_counts += numpy.bincount(e.tracks, minlength=len(sums))
	bad = numpy.flatnonzero((got_sums != sums) | (got_counts != counts))
	if len(bad):
		raise RuntimeError('figwig file {}: {} rows differ from the table, first {}'.format(
			triangle, len(bad), bad[:5].tolist()))
	return {'sum': int(got_sums.sum()), 'entries': int(got_counts.sum())}


def hic_row(ctx):
	quick = ctx['quick']
	mcool = common.testdata(SOURCES['quick' if quick else 'full'])
	work = os.path.join(ctx['scratch'], 'hic')
	table = os.path.join(work, 'table')
	os.makedirs(work, exist_ok=True)
	n_windows, n_rows = (20, 4) if quick else (200, 200)
	row_budget = 20 if quick else 120
	raw = {}
	paths = {'cooler': os.path.join(work, 'table.cool'),
		'upper': os.path.join(work, 'upper.fw'), 'both': os.path.join(work, 'both.fw')}

	raw['table'] = common.worker(ctx, 'hic:extract', threads=1, mcool=mcool,
		resolution=RESOLUTION, out=table)
	meta = load_meta(table)

	# Writes: the median of the repeats, the last file kept for the reads.
	w_cooler, raw['write_cooler'] = common.repeated(ctx, 'hic:write_cooler', threads=1,
		table=table, out=paths['cooler'])
	w_upper, raw['write_figwig_upper'] = common.repeated(ctx, 'hic:write_figwig',
		threads=ctx['threads'], table=table, out=paths['upper'],
		spill_dir=os.path.join(work, 'spill'), **UPPER)
	raw['write_figwig_both'] = common.worker(ctx, 'hic:write_figwig', threads=ctx['threads'],
		table=table, out=paths['both'], spill_dir=os.path.join(work, 'spill'), **BOTH)
	size_cooler = common.size_mb(paths['cooler'])
	size_upper = common.size_mb(paths['upper'])
	size_both = common.size_mb(paths['both'])

	raw['totals'] = {
		'cooler': common.worker(ctx, 'hic:check_totals', threads=ctx['threads'],
			kind='cooler', path=paths['cooler'], table=table),
		'upper': common.worker(ctx, 'hic:check_totals', threads=ctx['threads'],
			kind='figwig', path=paths['upper'], table=table, triangle='upper'),
		'both': common.worker(ctx, 'hic:check_totals', threads=ctx['threads'],
			kind='figwig', path=paths['both'], table=table, triangle='both')}

	# Windows, one thread on both sides.
	args = dict(table=table, n=n_windows, budget=120, verify=20)
	r_cooler, raw['window_cooler'] = common.repeated(ctx, 'hic:read_window', threads=1,
		kind='cooler', path=paths['cooler'], **args)
	r_figwig, raw['window_figwig'] = common.repeated(ctx, 'hic:read_window', threads=1,
		kind='figwig', path=paths['upper'], **args)
	compared = same_hashes('2 Mb window', raw['window_cooler'][0]['hashes'],
		raw['window_figwig'][0]['hashes'])

	# Rows: cooler scans every earlier row, so each query is seconds; one
	# process of at least a dozen queries on its side.
	args = dict(table=table, n=n_rows, budget=row_budget, verify=5)
	row_cooler, raw['row_cooler'] = common.repeated(ctx, 'hic:read_row', threads=1, repeats=1,
		kind='cooler', path=paths['cooler'], **args)
	row_figwig, raw['row_figwig'] = common.repeated(ctx, 'hic:read_row', threads=1,
		kind='figwig', path=paths['both'], **args)
	same_hashes('row', raw['row_cooler'][0]['hashes'], raw['row_figwig'][0]['hashes'])

	for path in paths.values():
		common.fresh(path)
	common.fresh(work)

	kb = RESOLUTION / 1000
	contacts = '{:,.0f}M'.format(meta['sum'] / 1e6) if meta['sum'] >= 1e6 else '{:,}'.format(meta['sum'])
	raw['windows_compared'] = compared
	return {
		'setting': 'K562 Hi-C, {:g} kb, {} contacts; 2 Mb diagonal window as a dense '
			'{k} x {k} block'.format(kb, contacts, k=SPAN // RESOLUTION),
		'baseline': 'cooler (Python API)',
		'read': [common.item(r_figwig, r_cooler, label='window')],
		'write': [common.item(w_upper, w_cooler)],
		'size': [common.item(size_upper, size_cooler, unit='MB')],
		'inputs': {'mcool': os.path.relpath(mcool, common.TESTDATA), 'resolution': RESOLUTION,
			'contacts': meta['sum'], 'pixels': meta['nnz'], 'pixels_in_source': meta['nnz_source'],
			'repeated_pixels_summed': meta['repeated_pixels_summed'],
			'figwig_layout': 'bin coordinates, upper triangle, one track per bin, '
			'tile {tile}, groups of {group_size}'.format(**UPPER),
			'read_threads': READ_THREADS, 'write_threads': {'figwig': ctx['threads'], 'cooler': 1},
			'windows': n_windows},
		'raw': raw,
		'notes': ['With both triangles ({size} at {kb:g} kb) the figwig file also reads a bin\'s whole '
			'row in {fig} against cooler\'s {coo}.'.format(size=render.megabytes(size_both), kb=kb,
			fig=render.seconds(row_figwig), coo=render.seconds(row_cooler))],
	}


ROWS = {'hic': hic_row}

if __name__ == '__main__':
	sys.exit('run benchmarks/readme_table/run.py hic')
