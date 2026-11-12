# schic.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The single-cell Hi-C row of the README table: a figwig file against a
scool.

Input: Nagano et al. 2017 mouse ES cells (mm9, 3,882 cells) at 10 kb from the
test collection's `.scool`, rebuilt into a pixel table on disk (`hic_common.py`):
every cell's upper-triangle raw counts, sorted by (cell, bin1, bin2). Both
writers start from that table.

The figwig file has one track per cell in position order. Each pair of
chromosomes (c1 <= c2) is a pseudo-chromosome of n1 x n2 positions and the
pixel (i, j) sits at (i - o1) * n2 + (j - o2), upper triangle only. A pseudobulk
window is a 2 Mb diagonal window summed over the cells of each of four random
labels, a (4, 200, 200) array: figwig reads it with `labels=`, the scool by
reading the window of every cell and summing them. One whole cell is all of a
cell's pixels. Reads use one thread on both sides. The write includes the work
of turning the table into `write_sparse` calls (a spill by chromosome, a sort
by position). A second figwig file in track order (level 6; the writer's
default there is uncompressed) is built for the note: it reads a cell by
decoding that cell's track alone.

Every read is hashed against the scool's on the same queries, and every
file's counts are summed per cell against the table; a mismatch raises.
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
from hic_common import bin_starts
from hic_common import tile_calls
from hic_common import digest
from hic_common import mirror
from hic_common import time_queries
from hic_common import same_hashes

SOURCES = {
	'full': 'sc-hi-c/full/sc-hi-c_full_mESC_mm9_PE_nagano2017-10kb-raw.scool',
	'quick': 'sc-hi-c/large/sc-hi-c_large_mESC_mm9_PE_nagano2017-10kb-raw.scool',
}
RESOLUTION = 10000
SPAN = 2_000_000
N_LABELS = 4
TILE = 16384
READ_THREADS = 1


def read_cells(table, meta, first, last, columns=('bin1', 'bin2', 'count')):
	"""The pixels of cells [first, last), one int32 array per column."""

	a, b = int(meta['cell_offsets'][first]), int(meta['cell_offsets'][last])
	out = []
	for column in columns:
		with open(os.path.join(table, column + '.i32'), 'rb') as handle:
			out.append(numpy.fromfile(handle, dtype=numpy.int32, count=b - a, offset=4 * a))
	return out


def cell_batches(meta, max_pixels=20_000_000):
	"""[first, last) ranges of cells holding about max_pixels pixels."""

	offsets = meta['cell_offsets']
	first, n = 0, len(meta['cells'])
	while first < n:
		last = int(numpy.searchsorted(offsets, offsets[first] + max_pixels, side='right')) - 1
		last = min(n, max(first + 1, last))
		yield first, last
		first = last


def cell_sums(table, meta):
	"""Each cell's number of pixels and its sum of counts, from the table."""

	count = numpy.fromfile(os.path.join(table, 'count.i32'), dtype=numpy.int32)
	cumulative = numpy.concatenate([[0], numpy.cumsum(count, dtype=numpy.int64)])
	offsets = meta['cell_offsets']
	return numpy.diff(offsets), cumulative[offsets[1:]] - cumulative[offsets[:-1]]


def extract(ctx, scool, out):
	"""Every cell of an scool as a pixel table, sorted by (cell, bin1, bin2).
	Untimed. The Zenodo file stores each cell as a group at the top level
	(the layout before cooler 0.8.9); the test collection's subsets use
	`cells/`."""

	import h5py

	os.makedirs(out, exist_ok=True)
	with h5py.File(scool, 'r') as handle:
		group = handle['cells'] if 'cells' in handle else handle
		names = sorted(group.keys())
		first = group[names[0]]
		chroms = [c.decode() if isinstance(c, bytes) else str(c) for c in first['chroms/name'][:]]
		lengths = first['chroms/length'][:].astype(int).tolist()
		chrom_offsets = first['indexes/chrom_offset'][:].astype(int).tolist()
		nbins = int(first.attrs['nbins'])
		assert int(first.attrs['bin-size']) == RESOLUTION

		parts = {key: [] for key in ('bin1', 'bin2', 'count')}
		sizes = []
		for name in names:
			cell = group[name]
			assert int(cell.attrs['nbins']) == nbins
			bin1 = cell['pixels/bin1_id'][:].astype(numpy.int64)
			bin2 = cell['pixels/bin2_id'][:].astype(numpy.int64)
			count = cell['pixels/count'][:]
			assert (bin1 <= bin2).all() and (count > 0).all()
			order = numpy.argsort(bin1 * nbins + bin2, kind='stable')
			parts['bin1'].append(bin1[order].astype(numpy.int32))
			parts['bin2'].append(bin2[order].astype(numpy.int32))
			parts['count'].append(count[order].astype(numpy.int32))
			sizes.append(len(bin1))

	sizes = numpy.array(sizes, dtype=numpy.int64)
	for key, arrays in parts.items():
		numpy.concatenate(arrays).tofile(os.path.join(out, key + '.i32'))
	numpy.concatenate([[0], numpy.cumsum(sizes)]).astype(numpy.int64).tofile(
		os.path.join(out, 'cell_offsets.i64'))
	total = sum(int(a.sum(dtype=numpy.int64)) for a in parts['count'])
	meta = {'source': scool, 'resolution': RESOLUTION, 'chroms': chroms, 'lengths': lengths,
		'chrom_offsets': chrom_offsets, 'nbins': nbins, 'cells': names, 'nnz': int(sizes.sum()),
		'sum': total}
	with open(os.path.join(out, 'meta.json'), 'w') as handle:
		json.dump(meta, handle)
	return {'cells': len(names), 'nnz': meta['nnz'], 'sum': total,
		'nnz_per_cell_median': float(numpy.median(sizes))}


def write_scool(ctx, table, out):
	import collections.abc
	import cooler

	meta = load_meta(table)
	common.fresh(out)
	starts, chrom = bin_starts(meta)
	lengths = numpy.array(meta['lengths'])
	ends = numpy.minimum(starts + meta['resolution'], lengths[chrom])
	bins = pandas.DataFrame({'chrom': numpy.array(meta['chroms'])[chrom], 'start': starts,
		'end': ends})

	class LazyCells(collections.abc.Mapping):
		"""Cell name -> pixel DataFrame, read from the table when asked for."""

		def __init__(self):
			self.index = {name: i for i, name in enumerate(meta['cells'])}

		def __getitem__(self, name):
			i = self.index[name]
			bin1, bin2, count = read_cells(table, meta, i, i + 1)
			return pandas.DataFrame({'bin1_id': bin1.astype(numpy.int64),
				'bin2_id': bin2.astype(numpy.int64), 'count': count})

		def __iter__(self):
			return iter(meta['cells'])

		def __len__(self):
			return len(meta['cells'])

	tmp = os.path.join(os.path.dirname(out), 'scool-tmp')
	os.makedirs(tmp, exist_ok=True)
	start = time.perf_counter()
	cooler.create_scool(out, bins, LazyCells(), ordered=True, symmetric_upper=True,
		dtypes={'count': 'int32'}, temp_dir=tmp)
	seconds = time.perf_counter() - start
	shutil.rmtree(tmp, ignore_errors=True)
	return {'seconds': seconds, 'mb': common.size_mb(out), 'rss_mb': common.peak_rss_mb()}


def write_figwig(ctx, table, out, spill_dir, order, level):
	"""One track per cell, each chromosome pair a pseudo-chromosome. The
	entries are spilled by the row's chromosome, then each is sorted by
	(column chromosome, position) and written in calls that end on a tile."""

	from figwig import FigWigWriter

	meta = load_meta(table)
	common.fresh(out)
	offsets = meta['chrom_offsets']
	sizes = numpy.diff(offsets)
	names = meta['chroms']
	n_chroms = len(names)
	blocks = [(c1, c2) for c1 in range(n_chroms) for c2 in range(c1, n_chroms)]
	chrom_sizes = {'{}|{}'.format(names[c1], names[c2]): int(sizes[c1] * sizes[c2])
		for c1, c2 in blocks}
	assert max(chrom_sizes.values()) < 2 ** 32

	start = time.perf_counter()
	shutil.rmtree(spill_dir, ignore_errors=True)
	os.makedirs(spill_dir)
	handles = [[open(os.path.join(spill_dir, '{}.{}'.format(c, k)), 'wb')
		for k in ('cell', 'bin1', 'bin2', 'count')] for c in range(n_chroms)]
	for first, last in cell_batches(meta):
		bin1, bin2, count = read_cells(table, meta, first, last)
		cells = numpy.repeat(numpy.arange(first, last, dtype=numpy.int32),
			numpy.diff(meta['cell_offsets'][first:last + 1]))
		c1 = numpy.searchsorted(offsets, bin1, side='right') - 1
		by = numpy.argsort(c1, kind='stable')
		bounds = numpy.searchsorted(c1[by], numpy.arange(n_chroms + 1))
		for c in range(n_chroms):
			take = by[bounds[c]:bounds[c + 1]]
			for handle, array in zip(handles[c], (cells, bin1, bin2, count)):
				array[take].tofile(handle)
	for quad in handles:
		for handle in quad:
			handle.close()

	tracks = [{'name': name} for name in meta['cells']]
	kwargs = {'order': 'tracks'} if order == 'tracks' else {'groups': len(tracks), 'tile': TILE}
	with FigWigWriter(out, chrom_sizes, tracks, genome='mm9', level=level,
			n_jobs=ctx['threads'], attributes={'resolution': meta['resolution'],
			'layout': 'cell'}, **kwargs) as writer:
		for c1 in range(n_chroms):
			cells, bin1, bin2, count = [numpy.fromfile(os.path.join(spill_dir,
				'{}.{}'.format(c1, k)), dtype=numpy.int32) for k in ('cell', 'bin1', 'bin2', 'count')]
			c2 = numpy.searchsorted(offsets, bin2, side='right') - 1
			positions = (bin1.astype(numpy.int64) - offsets[c1]) * sizes[c2] + (bin2 - offsets[c2])
			by = numpy.argsort((c2.astype(numpy.int64) << 32) | positions, kind='stable')
			c2, positions, cells, count = c2[by], positions[by], cells[by], count[by]
			bounds = numpy.searchsorted(c2, numpy.arange(n_chroms + 1))
			for c in range(c1, n_chroms):
				a, b = bounds[c], bounds[c + 1]
				if b == a:
					continue
				name = '{}|{}'.format(names[c1], names[c])
				p, t, v = positions[a:b], cells[a:b], count[a:b]
				cuts = tile_calls(p, TILE, 10_000_000)
				for x, y in zip(cuts[:-1], cuts[1:]):
					writer.write_sparse(name, p[x:y], t[x:y], v[x:y])
	seconds = time.perf_counter() - start
	shutil.rmtree(spill_dir, ignore_errors=True)
	return {'seconds': seconds, 'mb': common.size_mb(out), 'rss_mb': common.peak_rss_mb()}


def sort_upper(bin1, bin2, count):
	bin1, bin2 = bin1.astype(numpy.int64), bin2.astype(numpy.int64)
	keep = bin1 <= bin2
	bin1, bin2, count = bin1[keep], bin2[keep], numpy.asarray(count)[keep]
	order = numpy.lexsort((bin2, bin1))
	return bin1[order], bin2[order], count[order]


def cell_labels(n_cells, seed=0):
	return numpy.random.default_rng(seed).permutation(n_cells) % N_LABELS


def windows(meta, span, n, seed):
	"""(chromosome index, start bin) of n diagonal windows of span bp, in
	whole bins (cooler refuses a region past the chromosome's end)."""

	k = span // meta['resolution']
	sizes = numpy.array(meta['lengths']) // meta['resolution']
	eligible = numpy.flatnonzero(sizes >= k + 1)
	p = sizes[eligible] / sizes[eligible].sum()
	random_state = numpy.random.default_rng(seed)
	chroms = random_state.choice(eligible, size=n, p=p)
	return [(int(c), int(random_state.integers(0, sizes[c] - k + 1))) for c in chroms]


class Scool:
	def __init__(self, path, meta):
		import cooler
		names = cooler.fileops.list_scool_cells(path)
		by_name = {name.rsplit('/', 1)[-1]: name for name in names}
		self.cells = [cooler.Cooler(path + '::' + by_name[name]) for name in meta['cells']]
		self.meta = meta

	def cell(self, i):
		pixels = self.cells[i].pixels()[:]
		return sort_upper(pixels['bin1_id'].to_numpy(), pixels['bin2_id'].to_numpy(),
			pixels['count'].to_numpy())

	def pseudobulk(self, chrom, start, k, labels):
		res = self.meta['resolution']
		region = (self.meta['chroms'][chrom], start * res, (start + k) * res)
		out = numpy.zeros((N_LABELS, k, k), dtype=numpy.int64)
		for c, label in zip(self.cells, labels):
			out[label] += c.matrix(balance=False).fetch(region)
		return out


class FwCell:
	"""One track per cell; chromosome pairs flattened, upper triangle."""

	def __init__(self, path, meta, jobs):
		from figwig import FigWigReader
		self.fw = FigWigReader(path)
		self.meta = meta
		self.jobs = jobs
		names = meta['chroms']
		offsets = meta['chrom_offsets']
		sizes = numpy.diff(offsets)
		block = {'{}|{}'.format(names[c1], names[c2]): (c1, c2)
			for c1 in range(len(names)) for c2 in range(c1, len(names))}
		file_chroms = list(self.fw.chrom_sizes)
		self.first_bin = numpy.array([offsets[block[n][0]] for n in file_chroms])
		self.second_bin = numpy.array([offsets[block[n][1]] for n in file_chroms])
		self.width = numpy.array([sizes[block[n][1]] for n in file_chroms])
		self.sizes = sizes

	def cell(self, i):
		e = self.fw.read_entries(tracks=[i], n_jobs=self.jobs)
		chroms = e.chroms.astype(numpy.int64)
		positions = e.positions.astype(numpy.int64)
		bin1 = self.first_bin[chroms] + positions // self.width[chroms]
		bin2 = self.second_bin[chroms] + positions % self.width[chroms]
		return sort_upper(bin1, bin2, e.values)

	def pseudobulk(self, chrom, start, k, labels):
		name = '{}|{}'.format(self.meta['chroms'][chrom], self.meta['chroms'][chrom])
		n = self.sizes[chrom]
		rows = numpy.arange(start, start + k)
		y = self.fw.read([name] * k, rows * n + start, k, labels=labels, n_jobs=self.jobs)
		y = numpy.rint(y).astype(numpy.int64).transpose(1, 0, 2)
		return numpy.stack([mirror(m) for m in y])


def reader_for(kind, path, meta, jobs):
	return Scool(path, meta) if kind == 'scool' else FwCell(path, meta, jobs)


def read_pseudobulk(ctx, kind, path, table, n, budget, verify, jobs=READ_THREADS):
	"""Median seconds per 2 Mb window summed over the cells of each label."""

	meta = load_meta(table)
	reader = reader_for(kind, path, meta, jobs)
	k = SPAN // meta['resolution']
	labels = cell_labels(len(meta['cells']))
	out = time_queries(lambda item: reader.pseudobulk(item[0], item[1], k, labels),
		windows(meta, SPAN, n, 0), windows(meta, SPAN, 2, 1000), budget, verify, digest)
	out['rss_mb'] = common.peak_rss_mb()
	return out


def read_cell(ctx, kind, path, table, n, budget, verify, jobs=READ_THREADS):
	"""Median seconds per whole cell: all of its pixels, sorted."""

	meta = load_meta(table)
	reader = reader_for(kind, path, meta, jobs)
	cells = lambda seed, m: numpy.random.default_rng(seed).integers(  # noqa: E731
		0, len(meta['cells']), m).tolist()
	out = time_queries(reader.cell, cells(0, n), cells(1000, 2), budget, verify,
		lambda pixels: digest(*pixels))
	out['rss_mb'] = common.peak_rss_mb()
	return out


def check_totals(ctx, kind, path, table):
	"""Each cell's number of pixels and sum of counts in the file against the
	table. Raises on a mismatch."""

	meta = load_meta(table)
	n_cells = len(meta['cells'])
	pixels, sums = cell_sums(table, meta)
	if kind == 'scool':
		import h5py
		got_pixels = numpy.zeros(n_cells, dtype=numpy.int64)
		got_sums = numpy.zeros(n_cells, dtype=numpy.int64)
		with h5py.File(path, 'r') as handle:
			group = handle['cells'] if 'cells' in handle else handle
			for i, name in enumerate(meta['cells']):
				count = group[name]['pixels/count'][:]
				got_pixels[i], got_sums[i] = len(count), count.sum(dtype=numpy.int64)
	else:
		from figwig import FigWigReader
		e = FigWigReader(path).read_entries(n_jobs=ctx['threads'])
		got_pixels = numpy.bincount(e.tracks, minlength=n_cells)
		got_sums = numpy.bincount(e.tracks, weights=e.values, minlength=n_cells).astype(numpy.int64)
	bad = numpy.flatnonzero((got_pixels != pixels) | (got_sums != sums))
	if len(bad):
		raise RuntimeError('{} file: {} cells differ from the table, first {}'.format(
			kind, len(bad), bad[:5].tolist()))
	return {'sum': int(got_sums.sum()), 'pixels': int(got_pixels.sum())}


def schic_row(ctx):
	quick = ctx['quick']
	scool = common.testdata(SOURCES['quick' if quick else 'full'])
	work = os.path.join(ctx['scratch'], 'schic')
	table = os.path.join(work, 'table')
	os.makedirs(work, exist_ok=True)
	n_windows, n_cells, budget = (3, 5, 60) if quick else (50, 100, 120)
	raw = {}
	paths = {'scool': os.path.join(work, 'cells.scool'), 'cell': os.path.join(work, 'cell.fw'),
		'tracks': os.path.join(work, 'tracks.fw')}
	spill = os.path.join(work, 'spill')

	raw['table'] = common.worker(ctx, 'schic:extract', threads=1, scool=scool, out=table)
	meta = load_meta(table)

	w_scool, raw['write_scool'] = common.repeated(ctx, 'schic:write_scool', threads=1,
		table=table, out=paths['scool'])
	w_figwig, raw['write_figwig'] = common.repeated(ctx, 'schic:write_figwig',
		threads=ctx['threads'], table=table, out=paths['cell'], spill_dir=spill,
		order='positions', level=None)
	raw['write_figwig_tracks'] = common.worker(ctx, 'schic:write_figwig', threads=ctx['threads'],
		table=table, out=paths['tracks'], spill_dir=spill, order='tracks', level=6)
	size_scool = common.size_mb(paths['scool'])
	size_figwig = common.size_mb(paths['cell'])
	size_tracks = common.size_mb(paths['tracks'])

	raw['totals'] = {
		'scool': common.worker(ctx, 'schic:check_totals', threads=ctx['threads'],
			kind='scool', path=paths['scool'], table=table),
		'figwig': common.worker(ctx, 'schic:check_totals', threads=ctx['threads'],
			kind='figwig', path=paths['cell'], table=table),
		'figwig_tracks': common.worker(ctx, 'schic:check_totals', threads=ctx['threads'],
			kind='figwig', path=paths['tracks'], table=table)}

	# Pseudobulk windows. The scool reads every cell's window (seconds a
	# query), figwig does one read with labels=.
	args = dict(table=table, n=n_windows, budget=budget)
	p_scool, raw['pseudobulk_scool'] = common.repeated(ctx, 'schic:read_pseudobulk', threads=1,
		repeats=1, kind='scool', path=paths['scool'], verify=3, **args)
	p_figwig, raw['pseudobulk_figwig'] = common.repeated(ctx, 'schic:read_pseudobulk', threads=1,
		kind='figwig', path=paths['cell'], verify=3, **args)
	compared_pseudobulk = same_hashes('pseudobulk', raw['pseudobulk_scool'][0]['hashes'],
		raw['pseudobulk_figwig'][0]['hashes'])

	# Whole cells: the position-order file decodes every chunk of the group
	# (seconds a cell), the scool and the track-order file read one cell.
	args = dict(table=table, n=n_cells, budget=budget, verify=3)
	c_scool, raw['cell_scool'] = common.repeated(ctx, 'schic:read_cell', threads=1,
		kind='scool', path=paths['scool'], **args)
	c_figwig, raw['cell_figwig'] = common.repeated(ctx, 'schic:read_cell', threads=1, repeats=1,
		kind='figwig', path=paths['cell'], **args)
	c_tracks, raw['cell_figwig_tracks'] = common.repeated(ctx, 'schic:read_cell', threads=1,
		kind='figwig', path=paths['tracks'], **args)
	for key in ('cell_figwig', 'cell_figwig_tracks'):
		same_hashes('cell', raw['cell_scool'][0]['hashes'], raw[key][0]['hashes'])
	# Also at the writer's thread count, for the record.
	raw['cell_figwig_threads'] = common.worker(ctx, 'schic:read_cell', threads=ctx['threads'],
		kind='figwig', path=paths['cell'], jobs=ctx['threads'], **args)

	for path in paths.values():
		common.fresh(path)
	common.fresh(work)

	contacts = '{:,.0f}M'.format(meta['sum'] / 1e6) if meta['sum'] >= 1e6 else '{:,}'.format(meta['sum'])
	k = SPAN // RESOLUTION
	raw['pseudobulk_compared'] = compared_pseudobulk
	return {
		'setting': 'Single-cell Hi-C, mouse ES cells (Nagano 2017), {:,} cells, {:g} kb, {} contacts; '
			'2 Mb diagonal window summed over the cells of {} labels, ({}, {k}, {k}), and one whole '
			'cell'.format(len(meta['cells']), RESOLUTION / 1000, contacts, N_LABELS, N_LABELS, k=k),
		'baseline': 'scool (cooler)',
		'read': [common.item(p_figwig, p_scool, label='pseudobulk window'),
			common.item(c_figwig, c_scool, label='one cell')],
		'write': [common.item(w_figwig, w_scool)],
		'size': [common.item(size_figwig, size_scool, unit='MB')],
		'inputs': {'scool': os.path.relpath(scool, common.TESTDATA), 'resolution': RESOLUTION,
			'cells': len(meta['cells']), 'pixels': meta['nnz'], 'contacts': meta['sum'],
			'figwig_layout': 'one track per cell in position order, chromosome pairs as '
			'pseudo-chromosomes, upper triangle, tile {}, one group'.format(TILE),
			'read_threads': READ_THREADS, 'write_threads': {'figwig': ctx['threads'], 'scool': 1},
			'pseudobulk_windows': n_windows, 'cells_read': n_cells},
		'raw': raw,
		'notes': ['With one track per cell in track order (level 6, {size}) the figwig file reads one '
			'whole cell in {t}.'.format(size=render.megabytes(size_tracks), t=render.seconds(c_tracks))],
	}


ROWS = {'schic': schic_row}

if __name__ == '__main__':
	sys.exit('run benchmarks/readme_table/run.py schic')
