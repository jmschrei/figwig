# test_surface.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Cross-cutting tests built from the failure classes the battle tests and
the persona rounds found: damage anywhere in a file, batches encoded in
several goes, sums under every threading and scratch condition, every
command's overwrite and missing-input behaviour, and abandoned writers."""

import os
import sys
import subprocess

import numpy
import pytest

import figwig.figwig

from figwig import FigWigReader
from figwig import FigWigWriter
from figwig import write_figwig
from figwig.__main__ import main

from .test_figwig import CHROM_SIZES
from .test_figwig import assert_identical

ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1')


def _random_file(path, rng, order='positions', level=6, tile=64,
	several_goes=False):
	"""A small file of five tracks of every class, written as windows,
	intervals and entries, whole tiles and partial ones, and the dense
	float64 expectation of every track over the genome."""

	tracks = [{'name': 'u8'}, {'name': 'i16', 'dtype': 'int16'}, {'name': 'f32',
		'dtype': 'float32'}, {'name': 'f64', 'dtype': 'float64'}, {'name': 'ones'}]
	expected = {chrom: numpy.zeros((5, length)) for chrom, length in
		CHROM_SIZES.items()}
	with FigWigWriter(path, CHROM_SIZES, tracks, order=order, level=level,
			tile=tile, groups=[['u8', 'i16'], ['f32', 'f64'], ['ones']],
			n_jobs=2) as writer:
		for chrom, length in CHROM_SIZES.items():
			at = 0
			while at < length:
				width = int(rng.integers(1, 5 * tile))
				width = min(width, length - at)
				values = numpy.zeros((1, 5, width))
				values[0, 0] = rng.integers(0, 200, width) * (rng.random(width) < 0.3)
				values[0, 1] = rng.integers(-300, 300, width) * (rng.random(width) <
					0.2)
				values[0, 2] = numpy.round(rng.normal(0, 3, width), 3) * (rng.random(
					width) < 0.5)
				values[0, 2][rng.random(width) < 0.05] = numpy.nan
				# The float32 track keeps float32's rounding of what it is given.
				values[0, 2] = values[0, 2].astype(numpy.float32)
				values[0, 3] = (rng.integers(0, 2 ** 40, width) + 0.5) * (rng.random(
					width) < 0.4)
				values[0, 4] = rng.random(width) < 0.1
				for group in (['u8', 'i16'], ['f32', 'f64'], ['ones']):
					ids = [t['name'] for t in tracks]
					rows = [ids.index(name) for name in group]
					writer.write(chrom, [at], values[:, rows], tracks=group)
				expected[chrom][:, at:at + width] = values[0]
				at += width
	return expected


@pytest.mark.parametrize('order', ['positions', 'tracks'])
@pytest.mark.parametrize('level', [6, 0])
def test_every_kind_of_write_reads_back(tmp_path, monkeypatch, order, level):
	"""Random tracks of every class, written in windows of random widths
	in both orders at both levels, with the encoders' batches cut small so
	that batches are encoded in several goes, read back exactly, with
	their checksums, as dense windows, entries, sums and tile sums."""

	monkeypatch.setattr(figwig.figwig, '_BATCH_BYTES', 3000)
	rng = numpy.random.default_rng(21)
	path = str(tmp_path / 'random.fw')
	expected = _random_file(path, rng, order=order, level=level)
	reader = FigWigReader(path, verify=True)
	assert [t['dtype'] for t in reader.tracks] == ['uint8', 'int16', 'float32',
		'float64', 'uint8']
	for chrom, length in CHROM_SIZES.items():
		y = reader.read(chrom, [0], length, dtype='float64', n_jobs=2)[0]
		assert_identical(y, expected[chrom].astype(numpy.float64))
		if reader.order == 'positions':
			sums = reader.tile_sums(chroms=chrom)
			dense = numpy.nan_to_num(expected[chrom], nan=0).astype(numpy.float32)
			padded = numpy.zeros((5, sums.shape[1] * reader.tile), dtype=numpy.float32)
			padded[:, :length] = dense
			binned = padded.reshape(5, -1, reader.tile).sum(-1)
			# float32 sums of values of both signs, in another order.
			numpy.testing.assert_allclose(sums[:, ~numpy.isnan(sums).any(0)],
				binned[:, ~numpy.isnan(sums).any(0)], rtol=1e-4, atol=1e-3)
	entries = reader.read_entries()
	assert len(entries.values) == sum(int((numpy.nan_to_num(expected[c], nan=1)
		!= 0).sum()) for c in CHROM_SIZES)
	starts = numpy.array([0, 100, 4000, 2900, 10])
	chroms = numpy.array(['chr1', 'chr1', 'chr1', 'chr2', 'chrM'])
	ends = starts + numpy.array([5000, 333, 999, 101, 60])
	totals = reader.sums(chroms, starts, ends, dtype='float64', nan='skip',
		n_jobs=2)
	for i in range(len(starts)):
		stop = min(int(ends[i]), CHROM_SIZES[chroms[i]])
		numpy.testing.assert_array_equal(totals[i], numpy.nansum(expected[
			chroms[i]][:, starts[i]:stop], axis=-1))


@pytest.mark.parametrize('level', [6, 0])
def test_damage_anywhere_is_refused_or_invisible(tmp_path, level):
	"""A bit flipped at any byte of a file, header, tables, index or chunks,
	leaves a read that raises ValueError or gives the same values; with
	verify=True that holds for uncompressed chunks too. Nothing but
	ValueError escapes."""

	rng = numpy.random.default_rng(22)
	path = str(tmp_path / 'intact.fw')
	_random_file(path, rng, level=level)
	with open(path, 'rb') as handle:
		intact = bytearray(handle.read())
	reader = FigWigReader(path)
	windows = (numpy.array(['chr1', 'chr2', 'chrM']), numpy.array([3000, 10, 0]))
	good = reader.read(*windows, 70, dtype='float64', n_jobs=1)
	damaged = str(tmp_path / 'damaged.fw')
	outcomes = {'refused': 0, 'same': 0}
	# Every fifth byte of the header, the tables and the index, where each
	# byte is part of some field, and every 61st byte of the chunks between
	# them, which the CRC-32 and the codec guard alike; every fifth byte of
	# the chunks was 16,000 flips and 25 s for the compressed file.
	tables_at = reader._tables_at
	offsets = (list(range(0, 128, 5)) + list(range(128, tables_at, 61))
		+ list(range(tables_at, len(intact), 5)))
	for offset in offsets:
		data = bytearray(intact)
		data[offset] ^= 1 << int(rng.integers(0, 8))
		with open(damaged, 'wb') as handle:
			handle.write(data)
		try:
			r = FigWigReader(damaged, verify=True)
			y = r.read(*windows, 70, dtype='float64', n_jobs=1)
			r.tile_sums()
		except ValueError:
			outcomes['refused'] += 1
			continue
		# The tables may change silently: a flipped attribute or statistic
		# is still a well-formed file, and a reader cannot know. The values
		# cannot.
		assert_identical(y, good)
		outcomes['same'] += 1
	assert outcomes['refused'] > 0 and outcomes['same'] > 0


@pytest.mark.parametrize('n_jobs', [1, 4])
def test_sums_agree_under_every_condition(tmp_path, monkeypatch, n_jobs):
	"""Binned sums, interval sums and label sums over windows that straddle
	tiles and batch cuts, with and without a scratch tile, equal the sums
	of the dense windows, in both orders, on one thread and on four."""

	rng = numpy.random.default_rng(23)
	expected = None
	paths = {}
	for order in ('positions', 'tracks'):
		paths[order] = str(tmp_path / (order + '.fw'))
		found = _random_file(paths[order], numpy.random.default_rng(23),
			order=order)
		expected = found if expected is None else expected
	n = 150
	names = numpy.array(list(CHROM_SIZES))
	chroms = names[rng.integers(0, 3, n)]
	lengths = numpy.array([CHROM_SIZES[c] for c in chroms])
	starts = (rng.random(n) * (lengths - 1)).astype(numpy.int64)
	ends = numpy.minimum(starts + rng.integers(1, 900, n), lengths)
	dense = [expected[chroms[i]][:, starts[i]:ends[i]] for i in range(n)]
	for scratch in ('scratch', 'none'):
		if scratch == 'none':
			monkeypatch.setattr(figwig.figwig, '_SCRATCH_BYTES', 0)
		for batches in ((256, 1 << 25), (2, 500)):
			monkeypatch.setattr(figwig.figwig, '_BATCH_CHUNKS', batches[0])
			monkeypatch.setattr(figwig.figwig, '_BATCH_BYTES', batches[1])
			for order, path in paths.items():
				reader = FigWigReader(path)
				y = reader.sums(chroms, starts, ends, dtype='float64', n_jobs=n_jobs)
				numpy.testing.assert_array_equal(y, numpy.array([d.sum(-1) for d
					in dense]))
				y = reader.sums(chroms, starts, ends, dtype='float64', nan='skip',
					n_jobs=n_jobs)
				numpy.testing.assert_array_equal(y, numpy.array([numpy.nansum(d,
					-1) for d in dense]))
				y = reader.sums(chroms, starts, ends, dtype='float64', nan='count',
					n_jobs=n_jobs)
				numpy.testing.assert_array_equal(y, numpy.array([(~numpy.isnan(d)
					).sum(-1) for d in dense]))
				inside = starts + 300 <= lengths
				y = reader.read(chroms[inside], starts[inside], 300, bins=50,
					dtype='float64', n_jobs=n_jobs)
				full = numpy.stack([expected[c][:, s:s + 300] for c, s in zip(
					chroms[inside], starts[inside])])
				numpy.testing.assert_array_equal(y, full.reshape(len(full), 5, 6,
					50).sum(-1))
				y = reader.read(chroms[inside], starts[inside], 300, labels=[0, 0,
					1, 1, 2], dtype='float64', n_jobs=n_jobs)
				numpy.testing.assert_array_equal(y[:, 0], full[:, :2].sum(1))
				numpy.testing.assert_array_equal(y[:, 2], full[:, 4])
		monkeypatch.setattr(figwig.figwig, '_SCRATCH_BYTES', 1 << 28)


def test_every_command_refuses_an_existing_output(tmp_path):
	"""Every converter refuses to write over an output without -f (or
	--force), a missing input is a message, and -v writes to stderr."""

	rng = numpy.random.default_rng(24)
	a = str(tmp_path / 'a.fw')
	_random_file(a, rng)
	b = str(tmp_path / 'b.fw')
	write_figwig(b, CHROM_SIZES, ['x'], 'chr1', [0], numpy.ones((1, 1, 10)),
		tile=64)
	cases = [
		(['merge', '-o', b, a], 'exists; -f writes over it'),
		(['reorder', a, b, '--order', 'tracks'], 'exists; -f writes over it'),
		(['merge', '-o', str(tmp_path / 'm.fw'), str(tmp_path / 'none.fw')],
			'is not a file'),
		(['reorder', str(tmp_path / 'none.fw'), str(tmp_path / 'o.fw'),
			'--order', 'tracks'], 'is not a file'),
		(['info', str(tmp_path / 'none.fw')], 'is not a file'),
		(['fw2bw', str(tmp_path / 'none.fw'), '-o', str(tmp_path / 'd')],
			'is not a file'),
		(['bw2fw', '-o', str(tmp_path / 'c.fw'), str(tmp_path / 'none.bw')],
			'is not a file'),
	]
	for argv, message in cases:
		with pytest.raises(SystemExit, match=message):
			main(argv)
	assert len(FigWigReader(b).tracks) == 1
	main(['merge', '-o', b, '-f', a])
	assert len(FigWigReader(b).tracks) == 5


def test_commands_write_progress_to_stderr(tmp_path, capsys):
	rng = numpy.random.default_rng(25)
	a = str(tmp_path / 'a.fw')
	_random_file(a, rng)
	main(['reorder', a, str(tmp_path / 't.fw'), '--order', 'tracks', '-v'])
	main(['merge', '-o', str(tmp_path / 'm.fw'), a, '-v'])
	captured = capsys.readouterr()
	assert captured.out == '' and 'tracks' in captured.err


def test_interrupt_exits_quietly(monkeypatch):
	import figwig.info

	def interrupted(argv, prog):
		raise KeyboardInterrupt

	monkeypatch.setattr(figwig.info, 'main', interrupted)
	with pytest.raises(SystemExit) as caught:
		main(['info', 'whatever.fw'])
	assert caught.value.code == 130


def test_abandoned_writers_leave_nothing(tmp_path):
	"""A new-file writer abandoned leaves no file and no partial file; an
	append writer abandoned leaves the file as it was."""

	path = str(tmp_path / 'left.fw')
	rng = numpy.random.default_rng(26)
	_random_file(path, rng)
	with open(path, 'rb') as handle:
		before = handle.read()
	for append in (False, True):
		target = path if append else str(tmp_path / 'new.fw')
		with pytest.raises(RuntimeError):
			with FigWigWriter(target, None if append else CHROM_SIZES, ['extra'],
					append=append, n_jobs=2) as writer:
				writer.write('chr1', [0], numpy.ones((1, 1, 640)))
				writer._write(len(writer._inflight))
				raise RuntimeError("stopped")
		assert [name for name in os.listdir(tmp_path) if 'partial' in name] == []
	with open(path, 'rb') as handle:
		assert handle.read() == before
	assert not os.path.exists(str(tmp_path / 'new.fw'))


def test_cli_runs_in_a_subprocess(tmp_path):
	"""The dispatcher runs each command in a fresh process and exits 0."""

	rng = numpy.random.default_rng(27)
	a = str(tmp_path / 'a.fw')
	_random_file(a, rng)
	result = subprocess.run([sys.executable, '-m', 'figwig', 'info', '-b', a],
		env=ENV, capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert 'unused bytes\t0' in result.stdout


###
# Read plans of a file of many tracks and groups

PLAN_SIZES = {'chr1': 40_000, 'chr2': 9_001, 'chrM': 70}
PLAN_WIDTH = 700


@pytest.fixture(scope='module')
def plan_file(tmp_path_factory):
	"""24 tracks in 4 groups of 6, tile 64: sparse counts, dense counts,
	float32 values in eighths, and one track of runs, written whole. Every
	value is an integer or an eighth, so every sum below is exact in float64
	in any order. Returns (path, {chrom: (24, length) float64})."""

	rng = numpy.random.default_rng(41)
	tracks = [{'name': 't{}'.format(i)} if i % 3 != 2 else {'name': 't{}'.format(
		i), 'dtype': 'float32'} for i in range(24)]
	expected = {}
	for chrom, length in PLAN_SIZES.items():
		values = numpy.zeros((24, length))
		for i in range(24):
			if i == 4:
				cuts = numpy.sort(rng.choice(length, max(1, length // 150),
					replace=False))
				values[i] = numpy.repeat(rng.integers(0, 9, len(cuts) + 1),
					numpy.diff(numpy.concatenate([[0], cuts, [length]])))
			elif i % 3 == 0:
				values[i] = rng.integers(1, 50, length) * (rng.random(length) < 0.02)
			elif i % 3 == 1:
				values[i] = rng.integers(0, 3000, length) * (rng.random(length) < 0.6)
			else:
				values[i] = rng.integers(-800, 800, length) / 8 * (rng.random(length)
					< 0.3)
		expected[chrom] = values

	path = str(tmp_path_factory.mktemp('plans') / 'plans.fw')
	with FigWigWriter(path, PLAN_SIZES, tracks, tile=64, groups=6,
			n_jobs=2) as writer:
		for chrom, values in expected.items():
			writer.write(chrom, [0], values[None])
	assert max(track['group'] for track in FigWigReader(path).tracks) == 3
	return path, expected


def _plan_windows(rng, n, clustered=False):
	"""Random windows of PLAN_WIDTH, some running past their chromosome's
	end, or with `clustered` all within 3,700 bases of chr1, so that a read
	takes few chunks but many (tile, window) pairs and its tasks are split;
	and those of them that end inside their chromosome."""

	names = numpy.array(list(PLAN_SIZES))
	lengths = numpy.array(list(PLAN_SIZES.values()))
	picked = rng.choice(3, size=n, p=[1, 0, 0] if clustered else [0.75, 0.24,
		0.01])
	chroms = names[picked]
	starts = (rng.integers(1000, 4000, n) if clustered else rng.integers(0,
		lengths[picked])).astype(numpy.int64)
	inside = starts + PLAN_WIDTH <= lengths[picked]
	return chroms, starts, inside


def _plan_expected(expected, chroms, starts, tracks=None):
	"""(n, tracks, PLAN_WIDTH) float64, NaN past a chromosome's end."""

	tracks = list(range(24)) if tracks is None else tracks
	out = numpy.full((len(starts), len(tracks), PLAN_WIDTH), numpy.nan)
	for i, (chrom, start) in enumerate(zip(chroms, starts)):
		values = expected[chrom][tracks, start:start + PLAN_WIDTH]
		out[i, :, :values.shape[1]] = values
	return out


def _check_plan_reads(path, expected, rng, n_jobs, clustered):
	reader = FigWigReader(path)
	chroms, starts, inside = _plan_windows(rng, 120, clustered)
	dense = _plan_expected(expected, chroms, starts)

	assert_identical(reader.read(chroms, starts, PLAN_WIDTH, dtype='float64',
		n_jobs=n_jobs), dense)

	# One track of 24, below one in _FEW_TRACKS, and a few out of order.
	for tracks in ([5], [23, 0, 12, 12]):
		assert_identical(reader.read(chroms, starts, PLAN_WIDTH, tracks=tracks,
			dtype='float64', n_jobs=n_jobs), _plan_expected(expected, chroms,
			starts, tracks))

	found = reader.read(chroms, starts, PLAN_WIDTH, sparse=True, dtype='float64',
		n_jobs=n_jobs)
	assert found.shape == dense.shape
	got = numpy.zeros(dense.shape)
	got[found.rows, found.channels, found.offsets] = found.values
	numpy.testing.assert_array_equal(got, numpy.nan_to_num(dense, nan=0))
	order = numpy.lexsort((found.offsets, found.channels, found.rows))
	numpy.testing.assert_array_equal(order, numpy.arange(len(order)))

	within = dense[inside]
	assert_identical(reader.read(chroms[inside], starts[inside], PLAN_WIDTH,
		bins=50, dtype='float64', n_jobs=n_jobs), within.reshape(len(within), 24,
		PLAN_WIDTH // 50, 50).sum(-1))
	labels = [i % 3 for i in range(24)]
	y = reader.read(chroms[inside], starts[inside], PLAN_WIDTH, labels=labels,
		dtype='float64', n_jobs=n_jobs)
	for label in range(3):
		numpy.testing.assert_array_equal(y[:, label], within[:, label::3].sum(1))

	lengths = numpy.array([PLAN_SIZES[chrom] for chrom in chroms])
	ends = numpy.minimum(starts + rng.integers(1, 5000, len(starts)), lengths)
	sums = reader.sums(chroms, starts, ends, dtype='float64', n_jobs=n_jobs)
	numpy.testing.assert_array_equal(sums, numpy.array([expected[chrom][:,
		start:end].sum(-1) for chrom, start, end in zip(chroms, starts, ends)]))


@pytest.mark.parametrize('name, value', [
	('_THREAD_OUT_BYTES', 1), ('_THREAD_OUT_BYTES', 1 << 40),
	('_GROUP_BATCH_BYTES', 1), ('_GROUP_BATCH_BYTES', 1 << 40),
	('_MIN_BATCH_CHUNKS', 1), ('_MIN_BATCH_CHUNKS', 1 << 20),
	('_READ_BATCHES_PER_THREAD', 1), ('_READ_BATCHES_PER_THREAD', 64),
	('_SPLIT_PAIRS', 1), ('_SPLIT_PAIRS', 1 << 40),
	('_POOL_BYTES', 1), ('_POOL_BYTES', 1 << 40)])
def test_read_plan_thresholds_never_change_values(plan_file, monkeypatch, name,
		value):
	"""Dense, few-track, sparse, binned, labelled and interval reads of a
	file of four groups give the file's values on eight threads with each
	threshold of the read planner at either extreme, in batches of the
	default size and of at most 8 chunks, over windows spread over the
	genome and windows packed into a few tiles, whose reads are fewer
	batches than threads and so are split into tasks. No other test
	sets these thresholds, and the small files elsewhere never reach most
	of them."""

	path, expected = plan_file
	monkeypatch.setattr(figwig.figwig, name, value)
	for batch_chunks in (figwig.figwig._BATCH_CHUNKS, 8):
		monkeypatch.setattr(figwig.figwig, '_BATCH_CHUNKS', batch_chunks)
		for clustered in (False, True):
			_check_plan_reads(path, expected, numpy.random.default_rng(42), 8,
				clustered)


def test_loader_batches_read_as_the_file_holds(plan_file):
	"""200 reads of 32 windows from one reader, as a data loader makes them,
	switching threads and tracks between reads, each give the file's values:
	nothing a read leaves on the reader (its pool, the group indexes it
	loaded) changes a later one."""

	path, expected = plan_file
	reader = FigWigReader(path)
	rng = numpy.random.default_rng(43)
	subsets = [None, [5], [23, 0, 12], list(range(0, 24, 2)), [1, 7, 13, 19]]
	for k in range(200):
		chroms, starts, _ = _plan_windows(rng, 32)
		tracks = subsets[k % len(subsets)]
		y = reader.read(chroms, starts, PLAN_WIDTH, tracks=tracks, dtype='float64',
			n_jobs=(1, 4, 2)[k % 3])
		assert_identical(y, _plan_expected(expected, chroms, starts, tracks))
