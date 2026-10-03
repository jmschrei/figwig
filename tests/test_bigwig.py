# test_bigwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import re
import sys
import copy
import zlib
import pickle
import shutil
import struct
import types
import pathlib
import warnings
import threading
import subprocess

import numpy
import pytest
import pybigtools

import figwig
import figwig._kernels

from figwig import BigWig
from figwig import read_bigwig

from .writers import write_bigwig
from .writers import write_raw_bigwig


def reference(path, chroms, starts, width, missing=0.0):
	"""pybigtools 0.2.5's values() for each window, cast to float32."""

	bw = pybigtools.open(path)
	out = numpy.empty((len(starts), width), dtype=numpy.float32)
	scratch = numpy.empty(width, dtype=numpy.float64)
	names = [chroms] * len(starts) if isinstance(chroms, str) else chroms
	for j, (name, start) in enumerate(zip(names, starts)):
		bw.values(name, int(start), int(start) + width, missing=float(missing),
			arr=scratch)
		out[j] = scratch
	return out


def pickle_round_trip(obj):
	return pickle.loads(pickle.dumps(obj))


def assert_identical(a, b):
	"""The same NaN positions, and the same bits everywhere else."""

	assert a.shape == b.shape and a.dtype == b.dtype
	nan_a, nan_b = numpy.isnan(a), numpy.isnan(b)
	numpy.testing.assert_array_equal(nan_a, nan_b)
	numpy.testing.assert_array_equal(a.view(numpy.uint32)[~nan_a],
		b.view(numpy.uint32)[~nan_b])


def random_windows(rng, chroms, n, width):
	"""Windows that start inside their chromosome; some run past its end."""

	names = numpy.array(list(chroms))
	picked = rng.choice(names, size=n)
	lengths = numpy.array([chroms[name] for name in picked])
	return picked, rng.integers(0, lengths).astype(numpy.int64)


###
# Values
###

@pytest.mark.parametrize('fixture', ['dense_bw', 'sparse_bw', 'deep_bw'])
@pytest.mark.parametrize('width', [1, 7, 1000])
def test_read_matches_pybigtools(request, fixture, width, batching):
	path, chroms = request.getfixturevalue(fixture)
	rng = numpy.random.default_rng(width)
	names, starts = random_windows(rng, chroms, 3000, width)

	X = BigWig(path).read(names, starts, width)
	assert_identical(X, reference(path, list(names), starts, width))


def test_read_matches_pybigtools_at_chromosome_edges(dense_bw, batching):
	path, chroms = dense_bw
	names, starts = [], []
	for name, length in chroms.items():
		for start in [0, 1, length - 1000, length - 999, length - 1]:
			names.append(name)
			starts.append(max(0, start))

	X = BigWig(path).read(names, starts, 1000)
	assert_identical(X, reference(path, names, starts, 1000))


def test_past_chromosome_end_is_nan(dense_bw):
	path, chroms = dense_bw
	length = chroms['chrM']
	X = BigWig(path).read('chrM', [length - 10, length + 3], 20)

	assert not numpy.isnan(X[0, :10]).any()
	assert numpy.isnan(X[0, 10:]).all()
	assert numpy.isnan(X[1]).all()


def test_uncovered_bases_are_zero(tmp_path):
	path = tmp_path / 'gaps.bw'
	write_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, 2.5),
		('chr1', 500, 501, -1.0)])

	X = BigWig(path).read('chr1', [95, 495], 20)
	numpy.testing.assert_array_equal(X[0], [0] * 5 + [2.5] * 10 + [0] * 5)
	numpy.testing.assert_array_equal(X[1], [0] * 5 + [-1.0] + [0] * 14)


def test_special_values(tmp_path):
	path = tmp_path / 'special.bw'
	values = [numpy.nan, numpy.inf, -numpy.inf, -0.0, 1e-45, 3.4e38, -3.4e38]
	entries = [('chr1', 10 * i, 10 * i + 5, float(v)) for i, v in enumerate(values)]
	write_bigwig(path, {'chr1': 1000}, entries)

	starts = numpy.arange(0, 100, 3)
	X = BigWig(path).read('chr1', starts, 25)
	assert_identical(X, reference(str(path), 'chr1', starts, 25))

	# An interval whose value is NaN covers nothing, so its bases are 0.
	numpy.testing.assert_array_equal(BigWig(path).read('chr1', [0], 5)[0], 0)
	assert numpy.signbit(BigWig(path).read('chr1', [30], 1)[0, 0])


@pytest.mark.parametrize('kind', [1, 2, 3])
@pytest.mark.parametrize('compress', [True, False])
def test_section_types(tmp_path, kind, compress, batching):
	rng = numpy.random.default_rng(kind)
	chroms = {'chr1': 50_000, 'chr2': 20_000}
	sections = []
	for name, length in chroms.items():
		for s in range(1000, length - 2000, 3000):
			if kind == 1:
				items, position = [], s
				for _ in range(40):
					end = position + int(rng.integers(1, 30))
					items.append((position, end, float(rng.normal())))
					position = end + int(rng.integers(0, 20))
				sections.append((name, 1, 0, 0, items))
			elif kind == 2:
				span = int(rng.integers(1, 9))
				gaps = rng.integers(0, 5, size=40)
				positions = s + numpy.cumsum(span + gaps) - span - gaps[0]
				items = [(int(p), float(rng.normal())) for p in positions]
				sections.append((name, 2, 0, span, items))
			else:
				span = int(rng.integers(1, 6))
				step = span + int(rng.integers(0, 4))
				sections.append((name, 3, step, span, (s, [float(v) for v in
					rng.normal(size=60)])))

	path = str(tmp_path / 'sections.bw')
	write_raw_bigwig(path, chroms, sections, compress=compress)

	names, starts = random_windows(rng, chroms, 2000, 300)
	X = BigWig(path).read(names, starts, 300)
	assert_identical(X, reference(path, list(names), starts, 300))
	assert (X[~numpy.isnan(X)] != 0).any()


###
# Arguments
###

def test_n_jobs_does_not_change_output(sparse_bw, batching):
	path, chroms = sparse_bw
	names, starts = random_windows(numpy.random.default_rng(3), chroms, 5000, 500)
	bw = BigWig(path)

	X = bw.read(names, starts, 500, n_jobs=1)
	for n_jobs in [2, 3, 8, 16, -1]:
		assert_identical(bw.read(names, starts, 500, n_jobs=n_jobs), X)


def test_n_jobs_every_cpu(dense_bw, monkeypatch):
	"""n_jobs=-1 is every CPU the process may use."""

	sizes = []
	pool = figwig.bigwig.ThreadPoolExecutor

	def recorder(max_workers):
		sizes.append(max_workers)
		return pool(max_workers)

	monkeypatch.setattr(figwig.bigwig, 'ThreadPoolExecutor', recorder)
	monkeypatch.setattr(figwig.bigwig, '_BATCH_BLOCKS', 1)
	monkeypatch.setattr(figwig.bigwig, '_cpu_count', lambda: 3)
	BigWig(dense_bw[0]).read('chr1', numpy.arange(0, 240_000, 1000), 100,
		n_jobs=-1)
	assert sizes == [3]


def test_cpu_count(monkeypatch):
	"""The CPUs in the process's affinity mask where the platform has one,
	as Linux does, and every CPU where it does not, as on macOS and
	Windows."""

	if hasattr(os, 'sched_getaffinity'):
		assert figwig.bigwig._cpu_count() == len(os.sched_getaffinity(0))
		monkeypatch.delattr(os, 'sched_getaffinity')

	assert figwig.bigwig._cpu_count() == os.cpu_count()


def test_order_and_repeats(dense_bw, batching):
	path, chroms = dense_bw
	rng = numpy.random.default_rng(4)
	names, starts = random_windows(rng, chroms, 1000, 200)
	names = numpy.concatenate([names, names[:300]])
	starts = numpy.concatenate([starts, starts[:300] + 1])

	bw = BigWig(path)
	X = bw.read(names, starts, 200)
	order = rng.permutation(len(starts))
	assert_identical(bw.read(names[order], starts[order], 200), X[order])


def test_chroms_forms(dense_bw):
	path, _ = dense_bw
	starts = numpy.array([5, 1000, 70_000])
	bw = BigWig(path)

	X = bw.read('chr2', starts, 50)
	assert_identical(bw.read(['chr2'] * 3, starts, 50), X)
	assert_identical(bw.read(numpy.array(['chr2'] * 3), starts, 50), X)
	assert_identical(bw.read(numpy.array(['chr2'] * 3, dtype=object), starts,
		50), X)
	assert_identical(bw.read('chr2', starts.tolist(), 50), X)
	assert_identical(bw.read('chr2', starts.astype(numpy.uint32), 50), X)


def test_out(dense_bw):
	path, _ = dense_bw
	bw = BigWig(path)
	out = numpy.full((2, 30), 7, dtype=numpy.float32)

	result = bw.read('chr1', [0, 100], 30, out=out)
	assert result is out
	assert_identical(out, bw.read('chr1', [0, 100], 30))

	# A read-only array raised numba's TypingError, and a list AttributeError.
	read_only = numpy.empty((2, 30), dtype=numpy.float32)
	read_only.flags.writeable = False
	for bad in [numpy.empty((3, 30), dtype=numpy.float32),
			numpy.empty((2, 30), dtype=numpy.float64),
			numpy.empty((30, 2), dtype=numpy.float32).T, read_only]:
		with pytest.raises(ValueError, match='out must be a writeable'):
			bw.read('chr1', [0, 100], 30, out=bad)

	with pytest.raises(TypeError, match='out must be a numpy array, not list'):
		bw.read('chr1', [0, 100], 30, out=[[0.0] * 30] * 2)


@pytest.mark.parametrize('chroms', [[], 'chr1'])
@pytest.mark.parametrize('starts', [numpy.empty(0, dtype=numpy.int64), [], ()],
	ids=['array', 'list', 'tuple'])
def test_no_windows(dense_bw, chroms, starts):
	"""An empty list of starts is float64 to numpy, and was refused as not
	being integers."""

	X = BigWig(dense_bw[0]).read(chroms, starts, 10)
	assert X.shape == (0, 10) and X.dtype == numpy.float32


@pytest.mark.parametrize('width', [0, -3])
def test_width_must_be_positive(dense_bw, width):
	with pytest.raises(ValueError, match='width must be at least 1'):
		BigWig(dense_bw[0]).read('chr1', [0], width)


@pytest.mark.parametrize('width', [1.5, True, '10'])
def test_width_must_be_an_integer(dense_bw, width):
	with pytest.raises(TypeError, match='width must be an integer'):
		BigWig(dense_bw[0]).read('chr1', [0], width)


@pytest.mark.parametrize('n_jobs', [0, -2])
def test_n_jobs_must_be_positive(dense_bw, n_jobs):
	with pytest.raises(ValueError, match=r'n_jobs must be at least 1, or -1 '
			r'for every CPU\.'):
		BigWig(dense_bw[0]).read('chr1', [0], 10, n_jobs=n_jobs)


@pytest.mark.parametrize('n_jobs', [2.0, True, None])
def test_n_jobs_must_be_an_integer(dense_bw, n_jobs):
	with pytest.raises(TypeError, match='n_jobs must be an integer'):
		BigWig(dense_bw[0]).read('chr1', [0], 10, n_jobs=n_jobs)


def test_starts_must_be_integers(dense_bw):
	with pytest.raises(TypeError, match='starts must be integers'):
		BigWig(dense_bw[0]).read('chr1', [0.0, 10.5], 10)


def test_starts_must_be_one_dimensional(dense_bw):
	with pytest.raises(ValueError, match='one-dimensional'):
		BigWig(dense_bw[0]).read('chr1', [[0, 10]], 10)


def test_one_chrom_per_start(dense_bw):
	with pytest.raises(ValueError, match='one name per start'):
		BigWig(dense_bw[0]).read(['chr1', 'chr2'], [0, 10, 20], 10)


@pytest.mark.parametrize('missing', [0.0, -1.0, numpy.nan, 3,
	numpy.float32(2.5)], ids=['zero', 'negative', 'nan', 'int', 'float32'])
@pytest.mark.parametrize('fixture', ['dense_bw', 'sparse_bw'])
def test_missing(request, fixture, missing):
	path, chroms = request.getfixturevalue(fixture)
	names, starts = random_windows(numpy.random.default_rng(13), chroms, 1000,
		200)
	X = BigWig(path).read(names, starts, 200, missing=missing)
	assert_identical(X, reference(path, list(names), starts, 200, missing))


@pytest.mark.parametrize('missing', [0.0, -1.0, numpy.nan])
def test_missing_nan_interval(tmp_path, missing):
	"""An interval whose value is NaN covers nothing, so its bases are
	`missing`."""

	path = str(tmp_path / 'nan.bw')
	write_raw_bigwig(path, {'chr1': 1000}, [_bedgraph('chr1', 100, [
		(0, 5, numpy.nan), (5, 10, 2.0)])])

	X = BigWig(path).read('chr1', [98], 14, missing=missing)
	expected = [missing] * 7 + [2.0] * 5 + [missing] * 2
	assert_identical(X, numpy.array([expected], dtype=numpy.float32))
	assert_identical(X, reference(path, 'chr1', [98], 14, missing))


@pytest.mark.parametrize('missing', [0.0, numpy.nan])
def test_absent_chromosome_is_missing(dense_bw, missing):
	"""A writer leaves out a chromosome without data. Windows on one raised;
	they are now `missing` throughout, even far past where a chromosome
	would end, and a warning names the chromosome."""

	path, _ = dense_bw
	bw = BigWig(path)
	with pytest.warns(UserWarning, match=r'^3 windows are on chromosomes not in '
			r".*, and are (0\.0|nan) throughout: 'chr7', 'chrZ'\. Its chromosomes "
			r"include 'chr1', 'chr2', 'chrM'\.$") as record:
		X = bw.read(['chr1', 'chrZ', 'chr7', 'chrZ', 'chr2'], [0, 0, 5, 2**31,
			100], 10, missing=missing)

	assert len(record) == 1 and record[0].filename == __file__
	assert_identical(X[[0, 4]], bw.read(['chr1', 'chr2'], [0, 100], 10,
		missing=missing))
	assert_identical(X[1:4], numpy.full((3, 10), missing, dtype=numpy.float32))


def test_absent_chromosome_names(dense_bw):
	"""Another naming scheme (1 against chr1) leaves every window missing; the
	warning shows the file's own names, and lists at most ten absent."""

	names = [str(i) for i in range(1, 13)]
	with pytest.warns(UserWarning, match=r"^12 windows .*: '1', '10', '11', "
			r"'12', '2', '3', '4', '5', '6', '7', and 2 more\. Its chromosomes "
			r"include 'chr1', 'chr2'"):
		X = BigWig(dense_bw[0]).read(names, [0] * 12, 5)

	assert (X == 0).all()


def test_absent_chromosome_names_are_quoted(dense_bw):
	"""An empty or padded name, as from slicing a string or reading a BED
	file with stray spaces, showed as nothing or as the name it resembles."""

	with pytest.warns(UserWarning, match=r"throughout: '', 'chr1 '\."):
		BigWig(dense_bw[0]).read(['', 'chr1 '], [0, 0], 5)


def test_range_is_checked_before_absent_chromosomes(dense_bw):
	with warnings.catch_warnings():
		warnings.simplefilter('error')
		with pytest.raises(ValueError, match='start before 0'):
			BigWig(dense_bw[0]).read(['chrZ', 'chr1'], [0, -1], 10)


@pytest.mark.parametrize('missing', ['0', None, True, [0.0]])
def test_missing_must_be_a_number(dense_bw, missing):
	with pytest.raises(TypeError, match='missing must be a number'):
		BigWig(dense_bw[0]).read('chr1', [0], 10, missing=missing)


@pytest.mark.parametrize('start, dtype', [
	(-1, numpy.int64),
	(2**32 - 10, numpy.int64),
	(2**63 - 5, numpy.int64),
	(2**63 + 5, numpy.uint64),
])
def test_out_of_range_window_raises(dense_bw, start, dtype):
	"""A start within `width` of 2**63 overflowed when the window's end was
	computed, passed the check and read garbage, and a uint64 start of 2**63
	or more was reported as negative."""

	starts = numpy.array([0, start], dtype=dtype)
	with pytest.raises(ValueError, match=r'1 windows start before 0 or end '
			r'past 2\*\*32 - 1, such as chr1:{}-{}\.'.format(start, start + 10)):
		BigWig(dense_bw[0]).read('chr1', starts, 10)


def test_window_ending_at_the_largest_position(dense_bw):
	X = BigWig(dense_bw[0]).read('chr1', [2**32 - 11], 10)
	assert numpy.isnan(X).all()


def test_read_bigwig(sparse_bw):
	path, chroms = sparse_bw
	names, starts = random_windows(numpy.random.default_rng(5), chroms, 2000, 100)
	X = BigWig(path).read(names, starts, 100)
	assert_identical(read_bigwig(path, names, starts, 100, n_jobs=3), X)
	assert_identical(read_bigwig(pathlib.Path(path), names, starts, 100), X)
	assert_identical(read_bigwig(BigWig(path), names, starts, 100), X)


@pytest.mark.parametrize('container', [list, tuple])
def test_read_bigwig_several_files(dense_bw, sparse_bw, deep_bw, container,
	batching):
	"""Channel i of a read of several files holds what reading bigwigs[i] on
	its own gives, with paths and BigWig objects mixed and a file given
	twice."""

	files = [dense_bw[0], BigWig(sparse_bw[0]), deep_bw[0], dense_bw[0]]
	rng = numpy.random.default_rng(14)
	names = numpy.array(['chr1'] * 400 + ['chr2'] * 100)
	starts = rng.integers(0, 110_000, size=500)

	# deep_bw has no chr2, so its channel warns for those windows alone.
	with pytest.warns(UserWarning, match='100 windows are on chromosomes not in '
			'.*deep.bw') as record:
		X = read_bigwig(container(files), names, starts, 300, n_jobs=4,
			missing=numpy.nan)

	assert len(record) == 1
	assert X.shape == (500, 4, 300) and X.dtype == numpy.float32
	assert X.flags['C_CONTIGUOUS']

	with pytest.warns(UserWarning, match='deep.bw'):
		for i, bigwig in enumerate(files):
			assert_identical(X[:, i], read_bigwig(bigwig, names, starts, 300,
				missing=numpy.nan))

	assert numpy.isnan(X[400:, 2]).all() and not numpy.isnan(X[:400, 2]).all()

	# A list of one file gives one channel.
	assert_identical(read_bigwig([files[0]], names, starts, 300),
		read_bigwig(files[0], names, starts, 300)[:, None])


def test_read_bigwig_several_files_share_a_pool(dense_bw, sparse_bw,
	monkeypatch):
	"""Each file below is one batch, so read one at a time no read could use
	more than one thread; read together, their batches share one pool."""

	sizes = []
	pool = figwig.bigwig.ThreadPoolExecutor

	def recorder(max_workers):
		sizes.append(max_workers)
		return pool(max_workers)

	monkeypatch.setattr(figwig.bigwig, 'ThreadPoolExecutor', recorder)
	read_bigwig([dense_bw[0], sparse_bw[0], dense_bw[0]], 'chr1', [0, 5000],
		100, n_jobs=8)
	assert sizes == [3]


def test_kernels_compile_on_the_calling_thread(tmp_path, monkeypatch):
	"""Each kernel is called first on the calling thread when it has not been
	compiled, the inflater for the first compressed file even when an
	uncompressed file comes before it. The inflater is called only where
	zlib's library can be loaded; elsewhere, as on Windows, blocks are
	inflated by zlib.decompress."""

	sections = [_bedgraph('chr1', s, [(0, 5, 1.0)]) for s in range(0, 9000,
		100)]
	for compress in [False, True]:
		write_raw_bigwig(tmp_path / '{}.bw'.format(compress), {'chr1': 10_000},
			sections, compress=compress)

	calls = {}

	class Uncompiled:
		signatures = []

		def __init__(self, name, function):
			self.name, self.function = name, function

		def __call__(self, *args):
			calls.setdefault(self.name, threading.current_thread())
			return self.function(*args)

	for name in ['_read_windows', '_inflate_blocks']:
		monkeypatch.setattr(figwig.bigwig, name, Uncompiled(name,
			getattr(figwig.bigwig, name)))

	monkeypatch.setattr(figwig.bigwig, '_BATCH_BLOCKS', 2)
	files = [tmp_path / 'False.bw', tmp_path / 'True.bw']
	X = read_bigwig(files, 'chr1', numpy.arange(0, 9000, 50), 10, n_jobs=4)
	assert calls['_read_windows'] is threading.main_thread()
	if figwig._kernels._zlib_uncompress() is not None:
		assert calls['_inflate_blocks'] is threading.main_thread()

	assert_identical(X[:, 0], X[:, 1])


def test_read_bigwig_out(dense_bw, sparse_bw):
	files = [dense_bw[0], sparse_bw[0]]
	out = numpy.full((2, 2, 30), 7, dtype=numpy.float32)
	assert read_bigwig(files, 'chr1', [0, 100], 30, out=out) is out
	assert_identical(out, read_bigwig(files, 'chr1', [0, 100], 30))

	with pytest.raises(ValueError, match=r'of shape \(2, 2, 30\)'):
		read_bigwig(files, 'chr1', [0, 100], 30, out=numpy.empty((2, 30),
			dtype=numpy.float32))
	with pytest.raises(ValueError, match=r'of shape \(2, 30\)'):
		read_bigwig(files[0], 'chr1', [0, 100], 30, out=out)


@pytest.mark.parametrize('bigwigs, error, match', [
	([], ValueError, 'at least one bigWig'),
	(5, TypeError, 'not int'),
	({'a': 1}, TypeError, 'not dict'),
	(['x.bw', None], TypeError, 'not NoneType'),
])
def test_read_bigwig_bad_bigwigs(bigwigs, error, match):
	with pytest.raises(error, match=match):
		read_bigwig(bigwigs, 'chr1', [0], 10)


def test_read_bigwig_warning_names_the_caller(dense_bw):
	with pytest.warns(UserWarning, match='chromosomes not in') as record:
		read_bigwig([dense_bw[0], dense_bw[0]], 'chrZ', [0], 10)

	assert len(record) == 2
	assert all(warning.filename == __file__ for warning in record)


def test_read_bigwig_failure_names_the_file(tmp_path, dense_bw):
	path = str(tmp_path / 'corrupt.bw')
	write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, b'not zlib')])

	with pytest.raises(ValueError, match='figwig cannot read in .*corrupt.bw: '):
		read_bigwig([dense_bw[0], path], 'chr1', [95], 20)


def test_file_removed_after_opening(tmp_path, dense_bw):
	"""A file removed after its index was read cannot be opened for the read,
	and the files opened before it are closed again."""

	shutil.copy(dense_bw[0], tmp_path / 'copy.bw')
	removed = BigWig(tmp_path / 'copy.bw')
	removed.read('chr1', [0], 10)
	os.remove(tmp_path / 'copy.bw')

	opened = len(os.listdir('/proc/self/fd')) if os.path.exists(
		'/proc/self/fd') else None
	with pytest.raises(FileNotFoundError):
		read_bigwig([dense_bw[0], removed], 'chr1', [0], 10)

	if opened is not None:
		assert len(os.listdir('/proc/self/fd')) == opened


def test_pathlike_and_attributes(dense_bw):
	path, chroms = dense_bw
	bw = BigWig(pathlib.Path(path))
	assert bw.path == path
	assert bw.chrom_sizes == chroms
	assert list(bw.chrom_sizes) == list(pybigtools.open(path).chroms())
	assert repr(bw) == "BigWig('{}', 3 chromosomes)".format(path)


def test_multi_level_chromosome_tree(tmp_path):
	"""A chromosome tree with a root over three leaves, as UCSC writes one for
	a genome with more than 256 chromosomes. Its leaves were read in reverse,
	so `chroms` came out in the wrong order.
	"""

	chroms = {'chr{:02d}'.format(i): 1000 + i for i in range(40)}
	sections = [_bedgraph(name, 100, [(0, 10, float(i))]) for i, name in
		enumerate(chroms)]
	path = str(tmp_path / 'tree.bw')
	write_raw_bigwig(path, chroms, sections, chrom_block_size=16)

	bw = BigWig(path)
	assert list(bw.chrom_sizes.items()) == list(chroms.items())
	assert list(bw.chrom_sizes) == list(pybigtools.open(path).chroms())

	names, starts = list(chroms), [95] * len(chroms)
	assert_identical(bw.read(names, starts, 20), reference(path, names, starts,
		20))


###
# Threads, processes and the zlib fallback
###

@pytest.mark.parametrize('duplicate', [pickle_round_trip, copy.deepcopy],
	ids=['pickle', 'deepcopy'])
def test_pickle(dense_bw, duplicate):
	"""A PyTorch DataLoader pickles its dataset into each worker under the
	spawn and forkserver start methods, and BigWig held a lock that cannot
	be pickled. Copies are taken before and after the index is read."""

	path, chroms = dense_bw
	names, starts = random_windows(numpy.random.default_rng(10), chroms, 500,
		100)
	bw = BigWig(path)
	before = duplicate(bw)
	X = bw.read(names, starts, 100)
	after = duplicate(bw)
	assert before._index is None and after._index is not None

	for other in [before, after]:
		assert other.path == bw.path and other.chrom_sizes == bw.chrom_sizes
		assert_identical(other.read(names, starts, 100, n_jobs=3), X)
		assert other._index_lock is not bw._index_lock


@pytest.mark.parametrize('function', ['preadv', 'pread', 'read'])
def test_short_reads(deep_bw, monkeypatch, function):
	"""POSIX lets a read return fewer bytes than asked for, and Linux returns
	at most about 2 GiB from one. A short read marked the blocks it did not
	reach as unreadable, and the windows over them raised. Here every read
	returns at most 1,000 bytes, through preadv, through the pread used
	where there is no preadv, or through the seek and read used where there
	is neither, as on Windows. That last one shares a file position between
	threads, so the read runs in small batches on several threads."""

	if function != 'read' and not hasattr(os, function):
		pytest.skip('os.{} does not exist here.'.format(function))

	path, chroms = deep_bw
	names, starts = random_windows(numpy.random.default_rng(12), chroms, 300,
		20_000)
	X = BigWig(path).read(names, starts, 20_000)

	if function == 'preadv':
		preadv = os.preadv
		monkeypatch.setattr(os, 'preadv', lambda fd, buffers, offset: preadv(
			fd, [buffers[0][:1000]], offset))
	elif function == 'pread':
		pread = os.pread
		monkeypatch.delattr(os, 'preadv', raising=False)
		monkeypatch.setattr(os, 'pread', lambda fd, n, offset: pread(fd,
			min(n, 1000), offset))
	else:
		read = os.read
		monkeypatch.delattr(os, 'preadv', raising=False)
		monkeypatch.delattr(os, 'pread', raising=False)
		monkeypatch.setattr(os, 'read', lambda fd, n: read(fd, min(n, 1000)))
		monkeypatch.setattr(figwig.bigwig, '_BATCH_BLOCKS', 2)

	assert_identical(BigWig(path).read(names, starts, 20_000, n_jobs=6), X)


@pytest.mark.parametrize('failing, expected', [
	(set(), 'libz.so.1'),
	({'libz.so.1': AttributeError}, 'libz.1.dylib'),
	({'libz.so.1', 'libz.1.dylib', 'libz.dylib'}, 'zlib1.dll'),
	({'libz.so.1', 'libz.1.dylib', 'libz.dylib', 'zlib1.dll', 'zlib.dll'},
		'found:z'),
	({'libz.so.1', 'libz.1.dylib', 'libz.dylib', 'zlib1.dll', 'zlib.dll',
		'found:z'}, 'found:zlib'),
	({'libz.so.1', 'libz.1.dylib', 'libz.dylib', 'zlib1.dll', 'zlib.dll',
		'found:z', 'found:zlib'}, None),
], ids=['linux', 'macos', 'windows', 'find_z', 'find_zlib', 'none'])
def test_zlib_loader(monkeypatch, failing, expected):
	"""zlib's library is looked for under its Linux, macOS and Windows names,
	then through ctypes.util.find_library. A library that cannot be loaded,
	or that has no uncompress (AttributeError), is passed over, and without
	one the loader gives None, so that blocks are inflated by
	zlib.decompress."""

	import ctypes
	import ctypes.util

	class Library:
		def __init__(self, name):
			if name in failing:
				raise (failing[name] if isinstance(failing, dict) else OSError)(
					name)

			self.uncompress = types.SimpleNamespace(name=name)

	monkeypatch.setattr(ctypes, 'CDLL', Library)
	monkeypatch.setattr(ctypes.util, 'find_library', lambda name: 'found:' +
		name)
	loaded = figwig._kernels._load_zlib_uncompress()

	if expected is None:
		assert loaded is None
	else:
		function, ulong = loaded
		assert function.name == expected
		assert function.restype is ctypes.c_int
		assert ulong == numpy.dtype(ctypes.c_ulong)


def test_zlib_loader_without_find_library(monkeypatch):
	import ctypes
	import ctypes.util

	def library(name):
		raise OSError(name)

	monkeypatch.setattr(ctypes, 'CDLL', library)
	monkeypatch.setattr(ctypes.util, 'find_library', lambda name: None)
	assert figwig._kernels._load_zlib_uncompress() is None


def test_numba_disable_jit(dense_bw, tmp_path):
	"""With NUMBA_DISABLE_JIT=1 the kernels are plain Python functions,
	without the `signatures` attribute that read checked, and every read
	raised AttributeError. NUMBA_DISABLE_JIT is read when numba is first
	imported, so this runs in a new process."""

	path, chroms = dense_bw
	names, starts = random_windows(numpy.random.default_rng(11), chroms, 20, 300)
	numpy.savez(tmp_path / 'windows.npz', names=names, starts=starts)

	code = ("import numpy, figwig; w = numpy.load({!r}); numpy.save({!r}, "
		"figwig.BigWig({!r}).read(w['names'], w['starts'], 300, n_jobs=2))")
	code = code.format(str(tmp_path / 'windows.npz'), str(tmp_path / 'X.npy'),
		path)
	env = dict(os.environ, NUMBA_DISABLE_JIT='1')
	result = subprocess.run([sys.executable, '-c', code], env=env,
		capture_output=True, text=True)
	assert result.returncode == 0, result.stderr

	assert_identical(numpy.load(tmp_path / 'X.npy'), BigWig(path).read(names,
		starts, 300))


def test_concurrent_reads_on_one_object(deep_bw, batching):
	path, chroms = deep_bw
	bw = BigWig(path)
	rng = numpy.random.default_rng(6)
	jobs = [random_windows(rng, chroms, 3000, 1000) for _ in range(6)]
	expected = [bw.read(names, starts, 1000, n_jobs=1) for names, starts in jobs]

	results = [None] * len(jobs)

	def work(i):
		results[i] = bw.read(*jobs[i], 1000, n_jobs=4)

	threads = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	for result, X in zip(results, expected):
		assert_identical(result, X)


def test_zlib_decompress_fallback(sparse_bw, monkeypatch):
	path, chroms = sparse_bw
	names, starts = random_windows(numpy.random.default_rng(7), chroms, 3000, 400)
	X = BigWig(path).read(names, starts, 400)

	monkeypatch.setattr(figwig._kernels, '_ZLIB_UNCOMPRESS', [None])
	assert figwig._kernels._zlib_uncompress() is None
	assert_identical(BigWig(path).read(names, starts, 400, n_jobs=4), X)


###
# What figwig refuses to read
###

def _bedgraph(name, start, items):
	return (name, 1, 0, 0, [(start + s, start + e, v) for s, e, v in items])


@pytest.mark.parametrize('items, reason', [
	([(0, 10, 1.0), (5, 15, 2.0)], 'overlap'),
	([(20, 30, 1.0), (0, 10, 2.0)], 'unsorted'),
])
def test_bad_intervals_raise(tmp_path, items, reason):
	path = str(tmp_path / '{}.bw'.format(reason))
	write_raw_bigwig(path, {'chr1': 10_000}, [_bedgraph('chr1', 100, [(0, 5, 1.0)]),
		_bedgraph('chr1', 5000, items)])

	bw = BigWig(path)
	numpy.testing.assert_array_equal(bw.read('chr1', [100], 5)[0], 1.0)
	with pytest.raises(ValueError, match='1 windows overlap data blocks that '
			'figwig cannot read.*chr1:4990-5010'):
		bw.read('chr1', [100, 4990], 20)


def test_unknown_section_type_raises(tmp_path):
	path = str(tmp_path / 'kind.bw')
	block = zlib.compress(struct.pack('<IIIIIBBH', 0, 100, 110, 0, 0, 4, 0, 1) +
		struct.pack('<IIf', 100, 110, 1.0))
	write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, block)])

	with pytest.raises(ValueError, match='figwig cannot read'):
		BigWig(path).read('chr1', [95], 20)


def test_corrupt_block_raises(tmp_path):
	path = str(tmp_path / 'corrupt.bw')
	write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, b'not zlib')])

	with pytest.raises(ValueError, match='figwig cannot read'):
		BigWig(path).read('chr1', [95], 20)


@pytest.mark.parametrize('raw', [b'', b'abcde'], ids=['empty', 'partial_word'])
def test_block_not_whole_words_raises(tmp_path, raw, batching):
	"""A block must inflate to a whole number of 32-bit words, through
	uncompress() or, with the tiny batches' 64-byte buffer, through
	zlib.decompress."""

	path = str(tmp_path / 'words.bw')
	write_raw_bigwig(path, {'chr1': 1000}, [_bedgraph('chr1', 10, [(0, 5, 1.0)]),
		('chr1', 100, 110, zlib.compress(raw))])

	bw = BigWig(path)
	numpy.testing.assert_array_equal(bw.read('chr1', [10], 5)[0], 1.0)
	with pytest.raises(ValueError, match='figwig cannot read.*chr1:95-115'):
		bw.read('chr1', [10, 95], 20)


def test_file_truncated_after_its_index_was_read(tmp_path):
	"""A file cut short after the index was read, as when it is rewritten
	under a reader, leaves blocks that cannot be read in full."""

	path, _, data_index = _two_chrom_bigwig(tmp_path)
	bw = BigWig(path)
	bw.read('chr1', [95], 10)

	with open(path, 'r+b') as handle:
		handle.truncate(data_index - 4)

	numpy.testing.assert_array_equal(bw.read('chr1', [95], 10)[0, 5:], 1.0)
	with pytest.raises(ValueError, match='figwig cannot read.*chr2:95-105'):
		bw.read('chr2', [95], 10)


def test_corrupt_files_raise_only_value_errors(tmp_path):
	"""Bytes of three small bigWigs overwritten at random, a few at a time,
	three times in four inside the header, the chromosome tree or the data
	index. bigWig has no checksums, so a corrupt value can read without
	error, but nothing other than a ValueError may escape.
	"""

	chroms = {'chr1': 20_000, 'chr2': 9_000}
	sections = []
	for name, length in chroms.items():
		for s in range(100, length - 2000, 1500):
			sections.append(_bedgraph(name, s, [(10 * i, 10 * i + 5, float(i))
				for i in range(50)]))
			sections.append((name, 2, 0, 3, [(s + 600 + 7 * i, float(i)) for i
				in range(40)]))
			sections.append((name, 3, 4, 2, (s + 1000, [float(i) for i in
				range(60)])))

	files = []
	for compress in [True, False]:
		path = tmp_path / 'raw_{}.bw'.format(compress)
		write_raw_bigwig(path, chroms, sections, compress=compress)
		files.append(path.read_bytes())

	path = tmp_path / 'pybigtools.bw'
	write_bigwig(path, chroms, [(name, s, s + 3, float(s)) for name in chroms
		for s in range(0, chroms[name] - 5, 7)])
	files.append(path.read_bytes())

	rng = numpy.random.default_rng(9)
	names = ['chr1'] * 40 + ['chr2'] * 20
	starts = list(range(0, 20_000, 500)) + list(range(0, 9_000, 450))
	path = tmp_path / 'corrupt.bw'
	outcomes = {'read': 0, 'ValueError': 0}
	for trial in range(600):
		data = bytearray(files[trial % 3])
		chrom_tree, _, data_index = struct.unpack('<QQQ', data[8:32])
		region = int(rng.choice([0, chrom_tree, data_index, -1]))
		for _ in range(int(rng.integers(1, 4))):
			if region == -1:
				position = int(rng.integers(0, len(data) - 8))
			else:
				position = min(region + int(rng.integers(0, 80)), len(data) - 8)

			if rng.random() < 0.5:
				data[position] = int(rng.integers(256))
			else:
				value = int(rng.choice([0, 1, 2 ** 31, 2 ** 32 - 1, 2 ** 63]))
				data[position:position + 8] = value.to_bytes(8, 'little')

		path.write_bytes(bytes(data))
		try:
			bw = BigWig(path)
			present = [name in bw.chrom_sizes for name in names]
			if any(present):
				bw.read([n for n, p in zip(names, present) if p], [s for s, p in
					zip(starts, present) if p], 500, n_jobs=1)
			outcomes['read'] += 1
		except ValueError:
			outcomes['ValueError'] += 1

	assert outcomes['read'] > 50 and outcomes['ValueError'] > 50


def test_unsorted_index_raises(tmp_path):
	path = str(tmp_path / 'index.bw')
	write_raw_bigwig(path, {'chr1': 10_000}, [_bedgraph('chr1', 5000, [(0, 5, 1.0)]),
		_bedgraph('chr1', 100, [(0, 5, 2.0)])])

	with pytest.raises(ValueError, match='its data blocks are unsorted'):
		BigWig(path).read('chr1', [0], 10)


def _fixedstep_block(chrom_id, start, values, end):
	"""A compressed block of one fixedStep section, with step and span 1,
	whose section header says it ends at `end`."""

	header = struct.pack('<IIIIIBBH', chrom_id, start, end, 1, 1, 3, 0,
		len(values))
	return zlib.compress(header + struct.pack('<{}f'.format(len(values)),
		*values))


def test_overlapping_index_entries(tmp_path):
	"""pyBigWig writes some fixedStep blocks with an index entry that ends 6
	bases past the block's last interval, so neighbouring entries can overlap
	even though no two intervals do. Such files were refused as having
	overlapping data blocks.
	"""

	rng = numpy.random.default_rng(8)
	sections, start = [], 100
	for _ in range(20):
		n = int(rng.integers(20, 60))
		values = rng.normal(size=n).astype(numpy.float32).tolist()
		sections.append(('chr1', start, start + n + 6, _fixedstep_block(0,
			start, values, start + n + 6)))
		start += n + int(rng.integers(1, 5))

	path = str(tmp_path / 'overlap.bw')
	write_raw_bigwig(path, {'chr1': 5000}, sections)

	starts = numpy.arange(0, start + 10, 7)
	X = BigWig(path).read('chr1', starts, 50)
	assert_identical(X, reference(path, 'chr1', starts, 50))
	assert (X != 0).mean() > 0.5


def test_intervals_overlapping_across_blocks_raise(tmp_path):
	"""Intervals in two blocks that overlap give a base two values, as two
	overlapping intervals in one block do, so a window over both raises."""

	path = str(tmp_path / 'across.bw')
	write_raw_bigwig(path, {'chr1': 10_000}, [
		_bedgraph('chr1', 100, [(0, 10, 1.0)]),
		_bedgraph('chr1', 1000, [(0, 20, 2.0)]),
		_bedgraph('chr1', 1015, [(0, 10, 3.0)]),
		_bedgraph('chr1', 3000, [(0, 10, 4.0)])])

	bw = BigWig(path)
	X = bw.read('chr1', [95, 1005, 2995], 10)
	numpy.testing.assert_array_equal(X[:, 5:], [[1.0] * 5, [2.0] * 5, [4.0] * 5])

	with pytest.raises(ValueError, match='1 windows overlap data blocks that '
			'figwig cannot read.*chr1:1010-1020'):
		bw.read('chr1', [95, 1010, 2995], 10)


@pytest.mark.parametrize('header, match', [
	(b'', 'is not a bigWig file'),
	(b'\0' * 64, 'is not a bigWig file'),
	(struct.pack('<I', 0x8789F2EB) + b'\0' * 60, 'is a bigBed file'),
	(struct.pack('>I', 0x888FFC26) + b'\0' * 60, 'is a big-endian file'),
])
def test_not_a_bigwig_raises(tmp_path, header, match):
	path = tmp_path / 'bad.bw'
	path.write_bytes(header)
	with pytest.raises(ValueError, match=match):
		BigWig(path)


def test_missing_file_raises(tmp_path):
	with pytest.raises(FileNotFoundError):
		BigWig(tmp_path / 'absent.bw')


def test_big_endian_machine_raises(dense_bw, monkeypatch):
	monkeypatch.setattr(figwig.bigwig.sys, 'byteorder', 'big')
	with pytest.raises(ValueError, match='only on little-endian machines'):
		BigWig(dense_bw[0])


def _two_chrom_bigwig(tmp_path):
	"""A raw bigWig with one bedGraph block on each of two chromosomes.

	Returns its path, and the offsets of its chromosome tree and data index.
	The chromosome tree's root, a leaf, starts 32 bytes after the tree, and
	the data index's root, also a leaf, 48 bytes after the index.
	"""

	path = tmp_path / 'two.bw'
	write_raw_bigwig(path, {'chr1': 1000, 'chr2': 1000}, [
		_bedgraph('chr1', 100, [(0, 10, 1.0)]),
		_bedgraph('chr2', 100, [(0, 10, 2.0)])])

	with open(path, 'rb') as handle:
		chrom_tree, _, data_index = struct.unpack('<QQQ', handle.read(64)[8:32])
	return path, chrom_tree, data_index


def _patch(path, offset, fmt, *values):
	with open(path, 'r+b') as handle:
		handle.seek(offset)
		handle.write(struct.pack(fmt, *values))


@pytest.mark.parametrize('field, match', [
	('offset', r'chromosome tree of .* cannot be read\.$'),
	('magic', r'chromosome tree of .* cannot be read\.$'),
	('value_size', r'chromosome tree of .* cannot be read\.$'),
	('cycle', 'chromosome tree of .* cannot be read: it has a cycle'),
])
def test_bad_chromosome_tree_raises(tmp_path, field, match):
	path, chrom_tree, _ = _two_chrom_bigwig(tmp_path)
	if field == 'offset':
		_patch(path, 8, '<Q', path.stat().st_size + 100)
	elif field == 'magic':
		_patch(path, chrom_tree, '<I', 0)
	elif field == 'value_size':
		_patch(path, chrom_tree + 12, '<I', 4)
	else:
		# The root becomes a non-leaf node whose one child is itself. The key
		# size is 4, so the child's offset follows a 4-byte key.
		_patch(path, chrom_tree + 32, '<BBH', 0, 0, 1)
		_patch(path, chrom_tree + 40, '<Q', chrom_tree + 32)

	with pytest.raises(ValueError, match=match):
		BigWig(path)


@pytest.mark.parametrize('field, match', [
	('offset', 'data index of .* cannot be read: unpack'),
	('magic', r'data index of .* cannot be read\.$'),
	('cycle', 'data index of .* cannot be read: it has a cycle'),
	('count', 'data index of .* cannot be read: the file ends inside it'),
	('two_chroms', 'a data block spans two chromosomes'),
])
def test_bad_data_index_raises(tmp_path, field, match):
	path, _, data_index = _two_chrom_bigwig(tmp_path)
	root = data_index + 48
	if field == 'offset':
		_patch(path, 24, '<Q', path.stat().st_size + 100)
	elif field == 'magic':
		_patch(path, data_index, '<I', 0)
	elif field == 'cycle':
		# A non-leaf entry is 16 bytes of range, then its child's offset.
		_patch(path, root, '<BBH', 0, 0, 1)
		_patch(path, root + 4 + 16, '<Q', root)
	elif field == 'count':
		_patch(path, root + 2, '<H', 65535)
	else:
		_patch(path, root + 4 + 8, '<I', 1)

	bw = BigWig(path)
	with pytest.raises(ValueError, match=match):
		bw.read('chr1', [0], 10)


@pytest.mark.parametrize('field, value', [
	('size', 2 ** 40),
	('size', 2 ** 63 + 5),
	('offset', 2 ** 40),
	('offset', 2 ** 64 - 1),
])
def test_data_index_past_end_of_file_raises(tmp_path, field, value):
	"""A block size or offset in the data index that points past the end of
	the file was read as given: a huge size was allocated, raising
	MemoryError, and an offset of 2**63 or more became negative and raised
	OSError."""

	path, _, data_index = _two_chrom_bigwig(tmp_path)
	entry = data_index + 48 + 4
	_patch(path, entry + (24 if field == 'size' else 16), '<Q', value)

	bw = BigWig(path)
	with pytest.raises(ValueError, match='data index of .* cannot be read: a '
			'data block lies past the end of the file'):
		bw.read('chr1', [95], 10)


@pytest.mark.parametrize('field, value', [
	('key_size', 2 ** 31),
	('count', 65535),
])
def test_chromosome_tree_past_end_of_file_raises(tmp_path, field, value):
	"""A chromosome tree node longer than the file was read as given: a huge
	key size was allocated, and a node cut short raised struct.error."""

	path, chrom_tree, _ = _two_chrom_bigwig(tmp_path)
	if field == 'key_size':
		_patch(path, chrom_tree + 8, '<I', value)
	else:
		_patch(path, chrom_tree + 34, '<H', value)

	with pytest.raises(ValueError, match='chromosome tree of .* cannot be '
			'read: the file ends inside it'):
		BigWig(path)


def test_version():
	assert re.fullmatch(r'\d+\.\d+\.\d+', figwig.__version__)
