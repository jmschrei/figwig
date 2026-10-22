"""Optional GPU checks, enabled explicitly on a CUDA test machine."""

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy
import pytest

import figwig
from figwig import cuda
from .test_bigwig import assert_identical, random_windows, _bedgraph
from .writers import write_raw_bigwig

pytestmark = pytest.mark.skipif(os.environ.get('FIGWIG_TEST_CUDA') != '1',
	reason='set FIGWIG_TEST_CUDA=1 on a CUDA machine')


@pytest.mark.parametrize('fixture', ['dense_bw', 'sparse_bw', 'deep_bw'])
@pytest.mark.parametrize('width', [1, 7, 1000])
def test_cuda_matches_cpu(request, fixture, width):
	cupy = pytest.importorskip('cupy')
	path, chroms = request.getfixturevalue(fixture)
	names, starts = random_windows(numpy.random.default_rng(2), chroms, 3000,
		width)
	expected = figwig.read_bigwig(path, names, starts, width)
	reader = cuda.BigWigReader(path)
	for _ in range(2):
		assert_identical(reader.read(names, starts, width).get(), expected)
	out = numpy.empty_like(expected)
	assert reader.read(names, starts, width, out=out) is out
	assert_identical(out, expected)
	with cupy.cuda.Stream(non_blocking=True):
		assert_identical(reader.read(names, starts, width).get(), expected)


@pytest.mark.parametrize('compress', [True, False])
def test_cuda_sections_and_missing(tmp_path, compress):
	path = tmp_path / 'sections.bw'
	write_raw_bigwig(path, {'chr1': 1000}, [
		_bedgraph('chr1', 10, [(0, 5, -0.0), (10, 15, numpy.nan)]),
		('chr1', 2, 0, 3, [(100, 1.0), (110, -2.0)]),
		('chr1', 3, 4, 2, (200, [1.0, -0.0, 2.0]))], compress=compress)
	reader = cuda.BigWigReader(path)
	names, starts = ['chr1'] * 5, [195, 0, 105, 195, 990]
	expected = figwig.read_bigwig(path, names, starts, 30, missing=numpy.nan)
	for _ in range(2):
		assert_identical(reader.read(names, starts, 30, missing=numpy.nan).get(),
			expected)
	with pytest.warns(UserWarning) as caught:
		assert numpy.all(reader.read('absent', [0], 30, missing=2).get() == 2)
	assert caught[0].filename == __file__
	with pytest.warns(UserWarning) as caught:
		cuda.read_bigwig(reader, 'absent', [0], 30)
	assert caught[0].filename == __file__


def test_cuda_multifile_pickle_concurrency(dense_bw):
	path, _ = dense_bw
	reader = pickle.loads(pickle.dumps(cuda.BigWigReader(path)))
	starts = [0, 999, 249990, 0]
	expected = figwig.read_bigwig([path, path], 'chr1', starts, 100)
	assert_identical(cuda.read_bigwig([reader, reader], 'chr1', starts,
		100).get(), expected)
	out = numpy.empty_like(expected)
	assert cuda.read_bigwig([reader, reader], 'chr1', starts, 100, out=out) is out
	assert_identical(out, expected)
	with ThreadPoolExecutor(2) as pool:
		results = list(pool.map(lambda _: reader.read('chr1', starts, 100).get(),
			range(4)))
	for result in results:
		assert_identical(result, expected[:, 0])


def test_cuda_rejects_corruption_before_nvcomp(tmp_path, monkeypatch):
	path = tmp_path / 'bad.bw'
	write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, b'not zlib')])
	def forbidden(*args):
		pytest.fail('invalid bytes reached nvCOMP')
	monkeypatch.setattr(cuda, '_inflate', forbidden)
	with pytest.raises(ValueError, match='figwig cannot read'):
		cuda.BigWigReader(path).read('chr1', [95], 20)


def test_cuda_small_workspace(dense_bw):
	path, _ = dense_bw
	reader = cuda.BigWigReader(path)
	expected = figwig.read_bigwig(path, 'chr1', [0, 100000, 200000], 100)
	for _ in range(2):
		assert_identical(reader.read('chr1', [0, 100000, 200000], 100,
			workspace=1 << 20).get(), expected)
	with pytest.raises(MemoryError, match='workspace'):
		reader.read('chr1', [0], 100, workspace=1)


def test_cuda_file_changes_invalidate_cache(tmp_path, monkeypatch):
	path = tmp_path / 'mutable.bw'
	def write(value):
		write_raw_bigwig(path, {'chr1': 1000}, [
			_bedgraph('chr1', 100, [(0, 10, value)])])
	write(1.0)
	reader = cuda.BigWigReader(path)
	assert numpy.all(reader.read('chr1', [100], 10).get() == 1)
	assert reader._validated.entries
	write(2.0)
	def forbidden(*args):
		pytest.fail('changed bytes reached nvCOMP without CPU validation')
	monkeypatch.setattr(cuda, '_inflate', forbidden)
	assert numpy.all(reader.read('chr1', [100], 10).get() == 2)


def test_cuda_overlap_checks_match_cpu(tmp_path):
	path = tmp_path / 'overlap.bw'
	write_raw_bigwig(path, {'chr1': 10000}, [
		_bedgraph('chr1', 1000, [(0, 20, 2.0)]),
		_bedgraph('chr1', 1015, [(0, 10, 3.0)])])
	reader = cuda.BigWigReader(path)
	for _ in range(2):
		assert numpy.all(reader.read('chr1', [1005], 10).get() == 2)
		with pytest.raises(ValueError, match='figwig cannot read'):
			reader.read('chr1', [1010], 10)


def test_cuda_allocation_retry(dense_bw, monkeypatch):
	path, _ = dense_bw
	reader = cuda.BigWigReader(path)
	starts = [0, 100000, 200000]
	expected = figwig.read_bigwig(path, 'chr1', starts, 100)
	reader.read('chr1', starts, 100)
	inflate, calls = cuda._inflate, []
	def limited(cupy, raws, lengths, budget):
		calls.append(len(raws))
		if len(raws) > 1:
			raise MemoryError('simulated allocation pressure')
		return inflate(cupy, raws, lengths, budget)
	monkeypatch.setattr(cuda, '_inflate', limited)
	assert_identical(reader.read('chr1', starts, 100).get(), expected)
	assert max(calls) > 1 and min(calls) == 1


def test_cuda_without_native_validation(dense_bw, monkeypatch):
	path, _ = dense_bw
	monkeypatch.setattr(cuda, '_uncompress2', lambda: None)
	reader = cuda.BigWigReader(path)
	expected = figwig.read_bigwig(path, 'chr1', [0, 100000], 100)
	for _ in range(2):
		assert_identical(reader.read('chr1', [0, 100000], 100).get(), expected)


def test_cuda_splits_mixed_cache_batches(dense_bw, monkeypatch):
	"""Blocks validated by an earlier read go to nvCOMP, and new blocks are
	uploaded as the CPU decoded them, so a batch holding both is split until
	each part takes one route."""

	path, _ = dense_bw
	monkeypatch.setattr(cuda, '_uncompress2', lambda: None)
	reader = cuda.BigWigReader(path)
	reader.read('chr1', [0], 100)

	routes = []
	scatter_part = cuda.BigWigReader._scatter_part
	def recorded(self, cupy, raws, metadata, decoded, *args):
		routes.append({data is None for data in decoded})
		return scatter_part(self, cupy, raws, metadata, decoded, *args)
	monkeypatch.setattr(cuda.BigWigReader, '_scatter_part', recorded)

	starts = [0, 100000, 200000]
	assert_identical(reader.read('chr1', starts, 100).get(),
		figwig.read_bigwig(path, 'chr1', starts, 100))
	assert {True} in routes and {False} in routes
	assert all(len(route) == 1 for route in routes)
