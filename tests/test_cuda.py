# test_cuda.py
# Reads onto a GPU through cupy: the device path gives the CPU read's values
# in every mode, by both of its routes, and hands the array to torch through
# DLPack. The tests skip without cupy or without a CUDA device, never on a
# failed import elsewhere.

import pickle

import numpy
import pytest

import figwig._cuda
from figwig import FigWigReader
from figwig import FigWigWriter
from figwig import read_figwig
from figwig import write_figwig

from .test_figwig import reference_tracks
from .test_figwig import write_reference

cupy = pytest.importorskip('cupy')
try:
	N_DEVICES = cupy.cuda.runtime.getDeviceCount()
except Exception:
	N_DEVICES = 0
pytestmark = pytest.mark.skipif(N_DEVICES == 0, reason="no CUDA device")


@pytest.fixture(scope='module')
def fw(tmp_path_factory):
	path = str(tmp_path_factory.mktemp('cuda') / 'reference.fw')
	write_reference(path, reference_tracks())
	return path


@pytest.fixture(params=['scatter', 'pinned'])
def route(request, monkeypatch):
	"""Force each of the two dense routes."""

	monkeypatch.setattr(figwig._cuda, '_SCATTER_BYTES_PER_ENTRY', 0 if
		request.param == 'scatter' else 10 ** 9)
	return request.param


def assert_same(device_array, host_array):
	assert isinstance(device_array, cupy.ndarray)
	assert device_array.shape == host_array.shape
	assert device_array.dtype == host_array.dtype
	got = cupy.asnumpy(device_array)
	numpy.testing.assert_array_equal(numpy.nan_to_num(got, nan=-7), numpy.nan_to_num(
		host_array, nan=-7))


def test_dense_reads_match_the_cpu(fw, route):
	reader = FigWigReader(fw)
	rng = numpy.random.default_rng(0)
	chroms = rng.choice(['chr1', 'chr2'], 40)
	starts = rng.integers(0, 4000, 40)
	for kwargs in ({}, {'dtype': 'float32'}, {'tracks': 'atac_plus'}, {'tracks':
			[3, 0]}, {'dtype': 'float64', 'tracks': ['signal', 'empty']}):
		assert_same(reader.read(chroms, starts, 300, device='cuda', **kwargs),
			reader.read(chroms, starts, 300, **kwargs))
	assert_same(read_figwig(fw, chroms, starts, 300, device=0), read_figwig(fw,
		chroms, starts, 300))
	assert_same(reader.read(chroms[:0], [], 300, device='cuda:0'), reader.read(
		chroms[:0], [], 300))
	# Windows past a chromosome's end: NaN in a float output, 0 in an
	# integer one, as on the CPU.
	assert_same(reader.read(['chr2', 'chrM'], [2900, 10], 300, device='cuda'),
		reader.read(['chr2', 'chrM'], [2900, 10], 300))
	assert_same(reader.read(['chr2', 'chrM'], [2900, 10], 300, tracks=[0, 1],
		dtype='uint16', device='cuda'), reader.read(['chr2', 'chrM'], [2900, 10],
		300, tracks=[0, 1], dtype='uint16'))


def test_sums_labels_bins_and_sparse_on_the_device(fw):
	reader = FigWigReader(fw)
	rng = numpy.random.default_rng(1)
	chroms = rng.choice(['chr1', 'chr2'], 30)
	starts = rng.integers(0, 4000, 30)
	ends = starts + rng.integers(1, 900, 30)
	assert_same(reader.sums(chroms, starts, ends, device='cuda'), reader.sums(
		chroms, starts, ends))
	assert_same(reader.sums(chroms, starts, ends, tracks=1, nan='skip',
		device='cuda'), reader.sums(chroms, starts, ends, tracks=1, nan='skip'))
	assert_same(reader.sums(chroms, starts, ends, tracks=[0, 1, 2, 3, 4][:len(
		starts)] * 6, paired=True, device='cuda'), reader.sums(chroms, starts,
		ends, tracks=[0, 1, 2, 3, 4][:len(starts)] * 6, paired=True))
	labels = [0, 0, 1, 1, 2]
	assert_same(reader.read(chroms, starts, 256, labels=labels, device='cuda'),
		reader.read(chroms, starts, 256, labels=labels))
	assert_same(reader.read(chroms, starts, 256, bins=64, dtype='float64',
		device='cuda'), reader.read(chroms, starts, 256, bins=64, dtype='float64'))
	a = reader.read(chroms, starts, 256, sparse=True, device='cuda')
	b = reader.read(chroms, starts, 256, sparse=True)
	assert a.shape == b.shape and isinstance(a.values, cupy.ndarray)
	for field in ('rows', 'channels', 'offsets', 'values'):
		assert_same(getattr(a, field), getattr(b, field))


def test_out_on_the_device(fw, route):
	reader = FigWigReader(fw)
	chroms, starts = ['chr1'] * 8, numpy.arange(0, 800, 100)
	want = reader.read(chroms, starts, 128, dtype='float32')
	out = cupy.full(want.shape, 9, dtype=cupy.float32)
	got = reader.read(chroms, starts, 128, dtype='float32', out=out,
		device='cuda')
	assert got is out
	assert_same(out, want)
	with pytest.raises(ValueError, match="C-contiguous cupy array of shape"):
		reader.read(chroms, starts, 128, out=numpy.zeros(want.shape,
			numpy.float32), dtype='float32', device='cuda')
	with pytest.raises(ValueError, match="C-contiguous cupy array of shape"):
		reader.read(chroms, starts, 128, out=cupy.zeros((8, 5, 64), cupy.float32),
			dtype='float32', device='cuda')
	with pytest.raises(ValueError, match="out= cannot be given with sparse"):
		reader.read(chroms, starts, 128, out=out, sparse=True, device='cuda')


def test_device_argument_and_pinned_buffers(fw):
	reader = FigWigReader(fw)
	for bad in ('gpu', 'cuda:x', True):
		with pytest.raises((ValueError, TypeError), match="device must be"):
			reader.read('chr1', [0], 64, device=bad)
	with pytest.raises(Exception):
		reader.read('chr1', [0], 64, device=N_DEVICES + 10)
	# A pinned buffer is kept per shape and dtype and does not survive a
	# pickle, which DataLoader workers take.
	reader.read('chr1', [0, 100], 64, labels=[0, 0, 1, 1, 1], device='cuda')
	assert len(reader._pinned) == 1
	copy = pickle.loads(pickle.dumps(reader))
	assert '_pinned' not in copy.__dict__
	assert_same(copy.read('chr1', [0, 100], 64, device='cuda'), reader.read(
		'chr1', [0, 100], 64))


def test_dlpack_to_torch(fw):
	torch = pytest.importorskip('torch')
	if not torch.cuda.is_available():
		pytest.skip("torch has no CUDA")
	reader = FigWigReader(fw)
	y = reader.read(['chr1'] * 4, [0, 500, 1000, 1500], 256, dtype='float32',
		device='cuda')
	t = torch.from_dlpack(y)
	assert t.is_cuda and t.shape == (4, 5, 256)
	numpy.testing.assert_array_equal(t.cpu().numpy(), reader.read(['chr1'] * 4,
		[0, 500, 1000, 1500], 256, dtype='float32'))


def test_device_needs_cupy(fw, monkeypatch):
	import builtins
	real_import = builtins.__import__

	def no_cupy(name, *args, **kwargs):
		if name == 'cupy':
			raise ImportError('no cupy')
		return real_import(name, *args, **kwargs)

	monkeypatch.setattr(builtins, '__import__', no_cupy)
	with pytest.raises(ImportError, match="device= needs cupy"):
		FigWigReader(fw).read('chr1', [0], 64, device='cuda')


def test_a_written_file_reads_on_the_device(tmp_path):
	path = str(tmp_path / 'w.fw')
	values = numpy.zeros((2, 3, 500), dtype=numpy.float32)
	values[0, 1, 10:20] = 2.5
	values[1, 2, ::7] = 1
	write_figwig(path, {'chr1': 1000}, ['a', 'b', 'c'], 'chr1', [0, 500], values,
		tile=64)
	with FigWigWriter(path, None, ['d'], append=True) as writer:
		writer.write('chr1', [0], numpy.ones((1, 1, 1000)))
	reader = FigWigReader(path)
	assert_same(reader.read('chr1', [0, 500], 500, device='cuda'), reader.read(
		'chr1', [0, 500], 500))
