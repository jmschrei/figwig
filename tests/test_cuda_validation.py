import struct
import zlib

import numpy
import pytest

from figwig._cuda_validation import _ValidatedBlocks


def block(kind=1):
	return struct.pack('<IIIIIBBH', 0, 100, 110, 0, 0, kind, 0, 1) + \
		struct.pack('<IIf', 100, 110, -0.0)


def test_cache_hashes_bytes_and_validation_context():
	cache = _ValidatedBlocks()
	raw = zlib.compress(block())
	info, decoded = cache.check(raw, True, 0, 100, 110)
	assert decoded == block()
	assert cache.check(raw, True, 0, 100, 110) == (info, None)
	assert cache.check(raw, True, 1, 100, 110) == (None, None)
	assert cache.check(raw[:-1] + bytes([raw[-1] ^ 1]), True, 0, 100, 110) == \
		(None, None)
	assert zlib.decompress(raw[2:info[3]], wbits=-15) == block()
	cache.clear()
	assert cache.bytes == 0 and not cache.entries


@pytest.mark.parametrize('raw', [b'', b'bad', zlib.compress(block(4)),
	zlib.compress(b'abcde'), zlib.compress(block())[:-1]])
def test_invalid_blocks_are_not_cached(raw):
	cache = _ValidatedBlocks()
	assert cache.check(raw, True, 0, 100, 110) == (None, None)
	assert cache.bytes == 0 and not cache.entries


def test_cache_eviction_and_trailing_bytes():
	cache = _ValidatedBlocks(limit=1024)
	for i in range(30):
		raw = zlib.compress(block()) + bytes([i])
		info, decoded = cache.check(raw, True, 0, 100, 110)
		assert decoded == block()
		assert zlib.decompress(raw[2:info[3]], wbits=-15) == block()
		assert cache.bytes <= cache.limit
	assert len(cache.entries) == 1
	assert cache.check(zlib.compress(block()) + b'\0', True, 0, 100, 110)[1] \
		== block()


def test_uncompressed_blocks_are_validated_and_cached():
	cache = _ValidatedBlocks()
	info, decoded = cache.check(block(), False, 0, 100, 110)
	assert decoded == block()
	assert cache.check(block(), False, 0, 100, 110) == (info, None)


def test_cuda_dependency_is_lazy(monkeypatch):
	import sys
	from figwig import cuda
	monkeypatch.setitem(sys.modules, 'cupy', None)
	with pytest.raises(ImportError, match=r'figwig\[cuda\]'):
		cuda._cupy()


def test_batch_cache_identity_and_shared_memory_bound():
	from figwig._cuda_validation import _group_key
	cache = _ValidatedBlocks(limit=2048)
	raw = zlib.compress(block())
	cache.check(raw, True, 0, 100, 110)
	data = numpy.frombuffer(raw, dtype='u1')
	leaves = numpy.array([1], dtype='i8')
	key = _group_key(data, leaves)
	metadata = numpy.array([[len(block()), 100, 110, len(raw)-4]])
	cache.put(key, metadata)
	assert cache.get(key) is metadata
	assert cache.get(_group_key(data[:-1], leaves)) is None
	assert cache.get(_group_key(data, leaves + 1)) is None
	for i in range(20):
		cache.put(_group_key(data, leaves + i), metadata)
		assert cache.bytes <= cache.limit
	assert cache.get(key) is None
	assert cache.bytes == sum(cost for _, cost in cache.entries.values())
	cache.put(key, numpy.zeros((1000, 4), dtype='i8'))
	assert not cache.entries and cache.bytes == 0


def test_numpy_output_uses_cpu_without_cuda(dense_bw, monkeypatch):
	import figwig
	from figwig import cuda
	from .test_bigwig import assert_identical
	path, _ = dense_bw
	class UnhashableReader(cuda.BigWigReader):
		__hash__ = None
	reader = UnhashableReader(path)
	def forbidden(*args):
		pytest.fail('NumPy output loaded CUDA or allocated device memory')
	monkeypatch.setattr(cuda, '_cupy', forbidden)
	monkeypatch.setattr(cuda, '_library', forbidden)
	monkeypatch.setattr(reader, '_device_read', forbidden)
	starts = [0, 1000, 249999]
	expected = figwig.read_bigwig(path, 'chr1', starts, 10)
	out = numpy.empty_like(expected)
	assert reader.read('chr1', starts, 10, out=out, workspace=1) is out
	assert_identical(out, expected)
	out = numpy.empty((3, 2, 10), dtype='f4')
	assert cuda.read_bigwig([reader, reader], 'chr1', starts, 10, out=out) is out
	assert_identical(out[:, 0], expected)
	assert_identical(out[:, 1], expected)
	for read in (reader.read, lambda *args, **kwargs:
			cuda.read_bigwig(reader, *args, **kwargs)):
		with pytest.warns(UserWarning, match='chromosomes not in') as caught:
			read('absent', [0], 10, out=numpy.empty((1, 10), dtype='f4'))
		assert caught[0].filename == __file__
	with pytest.raises(ValueError, match='workspace'):
		reader.read('chr1', starts, 10, out=out[:, 0].copy(), workspace=0)
	assert not reader._validated.entries


def test_numpy_output_refreshes_changed_files(tmp_path, monkeypatch):
	from figwig import cuda
	from .writers import write_raw_bigwig
	path = tmp_path / 'mutable.bw'
	def write(value):
		write_raw_bigwig(path, {'chr1': 1000}, [
			('chr1', 1, 0, 0, [(100, 110, value)])])
	write(1.0)
	reader = cuda.BigWigReader(path)
	out = numpy.empty((1, 10), dtype='f4')
	assert numpy.all(reader.read('chr1', [100], 10, out=out) == 1)
	write(2.0)
	assert numpy.all(reader.read('chr1', [100], 10, out=out) == 2)
	read = cuda._cpu_read
	def changed(*args, **kwargs):
		result = read(*args, **kwargs)
		write(3.0)
		return result
	with monkeypatch.context() as patch:
		patch.setattr(cuda, '_cpu_read', changed)
		with pytest.raises(OSError, match='changed during the read'):
			reader.read('chr1', [100], 10, out=out)
	assert not reader._validated.entries
	write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, b'not zlib')])
	with pytest.raises(ValueError, match='figwig cannot read'):
		reader.read('chr1', [100], 10, out=out)
