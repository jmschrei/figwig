# test_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import sys
import zlib
import types

import numpy
import pytest
import deflate

import figwig._kernels


def blocks_of(sizes, random_state=0):
	"""Blocks of these lengths, as (data, bounds): random bytes with few
	distinct values, so that they compress, and one repetitive block."""

	rng = numpy.random.default_rng(random_state)
	blocks = [rng.integers(0, 20, size=n, dtype=numpy.uint8).tobytes() for n in
		sizes]
	blocks.append(b'ACGT' * 250)
	data = numpy.frombuffer(b''.join(blocks), dtype=numpy.uint8)
	bounds = numpy.cumsum([0] + [len(block) for block in blocks]).astype(
		numpy.int64)
	return blocks, data, bounds


def split_streams(out, sizes):
	return numpy.split(out[:sizes.sum()], numpy.cumsum(sizes)[:-1])


##


@pytest.mark.parametrize('level', [1, 6, 9])
def test_deflate_blocks_zlib(level):
	"""Each block is compressed by compress2() into the stream zlib.compress
	gives at the same level, the streams are back to back in `out`, and an
	empty block is a stream of its own."""

	blocks, data, bounds = blocks_of([0, 1, 24, 32768, 32760])
	compress2, ulong = figwig._kernels._zlib_compress()
	out = numpy.empty(bounds[-1] * 2 + 1000, dtype=numpy.uint8)
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)

	status = figwig._kernels._deflate_blocks_zlib(compress2, level, data, bounds,
		out, sizes, numpy.zeros(1, dtype=ulong))

	assert status == -1
	streams = split_streams(out, sizes)
	assert [stream.tobytes() for stream in streams] == [zlib.compress(block,
		level) for block in blocks]


@pytest.mark.parametrize('level', [1, 6, 12])
def test_deflate_blocks_libdeflate(level):
	blocks, data, bounds = blocks_of([0, 1, 24, 32768, 32760])
	alloc, compress, free = figwig._kernels._libdeflate()
	out = numpy.empty(bounds[-1] * 2 + 1000, dtype=numpy.uint8)
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)

	compressor = alloc(level)
	try:
		status = figwig._kernels._deflate_blocks_libdeflate(compress, compressor,
			data, bounds, out, sizes)
	finally:
		free(compressor)

	assert status == -1
	streams = split_streams(out, sizes)
	assert [stream.tobytes() for stream in streams] == [deflate.zlib_compress(
		block, level) for block in blocks]
	assert [zlib.decompress(stream.tobytes()) for stream in streams] == blocks


def test_deflate_blocks_report_the_block_that_does_not_fit():
	"""When what is left of `out` cannot hold a block's stream, both kernels
	stop and return that block, here the fourth, after the first three have
	used 31 of the 100 bytes."""

	blocks, data, bounds = blocks_of([0, 1, 24, 32768])
	out = numpy.empty(100, dtype=numpy.uint8)

	compress2, ulong = figwig._kernels._zlib_compress()
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)
	assert figwig._kernels._deflate_blocks_zlib(compress2, 6, data, bounds, out,
		sizes, numpy.zeros(1, dtype=ulong)) == 3

	alloc, compress, free = figwig._kernels._libdeflate()
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)
	compressor = alloc(6)
	try:
		assert figwig._kernels._deflate_blocks_libdeflate(compress, compressor,
			data, bounds, out, sizes) == 3
	finally:
		free(compressor)


def test_zlib_compress_loader(monkeypatch):
	"""compress2() is looked for in the same library, the same way, as
	uncompress(), and without it the loader gives None."""

	import ctypes
	import ctypes.util

	loaded = figwig._kernels._load_zlib_compress()
	function, ulong = loaded
	assert function.restype is ctypes.c_int
	assert function.argtypes[-1] is ctypes.c_int
	assert ulong == numpy.dtype(ctypes.c_ulong)

	def library(name):
		raise OSError(name)

	monkeypatch.setattr(ctypes, 'CDLL', library)
	monkeypatch.setattr(ctypes.util, 'find_library', lambda name: None)
	assert figwig._kernels._load_zlib_compress() is None


def test_libdeflate_loader_without_deflate(monkeypatch):
	"""Without the `deflate` package, or when its library does not export
	libdeflate's functions, the loader gives None, and blocks are compressed
	with zlib."""

	with monkeypatch.context() as patch:
		patch.setitem(sys.modules, 'deflate._deflate', None)
		assert figwig._kernels._load_libdeflate() is None

	# A shared library that loads but has none of libdeflate's functions.
	import _ctypes
	monkeypatch.setattr(deflate, '_deflate', types.SimpleNamespace(
		__file__=_ctypes.__file__))
	assert figwig._kernels._load_libdeflate() is None
