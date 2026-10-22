# _cuda_inflate.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import ctypes
import ctypes.util
from functools import lru_cache

import numba
import numpy

from ._kernels import _check_block


@lru_cache(None)
def _uncompress2():
	"""zlib's checked decoder, including the consumed zlib-wrapper length."""

	if numba.config.DISABLE_JIT:
		return None
	try:
		name = ctypes.util.find_library('z')
		if name is None:
			return None
		library = ctypes.CDLL(name)
		function = library.uncompress2
		function.argtypes = [ctypes.c_void_p] * 4
		function.restype = ctypes.c_int
		return function, numpy.dtype(ctypes.c_ulong)
	except (OSError, AttributeError):
		return None


@numba.njit(nogil=True, cache=True)
def _validate(function, data, starts, sizes, chroms, bases, out, cap, metadata,
	length, consumed):
	"""Check checksums and sections in a native loop before allowing CUDA use."""

	words = out.view(numpy.uint32)
	for b in range(len(starts)):
		length[0], consumed[0] = cap, sizes[b]
		metadata[b, 0] = -1
		status = function(out.ctypes.data + b * cap, length.ctypes.data,
			data.ctypes.data + starts[b], consumed.ctypes.data)
		if status != 0:
			if status == -5:
				metadata[b, 0] = 0
			continue
		if length[0] == 0 or length[0] % 4:
			continue
		good, first, last = _check_block(words, b * (cap // 4),
			b * (cap // 4) + length[0] // 4, chroms[b], bases[b, 0], bases[b, 1])
		if good:
			metadata[b, 0] = length[0]
			metadata[b, 1] = first
			metadata[b, 2] = last
			metadata[b, 3] = consumed[0] - 4


@numba.njit(nogil=True, cache=True)
def _check_reaches(metadata, lo, hi, reach, failed):
	"""Carry interval-overlap checks across chunks without per-block Python calls."""

	for b in range(len(lo)):
		for row in range(lo[b], hi[b]):
			if metadata[b, 1] < reach[row]:
				failed[row] = True
			reach[row] = max(reach[row], metadata[b, 2])


@numba.njit(nogil=True, cache=True)
def _pack_loop(data, starts, sizes, offsets, out):
	"""Align validated Deflate payloads without per-block Python copies."""

	for b in range(len(starts)):
		start, offset, size = int(starts[b]), int(offsets[b]), int(sizes[b])
		for j in range(size):
			out[offset + j] = data[start + j]


@numba.njit(nogil=True, cache=True)
def _copy_payloads(function, data, starts, sizes, offsets, out):
	for b in range(len(starts)):
		function(out.ctypes.data + offsets[b], data.ctypes.data + starts[b],
			sizes[b])


def _pack(data, starts, sizes, offsets, out):
	"""Copy each validated payload with ctypes' memmove, which every platform
	has, or with a loop where numba is disabled and cannot call it."""

	if numba.config.DISABLE_JIT:
		_pack_loop(data, starts, sizes, offsets, out)
	else:
		_copy_payloads(ctypes.memmove, data, starts, sizes, offsets, out)
