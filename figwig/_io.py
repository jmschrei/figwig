# _io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import ctypes
import errno
import os
import sys
from functools import lru_cache

import numba
import numpy

from ._kernels import _pread_into

_PREAD = getattr(os, 'pread', None)
_PREADV = getattr(os, 'preadv', None)


@lru_cache(None)
def _native_reads():
	"""POSIX positional reads with the calling thread's errno accessor."""

	if sys.platform not in ('linux', 'darwin') or ctypes.sizeof(
			ctypes.c_void_p) != 8:
		return None
	try:
		library = ctypes.CDLL(None)
		read = library.pread
		read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
			ctypes.c_int64]
		read.restype = ctypes.c_ssize_t
		error = getattr(library, '__errno_location' if sys.platform == 'linux'
			else '__error')
		error.argtypes = []
		error.restype = ctypes.POINTER(ctypes.c_int)
		return read, error
	except (AttributeError, OSError):
		return None


@numba.njit(nogil=True, cache=True)
def _read_block_runs(read, error, fd, data, starts, ends, offsets, available):
	"""Coalesce adjacent blocks and mark short reads in the same native call."""

	a = 0
	while a < len(starts):
		b = a + 1
		while b < len(starts) and offsets[b] == offsets[b-1] + ends[b-1] - starts[b-1]:
			b += 1
		total, size = 0, ends[b-1] - starts[a]
		while total < size:
			n = read(fd, data.ctypes.data + starts[a] + total,
				size - total, offsets[a] + total)
			if n < 0:
				code = error()[0]
				if code == errno.EINTR:
					continue
				return code
			if n == 0:
				break
			total += n
		for i in range(a, b):
			available[i] = ends[i] <= starts[a] + total
		a = b
	return 0


def _pread_blocks(fd, data, starts, ends, offsets):
	"""Gather adjacent blocks, with a portable Python fallback."""

	available = numpy.ones(len(starts), dtype=numpy.int64)
	native = _native_reads()
	if native is not None and not numba.config.DISABLE_JIT and \
			getattr(os, 'pread', None) is _PREAD and \
			getattr(os, 'preadv', None) is _PREADV:
		code = _read_block_runs(*native, fd, data, starts, ends, offsets, available)
		if code:
			raise OSError(code, os.strerror(code))
	else:
		breaks = numpy.flatnonzero(offsets[1:] != offsets[:-1] +
			ends[:-1] - starts[:-1]) + 1
		for a, b in zip(numpy.append(0, breaks), numpy.append(breaks, len(starts))):
			if a == b:
				continue
			got = _pread_into(fd, data[starts[a]:ends[b-1]], int(offsets[a]))
			available[a:b] = ends[a:b] <= starts[a] + got
	return available
