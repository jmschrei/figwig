# _isal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Reading with ISA-L's inflate, through the functions that python-isal's
extension already exports, called from numba without the GIL. No compiler
is needed."""

import ctypes
import sys
import zlib
from functools import lru_cache

import numpy

from ._kernels import _isal_inflate_blocks

# struct inflate_state of each ISA-L version whose layout has been checked:
# its size in bytes, and the byte offsets of its block_state and crc_flag
# fields, from that version's include/igzip_lib.h on a 64-bit platform. Its
# first fields, which every version shares, are written by
# `_isal_inflate_blocks`.
_LAYOUTS = {'2.31.1': (87368, 21160, 21172)}


@lru_cache(None)
def _isal_functions():
	"""ISA-L's isal_inflate_init() and isal_inflate_stateless() as ctypes
	functions, with the layout of their state, or None where python-isal is
	not installed, its ISA-L's layout has not been checked, the platform is
	not 64-bit, or its extension does not export them."""

	try:
		import isal
		from isal import isal_zlib
	except ImportError:
		return None

	layout = _LAYOUTS.get(getattr(isal, 'ISAL_VERSION', None))
	if layout is None or sys.maxsize < 2 ** 32:
		return None

	try:
		library = ctypes.CDLL(isal_zlib.__file__)
		init, inflate = library.isal_inflate_init, library.isal_inflate_stateless
	except (OSError, AttributeError):
		return None

	init.argtypes, init.restype = [ctypes.c_void_p], None
	inflate.argtypes, inflate.restype = [ctypes.c_void_p], ctypes.c_int
	return init, inflate, layout


class _IsalDecoder:
	"""Inflates data blocks with ISA-L. Blocks it cannot inflate go to
	python-isal's decompress, then zlib's, as do all blocks where its
	functions cannot be called from numba, which is much slower."""

	batched_reads = True
	kernel = staticmethod(_isal_inflate_blocks)

	def __init__(self, functions):
		self.functions = functions

	def inflate(self, data, starts, sizes, buffer, blocks):
		"""Inflate blocks into `buffer`, returning the bytes used, or None to
		leave every block to `decompress`."""

		if self.functions is None:
			return None

		init, inflate, (size, block_state, crc_flag) = self.functions
		memory = numpy.empty(size + 64, dtype=numpy.uint8)
		state = memory[-memory.ctypes.data % 64:][:size]
		return self.kernel(init, inflate, state, block_state, crc_flag, data,
			starts, sizes, buffer, blocks)

	@staticmethod
	def decompress(raw):
		from isal import isal_zlib

		try:
			return isal_zlib.decompress(raw)
		except isal_zlib.error:
			return zlib.decompress(raw)


def _isal_decoder(required):
	"""An `_IsalDecoder`, or None where ISA-L cannot be called from numba and
	is not `required`. A required decoder needs python-isal, figwig[isal]."""

	functions = _isal_functions()
	if functions is not None:
		return _IsalDecoder(functions)
	if not required:
		return None

	try:
		import isal  # noqa: F401
	except ImportError as error:
		raise ImportError("engine='isal' requires figwig[isal].") from error
	return _IsalDecoder(None)
