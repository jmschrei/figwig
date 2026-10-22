# _nvcomp.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import ctypes
import importlib.metadata
import pathlib
from functools import lru_cache

import numpy

from ._cuda_inflate import _pack


class _Options(ctypes.Structure):
	_fields_ = [('backend', ctypes.c_int), ('sort', ctypes.c_int),
		('reserved', ctypes.c_char * 56)]


class _Alignment(ctypes.Structure):
	_fields_ = [('input', ctypes.c_size_t), ('output', ctypes.c_size_t),
		('temp', ctypes.c_size_t)]


def _check(status):
	if status:
		raise RuntimeError('nvCOMP failed with status {}.'.format(status))


@lru_cache(None)
def _library():
	"""Load the optional ABI for which the adapter has been tested."""

	try:
		import nvidia.libnvcomp
		version = importlib.metadata.version('nvidia-libnvcomp-cu12')
		if version != '6.0.0.12':
			raise ImportError('figwig CUDA requires nvCOMP 6.0.0.12.')
		root = pathlib.Path(nvidia.libnvcomp.__path__[0]) / 'lib64'
		library = ctypes.CDLL(str(next(root.glob('libnvcomp.so.*'))))
	except (ImportError, StopIteration, OSError) as error:
		raise ImportError('CUDA reading requires figwig[cuda] and CUDA 12.') from error
	p, size = ctypes.c_void_p, ctypes.c_size_t
	library.nvcompBatchedDeflateDecompressGetRequiredAlignments.argtypes = [
		_Options, ctypes.POINTER(_Alignment)]
	library.nvcompBatchedDeflateDecompressGetTempSize.argtypes = [size, size,
		_Options, ctypes.POINTER(size), size, p]
	library.nvcompBatchedDeflateDecompressAsync.argtypes = [p, p, p, p, size, p,
		size, p, _Options, p, p]
	return library


def _aligned(cupy, size, alignment):
	allocation = cupy.empty(size + alignment - 1, dtype=cupy.uint8)
	offset = -allocation.data.ptr % alignment
	return allocation[offset:offset + size]


def _inflate(cupy, raws, lengths, budget):
	"""Inflate CPU-validated raw Deflate payloads, `_DeflateBlocks`, respecting
	nvCOMP's alignments."""

	library, options = _library(), _Options()
	alignment = _Alignment()
	_check(library.nvcompBatchedDeflateDecompressGetRequiredAlignments(options,
		ctypes.byref(alignment)))
	stream = cupy.cuda.get_current_stream()
	count, cap = len(raws), int(max(lengths))
	sizes = raws.sizes
	if cap > min(budget, 2**32 - 1) or numpy.any(sizes >= 2**32):
		raise MemoryError('A block exceeds the CUDA workspace or nvCOMP limit.')
	cap = (cap + alignment.output - 1) // alignment.output * alignment.output
	padded = (sizes + alignment.input - 1) // alignment.input * alignment.input
	offsets = numpy.cumsum(padded) - padded
	temp_size = ctypes.c_size_t()
	_check(library.nvcompBatchedDeflateDecompressGetTempSize(count, cap, options,
		ctypes.byref(temp_size), int(sum(lengths)), stream.ptr))
	total = int(padded.sum())
	if total + count * cap + temp_size.value + count * 64 + 4096 > budget:
		raise MemoryError('CUDA workspace limit is too small for this batch.')
	staging = numpy.zeros(total, dtype=numpy.uint8)
	_pack(raws.data, raws.starts, sizes, offsets, staging)
	compressed = _aligned(cupy, total, alignment.input)
	compressed.set(staging)
	decoded = _aligned(cupy, count * cap, alignment.output)
	temp = _aligned(cupy, max(1, temp_size.value), alignment.temp)
	inputs = cupy.asarray(offsets + numpy.uint64(compressed.data.ptr))
	outputs = cupy.asarray(numpy.arange(count, dtype=numpy.uint64) * cap +
		numpy.uint64(decoded.data.ptr))
	device_sizes = cupy.asarray(sizes)
	capacities = cupy.full(count, cap, dtype=cupy.uint64)
	completion = cupy.empty(2 * count, dtype=cupy.uint64)
	actual = completion[:count]
	statuses = completion[count:].view(cupy.int32)[:count]
	_check(library.nvcompBatchedDeflateDecompressAsync(inputs.data.ptr,
		device_sizes.data.ptr, capacities.data.ptr, actual.data.ptr, count,
		temp.data.ptr, temp_size.value, outputs.data.ptr, options,
		statuses.data.ptr, stream.ptr))
	# Check completion before any kernel uses the decoded bytes.
	checked = completion.get()
	if numpy.any(checked[count:].view(numpy.int32)[:count]) or not \
			numpy.array_equal(checked[:count], lengths):
		raise RuntimeError('nvCOMP returned an invalid decoded block.')
	return decoded, cap, actual
