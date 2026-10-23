# _cuda.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Reads of a figwig file onto a GPU, through CuPy.

`read_on_device` runs a read on the CPU and lands its output in device
memory as a `cupy.ndarray`, which PyTorch, JAX, TensorFlow and CuPy take
without a copy through DLPack (`torch.from_dlpack(y)`,
`jax.dlpack.from_dlpack(y)`, `tf.experimental.dlpack.from_dlpack(
y.__dlpack__())`). A dense read goes one of two ways: when the tracks are
sparse enough that their nonzero bases take fewer bytes than the dense
windows, the read lists the entries (`sparse=True`), moves them and
scatters them into a zeroed device array; otherwise it decodes into a
pinned host buffer, kept on the reader, and copies that. Sums per label,
per bin and over intervals, whose outputs are small, go through the
pinned buffer. The device's stream is synchronized before the array is
returned, so the array is complete for any framework's stream.
"""

import numpy

# Below this fraction of nonzero bases, by the file's statistics, a dense
# read is scattered from its entries on the device rather than copied
# dense. Measured on an H200: at 0.002 percent (128 of 10,246 cells) the
# entries route took 0.05 s against 0.25 s for the pinned copy; at 1.1
# percent (ATAC and DNase counts) it took 0.18 s against 0.10 s.
_SCATTER_DENSITY = 0.002


def _cupy():
	try:
		import cupy
	except ImportError as e:
		raise ImportError("device= needs cupy, which reads onto a GPU: install "
			"figwig[cuda] (cupy-cuda12x), or cupy-cuda13x or cupy-cuda11x for "
			"CUDA 13 or 11.") from e
	return cupy


def _device_of(device):
	"""The CUDA device index that `device` names: 'cuda', 'cuda:1', or an
	int."""

	if isinstance(device, bool):
		raise TypeError("device must be 'cuda', 'cuda:N' or an int, not a bool.")
	if isinstance(device, (int, numpy.integer)):
		return int(device)
	if isinstance(device, str):
		if device == 'cuda':
			return 0
		if device.startswith('cuda:') and device[5:].isdigit():
			return int(device[5:])
	raise ValueError("device must be 'cuda', 'cuda:N' or an int, not {!r}."
		.format(device))


def _pinned(reader, shape, dtype):
	"""A pinned host array of `shape` and `dtype`, kept on the reader for
	the next read of the same shape; pinning is slow, so one buffer per
	shape is reused."""

	cupy = _cupy()
	key = (tuple(shape), numpy.dtype(dtype).str)
	cache = reader.__dict__.setdefault('_pinned', {})
	if key not in cache:
		if len(cache) >= 4:
			cache.pop(next(iter(cache)))
		n = int(numpy.prod(shape, dtype=numpy.int64))
		memory = cupy.cuda.alloc_pinned_memory(max(n * numpy.dtype(dtype).itemsize,
			1))
		cache[key] = numpy.frombuffer(memory, dtype=dtype, count=n).reshape(shape)
	return cache[key]


def _check_out(cupy, out, shape, dtype):
	if not isinstance(out, cupy.ndarray) or out.shape != tuple(shape) or \
			out.dtype != numpy.dtype(dtype) or not out.flags.c_contiguous:
		raise ValueError("with device=, out must be a C-contiguous cupy array of "
			"shape {} and dtype {}.".format(tuple(shape), numpy.dtype(dtype)))
	return out


def _density(reader, ids):
	"""The fraction of bases that are nonzero over the tracks `ids`, from
	the file's statistics."""

	bases = float(reader._n_tiles) * reader.tile * max(len(ids), 1)
	return float(reader.stats['nonzero'][ids].sum()) / bases


def _copy_in(cupy, host, out, stream):
	"""Copy a host array to `out` on the device, on `stream`."""

	out.set(numpy.ascontiguousarray(host), stream=stream)
	return out


def read_on_device(reader, device, chroms, starts, width, tracks, out, dtype,
	n_jobs, sparse, labels, bins, stops, nan):
	"""`FigWigReader.read` or `sums` with its output on `device`."""

	cupy = _cupy()
	with cupy.cuda.Device(_device_of(device)):
		stream = cupy.cuda.get_current_stream()
		if sparse:
			if out is not None:
				raise ValueError("out= cannot be given with sparse=True.")
			entries = reader._read(chroms, starts, width, tracks, None, dtype,
				n_jobs, True, labels, bins, stops, nan)
			moved = type(entries)(cupy.asarray(entries.rows), cupy.asarray(
				entries.channels), cupy.asarray(entries.offsets), cupy.asarray(
				entries.values), entries.shape)
			stream.synchronize()
			return moved

		# The dense shape and dtype, as the CPU read would give them.
		ids, single = reader._select(tracks)
		summing = labels is not None or bins is not None
		out_dtype = reader._dtype(ids, dtype, summing)
		n = len(numpy.atleast_1d(starts))
		if labels is not None:
			channels = int(numpy.max(labels)) + 1 if len(labels) else 0
		else:
			channels = len(ids)
		last = width // bins if bins else width
		shape = (n, last) if single and labels is None else (n, channels, last)
		if out is not None:
			_check_out(cupy, out, shape, out_dtype)

		dense = not summing and bins is None and stops is None
		if dense and _density(reader, ids) < _SCATTER_DENSITY:
			# Few nonzero bases: list them, move them and scatter.
			entries = reader._read(chroms, starts, width, tracks, None, out_dtype,
				n_jobs, True, None, None, None, nan)
			result = out if out is not None else cupy.empty(shape, dtype=out_dtype)
			result.fill(0)
			if len(entries.values):
				rows = cupy.asarray(entries.rows)
				offsets = cupy.asarray(entries.offsets)
				values = cupy.asarray(entries.values)
				if result.ndim == 2:
					result[rows, offsets] = values
				else:
					result[rows, cupy.asarray(entries.channels), offsets] = values
			# Bases past the end of a chromosome are NaN in a float output,
			# as the dense read gives them; the entries say nothing of them.
			if numpy.dtype(out_dtype).kind == 'f' and n:
				from .bigwig import _chrom_codes
				names, codes = _chrom_codes(chroms, n)
				lengths = numpy.array([reader.chrom_sizes[name] for name in names],
					dtype=numpy.int64)[codes]
				starts_ = numpy.asarray(starts, dtype=numpy.int64)
				for j in numpy.flatnonzero(starts_ + width > lengths).tolist():
					result[j, ..., max(0, int(lengths[j] - starts_[j])):] = numpy.nan
			stream.synchronize()
			return result

		host = _pinned(reader, shape, out_dtype)
		reader._read(chroms, starts, width, tracks, host, out_dtype, n_jobs,
			False, labels, bins, stops, nan)
		result = out if out is not None else cupy.empty(shape, dtype=out_dtype)
		_copy_in(cupy, host, result, stream)
		stream.synchronize()
		return result
