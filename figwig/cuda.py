# cuda.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Explicit CUDA reading with CPU validation of previously unseen blocks."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache

import numpy

from .bigwig import BigWigReader as _CPUReader
from .bigwig import _read as _cpu_read
from .bigwig import _Read, _Windows, _check_missing, _check_n_jobs
from ._cuda_validation import _ValidatedBlocks, _DecodedBatch
from ._cuda_validation import _StagedBlocks, _DeflateBlocks
from ._cuda_validation import _group_key
from ._cuda_inflate import _uncompress2, _validate, _check_reaches
from ._io import _pread_blocks
from ._nvcomp import _inflate, _library


def _cupy():
	try:
		import cupy
	except ImportError as error:
		raise ImportError('CUDA reading requires figwig[cuda].') from error
	return cupy


@lru_cache(None)
def _kernels(device):
	cupy = _cupy()
	module = cupy.RawModule(code=r'''
extern "C" {
__global__ void fill(float *out, const long long *valid, long long width,
    long long stride, float missing) {
    long long row = blockIdx.x;
    for (long long j = threadIdx.x; j < width; j += blockDim.x)
        out[row * stride + j] = j < valid[row] ? missing : __int_as_float(0x7fc00000);
}
__global__ void scatter(const unsigned char *data, long long cap,
    const unsigned long long *lengths, const long long *lo, const long long *hi,
    const long long *starts, const long long *rows, const long long *valid,
    long long width, long long stride, float *out) {
    long long b = blockIdx.x, offset = 0;
    const unsigned char *buf = data + b * cap;
    while (offset < lengths[b]) {
        const unsigned int *h = (const unsigned int *)(buf + offset);
        long long section = h[1], step = h[3], span = h[4];
        int kind = h[5] & 255;
        long long count = h[5] >> 16;
        int size = kind == 1 ? 3 : kind == 2 ? 2 : 1;
        const unsigned int *items = h + 6;
        for (long long i = threadIdx.x; i < count; i += blockDim.x) {
            long long a, e;
            float value;
            if (kind == 1) {
                a = items[3*i]; e = items[3*i+1]; value = __int_as_float(items[3*i+2]);
            } else if (kind == 2) {
                a = items[2*i]; e = a + span; value = __int_as_float(items[2*i+1]);
            } else {
                a = section + i * step; e = a + span; value = __int_as_float(items[i]);
            }
            if (isnan(value)) continue;
            long long left = lo[b], right = hi[b];
            while (left < right) {
                long long mid = (left + right) / 2;
                if (starts[mid] + width > a) right = mid; else left = mid + 1;
            }
            for (long long q = left; q < hi[b] && starts[q] < e; q++) {
                long long row = rows[q];
                long long x = a > starts[q] ? a : starts[q];
                long long end = starts[q] + valid[row];
                long long y = e < end ? e : end;
                for (long long j = x; j < y; j++) out[row * stride + j-starts[q]] = value;
            }
        }
        offset += 24 + count * size * 4;
    }
}
}''')
	return module.get_function('fill'), module.get_function('scatter')


def _identity(stat):
	return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _workspace(cupy, requested):
	if requested is not None:
		if isinstance(requested, bool) or not isinstance(requested,
				(int, numpy.integer)) or requested < 1:
			raise ValueError('workspace must be a positive number of bytes.')
		return int(requested)
	free, _ = cupy.cuda.runtime.memGetInfo()
	return min(4 << 30, max(0, free - (256 << 20)) // 4)


class BigWigReader(_CPUReader):
	"""Read bigWig windows into CuPy, or into an explicit NumPy output.

	Unseen compressed blocks are decoded and checked on the CPU. Only identical
	bytes encountered again may be decompressed by nvCOMP. Validation records
	use at most 64 MiB per reader, and are discarded when the file changes.
	NumPy output uses the CPU reader. CUDA dependencies are loaded on the
	first device read, not on module import.
	"""

	def __init__(self, path):
		super().__init__(path)
		self._cuda_lock = threading.Lock()
		self._validated = _ValidatedBlocks()
		self._identity = _identity(os.stat(self.path))

	def __getstate__(self):
		state = super().__getstate__()
		del state['_cuda_lock']
		# Worker processes validate their own file snapshots.
		state['_validated'] = _ValidatedBlocks()
		return state

	def __setstate__(self, state):
		super().__setstate__(state)
		self._cuda_lock = threading.Lock()

	def read(self, chroms, starts, width, out=None, n_jobs=8, missing=0.0, *,
		workspace=None):
		"""Read windows on the current CUDA device and stream.

		Return a CuPy float32 array unless `out` explicitly supplies a NumPy
		array, which uses the CPU reader. `workspace` limits temporary device
		storage in bytes; the default
		is the smaller of 4 GiB and one quarter of free memory after reserving
		256 MiB. Allocation failures retry smaller batches.
		"""

		return _read(self, chroms, starts, width, out, n_jobs, missing,
			workspace=workspace)

	def _refresh(self):
		"""Discard metadata and validation records when the file changes."""

		identity = _identity(os.stat(self.path))
		if identity != self._identity:
			_CPUReader.__init__(self, self.path)
			self._validated.clear()
			self._identity = identity
		return identity

	def _device_read(self, windows, out, n_jobs, missing, budget):
		cupy = _cupy()
		with self._cuda_lock:
			identity = self._refresh()
			plan = _Read(self, windows, numpy.empty((windows.n, 1, 0),
				dtype=numpy.float32), 0, missing, None)
			plan.open()
			try:
				if _identity(os.fstat(plan.fd)) != identity:
					raise OSError('The bigWig file changed while opening it.')
				self._decode(plan, out, n_jobs, missing, budget)
				if _identity(os.fstat(plan.fd)) != identity:
					self._validated.clear()
					raise OSError('The bigWig file changed during the read.')
			finally:
				cupy.cuda.get_current_stream().synchronize()
				plan.close()

	def _decode(self, plan, out, n_jobs, missing, budget):
		cupy = _cupy()
		windows, index, rows = plan.windows, plan.index, plan.rows
		fill, scatter = _kernels(cupy.cuda.Device().id)
		metadata_bytes = windows.n * 8 + len(rows) * 16
		if metadata_bytes >= budget:
			raise MemoryError('Window metadata cannot fit in CUDA workspace.')
		budget -= metadata_bytes
		valid = numpy.full(windows.n, windows.width, dtype=numpy.int64)
		valid[plan.order] = numpy.clip(rows[:, 3] - rows[:, 2], 0, windows.width)
		d_valid = cupy.asarray(valid)
		stride = numpy.int64(out.strides[0] // 4)
		fill((windows.n,), (128,), (out, d_valid, numpy.int64(windows.width),
			stride, missing))
		d_starts, d_rows = cupy.asarray(rows[:, 2]), cupy.asarray(rows[:, 4])
		reach = numpy.full(len(rows), numpy.iinfo(numpy.int64).min,
			dtype=numpy.int64)
		compressed = self._buffer_size > 0

		def gather(position, end):
			needed = plan.needed[position:end]
			offsets, sizes = index['offsets'][needed], index['sizes'][needed]
			ends = numpy.cumsum(sizes)
			starts = ends - sizes
			data = numpy.empty(int(ends[-1]), dtype=numpy.uint8)
			available = _pread_blocks(plan.fd, data, starts, ends, offsets)
			key = _group_key(data, needed) if available.all() else None
			return data, starts, sizes, _StagedBlocks(data, starts, sizes, available), key

		with ThreadPoolExecutor(1) as pool, ThreadPoolExecutor(n_jobs) as validators:
			cap = max(4, min(self._buffer_size, 1 << 20) // 4 * 4)
			position, batch_size = 0, min(2048, (64 << 20) // cap)
			future = pool.submit(gather, 0, min(batch_size, len(plan.needed))) if \
				len(plan.needed) else None
			while position < len(plan.needed):
				end = min(position + batch_size, len(plan.needed))
				data, starts, sizes, raws, key = future.result()
				future = pool.submit(gather, end, min(end + batch_size,
					len(plan.needed))) if end < len(plan.needed) else None
				needed = plan.needed[position:end]
				metadata = self._validated.get(key) if key is not None else None
				decoded = [None] * len(raws)
				if metadata is None:
					metadata, decoded = self._validate_batch(data, starts, sizes, raws,
						needed, index, compressed, cap, validators, n_jobs)
					if key is not None and numpy.all(metadata[:, 0] > 0):
						self._validated.put(key, metadata)
				lo = numpy.searchsorted(rows[:, 1], numpy.arange(position, end),
					side='right').astype(numpy.int64)
				hi = numpy.searchsorted(rows[:, 0], numpy.arange(position, end),
					side='right').astype(numpy.int64)
				for j in numpy.flatnonzero(metadata[:, 0] <= 0):
					plan.failed[lo[j]:hi[j]] = True
				plan.finish()
				_check_reaches(metadata, lo, hi, reach, plan.failed)
				plan.finish()
				self._scatter(cupy, raws, metadata, decoded, compressed, lo, hi,
					d_starts, d_rows, d_valid, out, stride, budget, scatter)
				position = end

	def _validate_batch(self, data, starts, sizes, raws, needed, index,
		compressed, cap, pool, n_jobs):
		"""Decode new bytes once on the CPU, in bounded native batches."""

		metadata = numpy.zeros((len(raws), 4), dtype=numpy.int64)
		sizes = sizes.copy()
		sizes[raws.available == 0] = 0
		native = _uncompress2() if compressed else None
		if native is not None:
			function, ulong = native
			output = numpy.empty(len(raws) * cap, dtype=numpy.uint8)
			chroms, bases = index['chroms'][needed], index['bases'][needed]
			step = max(1, (len(raws) + n_jobs - 1) // n_jobs)
			def validate(a):
				b = min(a + step, len(raws))
				_validate(function, data, starts[a:b], sizes[a:b], chroms[a:b],
					bases[a:b], output[a*cap:b*cap], cap, metadata[a:b],
					numpy.zeros(1, dtype=ulong), numpy.zeros(1, dtype=ulong))
			# Compile once before the worker threads enter the native loop.
			if not getattr(_validate, 'signatures', True):
				validate(0)
				first = step
			else:
				first = 0
			list(pool.map(validate, range(first, len(raws), step)))
			if numpy.all(metadata[:, 0] > 0):
				return metadata, _DecodedBatch(output, cap)
			decoded = [memoryview(output)[j*cap:j*cap+int(info[0])] if info[0] > 0
				else None for j, info in enumerate(metadata)]
		else:
			decoded = [None] * len(raws)
		for j in numpy.flatnonzero(metadata[:, 0] == 0):
			leaf = needed[j]
			info, decoded[j] = self._validated.check(raws[j], compressed,
				int(index['chroms'][leaf]), *index['bases'][leaf])
			if info is not None:
				metadata[j] = info
		return metadata, decoded

	def _scatter(self, cupy, raws, metadata, decoded, compressed, lo, hi,
		starts, rows, valid, out, stride, budget, scatter):
		"""Scatter a batch, in halves where it does not fit in the workspace or
		mixes blocks decoded on the CPU with blocks for nvCOMP, without changing
		the requested output type."""

		# Blocks decoded on the CPU are uploaded as they are, never inflated
		# twice, so a batch that mixes them with validated cache hits is split
		# until each part takes one route.
		mixed = compressed and not isinstance(decoded, _DecodedBatch) and \
			len({data is None for data in decoded}) > 1
		if not mixed:
			try:
				self._scatter_part(cupy, raws, metadata, decoded, compressed, lo, hi,
					starts, rows, valid, out, stride, budget, scatter)
				return
			except (MemoryError, cupy.cuda.memory.OutOfMemoryError):
				if len(raws) == 1:
					raise MemoryError('One bigWig block cannot fit in CUDA workspace.')
		# Leave the exception handler before retrying, releasing failed allocations.
		middle = len(raws) // 2
		for a, b in ((0, middle), (middle, len(raws))):
			self._scatter(cupy, raws[a:b], metadata[a:b], decoded[a:b],
				compressed, lo[a:b], hi[a:b], starts, rows, valid, out, stride,
				budget, scatter)

	def _scatter_part(self, cupy, raws, metadata, decoded, compressed, lo, hi,
		starts, rows, valid, out, stride, budget, scatter):
		lengths = [int(info[0]) for info in metadata]
		if isinstance(decoded, _DecodedBatch):
			cap = decoded.cap
			if decoded.data.nbytes + len(raws) * 32 > budget:
				raise MemoryError('CUDA workspace limit is too small.')
			buffer = cupy.asarray(decoded.data)
			actual = cupy.asarray(lengths, dtype=cupy.uint64)
		elif compressed and all(data is None for data in decoded):
			payloads = _DeflateBlocks(raws.data, raws.starts + 2,
				(metadata[:, 3] - 2).astype(numpy.uint64))
			buffer, cap, actual = _inflate(cupy, payloads, lengths, budget)
		else:
			cap = max(lengths)
			if len(raws) * cap + len(raws) * 32 > budget:
				raise MemoryError('CUDA workspace limit is too small.')
			host = numpy.zeros((len(raws), cap), dtype=numpy.uint8)
			for j, data in enumerate(decoded):
				data = raws[j] if data is None else data
				host[j, :len(data)] = numpy.frombuffer(data, dtype=numpy.uint8)
			buffer = cupy.asarray(host)
			actual = cupy.asarray(lengths, dtype=cupy.uint64)
		scatter((len(raws),), (128,), (buffer, numpy.int64(cap), actual,
			cupy.asarray(lo), cupy.asarray(hi), starts, rows, valid,
			numpy.int64(out.shape[1]), stride, out))


def _read(bigwigs, chroms, starts, width, out=None, n_jobs=8, missing=0.0,
	*, workspace=None):
	"""Read one or more bigWigs into CuPy, or an explicitly supplied NumPy array.

	Shapes and values match `figwig.read_bigwig`. CPU inputs and file planning
	use figwig's existing validation. NumPy output uses the CPU reader.
	"""

	single = not isinstance(bigwigs, (list, tuple))
	files = [bigwigs] if single else list(bigwigs)
	if not files:
		raise ValueError('bigwigs must hold at least one bigWig.')
	for file in files:
		if not isinstance(file, (str, os.PathLike, BigWigReader)):
			raise TypeError('bigwigs must be a path, a BigWigReader, or a list '
				'or tuple of them, not {}.'.format(type(file).__name__))
	files = [file if isinstance(file, BigWigReader) else BigWigReader(file)
		for file in files]
	if isinstance(out, numpy.ndarray):
		if workspace is not None:
			_workspace(None, workspace)
		# Lock duplicate readers only once, in a consistent order across calls.
		with ExitStack() as locks:
			readers = sorted({id(file): file for file in files}.values(), key=id)
			for file in readers:
				locks.enter_context(file._cuda_lock)
			identities = [file._refresh() for file in readers]
			result = _cpu_read(files, single, chroms, starts, width, out,
				n_jobs, missing, 'auto', stacklevel=4)
			for file, identity in zip(readers, identities):
				if _identity(os.stat(file.path)) != identity:
					file._validated.clear()
					raise OSError('The bigWig file changed during the read.')
			return result

	cupy = _cupy()
	_library()
	n_jobs, missing = _check_n_jobs(n_jobs), _check_missing(missing)
	windows = _Windows(chroms, starts, width)
	shape = (windows.n, windows.width) if single else (windows.n, len(files),
		windows.width)
	if out is None:
		out = cupy.empty(shape, dtype=cupy.float32)
	elif not isinstance(out, cupy.ndarray):
		raise TypeError('out must be a NumPy or CuPy array.')
	elif out.shape != shape or out.dtype != numpy.float32 or not \
			out.flags.c_contiguous:
		raise ValueError('out must be a writeable, C-contiguous float32 array '
			'of shape {}.'.format(shape))
	if isinstance(out, cupy.ndarray) and out.device.id != cupy.cuda.Device().id:
		raise ValueError('out must be on the current CUDA device.')
	budget = _workspace(cupy, workspace)
	if windows.n == 0:
		return out
	windows.check_range()
	for file in files:
		with file._cuda_lock:
			file._refresh()
		file._warn_absent(windows, missing, stacklevel=3)
	for i, file in enumerate(files):
		target = out if single else out[:, i]
		file._device_read(windows, target, n_jobs, missing, budget)

	return out


def read_bigwig(bigwigs, chroms, starts, width, out=None, n_jobs=8, missing=0.0,
	*, workspace=None):
	"""Read one or more bigWigs into CuPy or an explicit NumPy output array.

	Shapes and values match `figwig.read_bigwig`. Paths and CUDA readers may
	be mixed. `workspace` has the same meaning as in `BigWigReader.read`.
	"""

	return _read(bigwigs, chroms, starts, width, out, n_jobs, missing,
		workspace=workspace)
