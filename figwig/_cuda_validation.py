# _cuda_validation.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import hashlib
import sys
import zlib
from collections import OrderedDict

import numpy

from ._kernels import _check_block


class _ValidatedBlocks:
	"""Bounded records of bytes checked on the CPU before GPU decompression."""

	def __init__(self, limit=64 << 20):
		self.limit = limit
		self.entries = OrderedDict()
		self.bytes = 0

	def clear(self):
		self.entries.clear()
		self.bytes = 0

	def get(self, key):
		"""The value stored under `key`, now the most recently used, or None."""

		if key not in self.entries:
			return None
		self.entries.move_to_end(key)
		return self.entries[key][0]

	def put(self, key, value):
		"""Store `value` under `key`, dropping the least recently used entries
		until it fits. A value larger than the whole limit is not kept."""

		old = self.entries.pop(key, None)
		if old is not None:
			self.bytes -= old[1]
		cost = _cost(key, value)
		while self.entries and self.bytes + cost > self.limit:
			_, (_, old_cost) = self.entries.popitem(last=False)
			self.bytes -= old_cost
		if cost <= self.limit:
			self.entries[key] = value, cost
			self.bytes += cost

	def check(self, raw, compressed, chrom, start, end):
		"""Return metadata and, on a cache miss, the already decoded bytes."""

		key = (hashlib.sha256(raw).digest(), compressed, int(chrom), int(start),
			int(end))
		metadata = self.get(key)
		if metadata is not None:
			return metadata, None

		try:
			if compressed:
				decoder = zlib.decompressobj()
				decoded = decoder.decompress(raw) + decoder.flush()
				if not decoder.eof:
					return None, None
				payload_end = len(raw) - len(decoder.unused_data) - 4
			else:
				decoded, payload_end = raw, len(raw)
		except zlib.error:
			return None, None
		if not decoded or len(decoded) % 4:
			return None, None
		words = numpy.frombuffer(decoded, dtype='<u4')
		good, first, last = _check_block(words, 0, len(words), chrom, start, end)
		if not good:
			return None, None
		metadata = (len(decoded), int(first), int(last), payload_end)
		self.put(key, metadata)
		return metadata, decoded


def _cost(key, value):
	"""The bytes an entry takes, counting its containers as well as its
	contents, rather than its hashes alone."""

	contents = value.nbytes if isinstance(value, numpy.ndarray) else sum(
		sys.getsizeof(item) for item in value)
	return 256 + sys.getsizeof(key) + sum(sys.getsizeof(item) for item in key) + \
		sys.getsizeof(value) + contents


class _DecodedBatch:
	"""Uniform CPU output buffers, uploaded without another padded host copy."""

	def __init__(self, data, cap):
		self.data, self.cap = data, cap

	def __getitem__(self, selection):
		return _DecodedBatch(self.data.reshape(-1, self.cap)[selection].reshape(-1),
			self.cap)


def _group_key(data, leaves):
	"""The cache key of a whole staged batch: its bytes and its blocks."""

	return ('batch', hashlib.sha256(data).digest(), leaves.tobytes())


class _StagedBlocks:
	"""Views of packed file bytes, retaining the exact bytes that were hashed."""

	def __init__(self, data, starts, sizes, available):
		self.data, self.starts, self.sizes = data, starts, sizes
		self.available = available

	def __len__(self):
		return len(self.starts)

	def __getitem__(self, selection):
		if isinstance(selection, slice):
			return _StagedBlocks(self.data, self.starts[selection],
				self.sizes[selection], self.available[selection])
		if not self.available[selection]:
			return b''
		a = int(self.starts[selection])
		return memoryview(self.data)[a:a + int(self.sizes[selection])]


class _DeflateBlocks:
	"""Validated raw Deflate payloads within a staged batch."""

	def __init__(self, data, starts, sizes):
		self.data, self.starts, self.sizes = data, starts, sizes

	def __len__(self):
		return len(self.starts)
