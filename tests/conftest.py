"""Shared fixtures for the figwig test suite: bigWig files written once per session.

- `dense_bw`: step-function floats on three chromosomes, with gaps.
- `sparse_bw`: integer counts at scattered positions, the way read-count
  tracks look.
- `deep_bw`: enough intervals on one chromosome that the data index has more
  than one level.

Each fixture returns (path, chroms), where chroms maps each name to its
length.

`batching` runs a test at the default batch size and at very small ones.
"""

import numpy
import pytest

import figwig.bigwig

from .writers import write_bigwig


def _step_entries(rng, chroms, gap_p, max_len, values):
	entries = []
	for name, length in chroms.items():
		position = int(rng.integers(0, 50))
		while position < length:
			end = min(length, position + int(rng.integers(1, max_len)))
			entries.append((name, position, end, float(values())))
			position = end + int(rng.geometric(gap_p)) - 1
	return entries


@pytest.fixture(params=[(256, 2 ** 20), (3, 2 ** 20), (1, 64)],
	ids=['default_batches', 'small_batches', 'tiny_batches'])
def batching(request, monkeypatch):
	"""Split each read into batches of this many blocks, with this buffer.

	`dense_bw` and `sparse_bw` hold fewer than 256 blocks, so at the default
	batch size a read of them is one batch on one thread. The small sizes are
	what reach the thread pool and the batch boundaries, and a 64-byte buffer
	sends every compressed block to the zlib.decompress fallback.
	"""

	batch_blocks, max_block_bytes = request.param
	monkeypatch.setattr(figwig.bigwig, '_BATCH_BLOCKS', batch_blocks)
	monkeypatch.setattr(figwig.bigwig, '_MAX_BLOCK_BYTES', max_block_bytes)
	return request.param


@pytest.fixture(scope='session')
def dense_bw(tmp_path_factory):
	rng = numpy.random.default_rng(0)
	chroms = {'chr1': 250_000, 'chr2': 120_013, 'chrM': 16_569}
	entries = _step_entries(rng, chroms, 0.6, 40, lambda: rng.normal(0, 3))
	path = tmp_path_factory.mktemp('bw') / 'dense.bw'
	write_bigwig(path, chroms, entries)
	return str(path), chroms


@pytest.fixture(scope='session')
def sparse_bw(tmp_path_factory):
	rng = numpy.random.default_rng(1)
	chroms = {'chr1': 400_000, 'chr2': 300_000, 'chrX': 200_000}
	entries = _step_entries(rng, chroms, 0.01, 3, lambda: rng.poisson(3) + 1)
	path = tmp_path_factory.mktemp('bw') / 'sparse.bw'
	write_bigwig(path, chroms, entries)
	return str(path), chroms


@pytest.fixture(scope='session')
def deep_bw(tmp_path_factory):
	rng = numpy.random.default_rng(2)
	chroms = {'chr1': 20_000_000}
	entries = _step_entries(rng, chroms, 0.05, 60, lambda: rng.normal(5, 2))
	path = tmp_path_factory.mktemp('bw') / 'deep.bw'
	write_bigwig(path, chroms, entries)
	return str(path), chroms
