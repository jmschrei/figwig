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
import figwig.figwig
import figwig._kernels
import figwig._summary_index

from .writers import write_bigwig
from .summary_files import Watch
from .summary_files import write_counts
from .summary_files import write_signal


def pytest_report_header(config):
	"""Which compression libraries loaded, since the kernels that call them
	run only where they do, and Python's zlib module is used elsewhere."""

	found = {False: 'not found', True: 'found'}
	return "figwig: zlib's uncompress {}, zlib's compress2 {}, libdeflate {}".format(
		found[figwig._kernels._zlib_uncompress() is not None],
		found[figwig._kernels._zlib_compress() is not None],
		found[figwig._kernels._libdeflate() is not None])


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


@pytest.fixture(scope='session')
def summary_signal(tmp_path_factory):
	"""{order: (ordinary, indexed)}: the signal tracks written once, without and
	with the regional summary index, in each order. `summary_pair` is one of the
	pairs, and the format and Kaitai tests read the indexed files with their own."""

	files = {}
	for order in ('positions', 'tracks'):
		root = tmp_path_factory.mktemp('summary-signal-' + order)
		files[order] = (write_signal(root / 'ordinary.fw', order, indexed=False),
			write_signal(root / 'indexed.fw', order))
	return files


@pytest.fixture(params=['positions', 'tracks'])
def summary_pair(summary_signal, request):
	"""(ordinary, indexed): the same signal tracks written without and with the
	regional summary index, in the order the test is run for."""

	return summary_signal[request.param]


@pytest.fixture(scope='session', params=['positions', 'tracks'])
def summary_counts(tmp_path_factory, request):
	"""(ordinary, indexed): four integer tracks, written the same way."""

	root = tmp_path_factory.mktemp('summary-counts-' + request.param)
	return (write_counts(root / 'ordinary.fw', request.param, indexed=False),
		write_counts(root / 'indexed.fw', request.param))


@pytest.fixture(scope='session', params=['positions', 'tracks'])
def summary_pages(tmp_path_factory, request):
	"""The signal tracks written with the index and windows of 8 bins, so that
	the summary has 3,132 windows in 31 descriptor pages, not the one or two a
	window of 2,048 bins gives a file this small."""

	root = tmp_path_factory.mktemp('summary-pages-' + request.param)
	with pytest.MonkeyPatch.context() as patch:
		patch.setattr(figwig._summary_index, 'MAX_WINDOW_BINS', 8)
		return write_signal(root / 'pages.fw', request.param)


@pytest.fixture
def small_levels_restart(monkeypatch):
	"""Levels of any size are restarted, so that the small files of a command's
	test get a regional summary index."""

	monkeypatch.setattr(figwig._summary_index, 'TINY_BYTES', 0)


@pytest.fixture
def small_windows(monkeypatch):
	"""Windows of 8 bins, and levels of any size restarted, so that the small
	files of a command's test have a summary of several windows and a read of one
	chromosome takes a few of them, as a read of a large file does."""

	monkeypatch.setattr(figwig._summary_index, 'MAX_WINDOW_BINS', 8)
	monkeypatch.setattr(figwig._summary_index, 'TINY_BYTES', 0)


@pytest.fixture
def watched(monkeypatch):
	"""A `Watch` of every pread of the figwig reader and every inflate of the
	summary index while the test runs."""

	watch = Watch()
	pread, inflate = figwig.figwig._pread, figwig._summary_index._inflate

	def counted_pread(fd, size, offset):
		watch.preads.append((offset, size))
		return pread(fd, size, offset)

	def counted_inflate(data, size, raw=False):
		watch.inflated.append((size, raw))
		return inflate(data, size, raw)

	monkeypatch.setattr(figwig.figwig, '_pread', counted_pread)
	monkeypatch.setattr(figwig._summary_index, '_inflate', counted_inflate)
	return watch
