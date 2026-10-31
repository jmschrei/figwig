"""Writer-to-reader regional summary queries, with independent v1/raw oracles."""

import pickle
import shutil
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy
import pytest

import figwig.figwig as runtime
from figwig import FigWigReader, FigWigWriter
from figwig import _summary_index

from . import format_reference as v1
from .summary_reference import read_index
from .test_summary_index import CHROMS
from .test_summary_index import pair as pair
from .test_summary_index import write


@pytest.fixture(scope='module')
def oracle(pair):
	return v1.File(pair[0])


def expected(file, ids, names, level):
	width = file.header.tile * 4 ** level
	counts = [-(-length // width) for length in file.chrom_sizes.values()]
	offsets = numpy.concatenate([[0], numpy.cumsum(counts)])
	out = numpy.zeros((file.header.n_tracks, int(offsets[-1])),
		dtype=numpy.float64 if level else numpy.float32)
	for members, group in zip(file.group_tracks, file.groups):
		bins, sums = group.zooms[level - 1][1:] if level else group.summary
		out[numpy.ix_(members, bins)] = sums.T
	pieces = []
	for name in names:
		tid = list(file.chrom_sizes).index(name)
		pieces.append(out[numpy.ix_(ids, numpy.arange(offsets[tid], offsets[tid + 1]))])
	return numpy.concatenate(pieces, axis=1)


@pytest.mark.parametrize('level', range(4))
def test_writer_to_regional_reader_keeps_exact_values_and_selection(pair, oracle, level):
	reader = FigWigReader(pair[1], verify=True)
	ids, names = [3, 0, 3, 6, 5], ['chrB', 'empty', 'chrB']
	actual = reader.tile_sums(ids, names, level)
	wanted = expected(oracle, ids, names, level)
	assert actual.dtype == wanted.dtype
	assert actual.tobytes() == wanted.tobytes()
	assert reader.tile_sums('track3', 'chrB', level).tobytes() == \
		expected(oracle, [3], ['chrB'], level)[0].tobytes()
	# Catalog discovery skips the full raw index and unrelated zoom payloads.
	assert isinstance(reader._summary_restart, _summary_index.SummaryIndexReader)
	assert all(index is None for index in reader._index)
	assert all(levels is None for levels in reader._zoom_levels)


def observe_reads(monkeypatch):
	reads, inflated = [], []
	pread, inflate = runtime._pread, _summary_index._inflate

	def read(fd, size, offset):
		reads.append((offset, size))
		return pread(fd, size, offset)

	def decode(data, size, raw=False):
		inflated.append((size, raw))
		return inflate(data, size, raw)

	monkeypatch.setattr(runtime, '_pread', read)
	monkeypatch.setattr(_summary_index, '_inflate', decode)
	return reads, inflated


def test_small_chromosome_reads_bounded_pages_and_fragments(pair, oracle, monkeypatch):
	verified = read_index(pair[1])
	reader = FigWigReader(pair[1])
	reads, inflated = observe_reads(monkeypatch)
	actual = reader.tile_sums('track0', 'chrB')
	assert actual.tobytes() == expected(oracle, [0], ['chrB'], 0)[0].tobytes()
	assert sum(size for _, size in reads) < 16000
	assert all(size <= 32768 and raw for size, raw in inflated)
	ref = verified['groups'][0]['levels'][0]
	assert (ref['source_offset'], ref['source_bytes']) not in reads


def test_whole_genome_keeps_complete_stream_checks(pair, oracle, monkeypatch):
	reader = FigWigReader(pair[1])
	_, inflated = observe_reads(monkeypatch)
	assert reader.tile_sums().tobytes() == expected(oracle, list(range(8)),
		list(CHROMS), 0).tobytes()
	assert reader._summary_restart is None and not inflated


@pytest.fixture(scope='module', params=['positions', 'tracks'])
def counts(tmp_path_factory, request):
	root = tmp_path_factory.mktemp('summary-counts-' + request.param)
	chroms = {'chrA': 16 * 32768 + 7, 'chrB': 643, 'empty': 263}
	signals = {}
	for name, length in chroms.items():
		x = numpy.arange(length, dtype=numpy.int32)
		values = numpy.stack([x % 11, x % 7 - 3, x % 17, numpy.full(length,
			2 ** 24 + 1, dtype=numpy.int32)])
		if name == 'chrA':
			values[:, 16384 * 16:18432 * 16] = 0
		if name == 'empty':
			values[:] = 0
		signals[name] = values
	paths = []
	for enabled in [False, True]:
		path = root / ('indexed.fw' if enabled else 'ordinary.fw')
		with FigWigWriter(path, chroms, [{'name': 'c{}'.format(i), 'dtype':
				'int32'} for i in range(4)], tile=16, groups=4, order=request.param,
				engine='zlib', n_jobs=1, zooms=2, summary_index=enabled) as writer:
			for chrom, values in signals.items():
				writer.write(chrom, [0], values[None])
		paths.append(path)
	return paths, signals


@pytest.mark.parametrize('nan', ['propagate', 'skip', 'count'])
def test_interval_sums_use_index_without_changing_exact_semantics(counts, monkeypatch, nan):
	paths, signals = counts
	names = ['chrA', 'chrB', 'chrA', 'chrA', 'empty', 'chrB']
	starts = numpy.array([17, 3, 31000 * 16 + 3, 16385 * 16 + 1, 7, 600])
	ends = numpy.array([16 * 51 + 7, 510, 31040 * 16 + 13, 16400 * 16 + 9, 230, 650])
	ids = [1, 0, 1, 2]
	wanted = numpy.array([[min(int(b), signals[name].shape[1]) - int(a)] * len(ids)
		if nan == 'count' else [signals[name][i, a:b].sum() for i in ids]
		for name, a, b in zip(names, starts, ends)], dtype=numpy.float64)
	reader = FigWigReader(paths[1], verify=True)
	legacy = FigWigReader(paths[0]).sums(names, starts, ends, tracks=ids,
		n_jobs=1, nan=nan, dtype='float64')
	reads, inflated = observe_reads(monkeypatch)
	actual = reader.sums(names, starts, ends, tracks=ids, n_jobs=1, nan=nan, dtype='float64')
	assert actual.tobytes() == wanted.tobytes()
	assert actual.tobytes() == legacy.tobytes()
	if nan != 'count':
		assert inflated and all(raw for _, raw in inflated)
		assert sum(size for size, _ in inflated) < 131072
		# The two main-chromosome intervals are far apart. No middle pages
		# or fragments are needed; raw reads cover only interval boundaries.
		assert sum(size for _, size in reads) < 800000
	else:
		assert not inflated


def test_aligned_integer_intervals_return_summary_interior_without_raw_read(counts, monkeypatch):
	paths, signals = counts
	reader = FigWigReader(paths[1])

	def unexpected(*args, **kwargs):
		raise AssertionError('aligned, exact whole tiles must not decode raw values')

	monkeypatch.setattr(reader, '_read', unexpected)
	starts, ends = [16 * 17, 16 * 30001], [16 * 43, 16 * 30025]
	actual = reader.sums('chrA', starts, ends, tracks=[0, 1], dtype='float64')
	wanted = numpy.array([[signals['chrA'][i, a:b].sum() for i in [0, 1]]
		for a, b in zip(starts, ends)], dtype=numpy.float64)
	assert actual.tobytes() == wanted.tobytes()


def test_rounded_integer_summaries_fall_back_to_exact_raw_values(counts):
	paths, signals = counts
	starts, ends = [17, 16 * 41], [16 * 39 + 3, 16 * 65]
	actual = FigWigReader(paths[1]).sums('chrA', starts, ends, tracks=3, dtype='float64')
	wanted = numpy.array([signals['chrA'][3, a:b].sum() for a, b in zip(starts, ends)],
		dtype=numpy.float64)
	assert actual.tobytes() == wanted.tobytes()


@pytest.mark.parametrize('n_jobs', [0, 'bad', 1.5])
def test_aligned_summary_interior_still_validates_threads(counts, n_jobs):
	with pytest.raises((TypeError, ValueError), match='n_jobs'):
		FigWigReader(counts[0][1]).sums('chrA', [16 * 17], [16 * 43],
			tracks=0, n_jobs=n_jobs)


def test_empty_window_uses_no_fragments_or_raw_chunks(counts, monkeypatch):
	reader = FigWigReader(counts[0][1])
	reads, inflated = observe_reads(monkeypatch)
	assert not reader.sums('chrA', [16400 * 16], [16450 * 16], tracks=[0, 1]).any()
	assert not inflated
	assert all(index is None for index in reader._index)
	assert sum(size for _, size in reads) < 10000


@pytest.mark.parametrize('where', ['root', 'locator', 'catalog', 'page', 'fragment'])
def test_production_reader_rejects_corruption_in_requested_index(pair, tmp_path, where):
	verified = read_index(pair[1])
	root = verified['offset']
	data = bytearray(pair[1].read_bytes())
	ref = verified['groups'][0]['levels'][0]
	window = next(w for w in ref['windows'] if w['ids'][-1] >= 12000)
	page = ref['index_offset'] + window['window'] // 102 * 4096
	at = {'root': root + 40, 'locator': root + 256 + 16,
		'catalog': struct.unpack_from('<Q', data, root + 256)[0] + 16,
		'page': page + 20, 'fragment': ref['source_offset'] +
			window['sums_relative_offset']}[where]
	data[at] ^= 1
	path = tmp_path / 'corrupt.fw'
	path.write_bytes(data)
	with pytest.raises(_summary_index.SummaryIndexError):
		FigWigReader(path).tile_sums('track0', 'chrB')


def test_integer_sum_does_not_hide_index_corruption(counts, tmp_path):
	path = tmp_path / 'corrupt-counts.fw'
	shutil.copyfile(counts[0][1], path)
	verified = read_index(path)
	data = bytearray(path.read_bytes())
	data[verified['offset'] + 40] ^= 1
	path.write_bytes(data)
	with pytest.raises(_summary_index.SummaryIndexError):
		FigWigReader(path).sums('chrA', [17], [16 * 40 + 3], tracks=0)


def repair_crc(data, at, size, field):
	struct.pack_into('<I', data, at + field, 0)
	struct.pack_into('<I', data, at + field, zlib.crc32(data[at:at + size]))


@pytest.mark.parametrize('where', ['catalog_offset', 'source_offset', 'page_offset',
	'fragment_offset', 'fragment_size', 'copied_sums', 'row_rank', 'raw_crc', 'dictionary'])
def test_crc_correct_malformed_indices_fail_before_unbounded_reads(pair, tmp_path,
	monkeypatch, where):
	verified = read_index(pair[1])
	data = bytearray(pair[1].read_bytes())
	locator = verified['offset'] + 256
	catalog = struct.unpack_from('<Q', data, locator)[0]
	count = struct.unpack_from('<I', data, locator + 8)[0]
	ref = verified['groups'][0]['levels'][0]
	window = next(w for w in ref['windows'] if w['ids'][-1] >= 12000)
	page = ref['index_offset'] + window['window'] // 102 * 4096
	page_size = 16 + 40 * struct.unpack_from('<I', data, page + 8)[0]
	slot = page + 16 + window['window'] % 102 * 40
	if where == 'catalog_offset':
		struct.pack_into('<Q', data, locator, 2 ** 64 - 1)
		repair_crc(data, locator, 32, 24)
	elif where in ('source_offset', 'page_offset'):
		struct.pack_into('<Q', data, catalog + (8 if where == 'page_offset' else 0),
			2 ** 64 - 1)
		struct.pack_into('<I', data, locator + 12, zlib.crc32(data[catalog:catalog + count * 32]))
		repair_crc(data, locator, 32, 24)
	elif where == 'dictionary':
		cmf = data[ref['source_offset']]
		flg = next(i for i in range(256) if i & 32 and (cmf * 256 + i) % 31 == 0)
		data[ref['source_offset'] + 1] = flg
	else:
		if where in ('fragment_offset', 'fragment_size'):
			struct.pack_into('<I', data, slot + (8 if where == 'fragment_offset' else 12),
				2 ** 32 - 1)
		elif where == 'copied_sums':
			data[slot + 24:slot + 40] = data[slot - 16:slot]
		elif where == 'row_rank':
			struct.pack_into('<I', data, slot + 4, 0)
		else:
			data[slot + 36] ^= 1
		repair_crc(data, page, page_size, 12)
	path = tmp_path / 'checked-malformed.fw'
	path.write_bytes(data)
	reader = FigWigReader(path)
	reads, _ = observe_reads(monkeypatch)
	with pytest.raises(_summary_index.SummaryIndexError):
		reader.tile_sums(0, 'chrB')
	assert all(size <= 65536 for _, size in reads)


def test_restart_decompression_caps_output_before_accepting_it(pair, tmp_path, monkeypatch):
	verified = read_index(pair[1])
	ref = verified['groups'][0]['levels'][0]
	data = bytearray(pair[1].read_bytes())
	compressor = zlib.compressobj(wbits=-15)
	bomb = compressor.compress(bytes(1 << 20)) + compressor.flush(zlib.Z_FULL_FLUSH)
	window = next(w for w in ref['windows'] if w['sums_compressed_bytes'] >= len(bomb))
	# Keep the bad output inside the original source extent and the window's
	# compressed-size bound, so the bounded inflater itself must reject it.
	page = ref['index_offset'] + window['window'] // 102 * 4096
	slot = page + 16 + window['window'] % 102 * 40
	raw_size = window['n'] * 4 * verified['groups'][0]['tracks']
	assert len(bomb) <= 2 * raw_size + 64
	offset = ref['source_offset'] + window['sums_relative_offset']
	data[offset:offset + len(bomb)] = bomb
	struct.pack_into('<II', data, slot + 28, len(bomb), zlib.crc32(bomb))
	page_size = 16 + 40 * struct.unpack_from('<I', data, page + 8)[0]
	repair_crc(data, page, page_size, 12)
	path = tmp_path / 'bounded-deflate.fw'
	path.write_bytes(data)
	reader = FigWigReader(path)
	first = int(window['ids'][0])
	decompressobj = zlib.decompressobj
	limits = []

	class BoundedDecoder:
		def __init__(self, *args):
			self.decoder = decompressobj(*args)

		def decompress(self, encoded, max_length=0):
			# Assert before calling zlib: an unbounded decode followed by a
			# length check must fail this test without allocating its output.
			assert 0 < max_length <= max(raw_size, 4 * window['n']) + 1
			limits.append(max_length)
			result = self.decoder.decompress(encoded, max_length)
			assert len(result) <= max_length
			return result

		def __getattr__(self, name):
			return getattr(self.decoder, name)

	monkeypatch.setattr(_summary_index.zlib, 'decompressobj', BoundedDecoder)
	with pytest.raises(_summary_index.SummaryIndexError):
		reader._summary_bins(numpy.array([0]), 0, [(first, first + 1)], reader._n_tiles)
	assert limits == [4 * window['n'] + 1, raw_size + 1]


def test_stale_index_uses_ordinary_summary_stream(pair, oracle, tmp_path):
	data = bytearray(pair[1].read_bytes())
	at = read_index(pair[1])['offset']
	data[at + 64 + 28] ^= 1
	struct.pack_into('<I', data, at + 12, 0)
	struct.pack_into('<I', data, at + 12, zlib.crc32(data[at:at + 256]))
	path = tmp_path / 'stale.fw'
	path.write_bytes(data)
	reader = FigWigReader(path)
	assert reader.tile_sums(0, 'chrB').tobytes() == expected(oracle, [0], ['chrB'], 0)[0].tobytes()
	assert reader._summary_restart is False


@pytest.mark.parametrize('version', [1, 9])
@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_unknown_extension_versions_keep_all_v1_values_and_zooms(tmp_path, monkeypatch, version, order):
	with monkeypatch.context() as previous:
		previous.setattr(_summary_index, 'VERSION', version)
		previous.setattr(runtime, '_SUMMARY_INDEX_VERSION', version)
		path = write(tmp_path / 'unsupported.fw', order, enabled=True)
	file = v1.File(path)
	root = file.metadata['org.figwig.summary-restarts']['offset']
	assert struct.unpack_from('<H', file.bytes, root + 8)[0] == version
	reader = FigWigReader(path, verify=True)
	for level in range(4):
		assert reader.tile_sums(chroms='chrB', level=level).tobytes() == \
			expected(file, list(range(8)), ['chrB'], level).tobytes()
	assert reader._summary_restart is False


def test_pickle_threaded_reads_and_replacement_keep_file_ownership(pair, oracle, tmp_path):
	path = tmp_path / 'owned.fw'
	shutil.copyfile(pair[1], path)
	reader = FigWigReader(path)
	wanted = expected(oracle, [0], ['chrB'], 0)[0]
	with ThreadPoolExecutor(4) as pool:
		results = list(pool.map(lambda _: reader.tile_sums(0, 'chrB'), range(12)))
	assert all(result.tobytes() == wanted.tobytes() for result in results)
	results[0][:] = -1
	assert reader.tile_sums(0, 'chrB').tobytes() == wanted.tobytes()
	copy = pickle.loads(pickle.dumps(reader))
	assert copy._summary_restart is None
	assert copy.tile_sums(0, 'chrB').tobytes() == wanted.tobytes()
	replacement = tmp_path / 'replacement.fw'
	shutil.copyfile(pair[1], replacement)
	replacement.replace(path)
	with pytest.raises(ValueError, match='replaced or rewritten'):
		reader.tile_sums(0, 'chrB')
