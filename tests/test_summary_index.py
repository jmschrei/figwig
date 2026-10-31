"""The optional directory against unchanged v1 readers and an independent oracle."""

import io
import shutil
import struct
import zlib

import kaitaistruct
import numpy
import pytest

from figwig import FigWigReader
from figwig import FigWigWriter
from figwig import write_figwig
from figwig.merge import _merge
from figwig.reorder import _reorder
from figwig import _summary_index

from . import format_reference as v1
from .kaitai_figwig import Figwig
from .summary_reference import read_index
from .test_kaitai import compare as compare_kaitai


KEY = 'org.figwig.summary-restarts'
CHROMS = {'chrA': 192000, 'chrB': 603, 'empty': 267}
TRACKS = [{'name': 'track{}'.format(i), 'dtype': 'float64' if i == 3 else
	'float32'} for i in range(8)]


def signals():
	rng = numpy.random.default_rng(1701)
	for chrom, length in CHROMS.items():
		tiles = -(-length // 16)
		rows = numpy.zeros((8, tiles))
		rows[0] = rng.normal(2, 1, tiles).astype(numpy.float32)
		rows[1] = numpy.arange(tiles) % 10 + 1
		rows[3] = 2 ** 24 + 1 + numpy.arange(tiles) % 3
		rows[4] = rng.uniform(0, 10, tiles).astype(numpy.float32)
		rows[4, ::37] = numpy.nan
		rows[6] = numpy.nan
		rows[7] = 0.25 + numpy.arange(tiles) % 7
		values = numpy.repeat(rows, 16, axis=1)[:, :length]
		values[2, ::2] = 1
		values[2, 1::2] = -1
		if chrom == 'chrA':
			values[:, 4096 * 16:6144 * 16] = 0
		elif chrom == 'empty':
			values[:] = 0
		yield chrom, values


def write(path, order='positions', enabled=False, n_jobs=2):
	with FigWigWriter(path, CHROMS, TRACKS, tile=16, groups=8, order=order,
			level=6, engine='zlib', zooms=3, n_jobs=n_jobs,
			summary_index=enabled) as writer:
		for chrom, values in signals():
			writer.write(chrom, [0], values[None])
	return path


@pytest.fixture(scope='module', params=['positions', 'tracks'])
def pair(tmp_path_factory, request):
	root = tmp_path_factory.mktemp('summary-' + request.param)
	return (write(root / 'ordinary.fw', request.param),
		write(root / 'indexed.fw', request.param, True))


def test_indexed_files_keep_every_legacy_value_and_summary(pair):
	ordinary, indexed = map(v1.File, pair)
	assert indexed.header.optional == 3 and indexed.header.version == 1
	assert indexed.header.zooms == 0 and not any(indexed.header.padding)
	assert read_index(pair[0]) is None
	verified = read_index(pair[1])
	assert any(level['window_bins'] for group in verified['groups'] for
		level in group['levels'])
	assert indexed.header.chroms - verified['offset'] <= 16 * 1024 ** 2
	for a, b in zip(ordinary.groups, indexed.groups):
		assert a.summary[0].tobytes() == b.summary[0].tobytes()
		assert a.summary[1].tobytes() == b.summary[1].tobytes()
		for x, y in zip(a.zooms, b.zooms):
			assert x[0] == y[0] and x[1].tobytes() == y[1].tobytes()
			assert x[2].tobytes() == y[2].tobytes()
	reader = FigWigReader(pair[1], verify=True)
	for chrom, values in signals():
		actual = reader.read(chrom, [0], values.shape[1], dtype='float64')[0]
		assert numpy.array_equal(actual, values, equal_nan=True)
		assert numpy.array_equal(numpy.signbit(actual), numpy.signbit(values))
	for g, group in enumerate(indexed.groups):
		ids, sums = reader._tile_sums_of(g)
		assert ids.tolist() == group.summary[0].tolist()
		assert sums.tobytes() == group.summary[1].tobytes()
		for actual, expected in zip(reader._zoom_levels_of(g), group.zooms):
			assert actual[0] == expected[0]
			assert actual[1].tolist() == expected[1].tolist()
			assert actual[2].tobytes() == expected[2].tobytes()
	parsed = Figwig(kaitaistruct.KaitaiStream(io.BytesIO(indexed.bytes)))
	compare_kaitai(parsed, indexed)


def test_global_window_can_span_chromosomes_without_changing_bins(pair):
	verified = read_index(pair[1])
	base = verified['groups'][0]['levels'][0]
	assert any(window['ids'][0] < 12000 <= window['ids'][-1] for
		window in base['windows'])
	assert any(window['ids'][0] >= 0 for window in base['windows'])
	# No descriptors or fragments describe the long all-zero middle gap.
	occupied = {window['window'] for window in base['windows']}
	for bin_id in range(4096, 6144, base['window_bins']):
		assert bin_id // base['window_bins'] not in occupied
	assert any(level['index_offset'] == 0 for group in verified['groups'] for
		level in group['levels'])


def test_empty_tracks_keep_available_empty_streams(pair):
	file = v1.File(pair[1])
	if file.header.order != 'tracks':
		return
	verified = read_index(pair[1])
	for group in [5]:
		for level in verified['groups'][group]['levels']:
			assert level['source_rows'] == 0 and level['source_bytes'] == 8
			assert level['window_bins'] == level['index_offset'] == 0
			assert level['windows'] is None
	# Track-order writing currently retains zero-sum rows for NaN-only tiles.
	# The accelerator must retain those IDs rather than infer spatial absence.
	missing = verified['groups'][6]['levels'][0]
	assert missing['source_rows'] > 0
	assert all(not window['sums'].any() for window in missing['windows'])


def test_disabled_flag_and_thread_counts_keep_deterministic_output(tmp_path):
	paths = [tmp_path / name for name in ['default.fw', 'disabled.fw',
		'one.fw', 'two.fw']]
	with FigWigWriter(paths[0], CHROMS, TRACKS, tile=16, groups=8,
			level=6, engine='zlib', zooms=3, n_jobs=2) as writer:
		for chrom, values in signals():
			writer.write(chrom, [0], values[None])
	write(paths[1])
	assert paths[0].read_bytes() == paths[1].read_bytes()
	write(paths[2], enabled=True, n_jobs=1)
	write(paths[3], enabled=True, n_jobs=2)
	assert paths[2].read_bytes() == paths[3].read_bytes()


@pytest.mark.parametrize('value', [None, 0, 1, 'yes', numpy.bool_(True)])
def test_flag_rejects_non_booleans_before_creating_files(tmp_path, value):
	path = tmp_path / 'invalid.fw'
	with pytest.raises(TypeError, match='summary_index'):
		FigWigWriter(path, {'chr1': 16}, ['a'], summary_index=value)
	assert not path.exists()


def test_append_rejection_and_default_invalidation(pair, tmp_path):
	path = tmp_path / 'append.fw'
	shutil.copyfile(pair[1], path)
	before = path.read_bytes()
	with pytest.raises(ValueError, match='append=True'):
		FigWigWriter(path, None, ['extra'], append=True, summary_index=True)
	assert path.read_bytes() == before
	locator = v1.File(path).metadata[KEY]
	with FigWigWriter(path, None, ['extra'], append=True, engine='zlib', n_jobs=1):
		pass
	after = v1.File(path)
	assert after.header.optional == 1 and after.metadata[KEY] == locator
	assert read_index(path) is None
	reader = FigWigReader(path, verify=True)
	assert len(reader.tracks) == 9
	assert numpy.array_equal(reader.read('chrB', [0], CHROMS['chrB'],
		tracks=['track0']), FigWigReader(pair[1]).read('chrB', [0],
		CHROMS['chrB'], tracks=['track0']), equal_nan=True)


def test_merge_and_reorder_safely_drop_the_extension(pair, tmp_path):
	merged, reordered = tmp_path / 'merge.fw', tmp_path / 'reorder.fw'
	_merge([pair[1]], merged)
	_reorder(pair[1], reordered, 'tracks' if v1.File(pair[1]).header.order ==
		'positions' else 'positions', n_jobs=1)
	for path in [merged, reordered]:
		file = v1.File(path)
		assert not file.header.optional & 2 and KEY not in file.metadata
		assert read_index(path) is None


def test_budget_and_global_summary_cutoff_keep_legacy_behavior(tmp_path, monkeypatch):
	monkeypatch.setattr(_summary_index, 'INDEX_BYTES', 512)
	path = write(tmp_path / 'budget.fw', enabled=True)
	file = v1.File(path)
	assert not file.header.optional & 2 and KEY not in file.metadata
	assert read_index(path) is None
	import figwig.figwig
	monkeypatch.setattr(figwig.figwig, '_SUMMARY_BYTES', 1)
	path = write(tmp_path / 'absent.fw', enabled=True)
	file = v1.File(path)
	assert all(group.summary is None and not group.zooms for group in file.groups)
	assert not file.header.optional & 2


def test_write_figwig_passes_through_the_option(tmp_path):
	path = tmp_path / 'convenience.fw'
	values = numpy.ones((1, 8, 160000), dtype=numpy.float32)
	write_figwig(path, {'chr1': 160000}, TRACKS, 'chr1', [0], values,
		tile=16, groups=8, n_jobs=1, engine='zlib', summary_index=True)
	assert read_index(path)['groups'][0]['levels'][0]['window_bins'] == 512


@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_summary_restarts_are_independent_of_raw_compression_codec(tmp_path, order):
	path = tmp_path / 'stored.fw'
	with FigWigWriter(path, {'chr1': 144000}, ['a', 'b'], tile=16, groups=2,
			order=order, level=0, n_jobs=1, summary_index=True) as writer:
		writer.write('chr1', [0], numpy.ones((1, 2, 144000), dtype=numpy.uint8))
	assert v1.File(path).header.codec == 0
	assert all(group['levels'][0]['window_bins'] for group in read_index(path)['groups'])
	assert FigWigReader(path, verify=True).tile_sums().shape == (2, 9000)


@pytest.mark.parametrize('where', ['root', 'locator', 'catalog', 'page',
	'row_rank', 'fragment_offset', 'fragment_crc', 'empty_window'])
def test_independent_verifier_rejects_malformed_directory(pair, tmp_path, where):
	file = v1.File(pair[1])
	verified = read_index(pair[1])
	root_at = verified['offset']
	data = bytearray(file.bytes)
	locator_at = root_at + 256
	catalog_at = struct.unpack_from('<Q', data, locator_at)[0]
	base = verified['groups'][0]['levels'][0]
	page_at = base['index_offset']
	if where in ['root', 'locator', 'catalog', 'page']:
		at = {'root': root_at + 40, 'locator': locator_at + 16,
			'catalog': catalog_at + 16, 'page': page_at + 16}[where]
		data[at] ^= 1
	else:
		window = base['windows'][0]['window']
		slot = page_at + 16 + window * 40
		if where == 'row_rank':
			struct.pack_into('<I', data, slot + 4, 1)
		elif where == 'fragment_offset':
			struct.pack_into('<I', data, slot + 8, 1)
		elif where == 'fragment_crc':
			data[slot + 16] ^= 1
		else:
			empty = 4096 // base['window_bins']
			page_at = base['index_offset'] + (empty // 102) * 4096
			slot = page_at + 16 + (empty % 102) * 40
			struct.pack_into('<I', data, slot + 4, 1)
		count = struct.unpack_from('<I', data, page_at + 8)[0]
		struct.pack_into('<I', data, page_at + 12, 0)
		struct.pack_into('<I', data, page_at + 12, zlib.crc32(
			data[page_at:page_at + 16 + count * 40]))
	path = tmp_path / (where + '.fw')
	path.write_bytes(data)
	with pytest.raises(v1.FormatError):
		read_index(path)


def test_stale_header_binding_falls_back(pair, tmp_path):
	file = v1.File(pair[1])
	at = file.metadata[KEY]['offset']
	data = bytearray(file.bytes)
	data[at + 64 + 28] ^= 1
	struct.pack_into('<I', data, at + 12, 0)
	struct.pack_into('<I', data, at + 12, zlib.crc32(data[at:at + 256]))
	path = tmp_path / 'stale.fw'
	path.write_bytes(data)
	assert read_index(path) is None


def test_failed_directory_finalization_keeps_previous_path(tmp_path, monkeypatch):
	path = tmp_path / 'previous.fw'
	previous = b'previous file remains until publication'
	path.write_bytes(previous)

	def fail(*args):
		raise RuntimeError('directory finalization failed')

	monkeypatch.setattr(_summary_index.SummaryIndex, 'serialize', fail)
	with pytest.raises(RuntimeError, match='finalization'):
		write(path, enabled=True)
	assert path.read_bytes() == previous
	assert list(tmp_path.iterdir()) == [path]
