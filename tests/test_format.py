# test_format.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The figwig file format against its specification, docs/format.rst.

The files of `tests/format_fixtures.py`, written by `FigWigWriter` in
every layout the format has, are read back by `tests/format_reference.py`,
a reader written from the specification alone, and every value, table and
index record it gives is compared with `FigWigReader`'s.
"""

import json
import struct

import numpy
import pytest

import figwig

from figwig import FigWigReader
from figwig.figwig import _HEADER

from . import format_reference as spec
from .format_fixtures import TILE
from .format_fixtures import write_all


@pytest.fixture(scope='module')
def files(tmp_path_factory):
	return write_all(tmp_path_factory.mktemp('format'))


def identical(a, b):
	a, b = numpy.asarray(a, dtype=numpy.float64), numpy.asarray(b,
		dtype=numpy.float64)
	return a.shape == b.shape and numpy.array_equal(a, b, equal_nan=True) and \
		numpy.array_equal(numpy.signbit(a), numpy.signbit(b))


def test_header_layout_matches_the_specification():
	"""The specification's header table is the package's: 128 bytes, the
	fields in this order at these offsets, and the CRC field at byte 104."""

	assert spec.HEADER.size == spec.HEADER_SIZE == _HEADER.itemsize == 128
	offsets = {}
	at = 0
	for name, code in zip(spec.HEADER_FIELDS, ['I', 'H', 'H', 'I', 'H', 'H',
			'I', 'I', 'I', 'I', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'Q', 'I', 'I',
			'I', '20s']):
		offsets[name] = at
		at += struct.calcsize('<' + code)
	assert at == 128
	assert offsets == {name: _HEADER.fields[name][1] for name in
		spec.HEADER_FIELDS}
	assert offsets['header_crc'] == spec.HEADER_CRC_AT
	assert spec.MAGIC == int.from_bytes(b'FIGW', 'little')
	assert spec.RUN.itemsize == 32 and spec.BLOCK.itemsize == 40
	assert spec.STATS.itemsize == 32


@pytest.mark.parametrize('name', ['zlib', 'stored', 'tracks', 'tracks_zlib',
	'appended', 'merged', 'blocks', 'entries'])
def test_reference_reader_agrees_with_figwig(files, name):
	"""Every table, index record, summary, zoom level and value the
	specification's reader finds is what FigWigReader finds."""

	path = files[name]
	file = spec.File(path)
	reader = FigWigReader(path, verify=True)
	header = file.header

	assert header.tile == reader.tile and header.order == reader.order
	assert header.codec == reader._codec and header.level == reader._level
	assert header.optional == reader._optional and header.length == \
		reader._file_size
	assert file.chrom_sizes == reader.chrom_sizes
	assert file.stats.tobytes() == reader.stats.tobytes()
	assert file.metadata == reader._metadata
	assert [t['name'] for t in file.tracks] == [t['name'] for t in reader.tracks]
	assert [m for m in file.group_tracks] == [g.tolist() for g in reader._groups]

	for g, group in enumerate(file.groups):
		if header.order == 'tracks':
			theirs = reader._group_index(g)
			assert group.blocks.tobytes() == theirs.tobytes()
		else:
			kinds, sizes, raws, offsets, crcs = reader._group_index(g)
			live = group.live['tile']
			assert live.tolist() == numpy.flatnonzero(kinds).tolist()
			for name_, theirs in (('kind', kinds), ('size', sizes), ('raw', raws),
					('crc', crcs)):
				assert group.live[name_].tolist() == theirs[live].tolist()
			assert group.offsets.tolist() == offsets[live].tolist()
			assert group.runs.tobytes() == reader._runs[g].tobytes()
		summary = reader._tile_sums_of(g)
		if summary is None:
			assert group.summary is None
		else:
			assert group.summary[0].tolist() == summary[0].tolist()
			assert group.summary[1].tobytes() == summary[1].tobytes()
		levels = reader._zoom_levels_of(g)
		assert len(group.zooms) == len(levels)
		for (factor, bins, sums), (f2, b2, s2) in zip(group.zooms, levels):
			assert factor == f2 and bins.tolist() == b2.tolist()
			assert sums.tobytes() == s2.tobytes()

	dense = file.dense()
	for chrom, length in file.chrom_sizes.items():
		theirs = reader.read(chrom, [0], length, dtype='float64')[0]
		assert identical(dense[chrom], theirs)


def test_every_kind_and_element_type_is_covered(files):
	"""The fixtures exercise every chunk kind, every element code, chunks
	stored as they are inside a compressed file, several runs per group,
	track-order blocks of more than one per chromosome, and dense, sparse
	and runs chunks written from entries."""

	seen = set()
	stored = set()
	runs_per_group = []
	for name in ['zlib', 'stored', 'appended', 'merged']:
		file = spec.File(files[name])
		for group in file.groups:
			kinds = group.live['kind']
			seen |= {(int(k & 7), int((k >> 3) & 15)) for k in kinds}
			stored |= {bool(k & 128) for k in kinds}
			runs_per_group.append(len(group.runs))
	kinds_seen = {kind for kind, _ in seen}
	codes_seen = {code for _, code in seen}
	assert kinds_seen == {spec.KIND_DENSE, spec.KIND_SPARSE, spec.KIND_MATRIX,
		spec.KIND_RUNS}
	assert codes_seen == set(range(9))
	assert stored == {True, False}
	assert max(runs_per_group) > 1
	# More than one element type per kind, and matrix chunks in both codecs.
	by_kind = {}
	for kind, code in seen:
		by_kind.setdefault(kind, set()).add(code)
	assert len(by_kind[spec.KIND_DENSE]) > 1 and len(by_kind[spec.KIND_RUNS]) > 1
	assert spec.CODE_ONE in by_kind[spec.KIND_MATRIX] and 0 in by_kind[
		spec.KIND_MATRIX]
	for name in ['zlib', 'stored']:
		file = spec.File(files[name])
		assert spec.KIND_MATRIX in {int(k & 7) for group in file.groups for k in
			group.live['kind']}

	blocks = spec.File(files['blocks'])
	assert len(blocks.groups[0].blocks) > 1
	assert (blocks.groups[0].blocks['n'] == spec.BLOCK_ENTRIES).any()
	codes = {int((k >> 3) & 15) for group in blocks.groups for k in
		group.blocks['kind']}
	assert len(codes) > 1 and spec.CODE_ONE in codes

	entries = spec.File(files['entries'])
	assert {spec.KIND_DENSE, spec.KIND_SPARSE, spec.KIND_RUNS} <= {int(k & 7)
		for group in entries.groups for k in group.live['kind']}


def test_live_tiles_are_the_nonzero_tiles(files):
	"""A tile absent from a group's live list holds only zeros, and every
	live chunk holds a nonzero value, so the specification's "zero kind"
	is exactly the tiles the index leaves out."""

	file = spec.File(files['zlib'])
	dense = file.dense()
	arrays = list(dense.values())
	for g, group in enumerate(file.groups):
		members = file.group_tracks[g]
		live = set(group.live['tile'].tolist())
		for tid, (chrom, length) in enumerate(file.chrom_sizes.items()):
			for local in range((length + TILE - 1) // TILE):
				values = arrays[tid][members, local * TILE:(local + 1) * TILE]
				nonzero = bool((values != 0).any())    # NaN counts as nonzero
				assert nonzero == (int(file.tile_offsets[tid]) + local in live)


def test_summaries_are_the_tile_sums(files):
	"""A group's summary lists the tiles whose sum is not zero with each
	track's float32 sum over the tile, NaN bases left out, and each zoom
	level sums those over 4**k tiles, bins never crossing a chromosome."""

	for name in ['zlib', 'blocks']:
		file = spec.File(files[name])
		dense = file.dense()
		arrays = list(dense.values())
		for g, group in enumerate(file.groups):
			members = file.group_tracks[g]
			tiles, sums = group.summary
			tile = file.header.tile
			expected = numpy.zeros((file.n_tiles, len(members)), dtype=numpy.float32)
			for tid, length in enumerate(file.chrom_sizes.values()):
				for local in range((length + tile - 1) // tile):
					block = arrays[tid][members, local * tile:(local + 1) * tile]
					block = numpy.where(numpy.isnan(block), 0.0, block)
					expected[int(file.tile_offsets[tid]) + local] = block.sum(axis=1)
			listed = numpy.zeros_like(expected)
			listed[tiles] = sums
			assert numpy.array_equal(listed, expected)
			assert set(tiles.tolist()) == set(numpy.flatnonzero(expected.any(
				axis=1)).tolist())
			for factor, bins, zsums in group.zooms:
				n_tiles = numpy.diff(file.tile_offsets)
				bin_starts = numpy.concatenate([[0], numpy.cumsum(-(-n_tiles //
					factor))])
				zexpected = numpy.zeros((int(bin_starts[-1]), len(members)))
				for tid in range(len(n_tiles)):
					for local in range(int(n_tiles[tid])):
						zexpected[int(bin_starts[tid]) + local // factor] += expected[
							int(file.tile_offsets[tid]) + local].astype(numpy.float64)
				zlisted = numpy.zeros_like(zexpected)
				zlisted[bins] = zsums
				assert numpy.array_equal(zlisted, zexpected)


def test_appended_file_keeps_dead_bytes_and_chunks_in_place(files):
	"""An append leaves the original's chunks at their offsets and its old
	tables and index as bytes no run or record points to; a merge into a
	new file drops them."""

	before = spec.File(files['zlib'])
	after = spec.File(files['appended'])
	merged = spec.File(files['merged'])
	for g in range(len(before.groups)):
		assert after.groups[g].offsets.tolist() == before.groups[g].offsets.tolist()
	assert after.header.length > before.header.length
	assert after.directory[-1] > before.header.length
	assert [t['name'] for t in after.tracks][-1] == 'late'
	assert merged.header.length < after.header.length
	assert identical(merged.dense()['chr1'], after.dense()['chr1'])


def test_metadata_is_the_json_the_specification_describes(files):
	file = spec.File(files['zlib'])
	raw = file.bytes[file.header.metadata:file.header.metadata +
		file.header.metadata_length]
	document = json.loads(raw)
	assert list(document) == ['figwig', 'genome', 'attributes', 'order',
		'tracks']
	assert document['figwig'] == figwig.__version__
	assert document['genome'] == 'toy' and document['order'] == 'positions'
	assert document['attributes'] == {'made': 'test_format', 'n': [1, 2]}
	assert document['tracks'][11] == {'name': 'signal', 'dtype': 'float32',
		'assay': 'ATAC', 'group': 8}
	assert raw == json.dumps(document, allow_nan=False).encode()


@pytest.mark.parametrize('where', ['header', 'tables', 'record', 'chunk',
	'summary_stream', 'zoom_stream'])
def test_corruption_is_refused_by_both_readers(files, tmp_path, where):
	"""A flipped byte anywhere a CRC-32 or a zlib checksum covers is
	refused by the specification's reader and by FigWigReader alike."""

	source = spec.File(files['zlib'])
	data = bytearray(source.bytes)
	header = source.header
	group = source.groups[0]
	if where == 'header':
		at = 20                                    # inside n_chroms
	elif where == 'tables':
		at = header.chroms + 2                     # the first chromosome's name
	elif where == 'record':
		at = int(source.directory[0]) + 8          # the first run's first tile
	elif where == 'chunk':
		at = int(group.offsets[0]) + 2
	elif where == 'summary_stream':
		at = _summary_stream_at(source, 0) + 1
	else:
		at = _zoom_stream_at(source, 0) + 1
	data[at] ^= 0x40
	path = tmp_path / (where + '.fw')
	path.write_bytes(bytes(data))

	with pytest.raises(spec.FormatError):
		file = spec.File(path)
		file.dense()
	with pytest.raises(ValueError):
		reader = FigWigReader(path, verify=True)
		for g in range(len(reader._groups)):
			reader._group_index(g)
			reader._tile_sums_of(g)
			reader._zoom_levels_of(g)
		reader.read('chr1', [0], 300)


def _summary_stream_at(file, g):
	at = int(file.directory[g]) + 8
	n_runs = len(file.groups[g].runs)
	at += n_runs * 32
	n_live, n_bytes = struct.unpack_from('<II', file.bytes, at)
	return at + 8 + n_bytes + 12


def _zoom_stream_at(file, g):
	at = _summary_stream_at(file, g)
	n_sum, n_bytes = struct.unpack_from('<II', file.bytes, at - 12)
	return at + n_bytes + 4 + 16
