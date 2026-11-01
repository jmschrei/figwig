# test_kaitai.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The Kaitai Struct definition of the format, docs/figwig.ksy, through
the Python parser compiled from it, tests/kaitai_figwig.py: every file of
tests/test_format.py parses, and what the parser finds is what the
specification's reader finds. When kaitai-struct-compiler is on the PATH,
the committed parser is checked to be the compiler's output."""

import io
import json
import pathlib
import re
import shutil
import subprocess
import sys
import zlib

import kaitaistruct
import numpy
import pytest

from . import format_reference as spec
from .format_fixtures import write_all
from .kaitai_figwig import Figwig
from .summary_reference import read_index


KSY = 'docs/figwig.ksy'
GENERATED = 'tests/kaitai_figwig.py'


@pytest.fixture(scope='module')
def files(tmp_path_factory, summary_signal):
	paths = write_all(tmp_path_factory.mktemp('kaitai'))
	paths['indexed'] = summary_signal['positions'][1]
	paths['indexed_tracks'] = summary_signal['tracks'][1]
	return paths


def regenerate(root, outdir):
	"""Compile the definition into `outdir` and return the parser's source
	as the committed file holds it: re-indented with tabs, one symbol per
	import line."""

	subprocess.run(['kaitai-struct-compiler', '-t', 'python', '--outdir',
		str(outdir), str(root / KSY)], check=True, capture_output=True)
	source = (outdir / 'figwig.py').read_text()
	source = source[source.index('import kaitaistruct'):]
	source = source.replace('from kaitaistruct import KaitaiStruct, '
		'KaitaiStream, BytesIO\n', 'from kaitaistruct import KaitaiStruct\n'
		'from kaitaistruct import KaitaiStream\nfrom kaitaistruct import BytesIO\n')
	lines = []
	for line in source.split('\n'):
		m = re.match(r'^( +)', line)
		if m:
			n = len(m.group(1))
			line = '\t' * (n // 4) + ' ' * (n % 4) + line[n:]
		lines.append(line)
	return '\n'.join(lines)


def chunks_of(parsed):
	"""Every chunk of a parsed position-order file, as (group, live index,
	tile, kind byte, decoded bytes), through the record's run list."""

	for g, record in enumerate(parsed.index.records):
		runs = record.chunk_runs
		while runs is not None:
			for chunk in runs.chunks.chunks:
				yield g, chunk.live_index, chunk.tile, chunk.kind_byte, chunk.inflated
			runs = runs.next


def decode(parsed, kind_byte, n_tracks, tile, data):
	"""The values of a chunk through the Kaitai chunk type, (n_tracks, tile)
	float64."""

	chunk = Figwig.Chunk(kind_byte, n_tracks, tile, kaitaistruct.KaitaiStream(
		io.BytesIO(data)))
	body = chunk.body
	out = numpy.zeros((n_tracks, tile))
	values = None if chunk.code == 7 else body.values.values
	if chunk.kind == 1:
		out[:] = numpy.array(values, dtype=numpy.float64).reshape(n_tracks, tile)
		return out
	if chunk.kind == 3:
		for j, slot in enumerate(body.tracks):
			for i in range(body.ptr[j], body.ptr[j + 1]):
				out[slot, body.offsets[i]] = 1.0 if values is None else values[i]
		return out
	d = 0
	for slot, track in enumerate(body.tracks):
		p = 0
		current, since = 0.0, 0
		for delta in track.deltas:
			p += delta.value
			value = 1.0 if values is None else values[d]
			if chunk.kind == 4:
				out[slot, since:p] = current
				current, since = value, p
			else:
				out[slot, p] = value
			d += 1
		if chunk.kind == 4:
			out[slot, since:] = current
	assert d == body.total
	return out


@pytest.mark.parametrize('name', ['zlib', 'stored', 'tracks', 'tracks_zlib',
	'appended', 'merged', 'blocks', 'entries', 'indexed', 'indexed_tracks'])
def test_kaitai_parser_agrees_with_the_reference_reader(files, name):
	path = files[name]
	file = spec.File(path)
	with open(path, 'rb') as handle:
		parsed = Figwig(kaitaistruct.KaitaiStream(handle))
		compare(parsed, file)


def compare(parsed, file):
	header = file.header
	h = parsed.header
	for field in ('version', 'flags', 'tile', 'codec', 'level', 'n_chroms',
			'n_tracks', 'n_groups', 'tables_crc', 'metadata_length', 'length',
			'required', 'optional', 'header_crc'):
		assert getattr(h, field) == getattr(header, field), field
	assert (h.chroms_offset, h.tracks_offset, h.index_offset, h.data_offset,
		h.metadata_offset, h.zooms_offset) == (header.chroms, header.tracks,
		header.index, header.data, header.metadata, header.zooms)
	assert h.track_order == (header.order == 'tracks') and h.has_zooms

	assert [(c.name, c.length) for c in parsed.chrom_table.entries] == list(
		file.chrom_sizes.items())
	assert [(s.sum, s.nonzero, s.min, s.max) for s in parsed.stats_table] == [
		tuple(float(r[k]) if k != 'nonzero' else int(r[k]) for k in ('sum',
		'nonzero', 'min', 'max')) for r in file.stats]
	assert json.loads(parsed.metadata) == file.metadata
	assert list(parsed.index.directory) == file.directory.tolist()

	for g, (record, group) in enumerate(zip(parsed.index.records, file.groups)):
		if header.order == 'tracks':
			assert record.num_entries == len(group.blocks)
			for k, (b, expected) in enumerate(zip(record.blocks, group.blocks)):
				assert (b.chrom, b.first, b.last, b.num_entries, b.offset,
					b.len_body, b.raw, b.kind, b.crc) == tuple(int(expected[k2]) for
					k2 in spec.BLOCK.names)
				body = b.body if header.codec == 0 else b.body_deflated
				positions, values = spec.decode_block(file.block(g, k), int(
					expected['kind']), int(expected['n']))
				assert list(body.positions) == positions.tolist()
				theirs = [1.0] * b.num_entries if (b.kind >> 3) & 15 == 7 else \
					list(body.values.values)
				assert numpy.array_equal(numpy.array(theirs, dtype=numpy.float64),
					values, equal_nan=True)
		else:
			assert record.num_entries == len(group.runs)
			assert [(r.first, r.last, r.offset, r.bytes) for r in record.runs] == [
				tuple(int(r[k]) for k in spec.RUN.names) for r in group.runs]
			live = record.live
			for name_, theirs in (('tile', live.tiles), ('kind', live.kinds),
					('size', live.sizes), ('raw', live.raws), ('crc', live.crcs)):
				assert list(theirs) == group.live[name_].tolist()

		summary = record.summary
		if group.summary is None:
			assert summary.len_stream == 0 and summary.stream is None
		else:
			tiles, sums = group.summary
			assert list(summary.stream.tiles) == tiles.tolist()
			assert numpy.array_equal(numpy.array(summary.stream.sums,
				dtype=numpy.float32), sums.ravel())
		assert record.zooms.n_zooms == len(group.zooms)
		for level, (factor, bins, sums) in zip(record.zooms.levels, group.zooms):
			assert level.factor == factor and list(level.stream.bins) == \
				bins.tolist()
			assert numpy.array_equal(numpy.array(level.stream.sums), sums.ravel())

	if header.order == 'positions':
		seen = 0
		for g, k, tile, kind_byte, data in chunks_of(parsed):
			group = file.groups[g]
			assert tile == group.live['tile'][k] and kind_byte == group.live[
				'kind'][k]
			assert data == file.chunk(g, k)
			expected = spec.decode_chunk(data, kind_byte, len(file.group_tracks[
				g]), header.tile)
			assert numpy.array_equal(decode(parsed, kind_byte, len(
				file.group_tracks[g]), header.tile, data), expected, equal_nan=True)
			seen += 1
		assert seen == sum(len(group.live['tile']) for group in file.groups)


def test_committed_parser_is_the_compilers_output(tmp_path):
	"""tests/kaitai_figwig.py is docs/figwig.ksy compiled; a change to the
	definition regenerates it. Skipped where the compiler is not installed,
	since it is a Java program; the parser is then still exercised above."""

	if shutil.which('kaitai-struct-compiler') is None:
		pytest.skip('kaitai-struct-compiler is not on the PATH')
	root = pathlib.Path(__file__).resolve().parent.parent
	fresh = regenerate(root, tmp_path)
	committed = (root / GENERATED).read_text()
	body = committed[committed.index('import kaitaistruct'):]
	assert fresh == body, 'regenerate tests/kaitai_figwig.py from docs/figwig.ksy'


def test_definition_is_the_runtime_version_the_parser_needs():
	assert tuple(int(x) for x in kaitaistruct.__version__.split('.')[:2]) >= \
		(0, 10)
	assert sys.version_info >= (3, 10)


@pytest.mark.parametrize('name', ['indexed', 'indexed_tracks'])
def test_kaitai_parses_the_summary_index_as_the_reference_reader_finds_it(
	files, name):
	"""summary_index and summary_page of docs/figwig.ksy find the root, the
	locators, the catalogs and the descriptors that summary_reference.py
	reads from "The regional summary index" of docs/format.rst, and the
	checksums of the fragments are those of the bytes."""

	path = files[name]
	data = path.read_bytes()
	index = read_index(path)
	stream = kaitaistruct.KaitaiStream(io.BytesIO(data))
	assert Figwig(stream).header.has_summary_index
	parsed = Figwig.SummaryIndex(index.root, stream)
	root = parsed.root
	assert (root.version, root.len_root, root.descriptors_per_page,
		root.window_target) == (2, 256, 102, 32768)
	assert root.directory_offset == index.root + 256
	assert root.n_groups == len(index.groups)
	assert bytes(root.header_snapshot) == index.header
	assert zlib.crc32(data[index.root:index.root + 256][:12] + bytes(4) +
		data[index.root + 16:index.root + 256]) == root.crc
	windows = 0
	for locator, group in zip(parsed.locators, index.groups):
		assert locator.levels_offset == group.catalog
		assert locator.n_levels == len(group.levels)
		for entry, level in zip(locator.levels, group.levels):
			for field in ENTRY_FIELDS:
				assert getattr(entry, field) == getattr(level, field), field
			windows += check_pages(stream, data, level)
	assert windows


# The fields of a catalog entry and of a descriptor, named alike in docs/format.rst,
# docs/figwig.ksy and the reference reader.
ENTRY_FIELDS = ['source_offset', 'index_offset', 'source_rows', 'source_bytes',
	'source_crc', 'window_bins']
DESCRIPTOR_FIELDS = ['n_rows', 'first_source_row', 'ids_offset', 'ids_bytes',
	'sums_offset', 'sums_bytes']


def check_pages(stream, data, level):
	"""Parse each page of `level` with summary_page and compare every
	descriptor with the reference reader's window, or with zeros, and the
	checksums with those of the bytes."""

	checked = 0
	count = -(-level.total // level.window_bins) if level.window_bins else 0
	body = data[level.source_offset:level.source_offset + level.source_bytes]
	for p, at in enumerate(level.pages):
		stream.seek(at)
		n = min(102, count - 102 * p)
		page = Figwig.SummaryPage(n, stream)
		assert (page.first_window, page.count) == (102 * p, n)
		for slot, window in enumerate(page.windows):
			found = level.windows.get(102 * p + slot)
			if found is None:
				assert not any(getattr(window, field) for field in DESCRIPTOR_FIELDS)
				continue
			for field in DESCRIPTOR_FIELDS:
				assert getattr(window, field) == getattr(found, field), field
			ids = body[found.ids_offset:found.ids_offset + found.ids_bytes]
			sums = body[found.sums_offset:found.sums_offset + found.sums_bytes]
			assert (window.ids_crc, window.ids_raw_crc, window.sums_crc,
				window.sums_raw_crc) == (zlib.crc32(ids), zlib.crc32(
				found.ids.astype('<u4').tobytes()), zlib.crc32(sums),
				zlib.crc32(found.sums.tobytes()))
			checked += 1
	return checked
