# test_coverage.py
# Tests of the figwig file format from the 2026-10-22 review's coverage gaps:
# forged files, every empty input, out= in every mode, the converters' edge
# cases and the kernels' element types.

import os
import sys
import json
import pickle
import struct
import zlib
import subprocess
import multiprocessing

import numpy
import pytest

import figwig
import figwig.figwig

from figwig import FigWigReader
from figwig import FigWigWriter
from figwig import write_figwig
from figwig import read_bigwig
from figwig import write_bigwig

from figwig.__main__ import main

from tests.test_figwig import CHROM_SIZES
from tests.test_figwig import TILE
from tests.test_figwig import TRACKS
from tests.test_figwig import assert_identical
from tests.test_figwig import reference_tracks
from tests.test_figwig import sparse_to_dense
from tests.test_figwig import write_reference

ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1')

_ZERO, _DENSE, _SPARSE, _MATRIX, _RUNS = 0, 1, 2, 3, 4
_U8, _U16, _U32, _I8, _I16, _I32, _F32, _ONE, _F64 = range(9)


@pytest.fixture(params=['scratch', 'no_scratch'])
def scratch(request, monkeypatch):
	"""Binned reads decode a chunk into a scratch tile, or, with none, bin
	each base straight into the window's row (what a read of thousands of
	tracks does)."""

	if request.param == 'no_scratch':
		monkeypatch.setattr(figwig.figwig, '_SCRATCH_BYTES', 0)
	return request.param


def kind_and_code(reader, g=0):
	kinds = reader._group_index(g)[0]
	return {(int(k & 7), int((k >> 3) & 15)) for k in kinds if k & 7}


###
# Every element code in every chunk kind, read in every mode


CODES = {'u8': (_U8, None), 'u16': (_U16, None), 'u32': (_U32, None),
	'i8': (_I8, None), 'i16': (_I16, None), 'i32': (_I32, None),
	'f32': (_F32, 'float32'), 'f64': (_F64, 'float64'), 'one': (_ONE, None)}
LENGTHS = {'c1': 64 * 5 + 23, 'c2': 10}


def code_values(code, rng, n):
	"""n nonzero values that need element type `code`, once the largest
	of them is in the chunk."""

	if code == 'u8':
		return rng.integers(1, 256, n).astype(numpy.float64)
	if code == 'u16':
		return rng.integers(256, 65536, n).astype(numpy.float64)
	if code == 'u32':
		return rng.integers(65536, 4_000_000_000, n).astype(numpy.float64)
	if code == 'i8':
		v = rng.integers(-128, 0, n).astype(numpy.float64)
		v[1::3] = rng.integers(1, 128, len(v[1::3]))
		return v
	if code == 'i16':
		return rng.integers(-30000, -128, n).astype(numpy.float64)
	if code == 'i32':
		return rng.integers(-2_000_000_000, -32769, n).astype(numpy.float64)
	if code == 'f32':
		return (rng.normal(0, 10, n) + 0.5).astype(numpy.float32).astype(
			numpy.float64)
	if code == 'f64':
		return rng.integers(1, 10 ** 6, n) + 1.0 / 3.0
	return numpy.ones(n)


def kind_truth(kind, code, rng, n_tracks, length, tile=64):
	"""The per-base values (tracks, length) whose tiles take a chunk of
	this kind."""

	truth = numpy.zeros((n_tracks, length))
	if kind == 'dense':
		truth[:] = code_values(code, rng, truth.size).reshape(truth.shape)
	elif kind == 'sparse':
		for t in range(n_tracks):
			where = rng.choice(length, max(1, length // 64), replace=False)
			truth[t, where] = code_values(code, rng, len(where))
	elif kind == 'runs':
		for t in range(n_tracks):
			for a in range(0, length, 16):
				truth[t, a:a + 16] = code_values(code, rng, 1)[0]
	else:
		for tile_start in range(0, length, tile):
			for t in rng.choice(n_tracks, 2, replace=False):
				truth[t, min(length - 1, tile_start + int(rng.integers(0, tile)))
					] = code_values(code, rng, 1)[0]
	return truth


def write_truth(path, truth, code, level, order='positions', tile=64):
	names = ['t{}'.format(i) for i in range(len(truth['c1']))]
	dtype = CODES[code][1]
	tracks = [{'name': n} if dtype is None else {'name': n, 'dtype': dtype}
		for n in names]
	with FigWigWriter(path, LENGTHS, tracks, tile=tile, groups=[names] if order ==
			'positions' else 1, level=level, order=order, n_jobs=2) as w:
		for chrom, values in truth.items():
			w.write(chrom, [0], values[None])


def check_every_mode(path, truth, tile=64):
	"""Windows, sparse windows, bins, labels, sums, entries, tile sums and
	a verified read all give what the per-base truth gives."""

	rng = numpy.random.default_rng(5)
	n_tracks = len(truth['c1'])
	reader = FigWigReader(path)
	for chrom, values in truth.items():
		length = values.shape[1]
		exp = numpy.full((1, n_tracks, length + 10), numpy.nan)
		exp[0, :, :length] = values
		for n_jobs in (1, 2):
			assert_identical(reader.read(chrom, [0], length + 10, dtype='float64',
				n_jobs=n_jobs), exp)

	chroms = rng.choice(list(truth), 60)
	starts = numpy.array([int(rng.integers(0, LENGTHS[c] + 5)) for c in chroms])
	exp = numpy.full((60, n_tracks, 100), numpy.nan)
	for j, (c, s) in enumerate(zip(chroms, starts)):
		piece = truth[c][:, s:s + 100]
		exp[j, :, :piece.shape[1]] = piece
	dense = reader.read(chroms, starts, 100, dtype='float64')
	assert_identical(dense, exp)
	assert_identical(FigWigReader(path, verify=True).read(chroms, starts, 100,
		dtype='float64'), exp)

	found = reader.read(chroms, starts, 100, sparse=True, dtype='float64')
	past = numpy.isnan(exp).all(axis=1)
	zeroed = numpy.nan_to_num(exp)
	zeroed[numpy.broadcast_to(past[:, None, :], exp.shape)] = 0
	numpy.testing.assert_array_equal(sparse_to_dense(found), zeroed)

	binned = reader.read(chroms, starts, 100, bins=10, dtype='float64')
	numpy.testing.assert_allclose(binned, zeroed.reshape(60, n_tracks, 10, 10
		).sum(-1), rtol=1e-9, atol=1e-6)

	labels = numpy.arange(n_tracks) % 3
	summed = reader.read(chroms, starts, 100, labels=labels, dtype='float64')
	# A base past a chromosome's end is NaN in the sum, as in a dense read.
	want = numpy.stack([exp[:, labels == k].sum(1) for k in range(3)], 1)
	numpy.testing.assert_array_equal(numpy.isnan(summed), numpy.isnan(want))
	numpy.testing.assert_allclose(numpy.nan_to_num(summed), numpy.nan_to_num(
		want), rtol=1e-9, atol=1e-6)

	for chrom in truth:
		length = LENGTHS[chrom]
		a = rng.integers(0, length + 20, 40)
		b = a + rng.integers(0, 150, 40)
		got = reader.sums(chrom, a, b, dtype='float64')
		padded = numpy.zeros((n_tracks, length + 400))
		padded[:, :length] = truth[chrom]
		want = numpy.stack([padded[:, x:y].sum(1) for x, y in zip(a, b)])
		numpy.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-6)

		entries = reader.read_entries(chroms=chrom)
		t, p = numpy.nonzero(truth[chrom])
		order = numpy.lexsort((p, t))
		numpy.testing.assert_array_equal(entries.tracks, t[order])
		numpy.testing.assert_array_equal(entries.positions, p[order])
		numpy.testing.assert_array_equal(entries.values.astype(numpy.float64),
			truth[chrom][t[order], p[order]])

		sums = reader.tile_sums(chroms=chrom)
		n_tiles = -(-length // tile)
		padded = numpy.zeros((n_tracks, n_tiles * tile))
		padded[:, :length] = truth[chrom]
		numpy.testing.assert_allclose(sums, padded.reshape(n_tracks, n_tiles, tile
			).sum(-1), rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize('level', [0, 6])
@pytest.mark.parametrize('kind', ['dense', 'sparse', 'runs', 'matrix'])
@pytest.mark.parametrize('code', list(CODES))
def test_every_code_in_every_chunk_kind(tmp_path, code, kind, level, scratch):
	"""A chunk of every kind in every element type, stored or deflated,
	reads, bins, sums and lists the same values as the truth."""

	if kind == 'dense' and code == 'one':
		pytest.skip("a dense chunk of ones is a runs chunk")
	rng = numpy.random.default_rng(hash((code, kind)) % 2 ** 32)
	n_tracks = 40 if kind == 'matrix' else 3
	truth = {c: kind_truth(kind, code, rng, n_tracks, n) for c, n in
		LENGTHS.items()}
	path = str(tmp_path / 'k.fw')
	write_truth(path, truth, code, level)
	if level == 0:
		kinds = kind_and_code(FigWigReader(path))
		want_kind = {'dense': _DENSE, 'sparse': _SPARSE, 'runs': _RUNS,
			'matrix': _MATRIX}[kind]
		assert want_kind in {k for k, _ in kinds}
		if code != 'one' or kind in ('sparse', 'matrix'):
			assert (want_kind, CODES[code][0]) in kinds
	check_every_mode(path, truth)


@pytest.mark.parametrize('code', list(CODES))
def test_every_code_in_track_order(tmp_path, code, scratch):
	"""A track-order file's blocks hold every element type, and read as
	windows, bins, entries and sums as the truth does."""

	rng = numpy.random.default_rng(11)
	truth = {c: kind_truth('sparse', code, rng, 3, n) for c, n in
		LENGTHS.items()}
	path = str(tmp_path / 't.fw')
	write_truth(path, truth, code, 0, order='tracks')
	assert FigWigReader(path).order == 'tracks'
	check_every_mode(path, truth)


###
# Files forged so that a check behind the checksums is what refuses them


HEADER = figwig.figwig._HEADER
RUN = figwig.figwig._RUN
BLOCK = figwig.figwig._BLOCK


@pytest.fixture(scope='module')
def small(tmp_path_factory):
	"""Two small files of the reference tracks: one with one group in
	position order and one in track order, both with the index's summaries."""

	ref = reference_tracks()
	directory = tmp_path_factory.mktemp('forge')
	paths = {}
	for name, order, groups in (('positions', 'positions', [['atac_plus',
			'atac_minus', 'signal', 'negative', 'empty']]), ('tracks', 'tracks',
			1)):
		path = str(directory / (name + '.fw'))
		with FigWigWriter(path, CHROM_SIZES, TRACKS, tile=64, groups=groups,
				order=order, level=6, n_jobs=2) as writer:
			for chrom, length in CHROM_SIZES.items():
				writer.write(chrom, [0], ref[chrom][None])
		paths[name] = path
	return paths


def load(path):
	with open(path, 'rb') as handle:
		data = bytearray(handle.read())
	return data, numpy.frombuffer(bytes(data[:HEADER.itemsize]), dtype=HEADER
		)[0]


def put(data, name, value):
	"""Set a header field."""

	dtype, at = HEADER.fields[name][:2]
	data[at:at + dtype.itemsize] = numpy.array(value, dtype=dtype).tobytes()


def seal(data):
	"""Recompute the header's checksums of the tables and the directory and
	of itself, as a forger who knows the format would."""

	header = numpy.frombuffer(bytes(data[:HEADER.itemsize]), dtype=HEADER)[0]
	chroms, index = int(header['chroms']), int(header['index'])
	n_groups = int(header['n_groups'])
	put(data, 'tables_crc', zlib.crc32(bytes(data[chroms:index + 8 * n_groups])))
	put(data, 'header_crc', figwig.figwig._header_crc(bytes(data[:HEADER
		.itemsize])))


def save(tmp_path, name, data):
	path = str(tmp_path / name)
	with open(path, 'wb') as handle:
		handle.write(bytes(data))
	return path


def metadata_of(data, header):
	at, size = int(header['metadata']), int(header['metadata_length'])
	return json.loads(bytes(data[at:at + size]).decode())


def put_metadata(data, header, document):
	"""Replace the metadata by `document`, padded with spaces to the length
	it had."""

	at, size = int(header['metadata']), int(header['metadata_length'])
	raw = json.dumps(document, separators=(',', ':')).encode()
	assert len(raw) <= size
	data[at:at + size] = raw + b' ' * (size - len(raw))


def refuses(path, match, call=None):
	with pytest.raises(ValueError, match=match):
		reader = FigWigReader(path)
		(call or (lambda r: r.read('chr1', [0], 10)))(reader)


def test_forged_metadata_is_refused(tmp_path, small):
	data, header = load(small['positions'])
	n = int(header['n_tracks'])

	def case(name, match, edit, replace=None):
		copy = bytearray(data)
		document = metadata_of(copy, header)
		edit(document)
		put_metadata(copy, header, document if replace is None else replace)
		seal(copy)
		refuses(save(tmp_path, name + '.fw', copy), match)

	case('names', "two tracks are named", lambda d: d['tracks'][1].update(
		name=d['tracks'][0]['name']))
	case('nodtype', "missing its name, dtype or group", lambda d: d['tracks'][
		0].pop('dtype'))
	case('dtype', "missing its name, dtype or group", lambda d: d['tracks'][
		0].update(dtype='float16'))
	case('group', "missing its name, dtype or group", lambda d: d['tracks'][
		0].update(group=9))
	case('boolgroup', "missing its name, dtype or group", lambda d: d['tracks'
		][0].update(group='0'))
	case('count', "does not describe its {} tracks".format(n), lambda d: d[
		'tracks'].pop())
	case('list', "is not a JSON object with attributes", lambda d: None,
		replace=[])

	# A header that counts two groups when every track is in group 0.
	copy = bytearray(data)
	put(copy, 'n_groups', 2)
	seal(copy)
	refuses(save(tmp_path, 'empty_group.fw', copy), "a group has no tracks")

	copy = bytearray(data)
	at = int(header['metadata'])
	copy[at:at + int(header['metadata_length'])] = b'\xff' * int(header[
		'metadata_length'])
	seal(copy)
	refuses(save(tmp_path, 'garbage.fw', copy), "not a JSON document")


def test_forged_chromosome_table_is_refused(tmp_path, small):
	data, header = load(small['positions'])
	at = int(header['chroms'])

	def name_at(i):
		position = at
		for _ in range(i):
			size = struct.unpack_from('<H', data, position)[0]
			position += 2 + size + 8
		return position

	second = name_at(1)
	copy = bytearray(data)
	copy[second + 2:second + 6] = b'chr1'
	seal(copy)
	refuses(save(tmp_path, 'twice.fw', copy), "distinct names")

	copy = bytearray(data)
	struct.pack_into('<Q', copy, second + 6, 0)
	seal(copy)
	refuses(save(tmp_path, 'zero.fw', copy), "distinct names")

	copy = bytearray(data)
	struct.pack_into('<Q', copy, second + 6, 2 ** 32)
	seal(copy)
	refuses(save(tmp_path, 'long.fw', copy), "distinct names")

	copy = bytearray(data)
	copy[second + 2:second + 4] = b'\xff\xfe'
	seal(copy)
	refuses(save(tmp_path, 'utf8.fw', copy), "not UTF-8")

	copy = bytearray(data)
	struct.pack_into('<H', copy, second, 65535)
	seal(copy)
	refuses(save(tmp_path, 'size.fw', copy), "ends inside its chromosome table")

	copy = bytearray(data)
	put(copy, 'n_chroms', 1000)
	seal(copy)
	refuses(save(tmp_path, 'count.fw', copy), "ends inside its chromosome table")


def test_forged_header_is_refused(tmp_path, small):
	data, header = load(small['positions'])

	copy = bytearray(data)
	put(copy, 'flags', 1)
	seal(copy)
	refuses(save(tmp_path, 'order.fw', copy), "needs a group per track")

	copy = bytearray(data)
	put(copy, 'n_tracks', 2 ** 28)
	seal(copy)
	refuses(save(tmp_path, 'tracks.fw', copy), "ends inside its track table")

	for field, value, match in (('tile', 48, "power of two"), ('tile', 8,
			"power of two"), ('codec', 5, "codec 5"), ('n_groups', 0,
			"no chromosomes, tracks or groups"), ('n_chroms', 0,
			"no chromosomes, tracks or groups"), ('metadata_length', 2 ** 40,
			"metadata lies outside"), ('length', 2 ** 40, "truncated")):
		copy = bytearray(data)
		put(copy, field, value)
		seal(copy)
		refuses(save(tmp_path, field + '.fw', copy), match)

	copy = bytearray(data)
	put(copy, 'magic', 0)
	refuses(save(tmp_path, 'magic.fw', copy), "is not a figwig file")
	refuses(save(tmp_path, 'short.fw', data[:100]), "too short")


def group_record(data, header, g=0, n_groups=None):
	"""The pieces of group g's index record, which must be the last in the
	file: (offset, runs, tiles, kinds, sizes, raws, crcs, summary), the
	live fields as arrays and the summary as (tiles, sums, tail)."""

	directory = numpy.frombuffer(bytes(data[int(header['index']):int(header[
		'index']) + 8 * int(header['n_groups'])]), dtype='<u8')
	offset = int(directory[g])
	assert offset == int(directory.max())
	n_runs = struct.unpack_from('<I', data, offset)[0]
	at = offset + 8
	runs = numpy.frombuffer(bytes(data[at:at + n_runs * RUN.itemsize]),
		dtype=RUN).copy()
	at += n_runs * RUN.itemsize
	n_live, n_bytes = struct.unpack_from('<II', data, at)
	raw = zlib.decompress(bytes(data[at + 8:at + 8 + n_bytes]))
	fields = [numpy.frombuffer(raw[start * n_live:stop * n_live], dtype=dtype
		).copy() for start, stop, dtype in ((0, 4, '<u4'), (4, 5, 'u1'),
		(5, 9, '<u4'), (9, 13, '<u4'), (13, 17, '<u4'))]
	at += 8 + n_bytes
	n_sum, n_sum_bytes = struct.unpack_from('<II', data, at)
	return offset, runs, fields, (n_sum, bytes(data[at + 8:at + 8 +
		n_sum_bytes]), bytes(data[at + 8 + n_sum_bytes:]))


def rebuild(data, header, offset, runs, fields, summary):
	"""The file with its last group's record written again from these
	pieces, checksum included."""

	n_sum, stream, tail = summary
	live = b''.join(f.tobytes() for f in fields)
	packed = zlib.compress(live)
	body = runs.tobytes() + struct.pack('<II', len(fields[0]), len(packed))
	crc = zlib.crc32(packed, zlib.crc32(body))
	record = struct.pack('<II', len(runs), crc) + body + packed + struct.pack(
		'<II', n_sum, len(stream)) + stream + tail
	copy = bytearray(data[:offset]) + record
	put(copy, 'length', len(copy))
	seal(copy)
	return copy


def test_forged_live_tiles_are_refused(tmp_path, small):
	data, header = load(small['positions'])
	offset, runs, fields, summary = group_record(data, header)
	assert len(fields[0]) >= 3

	def case(name, match, edit, call=None):
		f = [x.copy() for x in fields]
		r = runs.copy()
		changed = edit(f, r)
		r = changed if isinstance(changed, numpy.ndarray) else r
		refuses(save(tmp_path, name + '.fw', rebuild(data, header, offset, r, f,
			summary)), match, call)

	# Control: the record written again unchanged reads as the file did.
	same = save(tmp_path, 'same.fw', rebuild(data, header, offset, runs, fields,
		summary))
	assert_identical(FigWigReader(same).read('chr1', [0], 500, dtype='float64'),
		FigWigReader(small['positions']).read('chr1', [0], 500, dtype='float64'))
	case('code', "kind this figwig does not know", lambda f, r: f[1].__setitem__(
		0, (f[1][0] & 7) | (15 << 3)))
	case('kind', "kind this figwig does not know", lambda f, r: f[1].__setitem__(
		0, 7))
	case('size', "kind and size disagree", lambda f, r: f[2].__setitem__(0, 0))
	case('raw', "kind and size disagree", lambda f, r: f[3].__setitem__(0, 0))
	case('stored', "kind and size disagree", lambda f, r: (f[1].__setitem__(0,
		f[1][0] | 128), f[3].__setitem__(0, f[2][0] + 1)))
	case('order', "tiles are out of order", lambda f, r: (f[0].__setitem__(
		slice(0, 2), f[0][1::-1].copy())))
	case('past', "tiles are out of order", lambda f, r: f[0].__setitem__(-1,
		2 ** 20))
	case('uncovered', "lies in no run", lambda f, r: r[:0])
	case('overlap', "out of order or lies outside", lambda f, r: numpy.concatenate(
		[r, r]) if len(r) == 1 else None)


def test_forged_summary_is_refused(tmp_path, small):
	data, header = load(small['positions'])
	offset, runs, fields, (n_sum, stream, tail) = group_record(data, header)
	assert n_sum > 2
	n_tracks = int(header['n_tracks'])
	raw = zlib.decompress(stream)
	tiles = numpy.frombuffer(raw[:4 * n_sum], dtype='<u4').copy()
	sums = raw[4 * n_sum:]

	def forged(tiles_, n=None, packed=None):
		body = zlib.compress(tiles_.astype('<u4').tobytes() + sums)
		return rebuild(data, header, offset, runs, fields, (len(tiles_) if n is
			None else n, body if packed is None else packed, tail))

	def call(r):
		return r.tile_sums()

	swapped = tiles.copy()
	swapped[:2] = swapped[1::-1]
	refuses(save(tmp_path, 'swap.fw', forged(swapped)), "summary is out of order",
		call)
	big = tiles.copy()
	big[-1] = 2 ** 24
	refuses(save(tmp_path, 'big.fw', forged(big)), "summary is out of order",
		call)
	refuses(save(tmp_path, 'count.fw', forged(tiles, n=2 ** 30)),
		"ends inside it", call)
	refuses(save(tmp_path, 'short.fw', forged(tiles, packed=b'x' * 3)),
		"ends inside it|cannot be read", call)
	save(tmp_path, 'ok.fw', forged(tiles))
	assert FigWigReader(str(tmp_path / 'ok.fw')).tile_sums().shape[0] == n_tracks


def test_forged_track_order_blocks_are_refused(tmp_path, small):
	data, header = load(small['tracks'])
	directory = numpy.frombuffer(bytes(data[int(header['index']):int(header[
		'index']) + 8 * int(header['n_groups'])]), dtype='<u8')
	offset = int(directory[0])
	n_blocks = struct.unpack_from('<I', data, offset)[0]
	assert n_blocks >= 2
	at = offset + 8
	blocks = numpy.frombuffer(bytes(data[at:at + n_blocks * BLOCK.itemsize]),
		dtype=BLOCK).copy()

	def case(name, match, edit):
		b = blocks.copy()
		edit(b)
		copy = bytearray(data)
		copy[at:at + b.nbytes] = b.tobytes()
		struct.pack_into('<I', copy, offset + 4, zlib.crc32(b.tobytes()))
		refuses(save(tmp_path, name + '.fw', copy), match, lambda r: r.read(
			'chr1', [0], 10, tracks=0))

	case('chrom', "out of range", lambda b: b['chrom'].__setitem__(0, 99))
	case('empty', "out of range", lambda b: b['n'].__setitem__(0, 0))
	case('last', "out of range", lambda b: b['last'].__setitem__(0, 2 ** 31))
	case('first', "out of range", lambda b: b['first'].__setitem__(0, b['last'][
		0] + 1))
	case('kind', "out of range", lambda b: b['kind'].__setitem__(0, 2))
	case('code', "out of range", lambda b: b['kind'].__setitem__(0, 1 | (15 <<
		3)))
	case('offset', "outside the file", lambda b: b['offset'].__setitem__(0,
		2 ** 62))
	case('size', "outside the file", lambda b: b['size'].__setitem__(0,
		2 ** 31))
	case('order', "out of order", lambda b: b.__setitem__(slice(0, 2), b[1::-1]
		.copy()))
	with pytest.raises(ValueError, match="does not match its checksum"):
		copy = bytearray(data)
		copy[at] ^= 1
		FigWigReader(save(tmp_path, 'crc.fw', copy)).read('chr1', [0], 10,
			tracks=0)


###
# Arguments of read, sums, write and write_sparse


@pytest.fixture(scope='module')
def ref():
	return reference_tracks()


@pytest.fixture(scope='module')
def fw(tmp_path_factory, ref):
	path = str(tmp_path_factory.mktemp('proposed') / 'reference.fw')
	write_reference(path, ref)
	return path


@pytest.fixture(scope='module')
def track_fw(tmp_path_factory, ref):
	path = str(tmp_path_factory.mktemp('proposed') / 'tracks.fw')
	write_reference(path, ref, order='tracks')
	return path


@pytest.fixture(params=['positions', 'tracks'])
def either(request, fw, track_fw):
	return fw if request.param == 'positions' else track_fw


def test_tracks_in_every_form(fw, ref):
	reader = FigWigReader(fw)
	want = reader.read('chr1', [0, 70], 40, tracks=[1, 0])
	for tracks in (('atac_minus', 'atac_plus'), numpy.array(['atac_minus',
			'atac_plus']), numpy.array([1, 0]), [numpy.int64(1), 'atac_plus'],
			[1, 0]):
		assert_identical(reader.read('chr1', [0, 70], 40, tracks=tracks), want)
	assert_identical(reader.read('chr1', [0], 40, tracks=range(2)),
		reader.read('chr1', [0], 40, tracks=[0, 1]))
	one = reader.read('chr1', [0, 70], 40, tracks=numpy.int64(1))
	assert one.shape == (2, 40)
	assert_identical(one, want[:, 0])
	with pytest.raises(TypeError, match="a list of them"):
		reader.read('chr1', [0], 40, tracks=True)
	with pytest.raises(TypeError, match="names or indices"):
		reader.read('chr1', [0], 40, tracks=[True])


@pytest.mark.parametrize('dup', [[0, 0], ['atac_plus', 0, 'atac_plus'], [2, 2,
	2]])
def test_duplicate_tracks_are_read_once_each_and_summed_twice(either, dup):
	"""A track named twice has two channels, in a window read and in sums
	and bins; only labels refuse it."""

	reader = FigWigReader(either)
	rng = numpy.random.default_rng(1)
	starts = rng.integers(0, 4900, 20)
	y = reader.read('chr1', starts, 64, tracks=dup, dtype='float64')
	assert y.shape == (20, len(dup), 64)
	for k in range(len(dup)):
		assert_identical(y[:, k], reader.read('chr1', starts, 64, tracks=[dup[
			k]], dtype='float64')[:, 0])
	sums = reader.sums('chr1', starts, starts + 64, tracks=dup, dtype='float64')
	assert sums.shape == (20, len(dup))
	numpy.testing.assert_array_equal(sums[:, 0], sums[:, -1])
	binned = reader.read('chr1', starts, 64, tracks=dup, bins=16, dtype='float64')
	assert binned.shape == (20, len(dup), 4)
	found = reader.read('chr1', starts, 64, tracks=dup, sparse=True)
	assert found.shape == (20, len(dup), 64)


def test_starts_in_every_form(fw):
	reader = FigWigReader(fw)
	want = reader.read('chr1', [10, 20], 8)
	for starts in ([10, 20], (10, 20), numpy.array([10, 20], dtype=numpy.uint64),
			numpy.array([10, 20], dtype=numpy.int16), [numpy.int32(10),
			numpy.int64(20)]):
		assert_identical(reader.read('chr1', starts, 8), want)
	for starts in (numpy.array([True]), numpy.array([2.0]), ['1']):
		with pytest.raises(TypeError, match="must be integers"):
			reader.read('chr1', starts, 8)
	with pytest.raises(ValueError, match="one-dimensional"):
		reader.read('chr1', [[1, 2]], 8)
	with pytest.raises(ValueError, match="one name per start"):
		reader.read(['chr1', 'chr2'], [1], 8)
	assert reader.read(numpy.array(['chr1', 'chr2']), [1, 2], 8).shape == (2, 5,
		8)
	with pytest.raises(ValueError, match="start before 0 or end past"):
		reader.read('chr1', [2 ** 40], 8)
	with pytest.raises(ValueError, match="start before 0"):
		reader.read('chr1', numpy.array([2 ** 63 + 5], dtype=numpy.uint64), 8)


def test_one_base_windows_at_every_position(either, ref):
	"""Width-1 windows from 0 to two past each chromosome's end give the
	base, NaN past the end, in a dense read and as sums and bins."""

	reader = FigWigReader(either)
	for chrom, length in CHROM_SIZES.items():
		starts = numpy.arange(length + 2)
		y = reader.read(chrom, starts, 1, dtype='float64')
		assert y.shape == (length + 2, 5, 1)
		want = numpy.full((length + 2, 5), numpy.nan)
		want[:length] = ref[chrom].T
		assert_identical(y[:, :, 0], want)
		# A NaN in the file stays NaN in a sum; a base past the end is 0 there
		# and absent from the entries.
		inside = want.copy()
		inside[length:] = 0
		numpy.testing.assert_array_equal(reader.read(chrom, starts, 1, bins=1,
			dtype='float64')[:, :, 0], inside)
		numpy.testing.assert_array_equal(reader.sums(chrom, starts, starts + 1,
			dtype='float64'), inside)
		found = reader.read(chrom, starts, 1, sparse=True, dtype='float64')
		numpy.testing.assert_array_equal(sparse_to_dense(found)[:, :, 0], inside)


def test_windows_past_the_end_in_every_mode(either, ref):
	"""A window starting at a chromosome's end or after it, or longer than
	the chromosome: NaN in a dense float read and in sums per label, absent
	from sparse entries, and 0 in sums and bins (a base past the end counts
	as 0 there)."""

	reader = FigWigReader(either)
	chroms = ['chrM', 'chrM', 'chrM', 'chr2', 'chr1']
	starts = [70, 71, 5000, 3001 + 64, 4990]
	width = 150
	y = reader.read(chroms, starts, width, dtype='float64')
	assert numpy.isnan(y[:3]).all() and numpy.isnan(y[3]).all()
	assert numpy.isnan(y[4, :, 10:]).all() and not numpy.isnan(y[4, :, :10]
		).any()
	found = reader.read(chroms, starts, width, sparse=True, dtype='float64')
	assert not numpy.isin(found.rows, [0, 1, 2, 3]).any()
	assert (found.offsets[found.rows == 4] < 10).all()

	labelled = reader.read(chroms, starts, width, labels=[0, 0, 1, 1, 1],
		dtype='float64')
	assert numpy.isnan(labelled[:4]).all()

	zero = numpy.zeros((4, 5))
	numpy.testing.assert_array_equal(reader.sums(chroms[:4], starts[:4], [
		s + width for s in starts[:4]], dtype='float64'), zero)
	numpy.testing.assert_array_equal(reader.read(chroms[:4], starts[:4], width,
		bins=width, dtype='float64')[:, :, 0], zero)
	assert (reader.read('chrM', [0], 500, bins=500, dtype='float64')[0, :, 0] ==
		numpy.nan_to_num(ref['chrM']).sum(1)).all()


def test_sums_arguments(either):
	reader = FigWigReader(either)
	with pytest.raises(ValueError, match="end is before its start"):
		reader.sums('chr1', [5], [4])
	with pytest.raises(ValueError, match="one end per start"):
		reader.sums('chr1', [5], [4, 5])
	with pytest.raises(TypeError, match="ends must be integers"):
		reader.sums('chr1', [5], [6.0])
	with pytest.raises(ValueError, match="not one of the file's chromosomes"):
		reader.sums('chr9', [5], [6])
	with pytest.raises(ValueError, match="dtype must be"):
		reader.sums('chr1', [5], [6], dtype='float16')
	with pytest.raises(ValueError, match="cannot be read as"):
		reader.sums('chr1', [5], [6], tracks='signal', dtype='int64')
	with pytest.raises(ValueError, match="nan must be"):
		reader.sums('chr1', [5], [6], nan='drop')
	assert reader.sums('chr1', [], []).shape == (0, 5)
	assert reader.sums('chr1', [], [], tracks='signal').shape == (0,)
	assert reader.sums('chr1', [7], [7], tracks='signal') == 0


def test_labels_and_bins_together(either, ref):
	reader = FigWigReader(either)
	rng = numpy.random.default_rng(8)
	starts = rng.integers(0, 4900, 25)
	dense = numpy.nan_to_num(reader.read('chr1', starts, 128, dtype='float64'))
	got = reader.read('chr1', starts, 128, labels=[1, 0, -1, 1, 1], bins=32,
		dtype='float64')
	groups = [dense[:, [1]], dense[:, [0, 3, 4]]]
	want = numpy.stack([g.sum(1) for g in groups], 1).reshape(25, 2, 4, 32).sum(
		-1)
	numpy.testing.assert_allclose(got, want, rtol=1e-12)
	with pytest.raises(ValueError, match="nan= cannot be given with labels"):
		reader.read('chr1', starts, 128, labels=[0] * 5, bins=32, nan='skip')


def test_out_for_bins_and_labels(either):
	reader = FigWigReader(either)
	out = numpy.full((4, 5, 8), 9, dtype=numpy.float64)
	got = reader.read('chr1', [0, 100, 200, 4990], 64, bins=8, out=out,
		dtype='float64')
	assert got is out
	assert_identical(out, reader.read('chr1', [0, 100, 200, 4990], 64, bins=8,
		dtype='float64'))
	out = numpy.full((4, 2, 64), 9, dtype=numpy.float32)
	got = reader.read('chr1', [0, 100, 200, 4990], 64, labels=[0, 1, 0, 1, 1],
		out=out, dtype='float32')
	assert got is out
	assert_identical(out, reader.read('chr1', [0, 100, 200, 4990], 64,
		labels=[0, 1, 0, 1, 1], dtype='float32'))
	with pytest.raises(ValueError, match="shape"):
		reader.read('chr1', [0], 64, bins=8, out=numpy.zeros((1, 5, 64),
			dtype=numpy.float32))
	frozen = numpy.zeros((1, 5, 8), dtype=numpy.float32)
	frozen.setflags(write=False)
	with pytest.raises(ValueError, match="writeable"):
		reader.read('chr1', [0], 64, bins=8, out=frozen)


def test_dtype_given_as_a_torch_type(fw):
	torch = pytest.importorskip('torch')
	reader = FigWigReader(fw)
	assert reader.read('chr1', [0], 10, tracks='signal', dtype=torch.float32
		).dtype == numpy.float32
	assert reader.read('chr1', [0], 10, tracks='signal', dtype=torch.float64
		).dtype == numpy.float64
	assert reader.read('chr1', [0], 10, tracks='negative', dtype=torch.int32
		).dtype == numpy.int32
	for bad in (torch.bfloat16, torch.float16):
		with pytest.raises(ValueError, match="read float32 and cast"):
			reader.read('chr1', [0], 10, dtype=bad)
	with pytest.raises(ValueError, match="cannot be read as"):
		reader.read('chr1', [0], 10, tracks='signal', dtype=torch.int64)
	with pytest.raises(TypeError, match="not a dtype"):
		reader.read('chr1', [0], 10, dtype='nonsense')
	with pytest.raises(ValueError, match="32 or 64 bits"):
		reader.read('chr1', [0], 10, dtype=numpy.complex64)
	with pytest.raises(ValueError, match="does not hold"):
		reader.read('chr1', [0], 10, tracks='atac_plus', dtype=numpy.uint8)


def test_empty_requests(either):
	reader = FigWigReader(either)
	assert reader.read([], [], 10, sparse=True).shape == (0, 5, 10)
	assert reader.read([], [], 10, labels=[0, 0, 1, 1, -1]).shape == (0, 2, 10)
	assert reader.read([], [], 10, bins=5).shape == (0, 5, 2)
	assert reader.read([], [], 10, tracks='signal').shape == (0, 10)
	assert reader.read('chr1', [], 10, bins=10, tracks=[0, 1]).shape == (0, 2, 1)
	assert reader.read_entries(chroms=[]).values.shape == (0,)


def test_tile_sums_of_no_chromosomes(either):
	with pytest.raises(ValueError, match="chroms must name at least one"):
		FigWigReader(either).tile_sums(chroms=[])


def test_empty_writes_leave_a_valid_file(tmp_path):
	path = str(tmp_path / 'e.fw')
	with FigWigWriter(path, CHROM_SIZES, ['a', 'b'], tile=64) as writer:
		writer.write('chr1', [], numpy.zeros((0, 2, 5)))
		writer.write('chr1', numpy.array([], dtype=numpy.int64), numpy.zeros((0,
			2)), ends=[])
		writer.write_sparse('chr1', [], [], [])
		writer.write('chr1', [10], numpy.ones((1, 2, 3)))
	reader = FigWigReader(path)
	assert reader.stats['nonzero'].tolist() == [3, 3]
	numpy.testing.assert_array_equal(reader.read('chr1', [10], 3)[0], numpy.ones(
		(2, 3)))

	other = str(tmp_path / 'none.fw')
	write_figwig(other, CHROM_SIZES, ['a'], [], [], numpy.zeros((0, 1, 5)))
	reader = FigWigReader(other)
	assert reader.stats['nonzero'].tolist() == [0]
	assert reader.read('chr1', [0], 20).sum() == 0
	assert len(reader.read_entries().values) == 0
	assert reader.tile_sums().sum() == 0


def test_closed_and_misused_writers(tmp_path):
	path = str(tmp_path / 'w.fw')
	writer = FigWigWriter(path, CHROM_SIZES, ['a', 'b'], tile=64, groups=1)
	with pytest.raises(TypeError, match="a list of them"):
		writer.write('chr1', [0], numpy.ones((1, 1, 3)), tracks=3.5)
	with pytest.raises(ValueError, match="at least one track"):
		writer.write('chr1', [0], numpy.ones((1, 1, 3)), tracks=[])
	with pytest.raises(ValueError, match="not one of the tracks|is not one of"):
		writer.write('chr1', [0], numpy.ones((1, 1, 3)), tracks='c')
	writer.write('chr1', [0], numpy.ones((1, 3)), tracks='a')
	writer.close()
	writer.close()
	for call in (lambda: writer.write('chr1', [10], numpy.ones((1, 1, 3)),
			tracks='b'), lambda: writer.write_sparse('chr1', [10], [0], [1]),
			lambda: writer.add_file(path)):
		with pytest.raises(ValueError, match="the writer is closed"):
			call()
	assert FigWigReader(path).read('chr1', [0], 3, tracks='a').sum() == 3


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_value_checks_of_the_writer(tmp_path):
	# The filter is for NUMBA_DISABLE_JIT=1, where numpy warns on the cast of
	# a value past float32's range that the compiled kernel flags silently.
	path = str(tmp_path / 'v.fw')
	for tracks in (['a'], [{'name': 'a', 'dtype': 'float32'}]):
		with pytest.raises(ValueError, match="beyond float32's range"):
			write_figwig(path, CHROM_SIZES, tracks, 'chr1', [0], numpy.array([[[
				1e39]]]))
		with pytest.raises(ValueError, match="beyond float32's range"):
			write_figwig(path, CHROM_SIZES, tracks, 'chr1', [0], numpy.array([[
				-1e39]]), ends=[5])
	write_figwig(path, CHROM_SIZES, [{'name': 'a', 'dtype': 'float64'}], 'chr1',
		[0], numpy.array([[[1e39]]]))
	assert FigWigReader(path).read('chr1', [0], 1)[0, 0, 0] == 1e39

	tracks = [{'name': 'a', 'dtype': 'uint8'}]
	for value, match in ((1.5, "not an integer"), (300, "outside 0 to 255"),
			(-1, "outside 0 to 255"), (numpy.nan, "a value is NaN")):
		with pytest.raises(ValueError, match=match):
			with FigWigWriter(path, CHROM_SIZES, tracks, tile=64) as writer:
				writer.write_sparse('chr1', [3], [0], [value])
		with pytest.raises(ValueError, match=match):
			write_figwig(path, CHROM_SIZES, tracks, 'chr1', [0], numpy.full((1, 1,
				4), value))

	with pytest.raises(ValueError, match="bases long|must be from 1 to"):
		FigWigWriter(path, {'c': 2 ** 32}, ['a'])
	chroms = {'chr{}'.format(i): 100 for i in range(10)}
	with pytest.raises(ValueError, match="not one of the writer's chromosomes"):
		with FigWigWriter(path, chroms, ['a'], tile=64) as writer:
			writer.write('nope', [0], numpy.ones((1, 1, 5)))

	# A reader of a file with many chromosomes names six and counts the rest.
	with FigWigWriter(path, chroms, ['a'], tile=64) as writer:
		writer.write('chr9', [0], numpy.ones((1, 1, 5)))
	reader = FigWigReader(path)
	for call in (lambda: reader.read('nope', [0], 5), lambda: reader.sums(
			'nope', [0], [5]), lambda: reader.tile_sums(chroms='nope'),
			lambda: reader.read_entries(chroms=['nope'])):
		with pytest.raises(ValueError, match="'chr5', \\.\\.\\. \\(10 in all\\)"):
			call()


###
# Pickles, forks, processes


def _read_in_child(reader, queue):
	queue.put(reader.read('chr1', [0, 1000, 4000], 200, n_jobs=4))


def test_pickled_reader_reads_in_another_process(fw, ref):
	reader = FigWigReader(fw)
	want = reader.read('chr1', [0, 1000, 4000], 200, n_jobs=4)
	clone = pickle.loads(pickle.dumps(reader))
	assert_identical(clone.read('chr1', [0, 1000, 4000], 200), want)
	assert clone._pool is None and reader._index[0] is not None
	context = multiprocessing.get_context('spawn')
	queue = context.Queue()
	process = context.Process(target=_read_in_child, args=(reader, queue))
	process.start()
	try:
		got = queue.get(timeout=120)
	finally:
		process.join(60)
	assert process.exitcode == 0
	assert_identical(got, want)


@pytest.mark.skipif(not hasattr(os, 'fork'), reason="no fork on this platform")
def test_reader_with_a_thread_pool_reads_in_a_forked_child(fw, monkeypatch):
	"""A child that forks after the parent's pool was made gets a pool of
	its own, whose threads exist, instead of waiting on the parent's."""

	import warnings
	monkeypatch.setattr(figwig.figwig, '_CHUNKS_PER_THREAD', 1)
	reader = FigWigReader(fw)
	starts = numpy.arange(0, 4000, 100)
	want = reader.read('chr1', starts, 100, n_jobs=4)
	assert reader._pool is not None
	read_end, write_end = os.pipe()
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		pid = os.fork()
	if pid == 0:
		code = 1
		try:
			got = reader.read('chr1', starts, 100, n_jobs=4)
			os.write(write_end, b'1' if (got.tobytes() == want.tobytes() and
				reader._pool[0] == os.getpid()) else b'0')
			code = 0
		finally:
			os._exit(code)
	os.close(write_end)
	answer = os.read(read_end, 1)
	os.close(read_end)
	_, status = os.waitpid(pid, 0)
	assert status == 0 and answer == b'1'


def test_a_writer_cannot_be_pickled(tmp_path):
	writer = FigWigWriter(str(tmp_path / 'w.fw'), CHROM_SIZES, ['a'], tile=64)
	try:
		with pytest.raises((TypeError, pickle.PicklingError, AttributeError)):
			pickle.dumps(writer)
	finally:
		writer.close()


###
# append, add_file and merge together


def extra_tracks(chroms_sizes=CHROM_SIZES):
	"""Entries of two more tracks, as (chrom, positions, track, values)."""

	rng = numpy.random.default_rng(77)
	out = []
	for chrom, length in chroms_sizes.items():
		positions = numpy.unique(rng.integers(0, length, length // 20 + 1))
		tracks = rng.integers(0, 2, len(positions))
		values = rng.integers(1, 9, len(positions)).astype(numpy.float64)
		out.append((chrom, positions, tracks, values))
	return out


def dense_of(entries, names):
	truth = {c: numpy.zeros((len(names), n)) for c, n in CHROM_SIZES.items()}
	for chrom, positions, tracks, values in entries:
		truth[chrom][tracks, positions] = values
	return truth


@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_repeated_appends_then_a_copy_read_like_one_write(tmp_path, ref, order):
	"""Three appends in a row, each adding tracks, then the file copied by
	`add_file` into a fresh writer (which leaves the old tables behind) and
	by `merge`: every track, the stats and the tile sums are those of the
	pieces."""

	path = str(tmp_path / 'grow.fw')
	write_reference(path, ref, order=order)
	entries = extra_tracks()
	truth = []
	for round_, names in enumerate((['x0', 'x1'], ['y0', 'y1'], ['z0', 'z1'])):
		with FigWigWriter(path, None, names, append=True, n_jobs=2) as writer:
			for chrom, positions, tracks, values in entries:
				writer.write_sparse(chrom, positions, [names[t] for t in tracks],
					values + round_)
		truth.append({c: v + (v != 0) * round_ for c, v in dense_of(entries,
			names).items()})

	copy = str(tmp_path / 'copy.fw')
	with FigWigWriter(copy, CHROM_SIZES, None, tile=TILE, level=6,
			order=order) as writer:
		writer.add_file(path)
	merged = str(tmp_path / 'merged.fw')
	main(['merge', path, '-o', merged])

	original = FigWigReader(path)
	for name in (path, copy, merged):
		reader = FigWigReader(name, verify=True)
		assert [t['name'] for t in reader.tracks] == [t['name'] for t in
			TRACKS] + ['x0', 'x1', 'y0', 'y1', 'z0', 'z1']
		for chrom, length in CHROM_SIZES.items():
			y = reader.read(chrom, [0], length, dtype='float64')[0]
			assert_identical(y[:5], reader.read(chrom, [0], length, tracks=list(
				range(5)), dtype='float64')[0])
			numpy.testing.assert_array_equal(numpy.nan_to_num(y[:5]),
				numpy.nan_to_num(ref[chrom]))
			for k, values in enumerate(truth):
				numpy.testing.assert_array_equal(y[5 + 2 * k:7 + 2 * k],
					values[chrom])
		for field in original.stats.dtype.names:
			numpy.testing.assert_array_equal(reader.stats[field],
				original.stats[field])
		numpy.testing.assert_array_equal(reader.tile_sums(), original.tile_sums())
	assert os.path.getsize(copy) < os.path.getsize(path)


def test_add_file_between_own_writes(tmp_path, ref):
	"""A writer that wrote tracks, took another file's and wrote more keeps
	every group's values, refuses writes to the file's tracks, and refuses
	a file it already holds."""

	source = str(tmp_path / 'source.fw')
	with FigWigWriter(source, CHROM_SIZES, ['s0', 's1'], tile=TILE, level=6
			) as writer:
		writer.write('chr1', [0], numpy.stack([numpy.arange(200.0) % 7,
			numpy.arange(200.0) % 5])[None])
	path = str(tmp_path / 'both.fw')
	with FigWigWriter(path, CHROM_SIZES, ['a', 'b'], tile=TILE, groups=1,
			level=6) as writer:
		writer.write('chr1', [0], numpy.full((1, 1, 100), 1.0), tracks='a')
		writer.add_file(source)
		writer.write('chr1', [0], numpy.full((1, 1, 100), 2.0), tracks='b')
		with pytest.raises(ValueError, match="came from another file"):
			writer.write('chr1', [500], numpy.ones((1, 1, 5)), tracks='s0')
		with pytest.raises(ValueError, match="already one of the writer's"):
			writer.add_file(source)
	reader = FigWigReader(path, verify=True)
	assert [t['name'] for t in reader.tracks] == ['a', 'b', 's0', 's1']
	assert [t['group'] for t in reader.tracks] == [0, 1, 2, 2]
	y = reader.read('chr1', [0], 200, dtype='float64')[0]
	numpy.testing.assert_array_equal(y[0], [1.0] * 100 + [0.0] * 100)
	numpy.testing.assert_array_equal(y[1], [2.0] * 100 + [0.0] * 100)
	numpy.testing.assert_array_equal(y[2], numpy.arange(200.0) % 7)
	numpy.testing.assert_array_equal(y[3], numpy.arange(200.0) % 5)


def test_add_file_refuses_files_of_another_layout(tmp_path, ref):
	path = str(tmp_path / 'a.fw')
	write_reference(path, ref)
	for name, kwargs, match in (('tile', dict(tile=128), "tile 128"), ('level',
			dict(level=1), "level 1"), ('order', dict(order='tracks'),
			"order tracks")):
		other = str(tmp_path / (name + '.fw'))
		options = dict(tile=64, level=6, order='positions')
		options.update(kwargs)
		with FigWigWriter(other, CHROM_SIZES, ['n_' + name], **options):
			pass
		with pytest.raises(ValueError, match=match):
			with FigWigWriter(str(tmp_path / 'o.fw'), CHROM_SIZES, None, tile=64,
					level=6) as writer:
				writer.add_file(other)
	other = str(tmp_path / 'sizes.fw')
	with FigWigWriter(other, {'chr1': 5000, 'chr2': 3000, 'chrM': 70}, ['n'],
			tile=64, level=6):
		pass
	with pytest.raises(ValueError, match="other chromosomes or lengths.*chr2"):
		with FigWigWriter(str(tmp_path / 'o2.fw'), CHROM_SIZES, None, tile=64,
				level=6) as writer:
			writer.add_file(other)


def test_files_past_the_summary_cap_stay_readable_through_add_file(tmp_path, ref,
	monkeypatch):
	"""A file with no summaries is still read, `tile_sums` of it says so,
	and a writer that takes it keeps none for any track."""

	monkeypatch.setattr(figwig.figwig, '_SUMMARY_BYTES', 0)
	bare = str(tmp_path / 'bare.fw')
	write_reference(bare, ref)
	monkeypatch.undo()
	reader = FigWigReader(bare, verify=True)
	with pytest.raises(ValueError, match="keeps no summary"):
		reader.tile_sums()
	with pytest.raises(ValueError, match="keeps no summary"):
		reader.tile_sums(tracks='signal')
	assert reader.read('chr1', [0], 100).shape == (1, 5, 100)

	good = str(tmp_path / 'good.fw')
	write_figwig(good, CHROM_SIZES, ['extra'], 'chr1', [0], numpy.ones((1, 1,
		10)), tile=64)
	merged = str(tmp_path / 'merged.fw')
	main(['merge', good, bare, '-o', merged])
	both = FigWigReader(merged)
	with pytest.raises(ValueError, match="keeps no summary"):
		both.tile_sums()
	assert both.sums('chr1', [0], [10], tracks='extra')[0] == 10


def test_append_keeps_the_genome_and_may_replace_it(tmp_path, ref):
	path = str(tmp_path / 'g.fw')
	write_reference(path, ref)
	with FigWigWriter(path, None, ['more'], append=True):
		pass
	assert FigWigReader(path).genome == 'hg38'
	with FigWigWriter(path, None, ['more2'], append=True, genome='mm10'):
		pass
	assert FigWigReader(path).genome == 'mm10'
	assert [t['name'] for t in FigWigReader(path).tracks][-2:] == ['more',
		'more2']


###
# Commands


def test_bw2fw_missing_nan_and_back(tmp_path, capsys):
	"""A measured 0 stays apart from an uncovered base through bw2fw and
	fw2bw with --missing nan, and merges into the 0 without it."""

	sizes = {'chr1': 100}
	source = str(tmp_path / 'a.bw')
	write_bigwig(source, sizes, ['chr1'] * 2, [10, 30], numpy.array([3.0, 0.0]),
		ends=[20, 40], zooms=0, missing=numpy.nan)
	want = read_bigwig(source, 'chr1', [0], 100, missing=numpy.nan)[0]
	assert numpy.isnan(want[[5, 60]]).all() and want[35] == 0 and want[15] == 3

	nan_fw = str(tmp_path / 'nan.fw')
	main(['bw2fw', '-o', nan_fw, '--missing', 'nan', '-v', '-p', '2', source])
	assert 'group 1 of 1: a' in capsys.readouterr().err
	y = FigWigReader(nan_fw).read('chr1', [0], 100)[0, 0]
	assert numpy.isnan(y[[5, 60]]).all() and y[35] == 0 and y[15] == 3
	zero_fw = str(tmp_path / 'zero.fw')
	main(['bw2fw', '-o', zero_fw, source])
	numpy.testing.assert_array_equal(FigWigReader(zero_fw).read('chr1', [0],
		100)[0, 0], numpy.nan_to_num(want))

	back = tmp_path / 'back'
	main(['fw2bw', nan_fw, '-o', str(back), '--missing', 'nan'])
	numpy.testing.assert_array_equal(read_bigwig(str(back / 'a.bw'), 'chr1',
		[0], 100, missing=numpy.nan)[0], want)
	# Without --missing, fw2bw follows the file's `missing` attribute, nan
	# here, so the round trip is exact; --missing 0 drops the zero bases.
	same = tmp_path / 'same'
	main(['fw2bw', nan_fw, '-o', str(same)])
	numpy.testing.assert_array_equal(read_bigwig(str(same / 'a.bw'), 'chr1',
		[0], 100, missing=numpy.nan)[0], want)
	lossy = tmp_path / 'lossy'
	main(['fw2bw', nan_fw, '-o', str(lossy), '--missing', '0'])
	got = read_bigwig(str(lossy / 'a.bw'), 'chr1', [0], 100, missing=numpy.nan)[0]
	assert numpy.isnan(got[35]) and got[15] == 3


def test_track_names_with_separators_make_one_bigwig_each(tmp_path):
	path = str(tmp_path / 'n.fw')
	write_figwig(path, {'chr1': 100}, ['a/b', 'a_b2'], 'chr1', [0],
		numpy.ones((1, 2, 10)), tile=64)
	main(['fw2bw', path, '-o', str(tmp_path / 'out')])
	assert sorted(os.listdir(tmp_path / 'out')) == ['a_b.bw', 'a_b2.bw', 'n.tsv']
	assert read_bigwig(str(tmp_path / 'out' / 'a_b.bw'), 'chr1', [0], 10).sum(
		) == 10


def test_track_names_that_make_one_file_name_are_not_merged_silently(tmp_path):
	"""Tracks 'a/b' and 'a_b' both sanitize to a_b.bw; the second gets its
	track index appended, and the table says which file is whose."""

	clash = str(tmp_path / 'clash.fw')
	write_figwig(clash, {'chr1': 100}, ['a/b', 'a_b'], 'chr1', [0],
		numpy.stack([numpy.full(10, 1.0), numpy.full(10, 2.0)])[None], tile=64)
	main(['fw2bw', clash, '-o', str(tmp_path / 'clash')])
	bigwigs = sorted(f for f in os.listdir(tmp_path / 'clash') if f.endswith(
		'.bw'))
	assert bigwigs == ['a_b.1.bw', 'a_b.bw']
	assert read_bigwig(str(tmp_path / 'clash' / 'a_b.bw'), 'chr1', [0], 10).sum(
		) == 10
	assert read_bigwig(str(tmp_path / 'clash' / 'a_b.1.bw'), 'chr1', [0], 10
		).sum() == 20
	with open(tmp_path / 'clash' / 'clash.tsv') as handle:
		rows = [line.rstrip('\n').split('\t') for line in handle][1:]
	assert [(row[1], row[0]) for row in rows] == [('a/b', 'a_b.bw'), ('a_b',
		'a_b.1.bw')]


@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_info_bytes_account_for_every_byte(tmp_path, ref, order, capsys):
	path = str(tmp_path / 'i.fw')
	write_reference(path, ref, order=order)
	main(['info', '-b', path])
	text = capsys.readouterr().out
	fields = dict(line.split('\t')[:2] for line in text.splitlines() if '\t' in
		line and line.split('\t')[0] in ('bytes', 'chunk bytes', 'table and '
		'index bytes', 'unused bytes'))
	assert int(fields['unused bytes']) == 0
	assert int(fields['bytes']) == os.path.getsize(path)
	assert int(fields['bytes']) == 128 + int(fields['chunk bytes']) + int(fields[
		'table and index bytes'])


def test_info_stops_quietly_when_its_reader_goes(tmp_path):
	"""`figwig info file | head -1` ends with status 0 and no message."""

	path = str(tmp_path / 'many.fw')
	names = ['track_{:05d}_with_a_long_descriptive_name'.format(i) for i in
		range(4000)]
	write_figwig(path, {'chr1': 100}, names, 'chr1', [0], numpy.ones((1, 4000,
		4)), tile=64, groups=500)
	process = subprocess.Popen([sys.executable, '-m', 'figwig', 'info', path],
		env=ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	assert process.stdout.readline()
	process.stdout.close()
	error = process.stderr.read()
	process.stderr.close()
	assert process.wait(timeout=120) == 0 and error == b''


def test_reorder_keeps_exact_values_in_every_class(tmp_path):
	"""Float64 values, integers above 2**24 beside fractions and negative
	zero survive reorder to track order and back bit for bit."""

	path = str(tmp_path / 'c.fw')
	tracks = [{'name': 'f64', 'dtype': 'float64'}, {'name': 'big'}, {'name':
		'f32', 'dtype': 'float32'}, {'name': 'neg'}]
	rng = numpy.random.default_rng(3)
	values = numpy.zeros((1, 4, 300))
	values[0, 0, ::7] = rng.integers(1, 10 ** 9, len(values[0, 0, ::7])) + 1 / 3
	values[0, 1, 5::11] = rng.integers(2 ** 24, 2 ** 32, len(values[0, 1, 5::11]))
	values[0, 2, 3::5] = rng.normal(size=len(values[0, 2, 3::5])).astype(numpy.float32)
	values[0, 3, 9::13] = -rng.integers(1, 100000, len(values[0, 3, 9::13]))
	write_figwig(path, {'chr1': 300}, tracks, 'chr1', [0], values, tile=64,
		groups=2)
	want = FigWigReader(path).read('chr1', [0], 300, dtype='float64')
	for order in ('tracks', 'positions', 'tracks'):
		out = str(tmp_path / (order + '.fw'))
		main(['reorder', path, out, '--order', order, '-f'])
		got = FigWigReader(out, verify=True)
		assert got.order == order
		assert_identical(got.read('chr1', [0], 300, dtype='float64'), want)
		# The sums are added in another order, so they agree to rounding.
		numpy.testing.assert_allclose(got.stats['sum'], FigWigReader(path
			).stats['sum'], rtol=1e-14)
		numpy.testing.assert_array_equal(got.stats['nonzero'], FigWigReader(path
			).stats['nonzero'])
		path = out
	# Track order defaults to uncompressed chunks and position order to 6.
	assert FigWigReader(str(tmp_path / 'tracks.fw'))._level == 0
	assert FigWigReader(str(tmp_path / 'positions.fw'))._level == 6
	assert FigWigReader(str(tmp_path / 'positions.fw')).tile == \
		figwig.figwig._DEFAULT_TILE


###
# Entries, wide values, long varints and NaN in every chunk kind


@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_write_sparse_with_explicit_zeros_and_nan(tmp_path, order):
	"""An entry whose value is 0 writes nothing (a tile of only zeros takes
	no chunk), a NaN entry in a float track is kept, and both orders read
	the same bases."""

	path = str(tmp_path / 'z.fw')
	tracks = [{'name': 'ints'}, {'name': 'floats', 'dtype': 'float32'}]
	positions = numpy.array([3, 4, 70, 71, 130, 131, 200, 201])
	which = numpy.array([0, 1, 0, 1, 0, 1, 0, 1])
	values = numpy.array([0, numpy.nan, 5, 0, 0, 0, 7, -0.0])
	groups = [['ints', 'floats']] if order == 'positions' else 1
	with FigWigWriter(path, {'chr1': 256}, tracks, tile=64, order=order,
			groups=groups) as writer:
		writer.write_sparse('chr1', positions, which, values)
	reader = FigWigReader(path, verify=True)
	y = reader.read('chr1', [0], 256, dtype='float64')[0]
	want = numpy.zeros((2, 256))
	want[0, 70], want[0, 200], want[1, 4] = 5, 7, numpy.nan
	assert_identical(y, want)
	entries = reader.read_entries()
	assert entries.positions.tolist() == [70, 200, 4]
	assert entries.tracks.tolist() == [0, 0, 1]
	if order == 'positions':
		kinds = reader._group_index(0)[0]
		assert (kinds != 0).tolist() == [True, True, False, True]
	assert reader.stats['nonzero'].tolist() == [2, 0]  # NaN is not counted


@pytest.mark.parametrize('order', ['positions', 'tracks'])
@pytest.mark.parametrize('tile', [64, 32768, 65536])
def test_wide_values_and_long_varint_offsets(tmp_path, order, tile):
	"""Integers beyond 32 bits are kept exactly in a float64 track, and
	nonzero bases 20,000 apart in a tile of 32,768 or more need three-byte
	offsets, which read back, bin and list the same as in a small tile."""

	length = 200_000
	path = str(tmp_path / 'w.fw')
	positions = numpy.array([0, 20_000, 50_000, 70_000, 199_999])
	values = numpy.array([1.0, -2.0 ** 40, 2.0 ** 40 + 1, 3.25, 9.0])
	tracks = [{'name': 'wide', 'dtype': 'float64'}, {'name': 'small'}]
	groups = [['wide', 'small']] if order == 'positions' else 1
	with FigWigWriter(path, {'chr1': length}, tracks, tile=tile, order=order,
			groups=groups) as writer:
		writer.write_sparse('chr1', numpy.concatenate([positions, positions[::2]
			]), numpy.array([0] * 5 + [1] * 3), numpy.concatenate([values,
			[4, 5, 6]]))
	reader = FigWigReader(path, verify=True)
	truth = numpy.zeros((2, length))
	truth[0, positions] = values
	truth[1, positions[::2]] = [4, 5, 6]
	assert_identical(reader.read('chr1', [0], length, dtype='float64')[0], truth)
	rng = numpy.random.default_rng(2)
	starts = rng.integers(0, length - 40_000, 30)
	starts[:3] = [0, 19_990, 49_990]
	windows = reader.read('chr1', starts, 40_000, dtype='float64', bins=1000)
	want = numpy.stack([truth[:, s:s + 40_000].reshape(2, 40, 1000).sum(-1)
		for s in starts])
	numpy.testing.assert_array_equal(windows, want)
	entries = reader.read_entries(tracks='wide')
	numpy.testing.assert_array_equal(entries.positions, positions)
	numpy.testing.assert_array_equal(entries.values, values)
	assert reader.sums('chr1', [0], [length], tracks='wide', dtype='float64')[0
		] == values.sum()


def test_both_orders_store_the_same_values(tmp_path):
	"""An undeclared track that reads as float32 keeps an integer above
	2**24 exactly in a position-order file, where its chunk is float64, and
	rounded to float32 in a track-order file; the two orders agree in the
	track's own dtype, which is what the documentation promises."""

	reads = {}
	for order in ('positions', 'tracks'):
		path = str(tmp_path / (order + '.fw'))
		groups = [['a', 'b']] if order == 'positions' else 1
		with FigWigWriter(path, {'chr1': 100_000}, ['a', 'b'], tile=64, order=
				order, groups=groups) as writer:
			writer.write_sparse('chr1', [0, 50, 90_000, 90_001], ['a', 'a', 'a',
				'b'], [1, 2.0 ** 40 + 1, 0.5, 3.5])
		reader = FigWigReader(path)
		assert reader.tracks[0]['dtype'] == 'float32'
		reads[order] = reader.read('chr1', [0, 50], 1, tracks='a')
	assert reads['positions'].dtype == numpy.float32
	assert_identical(reads['positions'], reads['tracks'])
	assert reads['tracks'][1, 0] == numpy.float32(2.0 ** 40 + 1)


@pytest.fixture(params=['positions', 'tracks'])
def nan_file(request, tmp_path):
	"""Float32 tracks with NaN scattered in dense, sparse, runs and matrix
	chunks, and the truth."""

	rng = numpy.random.default_rng(request.param == 'tracks')
	sizes = {'chr1': 64 * 4 + 20}
	tracks = [{'name': 'f{}'.format(i), 'dtype': 'float32'} for i in range(4)]
	truth = numpy.zeros((4, 276))
	truth[0] = rng.normal(size=276).astype(numpy.float32)
	truth[1, ::50] = rng.normal(size=6).astype(numpy.float32)
	truth[2, :128] = numpy.repeat(rng.normal(size=8).astype(numpy.float32), 16)
	truth[3, 200:210] = 0.5
	for t, base in ((0, 3), (0, 100), (1, 100), (2, 70), (3, 205), (3, 270),
			(1, 255)):
		truth[t, base] = numpy.nan
	path = str(tmp_path / 'n.fw')
	groups = [[t['name'] for t in tracks]] if request.param == 'positions' else 1
	with FigWigWriter(path, sizes, tracks, tile=64, order=request.param,
			groups=groups, level=0) as writer:
		writer.write('chr1', [0], truth[None])
	return FigWigReader(path), truth


@pytest.mark.parametrize('mode', ['propagate', 'skip', 'count'])
def test_nan_modes_in_every_chunk_kind(nan_file, mode, scratch):
	reader, truth = nan_file
	rng = numpy.random.default_rng(4)
	starts = rng.integers(0, 276 - 64, 40)
	windows = numpy.stack([truth[:, s:s + 64] for s in starts])
	pieces = windows.reshape(40, 4, 8, 8)
	if mode == 'propagate':
		want = pieces.sum(-1)
	elif mode == 'skip':
		want = numpy.nansum(pieces, -1)
	else:
		want = (~numpy.isnan(pieces)).sum(-1)
	got = reader.read('chr1', starts, 64, bins=8, nan=mode, dtype='float64')
	numpy.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
	assert (numpy.isnan(got) == numpy.isnan(want)).all()

	ends = starts + rng.integers(0, 60, 40)
	sums = reader.sums('chr1', starts, ends, nan=mode, dtype='float64')
	cut = [truth[:, s:e] for s, e in zip(starts, ends)]
	if mode == 'propagate':
		want = numpy.stack([c.sum(1) for c in cut])
	elif mode == 'skip':
		want = numpy.stack([numpy.nansum(c, 1) for c in cut])
	else:
		want = numpy.stack([(~numpy.isnan(c)).sum(1) for c in cut])
	numpy.testing.assert_allclose(sums, want, rtol=1e-12, atol=1e-12)
	assert (numpy.isnan(sums) == numpy.isnan(want)).all()


def test_one_track_written_as_intervals_takes_one_value_each(tmp_path):
	path = str(tmp_path / 'one.fw')
	with FigWigWriter(path, CHROM_SIZES, ['a', 'b'], tile=64, groups=1) as writer:
		writer.write('chr1', [0, 10], numpy.array([1.0, 2.0]), ends=[5, 15],
			tracks='a')
		writer.write('chr1', [0], numpy.array([[7.0, 8.0, 9.0]]), tracks='b')
		with pytest.raises(ValueError, match="one channel per track, \\(1, 1\\), not \\(1, 2\\)"):
			writer.write('chr1', [100], numpy.ones((1, 2)), ends=[105],
				tracks='a')
	reader = FigWigReader(path)
	a = reader.read('chr1', [0], 16, tracks='a', dtype='float64')[0]
	numpy.testing.assert_array_equal(a, [1] * 5 + [0] * 5 + [2] * 5 + [0])
	numpy.testing.assert_array_equal(reader.read('chr1', [0], 4, tracks='b',
		dtype='float64')[0], [7, 8, 9, 0])


# The review round's additions: the header's own checksum and feature bits,
# metadata keys kept through an append, the append lock, the 2**53 limit,
# paired sums, and crafted chunks that once crashed the process.


def test_header_checksum_and_feature_bits(tmp_path, small):
	data, header = load(small['positions'])
	assert header['version'] == figwig.figwig._VERSION

	copy = bytearray(data)
	at = HEADER.fields['padding'][1]
	copy[at] ^= 1
	refuses(save(tmp_path, 'pad.fw', copy), "header does not match its checksum")

	copy = bytearray(data)
	put(copy, 'required', 4)
	seal(copy)
	refuses(save(tmp_path, 'req.fw', copy), r"needs a feature this figwig does "
		r"not know \(bits 0x4\)")

	copy = bytearray(data)
	put(copy, 'flags', 8)
	seal(copy)
	refuses(save(tmp_path, 'flag.fw', copy), "flags hold a bit this figwig does "
		"not know")

	copy = bytearray(data)
	put(copy, 'optional', 2 ** 32 - 1)
	seal(copy)
	path = save(tmp_path, 'opt.fw', copy)
	assert_identical(FigWigReader(path).read('chr1', [0], 100), FigWigReader(
		small['positions']).read('chr1', [0], 100))


def test_metadata_keys_survive_an_append(tmp_path, small):
	"""A top-level metadata key this figwig does not know is kept when the
	file is appended to, and the writer records its own version."""

	data, header = load(small['positions'])
	document = metadata_of(data, header)
	assert document['figwig'] == figwig.__version__
	document['attributes'] = {}
	document['later'] = {'k': [1, 2]}
	put_metadata(data, header, document)
	seal(data)
	path = save(tmp_path, 'keys.fw', data)
	reader = FigWigReader(path)
	assert reader.written_by == figwig.__version__
	assert reader._metadata['later'] == {'k': [1, 2]}

	with FigWigWriter(path, None, ['more'], append=True) as writer:
		writer.write('chr1', [0], numpy.ones((1, 1, 10)))
	reader = FigWigReader(path)
	assert reader._metadata['later'] == {'k': [1, 2]}
	assert reader.written_by == figwig.__version__
	assert [t['name'] for t in reader.tracks][-1] == 'more'


@pytest.mark.skipif(sys.platform == 'win32', reason="fcntl locks")
def test_a_second_appending_writer_is_refused(tmp_path, small):
	path = str(tmp_path / 'lock.fw')
	with open(small['positions'], 'rb') as source, open(path, 'wb') as copy:
		copy.write(source.read())
	with FigWigWriter(path, None, ['x'], append=True) as writer:
		with pytest.raises(ValueError, match="is being appended to by another"):
			FigWigWriter(path, None, ['y'], append=True)
		writer.write('chr1', [0], numpy.ones((1, 1, 10)))
	# The lock goes with the first writer.
	with FigWigWriter(path, None, ['y'], append=True) as writer:
		writer.write('chr1', [0], numpy.ones((1, 1, 10)))
	assert [t['name'] for t in FigWigReader(path).tracks][-2:] == ['x', 'y']


def test_integers_beyond_2_53_are_refused(tmp_path):
	path = str(tmp_path / 'big.fw')
	with FigWigWriter(path, {'chr1': 100}, ['a'], tile=64) as writer:
		writer.write('chr1', [0], numpy.array([[[2 ** 53, 1]]], dtype=numpy.int64))
		writer.write_sparse('chr1', [70], [0], numpy.array([2 ** 53],
			dtype=numpy.uint64))
		for values in (numpy.array([[[2 ** 53 + 1]]], dtype=numpy.int64),
				numpy.array([[[-2 ** 53 - 1]]], dtype=numpy.int64), numpy.array([[[
				2 ** 60]]], dtype=numpy.uint64)):
			with pytest.raises(ValueError, match=r"beyond 2\*\*53"):
				writer.write('chr1', [10], values)
			with pytest.raises(ValueError, match=r"beyond 2\*\*53"):
				writer.write_sparse('chr1', [10], [0], values.ravel())
	y = FigWigReader(path).read('chr1', [0], 100, dtype='float64')[0, 0]
	assert y[[0, 1, 70]].tolist() == [2 ** 53, 1, 2 ** 53] and y.sum() == 2 ** 54 + 1


@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_paired_sums(tmp_path, order, monkeypatch):
	"""sums(paired=True) sums interval i over track tracks[i] alone."""

	path = str(tmp_path / 'p.fw')
	rng = numpy.random.default_rng(5)
	values = rng.integers(0, 4, (1, 3, 1000)).astype(numpy.float64)
	values[0, 1, 100:900] = 0
	tracks = ['r0', 'r1', 'r2']
	write_figwig(path, {'chr1': 1000}, tracks, 'chr1', [0], values, tile=64,
		order=order, groups=[tracks[:2], tracks[2:]] if order == 'positions'
		else 1)
	reader = FigWigReader(path)
	starts = rng.integers(0, 900, 40)
	ends = starts + rng.integers(1, 100, 40)
	which = rng.integers(0, 3, 40)
	want = numpy.array([values[0, t, s:e].sum() for t, s, e in zip(which,
		starts, ends)])
	for named in (which, [tracks[t] for t in which]):
		got = reader.sums('chr1', starts, ends, tracks=named, paired=True)
		assert got.shape == (40,) and got.dtype == numpy.float64
		numpy.testing.assert_array_equal(got, want)
	every = reader.sums('chr1', starts, ends)
	numpy.testing.assert_array_equal(every[numpy.arange(40), which], want)
	assert reader.sums('chr1', [], [], tracks=[], paired=True).shape == (0,)
	got = reader.sums('chr1', starts, ends, tracks=which, paired=True,
		dtype='int32')
	assert got.dtype == numpy.int32 and got.tolist() == want.tolist()
	# Batches of two tracks, and tracks spread over two groups when the
	# file has them, give the same sums.
	monkeypatch.setattr(figwig.figwig, '_PAIRED_TRACKS', 2)
	numpy.testing.assert_array_equal(reader.sums('chr1', starts, ends,
		tracks=which, paired=True), want)
	monkeypatch.setattr(figwig.figwig, '_PAIRED_TRACKS', 16)
	with pytest.raises(ValueError, match="one track per interval"):
		reader.sums('chr1', starts, ends, paired=True)
	with pytest.raises(ValueError, match="one track per interval"):
		reader.sums('chr1', starts, ends, tracks=[0, 1], paired=True)
	with pytest.raises(ValueError, match="is out of range"):
		reader.sums('chr1', starts, ends, tracks=[3] * 40, paired=True)


def test_crafted_matrix_chunk_fails_without_a_crash():
	"""A matrix chunk whose middle row pointer is far past its values once
	read memory past the chunk; it is now a failed chunk."""

	from figwig._figwig_kernels import _check_chunks

	rows = struct.pack('<I', 2) + struct.pack('<2I', 0, 1)
	for ptr in ((0, 2 ** 31, 2), (0, 1, 2 ** 32 - 1), (1, 2, 2), (0, 2, 1)):
		raw = rows + struct.pack('<3I', *ptr) + struct.pack('<2H', 0, 1) + \
			b'\1\1\0\0'
		data = numpy.frombuffer(raw, dtype=numpy.uint8).copy()
		failed = numpy.zeros(1, dtype=numpy.int64)
		n = _check_chunks(16, data, numpy.array([0]), numpy.array([len(raw)]),
			numpy.array([_MATRIX], dtype=numpy.uint8), 5, numpy.arange(5), failed)
		assert n == 1 and failed[0] == 1, ptr
	good = rows + struct.pack('<3I', 0, 1, 2) + struct.pack('<2H', 0, 1) + \
		b'\1\1\0\0'
	data = numpy.frombuffer(good, dtype=numpy.uint8).copy()
	failed = numpy.zeros(1, dtype=numpy.int64)
	assert _check_chunks(16, data, numpy.array([0]), numpy.array([len(good)]),
		numpy.array([_MATRIX], dtype=numpy.uint8), 5, numpy.arange(5), failed) == 0


def test_only_regular_files_are_opened(tmp_path):
	with pytest.raises(ValueError, match="is not a regular file"):
		FigWigReader(str(tmp_path))
	with pytest.raises(ValueError, match="absent.fw cannot be read: "):
		FigWigReader(str(tmp_path / 'absent.fw'))


def test_oversized_chunk_sizes_are_refused(tmp_path, small):
	data, header = load(small['positions'])
	offset, runs, fields, summary = group_record(data, header)
	f = [x.copy() for x in fields]
	f[3][0] = 2 ** 31
	refuses(save(tmp_path, 'raw.fw', rebuild(data, header, offset, runs, f,
		summary)), "larger than any chunk of 5 tracks")


def test_infinite_values_are_refused(tmp_path):
	"""No figwig file holds an infinite value: `write` and `write_sparse`
	refuse one before writing anything, NaN stays allowed."""

	path = str(tmp_path / 'inf.fw')
	with FigWigWriter(path, {'chr1': 100}, [{'name': 'a', 'dtype': 'float32'}, {
			'name': 'b', 'dtype': 'float64'}], tile=64) as writer:
		writer.write('chr1', [0], numpy.array([[[1.0, numpy.nan], [2.0, 0.0]]]))
		for bad in (numpy.inf, -numpy.inf):
			with pytest.raises(ValueError, match="finite or NaN"):
				writer.write('chr1', [10], numpy.array([[[bad], [1.0]]]))
			with pytest.raises(ValueError, match="finite or NaN"):
				writer.write_sparse('chr1', [10], [1], numpy.array([bad]))
			with pytest.raises(ValueError, match="finite or NaN"):
				writer.write('chr1', [10], numpy.array([[[1.0], [bad]]], dtype=numpy.float32))
	y = FigWigReader(path).read('chr1', [0], 100, dtype='float64')[0]
	assert y[0, 0] == 1 and numpy.isnan(y[0, 1]) and y[1, 0] == 2
	assert not numpy.isinf(y).any()
