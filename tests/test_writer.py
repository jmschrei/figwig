# test_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import sys
import time
import struct
import threading
import pathlib
import zlib
import types

import numpy
import pytest
import deflate
import pybigtools

import figwig
import figwig._kernels
import figwig.writer

# pyBigWig, the writer's byte-for-byte oracle, does not build on Windows, so
# the tests that compare against it run everywhere else.
if sys.platform != 'win32':
	import pyBigWig

needs_pybigwig = pytest.mark.skipif(sys.platform == 'win32',
	reason='pyBigWig does not build on Windows')

# On Windows, neither zlib's library nor libdeflate's functions, from the
# deflate package's extension module, can be loaded with ctypes, so blocks
# are compressed by Python's zlib module and engine='libdeflate' raises. The
# tests of those libraries, and of files written with libdeflate, run
# everywhere else. The pytest header says which libraries loaded.
needs_c_libraries = pytest.mark.skipif(sys.platform == 'win32',
	reason="zlib's library and libdeflate's functions do not load on Windows")


def blocks_of(sizes, random_state=0):
	"""Blocks of these lengths, as (data, bounds): random bytes with few
	distinct values, so that they compress, and one repetitive block."""

	rng = numpy.random.default_rng(random_state)
	blocks = [rng.integers(0, 20, size=n, dtype=numpy.uint8).tobytes() for n in
		sizes]
	blocks.append(b'ACGT' * 250)
	data = numpy.frombuffer(b''.join(blocks), dtype=numpy.uint8)
	bounds = numpy.cumsum([0] + [len(block) for block in blocks]).astype(
		numpy.int64)
	return blocks, data, bounds


def split_streams(out, sizes):
	return numpy.split(out[:sizes.sum()], numpy.cumsum(sizes)[:-1])


##


@needs_c_libraries
@pytest.mark.parametrize('level', [1, 6, 9])
def test_deflate_blocks_zlib(level):
	"""Each block is compressed by compress2() into the stream zlib.compress
	gives at the same level, the streams are back to back in `out`, and an
	empty block is a stream of its own."""

	blocks, data, bounds = blocks_of([0, 1, 24, 32768, 32760])
	compress2, ulong = figwig._kernels._zlib_compress()
	out = numpy.empty(bounds[-1] * 2 + 1000, dtype=numpy.uint8)
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)

	status = figwig._kernels._deflate_blocks_zlib(compress2, level, data, bounds,
		out, sizes, numpy.zeros(1, dtype=ulong))

	assert status == -1
	streams = split_streams(out, sizes)
	assert [stream.tobytes() for stream in streams] == [zlib.compress(block,
		level) for block in blocks]


@needs_c_libraries
@pytest.mark.parametrize('level', [1, 6, 12])
def test_deflate_blocks_libdeflate(level):
	blocks, data, bounds = blocks_of([0, 1, 24, 32768, 32760])
	alloc, compress, free = figwig._kernels._libdeflate()
	out = numpy.empty(bounds[-1] * 2 + 1000, dtype=numpy.uint8)
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)

	compressor = alloc(level)
	try:
		status = figwig._kernels._deflate_blocks_libdeflate(compress, compressor,
			data, bounds, out, sizes)
	finally:
		free(compressor)

	assert status == -1
	streams = split_streams(out, sizes)
	assert [stream.tobytes() for stream in streams] == [deflate.zlib_compress(
		block, level) for block in blocks]
	assert [zlib.decompress(stream.tobytes()) for stream in streams] == blocks


@needs_c_libraries
def test_deflate_blocks_report_the_block_that_does_not_fit():
	"""When what is left of `out` cannot hold a block's stream, both kernels
	stop and return that block, here the fourth, after the first three have
	used 31 of the 100 bytes."""

	blocks, data, bounds = blocks_of([0, 1, 24, 32768])
	out = numpy.empty(100, dtype=numpy.uint8)

	compress2, ulong = figwig._kernels._zlib_compress()
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)
	assert figwig._kernels._deflate_blocks_zlib(compress2, 6, data, bounds, out,
		sizes, numpy.zeros(1, dtype=ulong)) == 3

	alloc, compress, free = figwig._kernels._libdeflate()
	sizes = numpy.zeros(len(blocks), dtype=numpy.int64)
	compressor = alloc(6)
	try:
		assert figwig._kernels._deflate_blocks_libdeflate(compress, compressor,
			data, bounds, out, sizes) == 3
	finally:
		free(compressor)


@needs_c_libraries
def test_zlib_compress_loader(monkeypatch):
	"""compress2() is looked for in the same library, the same way, as
	uncompress(), and without it the loader gives None."""

	import ctypes
	import ctypes.util

	loaded = figwig._kernels._load_zlib_compress()
	function, ulong = loaded
	assert function.restype is ctypes.c_int
	assert function.argtypes[-1] is ctypes.c_int
	assert ulong == numpy.dtype(ctypes.c_ulong)

	def library(name):
		raise OSError(name)

	monkeypatch.setattr(ctypes, 'CDLL', library)
	monkeypatch.setattr(ctypes.util, 'find_library', lambda name: None)
	assert figwig._kernels._load_zlib_compress() is None


def test_libdeflate_loader_without_deflate(monkeypatch):
	"""Without the `deflate` package, or when its library does not export
	libdeflate's functions, the loader gives None, and blocks are compressed
	with zlib."""

	with monkeypatch.context() as patch:
		patch.setitem(sys.modules, 'deflate._deflate', None)
		assert figwig._kernels._load_libdeflate() is None

	# A shared library that loads but has none of libdeflate's functions: one
	# of numpy's extension modules, which is a file in every Python, where
	# _ctypes is built into some interpreters.
	monkeypatch.setattr(deflate, '_deflate', types.SimpleNamespace(
		__file__=numpy.random.mtrand.__file__))
	assert figwig._kernels._load_libdeflate() is None


## Helpers: the same calls through figwig and through pyBigWig, and a parser
## of the parts of a bigWig that a writer lays out.


ZOOM_RECORD = numpy.dtype([('tid', '<u4'), ('start', '<u4'), ('end', '<u4'),
	('count', '<u4'), ('min', '<f4'), ('max', '<f4'), ('sum', '<f4'),
	('sum_squares', '<f4')])


def dense_runs(start, values, missing=0.0):
	"""The (start, values) runs of a dense array that are written."""

	values = numpy.asarray(values, dtype=numpy.float32)
	keep = ~numpy.isnan(values)
	if not numpy.isnan(missing):
		keep &= values != numpy.float32(missing)
	edges = numpy.flatnonzero(numpy.diff(keep.astype(numpy.int8), prepend=0,
		append=0))
	return [(start + int(a), values[a:b]) for a, b in zip(edges[::2],
		edges[1::2])]


def write_figwig(path, chroms, calls, **kwargs):
	"""Make each call, ('intervals', chrom, starts, ends, values), ('bases',
	chrom, positions, values) or ('dense', chrom, start, values), with
	BigWigWriter.write: single bases as windows of width 1, and a dense
	array as one window. Intervals and single bases are written with
	missing=NaN, so that every one is written, as pyBigWig writes them."""

	with figwig.BigWigWriter(path, chroms, **kwargs) as writer:
		for call in calls:
			if call[0] == 'intervals':
				writer.write(call[1], call[2], call[4], ends=call[3],
					missing=numpy.nan)
			elif call[0] == 'bases':
				writer.write(call[1], call[2], numpy.asarray(call[3])[:, None],
					missing=numpy.nan)
			else:
				writer.write(call[1], [call[2]], numpy.asarray(call[3])[None])


def write_pybigwig(path, chroms, calls, zooms):
	"""Make the same calls with pyBigWig, a dense array as one fixedStep call
	per run of written bases."""

	bw = pyBigWig.open(str(path), 'w')
	bw.addHeader(list(chroms.items()) if isinstance(chroms, dict) else chroms,
		maxZooms=zooms)
	for call in calls:
		if call[0] == 'intervals':
			bw.addEntries([call[1]] * len(call[2]), numpy.asarray(call[2]),
				ends=numpy.asarray(call[3]), values=numpy.asarray(call[4],
				dtype=numpy.float64))
		elif call[0] == 'bases':
			bw.addEntries(call[1], numpy.asarray(call[2]), values=numpy.asarray(
				call[3], dtype=numpy.float64), span=1)
		else:
			for start, values in dense_runs(call[2], call[3]):
				bw.addEntries(call[1], start, values=values.astype(numpy.float64),
					span=1, step=1)
	bw.close()


def index_leaves(buf, offset):
	leaves = []

	def node(position):
		is_leaf, _, count = struct.unpack_from('<BBH', buf, position)
		position += 4
		for _ in range(count):
			if is_leaf:
				leaves.append(struct.unpack_from('<IIIIQQ', buf, position))
				position += 32
			else:
				node(struct.unpack_from('<IIIIQ', buf, position)[4])
				position += 24

	node(offset + 48)
	return leaves


def parse(path):
	"""The header, total summary, chromosome tree, data blocks and zoom
	levels of a bigWig, each decompressed, for field by field comparison."""

	buf = pathlib.Path(path).read_bytes()
	header = struct.unpack_from('<IHHQQQHHQQIQ', buf, 0)
	n_levels, chrom_tree, data, index = header[2], header[3], header[4], header[5]
	summary = buf[header[9]:header[9] + 40]

	blocks = []
	if index:
		for leaf in index_leaves(buf, index):
			raw = zlib.decompress(buf[leaf[4]:leaf[4] + leaf[5]])
			blocks.append((leaf[:4], raw))

	levels = []
	for i in range(n_levels):
		reduction, _, zoom_data, zoom_index = struct.unpack_from('<IIQQ', buf,
			64 + 24 * i)
		leaves = index_leaves(buf, zoom_index)
		records = [numpy.frombuffer(zlib.decompress(buf[o:o + s]), ZOOM_RECORD)
			for *_, o, s in leaves]
		levels.append({'reduction': reduction, 'n_blocks': struct.unpack_from(
			'<I', buf, zoom_data)[0], 'entries': [leaf[:4] for leaf in leaves],
			'records': records, 'items_per_slot': struct.unpack_from('<I', buf,
			zoom_index + 40)[0]})

	return {'header': header, 'summary': summary, 'chrom_tree': buf[chrom_tree:
		data], 'n_blocks': struct.unpack_from('<Q', buf, data)[0], 'blocks':
		blocks, 'levels': levels, 'magic': buf[-4:]}


def random_bases(rng, length, n, values='counts'):
	positions = numpy.sort(rng.choice(length, n, replace=False)).astype(
		numpy.int64)
	if values == 'counts':
		return positions, rng.integers(1, 9, n).astype(numpy.float64)
	return positions, rng.normal(0, 3, n)


def random_intervals(rng, length, n):
	cuts = numpy.sort(rng.choice(length, 2 * n, replace=False)).astype(
		numpy.int64)
	return cuts[::2], cuts[1::2], rng.normal(0, 3, n)


CHROMS = {'chr1': 3_000_000, 'chr2': 1_000_000, 'chrM': 16_569}


@pytest.fixture(params=[(1 << 20, 64, 8, 64, 1 << 16), (5000, 3, 2, 2, 777),
	(1, 1, 1, 1, 1), (1 << 20, 64, 8, 64, 5)], ids=['default_batches',
	'small_batches', 'tiny_batches', 'small_scratch'])
def writer_batching(request, monkeypatch):
	"""Lay items out in batches of this many, read blocks back for the zoom
	levels this many at a time and up to this many chunks ahead, compress
	zoom blocks this many to a task, and let the record kernel write this
	many records before it stops. The fixtures are smaller than a default
	batch, so the small sizes are what reach the batch boundaries.
	'small_scratch' stops the kernel part way through chunks of the read-back
	that hold more than one chromosome."""

	items, zoom_blocks, zoom_ahead, zoom_compress, zoom_scratch = request.param
	monkeypatch.setattr(figwig.writer, '_BATCH_ITEMS', items)
	monkeypatch.setattr(figwig.writer, '_ZOOM_CHUNK_BLOCKS', zoom_blocks)
	monkeypatch.setattr(figwig.writer, '_ZOOM_READ_AHEAD', zoom_ahead)
	monkeypatch.setattr(figwig.writer, '_ZOOM_COMPRESS_BLOCKS', zoom_compress)
	monkeypatch.setattr(figwig.writer, '_ZOOM_SCRATCH_RECORDS', zoom_scratch)
	return request.param


def calls_of(kind, random_state=0):
	"""A sequence of calls of one kind over two chromosomes, chr1 in two calls
	and chrM left without values. Enough items for several data blocks."""

	rng = numpy.random.default_rng(random_state)
	if kind == 'bases':
		p1, v1 = random_bases(rng, 1_500_000, 6000)
		p2, v2 = random_bases(rng, 1_500_000, 3000, 'normal')
		p3, v3 = random_bases(rng, 1_000_000, 5000)
		return [('bases', 'chr1', p1, v1), ('bases', 'chr1', p2 + 1_500_000, v2),
			('bases', 'chr2', p3, v3)]
	elif kind == 'intervals':
		s1, e1, v1 = random_intervals(rng, 1_500_000, 4000)
		s2, e2, v2 = random_intervals(rng, 1_500_000, 2000)
		s3, e3, v3 = random_intervals(rng, 1_000_000, 3000)
		return [('intervals', 'chr1', s1, e1, v1), ('intervals', 'chr1', s2 +
			1_500_000, e2 + 1_500_000, v2), ('intervals', 'chr2', s3, e3, v3)]
	else:
		d1 = rng.normal(0, 1, 20_000).astype(numpy.float32)
		d1[rng.random(d1.size) < 0.002] = 0
		d2 = rng.normal(0, 1, 9000).astype(numpy.float32)
		d3 = numpy.where(rng.random(30_000) < 0.01, 0, rng.normal(0, 1, 30_000))
		return [('dense', 'chr1', 1000, d1), ('dense', 'chr1', 21_000, d2),
			('dense', 'chr2', 5, d3.astype(numpy.float32))]


##


@needs_pybigwig
@pytest.mark.parametrize('kind', ['bases', 'intervals'])
@pytest.mark.parametrize('n_jobs', [1, 3])
def test_bytes_identical_to_pybigwig(tmp_path, writer_batching, kind, n_jobs):
	"""Without zoom levels, a file of single bases or intervals written with
	zlib at level 6 is byte for byte the file pyBigWig writes from the same
	calls, whatever the number of threads or the batch size: the header, the
	total summary, the chromosome tree, the data blocks and the index."""

	calls = calls_of(kind)
	write_figwig(tmp_path / 'a.bw', CHROMS, calls, zooms=0, engine='zlib',
		level=6, n_jobs=n_jobs)
	write_pybigwig(tmp_path / 'b.bw', CHROMS, calls, zooms=0)

	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


@needs_pybigwig
def test_mixed_sections_match_pybigwig_but_for_empty_blocks(tmp_path):
	"""Intervals, then single bases, then intervals again on one chromosome
	each start a new data block, as they do in pyBigWig, and calls of the same
	type in a row continue one. Every block is pyBigWig's, byte for byte, but
	libBigWig also writes an empty block, a section header without items,
	where it flushes its buffer twice on changing section type: before single
	bases or a dense array that follow another type, and before intervals on
	a new chromosome that follow another type. Those are not written here."""

	rng = numpy.random.default_rng(3)
	s1, e1, v1 = random_intervals(rng, 100_000, 50)
	p2, v2 = random_bases(rng, 100_000, 70)
	p3, v3 = random_bases(rng, 100_000, 20)
	s4, e4, v4 = random_intervals(rng, 100_000, 30)
	calls = [('intervals', 'chr1', s1, e1, v1), ('bases', 'chr1', p2 + 100_000,
		v2), ('bases', 'chr1', p3 + 200_000, v3), ('intervals', 'chr1', s4 +
		300_000, e4 + 300_000, v4), ('bases', 'chr2', p2, v2)]

	write_figwig(tmp_path / 'a.bw', CHROMS, calls, zooms=0, engine='zlib',
		level=6, n_jobs=1)
	write_pybigwig(tmp_path / 'b.bw', CHROMS, calls, zooms=0)
	a, b = parse(tmp_path / 'a.bw'), parse(tmp_path / 'b.bw')

	empty = [block for block in b['blocks'] if len(block[1]) == 24]
	assert len(empty) == 2
	assert all(struct.unpack_from('<H', raw, 22)[0] == 0 for _, raw in empty)
	assert a['blocks'] == [block for block in b['blocks'] if len(block[1]) > 24]
	assert len(a['blocks']) == a['n_blocks'] == 4
	assert a['chrom_tree'] == b['chrom_tree']

	# libBigWig adds span * pow(value, 2) to the sum of squares, which the
	# macOS build of pyBigWig computes with a fused multiply-add, rounding
	# once where figwig and the Linux build round twice.
	assert a['summary'][:32] == b['summary'][:32]
	assert struct.unpack('<d', a['summary'][32:]) == pytest.approx(
		struct.unpack('<d', b['summary'][32:]), rel=1e-12)


@needs_pybigwig
@pytest.mark.parametrize('values', [[5.0, 1.0, 2.0], [-3.0, -1.0, -2.0],
	[0.0, 0.0]], ids=['first_largest', 'negative', 'zero'])
def test_header_maximum(tmp_path, values):
	"""The header's maximum is the largest value. libBigWig's is not when the
	first value is the largest, which it compares only with the minimum, or
	when no value is positive, where it keeps the smallest positive double it
	starts from. Everything else in the header matches pyBigWig's."""

	calls = [('bases', 'chr1', numpy.arange(len(values)) * 10, values)]
	write_figwig(tmp_path / 'a.bw', CHROMS, calls, zooms=0, engine='zlib')
	write_pybigwig(tmp_path / 'b.bw', CHROMS, calls, zooms=0)

	a = struct.unpack('<Qdddd', parse(tmp_path / 'a.bw')['summary'])
	b = struct.unpack('<Qdddd', parse(tmp_path / 'b.bw')['summary'])
	assert a[2] == max(values) != b[2]
	assert a[:2] == b[:2] and a[3:] == b[3:]
	assert pyBigWig.open(str(tmp_path / 'a.bw')).header()['maxVal'] == max(
		values)


@needs_pybigwig
@pytest.mark.parametrize('zooms', [0, 10])
def test_bytes_identical_to_pybigwig_without_values(tmp_path, zooms):
	"""A file with chromosomes but no values has no index and no zoom levels,
	but the space for the zoom headers is kept, as in pyBigWig's."""

	write_figwig(tmp_path / 'a.bw', CHROMS, [], zooms=zooms, engine='zlib')
	write_pybigwig(tmp_path / 'b.bw', CHROMS, [], zooms=zooms)
	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


@needs_pybigwig
def test_bytes_identical_to_pybigwig_many_chromosomes(tmp_path):
	"""With more than 32,767 chromosomes, the chromosome tree has a root over
	several leaves, laid out as libBigWig lays it out."""

	chroms = {'ctg{:05d}'.format(i): 1000 + i for i in range(40_000)}
	calls = [('bases', 'ctg00007', numpy.array([5, 9]), numpy.array([1.0, 2.0])),
		('bases', 'ctg39999', numpy.array([0]), numpy.array([3.0]))]

	write_figwig(tmp_path / 'a.bw', chroms, calls, zooms=0, engine='zlib')
	write_pybigwig(tmp_path / 'b.bw', chroms, calls, zooms=0)
	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


@needs_pybigwig
def test_fixedstep_matches_pybigwig_but_for_block_ends(tmp_path,
	writer_batching):
	"""A dense window whose runs are long enough to be written as fixedStep
	is written as pyBigWig writes one fixedStep call per run of its bases
	that are not 0, with two contiguous windows continuing one run, except
	that each block's end is the end of its last item. libBigWig puts the
	end of the last block of each call 6 bases further on, in the section
	header and in the index."""

	calls = calls_of('dense')
	write_figwig(tmp_path / 'a.bw', CHROMS, calls, zooms=0, engine='zlib',
		n_jobs=2)
	write_pybigwig(tmp_path / 'b.bw', CHROMS, calls, zooms=0)
	a, b = parse(tmp_path / 'a.bw'), parse(tmp_path / 'b.bw')

	assert a['summary'] == b['summary']
	assert a['chrom_tree'] == b['chrom_tree']
	assert a['n_blocks'] == b['n_blocks'] == len(a['blocks'])

	shifted = 0
	for (entry_a, raw_a), (entry_b, raw_b) in zip(a['blocks'], b['blocks']):
		tid, start, end, step, span, kind, _, count = struct.unpack_from(
			'<IIIIIBBH', raw_a)
		assert (kind, step, span) == (3, 1, 1)
		assert end == start + count
		assert entry_a == (tid, start, tid, end)
		assert raw_a[:8] == raw_b[:8] and raw_a[12:] == raw_b[12:]

		end_b = struct.unpack_from('<I', raw_b, 8)[0]
		assert end_b in (end, end + 6) and entry_b == (tid, start, tid, end_b)
		shifted += end_b != end

	assert 0 < shifted < len(a['blocks'])


@needs_pybigwig
def test_zoom_levels_match_pybigwig(tmp_path, writer_batching):
	"""The zoom levels pyBigWig writes are written here with the same
	reductions, records, blocks and index entries, but for the sums of the
	last record of each zoom block, which libBigWig leaves at 0. Those are
	checked against sums computed here from the values. Coarser levels, which
	libBigWig stops before once a level needs as many blocks as the one
	before it, are written while they have fewer records than the last."""

	calls = calls_of('bases', random_state=1) + [('intervals', 'chrM', [0, 9000],
		[5000, 16_000], [2.5, -1.0])]
	write_figwig(tmp_path / 'a.bw', CHROMS, calls, zooms=10, engine='zlib',
		n_jobs=4)
	write_pybigwig(tmp_path / 'b.bw', CHROMS, calls, zooms=10)
	a, b = parse(tmp_path / 'a.bw'), parse(tmp_path / 'b.bw')

	assert a['summary'] == b['summary']
	assert a['blocks'] == [block for block in b['blocks'] if len(block[1]) > 24]
	assert len(a['levels']) > len(b['levels']) > 1

	items = [(CHROMS_IDS[call[1]], int(s), int(e), numpy.float32(v)) for call in
		calls for s, e, v in (zip(call[2], call[3], call[4]) if call[0] ==
		'intervals' else zip(call[2], numpy.asarray(call[2]) + 1, call[3]))]

	for level_a, level_b in zip(a['levels'], b['levels']):
		assert level_a['reduction'] == level_b['reduction']
		assert level_a['entries'] == level_b['entries']
		assert level_a['items_per_slot'] == 1024
		assert level_a['n_blocks'] == len(level_a['records'])

		for block_a, block_b in zip(level_a['records'], level_b['records']):
			assert len(block_a) == len(block_b) <= 1023
			for field in ('tid', 'start', 'end', 'count', 'min', 'max'):
				numpy.testing.assert_array_equal(block_a[field], block_b[field])

			# Every record but the last of a block matches pyBigWig's exactly.
			numpy.testing.assert_array_equal(block_a[:-1].view('<u4'),
				block_b[:-1].view('<u4'))
			assert block_b[-1]['sum'] == 0 and block_b[-1]['sum_squares'] == 0
			assert (block_a[-1]['sum'], block_a[-1]['sum_squares']) == \
				record_sums(items, block_a[-1])

	counts = [sum(len(block) for block in level['records']) for level in
		a['levels']]
	assert counts == sorted(counts, reverse=True) and len(set(counts)) == len(
		counts)


CHROMS_IDS = {name: i for i, name in enumerate(CHROMS)}


def record_sums(items, record):
	"""A zoom record's sum and sum of squares, from the values of the items it
	covers, adding each overlap's float32 product as libBigWig does."""

	total, squares = 0.0, 0.0
	for tid, start, end, value in items:
		if tid != record['tid'] or end <= record['start'] or start >= \
				record['end']:
			continue

		overlap = min(end, int(record['end'])) - max(start, int(record['start']))
		total += float(numpy.float32(overlap) * value)
		squares += overlap * float(value) ** 2

	return numpy.float32(total), numpy.float32(squares)


@needs_pybigwig
def test_zoom_levels_read_by_pybigwig(tmp_path):
	"""pyBigWig takes the statistics of a whole chromosome from the zoom
	levels, and on a file written here they agree with the exact ones to
	float32 rounding. On pyBigWig's own file of the same values they do not,
	because of the records it leaves without a sum."""

	rng = numpy.random.default_rng(5)
	p1, v1 = random_bases(rng, 3_000_000, 20_000)
	p2, v2 = random_bases(rng, 1_000_000, 9000, 'normal')
	figwig.write_bigwig(tmp_path / 'a.bw', CHROMS, ['chr1'] * len(p1) + ['chr2'] *
		len(p2), numpy.concatenate([p1, p2]), numpy.concatenate([v1, v2])[:, None])
	write_pybigwig(tmp_path / 'b.bw', CHROMS, [('bases', 'chr1', p1, v1),
		('bases', 'chr2', p2, v2)], zooms=10)

	bw = pyBigWig.open(str(tmp_path / 'a.bw'))
	assert bw.header()['nLevels'] > 3
	for chrom in ('chr1', 'chr2'):
		for kind in ('mean', 'min', 'max', 'coverage', 'std'):
			numpy.testing.assert_allclose(bw.stats(chrom, type=kind, nBins=1),
				bw.stats(chrom, type=kind, nBins=1, exact=True), rtol=1e-6)

	bw = pyBigWig.open(str(tmp_path / 'b.bw'))
	zoomed = bw.stats('chr2', type='mean', nBins=1)[0]
	exact = bw.stats('chr2', type='mean', nBins=1, exact=True)[0]
	assert abs(zoomed - exact) > 1e-3 * abs(exact)


## Reading back


def read_back(path, chrom, width, missing=0.0):
	"""Every base of a chromosome, read by figwig and by pybigtools."""

	y = figwig.BigWigReader(str(path)).read(chrom, [0], width=width,
		missing=missing)
	bw = pybigtools.open(str(path))
	reference = numpy.empty(width, dtype=numpy.float64)
	bw.values(chrom, 0, width, missing=float(missing), arr=reference)
	return y[0], reference.astype(numpy.float32)


def assert_bits(a, b):
	assert a.dtype == b.dtype == numpy.float32
	nan_a, nan_b = numpy.isnan(a), numpy.isnan(b)
	numpy.testing.assert_array_equal(nan_a, nan_b)
	numpy.testing.assert_array_equal(a.view('<u4')[~nan_a], b.view('<u4')[~nan_b])


@pytest.mark.parametrize('engine', ['zlib', pytest.param('libdeflate',
	marks=needs_c_libraries), 'isal'])
def test_round_trip(tmp_path, engine):
	"""Intervals, single bases and a dense window, with each engine, read
	back bit for bit with figwig and with pybigtools, zoom levels
	included."""

	rng = numpy.random.default_rng(7)
	length = 200_000
	expected = numpy.zeros(length, dtype=numpy.float32)

	s, e, v = random_intervals(rng, 60_000, 500)
	p, w = random_bases(rng, 60_000, 800, 'normal')
	d = rng.normal(0, 1, 60_000).astype(numpy.float32)
	d[rng.random(d.size) < 0.3] = 0
	for a, b, value in zip(s, e, v.astype(numpy.float32)):
		expected[a:b] = value
	expected[60_000 + p] = w.astype(numpy.float32)
	expected[130_000:190_000] = d

	with figwig.BigWigWriter(tmp_path / 'a.bw', {'chr1': length, 'chr2': 10},
			engine=engine, level=1, n_jobs=2) as writer:
		writer.write('chr1', s, v, ends=e)
		writer.write('chr1', p + 60_000, w[:, None])
		writer.write('chr1', [130_000], d[None])

	y, reference = read_back(tmp_path / 'a.bw', 'chr1', length)
	assert_bits(y, expected)
	assert_bits(reference, expected)


@pytest.mark.parametrize('missing', [0.0, -1.0, numpy.nan])
def test_dense_missing(tmp_path, missing):
	"""Bases equal to `missing`, and NaN, are not written, so reading with the
	same `missing` gives the array back, NaN read as `missing`."""

	values = numpy.array([0, 1.5, -1, numpy.nan, 0, 0, 2, -1, 3],
		dtype=numpy.float32)
	figwig.write_bigwig(tmp_path / 'a.bw', {'chr1': 20}, 'chr1', [0], values[None],
		missing=missing)

	expected = numpy.full(20, missing, dtype=numpy.float32)
	expected[:9] = numpy.where(numpy.isnan(values), missing, values)
	y, reference = read_back(tmp_path / 'a.bw', 'chr1', 20, missing=missing)
	assert_bits(y, expected)
	assert_bits(reference, expected)

	written = sum(struct.unpack_from('<H', raw, 22)[0] for _, raw in parse(
		tmp_path / 'a.bw')['blocks'])
	assert written == int((~numpy.isnan(values) & (values != numpy.float32(
		missing))).sum())


def test_dense_negative_zero(tmp_path):
	"""-0.0 equals 0.0, so with missing=0.0 it is left out and reads back as
	0.0, without its sign bit."""

	figwig.write_bigwig(tmp_path / 'a.bw', {'chr1': 4}, 'chr1', [0], [[-0.0, 1.0,
		-0.0, 2.0]])
	y, _ = read_back(tmp_path / 'a.bw', 'chr1', 4)
	assert_bits(y, numpy.array([0.0, 1.0, 0.0, 2.0], dtype=numpy.float32))


## Writing windows and intervals


def tiled_windows(rng, chrom_sizes, width, gap):
	"""Windows of one width tiling each chromosome with gaps, the last few
	running past its end or starting after it, in a random order."""

	chroms, starts = [], []
	for name, length in chrom_sizes.items():
		first = numpy.arange(int(rng.integers(0, 50)), length + width // 2,
			width + gap)
		chroms += [name] * len(first)
		starts += first.tolist()

	order = rng.permutation(len(starts))
	return numpy.array(chroms)[order], numpy.array(starts)[order]


@pytest.mark.parametrize('missing', [0.0, numpy.nan])
def test_write_reads_back_what_read_gives(tmp_path, dense_bw, sparse_bw,
	writer_batching, missing):
	"""Windows read from a file, in any order, on several chromosomes, and
	running past the ends of their chromosomes, are written in one call and
	read back bit for bit."""

	for path, chrom_sizes in [dense_bw, sparse_bw]:
		rng = numpy.random.default_rng(0)
		chroms, starts = tiled_windows(rng, chrom_sizes, 700, 400)
		y = figwig.read_bigwig(path, chroms, starts, 700, missing=missing)
		with figwig.BigWigWriter(tmp_path / 'a.bw', chrom_sizes, n_jobs=3) as \
				writer:
			writer.write(chroms, starts, y, missing=missing)

		assert_bits(figwig.read_bigwig(tmp_path / 'a.bw', chroms, starts, 700,
			missing=missing), y)


def test_write_sorts_items(tmp_path):
	"""Windows given in any order write the file that the same windows
	sorted by chromosome, in the order of chrom_sizes, and start write."""

	rng = numpy.random.default_rng(1)
	chroms, starts = tiled_windows(rng, {'chr2': 30_000, 'chr1': 60_000}, 300,
		50)
	y = rng.normal(0, 1, (len(starts), 300)).astype(numpy.float32)
	y[rng.random(y.shape) < 0.2] = 0
	lengths = numpy.where(chroms == 'chr1', 60_000, 30_000)
	y[numpy.arange(300) >= (lengths - starts)[:, None]] = numpy.nan

	order = numpy.lexsort((starts, chroms == 'chr1'))
	for name, index in [('a', numpy.arange(len(starts))), ('b', order)]:
		with figwig.BigWigWriter(tmp_path / (name + '.bw'), {'chr2': 30_000,
				'chr1': 60_000}) as writer:
			writer.write(chroms[index], starts[index], y[index])

	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


def test_write_intervals(tmp_path, writer_batching):
	"""Intervals in any order, on several chromosomes, read back as their
	values, and those whose value is `missing` or NaN are not written."""

	rng = numpy.random.default_rng(9)
	s1, e1, v1 = random_intervals(rng, 3_000_000, 4000)
	s2, e2, v2 = random_intervals(rng, 1_000_000, 3000)
	v1[::7] = 0
	v2[::11] = numpy.nan

	chroms = numpy.array(['chr1'] * 4000 + ['chr2'] * 3000)
	starts, ends = numpy.concatenate([s1, s2]), numpy.concatenate([e1, e2])
	values = numpy.concatenate([v1, v2])
	order = rng.permutation(len(starts))
	with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS, n_jobs=2) as writer:
		writer.write(chroms[order], starts[order], values[order],
			ends=ends[order])

	for chrom, (s, e, v) in [('chr1', (s1, e1, v1)), ('chr2', (s2, e2, v2))]:
		expected = numpy.zeros(CHROMS[chrom], dtype=numpy.float32)
		for a, b, value in zip(s, e, numpy.nan_to_num(v).astype(numpy.float32)):
			expected[a:b] = value

		y, reference = read_back(tmp_path / 'a.bw', chrom, CHROMS[chrom])
		assert_bits(y, expected)
		assert_bits(reference, expected)

	written = sum(struct.unpack_from('<H', raw, 22)[0] for _, raw in parse(
		tmp_path / 'a.bw')['blocks'])
	assert written == int(((values != 0) & ~numpy.isnan(values)).sum())


def test_wide_windows_in_pieces(tmp_path, monkeypatch, writer_batching):
	"""A window wider than _SEGMENT_BASES is converted and written a piece of
	whole segments at a time, and the file does not depend on the batch
	size."""

	monkeypatch.setattr(figwig.writer, '_SEGMENT_BASES', 1000)
	rng = numpy.random.default_rng(10)
	y = rng.normal(0, 1, (2, 23_456))
	y[rng.random(y.shape) < 0.1] = 0

	def write(path):
		with figwig.BigWigWriter(path, CHROMS, n_jobs=2) as writer:
			writer.write(['chr2', 'chr1'], [5, 100_000], y)

	write(tmp_path / 'a.bw')
	monkeypatch.setattr(figwig.writer, '_BATCH_ITEMS', 1 << 20)
	write(tmp_path / 'b.bw')

	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()
	assert_bits(figwig.read_bigwig(tmp_path / 'a.bw', ['chr2', 'chr1'], [5,
		100_000], 23_456), y.astype(numpy.float32))


def section_types(path):
	"""The section type of each data block, in file order."""

	return [raw[20] for _, raw in parse(path)['blocks']]


def test_write_chooses_the_smallest_section_type(tmp_path, monkeypatch):
	"""Each window, or segment of a wider one, is written as the section type
	that takes it in the fewest bytes: fixedStep for a run of distinct
	values, varStep for scattered bases, and bedGraph for stretches of one
	value. The values read back whichever is chosen."""

	monkeypatch.setattr(figwig.writer, '_SEGMENT_BASES', 1000)
	rng = numpy.random.default_rng(11)
	runs = rng.normal(0, 1, 1000)
	scattered = numpy.zeros(1000)
	scattered[rng.choice(1000, 40, replace=False)] = rng.normal(0, 1, 40)
	stretches = numpy.repeat(rng.normal(0, 1, 10), 100)
	y = numpy.concatenate([runs, scattered, stretches])[None]

	with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS) as writer:
		writer.write('chr1', [0], y[:, :1000])
		writer.write('chr1', [1000], y[:, 1000:2000])
		writer.write('chr1', [2000], y[:, 2000:])
		writer.write('chr2', [0], y)

	assert section_types(tmp_path / 'a.bw') == [3, 2, 1, 3, 2, 1]
	y = y.astype(numpy.float32)
	assert_bits(figwig.read_bigwig(tmp_path / 'a.bw', ['chr1', 'chr2'], [0, 0],
		3000), numpy.concatenate([y, y]))


@pytest.mark.parametrize('call, error, match', [
	(lambda w: w.write(1, [0], [[1]]), ValueError, 'one name per start'),
	(lambda w: w.write(['chr1', 'chr1'], [0], [[1]]), ValueError,
		'one name per start'),
	(lambda w: w.write('chr3', [0], [[1]]), ValueError, 'not one of the'),
	(lambda w: w.write('chr1', [0], [['a']]), TypeError, 'must be numbers'),
	(lambda w: w.write('chr1', [0.0], [[1]]), TypeError, 'must be integers'),
	(lambda w: w.write('chr1', [[0]], [[1]]), ValueError, 'one-dimensional'),
	(lambda w: w.write('chr1', [0, 5], [1, 2]), ValueError,
		'pass ends to write intervals'),
	(lambda w: w.write('chr1', [0, 5], [[1, 2]]), ValueError,
		'one row per start'),
	(lambda w: w.write('chr1', [0], numpy.zeros((1, 0))), ValueError,
		'at least one column'),
	(lambda w: w.write('chr1', [0], [[1]], ends=[5]), ValueError,
		'one value per interval'),
	(lambda w: w.write('chr1', [0, 9], [1, 2], ends=[5]), ValueError,
		'same length'),
	(lambda w: w.write('chr1', [-1], [[1]]), ValueError, 'before 0'),
	(lambda w: w.write('chr1', [2 ** 32 - 3], [[numpy.nan] * 5]), ValueError,
		r'past 2\*\*32 - 1'),
	(lambda w: w.write('chr1', [5], [1], ends=[5]), ValueError,
		'at or before its start'),
	(lambda w: w.write('chr2', [10], [1], ends=[1_000_001]), ValueError,
		'past its length'),
	(lambda w: w.write('chrM', [16_000], [numpy.ones(570)]), ValueError,
		'where its values must be NaN'),
	(lambda w: w.write('chr2', [999_999, 1_000_000], [[1], [2]]), ValueError,
		'window 1 on .chr2. runs past the end'),
	(lambda w: w.write('chr1', [0], [[1, numpy.inf]]), ValueError,
		'finite or NaN'),
	(lambda w: w.write('chr1', [0], [numpy.inf], ends=[2]), ValueError,
		'finite or NaN'),
	(lambda w: w.write('chr1', [0], [[1e39]]), ValueError,
		"outside float32's range"),
	(lambda w: w.write('chr1', [0, 4], [[1] * 5, [2] * 5]), ValueError,
		'window 1 on .chr1., at 4, starts before window 0 ends'),
	(lambda w: w.write('chr1', [4, 0], [2, 1], ends=[9, 5]), ValueError,
		'interval 0 on .chr1., at 4, starts before interval 1 ends'),
	(lambda w: w.write('chr1', [5, 5], [[1], [2]]), ValueError,
		'must not overlap'),
	(lambda w: w.write('chr1', [0], [[1]], missing='0'), TypeError,
		'missing must be a number'),
], ids=['chrom_type', 'n_chroms', 'chrom_unknown', 'values_strings',
	'float_starts', 'starts_2d', 'windows_1d', 'n_rows', 'no_columns',
	'intervals_2d', 'n_ends', 'negative', 'past_2_32', 'empty_interval',
	'interval_past_end', 'window_past_end', 'base_past_end', 'window_inf',
	'interval_inf',
	'float32_range', 'overlapping_windows', 'overlapping_intervals', 'repeat',
	'missing_type'])
def test_bad_write(tmp_path, call, error, match):
	with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS) as writer:
		with pytest.raises(error, match=match):
			call(writer)


@pytest.mark.parametrize('first, second', [
	(('chr1', [100], [1], [200]), ('chr1', [150], [2], [300])),
	(('chr1', [100], [[1, 2]], None), ('chr1', [101], [[3]], None)),
	(('chr1', [100], [[0, 0]], None), ('chr1', [101], [[3]], None)),
	(('chr2', [5], [[1]], None), ('chr1', [5], [[1]], None)),
], ids=['intervals', 'windows', 'missing_window', 'chromosome_order'])
def test_write_out_of_order(tmp_path, first, second):
	"""Each call's items start at or after the end of the last call's last
	item on their chromosome, a window whose values are all `missing`
	included, and chromosomes come in the order of `chrom_sizes`."""

	with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS) as writer:
		writer.write(*first[:3], ends=first[3])
		with pytest.raises(ValueError, match="before"):
			writer.write(*second[:3], ends=second[3])


def test_write_that_raises_writes_nothing(tmp_path, monkeypatch):
	"""A call that raises, here on an infinite value in a late piece of its
	windows, writes none of them, and does not count as reaching their
	chromosome."""

	monkeypatch.setattr(figwig.writer, '_BATCH_ITEMS', 10)
	y = numpy.ones((50, 10))
	y[40, 3] = numpy.inf

	with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS) as writer:
		writer.write('chr1', [0], [[5.0]])
		with pytest.raises(ValueError, match="window 40 on 'chr2'"):
			writer.write('chr2', numpy.arange(50) * 10, y)
		writer.write('chr2', [5], [[2.0]])

	with figwig.BigWigWriter(tmp_path / 'b.bw', CHROMS) as writer:
		writer.write('chr1', [0], [[5.0]])
		writer.write('chr2', [5], [[2.0]])

	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


## Engines, levels, threads and batches


def test_engine_auto(tmp_path, monkeypatch):
	"""'auto' is libdeflate when libdeflate's functions load from the deflate
	package, which they do not on Windows, and zlib otherwise. Asking for
	libdeflate where they do not load says whether the package is missing
	or only its functions."""

	with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS) as writer:
		assert writer.engine == ('zlib' if sys.platform == 'win32' else
			'libdeflate')

	monkeypatch.setattr(figwig.writer, '_libdeflate', lambda: None)
	with figwig.BigWigWriter(tmp_path / 'b.bw', CHROMS) as writer:
		assert writer.engine == 'zlib'
	with pytest.raises(ValueError, match="could not be loaded from the deflate"):
		figwig.BigWigWriter(tmp_path / 'c.bw', CHROMS, engine='libdeflate')

	monkeypatch.setitem(sys.modules, 'deflate', None)
	with pytest.raises(ValueError, match="needs the deflate package"):
		figwig.BigWigWriter(tmp_path / 'd.bw', CHROMS, engine='libdeflate')


def test_engine_isal_needs_isal(tmp_path, monkeypatch):
	monkeypatch.setitem(sys.modules, 'isal.isal_zlib', None)
	with pytest.raises(ValueError, match="needs the isal package"):
		figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS, engine='isal')


@pytest.mark.parametrize('engine, level', [('zlib', 6), ('zlib', 1),
	pytest.param('libdeflate', 6, marks=needs_c_libraries),
	pytest.param('libdeflate', 12, marks=needs_c_libraries)])
def test_same_bytes_whatever_threads_and_batches(tmp_path, monkeypatch, engine,
	level):
	"""zlib and libdeflate files are the same byte for byte whatever the
	number of threads or the batch size, and from one run to the next."""

	dense = numpy.random.default_rng(4).normal(0, 1, 16_000).astype(
		numpy.float32)
	calls = calls_of('bases', random_state=2) + [('dense', 'chrM', 5, dense)]
	outputs = set()
	for n_jobs, batch in [(1, 1 << 20), (4, 1 << 20), (3, 777), (1, 777)]:
		monkeypatch.setattr(figwig.writer, '_BATCH_ITEMS', batch)
		path = tmp_path / '{}-{}.bw'.format(n_jobs, batch)
		write_figwig(path, CHROMS, calls, engine=engine, level=level,
			n_jobs=n_jobs)
		outputs.add(path.read_bytes())

	assert len(outputs) == 1


@pytest.mark.parametrize('n_jobs', [1, 4])
def test_same_bytes_without_pread(tmp_path, monkeypatch, writer_batching,
	n_jobs):
	"""Where there is neither preadv nor pread, as on Windows, blocks are read
	back for the zoom levels by seeking, so every zoom level is held to the
	end rather than the finest written as it is compressed. The file is the
	same."""

	calls = calls_of('intervals', random_state=3)
	write_figwig(tmp_path / 'a.bw', CHROMS, calls, n_jobs=n_jobs)
	monkeypatch.delattr(os, 'preadv', raising=False)
	monkeypatch.delattr(os, 'pread', raising=False)
	write_figwig(tmp_path / 'b.bw', CHROMS, calls, n_jobs=n_jobs)

	assert len(parse(tmp_path / 'a.bw')['levels']) > 1
	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


@pytest.mark.skipif(not (hasattr(os, 'preadv') or hasattr(os, 'pread')),
	reason='without pread every zoom level is held to the end')
def test_finest_zoom_level_written_during_pass(tmp_path, monkeypatch):
	"""The finest zoom level's blocks are written as they are compressed,
	before the levels are finished; on one thread, each is compressed as
	soon as its records are, so all are written by then."""

	monkeypatch.setattr(figwig.writer, '_ZOOM_CHUNK_BLOCKS', 1)
	monkeypatch.setattr(figwig.writer, '_ZOOM_COMPRESS_BLOCKS', 1)
	seen = []
	finish = figwig.writer._ZoomLevel.finish

	def recorded(self):
		seen.append((self.size, self.n_written, len(self.blocks)))
		return finish(self)

	monkeypatch.setattr(figwig.writer._ZoomLevel, 'finish', recorded)
	write_figwig(tmp_path / 'a.bw', CHROMS, calls_of('bases'), n_jobs=1)

	size, n_written, n_blocks = seen[0]
	assert size == min(size for size, _, _ in seen)
	assert n_written == n_blocks > 1
	assert all(n_written == 0 for _, n_written, _ in seen[1:])


def test_zoom_read_ahead_is_bounded(tmp_path, monkeypatch):
	"""However many threads there are, blocks are read back for the zoom
	levels at most _ZOOM_READ_AHEAD chunks past the one the slowest level is
	on. With summaries slowed down, the reads reach that bound."""

	monkeypatch.setattr(figwig.writer, '_ZOOM_CHUNK_BLOCKS', 1)
	monkeypatch.setattr(figwig.writer, '_ZOOM_READ_AHEAD', 2)
	lock = threading.Lock()
	n_reads, summarized, most = [0], {}, [0]
	init = figwig.writer._ZoomLevel.__init__
	add = figwig.writer._ZoomLevel.add
	read_items = figwig.writer._read_items

	def registered(self, size):
		init(self, size)
		summarized[size] = 0

	def slow(self, *args):
		time.sleep(0.002)
		records = add(self, *args)
		with lock:
			summarized[self.size] += 1
		return records

	def counted(*args):
		with lock:
			n_reads[0] += 1
			most[0] = max(most[0], n_reads[0] - min(summarized.values()))
		return read_items(*args)

	monkeypatch.setattr(figwig.writer._ZoomLevel, '__init__', registered)
	monkeypatch.setattr(figwig.writer._ZoomLevel, 'add', slow)
	monkeypatch.setattr(figwig.writer, '_read_items', counted)
	positions = numpy.arange(0, 2_000_000, 20)
	values = numpy.random.default_rng(0).random(len(positions)).astype(
		numpy.float32)
	write_figwig(tmp_path / 'a.bw', CHROMS, [('bases', 'chr1', positions,
		values)], n_jobs=8)

	assert n_reads[0] > 10
	assert most[0] == 3


@needs_c_libraries
def test_levels_change_bytes_not_values(tmp_path):
	calls = calls_of('intervals')
	for name, engine, level in [('a', 'zlib', 1), ('b', 'zlib', 9), ('c',
			'libdeflate', 0), ('d', 'isal', 3)]:
		write_figwig(tmp_path / (name + '.bw'), CHROMS, calls, engine=engine,
			level=level)

	files = [parse(tmp_path / (name + '.bw')) for name in 'abcd']
	assert len({(tmp_path / (name + '.bw')).read_bytes() for name in 'abcd'}) == 4
	for f in files[1:]:
		assert [raw for _, raw in f['blocks']] == [raw for _, raw in files[0][
			'blocks']]
		assert f['summary'] == files[0]['summary']
		assert [level['records'][0].tobytes() for level in f['levels']] == [
			level['records'][0].tobytes() for level in files[0]['levels']]


def test_zlib_fallbacks(tmp_path, monkeypatch):
	"""Where zlib's library cannot be loaded, blocks are compressed with
	zlib.compress and read back for the zoom levels with zlib.decompress, and
	the file is the same."""

	calls = calls_of('bases')
	write_figwig(tmp_path / 'a.bw', CHROMS, calls, engine='zlib', n_jobs=2)

	monkeypatch.setattr(figwig.writer, '_zlib_compress', lambda: None)
	monkeypatch.setattr(figwig.writer, '_zlib_uncompress', lambda: None)
	write_figwig(tmp_path / 'b.bw', CHROMS, calls, engine='zlib', n_jobs=2)
	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


def test_n_jobs_all_cpus(tmp_path):
	writer = figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS, n_jobs=-1)
	assert writer.n_jobs == figwig.bigwig._cpu_count()
	writer.close()


## The kernels, on inputs small enough to check by hand


def test_zoom_records_by_hand():
	"""With records of 10 bases: [0, 5) at 1 starts a record, which [7, 12)
	at 2 extends to 10, covering 8 bases with sum 5 * 1 + 3 * 2 = 11 and sum
	of squares 5 + 12 = 17. The rest of that item, [10, 12), starts the next
	record. [30, 31) at 3 starts a third, still open at the end."""

	tids = numpy.zeros(3, dtype=numpy.int64)
	starts = numpy.array([0, 7, 30], dtype=numpy.int64)
	ends = numpy.array([5, 12, 31], dtype=numpy.int64)
	values = numpy.array([1, 2, 3], dtype=numpy.float32)
	open_ints = numpy.zeros(6, dtype=numpy.int64)
	open_floats = numpy.zeros(4)
	ints = numpy.zeros((10, 4), dtype=numpy.uint32)
	floats = numpy.zeros((10, 4), dtype=numpy.float32)
	progress = numpy.array([0, -1], dtype=numpy.int64)

	n = figwig._kernels._zoom_records(10, tids, starts, ends, values, progress,
		open_ints, open_floats, ints, floats)

	assert n == 2
	assert ints[:2].tolist() == [[0, 0, 10, 8], [0, 10, 12, 2]]
	assert floats[:2].tolist() == [[1, 2, 11, 17], [2, 2, 4, 8]]
	assert open_ints.tolist() == [1, 3, 0, 30, 31, 1]
	assert open_floats.tolist() == [3, 3, 3, 9]
	assert progress.tolist() == [3, -1]


def test_zoom_records_stops_when_full():
	"""With room for one record and records of 10 bases, [0, 35) at 2 fills
	the buffer with [0, 10) and stops at base 20, where [10, 20) would close,
	then at base 30. The third call finishes the item and stops at the start
	of [40, 41), where [30, 35) would close, and the fourth finishes. No call
	changes the open record after its last write."""

	tids = numpy.zeros(2, dtype=numpy.int64)
	starts = numpy.array([0, 40], dtype=numpy.int64)
	ends = numpy.array([35, 41], dtype=numpy.int64)
	values = numpy.array([2, 3], dtype=numpy.float32)
	open_ints = numpy.zeros(6, dtype=numpy.int64)
	open_floats = numpy.zeros(4)
	ints = numpy.zeros((1, 4), dtype=numpy.uint32)
	floats = numpy.zeros((1, 4), dtype=numpy.float32)
	progress = numpy.array([0, -1], dtype=numpy.int64)

	calls = [
		([[0, 0, 10, 10]], [[2, 2, 20, 40]], [0, 20], [1, 2, 0, 10, 20, 10]),
		([[0, 10, 20, 10]], [[2, 2, 20, 40]], [0, 30], [1, 3, 0, 20, 30, 10]),
		([[0, 20, 30, 10]], [[2, 2, 20, 40]], [1, 40], [1, 4, 0, 30, 35, 5]),
		([[0, 30, 35, 5]], [[2, 2, 10, 20]], [2, -1], [1, 5, 0, 40, 41, 1]),
	]

	for record_ints, record_floats, where, opened in calls:
		n = figwig._kernels._zoom_records(10, tids, starts, ends, values,
			progress, open_ints, open_floats, ints, floats)

		assert n == 1
		assert ints.tolist() == record_ints
		assert floats.tolist() == record_floats
		assert progress.tolist() == where
		assert open_ints.tolist() == opened

	assert open_floats.tolist() == [3, 3, 3, 9]


@pytest.mark.parametrize('room', [1, 2, 7, 1_000_000])
def test_zoom_records_any_buffer_size(room):
	"""Calling the kernel until every item is summarized gives the same
	records, and the same open record, whatever its output buffer holds:
	items of several chromosomes, spanning many records and few, with more
	than 1023 records to a block."""

	rng = numpy.random.default_rng(0)
	widths = rng.integers(1, 60, 5000)
	gaps = rng.integers(0, 30, 5000)
	starts = numpy.cumsum(gaps + widths) - widths
	ends = starts + widths
	tids = numpy.repeat(numpy.arange(4), [1000, 2500, 1, 1499])
	values = rng.random(5000).astype(numpy.float32)

	def records(room):
		open_ints = numpy.zeros(6, dtype=numpy.int64)
		open_floats = numpy.zeros(4)
		ints = numpy.zeros((room, 4), dtype=numpy.uint32)
		floats = numpy.zeros((room, 4), dtype=numpy.float32)
		progress = numpy.array([0, -1], dtype=numpy.int64)

		out_ints, out_floats = [], []
		while progress[0] < len(values):
			n = figwig._kernels._zoom_records(10, tids, starts, ends, values,
				progress, open_ints, open_floats, ints, floats)
			out_ints.append(ints[:n].copy())
			out_floats.append(floats[:n].copy())

		return (numpy.concatenate(out_ints), numpy.concatenate(out_floats),
			open_ints, open_floats)

	expected = records(1_000_000)
	assert len(expected[0]) > 2 * 1023

	for x, y in zip(records(room), expected):
		numpy.testing.assert_array_equal(x, y)


def test_zoom_records_full_block():
	"""Once the open record is the 1023rd of its block, the next item starts a
	new record in a new block, even though it falls within the open record's
	10 bases."""

	starts = numpy.append(numpy.arange(1023, dtype=numpy.int64) * 100, 102_205)
	tids = numpy.zeros(1024, dtype=numpy.int64)
	ends = starts + 1
	values = numpy.ones(1024, dtype=numpy.float32)
	open_ints = numpy.zeros(6, dtype=numpy.int64)
	open_floats = numpy.zeros(4)
	ints = numpy.zeros((1100, 4), dtype=numpy.uint32)
	floats = numpy.zeros((1100, 4), dtype=numpy.float32)
	progress = numpy.array([0, -1], dtype=numpy.int64)

	n = figwig._kernels._zoom_records(10, tids, starts, ends, values, progress,
		open_ints, open_floats, ints, floats)

	assert n == 1023
	assert ints[1022].tolist() == [0, 102_200, 102_201, 1]
	assert open_ints.tolist() == [1, 1, 0, 102_205, 102_206, 1]


def test_update_summary_by_hand():
	"""The minimum and maximum are those of the values, the first value
	included. The sum adds each span times value as float32, so 3 * 0.1
	rounds to float32 before it is added, as in libBigWig."""

	counts = numpy.zeros(2, dtype=numpy.uint64)
	stats = numpy.array([numpy.inf, -numpy.inf, 0.0, 0.0])
	figwig._kernels._update_summary(numpy.array([3, 1], dtype=numpy.uint32),
		numpy.array([0.1, 0.5], dtype=numpy.float32), counts, stats)
	assert stats[1] == 0.5

	counts = numpy.zeros(2, dtype=numpy.uint64)
	stats = numpy.array([numpy.inf, -numpy.inf, 0.0, 0.0])
	figwig._kernels._update_summary(numpy.array([3, 1], dtype=numpy.uint32),
		numpy.array([0.5, 0.1], dtype=numpy.float32)[::-1].copy(), counts, stats)

	assert counts.tolist() == [4, 2]
	assert stats[0] == float(numpy.float32(0.1))
	assert stats[1] == 0.5
	assert stats[2] == float(numpy.float32(numpy.float32(3) * numpy.float32(
		0.1))) + 0.5
	assert stats[3] == 3 * float(numpy.float32(0.1)) ** 2 + 0.25


def test_zoom_reductions():
	"""16 times the mean item width, or 10 if that is less, capped at the
	longest chromosome, then 4 times larger while it fits."""

	assert figwig.writer._zoom_reductions(100, 100, 10_000, 10) == [16, 64, 256,
		1024, 4096]
	assert figwig.writer._zoom_reductions(1000, 100, 10_000_000, 3) == [160, 640,
		2560]
	assert figwig.writer._zoom_reductions(100, 400, 1000, 10) == [10, 40, 160,
		640]
	assert figwig.writer._zoom_reductions(100, 1, 50, 10) == [50]
	assert figwig.writer._zoom_reductions(2 ** 28 - 1, 1, 2 ** 32 - 1, 10) == [
		2 ** 32 - 16]
	assert figwig.writer._zoom_reductions(2 ** 28, 1, 2 ** 32 - 1, 10) == []
	assert figwig.writer._zoom_reductions(2 ** 31, 1, 2 ** 32 - 1, 10) == []


## Arguments


@pytest.mark.parametrize('chrom_sizes, error, match', [
	({}, ValueError, 'at least one chromosome'),
	(5, TypeError, 'chrom_sizes must be a dict'),
	([('chr1', 10, 3)], ValueError, r'\(name, length\) pair'),
	([(1, 10)], TypeError, 'names must be strings'),
	([('', 10)], ValueError, 'non-empty'),
	([('chr\x001', 10)], ValueError, 'null characters'),
	([('chr1', 10), ('chr1', 20)], ValueError, 'named twice'),
	([('chr1', 10.0)], TypeError, 'must be an integer'),
	([('chr1', True)], TypeError, 'must be an integer'),
	([('chr1', 0)], ValueError, 'from 1 to 2\\*\\*32 - 1'),
	([('chr1', 2 ** 32)], ValueError, 'from 1 to 2\\*\\*32 - 1'),
], ids=['empty', 'not_iterable', 'triple', 'name_type', 'empty_name', 'null',
	'duplicate', 'float_length', 'bool_length', 'zero_length', 'long'])
def test_bad_chrom_sizes(tmp_path, chrom_sizes, error, match):
	with pytest.raises(error, match=match):
		figwig.BigWigWriter(tmp_path / 'a.bw', chrom_sizes)
	assert not (tmp_path / 'a.bw').exists()


@pytest.mark.parametrize('kwargs, error, match', [
	({'zooms': 1.0}, TypeError, 'zooms must be an integer'),
	({'zooms': -1}, ValueError, 'zooms must be from 0 to 65535'),
	({'zooms': 65536}, ValueError, 'zooms must be from 0 to 65535'),
	({'engine': 'gzip'}, ValueError, 'engine must be one of'),
	({'level': 6.0}, TypeError, 'level must be an integer'),
	({'engine': 'zlib', 'level': 10}, ValueError, 'from 0 to 9'),
	({'engine': 'zlib', 'level': -1}, ValueError, 'from 0 to 9'),
	pytest.param({'engine': 'libdeflate', 'level': 13}, ValueError,
		'from 0 to 12', marks=needs_c_libraries),
	({'engine': 'isal', 'level': 4}, ValueError, 'from 0 to 3'),
	({'n_jobs': 0}, ValueError, 'n_jobs must be at least 1'),
	({'n_jobs': 2.0}, TypeError, 'n_jobs must be an integer'),
], ids=['zooms_type', 'zooms_low', 'zooms_high', 'engine', 'level_type',
	'zlib_high', 'zlib_low', 'libdeflate_high', 'isal_high', 'n_jobs',
	'n_jobs_type'])
def test_bad_arguments(tmp_path, kwargs, error, match):
	with pytest.raises(error, match=match):
		figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS, **kwargs)
	assert not (tmp_path / 'a.bw').exists()


## Behaviour


def test_empty_writes_change_nothing(tmp_path):
	"""Writing no windows or intervals, or windows whose values are all
	`missing`, writes no data, and nothing about the file changes."""

	with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS, zooms=0) as writer:
		writer.write('chr1', [], numpy.zeros((0, 5)))
		writer.write('chr1', [], [], ends=[])
		writer.write('chr1', [0, 10], numpy.zeros((2, 5)))
		writer.write('chr1', [20], [[1.0]])

	write_figwig(tmp_path / 'b.bw', CHROMS, [('bases', 'chr1', [20], [1.0])],
		zooms=0)
	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


@pytest.mark.parametrize('dtype', [numpy.int32, numpy.uint32, numpy.int64,
	numpy.uint64, 'list'])
@pytest.mark.parametrize('value_dtype', [numpy.float32, numpy.float64,
	numpy.int16, numpy.uint8, bool, 'list'])
def test_input_types(tmp_path, dtype, value_dtype):
	"""Starts and ends of any integer type and values of any real type, or
	lists of them, write the same file as int64 positions and float64
	values."""

	starts = numpy.array([3, 8, 9, 100])
	values = numpy.array([1, 0, 1, 1])

	def cast(x, t):
		return x.tolist() if t == 'list' else x.astype(t)

	for name, x, ends in [('windows', values[:, None], None), ('intervals',
			values, starts + 1)]:
		figwig.write_bigwig(tmp_path / 'a.bw', CHROMS, 'chr1', cast(starts,
			dtype), cast(x, value_dtype), ends=None if ends is None else cast(
			ends, dtype))
		figwig.write_bigwig(tmp_path / 'b.bw', CHROMS, 'chr1', starts,
			x.astype(numpy.float64), ends=ends)
		assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


def test_path_types(tmp_path):
	figwig.write_bigwig(str(tmp_path / 'a.bw'), CHROMS, 'chr1', [0], [[1.0]])
	figwig.write_bigwig(tmp_path / 'b.bw', list(CHROMS.items()), 'chr1', [0],
		[[1.0]])
	assert (tmp_path / 'a.bw').read_bytes() == (tmp_path / 'b.bw').read_bytes()


@pytest.mark.parametrize('container', [list, tuple])
def test_write_bigwig_several_files(tmp_path, container):
	"""Given a list of paths, channel i of the values is written to the i-th
	path, as BigWigWriter.write writes it, and read_bigwig of the same paths
	reads the values back."""

	rng = numpy.random.default_rng(12)
	chrom_sizes = {'chr1': 60_000, 'chr2': 20_000}
	chroms, starts = tiled_windows(rng, chrom_sizes, 500, 100)
	lengths = numpy.where(chroms == 'chr1', 60_000, 20_000)
	y = rng.normal(0, 1, (len(starts), 3, 500)).astype(numpy.float32)
	y[rng.random(y.shape) < 0.5] = 0
	y[numpy.broadcast_to(numpy.arange(500) >= (lengths - starts)[:, None, None],
		y.shape)] = numpy.nan

	paths = [tmp_path / '{}.bw'.format(i) for i in range(3)]
	figwig.write_bigwig(container(paths), chrom_sizes, chroms, starts, y,
		n_jobs=2)
	for i in range(3):
		figwig.write_bigwig(tmp_path / 'one.bw', chrom_sizes, chroms, starts,
			y[:, i], n_jobs=2)
		assert paths[i].read_bytes() == (tmp_path / 'one.bw').read_bytes()

	assert_bits(figwig.read_bigwig(paths, chroms, starts, 500), y)

	inside = starts + 50 <= lengths
	chroms, starts, v = chroms[inside], starts[inside], y[inside, :, 0]
	figwig.write_bigwig(paths[:2], chrom_sizes, chroms, starts, v[:, :2],
		ends=starts + 50)
	for i in range(2):
		figwig.write_bigwig(tmp_path / 'one.bw', chrom_sizes, chroms, starts,
			v[:, i], ends=starts + 50)
		assert paths[i].read_bytes() == (tmp_path / 'one.bw').read_bytes()


@pytest.mark.parametrize('kwargs, error, match', [
	({'paths': []}, ValueError, 'at least one path'),
	({'paths': [1]}, TypeError, 'paths must be a path'),
	({'values': numpy.zeros((2, 3, 5))}, ValueError,
		r'one channel per path, of shape \(n, 2, width\)'),
	({'values': numpy.zeros((2, 5))}, ValueError, 'one channel per path'),
	({'values': numpy.zeros((2, 2, 5)), 'ends': [5, 15]}, ValueError,
		r'one channel per path, of shape \(n, 2\)'),
], ids=['no_paths', 'path_type', 'n_channels', 'no_channels', 'intervals'])
def test_write_bigwig_bad_arguments(tmp_path, kwargs, error, match):
	arguments = {'paths': [tmp_path / 'a.bw', tmp_path / 'b.bw'], 'chrom_sizes':
		CHROMS, 'chroms': 'chr1', 'starts': [0, 10], 'values': numpy.zeros((2, 2,
		5))}
	arguments.update(kwargs)
	with pytest.raises(error, match=match):
		figwig.write_bigwig(**arguments)


def test_failure_leaves_no_header(tmp_path):
	"""When writing fails inside the with block, the file is closed without a
	header, so that no reader takes it for a bigWig, and the threads stop."""

	with pytest.raises(ValueError):
		with figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS, n_jobs=4) as writer:
			writer.write('chr1', [5], [[1.0]])
			writer.write('chr1', [4], [[1.0]])

	assert writer._pool is None and writer._file is None
	with pytest.raises(ValueError, match="not a bigWig"):
		figwig.BigWigReader(str(tmp_path / 'a.bw'))


def test_close(tmp_path):
	"""close writes the file; closing again does nothing, and writing after
	it raises."""

	writer = figwig.BigWigWriter(tmp_path / 'a.bw', CHROMS)
	writer.write('chr1', [5], [[2.0]])
	writer.close()
	data = (tmp_path / 'a.bw').read_bytes()
	writer.close()
	assert (tmp_path / 'a.bw').read_bytes() == data

	with pytest.raises(ValueError, match="has been closed"):
		writer.write('chr1', [6], [[1.0]])

	assert repr(writer) == "BigWigWriter({!r})".format(str(tmp_path / 'a.bw'))
	assert writer.chrom_sizes == CHROMS
	assert figwig.BigWigReader(str(tmp_path / 'a.bw')).read('chr1', [5],
		2).tolist() == [[2.0, 0.0]]
