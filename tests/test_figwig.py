# test_figwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import sys
import pickle
import struct
import subprocess

import numpy
import pytest

import figwig
import figwig.figwig
import figwig._figwig_kernels

from figwig import FigWigReader
from figwig import FigWigWriter
from figwig import read_figwig
from figwig import write_figwig
from figwig import SparseEntries
from figwig import SparseWindows

from figwig._figwig_kernels import _check_chunks
from figwig._figwig_kernels import _encode_tiles
from figwig._figwig_kernels import _read_tiles


CHROM_SIZES = {'chr1': 5000, 'chr2': 3001, 'chrM': 70}
TILE = 64
TRACKS = [
	{'name': 'atac_plus', 'assay': 'ATAC-seq', 'strand': '+'},
	{'name': 'atac_minus', 'assay': 'ATAC-seq', 'strand': '-'},
	{'name': 'signal', 'dtype': 'float32', 'assay': 'DNase-seq'},
	{'name': 'negative', 'assay': 'test'},
	{'name': 'empty', 'assay': 'test'},
]
GROUPS = [['atac_plus', 'atac_minus'], ['signal'], ['negative', 'empty']]


def reference_tracks(random_state=0):
	"""Dense per-base values of five tracks on three chromosomes: scattered
	counts, a dense stretch, float32 signal with a NaN, negative integers,
	and an empty track."""

	rng = numpy.random.default_rng(random_state)
	ref = {}
	for chrom, length in CHROM_SIZES.items():
		values = numpy.zeros((5, length), dtype=numpy.float64)
		for t in range(4):
			idx = rng.choice(length, length // 10, replace=False)
			if t == 2:
				values[t, idx] = rng.normal(0, 1, len(idx)).astype(numpy.float32)
			else:
				values[t, idx] = rng.integers(1, 400, len(idx))
		if length > 900:
			values[1, 100:900] = rng.integers(0, 3, 800)
		values[3] = -values[3]
		if chrom == 'chr1':
			values[2, 777] = numpy.nan
		ref[chrom] = values
	return ref


def write_reference(path, ref, n_jobs=2, tile=TILE, level=6, groups=GROUPS,
	calls=2, width=100, order='positions'):
	"""Write `ref` as windows of `width`, each chromosome in `calls` calls."""

	with FigWigWriter(path, CHROM_SIZES, TRACKS, genome='hg38', attributes={
			'made_by': 'test_figwig'}, tile=tile, groups=groups, level=level,
			n_jobs=n_jobs, order=order) as writer:
		for chrom, length in CHROM_SIZES.items():
			starts = numpy.arange(0, length, width)
			values = numpy.full((len(starts), 5, width), numpy.nan)
			for j, start in enumerate(starts):
				piece = ref[chrom][:, start:start + width]
				values[j, :, :piece.shape[1]] = piece
			for part in numpy.array_split(numpy.arange(len(starts)), calls):
				if len(part) > 0:
					writer.write(chrom, starts[part], values[part])


def expected(ref, chroms, starts, width, tracks, dtype):
	"""The values `read` should give for these windows: NaN past the end
	for floats, 0 for integers."""

	fill = numpy.nan if numpy.dtype(dtype).kind == 'f' else 0
	out = numpy.full((len(starts), len(tracks), width), fill, dtype=dtype)
	for j, (chrom, start) in enumerate(zip(chroms, starts)):
		length = CHROM_SIZES[chrom]
		if start < length:
			piece = ref[chrom][tracks, start:min(length, start + width)]
			out[j, :, :piece.shape[1]] = piece
	return out


def assert_identical(a, b):
	"""The same NaN positions, and the same values everywhere else."""

	assert a.shape == b.shape and a.dtype == b.dtype
	if a.dtype.kind == 'f':
		nan_a, nan_b = numpy.isnan(a), numpy.isnan(b)
		numpy.testing.assert_array_equal(nan_a, nan_b)
		numpy.testing.assert_array_equal(a[~nan_a], b[~nan_b])
	else:
		numpy.testing.assert_array_equal(a, b)


def random_windows(rng, n, width):
	chroms = rng.choice(list(CHROM_SIZES), size=n)
	starts = rng.integers(0, max(CHROM_SIZES.values()), size=n)
	return chroms, starts.astype(numpy.int64)


@pytest.fixture(scope='session')
def ref():
	return reference_tracks()


@pytest.fixture(scope='session')
def fw(tmp_path_factory, ref):
	path = str(tmp_path_factory.mktemp('fw') / 'reference.fw')
	write_reference(path, ref)
	return path


@pytest.fixture(scope='session')
def track_fw(tmp_path_factory, ref):
	path = str(tmp_path_factory.mktemp('fw') / 'tracks.fw')
	write_reference(path, ref, order='tracks')
	return path


@pytest.fixture(params=[(256, 1 << 25), (1, 1 << 25), (3, 400)],
	ids=['default_batches', 'one_chunk_batches', 'small_byte_batches'])
def read_batching(request, monkeypatch):
	"""Read in batches of this many chunks and this many bytes."""

	chunks, bytes_ = request.param
	monkeypatch.setattr(figwig.figwig, '_BATCH_CHUNKS', chunks)
	monkeypatch.setattr(figwig.figwig, '_BATCH_BYTES', bytes_)
	return request.param


###
# Values


def test_read_whole_chromosomes(fw, ref):
	reader = FigWigReader(fw)
	for chrom, length in CHROM_SIZES.items():
		y = reader.read(chrom, [0], length, dtype='float64')
		exp = ref[chrom].copy()
		exp[2] = exp[2].astype(numpy.float32)
		assert_identical(y[0], exp)


def test_read_random_windows(fw, ref, read_batching):
	rng = numpy.random.default_rng(1)
	chroms, starts = random_windows(rng, 400, 150)
	reader = FigWigReader(fw)
	y = reader.read(chroms, starts, 150, dtype='float64', n_jobs=3)
	exp = expected(ref, chroms, starts, 150, [0, 1, 2, 3, 4], 'float64')
	exp[:, 2] = exp[:, 2].astype(numpy.float32)
	assert_identical(y, exp)


def test_read_track_subset_and_order(fw, ref):
	rng = numpy.random.default_rng(2)
	chroms, starts = random_windows(rng, 100, 37)
	reader = FigWigReader(fw)
	y = reader.read(chroms, starts, 37, tracks=['negative', 'atac_plus',
		'empty'], dtype='float32')
	assert_identical(y, expected(ref, chroms, starts, 37, [3, 0, 4], 'float32'))

	y = reader.read(chroms, starts, 37, tracks=[1, 1], dtype='float32')
	assert_identical(y, expected(ref, chroms, starts, 37, [1, 1], 'float32'))


def test_read_single_track_gives_two_dimensions(fw, ref):
	reader = FigWigReader(fw)
	y = reader.read('chr1', [0, 64, 4990], 100, tracks='atac_plus')
	assert y.shape == (3, 100) and y.dtype == numpy.uint16
	exp = expected(ref, ['chr1'] * 3, [0, 64, 4990], 100, [0], 'uint16')
	assert_identical(y, exp[:, 0])

	y = reader.read('chr1', [0], 100, tracks=2)
	assert y.shape == (1, 100) and y.dtype == numpy.float32


def test_read_row_order_follows_windows(fw):
	reader = FigWigReader(fw)
	chroms = ['chr2', 'chr1', 'chrM', 'chr1']
	starts = [2900, 10, 0, 0]
	y = reader.read(chroms, starts, 80)
	for j in range(4):
		one = reader.read(chroms[j], [starts[j]], 80)
		assert_identical(y[j:j + 1], one)


def test_read_past_the_end(fw, ref):
	reader = FigWigReader(fw)
	y = reader.read('chrM', [50, 70, 100], 40, dtype='float32')
	assert numpy.isnan(y[0, :, 20:]).all() and not numpy.isnan(y[0, :, :20]).any()
	assert numpy.isnan(y[1]).all() and numpy.isnan(y[2]).all()
	assert_identical(y[0, :, :20], ref['chrM'][:, 50:70].astype(numpy.float32))

	y = reader.read('chrM', [50, 70], 40, tracks=['atac_plus', 'negative'])
	assert y.dtype == numpy.int32
	assert (y[0, :, 20:] == 0).all() and (y[1] == 0).all()


def test_read_default_dtype(fw):
	reader = FigWigReader(fw)
	assert reader.read('chr1', [0], 10, tracks=['atac_plus']).dtype == numpy.uint16
	assert reader.read('chr1', [0], 10, tracks=['negative']).dtype == numpy.int16
	assert reader.read('chr1', [0], 10, tracks=['empty']).dtype == numpy.uint8
	assert reader.read('chr1', [0], 10, tracks=['atac_plus', 'negative']).dtype \
		== numpy.int32
	assert reader.read('chr1', [0], 10, tracks=['atac_plus', 'empty']).dtype == \
		numpy.uint16
	assert reader.read('chr1', [0], 10).dtype == numpy.float32


def test_read_requested_dtype(fw, ref):
	reader = FigWigReader(fw)
	for dtype in ('uint32', 'int64', 'float64', numpy.float32, 'int16'):
		y = reader.read('chr1', [0], 200, tracks=['atac_plus'], dtype=dtype)
		assert y.dtype == numpy.dtype(dtype)
		numpy.testing.assert_array_equal(y[0, 0], ref['chr1'][0, :200])


def test_read_dtype_that_cannot_hold_the_values_raises(fw):
	reader = FigWigReader(fw)
	with pytest.raises(ValueError, match="does not hold"):
		reader.read('chr1', [0], 10, tracks=['atac_plus'], dtype='uint8')
	with pytest.raises(ValueError, match="does not hold"):
		reader.read('chr1', [0], 10, tracks=['negative'], dtype='uint16')
	with pytest.raises(ValueError, match="float32 and cannot be read"):
		reader.read('chr1', [0], 10, tracks=['signal'], dtype='int32')
	with pytest.raises(ValueError, match="integer or float"):
		reader.read('chr1', [0], 10, dtype='bool')
	with pytest.raises(TypeError, match="not a dtype"):
		reader.read('chr1', [0], 10, dtype=object())


def test_read_out(fw, ref):
	reader = FigWigReader(fw)
	out = numpy.full((3, 2, 50), 7, dtype=numpy.float32)
	y = reader.read('chr1', [0, 100, 200], 50, tracks=[0, 2], out=out)
	assert y is out
	assert_identical(out, expected(ref, ['chr1'] * 3, [0, 100, 200], 50, [0, 2],
		'float32'))

	out = numpy.zeros((3, 50), dtype=numpy.uint16)
	y = reader.read('chr1', [0, 100, 200], 50, tracks='atac_plus', out=out)
	assert y is out

	with pytest.raises(ValueError, match="shape"):
		reader.read('chr1', [0], 50, tracks=[0, 2], out=numpy.zeros((1, 50),
			dtype=numpy.float32))
	with pytest.raises(ValueError, match="float32"):
		reader.read('chr1', [0], 50, tracks=[0, 2], out=numpy.zeros((1, 2, 50),
			dtype=numpy.float64))
	with pytest.raises(ValueError, match="C-contiguous"):
		reader.read('chr1', [0], 50, tracks=[0, 2], out=numpy.zeros((1, 2, 100),
			dtype=numpy.float32)[:, :, ::2])
	with pytest.raises(TypeError, match="numpy array"):
		reader.read('chr1', [0], 50, out=[0])


def test_read_threads_and_batches_do_not_change_values(fw, read_batching):
	rng = numpy.random.default_rng(3)
	chroms, starts = random_windows(rng, 300, 90)
	reader = FigWigReader(fw)
	one = reader.read(chroms, starts, 90, n_jobs=1)
	for n_jobs in (2, 8, -1):
		assert_identical(reader.read(chroms, starts, 90, n_jobs=n_jobs), one)


@pytest.mark.parametrize('without', [('libdeflate',), ('libdeflate', 'zlib')])
def test_read_without_inflate_libraries(fw, ref, monkeypatch, without):
	"""Chunks are inflated by libdeflate, by zlib's library where libdeflate
	cannot be loaded, and by the zlib module where neither can, with the
	same values each way."""

	if 'libdeflate' in without:
		monkeypatch.setattr(figwig.figwig, '_libdeflate_inflate', lambda: None)
	if 'zlib' in without:
		monkeypatch.setattr(figwig.figwig, '_zlib_uncompress', lambda: None)
	reader = FigWigReader(fw)
	rng = numpy.random.default_rng(4)
	chroms, starts = random_windows(rng, 100, 150)
	y = reader.read(chroms, starts, 150, dtype='float64')
	exp = expected(ref, chroms, starts, 150, [0, 1, 2, 3, 4], 'float64')
	exp[:, 2] = exp[:, 2].astype(numpy.float32)
	assert_identical(y, exp)


def test_read_empty(fw):
	reader = FigWigReader(fw)
	y = reader.read([], [], 10)
	assert y.shape == (0, 5, 10)


def test_read_argument_errors(fw):
	reader = FigWigReader(fw)
	with pytest.raises(ValueError, match="not one of the file's chromosomes"):
		reader.read('chr9', [0], 10)
	with pytest.raises(ValueError, match="start before 0"):
		reader.read('chr1', [-1], 10)
	with pytest.raises(ValueError, match="width must be at least 1"):
		reader.read('chr1', [0], 0)
	with pytest.raises(ValueError, match="not one of the file's tracks"):
		reader.read('chr1', [0], 10, tracks=['nope'])
	with pytest.raises(ValueError, match="out of range"):
		reader.read('chr1', [0], 10, tracks=[5])
	with pytest.raises(ValueError, match="at least one track"):
		reader.read('chr1', [0], 10, tracks=[])
	with pytest.raises(TypeError, match="names or indices"):
		reader.read('chr1', [0], 10, tracks=[1.5])
	with pytest.raises(TypeError, match="a list of them"):
		reader.read('chr1', [0], 10, tracks={'a': 1})
	with pytest.raises(TypeError, match="n_jobs"):
		reader.read('chr1', [0], 10, n_jobs=1.5)


def test_select(fw):
	reader = FigWigReader(fw)
	numpy.testing.assert_array_equal(reader.select(assay='ATAC-seq'), [0, 1])
	numpy.testing.assert_array_equal(reader.select(assay='ATAC-seq', strand='-'),
		[1])
	numpy.testing.assert_array_equal(reader.select(name='signal'), [2])
	numpy.testing.assert_array_equal(reader.select(dtype='float32'), [2])
	assert reader.select(assay='none').shape == (0,)
	assert reader.select(strand='+').dtype == numpy.int64


def test_metadata_round_trip(fw):
	reader = FigWigReader(fw)
	assert reader.genome == 'hg38'
	assert reader.attributes == {'made_by': 'test_figwig'}
	assert reader.chrom_sizes == CHROM_SIZES
	assert reader.tile == TILE
	assert [track['name'] for track in reader.tracks] == [track['name'] for
		track in TRACKS]
	assert [track['dtype'] for track in reader.tracks] == ['uint16', 'uint16',
		'float32', 'int16', 'uint8']
	assert [track['group'] for track in reader.tracks] == [0, 0, 1, 2, 2]
	assert reader.tracks[0]['strand'] == '+'
	assert reader.tracks[2]['assay'] == 'DNase-seq'
	assert 'FigWigReader' in repr(reader) and '5 tracks' in repr(reader)


def test_stats(fw, ref):
	reader = FigWigReader(fw)
	for i in range(5):
		values = numpy.concatenate([ref[chrom][i] for chrom in CHROM_SIZES])
		if i == 2:
			values = values.astype(numpy.float32).astype(numpy.float64)
		nonzero = values != 0
		assert reader.stats['nonzero'][i] == nonzero.sum()
		if i == 2:
			assert numpy.isnan(reader.stats['sum'][i])
			assert reader.stats['min'][i] == numpy.nanmin(values)
			assert reader.stats['max'][i] == numpy.nanmax(values)
		elif i == 4:
			assert reader.stats['sum'][i] == 0 and reader.stats['min'][i] == 0
			assert reader.stats['max'][i] == 0
		else:
			assert reader.stats['sum'][i] == values.sum()
			assert reader.stats['min'][i] == min(0, values.min())
			assert reader.stats['max'][i] == max(0, values.max())


def test_pickle(fw):
	reader = FigWigReader(fw)
	reader.read('chr1', [0], 10, tracks=['signal'])
	copy = pickle.loads(pickle.dumps(reader))
	assert copy._index[1] is not None and copy._index[0] is None
	assert_identical(copy.read('chr2', [100], 500), reader.read('chr2', [100],
		500))


def test_read_from_several_threads(fw):
	import threading

	reader = FigWigReader(fw)
	results = {}

	def run(k):
		results[k] = reader.read('chr1', [k * 100], 300, n_jobs=2)

	threads = [threading.Thread(target=run, args=(k,)) for k in range(6)]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	for k in range(6):
		assert_identical(results[k], reader.read('chr1', [k * 100], 300))


def test_read_figwig(fw, ref):
	reader = FigWigReader(fw)
	y = read_figwig(fw, 'chr1', [0, 100], 50, tracks=['atac_plus'])
	assert_identical(y, reader.read('chr1', [0, 100], 50, tracks=['atac_plus']))
	y = read_figwig(reader, 'chr1', [0, 100], 50, dtype='float64', n_jobs=1)
	assert y.shape == (2, 5, 50) and y.dtype == numpy.float64
	with pytest.raises(TypeError, match="path or a FigWigReader"):
		read_figwig(3, 'chr1', [0], 10)


###
# Writing


def test_same_bytes_whatever_threads_and_calls(tmp_path, ref):
	"""The file depends on the values and on which calls wrote which groups,
	not on the threads or on how the values were split between calls."""

	paths = [str(tmp_path / name) for name in ('a.fw', 'b.fw', 'c.fw', 'd.fw')]
	write_reference(paths[0], ref, n_jobs=1, calls=1)
	write_reference(paths[1], ref, n_jobs=4, calls=1)
	write_reference(paths[2], ref, n_jobs=1, calls=1, groups=5)
	write_reference(paths[3], ref, n_jobs=3, calls=5, width=37, groups=5)
	contents = []
	for path in paths:
		with open(path, 'rb') as handle:
			contents.append(handle.read())

	assert contents[0] == contents[1]
	assert contents[2] == contents[3]

	# With several groups, calls interleave the groups' chunks in the file,
	# and the chunks themselves are still the same.
	path = str(tmp_path / 'e.fw')
	write_reference(path, ref, n_jobs=3, calls=5, width=37)
	one, five = FigWigReader(paths[0]), FigWigReader(path)
	for g in range(3):
		kinds, sizes, _, _ = one._group_index(g)
		kinds_5, sizes_5, _, _ = five._group_index(g)
		numpy.testing.assert_array_equal(kinds, kinds_5)
		numpy.testing.assert_array_equal(sizes, sizes_5)


@pytest.mark.parametrize('tile', [16, 64, 1024, 65536])
@pytest.mark.parametrize('level', [0, 1, 6])
def test_tile_sizes_and_levels_give_the_same_values(tmp_path, ref, tile, level):
	path = str(tmp_path / 'tile.fw')
	write_reference(path, ref, tile=tile, level=level, n_jobs=1)
	reader = FigWigReader(path)
	assert reader.tile == tile
	rng = numpy.random.default_rng(5)
	chroms, starts = random_windows(rng, 60, 200)
	y = reader.read(chroms, starts, 200, dtype='float64')
	exp = expected(ref, chroms, starts, 200, [0, 1, 2, 3, 4], 'float64')
	exp[:, 2] = exp[:, 2].astype(numpy.float32)
	assert_identical(y, exp)


def test_intervals_write_what_windows_write(tmp_path, ref):
	"""Intervals of constant value and windows holding the same values give
	the same bytes."""

	windows, intervals = str(tmp_path / 'w.fw'), str(tmp_path / 'i.fw')
	chrom_sizes = {'chr1': 1000}
	tracks = ['a', 'b']
	bounds = [0, 10, 64, 65, 200, 500, 999, 1000]
	values = numpy.array([[1, 0], [0, 2], [3, 3], [0, 0], [5, 1], [0, 0],
		[9, 9]], dtype=numpy.float64)

	dense = numpy.zeros((2, 1000))
	for (a, b), v in zip(zip(bounds[:-1], bounds[1:]), values):
		dense[:, a:b] = v[:, None]

	write_figwig(windows, chrom_sizes, tracks, 'chr1', [0], dense[None], tile=64)
	write_figwig(intervals, chrom_sizes, tracks, 'chr1', bounds[:-1], values,
		ends=bounds[1:], tile=64)
	with open(windows, 'rb') as a, open(intervals, 'rb') as b:
		assert a.read() == b.read()

	y = FigWigReader(intervals).read('chr1', [0], 1000)
	numpy.testing.assert_array_equal(y[0], dense)


def test_groups_as_a_count(tmp_path):
	path = str(tmp_path / 'g.fw')
	with FigWigWriter(path, {'chr1': 100}, ['a', 'b', 'c', 'd', 'e'], groups=2,
			tile=16) as writer:
		writer.write('chr1', [0], numpy.ones((1, 5, 100)))

	reader = FigWigReader(path)
	assert [track['group'] for track in reader.tracks] == [0, 0, 1, 1, 2]
	numpy.testing.assert_array_equal(reader.read('chr1', [0], 100), 1)


def test_write_whole_groups_only(tmp_path):
	path = str(tmp_path / 'g.fw')
	writer = FigWigWriter(path, {'chr1': 100}, ['a', 'b', 'c'], groups=[['a',
		'b'], ['c']], tile=16)
	with pytest.raises(ValueError, match="whole groups"):
		writer.write('chr1', [0], numpy.ones((1, 1, 10)), tracks=['a'])
	with pytest.raises(ValueError, match="whole groups"):
		writer.write('chr1', [0], numpy.ones((1, 2, 10)), tracks=['a', 'c'])
	with pytest.raises(ValueError, match="named twice"):
		writer.write('chr1', [0], numpy.ones((1, 2, 10)), tracks=['c', 'c'])
	with pytest.raises(ValueError, match="not one of the tracks"):
		writer.write('chr1', [0], numpy.ones((1, 1, 10)), tracks=['z'])
	with pytest.raises(ValueError, match="out of range"):
		writer.write('chr1', [0], numpy.ones((1, 1, 10)), tracks=[3])
	with pytest.raises(TypeError, match="names or indices"):
		writer.write('chr1', [0], numpy.ones((1, 1, 10)), tracks=[None])

	writer.write('chr1', [0], numpy.ones((1, 2, 10)), tracks=['b', 'a'])
	writer.write('chr1', [0], numpy.full((1, 10), 2.0), tracks='c')
	writer.write('chr1', [10], numpy.full((1, 10), 3.0), tracks=2)
	writer.close()

	y = FigWigReader(path).read('chr1', [0], 20)
	numpy.testing.assert_array_equal(y[0, :2, :10], 1)
	numpy.testing.assert_array_equal(y[0, 2, :10], 2)
	numpy.testing.assert_array_equal(y[0, 2, 10:], 3)
	numpy.testing.assert_array_equal(y[0, :2, 10:], 0)


def test_write_order_and_overlap_errors(tmp_path):
	path = str(tmp_path / 'o.fw')
	writer = FigWigWriter(path, {'chr1': 1000, 'chr2': 1000}, ['a'], tile=64)
	ones = numpy.ones((2, 1, 50))
	with pytest.raises(ValueError, match="must not overlap"):
		writer.write('chr1', [0, 40], ones)
	with pytest.raises(ValueError, match="must not overlap"):
		writer.write('chr1', [0, 10], numpy.ones((2, 1)), ends=[20, 30])
	with pytest.raises(ValueError, match="starts before 0"):
		writer.write('chr1', [-1], ones[:1])
	with pytest.raises(ValueError, match="not one of the writer's"):
		writer.write('chr3', [0], ones[:1])
	with pytest.raises(ValueError, match="past its length"):
		writer.write('chr1', [980], ones[:1])
	with pytest.raises(ValueError, match="past its length"):
		writer.write('chr1', [980], numpy.ones((1, 1)), ends=[1001])
	with pytest.raises(ValueError, match="at or before its start"):
		writer.write('chr1', [10], numpy.ones((1, 1)), ends=[10])
	with pytest.raises(ValueError, match="one end per start"):
		writer.write('chr1', [10], numpy.ones((1, 1)), ends=[20, 30])
	with pytest.raises(ValueError, match="shape"):
		writer.write('chr1', [10], numpy.ones((2, 1, 5)))
	with pytest.raises(ValueError, match="shape"):
		writer.write('chr1', [10], numpy.ones((1, 2, 5)))
	with pytest.raises(ValueError, match="one base wide"):
		writer.write('chr1', [10], numpy.ones((1, 1, 0)))
	with pytest.raises(TypeError, match="numbers"):
		writer.write('chr1', [10], numpy.ones((1, 1, 5), dtype=bool))
	with pytest.raises(TypeError, match="integers"):
		writer.write('chr1', [1.5], ones[:1])
	with pytest.raises(ValueError, match="one-dimensional"):
		writer.write('chr1', [[1]], ones[:1])

	writer.write('chr2', [100, 0], ones)
	with pytest.raises(ValueError, match="before the end of what was written"):
		writer.write('chr2', [120], ones[:1])
	writer.write('chr2', [150], ones[:1])
	writer.write('chr1', [0], ones[:1])
	with pytest.raises(ValueError, match="written again after"):
		writer.write('chr2', [500], ones[:1])
	writer.close()

	reader = FigWigReader(path)
	y = reader.read('chr2', [0], 1000)[0, 0]
	assert y[:50].sum() == 50 and y[50:100].sum() == 0 and y[100:200].sum() == 100
	assert y[200:].sum() == 0
	with pytest.raises(ValueError, match="closed"):
		writer.write('chr1', [500], ones[:1])
	writer.close()


def test_window_past_the_end_needs_nan(tmp_path):
	path = str(tmp_path / 'p.fw')
	values = numpy.full((1, 2, 100), numpy.nan)
	values[0, :, :30] = 4
	with FigWigWriter(path, {'chr1': 80}, ['a', {'name': 'b', 'dtype':
			'float32'}], tile=16) as writer:
		writer.write('chr1', [50], values)
	y = FigWigReader(path).read('chr1', [0], 80, dtype='float64')
	numpy.testing.assert_array_equal(y[0, :, 50:], 4)
	numpy.testing.assert_array_equal(y[0, :, :50], 0)

	with FigWigWriter(path, {'chr1': 80}, ['a', 'b'], tile=16) as writer:
		values[0, 1, 35] = 1
		with pytest.raises(ValueError, match="only where its values are NaN or 0"):
			writer.write('chr1', [50], values)
		values[0, 1, 35] = 0
		writer.write('chr1', [50], values)
		with pytest.raises(ValueError, match="at or past its length"):
			writer.write('chr1', [80], numpy.full((1, 2, 10), numpy.nan))


def test_declared_dtypes_are_checked(tmp_path):
	path = str(tmp_path / 'd.fw')
	writer = FigWigWriter(path, {'chr1': 100}, [{'name': 'a', 'dtype': 'uint8'},
		{'name': 'b', 'dtype': 'int16'}], groups=1, tile=16)
	with pytest.raises(ValueError, match="is NaN"):
		writer.write('chr1', [0], numpy.full((1, 10), numpy.nan), tracks='a')
	with pytest.raises(ValueError, match="not an integer"):
		writer.write('chr1', [0], numpy.full((1, 10), 1.5), tracks='a')
	with pytest.raises(ValueError, match="outside 0 to 255"):
		writer.write('chr1', [0], numpy.full((1, 10), 256), tracks='a')
	with pytest.raises(ValueError, match="outside -32768 to 32767"):
		writer.write('chr1', [0], numpy.full((1, 10), 40000), tracks='b')
	writer.write('chr1', [0], numpy.full((1, 10), 255), tracks='a')
	writer.write('chr1', [0], numpy.full((1, 10), -300), tracks='b')
	writer.close()

	reader = FigWigReader(path)
	assert [track['dtype'] for track in reader.tracks] == ['uint8', 'int16']
	y = reader.read('chr1', [0], 10)
	assert y.dtype == numpy.int16
	numpy.testing.assert_array_equal(y[0], [[255] * 10, [-300] * 10])


@pytest.mark.parametrize('values, dtype', [
	([0, 0], 'uint8'), ([1, 255], 'uint8'), ([256, 7], 'uint16'),
	([65536, 1], 'uint32'), ([4294967295, 0], 'uint32'),
	([-1, 127], 'int8'), ([-129, 5], 'int16'), ([-40000, 2], 'int32'),
	([-1, 2147483648], 'float32'), ([1.5, 2], 'float32'),
	([numpy.nan, 1], 'float32'), ([4294967296, 0], 'float32')])
def test_inferred_dtype(tmp_path, values, dtype):
	path = str(tmp_path / 'i.fw')
	write_figwig(path, {'chr1': 32}, ['a'], 'chr1', [0], numpy.array(values,
		dtype=numpy.float64)[None, None, :], tile=16)
	reader = FigWigReader(path)
	assert reader.tracks[0]['dtype'] == dtype
	y = reader.read('chr1', [0], 2, dtype='float64')[0, 0]
	numpy.testing.assert_array_equal(y, numpy.array(values, dtype=numpy.float32
		if dtype == 'float32' else numpy.float64))


def test_chunk_kinds(tmp_path):
	"""A tile with no values takes no bytes, scattered values are stored
	sparsely, and a filled tile densely, in the narrowest type."""

	path = str(tmp_path / 'k.fw')
	values = numpy.zeros((1, 2, 256))
	values[0, 0, 70] = 3                 # tile 1: sparse, uint8
	values[0, :, 128:192] = 70000        # tile 2: dense, uint32
	values[0, 1, 200] = -2               # tile 3: sparse, int8
	write_figwig(path, {'chr1': 256}, ['a', 'b'], 'chr1', [0], values, tile=64)
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets = reader._group_index(0)
	assert [(int(k & 3), int(k >> 2)) for k in kinds] == [(0, 0), (2, 0),
		(1, 2), (2, 3)]
	assert sizes[0] == 0 and (sizes[1:] > 0).all() and (raws[1:] > 0).all()
	numpy.testing.assert_array_equal(reader.read('chr1', [0], 256,
		dtype='float64')[0], values[0])


def test_zero_tiles_take_no_space(tmp_path):
	sparse, dense = str(tmp_path / 's.fw'), str(tmp_path / 'd.fw')
	chrom_sizes = {'chr1': 1 << 20}
	write_figwig(sparse, chrom_sizes, ['a'], 'chr1', [5], numpy.ones((1, 1, 1)),
		tile=1024)
	write_figwig(dense, chrom_sizes, ['a'], 'chr1', [0], numpy.ones((1, 1,
		1 << 20)), tile=1024)
	assert os.path.getsize(sparse) < 10_000 < os.path.getsize(dense)


def test_writer_arguments(tmp_path):
	path = str(tmp_path / 'a.fw')
	with pytest.raises(TypeError, match="tile must be an integer"):
		FigWigWriter(path, {'chr1': 10}, ['a'], tile=16.0)
	for tile in (8, 100, 1 << 17):
		with pytest.raises(ValueError, match="power of two"):
			FigWigWriter(path, {'chr1': 10}, ['a'], tile=tile)
	with pytest.raises(TypeError, match="list of names or of dicts"):
		FigWigWriter(path, {'chr1': 10}, 'a')
	with pytest.raises(TypeError, match="name or a dict"):
		FigWigWriter(path, {'chr1': 10}, [1])
	with pytest.raises(ValueError, match="at least one track"):
		FigWigWriter(path, {'chr1': 10}, [])
	with pytest.raises(ValueError, match="non-empty string 'name'"):
		FigWigWriter(path, {'chr1': 10}, [{'dtype': 'uint8'}])
	with pytest.raises(ValueError, match="named twice"):
		FigWigWriter(path, {'chr1': 10}, ['a', 'a'])
	with pytest.raises(ValueError, match="'group' key"):
		FigWigWriter(path, {'chr1': 10}, [{'name': 'a', 'group': 0}])
	with pytest.raises(ValueError, match="dtype of track 'a' must be one of"):
		FigWigWriter(path, {'chr1': 10}, [{'name': 'a', 'dtype': 'float64'}])
	with pytest.raises(TypeError, match="not a dtype"):
		FigWigWriter(path, {'chr1': 10}, [{'name': 'a', 'dtype': 3.5}])
	with pytest.raises(TypeError, match="JSON serializable"):
		FigWigWriter(path, {'chr1': 10}, [{'name': 'a', 'x': object()}])
	with pytest.raises(TypeError, match="JSON serializable"):
		FigWigWriter(path, {'chr1': 10}, ['a'], attributes={'x': object()})
	with pytest.raises(TypeError, match="attributes must be a dict"):
		FigWigWriter(path, {'chr1': 10}, ['a'], attributes=[1])
	with pytest.raises(TypeError, match="genome must be a string"):
		FigWigWriter(path, {'chr1': 10}, ['a'], genome=38)
	with pytest.raises(ValueError, match="at least 1"):
		FigWigWriter(path, {'chr1': 10}, ['a'], groups=0)
	with pytest.raises(TypeError, match="groups must be"):
		FigWigWriter(path, {'chr1': 10}, ['a'], groups=True)
	with pytest.raises(TypeError, match="groups must be"):
		FigWigWriter(path, {'chr1': 10}, ['a'], groups=3.5)
	with pytest.raises(ValueError, match="at least one track"):
		FigWigWriter(path, {'chr1': 10}, ['a'], groups=[[]])
	with pytest.raises(ValueError, match="not one of the tracks"):
		FigWigWriter(path, {'chr1': 10}, ['a'], groups=[['b']])
	with pytest.raises(ValueError, match="in two groups"):
		FigWigWriter(path, {'chr1': 10}, ['a'], groups=[['a'], ['a']])
	with pytest.raises(ValueError, match="every track must be in a group"):
		FigWigWriter(path, {'chr1': 10}, ['a', 'b'], groups=[['a']])
	with pytest.raises(ValueError, match="more than 2\\*\\*29 values"):
		FigWigWriter(path, {'chr1': 10}, [str(i) for i in range(8193)],
			groups=8193, tile=65536)
	with pytest.raises(ValueError, match="engine must be one of"):
		FigWigWriter(path, {'chr1': 10}, ['a'], engine='brotli')
	with pytest.raises(ValueError, match="level must be from"):
		FigWigWriter(path, {'chr1': 10}, ['a'], engine='zlib', level=10)
	with pytest.raises(ValueError, match="n_jobs"):
		FigWigWriter(path, {'chr1': 10}, ['a'], n_jobs=0)
	with pytest.raises(ValueError, match="at least one chromosome"):
		FigWigWriter(path, {}, ['a'])

	writer = FigWigWriter(path, {'chr1': 10}, ['a', {'name': 'b', 'dtype':
		numpy.uint8}], genome='https://example.org/hg38.fa')
	assert 'FigWigWriter' in repr(writer) and '2 tracks' in repr(writer)
	assert writer.tracks[1]['dtype'] == 'uint8'
	writer.close()
	assert FigWigReader(path).genome == 'https://example.org/hg38.fa'


def test_abandoned_file_is_not_a_figwig(tmp_path):
	path = str(tmp_path / 'x.fw')
	with pytest.raises(RuntimeError):
		with FigWigWriter(path, {'chr1': 100}, ['a'], tile=16) as writer:
			writer.write('chr1', [0], numpy.ones((1, 1, 10)))
			raise RuntimeError('stop')

	with pytest.raises(ValueError, match="not a figwig file"):
		FigWigReader(path)


def test_empty_file_round_trip(tmp_path):
	path = str(tmp_path / 'e.fw')
	with FigWigWriter(path, {'chr1': 100, 'chr2': 50}, ['a', 'b'], tile=16):
		pass

	reader = FigWigReader(path)
	assert [track['dtype'] for track in reader.tracks] == ['uint8', 'uint8']
	y = reader.read(['chr1', 'chr2'], [0, 40], 100, dtype='float32')
	numpy.testing.assert_array_equal(y[0], 0)
	numpy.testing.assert_array_equal(y[1, :, :10], 0)
	assert numpy.isnan(y[1, :, 10:]).all()


def test_write_figwig_round_trip(tmp_path):
	path = str(tmp_path / 'w.fw')
	rng = numpy.random.default_rng(6)
	chroms = numpy.array(['chr2', 'chr1', 'chr1'])
	starts = numpy.array([300, 2500, 1000])
	y_hat = rng.random((3, 2, 1000), dtype=numpy.float32)
	write_figwig(path, CHROM_SIZES, ['plus', 'minus'], chroms, starts, y_hat,
		genome='hg38', tile=1024, level=1, n_jobs=1)
	y = read_figwig(path, chroms, starts, 1000)
	assert y.dtype == numpy.float32
	numpy.testing.assert_array_equal(y, y_hat)

	write_figwig(path, CHROM_SIZES, ['plus', 'minus'], 'chr1', [10, 20],
		[[1, 2], [3, 4]], ends=[15, 30])
	y = read_figwig(path, 'chr1', [10], 20)
	numpy.testing.assert_array_equal(y[0, :, :5], [[1] * 5, [2] * 5])
	numpy.testing.assert_array_equal(y[0, :, 5:10], 0)
	numpy.testing.assert_array_equal(y[0, :, 10:], [[3] * 10, [4] * 10])


def test_many_tracks_and_groups(tmp_path):
	"""A file of many small groups, read a few tracks at a time and all at
	once."""

	path = str(tmp_path / 'm.fw')
	rng = numpy.random.default_rng(7)
	n_tracks = 37
	values = rng.poisson(0.05, (3, n_tracks, 500)).astype(numpy.float64)
	names = ['t{}'.format(i) for i in range(n_tracks)]
	write_figwig(path, {'chr1': 2000}, names, 'chr1', [0, 500, 1500], values,
		groups=5, tile=128)
	reader = FigWigReader(path)
	assert max(track['group'] for track in reader.tracks) == 7
	y = reader.read('chr1', [0, 500, 1500], 500)
	assert y.dtype == numpy.uint8
	numpy.testing.assert_array_equal(y, values)
	picked = [36, 0, 17, 18]
	y = reader.read('chr1', [250], 500, tracks=picked, dtype='float64')
	numpy.testing.assert_array_equal(y[0, :, :250], values[0, picked, 250:])
	numpy.testing.assert_array_equal(y[0, :, 250:], values[1, picked, :250])


###
# Corrupt files


def corrupt_cases(fw):
	with open(fw, 'rb') as handle:
		data = bytearray(handle.read())

	cases = {}
	for n in (0, 10, 127, 128, 200, len(data) // 2, len(data) - 1):
		cases['truncated_{}'.format(n)] = bytes(data[:n])

	header = numpy.frombuffer(bytes(data[:128]), dtype=figwig.figwig._HEADER)

	def with_header(**fields):
		copy = header.copy()
		for key, value in fields.items():
			copy[key] = value
		return bytes(copy.tobytes()) + bytes(data[128:])

	cases['bad_magic'] = with_header(magic=1)
	cases['bad_version'] = with_header(version=2)
	cases['bad_tile'] = with_header(tile=100)
	cases['bad_codec'] = with_header(codec=7)
	cases['long_length'] = with_header(length=len(data) + 1)
	cases['short_length'] = with_header(length=100)
	cases['table_outside'] = with_header(chroms=len(data) + 5)
	cases['no_tracks'] = with_header(n_tracks=0)
	cases['more_tracks'] = with_header(n_tracks=6)
	cases['no_groups'] = with_header(n_groups=0)
	cases['more_groups'] = with_header(n_groups=4)
	cases['bad_metadata'] = with_header(metadata_length=3)

	for key in ('chroms', 'tracks', 'metadata', 'index'):
		offset = int(header[key][0])
		copy = bytearray(data)
		for k in range(offset, min(offset + 40, len(copy))):
			copy[k] ^= 0xFF
		cases['flipped_' + key] = bytes(copy)

	# A corrupt chunk: the first chunk's bytes, which lie right after the
	# header, with their middle flipped, and a kind byte of a kind the
	# format does not have.
	copy = bytearray(data)
	for k in range(140, 160):
		copy[k] ^= 0x55
	cases['flipped_chunk'] = bytes(copy)
	return cases


def test_corrupt_files_raise_only_value_errors(tmp_path, fw):
	for name, content in corrupt_cases(fw).items():
		path = str(tmp_path / (name + '.fw'))
		with open(path, 'wb') as handle:
			handle.write(content)

		try:
			reader = FigWigReader(path)
			reader.read(['chr1', 'chr2', 'chrM'], [0, 1000, 0], 2000)
			reader.read('chr1', [0], 5000)
		except ValueError as e:
			assert path in str(e) or 'figwig' in str(e).lower(), (name, e)
		except Exception as e:
			raise AssertionError("{} raised {}: {}".format(name, type(e).__name__,
				e))


def test_corrupt_index_runs_raise(tmp_path, fw):
	reader = FigWigReader(fw)
	with open(fw, 'rb') as handle:
		data = bytearray(handle.read())

	offset = int(reader._directory[0])
	n_runs = struct.unpack_from('<I', data, offset)[0]
	assert n_runs >= 1
	runs = numpy.frombuffer(bytes(data[offset + 4:offset + 4 + n_runs *
		figwig.figwig._RUN.itemsize]), dtype=figwig.figwig._RUN).copy()

	def write(runs_, name):
		path = str(tmp_path / name)
		copy = bytearray(data)
		copy[offset + 4:offset + 4 + runs.nbytes] = runs_.tobytes()
		with open(path, 'wb') as handle:
			handle.write(copy)
		return path

	bad = runs.copy()
	bad['bytes'][0] += 1
	with pytest.raises(ValueError, match="do not add up"):
		FigWigReader(write(bad, 'bytes.fw')).read('chr1', [0], 10)

	bad = runs.copy()
	bad['last'][0] = reader._n_tiles + 10
	with pytest.raises(ValueError, match="out of order or lies outside"):
		FigWigReader(write(bad, 'last.fw')).read('chr1', [0], 10)

	bad = runs.copy()
	bad['offset'][0] = 2 ** 63
	with pytest.raises(ValueError, match="outside the file"):
		FigWigReader(write(bad, 'offset.fw')).read('chr1', [0], 10)


def test_truncated_chunk_raises(tmp_path, fw):
	"""A file whose header is intact but whose chunks were cut short."""

	reader = FigWigReader(fw)
	kinds, sizes, raws, offsets = reader._group_index(0)
	live = numpy.flatnonzero(kinds)
	last = int(offsets[live[-1]] + sizes[live[-1]])
	with open(fw, 'rb') as handle:
		data = handle.read()

	path = str(tmp_path / 'cut.fw')
	with open(path, 'wb') as handle:
		handle.write(data[:last - 3] + data[last:])

	with pytest.raises(ValueError):
		FigWigReader(path).read('chr1', [0], 5000)


###
# Kernels


def test_encode_tiles_layout():
	"""The bytes of a dense and of a sparse chunk, read back by hand."""

	tile = 16
	values = numpy.zeros((1, 2, 32))
	values[0, 0, :16] = numpy.arange(16)         # tile 0: dense, uint8
	values[0, 1, :16] = 1
	values[0, 0, 20] = 300                        # tile 1: sparse, uint16
	values[0, 1, 17] = 5
	values[0, 1, 31] = 6
	starts, ends = numpy.array([0]), numpy.array([32])
	tiles = numpy.array([0, 1])
	lo, hi = numpy.array([0, 0]), numpy.array([1, 1])
	buffer = numpy.zeros((2, tile))
	out = numpy.zeros(1000, dtype=numpy.uint8)
	bounds = numpy.zeros(3, dtype=numpy.int64)
	kinds = numpy.zeros(2, dtype=numpy.uint8)
	stats = numpy.zeros((2, 5))
	stats[:, 2], stats[:, 3] = numpy.inf, -numpy.inf

	n = _encode_tiles(tile, tiles, lo, hi, starts, ends, values, True, buffer,
		False, False, True, out, bounds, kinds, stats)
	assert n == 2 and not buffer.any()
	assert kinds[0] == 1 | (0 << 2) and kinds[1] == 2 | (1 << 2)
	# The sparse chunk: two counts, where each track's offsets start and
	# the last end, three one-byte varints padded to 4, three uint16
	# values, padded to 8.
	assert bounds.tolist() == [0, 32, 32 + 8 + 12 + 4 + 6 + 2]
	numpy.testing.assert_array_equal(out[:16], numpy.arange(16))
	numpy.testing.assert_array_equal(out[16:32], 1)
	counts = out[32:40].view(numpy.uint32)
	assert counts.tolist() == [1, 2]
	assert out[40:52].view(numpy.uint32).tolist() == [0, 1, 3]
	assert out[52:56].tolist() == [4, 1, 14, 0]
	sparse_values = out[56:62].view(numpy.uint16)
	assert sparse_values.tolist() == [300, 5, 6]
	assert stats[:, 1].tolist() == [16, 18] and stats[:, 0].tolist() == [420, 27]
	assert stats[:, 3].tolist() == [300, 6] and stats[:, 4].tolist() == [0, 0]


def test_encode_tiles_varints_and_ones():
	"""Offsets of 128 and more take two varint bytes, and a tile whose
	nonzero values are all 1 stores no values."""

	tile = 1024
	values = numpy.zeros((1, 1, 1024))
	values[0, 0, [0, 127, 128, 1023]] = 1
	buffer = numpy.zeros((1, tile))
	out = numpy.zeros(100, dtype=numpy.uint8)
	bounds = numpy.zeros(2, dtype=numpy.int64)
	kinds = numpy.zeros(1, dtype=numpy.uint8)
	stats = numpy.zeros((1, 5))
	_encode_tiles(tile, numpy.array([0]), numpy.array([0]), numpy.array([1]),
		numpy.array([0]), numpy.array([1024]), values, True, buffer, False,
		False, True, out, bounds, kinds, stats)
	assert kinds[0] == 2 | (7 << 2)
	assert out[:4].view(numpy.uint32)[0] == 4
	assert out[4:12].view(numpy.uint32).tolist() == [0, 5]
	# The differences 0, 127 and 1 take a byte each, and 895 two.
	assert out[12:18].tolist() == [0, 127, 1, 0x80 | (895 & 127), 895 >> 7, 0]
	assert bounds[1] == 24 and stats[0].tolist() == [4, 4, 0, 1, 0]


def test_encode_tiles_keeps_an_open_tile():
	tile = 16
	values = numpy.ones((1, 1, 24))
	buffer = numpy.zeros((1, tile))
	out = numpy.zeros(100, dtype=numpy.uint8)
	bounds = numpy.zeros(3, dtype=numpy.int64)
	kinds = numpy.zeros(2, dtype=numpy.uint8)
	stats = numpy.zeros((1, 5))
	n = _encode_tiles(tile, numpy.array([0, 1]), numpy.array([0, 0]),
		numpy.array([1, 1]), numpy.array([0]), numpy.array([24]), values, True,
		buffer, False, True, True, out, bounds, kinds, stats)
	assert n == 1 and kinds[0] == 1 and buffer[0, :8].sum() == 8
	assert buffer[0, 8:].sum() == 0 and stats[0, 1] == 16

	# The open tile is then encoded by scanning the buffer, with the values
	# a later call adds.
	more = numpy.full((1, 1, 8), 2.0)
	n = _encode_tiles(tile, numpy.array([1]), numpy.array([0]), numpy.array([1]),
		numpy.array([24]), numpy.array([32]), more, True, buffer, True, False,
		True, out, bounds, kinds, stats)
	assert n == 1 and kinds[0] == 1 and not buffer.any()
	numpy.testing.assert_array_equal(out[:16], [1] * 8 + [2] * 8)
	assert stats[0, 1] == 32 and stats[0, 0] == 40


def test_check_chunks_rejects_short_and_overrunning_chunks():
	tile, n_tracks = 16, 1
	data = numpy.zeros(64, dtype=numpy.uint8)
	failed = numpy.zeros(4, dtype=numpy.int64)
	positions = numpy.array([0, 0, 0, 0], dtype=numpy.int64)
	# dense uint8 of 16 bytes: a length of 16 passes, 15 fails; an unknown
	# kind fails; a sparse chunk whose one offset is past the tile fails.
	data[:4].view(numpy.uint32)[0] = 1
	data[4:12].view(numpy.uint32)[:] = [0, 1]
	data[12] = 16
	slots = numpy.array([0])
	kinds = numpy.array([1, 1, 3, 2], dtype=numpy.uint8)
	lengths = numpy.array([16, 15, 16, 20], dtype=numpy.int64)
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 3
	assert failed.tolist() == [0, 1, 1, 1]

	# An offset of 15 passes; a varint that runs past the track's bytes, or
	# bytes the varints do not fill, fail; the offsets of a track that is
	# not read are not checked.
	data[12] = 15
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 2 and failed[3] == 0
	data[12] = 128 | 15
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 3
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		numpy.array([-1]), failed) == 2
	data[12] = 15
	data[8:12].view(numpy.uint32)[0] = 2
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 3


def test_read_tiles_zero_chunk_writes_zeros():
	out = numpy.full((1, 2, 10), 9, dtype=numpy.int32)
	_read_tiles(16, numpy.zeros(8, dtype=numpy.uint8), numpy.zeros(0,
		dtype=numpy.int64), numpy.zeros(0, dtype=numpy.uint8), 2,
		numpy.array([1, -1]), numpy.array([-1]), numpy.array([0]),
		numpy.array([4]), numpy.array([0]), out, False)
	numpy.testing.assert_array_equal(out[0, 1], 0)
	numpy.testing.assert_array_equal(out[0, 0], 9)


def test_without_jit(tmp_path, ref):
	"""With NUMBA_DISABLE_JIT=1 the kernels are plain Python functions, and
	a small file writes and reads the same values."""

	script = """
import numpy, sys
from figwig import FigWigWriter
from figwig import FigWigReader
values = numpy.zeros((1, 2, 100))
values[0, 0, 5] = 3
values[0, 1, 40:70] = 2.5
with FigWigWriter(sys.argv[1], {'chr1': 100}, ['a', 'b'], tile=32) as w:
	w.write('chr1', [0], values)
y = FigWigReader(sys.argv[1]).read('chr1', [0, 50], 60, dtype='float64')
assert numpy.array_equal(y[0], values[0, :, :60]), y[0]
assert numpy.array_equal(y[1, :, :50], values[0, :, 50:]) and numpy.isnan(y[1, :, 50:]).all()
print('ok')
"""
	result = subprocess.run([sys.executable, '-c', script, str(tmp_path /
		'nojit.fw')], env=dict(os.environ, NUMBA_DISABLE_JIT='1',
		OPENBLAS_NUM_THREADS='1'), capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert result.stdout.strip() == 'ok'


###
# Sparse reads, sums per label, and nonzero bases as entries


def sparse_to_dense(found, dtype='float64'):
	out = numpy.zeros(found.shape, dtype=dtype)
	out[found.rows, found.channels, found.offsets] = found.values
	return out


def test_sparse_read_matches_dense(fw, ref, read_batching):
	rng = numpy.random.default_rng(20)
	chroms, starts = random_windows(rng, 200, 150)
	reader = FigWigReader(fw)
	found = reader.read(chroms, starts, 150, sparse=True, dtype='float64')
	assert isinstance(found, SparseWindows) and found.shape == (200, 5, 150)
	dense = reader.read(chroms, starts, 150, dtype='float64')
	# Bases past a chromosome's end are NaN in every channel of the dense
	# read and absent from the entries; a NaN value is an entry.
	past = numpy.isnan(dense).all(axis=1)
	dense[numpy.broadcast_to(past[:, None, :], dense.shape)] = 0
	numpy.testing.assert_array_equal(sparse_to_dense(found), dense)
	assert (found.values != 0).all()
	order = numpy.lexsort((found.offsets, found.channels, found.rows))
	numpy.testing.assert_array_equal(order, numpy.arange(len(order)))

	found = reader.read(chroms, starts, 150, tracks=['negative', 'atac_plus',
		'negative'], sparse=True)
	assert found.values.dtype == numpy.int32 and found.shape == (200, 3, 150)
	dense = reader.read(chroms, starts, 150, tracks=['negative', 'atac_plus',
		'negative'])
	numpy.testing.assert_array_equal(sparse_to_dense(found, 'int32'), dense)

	found = reader.read(chroms, starts, 150, tracks='signal', sparse=True)
	assert found.shape == (200, 1, 150) and found.values.dtype == numpy.float32
	empty = reader.read([], [], 10, sparse=True)
	assert empty.shape == (0, 5, 10) and len(empty.values) == 0


def test_labels_sum_tracks(fw, ref, read_batching):
	rng = numpy.random.default_rng(21)
	chroms, starts = random_windows(rng, 100, 150)
	reader = FigWigReader(fw)
	labels = numpy.array([0, 0, 1, -1, 1])
	y = reader.read(chroms, starts, 150, labels=labels)
	assert y.shape == (100, 2, 150) and y.dtype == numpy.float32
	dense = reader.read(chroms, starts, 150, dtype='float64')
	exp = numpy.stack([dense[:, 0] + dense[:, 1], dense[:, 2] + dense[:, 4]],
		axis=1)
	assert_identical(y.astype(numpy.float64), exp.astype(numpy.float32).astype(
		numpy.float64))

	y = reader.read(chroms, starts, 150, tracks=['atac_minus', 'atac_plus'],
		labels=[0, 0], dtype='int64')
	assert y.dtype == numpy.int64
	numpy.testing.assert_array_equal(y[:, 0], numpy.nan_to_num(dense[:, 0] +
		dense[:, 1]))

	with pytest.raises(ValueError, match="one integer per track"):
		reader.read(chroms, starts, 150, labels=[0, 1])
	with pytest.raises(ValueError, match="cannot both"):
		reader.read(chroms, starts, 150, labels=labels, sparse=True)
	with pytest.raises(ValueError, match="-1 or from 0"):
		reader.read(chroms, starts, 150, labels=[0, 0, 0, -2, 0])
	with pytest.raises(ValueError, match="each track is read once"):
		reader.read(chroms, starts, 150, tracks=[0, 0], labels=[0, 0])
	with pytest.raises(ValueError, match="out cannot be given"):
		reader.read(chroms, starts, 150, sparse=True, out=numpy.zeros(1))


def test_dense_guard(fw, monkeypatch):
	reader = FigWigReader(fw)
	monkeypatch.setattr(figwig.figwig, '_MAX_DENSE_BYTES', 1000)
	with pytest.raises(ValueError, match="sparse=True or labels="):
		reader.read('chr1', [0], 1000)
	out = numpy.empty((1, 5, 1000), dtype=numpy.float32)
	assert reader.read('chr1', [0], 1000, out=out) is out
	assert reader.read('chr1', [0], 1000, sparse=True).shape == (1, 5, 1000)


def test_read_entries(fw, ref):
	reader = FigWigReader(fw)
	entries = reader.read_entries(tracks=['atac_plus', 'signal'], chroms=['chr2',
		'chrM'])
	assert isinstance(entries, SparseEntries)
	expected = []
	for tid, chrom in ((1, 'chr2'), (2, 'chrM')):
		for track in (0, 2):
			positions = numpy.flatnonzero(ref[chrom][track] != 0)
			expected.append(numpy.stack([numpy.full(len(positions), tid), positions,
				numpy.full(len(positions), track), numpy.nan_to_num(ref[chrom][
				track, positions]).astype(numpy.float32)], axis=1))
	expected = numpy.concatenate(expected)
	got = numpy.stack([entries.chroms, entries.positions, entries.tracks,
		entries.values], axis=1)
	numpy.testing.assert_array_equal(got, expected)

	everything = reader.read_entries()
	assert len(everything.values) == int(reader.stats['nonzero'].sum())
	with pytest.raises(ValueError, match="not one of the file's chromosomes"):
		reader.read_entries(chroms=['chr9'])


###
# Nonzero bases as input, and matrix chunks


def cells_reference(n_cells=120, random_state=30):
	"""Sparse per-cell counts on the test chromosomes, as (dense reference,
	entries per chromosome)."""

	rng = numpy.random.default_rng(random_state)
	ref = {chrom: numpy.zeros((n_cells, length)) for chrom, length in
		CHROM_SIZES.items()}
	entries = {}
	for chrom, length in CHROM_SIZES.items():
		cells, positions, values = [], [], []
		for cell in range(n_cells):
			where = numpy.unique(rng.integers(0, length, max(1, length // 400)))
			count = rng.integers(1, 4, len(where))
			ref[chrom][cell, where] = count
			cells.append(numpy.full(len(where), cell))
			positions.append(where)
			values.append(count)
		entries[chrom] = (numpy.concatenate(positions), numpy.concatenate(cells),
			numpy.concatenate(values))
	return ref, entries


@pytest.fixture(scope='session')
def cells(tmp_path_factory):
	ref, entries = cells_reference()
	names = ['cell{}'.format(i) for i in range(120)]
	paths = {}
	for order, groups in (('positions', 120), ('positions', 8), ('tracks', 1)):
		path = str(tmp_path_factory.mktemp('cells') / '{}_{}.fw'.format(order,
			groups))
		with FigWigWriter(path, CHROM_SIZES, names, tile=64, groups=groups,
				order=order, n_jobs=3, level=0 if groups == 120 else None) as writer:
			for chrom, (positions, which, values) in entries.items():
				writer.write_sparse(chrom, positions, which, values)
		paths[order, groups] = path
	return ref, entries, paths


@pytest.mark.parametrize('key', [('positions', 120), ('positions', 8),
	('tracks', 1)])
def test_write_sparse_reads_back(cells, key, read_batching):
	ref, entries, paths = cells
	reader = FigWigReader(paths[key])
	assert reader.order == key[0]
	rng = numpy.random.default_rng(31)
	chroms, starts = random_windows(rng, 150, 200)
	subset = numpy.sort(rng.choice(120, 30, replace=False))
	# One cell of 120 is looked up among a matrix chunk's present tracks by
	# binary search; thirty are found by walking them.
	for tracks in (None, subset, subset[:1]):
		picked = numpy.arange(120) if tracks is None else tracks
		y = reader.read(chroms, starts, 200, tracks=tracks, dtype='float64')
		exp = numpy.zeros((150, len(picked), 200))
		for j, (chrom, start) in enumerate(zip(chroms, starts)):
			length = CHROM_SIZES[chrom]
			if start < length:
				piece = ref[chrom][picked, start:min(length, start + 200)]
				exp[j, :, :piece.shape[1]] = piece
			exp[j, :, max(0, length - start):] = numpy.nan
		assert_identical(y, exp)
		found = reader.read(chroms, starts, 200, tracks=tracks, sparse=True,
			dtype='float64')
		numpy.testing.assert_array_equal(sparse_to_dense(found), numpy.nan_to_num(
			exp))

	labels = rng.integers(0, 4, 120)
	y = reader.read(chroms, starts, 200, labels=labels, dtype='float64')
	exp = numpy.zeros((150, 4, 200))
	for j, (chrom, start) in enumerate(zip(chroms, starts)):
		length = CHROM_SIZES[chrom]
		for label in range(4):
			piece = ref[chrom][labels == label, start:min(length, start + 200)]
			exp[j, label, :piece.shape[1]] = piece.sum(axis=0)
		exp[j, :, max(0, length - start):] = numpy.nan
	assert_identical(y, exp)

	entries_out = reader.read_entries(tracks=[3, 77])
	for chrom_id, position, track, value in zip(*entries_out):
		assert ref[list(CHROM_SIZES)[chrom_id]][track, position] == value
	assert len(entries_out.values) == sum(int((ref[chrom][[3, 77]] != 0).sum())
		for chrom in CHROM_SIZES)
	assert [track['dtype'] for track in reader.tracks] == ['uint8'] * 120


def test_matrix_chunks_are_chosen_for_many_sparse_tracks(cells, tmp_path,
	monkeypatch):
	ref, entries, paths = cells
	reader = FigWigReader(paths['positions', 120])
	kinds, sizes, raws, offsets = reader._group_index(0)
	counts = numpy.bincount(kinds & 3, minlength=4)
	assert counts[3] > 0

	# Without matrix chunks the same uncompressed values take more bytes,
	# and a compressed file has none.
	monkeypatch.setattr(figwig.figwig, '_MATRIX_CHUNKS', False)
	path = str(tmp_path / 'old.fw')
	with FigWigWriter(path, CHROM_SIZES, ['cell{}'.format(i) for i in range(
			120)], tile=64, groups=120, level=0) as writer:
		for chrom, (positions, which, values) in entries.items():
			writer.write_sparse(chrom, positions, which, values)
	old = FigWigReader(path)
	kinds, _, _, _ = old._group_index(0)
	assert numpy.bincount(kinds & 3, minlength=4)[3] == 0
	assert os.path.getsize(path) > os.path.getsize(paths['positions', 120])
	y = reader.read('chr1', [0], 5000, dtype='float64')
	assert_identical(old.read('chr1', [0], 5000, dtype='float64'), y)
	kinds, _, _, _ = FigWigReader(paths['positions', 8])._group_index(0)
	assert numpy.bincount(kinds & 3, minlength=4)[3] == 0


def test_write_sparse_arguments(tmp_path):
	path = str(tmp_path / 's.fw')
	writer = FigWigWriter(path, {'chr1': 1000, 'chr2': 500}, ['a', 'b', {'name':
		'c', 'dtype': 'uint8'}], groups=[['a', 'b'], ['c']], tile=64)
	with pytest.raises(ValueError, match="outside the chromosome"):
		writer.write_sparse('chr1', [1000], 'c', [1])
	with pytest.raises(ValueError, match="outside the chromosome"):
		writer.write_sparse('chr1', [-1], 'c', [1])
	with pytest.raises(ValueError, match="both name base"):
		writer.write_sparse('chr1', [5, 5], ['c', 'c'], [1, 2])
	with pytest.raises(ValueError, match="one value per position"):
		writer.write_sparse('chr1', [5, 6], 'c', [1])
	with pytest.raises(ValueError, match="one track per position"):
		writer.write_sparse('chr1', [5, 6], ['c'], [1, 2])
	with pytest.raises(ValueError, match="out of range"):
		writer.write_sparse('chr1', [5], [7], [1])
	with pytest.raises(TypeError, match="names or indices"):
		writer.write_sparse('chr1', [5], [1.5], [1])
	with pytest.raises(TypeError, match="numbers"):
		writer.write_sparse('chr1', [5], 'c', ['x'])
	with pytest.raises(ValueError, match="is uint8, and a value is outside"):
		writer.write_sparse('chr1', [5], 'c', [300])
	with pytest.raises(ValueError, match="not one of the writer's"):
		writer.write_sparse('chr3', [5], 'c', [1])

	writer.write_sparse('chr1', [5, 70, 6], numpy.array(['a', 'b', 'b']), [1, 2,
		3])
	writer.write_sparse('chr1', [200], 2, [9])
	with pytest.raises(ValueError, match="before the end of what was written"):
		writer.write_sparse('chr1', [60], 'a', [1])
	with pytest.raises(ValueError, match="which an earlier call already wrote"):
		writer.write_sparse('chr1', [71], 'a', [1])
	with pytest.raises(ValueError, match="which an earlier call already wrote"):
		writer.write('chr1', [72], numpy.ones((1, 2, 10)), tracks=['a', 'b'])
	writer.write('chr1', [128], numpy.ones((1, 2, 10)), tracks=['a', 'b'])
	with pytest.raises(ValueError, match="was begun by write"):
		writer.write_sparse('chr1', [140], 'a', [1])
	writer.write_sparse('chr1', [300], 'a', [4])
	writer.write_sparse('chr2', [], [], [])
	writer.close()

	reader = FigWigReader(path)
	y = reader.read('chr1', [0], 400, dtype='float64')[0]
	assert y[0, 5] == 1 and y[1, 70] == 2 and y[1, 6] == 3 and y[2, 200] == 9
	assert y[0, 128:138].sum() == 10 and y[1, 128:138].sum() == 10
	assert y[0, 300] == 4 and y.sum() == 1 + 2 + 3 + 9 + 20 + 4
	with pytest.raises(ValueError, match="closed"):
		writer.write_sparse('chr2', [1], 'c', [1])


###
# Track order


def test_track_order_reads_as_position_order(fw, track_fw, ref, read_batching):
	one, other = FigWigReader(fw), FigWigReader(track_fw)
	assert other.order == 'tracks' and one.order == 'positions'
	assert 'track order' in repr(other)
	assert [track['dtype'] for track in other.tracks] == [track['dtype'] for
		track in one.tracks]
	assert [track['group'] for track in other.tracks] == [0, 1, 2, 3, 4]
	numpy.testing.assert_array_equal(other.stats['nonzero'], one.stats['nonzero'])
	numpy.testing.assert_array_equal(other.stats['max'], one.stats['max'])

	rng = numpy.random.default_rng(40)
	chroms, starts = random_windows(rng, 150, 120)
	for tracks in (None, ['negative', 'atac_plus', 'negative'], 'signal', [4]):
		a = one.read(chroms, starts, 120, tracks=tracks, dtype='float64')
		b = other.read(chroms, starts, 120, tracks=tracks, dtype='float64')
		assert_identical(a, b)
		a = one.read(chroms, starts, 120, tracks=tracks, sparse=True)
		b = other.read(chroms, starts, 120, tracks=tracks, sparse=True)
		for x, y in zip(a, b):
			numpy.testing.assert_array_equal(x, y)
	a = one.read(chroms, starts, 120, labels=[0, 1, 0, 1, -1])
	b = other.read(chroms, starts, 120, labels=[0, 1, 0, 1, -1])
	assert_identical(a, b)
	for x, y in zip(one.read_entries(), other.read_entries()):
		numpy.testing.assert_array_equal(x, y)

	out = numpy.ones((3, 5, 120), dtype=numpy.float32)
	other.read(chroms[:3], starts[:3], 120, out=out)
	assert_identical(out, one.read(chroms[:3], starts[:3], 120))


def test_track_order_blocks_and_calls(tmp_path, ref, monkeypatch):
	"""Blocks of a few entries, written over several calls, read the same."""

	monkeypatch.setattr(figwig.figwig, '_BLOCK_ENTRIES', 16)
	path = str(tmp_path / 'blocks.fw')
	write_reference(path, ref, order='tracks', calls=7, width=37, n_jobs=1)
	reader = FigWigReader(path)
	assert len(reader._group_index(0)) > 20
	assert reader.read('chr1', [0], 5000, dtype='float64')[0, 2].tolist() == 		ref['chr1'][2].astype(numpy.float32).astype(numpy.float64).tolist() or 		numpy.isnan(ref['chr1'][2]).any()
	a = FigWigReader(path).read('chr1', [0], 5000, tracks=['atac_plus'])
	numpy.testing.assert_array_equal(a[0, 0], ref['chr1'][0])
	assert reader.read_entries(tracks=['atac_plus']).values.shape == (int(
		reader.stats['nonzero'][0]),)


def test_track_order_writer(tmp_path):
	path = str(tmp_path / 't.fw')
	writer = FigWigWriter(path, {'chr1': 300, 'chr2': 100}, ['a', 'b'],
		order='csr', groups=[['a', 'b']])
	assert writer.order == 'tracks' and writer.level == 0
	assert [track['group'] for track in writer.tracks] == [0, 1]
	writer.write('chr1', [0, 10], [[1.5, 0], [0, 2]], ends=[5, 20], tracks=['a',
		'b'])
	writer.write('chr1', [100], numpy.array([[0, 3, 0, 4]]), tracks='b')
	with pytest.raises(ValueError, match="before the end of what was written"):
		writer.write('chr1', [50], numpy.array([[1, 1]]), tracks='b')
	writer.write('chr1', [50], numpy.array([[1, 1]]), tracks='a')
	writer.write('chr2', [0], numpy.array([[7]]), tracks='a')
	with pytest.raises(ValueError, match="written again after"):
		writer.write('chr1', [200], numpy.array([[1]]), tracks='a')
	writer.close()

	reader = FigWigReader(path)
	y = reader.read(['chr1', 'chr2'], [0, 0], 300, dtype='float64')
	numpy.testing.assert_array_equal(y[0, 0, :5], 1.5)
	numpy.testing.assert_array_equal(y[0, 0, 50:52], 1)
	numpy.testing.assert_array_equal(y[0, 1, 10:20], 2)
	assert y[0, 1, 101] == 3 and y[0, 1, 103] == 4 and y[1, 0, 0] == 7
	assert numpy.isnan(y[1, :, 100:]).all()
	assert y[0].sum() == 1.5 * 5 + 2 + 20 + 7 and numpy.nansum(y[1]) == 7
	assert [track['dtype'] for track in reader.tracks] == ['float32', 'uint8']

	with pytest.raises(ValueError, match="order must be"):
		FigWigWriter(path, {'chr1': 300}, ['a'], order='rows')


def test_track_order_compressed(tmp_path, ref):
	path = str(tmp_path / 'c.fw')
	write_reference(path, ref, order='tracks', level=6)
	reader = FigWigReader(path)
	assert reader._codec == 1
	y = reader.read('chr1', [100], 500, dtype='float64')[0]
	exp = ref['chr1'][:, 100:600].copy()
	exp[2] = exp[2].astype(numpy.float32)
	assert_identical(y, exp)


def test_corrupt_track_order_files_raise_only_value_errors(tmp_path, track_fw):
	with open(track_fw, 'rb') as handle:
		data = bytearray(handle.read())
	reader = FigWigReader(track_fw)
	offset = int(reader._directory[0])
	cases = {'truncated': bytes(data[:len(data) // 2]), 'flipped_index':
		bytes(data[:offset + 8]) + bytes(b ^ 0xFF for b in data[offset + 8:offset
		+ 48]) + bytes(data[offset + 48:]), 'flipped_block': bytes(data[:140]) +
		bytes(b ^ 0x55 for b in data[140:170]) + bytes(data[170:])}
	for name, content in cases.items():
		path = str(tmp_path / (name + '.fw'))
		with open(path, 'wb') as handle:
			handle.write(content)
		try:
			FigWigReader(path).read(['chr1', 'chr2'], [0, 0], 3000)
		except ValueError:
			pass
		except Exception as e:
			raise AssertionError("{} raised {}".format(name, type(e).__name__))


def test_labels_sum_the_same_on_threads(cells, monkeypatch):
	"""Sums per label gather tracks from several groups or tracks, which on
	several threads land in the same cells of the output."""

	monkeypatch.setattr(figwig.figwig, '_BATCH_CHUNKS', 2)
	ref, entries, paths = cells
	rng = numpy.random.default_rng(50)
	chroms, starts = random_windows(rng, 200, 300)
	labels = rng.integers(0, 3, 120)
	for key in (('positions', 8), ('tracks', 1), ('positions', 120)):
		reader = FigWigReader(paths[key])
		one = reader.read(chroms, starts, 300, labels=labels, n_jobs=1)
		for _ in range(5):
			many = reader.read(chroms, starts, 300, labels=labels, n_jobs=8)
			assert_identical(many, one)
