# test_figwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import sys
import pickle
import shutil
import struct
import zlib
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
		# A NaN base is stored and read back, but is no part of the sum
		# and no nonzero base, so a file written with its uncovered bases
		# NaN keeps the statistics of its covered ones.
		nonzero = (values != 0) & ~numpy.isnan(values)
		assert reader.stats['nonzero'][i] == nonzero.sum()
		if i == 2:
			assert reader.stats['sum'][i] == pytest.approx(numpy.nansum(values),
				rel=1e-12)
			assert reader.stats['min'][i] == numpy.nanmin(values)
			assert reader.stats['max'][i] == numpy.nanmax(values)
		elif i == 4:
			assert reader.stats['sum'][i] == 0 and reader.stats['min'][i] == 0
			assert reader.stats['max'][i] == 0
		else:
			# The minimum and maximum are those of the nonzero bases, so a
			# track of negative values has a negative maximum.
			assert reader.stats['sum'][i] == values.sum()
			assert reader.stats['min'][i] == values[values != 0].min()
			assert reader.stats['max'][i] == values[values != 0].max()


def test_pickle(fw):
	reader = FigWigReader(fw)
	reader.read('chr1', [0], 10, tracks=['signal'])
	copy = pickle.loads(pickle.dumps(reader))
	# A pickled reader carries no index, and reads it again where it lands.
	assert copy._index[1] is None and copy._index[0] is None
	assert_identical(copy.read('chr2', [100], 500), reader.read('chr2', [100],
		500))
	assert copy._index[1] is not None


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
		kinds, sizes, _, _, _ = one._group_index(g)
		kinds_5, sizes_5, _, _, _ = five._group_index(g)
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
	with pytest.raises(ValueError, match="not one of the writer's tracks"):
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
	([-1, 2147483648], 'float64'), ([1.5, 2], 'float32'),
	([numpy.nan, 1], 'float32'), ([4294967296, 0], 'float64'),
	([16777217, 0.5], 'float32'), ([0.1, 1e-50], 'float32')])
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
	values[0, :, 128:192] = 70000 + numpy.arange(64)   # tile 2: dense, uint32
	values[0, 1, 200] = -2               # tile 3: sparse, int8
	write_figwig(path, {'chr1': 256}, ['a', 'b'], 'chr1', [0], values, tile=64)
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets, _ = reader._group_index(0)
	assert [(int(k & 7), int((k >> 3) & 15)) for k in kinds] == [(0, 0),
		(2, 0), (1, 2), (2, 3)]
	assert sizes[0] == 0 and (sizes[1:] > 0).all() and (raws[1:] > 0).all()
	numpy.testing.assert_array_equal(reader.read('chr1', [0], 256,
		dtype='float64')[0], values[0])

	# A tile of one value is stored as runs, and a filled tile of a few
	# values too, when that is smaller than storing every base.
	values[0, :, 128:192] = 70000
	write_figwig(path, {'chr1': 256}, ['a', 'b'], 'chr1', [0], values, tile=64)
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets, _ = reader._group_index(0)
	assert (int(kinds[2] & 7), int((kinds[2] >> 3) & 15)) == (4, 2)
	assert raws[2] < 64
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
	with pytest.raises(ValueError, match="no tracks"):
		with FigWigWriter(path, {'chr1': 10}, []):
			pass
	with pytest.raises(ValueError, match="non-empty string 'name'"):
		FigWigWriter(path, {'chr1': 10}, [{'dtype': 'uint8'}])
	with pytest.raises(ValueError, match="named twice"):
		FigWigWriter(path, {'chr1': 10}, ['a', 'a'])
	with pytest.raises(ValueError, match="'group' key"):
		FigWigWriter(path, {'chr1': 10}, [{'name': 'a', 'group': 0}])
	with pytest.raises(ValueError, match="dtype of track 'a' must be one of"):
		FigWigWriter(path, {'chr1': 10}, [{'name': 'a', 'dtype': 'float16'}])
	with pytest.raises(ValueError, match="NaN and infinity are not JSON"):
		FigWigWriter(path, {'chr1': 10}, [{'name': 'a', 'x': float('nan')}])
	with pytest.raises(ValueError, match="longer than 65535 bytes"):
		FigWigWriter(path, {'c' * 70000: 10}, ['a'])
	with pytest.raises(ValueError, match="holds at most"):
		FigWigWriter(path, {'chr1': 2 ** 32 - 1}, ['a'], tile=16)
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

	# The partial file is removed, and the path holds what it held before:
	# here, nothing.
	assert not os.path.exists(path)
	assert not any(name.endswith('.partial') for name in os.listdir(tmp_path))


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
	cases['bad_version'] = with_header(version=0)
	cases['newer_version'] = with_header(version=2)
	cases['huge_metadata'] = with_header(metadata_length=2 ** 50)
	cases['huge_tracks'] = with_header(n_tracks=2 ** 32 - 1)
	cases['huge_groups'] = with_header(n_groups=2 ** 32 - 1)
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
	at = offset + 8
	runs = numpy.frombuffer(bytes(data[at:at + n_runs *
		figwig.figwig._RUN.itemsize]), dtype=figwig.figwig._RUN).copy()
	n_bytes = struct.unpack_from('<II', data, at + runs.nbytes)[1]
	end = at + runs.nbytes + 8 + n_bytes

	def write(runs_, name, recompute=True):
		# The record's CRC is recomputed, so that the checks on the runs
		# themselves are what refuses the file.
		path = str(tmp_path / name)
		copy = bytearray(data)
		copy[at:at + runs.nbytes] = runs_.tobytes()
		if recompute:
			struct.pack_into('<I', copy, offset + 4, zlib.crc32(copy[at:end]))
		with open(path, 'wb') as handle:
			handle.write(copy)
		return path

	bad = runs.copy()
	bad['bytes'][0] += 1
	with pytest.raises(ValueError, match="does not match its checksum"):
		FigWigReader(write(bad, 'crc.fw', recompute=False)).read('chr1', [0], 10)
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
	kinds, sizes, raws, offsets, _ = reader._group_index(0)
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
	classes = numpy.zeros(2, dtype=numpy.int64)

	n = _encode_tiles(tile, tiles, lo, hi, starts, ends, values, True, buffer,
		False, classes, True, False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert n == 2 and not buffer.any()
	assert kinds[0] == 1 | (0 << 3) and kinds[1] == 2 | (1 << 3)
	# The sparse chunk: the entries in all, the bytes of each track's
	# offsets, three one-byte varints padded to 4, three uint16 values,
	# padded to 8.
	assert bounds.tolist() == [0, 32, 32 + 4 + 8 + 4 + 6 + 2]
	numpy.testing.assert_array_equal(out[:16], numpy.arange(16))
	numpy.testing.assert_array_equal(out[16:32], 1)
	assert out[32:44].view(numpy.uint32).tolist() == [3, 1, 2]
	assert out[44:48].tolist() == [4, 1, 14, 0]
	sparse_values = out[48:54].view(numpy.uint16)
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
	out = numpy.zeros(10000, dtype=numpy.uint8)
	bounds = numpy.zeros(2, dtype=numpy.int64)
	kinds = numpy.zeros(1, dtype=numpy.uint8)
	stats = numpy.zeros((1, 5))
	classes = numpy.zeros(1, dtype=numpy.int64)
	_encode_tiles(tile, numpy.array([0]), numpy.array([0]), numpy.array([1]),
		numpy.array([0]), numpy.array([1024]), values, True, buffer, False, classes, True, False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert kinds[0] == 2 | (7 << 3)
	assert out[:8].view(numpy.uint32).tolist() == [4, 5]
	# The differences 0, 127 and 1 take a byte each, and 895 two.
	assert out[8:14].tolist() == [0, 127, 1, 0x80 | (895 & 127), 895 >> 7, 0]
	assert bounds[1] == 16 and stats[0].tolist() == [4, 4, 0, 1, 0]


def test_encode_tiles_runs_and_float64():
	"""A tile of one value is stored as runs: the changes of value, as
	varints, then the value from each change on. An integer above 2**24
	next to a fraction makes the chunk float64."""

	tile = 64
	values = numpy.zeros((1, 2, 64))
	values[0, 0, :] = 3                 # one run
	values[0, 1, 10:20] = 2.5           # a run, back to 0, then a run
	values[0, 1, 40:50] = 2.5
	buffer = numpy.zeros((2, tile))
	out = numpy.zeros(4000, dtype=numpy.uint8)
	bounds = numpy.zeros(2, dtype=numpy.int64)
	kinds = numpy.zeros(1, dtype=numpy.uint8)
	stats = numpy.zeros((2, 5))
	classes = numpy.zeros(2, dtype=numpy.int64)
	_encode_tiles(tile, numpy.array([0]), numpy.array([0]), numpy.array([1]),
		numpy.array([0]), numpy.array([64]), values, True, buffer, False, classes, True, False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert kinds[0] == 4 | (6 << 3)
	# Five changes: the bytes of each track's varints, the varints (0; 10,
	# 10, 20, 10), padding to 4, five float32 values.
	assert out[:12].view(numpy.uint32).tolist() == [5, 1, 4]
	assert out[12:17].tolist() == [0, 10, 10, 20, 10]
	assert out[20:40].view(numpy.float32).tolist() == [3, 2.5, 0, 2.5, 0]
	assert bounds[1] == 40 and stats[:, 1].tolist() == [64, 20]

	values[0, 0, 5] = 16777217
	kinds[:] = 0
	_encode_tiles(tile, numpy.array([0]), numpy.array([0]), numpy.array([1]),
		numpy.array([0]), numpy.array([64]), values, True, buffer, False, classes, True, False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert kinds[0] == 4 | (8 << 3)
	assert out[20:28].view(numpy.float64).tolist() == [3]
	assert out[28:36].view(numpy.float64).tolist() == [16777217]

	# A float32 track keeps float32's rounding of the same value.
	classes[0] = 1
	kinds[:] = 0
	_encode_tiles(tile, numpy.array([0]), numpy.array([0]), numpy.array([1]),
		numpy.array([0]), numpy.array([64]), values, True, buffer, False, classes, True, False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert kinds[0] == 4 | (6 << 3)
	assert out[24:28].view(numpy.float32).tolist() == [16777216]


def test_encode_tiles_keeps_an_open_tile():
	tile = 16
	values = numpy.ones((1, 1, 24))
	buffer = numpy.zeros((1, tile))
	out = numpy.zeros(400, dtype=numpy.uint8)
	bounds = numpy.zeros(3, dtype=numpy.int64)
	kinds = numpy.zeros(2, dtype=numpy.uint8)
	stats = numpy.zeros((1, 5))
	classes = numpy.zeros(1, dtype=numpy.int64)
	n = _encode_tiles(tile, numpy.array([0, 1]), numpy.array([0, 0]),
		numpy.array([1, 1]), numpy.array([0]), numpy.array([24]), values, True,
		buffer, True, classes, True, False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	# A tile of ones is one run: its change at 0, to 1, as uint8.
	assert n == 1 and kinds[0] == 4 and buffer[0, :8].sum() == 8
	assert out[:8].view(numpy.uint32).tolist() == [1, 1]
	assert out[8:13].tolist() == [0, 0, 0, 0, 1]
	assert buffer[0, 8:].sum() == 0 and stats[0, 1] == 16

	# The open tile is then encoded by scanning the buffer, with the values
	# a later call adds: two runs.
	more = numpy.full((1, 1, 8), 2.0)
	n = _encode_tiles(tile, numpy.array([1]), numpy.array([0]), numpy.array([1]),
		numpy.array([24]), numpy.array([32]), more, True, buffer, False,
		classes, True, False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert n == 1 and kinds[0] == 4 and not buffer.any()
	assert out[:8].view(numpy.uint32).tolist() == [2, 2]
	assert out[8:14].tolist() == [0, 8, 0, 0, 1, 2]
	assert stats[0, 1] == 32 and stats[0, 0] == 40


def test_check_chunks_rejects_short_and_overrunning_chunks():
	tile, n_tracks = 16, 1
	data = numpy.zeros(64, dtype=numpy.uint8)
	failed = numpy.zeros(4, dtype=numpy.int64)
	positions = numpy.array([0, 0, 0, 0], dtype=numpy.int64)
	# dense uint8 of 16 bytes: a length of 16 passes, 15 fails; an unknown
	# kind fails; a sparse chunk whose one offset is past the tile fails.
	data[:8].view(numpy.uint32)[:] = [1, 1]
	data[8] = 16
	slots = numpy.array([0])
	kinds = numpy.array([1, 1, 3, 2], dtype=numpy.uint8)
	lengths = numpy.array([16, 15, 16, 16], dtype=numpy.int64)
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 3
	assert failed.tolist() == [0, 1, 1, 1]

	# An offset of 15 passes; a varint that runs past the track's bytes, or
	# bytes the varints do not fill, fail. The offsets of a track that is
	# not read are only counted: an offset past the tile passes there, and
	# a varint that does not end is one short of the total.
	data[8] = 15
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 2 and failed[3] == 0
	data[8] = 128 | 15
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 3
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		numpy.array([-1]), failed) == 3
	data[8] = 16
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		numpy.array([-1]), failed) == 2
	data[8] = 15
	data[4:8].view(numpy.uint32)[0] = 2
	failed[:] = 0
	assert _check_chunks(tile, data, positions, lengths, kinds, n_tracks,
		slots, failed) == 3


def test_read_tiles_zero_chunk_writes_zeros():
	out = numpy.full((1, 2, 10), 9, dtype=numpy.int32)
	_read_tiles(16, numpy.zeros(8, dtype=numpy.uint8), numpy.zeros(0,
		dtype=numpy.int64), numpy.zeros(0, dtype=numpy.uint8), 2,
		numpy.array([1, -1]), numpy.array([-1]), numpy.array([0]),
		numpy.array([4]), numpy.array([0]), out, False, numpy.zeros((0, 0)), 1,
		numpy.zeros(0, dtype=numpy.int64), 0)
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


@pytest.mark.parametrize('ascending', [True, False])
def test_sparse_read_sorts_on_threads(fw, read_batching, monkeypatch, ascending):
	"""A sparse read's entries, sorted from its tasks' parts by a counting
	sort cut among threads, or by keys where a bucket's offsets would not
	ascend, are those one thread gives, in the same type."""

	reader = FigWigReader(fw)
	rng = numpy.random.default_rng(22)
	chroms, starts = random_windows(rng, 300, 150)
	picks = (None, ['negative', 'atac_plus', 'signal'])
	ones = [reader.read(chroms, starts, 150, tracks=tracks, sparse=True,
		n_jobs=1) for tracks in picks]
	monkeypatch.setattr(figwig.figwig, '_GATHER_PER_THREAD', 1)
	monkeypatch.setattr(figwig.figwig, '_GATHER_ROW_BYTES', 1 << 20)
	if not ascending:
		monkeypatch.setattr(figwig.figwig, '_bucket_ascending', lambda *a: False)
	for tracks, one in zip(picks, ones):
		several = reader.read(chroms, starts, 150, tracks=tracks, sparse=True,
			n_jobs=4)
		for a, b in zip((one.rows, one.channels, one.offsets, one.values), (
				several.rows, several.channels, several.offsets, several.values)):
			assert a.dtype == b.dtype
			numpy.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('n_threads', [1, 4])
def test_sparse_gather_sorts_offsets_that_do_not_ascend(fw, monkeypatch,
	n_threads):
	"""Parts whose offsets ascend within each (row, channel) are sorted by
	the counting sort alone; parts whose offsets do not are still sorted by
	row, channel and offset."""

	class Task:
		def __init__(self, rows, channels, offsets, values):
			self.entries = (numpy.array(rows), numpy.array(channels),
				numpy.array(offsets), numpy.array(values, dtype=numpy.float32))

	monkeypatch.setattr(figwig.figwig, '_GATHER_PER_THREAD', 1)
	monkeypatch.setattr(figwig.figwig, '_GATHER_ROW_BYTES', 1 << 20)
	reader = FigWigReader(fw)
	ascending = [([0, 0, 1], [0, 1, 0], [1, 4, 2], [1, 2, 3]), ([0, 1, 1], [0,
		0, 1], [5, 3, 0], [4, 5, 6])]
	descending = [([0, 1], [0, 1], [8, 2], [1, 2]), ([0, 1], [0, 1], [3, 1],
		[3, 4])]
	# The one pair out of order straddles the threads' halves of the check.
	straddling = [([0, 0], [0, 1], [5, 7], [1, 2]), ([0, 1], [1, 1], [3, 1],
		[3, 4])]
	for parts in (ascending, descending, straddling):
		found = reader._gather([Task(*part) for part in parts], [], 2, 2, 10,
			numpy.dtype('float32'), n_threads)
		rows, channels, offsets, values = (numpy.concatenate([part[k] for part in
			parts]) for k in range(4))
		order = numpy.lexsort((offsets, channels, rows))
		for a, b in zip((found.rows, found.channels, found.offsets, found.values),
				(rows[order], channels[order], offsets[order], numpy.array(values,
				dtype=numpy.float32)[order])):
			numpy.testing.assert_array_equal(a, b)
		assert found.values.dtype == numpy.float32 and found.shape == (2, 2, 10)


@pytest.mark.parametrize('n_windows', [2, 50, 1000])
def test_sparse_gather_counts_are_bounded(fw, monkeypatch, n_windows):
	"""The counting sort's runs, a row of buckets each, take no more than
	one row and a byte per entry, however many threads there are, and sort
	as one thread does."""

	shapes = []
	starts_of = figwig.figwig._bucket_starts

	def spy(counts):
		shapes.append(counts.shape)
		return starts_of(counts)

	class Task:
		def __init__(self, rows, offsets):
			self.entries = (rows, numpy.zeros(len(rows), dtype=numpy.int64),
				offsets, numpy.ones(len(rows), dtype=numpy.float32))

	monkeypatch.setattr(figwig.figwig, '_GATHER_PER_THREAD', 1)
	monkeypatch.setattr(figwig.figwig, '_bucket_starts', spy)
	reader = FigWigReader(fw)
	rng = numpy.random.default_rng(3)
	rows = numpy.sort(rng.integers(0, n_windows, 4096))

	def tasks():
		return [Task(rows[a:a + 64], numpy.arange(a, a + 64)) for a in range(0,
			4096, 64)]

	n_buckets = n_windows * 4
	one = reader._gather(tasks(), [], n_windows, 4, 100, numpy.dtype('float32'), 1)
	found = reader._gather(tasks(), [], n_windows, 4, 100, numpy.dtype('float32'),
		8)
	runs, buckets = shapes[-1]
	assert buckets == n_buckets and (runs - 1) * n_buckets * 8 <= 4096
	assert runs == {2: 8, 50: 3, 1000: 1}[n_windows]
	for a, b in zip((one.rows, one.channels, one.offsets, one.values), (
			found.rows, found.channels, found.offsets, found.values)):
		numpy.testing.assert_array_equal(a, b)


def test_batch_cuts_are_the_planners_loop():
	"""The kernel that cuts a read's chunks into batches gives the batches
	and bytes of the loop the planner ran before, binned or not, and finds
	a window cut inside a batch where the loop did."""

	from figwig._figwig_kernels import _batch_cuts

	def loop(bounds, live, cap, most, continues, binned):
		firsts, totals, cut_inside, a = [0], [], False, 0
		while a < len(bounds):
			total_bytes, n_live, b = 0, 0, a
			while b < len(bounds) and (b == a or (total_bytes + bounds[b] <= cap
					and n_live < most)):
				total_bytes += int(bounds[b])
				n_live += int(live[b])
				b += 1
			if binned:
				while b < len(bounds) and continues[b - 1]:
					if total_bytes + bounds[b] > 2 * cap or n_live >= 2 * most:
						cut_inside = True
						break
					total_bytes += int(bounds[b])
					n_live += int(live[b])
					b += 1
			totals.append(total_bytes)
			firsts.append(b)
			a = b
		return firsts, totals, cut_inside

	rng = numpy.random.default_rng(7)
	seen = set()
	for trial in range(300):
		n = int(rng.integers(0, 60))
		bounds = rng.integers(8, 400, n).astype(numpy.int64)
		live = rng.integers(0, 2, n).astype(numpy.int64)
		continues = rng.random(n) < 0.6
		cap, most = int(rng.integers(8, 1200)), int(rng.integers(1, 9))
		for binned in (False, True):
			firsts, totals, cut_inside = _batch_cuts(bounds, live, cap, most,
				continues if binned else numpy.zeros(0, dtype=bool), binned)
			expected = loop(bounds, live, cap, most, continues, binned)
			assert (firsts.tolist(), totals.tolist(), bool(cut_inside)) == expected
			seen.add(bool(cut_inside))
	assert seen == {False, True}


def test_labels_sum_tracks(fw, ref, read_batching):
	rng = numpy.random.default_rng(21)
	chroms, starts = random_windows(rng, 100, 150)
	reader = FigWigReader(fw)
	labels = numpy.array([0, 0, 1, -1, 1])
	y = reader.read(chroms, starts, 150, labels=labels)
	assert y.shape == (100, 2, 150) and y.dtype == numpy.float64
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
	with pytest.raises(ValueError, match="sparse cannot be given"):
		reader.read(chroms, starts, 150, labels=labels, sparse=True)
	with pytest.raises(ValueError, match="-1 or from 0"):
		reader.read(chroms, starts, 150, labels=[0, 0, 0, -2, 0])
	with pytest.raises(ValueError, match="each track is read once"):
		reader.read(chroms, starts, 150, tracks=[0, 0], labels=[0, 0])
	with pytest.raises(ValueError, match="out cannot be given"):
		reader.read(chroms, starts, 150, sparse=True, out=numpy.zeros(1))


def test_dense_guard(fw, track_fw, monkeypatch):
	reader = FigWigReader(fw)
	monkeypatch.setattr(figwig.figwig, '_MAX_DENSE_BYTES', 1000)
	with pytest.raises(ValueError, match="sparse=True or labels="):
		reader.read('chr1', [0], 1000)
	out = numpy.empty((1, 5, 1000), dtype=numpy.float32)
	assert reader.read('chr1', [0], 1000, out=out) is out
	assert reader.read('chr1', [0], 1000, sparse=True).shape == (1, 5, 1000)

	# A summed read in track order counts its copies per thread.
	monkeypatch.setattr(figwig.figwig, '_MAX_DENSE_BYTES', 1000)
	with pytest.raises(ValueError, match="the dense output would take"):
		FigWigReader(track_fw).read('chr1', [1], 4992, bins=3, dtype='float64',
			n_jobs=4)


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
	# The NaN bases are entries, but no nonzero bases of the statistics.
	assert len(everything.values) == int(reader.stats['nonzero'].sum()) + int(
		numpy.isnan(everything.values).sum())
	with pytest.raises(ValueError, match="not one of the file's chromosomes"):
		reader.read_entries(chroms=['chr9'])


@pytest.mark.parametrize('order', ['positions', 'tracks'])
@pytest.mark.parametrize('per_read', [1, 2])
def test_read_entries_in_several_reads(fw, track_fw, monkeypatch, order,
	per_read):
	"""Chromosomes whose sort keys would pass the limit together are read a
	few at a time, and give the entries one read of them all gives."""

	reader = FigWigReader(fw if order == 'positions' else track_fw)
	tracks = ['atac_plus', 'signal', 'negative']
	one = reader.read_entries(tracks=tracks, n_jobs=2)
	widest = max(reader.chrom_sizes.values())
	monkeypatch.setattr(figwig.figwig, '_SPARSE_KEYS', per_read * len(tracks) *
		widest)
	calls = []
	read = FigWigReader.read

	def counted(self, chroms, *args, **kwargs):
		calls.append(len(chroms))
		return read(self, chroms, *args, **kwargs)

	monkeypatch.setattr(FigWigReader, 'read', counted)
	several = reader.read_entries(tracks=tracks, n_jobs=2)
	n_chroms = len(reader.chrom_sizes)
	assert calls == [min(per_read, n_chroms - a) for a in range(0, n_chroms,
		per_read)]
	for a, b in zip((one.chroms, one.positions, one.tracks, one.values), (
			several.chroms, several.positions, several.tracks, several.values)):
		assert a.dtype == b.dtype
		numpy.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_read_entries_plans_within_the_longest_chromosome(fw, track_fw,
	monkeypatch, order):
	"""With no budget of its own, read_entries reads chromosomes together
	only while their tiles stay within the longest chromosome's, and gives
	the entries one read gives."""

	reader = FigWigReader(fw if order == 'positions' else track_fw)
	tracks = ['atac_plus', 'signal', 'negative']
	one = reader.read_entries(tracks=tracks, n_jobs=2)
	monkeypatch.setattr(figwig.figwig, '_ENTRIES_PAIRS', 0)
	calls = []
	read = FigWigReader.read

	def counted(self, chroms, *args, **kwargs):
		calls.append(len(chroms))
		return read(self, chroms, *args, **kwargs)

	monkeypatch.setattr(FigWigReader, 'read', counted)
	several = reader.read_entries(tracks=tracks, n_jobs=2)
	tiles = [-(-size // reader.tile) for size in reader.chrom_sizes.values()]
	expected, total = [0], 0
	for n in tiles:
		if expected[-1] and total + n > max(tiles):
			expected.append(0)
			total = 0
		expected[-1] += 1
		total += n
	assert calls == expected and len(expected) > 1
	for a, b in zip((one.chroms, one.positions, one.tracks, one.values), (
			several.chroms, several.positions, several.tracks, several.values)):
		assert a.dtype == b.dtype
		numpy.testing.assert_array_equal(a, b)


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
	kinds, sizes, raws, offsets, _ = reader._group_index(0)
	counts = numpy.bincount(kinds & 7, minlength=5)
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
	kinds, _, _, _, _ = old._group_index(0)
	assert numpy.bincount(kinds & 7, minlength=5)[3] == 0
	assert os.path.getsize(path) > os.path.getsize(paths['positions', 120])
	y = reader.read('chr1', [0], 5000, dtype='float64')
	assert_identical(old.read('chr1', [0], 5000, dtype='float64'), y)
	kinds, _, _, _, _ = FigWigReader(paths['positions', 8])._group_index(0)
	assert numpy.bincount(kinds & 7, minlength=5)[3] == 0


@pytest.mark.parametrize('level', [6, 0])
def test_write_sparse_writes_what_write_writes(tmp_path, level):
	"""A tile written as entries is stored as the kind `write` picks for
	the rows they fill, so the two give every tile the same chunk, though
	they may lay a call's groups out in another order: depth-like runs,
	float32 noise, ones, negative runs, float64 values, scattered counts and
	a track with none, in groups where a track is absent from tiles, on
	chromosomes that end inside a tile. Runs broken by NaN read the same and
	take fewer decoded bytes, a stretch of NaN being one change of value;
	`write` counts each NaN base as a change when it surveys a row, so it
	stores such a tile sparse."""

	rng = numpy.random.default_rng(37)
	sizes = {'a': 3000, 'b': 1000, 'c': 70}
	tracks = [{'name': 'depth'}, {'name': 'noise', 'dtype': 'float32'},
		{'name': 'ones'}, {'name': 'negative'}, {'name': 'wide', 'dtype':
		'float64'}, {'name': 'scattered'}, {'name': 'empty'}]
	groups = [['depth', 'empty'], ['noise'], ['scattered'], ['ones'], ['negative',
		'wide']]
	rows, nan_rows = {}, {}
	for chrom, length in sizes.items():
		depth = numpy.cumsum(rng.integers(-1, 2, length)).clip(0, None)
		values = numpy.zeros((7, length))
		values[0] = depth
		values[1] = rng.random(length).astype(numpy.float32)
		values[1, length // 3:length // 2] = 0
		values[2] = rng.random(length) < 0.7
		values[3] = -depth
		values[3, :length // 2] = 0
		values[4] = numpy.where(depth > 0, depth * 2.0 ** 30 + 0.25, 0)
		values[5] = (rng.random(length) < 0.02) * rng.integers(1, 9, length)
		rows[chrom] = values
		nan_rows[chrom] = depth[None].astype(numpy.float64)
		nan_rows[chrom][0, 10:40] = numpy.nan
		nan_rows[chrom][0, length // 2] = numpy.nan

	def write_both(name, tracks, groups, rows):
		paths = str(tmp_path / (name + '_dense.fw')), str(tmp_path / (name +
			'_entries.fw'))
		with FigWigWriter(paths[0], sizes, tracks, tile=64, groups=groups,
				level=level) as writer:
			for chrom, values in rows.items():
				writer.write(chrom, [0], values[None])
		with FigWigWriter(paths[1], sizes, tracks, tile=64, groups=groups,
				level=level) as writer:
			for chrom, values in rows.items():
				which, positions = numpy.nonzero(values)
				order = rng.permutation(len(which))
				writer.write_sparse(chrom, positions[order], which[order], values[
					which, positions][order])
		reader = FigWigReader(paths[1])
		for chrom, values in rows.items():
			assert_identical(reader.read(chrom, [0], sizes[chrom], dtype='float64')[
				0], values)
		return paths, reader

	def chunks(path, group):
		kinds, sizes, _, offsets, _ = FigWigReader(path)._group_index(group)
		with open(path, 'rb') as handle:
			data = handle.read()
		return kinds.tolist(), [data[o:o + n] for o, n in zip(offsets.tolist(),
			sizes.tolist())]

	(dense, entries), reader = write_both('all', tracks, groups, rows)
	for g in range(len(groups)):
		assert chunks(dense, g) == chunks(entries, g)
	kinds = numpy.concatenate([reader._group_index(g)[0] for g in range(len(
		groups))])
	counts = numpy.bincount(kinds & 7, minlength=5)
	assert counts[1] > 0 and counts[2] > 0 and counts[4] > 0

	(dense, entries), reader = write_both('nan', [{'name': 'nan', 'dtype':
		'float32'}], 1, nan_rows)
	assert FigWigReader(entries)._group_index(0)[2].sum() < FigWigReader(
		dense)._group_index(0)[2].sum()
	assert (reader._group_index(0)[0] & 7 == 4).any()


def test_entry_runs_are_the_changes_of_the_rows():
	"""The changes of value found from a tile's entries are those
	`_collect_runs` finds in the rows they fill, and take the bytes
	`_run_bytes` counts: runs, gaps far enough apart to need two- and
	three-byte offsets, NaN stretches, a run to the tile's end and a track
	with no entries."""

	kernels = figwig._figwig_kernels
	rng = numpy.random.default_rng(41)
	tile = 65536
	rows = numpy.zeros((5, tile))
	rows[0] = numpy.repeat(rng.integers(0, 3, tile // 16), 16)
	rows[1, [3, 4, 5, 300, 301, 40000]] = [2, 2, 7, 1, 1, 9]
	rows[2] = rows[0]
	rows[2, 100:140] = numpy.nan
	rows[2, 141] = numpy.nan
	rows[3, tile - 50:] = 4
	which, offsets = numpy.nonzero(rows)
	touched = numpy.unique(which).astype(numpy.int64)
	nnz = numpy.bincount(which, minlength=5).astype(numpy.int64)
	offs = offsets.astype(numpy.int64)
	vals = rows[which, offsets]

	npts = numpy.zeros(5, dtype=numpy.int64)
	roffs = numpy.empty(2 * len(offs), dtype=numpy.int64)
	rvals = numpy.empty(2 * len(offs), dtype=numpy.float64)
	n_points, run_bytes = kernels._entry_runs(tile, touched, len(touched), nnz,
		offs, vals, npts, roffs, rvals)

	expected_offs = numpy.empty(5 * tile, dtype=numpy.int64)
	expected_vals = numpy.empty(5 * tile, dtype=numpy.float64)
	d, expected_bytes = 0, 0
	for g in touched:
		n = kernels._collect_runs(rows[g], tile, 0, expected_offs, expected_vals, d)
		assert npts[g] == n
		d += n
		expected_bytes += kernels._run_bytes(rows[g], tile, 0)
	assert n_points == d and run_bytes == expected_bytes
	numpy.testing.assert_array_equal(roffs[:d], expected_offs[:d])
	assert_identical(rvals[:d], expected_vals[:d])


@pytest.mark.parametrize('n_chunks', [1, 3, 8])
def test_radix_order_is_a_stable_argsort(n_chunks):
	"""The radix sort gives numpy's stable argsort and the sorted keys, for
	keys of any width with many ties, whatever the number of chunks."""

	from concurrent.futures import ThreadPoolExecutor
	from figwig.figwig import _radix_order

	rng = numpy.random.default_rng(n_chunks)
	with ThreadPoolExecutor(4) as pool:
		for bits in (1, 5, 16, 17, 31, 40, 63):
			for n in (0, 1, 7, 5000):
				keys = rng.integers(0, 1 << min(bits, 62), n, dtype=numpy.int64)
				if bits == 63:
					keys |= rng.integers(0, 2, n, dtype=numpy.int64) << 62
				keys[::3] = keys[0] if n else 0
				expected = numpy.argsort(keys, kind='stable')
				order, sorted_keys = _radix_order(keys.copy(), bits, pool if
					n_chunks > 1 else None, n_chunks)
				numpy.testing.assert_array_equal(order, expected)
				numpy.testing.assert_array_equal(sorted_keys, keys[expected])
		order, _ = _radix_order(numpy.full(100, 5), 4, None, 1)
		numpy.testing.assert_array_equal(order, numpy.arange(100))


@pytest.mark.parametrize('n_jobs', [1, 4])
def test_write_sparse_radix_sort_writes_what_the_numpy_sort_writes(tmp_path,
	monkeypatch, n_jobs):
	"""Entries sorted by the radix sort, in one chunk or several, give the
	file the numpy sort of the composite keys gives, byte for byte, over
	several chromosomes and groups in one call, values of every kind, and
	the same message for two entries on one base."""

	monkeypatch.setattr(figwig.figwig, '_RADIX_PARALLEL', 1)
	# The radix sort, whatever order the keys come in.
	monkeypatch.setattr(figwig.figwig, '_MERGE_RUNS', -1)
	monkeypatch.setattr(figwig.figwig, '_MERGE_REACH', 1)
	monkeypatch.setattr(figwig.figwig, '_MERGE_FAR', 10 ** 9)
	original = FigWigWriter._radix_entries
	radix, used = [True], []

	def chosen(self, *args):
		result = original(self, *args) if radix[0] else None
		used.append(result is not None)
		return result

	monkeypatch.setattr(FigWigWriter, '_radix_entries', chosen)
	rng = numpy.random.default_rng(5)
	tracks = ['t{}'.format(i) for i in range(40)]
	groups = [tracks[:1], tracks[1:9], tracks[9:30], tracks[30:]]
	n = 6000
	chroms = rng.choice(['chr1', 'chr2', 'chrM'], n)
	sizes = numpy.array([CHROM_SIZES[c] for c in chroms])
	positions = (rng.random(n) * sizes).astype(numpy.int64)
	which = rng.integers(0, 40, n)
	key = numpy.unique(numpy.stack([numpy.searchsorted(['chr1', 'chr2', 'chrM'],
		chroms), which, positions]), axis=1, return_index=True)[1]
	chroms, positions, which = chroms[key], positions[key], which[key]
	values = {'int32': rng.integers(1, 50, len(key)).astype(numpy.int32),
		'float16': rng.random(len(key)).astype(numpy.float16) + 1,
		'>f8': (rng.random(len(key)) - 0.5).astype('>f8'),
		'longdouble': (rng.random(len(key)) * 3).astype(numpy.longdouble),
		'uint8': numpy.ones(len(key), dtype=numpy.uint8)}

	def write(use_radix, vals, names=None):
		radix[0] = use_radix
		path = str(tmp_path / '{}.fw'.format(use_radix))
		with FigWigWriter(path, CHROM_SIZES, tracks, tile=64, groups=groups,
				n_jobs=n_jobs) as writer:
			if names is None:
				writer.write_sparse(chroms, positions, which, vals)
			else:
				for chrom in ['chr1', 'chr2', 'chrM']:
					take = chroms == chrom
					writer.write_sparse(chrom, positions[take], names[take], vals[
						take])
		with open(path, 'rb') as handle:
			return handle.read()

	for kind, vals in values.items():
		assert write(True, vals) == write(False, vals), kind
	names = numpy.array(tracks)[which]
	assert write(True, values['int32'], names) == write(False, values['int32'],
		names)

	messages = []
	for use_radix in (True, False):
		radix[0] = use_radix
		with FigWigWriter(str(tmp_path / 'c.fw'), CHROM_SIZES, tracks, tile=64,
				groups=groups, n_jobs=n_jobs) as writer:
			with pytest.raises(ValueError, match='both name base') as error:
				writer.write_sparse(numpy.append(chroms, chroms[7]), numpy.append(
					positions, positions[7]), numpy.append(which, which[7]),
					numpy.ones(len(key) + 1))
			messages.append(str(error.value))
	assert messages[0] == messages[1]
	assert used.count(True) == len(values) + 3 + 1


@pytest.mark.parametrize('n_jobs', [1, 3])
def test_write_sparse_sorts_by_either_sort_alike(tmp_path, monkeypatch, n_jobs):
	"""Keys sorted by numpy's merge sort or by the radix sort give the same
	file, byte for byte; on one thread, keys out of order over the whole
	call go to the radix sort, and keys out of order only near each other,
	or in a few runs, to the merge sort."""

	monkeypatch.setattr(figwig.figwig, '_RADIX_PARALLEL', 1)
	rng = numpy.random.default_rng(12)
	tracks = ['t{}'.format(i) for i in range(64)]
	groups = [tracks[i:i + 4] for i in range(0, 64, 4)]

	def in_position_order(positions, which):
		order = numpy.lexsort((which, positions))
		return positions[order], which[order]

	# Each track over its own stretch, as reads are, in position order: keys
	# out of order only near each other.
	stretches = [numpy.sort(rng.choice(300, 150, replace=False)) + t * 70 for t in
		range(64)]
	local = in_position_order(numpy.concatenate(stretches), numpy.repeat(
		numpy.arange(64), 150))
	# Every track over the whole chromosome, in position order: each tile's
	# keys go back to the first group's.
	key = numpy.unique(rng.integers(0, 64 * CHROM_SIZES['chr1'], 20000))
	far = in_position_order(key // 64, key % 64)
	# Keys in order, in two runs.
	p, t = far
	order = numpy.lexsort((p % 64, t % 4, p // 64, t // 4))
	half = len(order) // 2
	runs = (numpy.concatenate([p[order][half:], p[order][:half]]),
		numpy.concatenate([t[order][half:], t[order][:half]]))
	inputs = {'local': local, 'far': far, 'runs': runs}
	radix_runs = []
	original = figwig.figwig._radix_order

	def spy(*args):
		radix_runs.append(True)
		return original(*args)

	monkeypatch.setattr(figwig.figwig, '_radix_order', spy)

	def write(name, p, t, **constants):
		with monkeypatch.context() as m:
			for constant, value in constants.items():
				m.setattr(figwig.figwig, constant, value)
			path = str(tmp_path / '{}.fw'.format(name))
			with FigWigWriter(path, CHROM_SIZES, tracks, tile=64, groups=groups,
					n_jobs=n_jobs) as writer:
				writer.write_sparse('chr1', p, t, numpy.arange(1, len(p) + 1))
		with open(path, 'rb') as handle:
			return handle.read()

	for name, (p, t) in inputs.items():
		del radix_runs[:]
		chosen = write(name, p, t)
		took_radix = len(radix_runs) > 0
		merged = write(name + '_merge', p, t, _MERGE_RUNS=10 ** 9)
		radix = write(name + '_radix', p, t, _MERGE_RUNS=-1, _MERGE_REACH=1,
			_MERGE_FAR=10 ** 9)
		assert chosen == merged == radix, name
		if n_jobs == 1:
			assert took_radix == (name == 'far'), name
		else:
			assert took_radix == (name != 'runs'), name


def test_write_sparse_packs_group_batches_into_tasks(tmp_path, monkeypatch):
	"""A call's group batches written inline, packed several to a pool task
	or one to a task give the same file, byte for byte, over calls that
	continue a group's runs of the file and calls that do not."""

	rng = numpy.random.default_rng(9)
	tracks = ['t{}'.format(i) for i in range(30)]
	groups = [tracks[i:i + 3] for i in range(0, 30, 3)]
	calls = []
	for chrom in ('chr1', 'chr2'):
		for lo, hi in ((0, 1600), (1600, 2432), (2432, CHROM_SIZES[chrom])):
			key = numpy.unique(rng.integers(0, 30 * (hi - lo), 900))
			calls.append((chrom, lo + key % (hi - lo), key // (hi - lo), rng.integers(
				1, 9, len(key)).astype(numpy.int32)))

	def write(name, n_jobs, per_thread):
		monkeypatch.setattr(figwig.figwig, '_BATCHES_PER_THREAD', per_thread)
		path = str(tmp_path / name)
		with FigWigWriter(path, CHROM_SIZES, tracks, tile=64, groups=layout,
				n_jobs=n_jobs) as writer:
			for chrom, positions, which, values in calls:
				order = numpy.argsort(positions, kind='stable')
				writer.write_sparse(chrom, positions[order], which[order], values[
					order])
		with open(path, 'rb') as handle:
			return handle.read()

	# Ten groups, whose batches interleave in the file, and one group cut
	# into batches of a few tiles, whose chunks follow one another.
	for layout, per_batch in ((groups, 1 << 20), ([tracks], 64)):
		monkeypatch.setattr(figwig.figwig, '_ENTRIES_PER_BATCH', per_batch)
		monkeypatch.setattr(figwig.figwig, '_BATCH_BYTES', 1 << 25 if per_batch >
			64 else 8 * per_batch)
		inline = write('inline.fw', 1, 8)
		assert write('packed.fw', 3, 1) == inline
		assert write('single.fw', 3, 1 << 40) == inline
		reader = FigWigReader(str(tmp_path / 'packed.fw'))
		for chrom, positions, which, values in calls:
			y = reader.read([chrom] * len(positions), positions, 1, dtype='int64')
			numpy.testing.assert_array_equal(y[numpy.arange(len(positions)), which,
				0], values)


def test_track_order_packs_block_flushes_into_tasks(tmp_path, monkeypatch):
	"""Tracks' blocks in track order written inline, packed several to a
	pool task or one to a task give the same file, byte for byte, with
	blocks cut where a track's entries pass a block and where its
	chromosome changes."""

	monkeypatch.setattr(figwig.figwig, '_BLOCK_ENTRIES', 40)
	rng = numpy.random.default_rng(10)
	tracks = ['c{}'.format(i) for i in range(25)]
	calls = []
	for chrom in ('chr1', 'chr2', 'chrM'):
		key = numpy.unique(rng.integers(0, 25 * CHROM_SIZES[chrom], 1500))
		positions, which = key % CHROM_SIZES[chrom], key // CHROM_SIZES[chrom]
		order = numpy.argsort(positions, kind='stable')
		cut = len(order) // 2
		for part in (order[:cut], order[cut:]):
			calls.append((chrom, positions[part], which[part], rng.integers(1, 9,
				len(part)).astype(numpy.int32)))

	def write(name, n_jobs, per_thread):
		monkeypatch.setattr(figwig.figwig, '_BATCHES_PER_THREAD', per_thread)
		path = str(tmp_path / name)
		with FigWigWriter(path, CHROM_SIZES, tracks, order='tracks', level=6,
				n_jobs=n_jobs) as writer:
			for chrom, positions, which, values in calls:
				writer.write_sparse(chrom, positions, which, values)
		with open(path, 'rb') as handle:
			return handle.read()

	inline = write('inline.fw', 1, 8)
	assert write('packed.fw', 3, 1) == inline
	assert write('single.fw', 3, 1 << 40) == inline
	entries = FigWigReader(str(tmp_path / 'packed.fw')).read_entries()
	assert len(entries.values) == sum(len(call[1]) for call in calls)


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


###
# The format's second round: groups in any order, exact integers, runs,
# stored chunks, sums per bin, and the writer's and reader's guards


def test_groups_in_any_order_keep_their_tracks(tmp_path):
	"""A group listed in another order than the file's tracks still reads
	each track under its own name."""

	path = str(tmp_path / 'g.fw')
	values = numpy.zeros((1, 3, 1000))
	values[0, 0, :10] = 5
	values[0, 1, 500:] = 7
	values[0, 2, 100] = 1
	write_figwig(path, {'chr1': 1000}, ['a', 'b', 'c'], 'chr1', [0], values,
		groups=[['c', 'b', 'a']], tile=64)
	reader = FigWigReader(path)
	assert [t['group'] for t in reader.tracks] == [0, 0, 0]
	for name, k in (('a', 0), ('b', 1), ('c', 2)):
		numpy.testing.assert_array_equal(reader.read('chr1', [0], 1000,
			tracks=name)[0], values[0, k])
	numpy.testing.assert_array_equal(reader.stats['sum'], [50, 3500, 1])


def test_integers_next_to_floats_stay_exact(tmp_path):
	"""A chunk that holds a fraction and an integer above 2**24 is stored
	as float64, so a uint32 track grouped with a float track keeps its
	integers; a float32 track keeps float32's rounding; integers beyond
	uint32 are inferred float64; and a value beyond float32 raises."""

	path = str(tmp_path / 'e.fw')
	tracks = [{'name': 'counts', 'dtype': 'uint32'}, {'name': 'fold',
		'dtype': 'float32'}, {'name': 'wide'}, {'name': 'exact', 'dtype':
		'float64'}]
	values = numpy.zeros((1, 4, 64))
	values[0, 0, 3] = 16777217
	values[0, 0, 4] = 3000000001
	values[0, 1, 3] = 16777217
	values[0, 1, 5] = 0.1
	values[0, 2, 7] = 4294967297
	values[0, 3, 8] = 0.1
	write_figwig(path, {'chr1': 64}, tracks, 'chr1', [0], values, groups=4,
		tile=64)
	reader = FigWigReader(path)
	assert [t['dtype'] for t in reader.tracks] == ['uint32', 'float32',
		'float64', 'float64']
	kinds, _, _, _, _ = reader._group_index(0)
	assert (kinds[0] >> 3) & 15 == 8
	y = reader.read('chr1', [0], 64, dtype='float64')[0]
	assert y[0, 3] == 16777217 and y[0, 4] == 3000000001
	assert y[1, 3] == 16777216 and y[1, 5] == numpy.float32(0.1)
	assert y[2, 7] == 4294967297 and y[3, 8] == 0.1
	assert reader.read('chr1', [3], 1, tracks='counts')[0, 0] == 16777217
	assert reader.stats['max'].tolist() == [3000000001, 16777216, 4294967297,
		0.1]
	found = reader.read('chr1', [0], 64, sparse=True)
	assert found.values.dtype == numpy.float64 and found.values[0] == 16777217
	entries = reader.read_entries()
	assert entries.values.dtype == numpy.float64 and 3000000001 in \
		entries.values

	# A float32 track with a fraction and a uint32 track in one chunk, with
	# nothing above 2**24 and no float64 value, stays float32.
	values[0, 0, 3:5] = 100
	values[0, 2, 7] = 0
	values[0, 3, 8] = 0
	write_figwig(path, {'chr1': 64}, tracks, 'chr1', [0], values, groups=4,
		tile=64)
	kinds, _, _, _, _ = FigWigReader(path)._group_index(0)
	assert (kinds[0] >> 3) & 15 == 6

	with pytest.raises(ValueError, match="beyond float32's range"):
		write_figwig(path, {'chr1': 64}, ['a'], 'chr1', [0], numpy.full((1, 1,
			64), 1e39), tile=64)
	write_figwig(path, {'chr1': 64}, [{'name': 'a', 'dtype': 'float64'}],
		'chr1', [0], numpy.full((1, 1, 64), 1e39), tile=64)
	assert FigWigReader(path).read('chr1', [0], 1)[0, 0] == 1e39

	# A value too small for float32 is 0, and not counted as nonzero.
	write_figwig(path, {'chr1': 64}, ['a'], 'chr1', [0], numpy.full((1, 1,
		64), 1e-50), tile=64)
	assert FigWigReader(path).stats['nonzero'][0] == 0


def test_negative_zero_reads_as_zero(tmp_path):
	path = str(tmp_path / 'z.fw')
	values = numpy.zeros((1, 1, 128))
	values[0, 0, :64] = -0.0
	values[0, 0, 64:] = numpy.arange(64) * 0.5
	values[0, 0, 70] = -0.0
	write_figwig(path, {'chr1': 128}, ['a'], 'chr1', [0], values, tile=64)
	y = FigWigReader(path).read('chr1', [0], 128)[0, 0]
	assert not numpy.signbit(y).any()


def test_runs_chunks_hold_binned_signal(tmp_path):
	"""Intervals of one value over many bases are stored as runs, read
	back exactly in windows and as entries, and take far less than a
	dense chunk."""

	path = str(tmp_path / 'r.fw')
	rng = numpy.random.default_rng(3)
	length = 50000
	starts = numpy.arange(0, length, 5000)
	ends = numpy.minimum(starts + 5000, length)
	values = rng.normal(size=(len(starts), 2)).astype(numpy.float32)
	values[3, 0] = 0
	with FigWigWriter(path, {'chr1': length}, ['a', 'b'], tile=4096) as writer:
		writer.write('chr1', starts, values, ends=ends)
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets, _ = reader._group_index(0)
	assert set((kinds & 7).tolist()) == {4} and (raws < 100).all()
	assert os.path.getsize(path) < 4000

	dense = numpy.repeat(values.T, 5000, axis=1)[:, :length]
	numpy.testing.assert_array_equal(reader.read('chr1', [0], length)[0],
		dense)
	windows = rng.integers(0, length, 300)
	y = reader.read('chr1', windows, 777, dtype='float32')
	for j, start in enumerate(windows):
		piece = dense[:, start:start + 777]
		numpy.testing.assert_array_equal(y[j, :, :piece.shape[1]], piece)
		assert numpy.isnan(y[j, :, piece.shape[1]:]).all()
	found = reader.read('chr1', windows, 777, sparse=True)
	assert_identical(sparse_to_dense(found, 'float32'), numpy.nan_to_num(y))
	entries = reader.read_entries()
	assert len(entries.values) == int((dense != 0).sum())
	y = reader.read('chr1', windows, 777, labels=[0, 0])
	numpy.testing.assert_allclose(numpy.nan_to_num(y[:, 0]), numpy.nan_to_num(
		reader.read('chr1', windows, 777, dtype='float64').sum(axis=1)),
		rtol=1e-6)


def test_small_chunks_are_stored_raw(tmp_path):
	"""A chunk that deflate would grow is stored as it is, which the kind
	byte marks, and the file is smaller than with every chunk deflated."""

	path = str(tmp_path / 's.fw')
	values = numpy.zeros((1, 1, 64 * 40))
	values[0, 0, ::64] = 1
	values[0, 0, 1::64] = 2
	write_figwig(path, {'chr1': 64 * 40}, ['a'], 'chr1', [0], values, tile=64)
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets, _ = reader._group_index(0)
	stored = (kinds & 128) > 0
	assert stored.all() and (sizes == raws).all()
	numpy.testing.assert_array_equal(reader.read('chr1', [0], 64 * 40)[0, 0],
		values[0, 0])
	level0 = str(tmp_path / 'l0.fw')
	write_figwig(level0, {'chr1': 64 * 40}, ['a'], 'chr1', [0], values, tile=64,
		level=0)
	# The chunks take the same bytes either way; the index's kind bytes
	# differ, which deflate may put in a byte or two.
	assert abs(os.path.getsize(path) - os.path.getsize(level0)) <= 8


def test_matrix_chunks_in_compressed_files(tmp_path):
	"""With thousands of tracks of which a few have a value in a tile, a
	compressed file stores the tile as a matrix chunk, whose size follows
	the values rather than the tracks."""

	n_tracks = 3000
	path = str(tmp_path / 'm.fw')
	names = ['c{}'.format(i) for i in range(n_tracks)]
	rng = numpy.random.default_rng(5)
	positions = rng.choice(5000, 200, replace=False)
	cells = rng.integers(0, n_tracks, 200)
	with FigWigWriter(path, {'chr1': 5000}, names, tile=256, groups=n_tracks,
			level=1) as writer:
		writer.write_sparse('chr1', positions, cells, numpy.ones(200))
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets, _ = reader._group_index(0)
	assert set((kinds[kinds > 0] & 7).tolist()) == {3}
	assert int(sizes.sum()) < 10000
	found = reader.read('chr1', [0], 5000, sparse=True)
	assert sorted(zip(found.channels.tolist(), found.offsets.tolist())) == \
		sorted(zip(cells.tolist(), positions.tolist()))
	y = reader.read('chr1', [0], 5000, labels=numpy.zeros(n_tracks, dtype=int))
	assert y[0, 0].sum() == 200


def test_bins_sum_each_window(fw, ref, scratch_mode):
	reader = FigWigReader(fw)
	rng = numpy.random.default_rng(11)
	chroms, starts = random_windows(rng, 60, 256)
	dense = numpy.nan_to_num(reader.read(chroms, starts, 256, dtype='float64'))
	for bins in (1, 32, 256):
		y = reader.read(chroms, starts, 256, bins=bins)
		assert y.shape == (60, 5, 256 // bins) and y.dtype == numpy.float64
		numpy.testing.assert_allclose(y, dense.reshape(60, 5, -1, bins).sum(-1),
			rtol=1e-5)
	y = reader.read(chroms, starts, 256, bins=64, tracks='atac_plus')
	assert y.shape == (60, 4)
	numpy.testing.assert_allclose(y, dense[:, 0].reshape(60, -1, 64).sum(-1))
	y = reader.read(chroms, starts, 256, bins=64, labels=[0, 0, -1, 1, 1],
		dtype='float64')
	expected = numpy.stack([dense[:, :2].sum(1), dense[:, 3:].sum(1)], 1)
	numpy.testing.assert_allclose(y, expected.reshape(60, 2, -1, 64).sum(-1))
	y = reader.read(chroms, starts, 256, bins=256, tracks=['atac_plus',
		'atac_minus'], dtype='int64')
	numpy.testing.assert_array_equal(y[:, :, 0], dense[:, :2].sum(-1))
	with pytest.raises(ValueError, match="divisor of the width"):
		reader.read(chroms, starts, 256, bins=100)
	with pytest.raises(ValueError, match="divisor of the width"):
		reader.read(chroms, starts, 256, bins=0)
	with pytest.raises(TypeError, match="bins must be an integer"):
		reader.read(chroms, starts, 256, bins=2.0)
	with pytest.raises(ValueError, match="sparse cannot be given"):
		reader.read(chroms, starts, 256, bins=2, sparse=True)
	y = reader.read(chroms, starts, 256, bins=64, n_jobs=1)
	numpy.testing.assert_allclose(y, dense.reshape(60, 5, -1, 64).sum(-1),
		rtol=1e-5)


def test_bins_that_straddle_tiles_on_threads(fw, ref, read_batching,
	monkeypatch):
	"""A bin that straddles a tile boundary is filled from two tiles, so a
	binned read's batches are not cut inside a window, and the threads
	give the one-thread result without a copy of the output per thread;
	a window longer than a batch may grow is cut, and the read then sums
	into a copy per thread."""

	reader = FigWigReader(fw)
	seen = []
	run = reader._run

	def spy(tasks, n_threads, summed=None, copies=False):
		seen.append(copies)
		return run(tasks, n_threads, summed, copies)

	monkeypatch.setattr(reader, '_run', spy)
	width = 4992
	dense = expected(ref, numpy.array(['chr1']), numpy.array([1]), width,
		list(range(5)), numpy.float64)
	for bins in (width, 1664, 96, 3):
		for _ in range(3):
			y = reader.read('chr1', [1], width, bins=bins, dtype='float64',
				n_jobs=4)
			numpy.testing.assert_array_equal(y, dense.reshape(1, 5, -1, bins).sum(
				-1))
	# One window of 78 tiles: cut, and summed into copies, under batches
	# of one or three chunks; whole, and summed straight, under the default.
	assert seen and (all(seen) if read_batching[0] < 78 else not any(seen))

	seen.clear()
	reader.read('chr1', [1, 3], width, labels=[0, 0, 1, 1, 1], dtype='float64',
		n_jobs=4)
	assert seen == [False]
	# Windows of 300 bases that straddle many task cuts under small batches.
	rng = numpy.random.default_rng(18)
	chroms, starts = random_windows(rng, 200, 300)
	inside = starts + 300 <= numpy.array([CHROM_SIZES[c] for c in chroms])
	chroms, starts = chroms[inside], starts[inside]
	dense = expected(ref, chroms, starts, 300, list(range(5)), numpy.float64)
	seen.clear()
	for _ in range(3):
		y = reader.read(chroms, starts, 300, bins=100, dtype='float64', n_jobs=4)
		numpy.testing.assert_array_equal(y, dense.reshape(len(starts), 5, 3,
			100).sum(-1))
	if read_batching[0] >= 10:
		assert seen == [False, False, False]


@pytest.fixture(params=['scratch', 'no_scratch'])
def scratch_mode(request, monkeypatch):
	"""Binned reads decode a chunk into a scratch tile when many pairs
	fall on it and it fits _SCRATCH_BYTES, and bin each pair's bases
	straight into its row otherwise: here with the scratch tile as usual
	and with none, as for a read of thousands of tracks. (Chunks of fewer
	than _SCRATCH_PAIRS pairs take the direct path either way.)"""

	if request.param == 'no_scratch':
		monkeypatch.setattr(figwig.figwig, '_SCRATCH_BYTES', 0)
	return request.param


@pytest.mark.parametrize('n_jobs', [1, 4])
def test_sums_over_intervals_of_any_length(fw, track_fw, ref, read_batching,
	scratch_mode, n_jobs):
	"""sums takes intervals that differ in length and gives each track's
	total over each, in position and in track order, as the dense windows
	sum to; an empty interval sums to 0, and one past a chromosome's end
	counts the bases past it as 0."""

	rng = numpy.random.default_rng(14)
	n = 60
	names = numpy.array(list(CHROM_SIZES))
	chroms = names[rng.integers(0, 3, n)]
	lengths = numpy.array([CHROM_SIZES[c] for c in chroms])
	starts = (rng.random(n) * lengths).astype(numpy.int64)
	ends = starts + rng.integers(0, 700, n)
	ends[:3] = starts[:3]
	ends[3] = CHROM_SIZES[chroms[3]] + 50
	expected_sums = numpy.zeros((n, 5))
	for i in range(n):
		width = int(min(ends[i], CHROM_SIZES[chroms[i]]) - starts[i])
		if width > 0:
			expected_sums[i] = expected(ref, chroms[i:i + 1], starts[i:i + 1],
				width, list(range(5)), numpy.float64)[0].sum(-1)
	for path in (fw, track_fw):
		reader = FigWigReader(path)
		y = reader.sums(chroms, starts, ends, dtype='float64', n_jobs=n_jobs)
		assert y.shape == (n, 5)
		numpy.testing.assert_array_equal(y, expected_sums)
		# Sums of an integer track come back float64, which holds them exactly.
		y = reader.sums(chroms, starts, ends, tracks='atac_plus', n_jobs=n_jobs)
		assert y.shape == (n,) and y.dtype == numpy.float64
		numpy.testing.assert_array_equal(y, expected_sums[:, 0])
		# A float32 output accumulates the tiles' partial sums in float32.
		y = reader.sums(chroms, starts, ends, tracks=[2, 0], dtype='float32',
			n_jobs=n_jobs)
		assert y.dtype == numpy.float32
		numpy.testing.assert_allclose(y, expected_sums[:, [2, 0]], rtol=1e-5,
			atol=1e-2)
		assert reader.sums('chr1', [], [], dtype='float64').shape == (0, 5)
		with pytest.raises(ValueError, match="one end per start"):
			reader.sums(chroms, starts, ends[:-1])
		with pytest.raises(ValueError, match="end is before its start"):
			reader.sums('chr1', [10], [9])
		with pytest.raises(ValueError, match="'chr1', 'chr2', 'chrM'"):
			reader.sums('chr9', [0], [10])


@pytest.mark.parametrize('order', ['positions', 'tracks'])
@pytest.mark.parametrize('n_jobs', [1, 4])
def test_sums_of_integer_tracks_over_whole_tiles(tmp_path, order, n_jobs):
	"""Sums of integer tracks over intervals that cover whole tiles take
	those tiles from the tile sums: equal to the dense sums whatever the
	output type, with a tile whose sum is 2**24 or more (rounded in its
	float32 tile sum) decoded instead, intervals past a chromosome's end
	counted to it, and a float track among the tracks read as before."""

	rng = numpy.random.default_rng(31)
	sizes = {'chrA': 4000, 'chrB': 1500}
	tracks = [{'name': 'depth'}, {'name': 'signed'}, {'name': 'big'},
		{'name': 'signal', 'dtype': 'float32'}]
	ref = {}
	for chrom, length in sizes.items():
		values = numpy.zeros((4, length))
		values[0] = rng.integers(0, 60, length)
		values[1] = rng.integers(-500, 500, length)
		values[2, ::7] = rng.integers(0, 200, len(values[2, ::7]))
		values[3] = rng.normal(0, 1, length).astype(numpy.float32)
		ref[chrom] = values
	ref['chrA'][2, 640:704] = 300000
	ref['chrA'][2, 640] += 1
	path = str(tmp_path / 'depth.fw')
	with FigWigWriter(path, sizes, tracks, tile=TILE, groups=[['depth',
			'signed'], ['big'], ['signal']], order=order) as writer:
		for chrom, length in sizes.items():
			writer.write(chrom, [0], ref[chrom][None])
	reader = FigWigReader(path)
	assert reader.tile_sums('big', 'chrA')[10] != ref['chrA'][2, 640:704].sum()

	n = 80
	chroms = numpy.array(list(sizes))[rng.integers(0, 2, n)]
	chroms[:4] = 'chrA'
	lengths = numpy.array([sizes[c] for c in chroms])
	starts = (rng.random(n) * lengths).astype(numpy.int64)
	ends = starts + rng.integers(0, 2000, n)
	starts[:4], ends[:4] = [0, 64, 600, 10], [4000, 128, 720, 10]
	ends[4] = lengths[4] + 30
	clipped = numpy.minimum(ends, lengths)
	dense = numpy.array([ref[c][:, s:e].sum(-1) for c, s, e in zip(chroms,
		starts, clipped)])
	assert ((ends // TILE) - (-(-starts // TILE)) > 2).sum() > 20

	for dtype in ('float64', 'float32', 'int64', None):
		y = reader.sums(chroms, starts, ends, tracks=['depth', 'signed', 'big'],
			dtype=dtype, n_jobs=n_jobs)
		assert_identical(y, dense[:, :3].astype(y.dtype))
	assert_identical(reader.sums(chroms, starts, ends, tracks='big', n_jobs=
		n_jobs, nan='skip'), dense[:, 2])
	y = reader.sums(chroms, starts, ends, dtype='float64', n_jobs=n_jobs)
	assert_identical(y[:, :3], dense[:, :3])
	numpy.testing.assert_allclose(y[:, 3], dense[:, 3], rtol=1e-6, atol=1e-4)
	with pytest.raises(ValueError, match="outside what int16 holds"):
		reader.sums(chroms, starts, ends, tracks='depth', dtype='int16')


@pytest.mark.parametrize('n_jobs', [1, 4])
def test_nan_modes_of_sums(fw, track_fw, ref, scratch_mode, n_jobs):
	"""nan='skip' leaves NaN bases out of a sum, 'count' gives the bases
	that are not NaN, and the default makes a sum with a NaN base NaN; in
	both orders, through sums and through bins."""

	rng = numpy.random.default_rng(16)
	n = 40
	names = numpy.array(list(CHROM_SIZES))
	chroms = names[rng.integers(0, 3, n)]
	lengths = numpy.array([CHROM_SIZES[c] for c in chroms])
	starts = (rng.random(n) * lengths).astype(numpy.int64)
	ends = numpy.minimum(starts + rng.integers(1, 600, n), lengths)
	dense = [expected(ref, chroms[i:i + 1], starts[i:i + 1], int(ends[i] -
		starts[i]), list(range(5)), numpy.float64)[0] for i in range(n)]
	skipped = numpy.array([numpy.nansum(d, axis=-1) for d in dense])
	valid = numpy.array([(~numpy.isnan(d)).sum(-1) for d in dense])
	plain = numpy.array([d.sum(-1) for d in dense])
	assert numpy.isnan(plain).any() and (valid < (ends - starts)[:, None]).any()
	for path in (fw, track_fw):
		reader = FigWigReader(path)
		numpy.testing.assert_array_equal(reader.sums(chroms, starts, ends,
			dtype='float64', n_jobs=n_jobs), plain)
		numpy.testing.assert_array_equal(reader.sums(chroms, starts, ends,
			dtype='float64', n_jobs=n_jobs, nan='skip'), skipped)
		numpy.testing.assert_array_equal(reader.sums(chroms, starts, ends,
			dtype='float64', n_jobs=n_jobs, nan='count'), valid)
		y = reader.read(chroms, starts, 300, bins=100, dtype='float64',
			n_jobs=n_jobs, nan='count')
		full = numpy.nan_to_num(expected(ref, chroms, starts, 300, list(range(
			5)), numpy.float64), nan=-1)
		window = expected(ref, chroms, starts, 300, list(range(5)), numpy.float64)
		inside = (starts[:, None] + numpy.arange(300)[None]) < lengths[:, None]
		counted = (~numpy.isnan(window) & inside[:, None, :]).reshape(n, 5, 3,
			100).sum(-1)
		numpy.testing.assert_array_equal(y, counted)
		del full
		with pytest.raises(ValueError, match="nan must be"):
			reader.sums(chroms, starts, ends, nan='ignore')
		with pytest.raises(ValueError, match="give bins too"):
			reader.read(chroms, starts, 300, nan='skip')


def test_bins_in_track_order(track_fw, fw):
	rng = numpy.random.default_rng(12)
	chroms, starts = random_windows(rng, 40, 512)
	a = FigWigReader(fw).read(chroms, starts, 512, bins=128, dtype='float64')
	b = FigWigReader(track_fw).read(chroms, starts, 512, bins=128,
		dtype='float64')
	numpy.testing.assert_allclose(a, b)
	b = FigWigReader(track_fw).read(chroms, starts, 512, bins=128, labels=[0,
		0, 1, 1, 1], dtype='float64', n_jobs=3)
	numpy.testing.assert_allclose(b[:, 0], a[:, 0] + a[:, 1])


def test_sums_into_an_integer_type_are_checked(fw):
	reader = FigWigReader(fw)
	with pytest.raises(ValueError, match="outside what uint8 holds"):
		reader.read('chr1', [0], 5000, tracks=['atac_plus', 'atac_minus'],
			labels=[0, 0], dtype='uint8')
	with pytest.raises(ValueError, match="outside what int16 holds"):
		reader.read('chr1', [0], 5000, tracks='atac_plus', bins=5000,
			dtype='int16')
	y = reader.read('chr1', [0], 5000, tracks='atac_plus', bins=5000,
		dtype='int64')
	assert y[0, 0] == reader.stats['sum'][0] - reader.read('chr2', [0], 3001,
		tracks='atac_plus', dtype='int64').sum() - reader.read('chrM', [0], 70,
		tracks='atac_plus', dtype='int64').sum()


def test_many_windows_in_one_tile(fw, ref):
	"""Windows packed into one tile are decoded from a scratch tile, on
	every thread, and read the same values."""

	reader = FigWigReader(fw)
	starts = numpy.concatenate([numpy.full(3000, 128), numpy.arange(100, 180)])
	y = reader.read('chr1', starts, 20, n_jobs=4)
	assert_identical(y, expected(ref, ['chr1'] * len(starts), starts, 20,
		list(range(5)), y.dtype))
	y = reader.read('chr1', starts, 20, labels=[0, 1, 1, 0, 0], n_jobs=4)
	dense = reader.read('chr1', starts, 20, dtype='float64')
	numpy.testing.assert_array_equal(y[:, 0], dense[:, [0, 3, 4]].sum(1))
	assert_identical(y[:, 1], (dense[:, 1] + dense[:, 2]).astype(y.dtype))


def test_write_sparse_that_raises_writes_nothing(tmp_path):
	"""A write_sparse call refused for a tile that write left open leaves
	the writer as it was, so the tile can still be finished by write."""

	path = str(tmp_path / 'o.fw')
	with FigWigWriter(path, {'chr1': 1000}, ['a', 'b'], groups=1, tile=64) \
			as writer:
		writer.write('chr1', [0], numpy.full((1, 2, 30), 2))
		with pytest.raises(ValueError, match="was begun by write"):
			writer.write_sparse('chr1', [40, 200], ['a', 'b'], [9, 9])
		writer.write('chr1', [30], numpy.full((1, 2, 34), 3))
		writer.write_sparse('chr1', [200], 'b', [9])
	y = FigWigReader(path).read('chr1', [0], 1000)[0]
	assert y[0, :30].tolist() == [2] * 30 and y[0, 30:64].tolist() == [3] * 34
	assert y[1, 200] == 9 and y[:, 64:200].sum() == 0 and y[:, 201:].sum() == 0


def test_failed_writer_leaves_the_path_as_it_was(tmp_path):
	path = str(tmp_path / 'p.fw')
	write_figwig(path, {'chr1': 100}, ['a'], 'chr1', [0], numpy.ones((1, 1,
		100)), tile=16)
	with open(path, 'rb') as handle:
		before = handle.read()
	with pytest.raises(ValueError):
		with FigWigWriter(path, {'chr1': 100}, ['a'], tile=16) as writer:
			writer.write('chr1', [0], numpy.ones((1, 1, 10)))
			writer.write('chr1', [5], numpy.ones((1, 1, 10)))
	with open(path, 'rb') as handle:
		assert handle.read() == before
	assert not any(name.endswith('.partial') for name in os.listdir(tmp_path))

	# A file can be written over itself: the reader holds the old inode
	# until the writer is done.
	reader = FigWigReader(path)
	with FigWigWriter(path, {'chr1': 100}, ['a'], tile=16) as writer:
		writer.write('chr1', [0], reader.read('chr1', [0], 100) * 2)
	assert FigWigReader(path).read('chr1', [0], 100).sum() == 200
	with pytest.raises(ValueError, match="replaced or rewritten"):
		reader.read('chr1', [0], 100)


def test_unclosed_writer_warns(tmp_path):
	import gc
	path = str(tmp_path / 'u.fw')
	writer = FigWigWriter(path, {'chr1': 100}, ['a'], tile=16)
	writer.write('chr1', [0], numpy.ones((1, 1, 10)))
	with pytest.warns(ResourceWarning, match="never closed"):
		del writer
		gc.collect()
	assert not os.path.exists(path)
	assert not any(name.endswith('.partial') for name in os.listdir(tmp_path))


def test_reader_keeps_to_the_file_it_opened(tmp_path):
	path = str(tmp_path / 'c.fw')
	write_figwig(path, {'chr1': 100}, ['a'], 'chr1', [0], numpy.ones((1, 1,
		100)), tile=16)
	reader = FigWigReader(path)
	copy = pickle.loads(pickle.dumps(reader))
	write_figwig(path, {'chr1': 100}, ['a'], 'chr1', [0], numpy.full((1, 1,
		100), 2), tile=16, level=0)
	for r in (reader, copy):
		with pytest.raises(ValueError, match="replaced or rewritten"):
			r.read('chr1', [0], 100)
	os.remove(path)
	with pytest.raises(ValueError, match="cannot be read"):
		reader.read('chr1', [0], 100)

	# An absolute path survives a change of directory.
	write_figwig(path, {'chr1': 100}, ['a'], 'chr1', [0], numpy.ones((1, 1,
		100)), tile=16)
	cwd = os.getcwd()
	os.chdir(tmp_path)
	try:
		reader = FigWigReader('c.fw')
		os.chdir(cwd)
		assert reader.path == os.path.abspath(path)
		assert reader.read('chr1', [0], 100).sum() == 100
	finally:
		os.chdir(cwd)


def test_crafted_counts_raise_value_errors(tmp_path, fw):
	"""A count in the header or the index that the file could not hold
	raises a ValueError before any memory is asked for."""

	with open(fw, 'rb') as handle:
		data = bytearray(handle.read())
	header = numpy.frombuffer(bytes(data[:128]), dtype=figwig.figwig._HEADER)
	index = int(header['index'][0])
	for name, mutate in (('n_runs', lambda d: d.__setitem__(slice(index + 24,
			index + 28), (2 ** 32 - 1).to_bytes(4, 'little'))), ('chrom_length',
			lambda d: d.__setitem__(slice(int(header['chroms'][0]) + 6, int(
			header['chroms'][0]) + 14), (2 ** 63).to_bytes(8, 'little')))):
		copy = bytearray(data)
		mutate(copy)
		path = str(tmp_path / (name + '.fw'))
		with open(path, 'wb') as handle:
			handle.write(bytes(copy))
		with pytest.raises(ValueError):
			FigWigReader(path).read('chr1', [0], 100)

	copy = bytearray(data)
	copy[4:6] = (2).to_bytes(2, 'little')
	path = str(tmp_path / 'v2.fw')
	with open(path, 'wb') as handle:
		handle.write(bytes(copy))
	with pytest.raises(ValueError, match="version 2 of the format.*newer figwig"):
		FigWigReader(path)


def test_track_order_rejects_overlap_across_calls(tmp_path):
	path = str(tmp_path / 't.fw')
	with FigWigWriter(path, {'chr1': 1000}, ['a'], tile=64, order='tracks') \
			as writer:
		values = numpy.zeros((1, 1, 100))
		values[0, 0, 10] = 1
		writer.write('chr1', [0], values)
		with pytest.raises(ValueError, match="before the end of what was"):
			writer.write('chr1', [50], numpy.ones((1, 1, 100)))
		writer.write('chr1', [100], numpy.ones((1, 1, 100)))


def test_index_of_empty_groups_is_small(tmp_path):
	"""A group's index is deflated and holds only the tiles with values,
	so a file of many empty groups on a long genome stays small."""

	path = str(tmp_path / 'i.fw')
	names = ['t{}'.format(i) for i in range(64)]
	with FigWigWriter(path, {'chr1': 2 ** 28}, names, groups=1, tile=4096) \
			as writer:
		writer.write('chr1', [2 ** 27], numpy.ones((1, 64, 10)))
	assert os.path.getsize(path) < 64 * 400
	reader = FigWigReader(path)
	assert reader.read('chr1', [2 ** 27 - 2], 14).sum() == 640
	kinds, sizes, raws, offsets, _ = reader._group_index(5)
	assert sizes.dtype == numpy.uint32 and (kinds > 0).sum() == 1
	copy = pickle.loads(pickle.dumps(reader))
	assert all(index is None for index in copy._index)


def test_same_stats_whatever_the_call_split(tmp_path):
	"""The sum of a float track is compensated, so splitting the values
	between calls differently gives the same statistics table."""

	rng = numpy.random.default_rng(8)
	values = rng.normal(size=(1, 1, 200000)).astype(numpy.float32) * 1000
	sums = []
	for width in (200000, 7000, 1000):
		path = str(tmp_path / '{}.fw'.format(width))
		starts = numpy.arange(0, 200000, width)
		pieces = numpy.full((len(starts), 1, width), numpy.nan)
		for j, start in enumerate(starts):
			piece = values[0, :, start:start + width]
			pieces[j, :, :piece.shape[1]] = piece
		with FigWigWriter(path, {'chr1': 200000}, ['a'], tile=4096) as writer:
			for j in range(len(starts)):
				writer.write('chr1', starts[j:j + 1], pieces[j:j + 1])
		sums.append(float(FigWigReader(path).stats['sum'][0]))
	assert sums[0] == sums[1] == sums[2]
	assert abs(sums[0] - float(values.astype(numpy.float64).sum())) < 1e-6


def test_encode_tiles_stops_before_a_tile_that_might_not_fit():
	"""The encoder returns the tiles it encoded when the next might not fit
	its buffer, and goes on from there in the next call."""

	tile = 16
	values = numpy.ones((1, 1, 64)) * 3
	buffer = numpy.zeros((1, tile))
	out = numpy.zeros(160, dtype=numpy.uint8)      # one tile at 8 bytes + 16
	bounds = numpy.zeros(5, dtype=numpy.int64)
	kinds = numpy.zeros(4, dtype=numpy.uint8)
	stats = numpy.zeros((1, 5))
	classes = numpy.zeros(1, dtype=numpy.int64)
	tiles, lo, hi = numpy.arange(4), numpy.zeros(4, dtype=numpy.int64), \
		numpy.ones(4, dtype=numpy.int64)
	n = _encode_tiles(tile, tiles, lo, hi, numpy.array([0]), numpy.array([64]),
		values, True, buffer, False, classes, True, False, out, bounds,
		kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert n == 2 and (kinds[:n] > 0).all() and not buffer.any()
	m = _encode_tiles(tile, tiles[n:], lo[n:], hi[n:], numpy.array([0]),
		numpy.array([64]), values, True, buffer, False, classes, True,
		False, out, bounds, kinds, stats, numpy.zeros((len(kinds), buffer.shape[0]), numpy.float32))
	assert m == 2 and stats[0, 1] == 64


###
# Summaries, checksums, appending and merging


def test_tile_sums(fw, ref):
	"""Each track's sum over every tile comes from the file's summaries,
	without decoding a chunk, and matches the values."""

	reader = FigWigReader(fw)
	sums = reader.tile_sums()
	# A NaN in a tile makes its sum NaN, as it makes the track's sum.
	expected = []
	for chrom, length in CHROM_SIZES.items():
		padded = numpy.pad(ref[chrom], ((0, 0), (0, -length % TILE)))
		expected.append(numpy.nansum(padded.reshape(5, -1, TILE), axis=-1))
	expected = numpy.concatenate(expected, axis=1).astype(numpy.float32)
	assert sums.shape == expected.shape and sums.dtype == numpy.float32
	numpy.testing.assert_allclose(sums, expected, rtol=1e-6)
	assert not numpy.isnan(sums).any()
	numpy.testing.assert_allclose(reader.tile_sums('signal', 'chr2'),
		expected[2, 79:79 + 47], rtol=1e-6)
	numpy.testing.assert_allclose(reader.tile_sums(['empty', 'atac_plus'],
		['chrM', 'chr1']), numpy.concatenate([expected[[4, 0], 126:],
		expected[[4, 0], :79]], axis=1), rtol=1e-6)
	numpy.testing.assert_allclose(sums.sum(axis=1), reader.stats['sum'],
		rtol=1e-5)
	with pytest.raises(ValueError, match="not one of the file's chromosomes"):
		reader.tile_sums(chroms='chr9')


def test_tile_sums_in_track_order(track_fw, fw):
	numpy.testing.assert_allclose(FigWigReader(track_fw).tile_sums(),
		FigWigReader(fw).tile_sums(), rtol=1e-6)


def test_files_past_the_summary_cap_keep_none(tmp_path, monkeypatch):
	monkeypatch.setattr(figwig.figwig, '_SUMMARY_BYTES', 100)
	path = str(tmp_path / 'big.fw')
	write_figwig(path, {'chr1': 64 * 100}, ['a'], 'chr1', [0], numpy.ones((1, 1,
		6400)), tile=64)
	reader = FigWigReader(path)
	assert reader.read('chr1', [0], 10).sum() == 10
	with pytest.raises(ValueError, match="keeps no summary"):
		reader.tile_sums()


def test_verify_checks_every_chunk(tmp_path):
	"""A flipped byte in an uncompressed chunk passes an ordinary read and
	fails one with verify=True; a compressed chunk fails either way."""

	for level in (0, 6):
		path = str(tmp_path / 'v{}.fw'.format(level))
		values = numpy.arange(2000.0)[None, None]
		write_figwig(path, {'chr1': 2000}, ['a'], 'chr1', [0], values, tile=256,
			level=level)
		with open(path, 'rb') as handle:
			data = bytearray(handle.read())
		data[300] ^= 0x40
		with open(path, 'wb') as handle:
			handle.write(bytes(data))
		with pytest.raises(ValueError, match="checksum|corrupt"):
			FigWigReader(path, verify=True).read('chr1', [0], 2000)
		if level == 0:
			y = FigWigReader(path).read('chr1', [0], 2000)
			assert not numpy.array_equal(y[0, 0], values[0, 0])
	reader = FigWigReader(path, verify=True)
	assert reader.verify
	copy = pickle.loads(pickle.dumps(reader))
	assert copy.verify


def test_append_adds_tracks_to_a_file(tmp_path, ref):
	"""append=True starts from the file's tracks, whose chunks are copied,
	and adds the new ones after them; the old tracks cannot be written."""

	path = str(tmp_path / 'grow.fw')
	write_reference(path, ref, tile=TILE)
	before = FigWigReader(path)
	old_sums = before.tile_sums()
	with FigWigWriter(path, None, [{'name': 'more', 'assay': 'new'}],
			append=True, n_jobs=2) as writer:
		assert [t['name'] for t in writer.tracks][:5] == [t['name'] for t in
			TRACKS]
		with pytest.raises(ValueError, match="came from another file"):
			writer.write('chr1', [0], numpy.ones((1, 1, 10)), tracks='atac_plus')
		writer.write('chr1', [10], numpy.full((1, 1, 20), 3.0))
		writer.write('chr2', [0], numpy.full((1, 1, 5), 2.0), tracks='more')
	after = FigWigReader(path, verify=True)
	assert [t['name'] for t in after.tracks] == [t['name'] for t in TRACKS] + \
		['more']
	assert [t['group'] for t in after.tracks] == [0, 0, 1, 2, 2, 3]
	assert after.tracks[5]['assay'] == 'new' and after.tracks[5]['dtype'] == \
		'uint8'
	assert after.genome == 'hg38' and after.attributes == before.attributes
	rng = numpy.random.default_rng(9)
	chroms, starts = random_windows(rng, 50, 300)
	y = after.read(chroms, starts, 300, dtype='float64')
	assert_identical(y[:, :5], expected(ref, chroms, starts, 300, list(range(
		5)), numpy.float64))
	numpy.testing.assert_array_equal(after.read('chr1', [0], 40, tracks='more')[
		0], [0] * 10 + [3] * 20 + [0] * 10)
	for field in before.stats.dtype.names:
		numpy.testing.assert_array_equal(before.stats[field], after.stats[field][
			:5])
	numpy.testing.assert_array_equal(after.tile_sums()[:5], old_sums)
	assert after.tile_sums('more', 'chr1')[0] == 60
	entries = after.read_entries(tracks='more')
	assert len(entries.values) == 25

	with pytest.raises(ValueError, match="must be the file's"):
		FigWigWriter(path, None, ['x'], append=True, tile=16)
	with pytest.raises(ValueError, match="must be the file's"):
		FigWigWriter(path, {'chr1': 10}, ['x'], append=True)
	with pytest.raises(ValueError, match="named twice|already one of"):
		FigWigWriter(path, None, ['more'], append=True)
	with pytest.raises(ValueError, match="no tracks"):
		with FigWigWriter(str(tmp_path / 'none.fw'), {'chr1': 10}, None):
			pass


def test_append_in_track_order(tmp_path, ref):
	path = str(tmp_path / 'grow_t.fw')
	write_reference(path, ref, order='tracks')
	with open(path, 'rb') as handle:
		before = handle.read()
	with FigWigWriter(path, None, ['more'], append=True) as writer:
		writer.write_sparse('chr2', [5, 2999], 'more', [4, 5])
	with open(path, 'rb') as handle:
		after_bytes = handle.read()
	assert after_bytes[128:len(before)] == before[128:]
	after = FigWigReader(path, verify=True)
	assert after.order == 'tracks' and len(after.tracks) == 6
	rng = numpy.random.default_rng(10)
	chroms, starts = random_windows(rng, 50, 300)
	assert_identical(after.read(chroms, starts, 300, tracks=list(range(5)),
		dtype='float64'), expected(ref, chroms, starts, 300, list(range(5)),
		numpy.float64))
	y = after.read('chr2', [0], 3001, tracks='more')[0]
	assert y[5] == 4 and y[2999] == 5 and y.sum() == 9

	# A copy of the file leaves its old index behind and moves its blocks.
	copied = str(tmp_path / 'copied_t.fw')
	with FigWigWriter(copied, CHROM_SIZES, None, tile=TILE, order='tracks',
			level=6) as \
			writer:
		writer.add_file(path)
	assert os.path.getsize(copied) < os.path.getsize(path)
	reader = FigWigReader(copied, verify=True)
	assert_identical(reader.read(chroms, starts, 300, tracks=list(range(5)),
		dtype='float64'), expected(ref, chroms, starts, 300, list(range(5)),
		numpy.float64))
	assert_identical(reader.read('chr2', [0], 3001, tracks='more'), y[None])


def test_append_writes_in_place(tmp_path, ref):
	"""append=True extends the file itself: its chunks stay where they are,
	the old tables and index are left behind them, the header is written
	last, a writer that fails leaves the file as it was, bytes and
	modification time, and a reader open across an append that finished is
	told to open the file again."""

	path = str(tmp_path / 'inplace.fw')
	write_reference(path, ref, tile=TILE)
	with open(path, 'rb') as handle:
		before = handle.read()
	reader = FigWigReader(path)
	rng = numpy.random.default_rng(12)
	chroms, starts = random_windows(rng, 50, 300)
	old = expected(ref, chroms, starts, 300, list(range(5)), numpy.float64)

	with pytest.raises(RuntimeError, match="part way"):
		with FigWigWriter(path, None, ['more'], append=True) as writer:
			writer.write('chr1', [0], numpy.full((1, 1, 2 * TILE), 3.0))
			writer._write(len(writer._inflight))
			writer._file.flush()
			assert os.path.getsize(path) > len(before)
			raise RuntimeError("stopped part way")
	with open(path, 'rb') as handle:
		assert handle.read() == before
	assert_identical(reader.read(chroms, starts, 300, dtype='float64'), old)

	# Bytes past the header's length, as a killed append leaves, are
	# ignored by a reader and written over by the next append.
	with open(path, 'ab') as handle:
		handle.write(b'\xff' * 1000)
	assert_identical(FigWigReader(path).read(chroms, starts, 300,
		dtype='float64'), old)
	with FigWigWriter(path, None, ['more'], append=True) as writer:
		writer.write('chr1', [10], numpy.full((1, 1, 20), 3.0))
	with open(path, 'rb') as handle:
		after = handle.read()
	assert after[128:len(before)] == before[128:]
	header = numpy.frombuffer(after[:128], dtype=figwig.figwig._HEADER)[0]
	assert int(header['chroms']) > len(before)
	assert int(header['length']) == len(after)
	with pytest.raises(ValueError, match="open it again"):
		reader.read(chroms, starts, 300)

	grown = FigWigReader(path, verify=True)
	assert [t['name'] for t in grown.tracks] == [t['name'] for t in TRACKS] + \
		['more']
	assert_identical(grown.read(chroms, starts, 300, tracks=list(range(5)),
		dtype='float64'), old)
	numpy.testing.assert_array_equal(grown.read('chr1', [0], 40, tracks='more')[
		0], [0] * 10 + [3] * 20 + [0] * 10)

	with FigWigWriter(path, None, ['third'], append=True) as writer:
		writer.write_sparse('chr2', [7], 'third', [9])
	twice = FigWigReader(path, verify=True)
	assert [t['group'] for t in twice.tracks] == [0, 0, 1, 2, 2, 3, 4]
	assert twice.read('chr2', [0], 10, tracks='third')[0, 7] == 9
	assert_identical(twice.read(chroms, starts, 300, tracks=list(range(5)),
		dtype='float64'), old)

	# A copy of the file holds only its chunks, at new offsets; a file with
	# no chunks at all has none to copy.
	empty = str(tmp_path / 'empty.fw')
	with FigWigWriter(empty, CHROM_SIZES, ['nothing'], tile=TILE):
		pass
	copied = str(tmp_path / 'copied.fw')
	with FigWigWriter(copied, CHROM_SIZES, None, tile=TILE) as writer:
		writer.add_file(empty)
		writer.add_file(path)
	assert FigWigReader(copied).tracks[0]['name'] == 'nothing'
	assert os.path.getsize(copied) < os.path.getsize(path)
	reader = FigWigReader(copied, verify=True)
	assert_identical(reader.read(chroms, starts, 300, tracks=list(range(1, 6)),
		dtype='float64'), old)
	assert reader.read('chr2', [0], 10, tracks='third')[0, 7] == 9
	numpy.testing.assert_array_equal(reader.tile_sums()[1:], twice.tile_sums())
	for field in reader.stats.dtype.names:
		numpy.testing.assert_array_equal(reader.stats[field][1:], twice.stats[
			field])


def test_add_file_copies_tracks(tmp_path, ref, cells):
	"""add_file takes another file's tracks whole, in their groups, and
	refuses a file written with another tile, level, order or genome."""

	first = str(tmp_path / 'first.fw')
	write_reference(first, ref)
	_, entries, paths = cells
	merged = str(tmp_path / 'merged.fw')
	with FigWigWriter(merged, CHROM_SIZES, ['own'], tile=TILE) as writer:
		writer.write('chr1', [0], numpy.ones((1, 1, 100)))
		writer.add_file(first)
		with pytest.raises(ValueError, match="already one of the writer's"):
			writer.add_file(first)
		with pytest.raises(ValueError, match="was written with (tile|level)"):
			writer.add_file(paths['positions', 120])
		with pytest.raises(ValueError, match="was written with order tracks"):
			writer.add_file(paths['tracks', 1])
	reader = FigWigReader(merged, verify=True)
	assert [t['name'] for t in reader.tracks] == ['own'] + [t['name'] for t in
		TRACKS]
	assert [t['group'] for t in reader.tracks] == [0, 1, 1, 2, 3, 3]
	rng = numpy.random.default_rng(11)
	chroms, starts = random_windows(rng, 50, 300)
	assert_identical(reader.read(chroms, starts, 300, tracks=list(range(1, 6)),
		dtype='float64'), expected(ref, chroms, starts, 300, list(range(5)),
		numpy.float64))
	assert reader.read('chr1', [0], 100, tracks='own').sum() == 100

	other = str(tmp_path / 'other.fw')
	write_figwig(other, {'chr1': 100}, ['x'], 'chr1', [0], numpy.ones((1, 1,
		100)), tile=TILE)
	with FigWigWriter(str(tmp_path / 'm2.fw'), CHROM_SIZES, None, tile=TILE) \
			as writer:
		with pytest.raises(ValueError, match="other chromosomes"):
			writer.add_file(other)
		writer.add_file(first)


###
# Found by the persona test


@pytest.mark.parametrize('fallback', [False, True])
def test_damaged_compressed_chunk_never_reads_wrong(fw, tmp_path, monkeypatch,
	fallback):
	"""A bit flipped anywhere in a compressed chunk is refused or, when the
	flip is invisible, read as before: a chunk whose stream libdeflate
	rejects is inflated by zlib up to its checksum, never short of it. With
	`fallback` every chunk takes that path, as when no C library loads."""

	if fallback:
		monkeypatch.setattr(figwig.figwig, '_libdeflate_inflate', lambda: None)
		monkeypatch.setattr(figwig.figwig, '_zlib_uncompress', lambda: None)
	path = str(tmp_path / 'flipped.fw')
	shutil.copy(fw, path)
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets, crcs = reader._group_index(0)
	t = int(numpy.argmax(sizes * (sizes != raws)))
	assert sizes[t] > 0 and sizes[t] != raws[t]
	offset, size = int(offsets[t]), int(sizes[t])
	tid = int(numpy.searchsorted(reader._tile_offsets, t, side='right') - 1)
	chrom = list(reader.chrom_sizes)[tid]
	start = (t - int(reader._tile_offsets[tid])) * reader.tile
	width = min(reader.tile, reader.chrom_sizes[chrom] - start)
	good = reader.read(chrom, [start], width, dtype='float64', n_jobs=1)
	with open(path, 'rb') as handle:
		handle.seek(offset)
		original = bytearray(handle.read(size))
	outcomes = {'refused': 0, 'same': 0}
	for i in range(size):
		damaged = bytearray(original)
		damaged[i] ^= 0x10
		with open(path, 'r+b') as handle:
			handle.seek(offset)
			handle.write(damaged)
		try:
			y = FigWigReader(path).read(chrom, [start], width, dtype='float64',
				n_jobs=1)
		except ValueError:
			outcomes['refused'] += 1
			continue
		assert_identical(y, good)
		outcomes['same'] += 1
	assert outcomes['refused'] > size // 2


@pytest.mark.skipif(sys.platform == 'win32', reason="no file modes")
def test_files_are_readable_as_the_umask_allows(tmp_path):
	path = str(tmp_path / 'mode.fw')
	write_figwig(path, {'chr1': 100}, ['x'], 'chr1', [0], numpy.ones((1, 1, 100)))
	mask = os.umask(0)
	os.umask(mask)
	assert os.stat(path).st_mode & 0o777 == 0o666 & ~mask


def test_writer_abandoned_from_a_pool_thread(tmp_path):
	"""A writer dropped with work in flight is finalized on a pool thread;
	its _abandon must not wait for that thread."""

	path = str(tmp_path / 'dropped.fw')
	writer = FigWigWriter(path, {'chr1': 100000}, ['x'], n_jobs=2)
	writer.write('chr1', [0], numpy.ones((1, 1, 100000)))
	partial = writer._partial
	writer._pool.submit(writer._abandon).result()
	assert writer._file is None and not os.path.exists(partial)
	assert not os.path.exists(path)


def test_reader_reads_through_an_append(tmp_path, ref):
	"""A reader open on a file keeps reading while a writer appends to it,
	since the chunks its index names stay where they are, and is told to
	open the file again once the append has closed, or when the file has
	been truncated."""

	path = str(tmp_path / 'during.fw')
	write_reference(path, ref, tile=TILE)
	reader = FigWigReader(path)
	rng = numpy.random.default_rng(13)
	chroms, starts = random_windows(rng, 30, 300)
	before = reader.read(chroms, starts, 300, dtype='float64')
	writer = FigWigWriter(path, None, ['more'], append=True, n_jobs=1)
	writer.write('chr1', [0], numpy.full((1, 1, 4 * TILE), 2.0))
	writer._write(len(writer._inflight))
	writer._file.flush()
	assert os.path.getsize(path) > reader._file_size
	assert_identical(reader.read(chroms, starts, 300, dtype='float64'), before)
	writer.close()
	with pytest.raises(ValueError, match="open it again"):
		reader.read(chroms, starts, 300)

	reader = FigWigReader(path)
	with open(path, 'r+b') as handle:
		handle.truncate(reader._file_size - 1)
	with pytest.raises(ValueError, match="truncated"):
		reader.read(chroms, starts, 300)


def test_messages_say_what_to_do(fw):
	reader = FigWigReader(fw)
	with pytest.raises(ValueError, match="no track has an attribute 'strnd'; "
			"the tracks' keys are"):
		reader.select(strnd='+')
	with pytest.raises(ValueError, match="'chr1', 'chr2', 'chrM'"):
		reader.read('chr3', [0], 10)
	with pytest.raises(ValueError, match="'chr1', 'chr2', 'chrM'"):
		reader.tile_sums(chroms='Chr1')
	with pytest.raises(ValueError, match="pass dtype="):
		reader.read('chr1', [0], 10, out=numpy.zeros((1, 5, 10), dtype='float64'))
	with pytest.raises(ValueError, match="give one per window"):
		reader.read('chr1', 0, 10)
	torch = pytest.importorskip('torch')
	assert reader.read('chr1', [0], 10, dtype=torch.float32).dtype == \
		numpy.float32
	assert reader.read('chr1', [0], 10, dtype=torch.int32, tracks='atac_plus'
		).dtype == numpy.int32


def test_chromosome_mismatch_is_described(tmp_path, ref):
	first = str(tmp_path / 'a.fw')
	write_reference(first, ref)
	other = str(tmp_path / 'b.fw')
	write_figwig(other, {'chr1': 5000, 'chr2': 3000, 'chrX': 9}, ['x'], 'chr1',
		[0], numpy.ones((1, 1, 10)), tile=TILE)
	with FigWigWriter(str(tmp_path / 'm.fw'), CHROM_SIZES, None, tile=TILE) as \
			writer:
		writer.add_file(first)
		with pytest.raises(ValueError, match="has a chromosome the writer lacks, "
				"such as 'chrX'; it lacks a chromosome the writer has, such as "
				"'chrM'; 'chr2' is 3000 bases long in it and 3001 in the writer"):
			writer.add_file(other)


@pytest.mark.parametrize('level', [6, 0])
def test_batches_encoded_in_several_goes(tmp_path, monkeypatch, level):
	"""A batch whose chunks outgrow the encoder's buffer is encoded in
	several goes and joined from the parts: float64 chunks, at eight bytes
	per value where the buffer holds four, and entries past _BATCH_BYTES.
	Each part is its own streams, not the front of a buffer the next go
	writes over."""

	path = str(tmp_path / 'parts.fw')
	n = 16 * 40
	y = (numpy.arange(n) + 0.5)[None, None]
	write_figwig(path, {'chr1': n}, [{'name': 't', 'dtype': 'float64'}], 'chr1',
		[0], y, tile=16, level=level)
	reader = FigWigReader(path, verify=True)
	assert_identical(reader.read('chr1', [0], n), y)
	numpy.testing.assert_array_equal(reader.read('chr1', [0], n, bins=16)[0, 0],
		y[0, 0].reshape(-1, 16).sum(-1))

	monkeypatch.setattr(figwig.figwig, '_BATCH_BYTES', 2000)
	rng = numpy.random.default_rng(15)
	positions = numpy.sort(rng.choice(20000, 6000, replace=False))
	values = rng.integers(1, 200, 6000)
	with FigWigWriter(path, {'chr1': 20000}, ['t'], tile=64, level=level) as \
			writer:
		writer.write_sparse('chr1', positions, 't', values)
	expected_track = numpy.zeros(20000, dtype=numpy.int64)
	expected_track[positions] = values
	reader = FigWigReader(path, verify=True)
	numpy.testing.assert_array_equal(reader.read('chr1', [0], 20000,
		dtype='int64')[0, 0], expected_track)
	assert len(reader._runs[0]) >= 1


def test_stored_chunks_in_a_compressed_file_are_checked(tmp_path, ref):
	"""A chunk stored as it is inside a compressed file has no zlib checksum,
	so its CRC is checked on every read, as the compressed chunks' Adler-32
	is; an uncompressed file's chunks are checked only with verify=True."""

	path = str(tmp_path / 'stored.fw')
	rng = numpy.random.default_rng(17)
	# Random bytes do not deflate, so these chunks are stored as they are.
	values = rng.integers(0, 256, (1, 1, 4 * TILE)).astype(numpy.float64)
	write_figwig(path, {'chr1': 4 * TILE}, ['noise'], 'chr1', [0], values,
		tile=TILE, level=6)
	reader = FigWigReader(path)
	kinds, sizes, raws, offsets, crcs = reader._group_index(0)
	assert ((kinds & figwig.figwig._STORED) > 0).all()
	good = reader.read('chr1', [0], 4 * TILE)
	with open(path, 'r+b') as handle:
		handle.seek(int(offsets[1]) + 5)
		byte = handle.read(1)
		handle.seek(int(offsets[1]) + 5)
		handle.write(bytes([byte[0] ^ 0x10]))
	with pytest.raises(ValueError, match="does not match its checksum"):
		FigWigReader(path).read('chr1', [0], 4 * TILE)
	with pytest.raises(ValueError, match="does not match its checksum"):
		FigWigReader(path, verify=True).read('chr1', [0], 4 * TILE)
	assert_identical(FigWigReader(path).read('chr1', [0], TILE), good[:, :,
		:TILE])


def test_append_merges_attributes(tmp_path, ref):
	path = str(tmp_path / 'attrs.fw')
	write_reference(path, ref, tile=TILE)
	before = FigWigReader(path).attributes
	assert before
	with FigWigWriter(path, None, ['more'], append=True, attributes={'batch':
			2, 'lab': 'x'}) as writer:
		writer.write('chr1', [0], numpy.ones((1, 1, 10)))
	after = FigWigReader(path).attributes
	assert after == {**before, 'batch': 2, 'lab': 'x'}


def test_window_that_ends_a_large_chunk_keeps_the_next_tile_open(tmp_path):
	"""A window that completes a tile whose chunk is large (float64 values)
	and runs into the next tile leaves that tile's values in the open
	buffer: the encoder, which stops before a tile that might not fit its
	buffer, must not stop before the tile it only keeps."""

	path = str(tmp_path / 'kept.fw')
	rng = numpy.random.default_rng(28)
	n = 3001
	expected = numpy.zeros(n)
	calls = [(1239, 1293), (1293, 1318), (1318, 1349), (1349, 1600)]
	with FigWigWriter(path, {'chr1': n}, [{'name': 'f64', 'dtype': 'float64'}],
			tile=64, level=6, n_jobs=1) as writer:
		for start, end in calls:
			values = (rng.integers(0, 2 ** 40, end - start) + 0.5) * (rng.random(
				end - start) < 0.5)
			writer.write('chr1', [start], values[None, None])
			expected[start:end] = values
	y = FigWigReader(path, verify=True).read('chr1', [0], n, dtype='float64')
	numpy.testing.assert_array_equal(y[0, 0], expected)


def test_split_tasks_cut_between_tiles_no_window_spans(fw, ref, monkeypatch):
	"""A read of many windows over few chunks is split by pairs across the
	threads; a binned read's pieces are cut only at tile boundaries that
	no window spans, since a window's pairs lie apart in tile order."""

	reader = FigWigReader(fw)
	rng = numpy.random.default_rng(31)
	n = 1200
	starts = rng.integers(0, 100, n)
	chroms = numpy.array(['chr1'] * n)
	dense = expected(ref, chroms, starts, 100, list(range(5)), numpy.float64)
	want = dense.reshape(n, 5, 2, 50).sum(-1)
	monkeypatch.setattr(figwig.figwig, '_CHUNKS_PER_THREAD', 1)
	seen = []
	run = reader._run

	def spy(tasks, n_threads, summed=None, copies=False):
		seen.append((len(tasks), copies))
		return run(tasks, n_threads, summed, copies)

	monkeypatch.setattr(reader, '_run', spy)
	for _ in range(10):
		y = reader.read(chroms, starts, 100, bins=50, dtype='float64', n_jobs=4)
		numpy.testing.assert_array_equal(y, want)
	# Every window spans the one tile boundary, so the pieces are cut
	# inside windows and sum into copies.
	assert seen[0][0] >= 4 and all(copies for _, copies in seen)

	# Windows inside the first tile of two chromosomes: the pieces are cut
	# at the tile boundary between them, which no window spans; no copies.
	seen.clear()
	starts = rng.integers(0, 30, n)
	chroms = numpy.where(numpy.arange(n) % 2 == 0, 'chr1', 'chr2')
	dense = expected(ref, chroms, starts, 30, list(range(5)), numpy.float64)
	y = reader.read(chroms, starts, 30, bins=10, dtype='float64', n_jobs=4)
	numpy.testing.assert_array_equal(y, dense.reshape(n, 5, 3, 10).sum(-1))
	assert seen[0][0] >= 4 and not seen[0][1]


def test_chunk_crcs_are_zlibs():
	"""The CRC-32 kernel that checksums every chunk gives zlib's values on
	pieces of every length, empty ones included."""

	import zlib

	from figwig.figwig import _crcs

	rng = numpy.random.default_rng(0)
	data = rng.integers(0, 256, 100_000, dtype=numpy.uint8).tobytes()
	cuts = numpy.concatenate([[0], numpy.sort(rng.integers(0, len(data), 300)),
		[len(data), len(data)]])
	expected = [zlib.crc32(data[int(a):int(b)]) for a, b in zip(cuts[:-1],
		cuts[1:])]
	assert _crcs(data, cuts).tolist() == expected
	for n in range(0, 40):
		piece = data[:n]
		assert _crcs(piece, numpy.array([0, n])).tolist() == [zlib.crc32(piece)]


# Dense tracks: short runs decode in place, and a chromosome of dense chunks
# is planned as enough batches for every thread without a chunk being
# inflated twice.


@pytest.fixture(scope='module')
def runs_fw(tmp_path_factory):
	"""A float32 track whose values hold for one to three bases, as
	conservation scores do, stored as runs chunks, beside a dense random
	track, at the default tile."""

	rng = numpy.random.default_rng(5)
	length = 4 * 16384 + 1000
	lengths = rng.integers(1, 4, length)
	values = numpy.repeat(rng.choice([-1.5, 0.0, 0.25, 2.0], length), lengths)[
		:length].astype(numpy.float32)
	dense = rng.normal(size=length).astype(numpy.float32)
	path = str(tmp_path_factory.mktemp('runs') / 'runs.fw')
	with FigWigWriter(path, {'chr1': length}, [{'name': 'cons', 'dtype':
			'float32'}, {'name': 'noise', 'dtype': 'float32'}], groups=[['cons'],
			['noise']], n_jobs=2) as writer:
		writer.write('chr1', [0], numpy.stack([values, dense])[None])
	return path, values, dense


def test_short_runs_read_as_written(runs_fw, read_batching):
	"""Windows, labels, bins, sums and entries of a track of runs a few
	bases long give what the values were, with the chunk kinds the writer
	chose: runs for the scores, dense for the noise."""

	path, values, dense = runs_fw
	reader = FigWigReader(path)
	kinds = reader._group_index(0)[0]
	assert set(((kinds[kinds > 0] & 7)).tolist()) == {4}
	noise = reader._group_index(1)[0]
	assert (noise[:4] & 7).tolist() == [1, 1, 1, 1]    # the last tile is partial
	length = len(values)
	rng = numpy.random.default_rng(6)
	starts = numpy.sort(rng.integers(0, length - 100, 300))
	starts = numpy.append(starts, [16384 - 50, 2 * 16384 - 1, length - 60])
	n = len(starts)
	expected = numpy.stack([numpy.stack([v[s:s + 100] if s + 100 <= length else
		numpy.append(v[s:], numpy.full(s + 100 - length, numpy.nan, numpy.float32))
		for s in starts]) for v in (values, dense)], axis=1)

	for n_jobs in (1, 3):
		got = reader.read('chr1', starts, 100, n_jobs=n_jobs)
		assert_identical(got, expected)
		# Accumulating paths: a label summing both tracks, and bins.
		both = reader.read('chr1', starts, 100, labels=[0, 0], n_jobs=n_jobs)
		assert numpy.allclose(both[:, 0], numpy.nansum(expected, axis=1).astype(
			numpy.float32) + 0 * numpy.where(numpy.isnan(expected).all(axis=1),
			numpy.nan, 0), equal_nan=True, atol=1e-5)
		binned = reader.read('chr1', starts, 100, bins=10, n_jobs=n_jobs)
		assert numpy.allclose(numpy.nan_to_num(binned), numpy.nansum(
			expected.reshape(n, 2, 10, 10), axis=3), atol=1e-5)
		sums = reader.sums('chr1', starts, numpy.minimum(starts + 100, length),
			n_jobs=n_jobs)
		assert numpy.allclose(sums, numpy.nansum(expected, axis=2), atol=1e-5)
	entries = reader.read('chr1', starts[:5], 100, sparse=True)
	rebuilt = numpy.zeros((5, 2, 100), dtype=numpy.float32)
	rebuilt[entries.rows, entries.channels, entries.offsets] = entries.values
	assert_identical(rebuilt, numpy.nan_to_num(expected[:5]))


def test_dense_chromosome_is_batched_for_every_thread(tmp_path, monkeypatch):
	"""A read of a chromosome of dense chunks plans at least as many
	batches as threads, each chunk in one batch, where one batch of
	_BATCH_BYTES held them all and the threads split its windows."""

	length = 600 * TILE
	rng = numpy.random.default_rng(7)
	values = rng.integers(1, 200, (2, length)).astype(numpy.float32)
	path = str(tmp_path / 'dense.fw')
	with FigWigWriter(path, {'chr1': length}, ['a', 'b'], tile=TILE,
			n_jobs=1) as writer:
		writer.write('chr1', [0], values[None])
	reader = FigWigReader(path)
	seen = []
	run = figwig.figwig.FigWigReader._run

	def recorded(self, tasks, n_threads, summed=None, copies=False):
		seen.append((tasks, n_threads))
		return run(self, tasks, n_threads, summed, copies)

	monkeypatch.setattr(figwig.figwig.FigWigReader, '_run', recorded)
	starts = numpy.arange(0, length - 200, 37)
	got = reader.read('chr1', starts, 200, n_jobs=8)
	numpy.testing.assert_array_equal(got, numpy.stack([values[:, s:s + 200] for
		s in starts]))
	tasks, n_threads = seen[-1]
	assert n_threads == 8 and len(tasks) >= 8
	tiles = numpy.concatenate([task.gtiles for task in tasks])
	assert len(tiles) == len(numpy.unique(tiles)) == 600
