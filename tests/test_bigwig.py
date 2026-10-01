# test_bigwig.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import re
import copy
import zlib
import pickle
import struct
import pathlib
import threading

import numpy
import pytest
import pybigtools

import figwig
import figwig._kernels

from figwig import BigWig
from figwig import read_windows

from .writers import write_bigwig
from .writers import write_raw_bigwig


def reference(path, chroms, starts, width):
	"""pybigtools 0.2.5's values() for each window, cast to float32."""

	bw = pybigtools.open(path)
	out = numpy.empty((len(starts), width), dtype=numpy.float32)
	scratch = numpy.empty(width, dtype=numpy.float64)
	names = [chroms] * len(starts) if isinstance(chroms, str) else chroms
	for j, (name, start) in enumerate(zip(names, starts)):
		bw.values(name, int(start), int(start) + width, arr=scratch)
		out[j] = scratch
	return out


def pickle_round_trip(obj):
	return pickle.loads(pickle.dumps(obj))


def assert_identical(a, b):
	"""The same NaN positions, and the same bits everywhere else."""

	assert a.shape == b.shape and a.dtype == b.dtype
	nan_a, nan_b = numpy.isnan(a), numpy.isnan(b)
	numpy.testing.assert_array_equal(nan_a, nan_b)
	numpy.testing.assert_array_equal(a.view(numpy.uint32)[~nan_a],
		b.view(numpy.uint32)[~nan_b])


def random_windows(rng, chroms, n, width):
	"""Windows that start inside their chromosome; some run past its end."""

	names = numpy.array(list(chroms))
	picked = rng.choice(names, size=n)
	lengths = numpy.array([chroms[name] for name in picked])
	return picked, rng.integers(0, lengths).astype(numpy.int64)


###
# Values
###

@pytest.mark.parametrize('fixture', ['dense_bw', 'sparse_bw', 'deep_bw'])
@pytest.mark.parametrize('width', [1, 7, 1000])
def test_read_matches_pybigtools(request, fixture, width, batching):
	path, chroms = request.getfixturevalue(fixture)
	rng = numpy.random.default_rng(width)
	names, starts = random_windows(rng, chroms, 3000, width)

	X = BigWig(path).read(names, starts, width)
	assert_identical(X, reference(path, list(names), starts, width))


def test_read_matches_pybigtools_at_chromosome_edges(dense_bw, batching):
	path, chroms = dense_bw
	names, starts = [], []
	for name, length in chroms.items():
		for start in [0, 1, length - 1000, length - 999, length - 1]:
			names.append(name)
			starts.append(max(0, start))

	X = BigWig(path).read(names, starts, 1000)
	assert_identical(X, reference(path, names, starts, 1000))


def test_past_chromosome_end_is_nan(dense_bw):
	path, chroms = dense_bw
	length = chroms['chrM']
	X = BigWig(path).read('chrM', [length - 10, length + 3], 20)

	assert not numpy.isnan(X[0, :10]).any()
	assert numpy.isnan(X[0, 10:]).all()
	assert numpy.isnan(X[1]).all()


def test_uncovered_bases_are_zero(tmp_path):
	path = tmp_path / 'gaps.bw'
	write_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, 2.5),
		('chr1', 500, 501, -1.0)])

	X = BigWig(path).read('chr1', [95, 495], 20)
	numpy.testing.assert_array_equal(X[0], [0] * 5 + [2.5] * 10 + [0] * 5)
	numpy.testing.assert_array_equal(X[1], [0] * 5 + [-1.0] + [0] * 14)


def test_special_values(tmp_path):
	path = tmp_path / 'special.bw'
	values = [numpy.nan, numpy.inf, -numpy.inf, -0.0, 1e-45, 3.4e38, -3.4e38]
	entries = [('chr1', 10 * i, 10 * i + 5, float(v)) for i, v in enumerate(values)]
	write_bigwig(path, {'chr1': 1000}, entries)

	starts = numpy.arange(0, 100, 3)
	X = BigWig(path).read('chr1', starts, 25)
	assert_identical(X, reference(str(path), 'chr1', starts, 25))

	# An interval whose value is NaN covers nothing, so its bases are 0.
	numpy.testing.assert_array_equal(BigWig(path).read('chr1', [0], 5)[0], 0)
	assert numpy.signbit(BigWig(path).read('chr1', [30], 1)[0, 0])


@pytest.mark.parametrize('kind', [1, 2, 3])
@pytest.mark.parametrize('compress', [True, False])
def test_section_types(tmp_path, kind, compress, batching):
	rng = numpy.random.default_rng(kind)
	chroms = {'chr1': 50_000, 'chr2': 20_000}
	sections = []
	for name, length in chroms.items():
		for s in range(1000, length - 2000, 3000):
			if kind == 1:
				items, position = [], s
				for _ in range(40):
					end = position + int(rng.integers(1, 30))
					items.append((position, end, float(rng.normal())))
					position = end + int(rng.integers(0, 20))
				sections.append((name, 1, 0, 0, items))
			elif kind == 2:
				span = int(rng.integers(1, 9))
				gaps = rng.integers(0, 5, size=40)
				positions = s + numpy.cumsum(span + gaps) - span - gaps[0]
				items = [(int(p), float(rng.normal())) for p in positions]
				sections.append((name, 2, 0, span, items))
			else:
				span = int(rng.integers(1, 6))
				step = span + int(rng.integers(0, 4))
				sections.append((name, 3, step, span, (s, [float(v) for v in
					rng.normal(size=60)])))

	path = str(tmp_path / 'sections.bw')
	write_raw_bigwig(path, chroms, sections, compress=compress)

	names, starts = random_windows(rng, chroms, 2000, 300)
	X = BigWig(path).read(names, starts, 300)
	assert_identical(X, reference(path, list(names), starts, 300))
	assert (X[~numpy.isnan(X)] != 0).any()


###
# Arguments
###

def test_n_jobs_does_not_change_output(sparse_bw, batching):
	path, chroms = sparse_bw
	names, starts = random_windows(numpy.random.default_rng(3), chroms, 5000, 500)
	bw = BigWig(path)

	X = bw.read(names, starts, 500, n_jobs=1)
	for n_jobs in [2, 3, 8, 16]:
		assert_identical(bw.read(names, starts, 500, n_jobs=n_jobs), X)


def test_order_and_repeats(dense_bw, batching):
	path, chroms = dense_bw
	rng = numpy.random.default_rng(4)
	names, starts = random_windows(rng, chroms, 1000, 200)
	names = numpy.concatenate([names, names[:300]])
	starts = numpy.concatenate([starts, starts[:300] + 1])

	bw = BigWig(path)
	X = bw.read(names, starts, 200)
	order = rng.permutation(len(starts))
	assert_identical(bw.read(names[order], starts[order], 200), X[order])


def test_chroms_forms(dense_bw):
	path, _ = dense_bw
	starts = numpy.array([5, 1000, 70_000])
	bw = BigWig(path)

	X = bw.read('chr2', starts, 50)
	assert_identical(bw.read(['chr2'] * 3, starts, 50), X)
	assert_identical(bw.read(numpy.array(['chr2'] * 3), starts, 50), X)
	assert_identical(bw.read(numpy.array(['chr2'] * 3, dtype=object), starts,
		50), X)
	assert_identical(bw.read('chr2', starts.tolist(), 50), X)
	assert_identical(bw.read('chr2', starts.astype(numpy.uint32), 50), X)


def test_out(dense_bw):
	path, _ = dense_bw
	bw = BigWig(path)
	out = numpy.full((2, 30), 7, dtype=numpy.float32)

	result = bw.read('chr1', [0, 100], 30, out=out)
	assert result is out
	assert_identical(out, bw.read('chr1', [0, 100], 30))

	for bad in [numpy.empty((3, 30), dtype=numpy.float32),
			numpy.empty((2, 30), dtype=numpy.float64),
			numpy.empty((30, 2), dtype=numpy.float32).T]:
		with pytest.raises(ValueError, match='out must be'):
			bw.read('chr1', [0, 100], 30, out=bad)


@pytest.mark.parametrize('chroms', [[], 'chr1'])
@pytest.mark.parametrize('starts', [numpy.empty(0, dtype=numpy.int64), [], ()],
	ids=['array', 'list', 'tuple'])
def test_no_windows(dense_bw, chroms, starts):
	"""An empty list of starts is float64 to numpy, and was refused as not
	being integers."""

	X = BigWig(dense_bw[0]).read(chroms, starts, 10)
	assert X.shape == (0, 10) and X.dtype == numpy.float32


@pytest.mark.parametrize('width', [0, -3])
def test_width_must_be_positive(dense_bw, width):
	with pytest.raises(ValueError, match='width must be at least 1'):
		BigWig(dense_bw[0]).read('chr1', [0], width)


@pytest.mark.parametrize('width', [1.5, True, '10'])
def test_width_must_be_an_integer(dense_bw, width):
	with pytest.raises(TypeError, match='width must be an integer'):
		BigWig(dense_bw[0]).read('chr1', [0], width)


@pytest.mark.parametrize('n_jobs', [0, -1])
def test_n_jobs_must_be_positive(dense_bw, n_jobs):
	with pytest.raises(ValueError, match='n_jobs must be at least 1'):
		BigWig(dense_bw[0]).read('chr1', [0], 10, n_jobs=n_jobs)


@pytest.mark.parametrize('n_jobs', [2.0, True, None])
def test_n_jobs_must_be_an_integer(dense_bw, n_jobs):
	with pytest.raises(TypeError, match='n_jobs must be an integer'):
		BigWig(dense_bw[0]).read('chr1', [0], 10, n_jobs=n_jobs)


def test_starts_must_be_integers(dense_bw):
	with pytest.raises(TypeError, match='starts must be integers'):
		BigWig(dense_bw[0]).read('chr1', [0.0, 10.5], 10)


def test_starts_must_be_one_dimensional(dense_bw):
	with pytest.raises(ValueError, match='one-dimensional'):
		BigWig(dense_bw[0]).read('chr1', [[0, 10]], 10)


def test_one_chrom_per_start(dense_bw):
	with pytest.raises(ValueError, match='one name per start'):
		BigWig(dense_bw[0]).read(['chr1', 'chr2'], [0, 10, 20], 10)


def test_missing_chromosome_raises(dense_bw):
	with pytest.raises(ValueError, match='Chromosomes not in .*: chr7, chrZ'):
		BigWig(dense_bw[0]).read(['chr1', 'chrZ', 'chr7'], [0, 0, 0], 10)


@pytest.mark.parametrize('start', [-1, 2**32 - 10])
def test_out_of_range_window_raises(dense_bw, start):
	with pytest.raises(ValueError, match='start before 0 or end past'):
		BigWig(dense_bw[0]).read('chr1', [0, start], 10)


def test_read_windows(sparse_bw):
	path, chroms = sparse_bw
	names, starts = random_windows(numpy.random.default_rng(5), chroms, 2000, 100)
	assert_identical(read_windows(path, names, starts, 100, n_jobs=3),
		BigWig(path).read(names, starts, 100))


def test_pathlike_and_attributes(dense_bw):
	path, chroms = dense_bw
	bw = BigWig(pathlib.Path(path))
	assert bw.path == path
	assert bw.chroms == chroms
	assert list(bw.chroms) == list(pybigtools.open(path).chroms())
	assert repr(bw) == "BigWig('{}', 3 chromosomes)".format(path)


def test_multi_level_chromosome_tree(tmp_path):
	"""A chromosome tree with a root over three leaves, as UCSC writes one for
	a genome with more than 256 chromosomes. Its leaves were read in reverse,
	so `chroms` came out in the wrong order.
	"""

	chroms = {'chr{:02d}'.format(i): 1000 + i for i in range(40)}
	sections = [_bedgraph(name, 100, [(0, 10, float(i))]) for i, name in
		enumerate(chroms)]
	path = str(tmp_path / 'tree.bw')
	write_raw_bigwig(path, chroms, sections, chrom_block_size=16)

	bw = BigWig(path)
	assert list(bw.chroms.items()) == list(chroms.items())
	assert list(bw.chroms) == list(pybigtools.open(path).chroms())

	names, starts = list(chroms), [95] * len(chroms)
	assert_identical(bw.read(names, starts, 20), reference(path, names, starts,
		20))


###
# Threads, processes and the zlib fallback
###

@pytest.mark.parametrize('duplicate', [pickle_round_trip, copy.deepcopy],
	ids=['pickle', 'deepcopy'])
def test_pickle(dense_bw, duplicate):
	"""A PyTorch DataLoader pickles its dataset into each worker under the
	spawn and forkserver start methods, and BigWig held a lock that cannot
	be pickled. Copies are taken before and after the index is read."""

	path, chroms = dense_bw
	names, starts = random_windows(numpy.random.default_rng(10), chroms, 500,
		100)
	bw = BigWig(path)
	before = duplicate(bw)
	X = bw.read(names, starts, 100)
	after = duplicate(bw)
	assert before._index is None and after._index is not None

	for other in [before, after]:
		assert other.path == bw.path and other.chroms == bw.chroms
		assert_identical(other.read(names, starts, 100, n_jobs=3), X)
		assert other._index_lock is not bw._index_lock


def test_concurrent_reads_on_one_object(deep_bw, batching):
	path, chroms = deep_bw
	bw = BigWig(path)
	rng = numpy.random.default_rng(6)
	jobs = [random_windows(rng, chroms, 3000, 1000) for _ in range(6)]
	expected = [bw.read(names, starts, 1000, n_jobs=1) for names, starts in jobs]

	results = [None] * len(jobs)

	def work(i):
		results[i] = bw.read(*jobs[i], 1000, n_jobs=4)

	threads = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	for result, X in zip(results, expected):
		assert_identical(result, X)


def test_zlib_decompress_fallback(sparse_bw, monkeypatch):
	path, chroms = sparse_bw
	names, starts = random_windows(numpy.random.default_rng(7), chroms, 3000, 400)
	X = BigWig(path).read(names, starts, 400)

	monkeypatch.setattr(figwig._kernels, '_ZLIB_UNCOMPRESS', [None])
	assert figwig._kernels._zlib_uncompress() is None
	assert_identical(BigWig(path).read(names, starts, 400, n_jobs=4), X)


###
# What figwig refuses to read
###

def _bedgraph(name, start, items):
	return (name, 1, 0, 0, [(start + s, start + e, v) for s, e, v in items])


@pytest.mark.parametrize('items, reason', [
	([(0, 10, 1.0), (5, 15, 2.0)], 'overlap'),
	([(20, 30, 1.0), (0, 10, 2.0)], 'unsorted'),
])
def test_bad_intervals_raise(tmp_path, items, reason):
	path = str(tmp_path / '{}.bw'.format(reason))
	write_raw_bigwig(path, {'chr1': 10_000}, [_bedgraph('chr1', 100, [(0, 5, 1.0)]),
		_bedgraph('chr1', 5000, items)])

	bw = BigWig(path)
	numpy.testing.assert_array_equal(bw.read('chr1', [100], 5)[0], 1.0)
	with pytest.raises(ValueError, match='1 windows overlap data blocks that '
			'figwig cannot read.*chr1:4990-5010'):
		bw.read('chr1', [100, 4990], 20)


def test_unknown_section_type_raises(tmp_path):
	path = str(tmp_path / 'kind.bw')
	block = zlib.compress(struct.pack('<IIIIIBBH', 0, 100, 110, 0, 0, 4, 0, 1) +
		struct.pack('<IIf', 100, 110, 1.0))
	write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, block)])

	with pytest.raises(ValueError, match='figwig cannot read'):
		BigWig(path).read('chr1', [95], 20)


def test_corrupt_block_raises(tmp_path):
	path = str(tmp_path / 'corrupt.bw')
	write_raw_bigwig(path, {'chr1': 1000}, [('chr1', 100, 110, b'not zlib')])

	with pytest.raises(ValueError, match='figwig cannot read'):
		BigWig(path).read('chr1', [95], 20)


def test_corrupt_files_raise_only_value_errors(tmp_path):
	"""Bytes of three small bigWigs overwritten at random, a few at a time,
	three times in four inside the header, the chromosome tree or the data
	index. bigWig has no checksums, so a corrupt value can read without
	error, but nothing other than a ValueError may escape.
	"""

	chroms = {'chr1': 20_000, 'chr2': 9_000}
	sections = []
	for name, length in chroms.items():
		for s in range(100, length - 2000, 1500):
			sections.append(_bedgraph(name, s, [(10 * i, 10 * i + 5, float(i))
				for i in range(50)]))
			sections.append((name, 2, 0, 3, [(s + 600 + 7 * i, float(i)) for i
				in range(40)]))
			sections.append((name, 3, 4, 2, (s + 1000, [float(i) for i in
				range(60)])))

	files = []
	for compress in [True, False]:
		path = tmp_path / 'raw_{}.bw'.format(compress)
		write_raw_bigwig(path, chroms, sections, compress=compress)
		files.append(path.read_bytes())

	path = tmp_path / 'pybigtools.bw'
	write_bigwig(path, chroms, [(name, s, s + 3, float(s)) for name in chroms
		for s in range(0, chroms[name] - 5, 7)])
	files.append(path.read_bytes())

	rng = numpy.random.default_rng(9)
	names = ['chr1'] * 40 + ['chr2'] * 20
	starts = list(range(0, 20_000, 500)) + list(range(0, 9_000, 450))
	path = tmp_path / 'corrupt.bw'
	outcomes = {'read': 0, 'ValueError': 0}
	for trial in range(600):
		data = bytearray(files[trial % 3])
		chrom_tree, _, data_index = struct.unpack('<QQQ', data[8:32])
		region = int(rng.choice([0, chrom_tree, data_index, -1]))
		for _ in range(int(rng.integers(1, 4))):
			if region == -1:
				position = int(rng.integers(0, len(data) - 8))
			else:
				position = min(region + int(rng.integers(0, 80)), len(data) - 8)

			if rng.random() < 0.5:
				data[position] = int(rng.integers(256))
			else:
				value = int(rng.choice([0, 1, 2 ** 31, 2 ** 32 - 1, 2 ** 63]))
				data[position:position + 8] = value.to_bytes(8, 'little')

		path.write_bytes(bytes(data))
		try:
			bw = BigWig(path)
			present = [name in bw.chroms for name in names]
			if any(present):
				bw.read([n for n, p in zip(names, present) if p], [s for s, p in
					zip(starts, present) if p], 500, n_jobs=1)
			outcomes['read'] += 1
		except ValueError:
			outcomes['ValueError'] += 1

	assert outcomes['read'] > 50 and outcomes['ValueError'] > 50


def test_unsorted_index_raises(tmp_path):
	path = str(tmp_path / 'index.bw')
	write_raw_bigwig(path, {'chr1': 10_000}, [_bedgraph('chr1', 5000, [(0, 5, 1.0)]),
		_bedgraph('chr1', 100, [(0, 5, 2.0)])])

	with pytest.raises(ValueError, match='its data blocks are unsorted'):
		BigWig(path).read('chr1', [0], 10)


def _fixedstep_block(chrom_id, start, values, end):
	"""A compressed block of one fixedStep section, with step and span 1,
	whose section header says it ends at `end`."""

	header = struct.pack('<IIIIIBBH', chrom_id, start, end, 1, 1, 3, 0,
		len(values))
	return zlib.compress(header + struct.pack('<{}f'.format(len(values)),
		*values))


def test_overlapping_index_entries(tmp_path):
	"""pyBigWig writes some fixedStep blocks with an index entry that ends 6
	bases past the block's last interval, so neighbouring entries can overlap
	even though no two intervals do. Such files were refused as having
	overlapping data blocks.
	"""

	rng = numpy.random.default_rng(8)
	sections, start = [], 100
	for _ in range(20):
		n = int(rng.integers(20, 60))
		values = rng.normal(size=n).astype(numpy.float32).tolist()
		sections.append(('chr1', start, start + n + 6, _fixedstep_block(0,
			start, values, start + n + 6)))
		start += n + int(rng.integers(1, 5))

	path = str(tmp_path / 'overlap.bw')
	write_raw_bigwig(path, {'chr1': 5000}, sections)

	starts = numpy.arange(0, start + 10, 7)
	X = BigWig(path).read('chr1', starts, 50)
	assert_identical(X, reference(path, 'chr1', starts, 50))
	assert (X != 0).mean() > 0.5


def test_intervals_overlapping_across_blocks_raise(tmp_path):
	"""Intervals in two blocks that overlap give a base two values, as two
	overlapping intervals in one block do, so a window over both raises."""

	path = str(tmp_path / 'across.bw')
	write_raw_bigwig(path, {'chr1': 10_000}, [
		_bedgraph('chr1', 100, [(0, 10, 1.0)]),
		_bedgraph('chr1', 1000, [(0, 20, 2.0)]),
		_bedgraph('chr1', 1015, [(0, 10, 3.0)]),
		_bedgraph('chr1', 3000, [(0, 10, 4.0)])])

	bw = BigWig(path)
	X = bw.read('chr1', [95, 1005, 2995], 10)
	numpy.testing.assert_array_equal(X[:, 5:], [[1.0] * 5, [2.0] * 5, [4.0] * 5])

	with pytest.raises(ValueError, match='1 windows overlap data blocks that '
			'figwig cannot read.*chr1:1010-1020'):
		bw.read('chr1', [95, 1010, 2995], 10)


@pytest.mark.parametrize('header, match', [
	(b'', 'is not a bigWig file'),
	(b'\0' * 64, 'is not a bigWig file'),
	(struct.pack('<I', 0x8789F2EB) + b'\0' * 60, 'is a bigBed file'),
	(struct.pack('>I', 0x888FFC26) + b'\0' * 60, 'is a big-endian file'),
])
def test_not_a_bigwig_raises(tmp_path, header, match):
	path = tmp_path / 'bad.bw'
	path.write_bytes(header)
	with pytest.raises(ValueError, match=match):
		BigWig(path)


def test_missing_file_raises(tmp_path):
	with pytest.raises(FileNotFoundError):
		BigWig(tmp_path / 'absent.bw')


def test_big_endian_machine_raises(dense_bw, monkeypatch):
	monkeypatch.setattr(figwig.bigwig.sys, 'byteorder', 'big')
	with pytest.raises(ValueError, match='only on little-endian machines'):
		BigWig(dense_bw[0])


def _two_chrom_bigwig(tmp_path):
	"""A raw bigWig with one bedGraph block on each of two chromosomes.

	Returns its path, and the offsets of its chromosome tree and data index.
	The chromosome tree's root, a leaf, starts 32 bytes after the tree, and
	the data index's root, also a leaf, 48 bytes after the index.
	"""

	path = tmp_path / 'two.bw'
	write_raw_bigwig(path, {'chr1': 1000, 'chr2': 1000}, [
		_bedgraph('chr1', 100, [(0, 10, 1.0)]),
		_bedgraph('chr2', 100, [(0, 10, 2.0)])])

	with open(path, 'rb') as handle:
		chrom_tree, _, data_index = struct.unpack('<QQQ', handle.read(64)[8:32])
	return path, chrom_tree, data_index


def _patch(path, offset, fmt, *values):
	with open(path, 'r+b') as handle:
		handle.seek(offset)
		handle.write(struct.pack(fmt, *values))


@pytest.mark.parametrize('field, match', [
	('offset', r'chromosome tree of .* cannot be read\.$'),
	('magic', r'chromosome tree of .* cannot be read\.$'),
	('value_size', r'chromosome tree of .* cannot be read\.$'),
	('cycle', 'chromosome tree of .* cannot be read: it has a cycle'),
])
def test_bad_chromosome_tree_raises(tmp_path, field, match):
	path, chrom_tree, _ = _two_chrom_bigwig(tmp_path)
	if field == 'offset':
		_patch(path, 8, '<Q', path.stat().st_size + 100)
	elif field == 'magic':
		_patch(path, chrom_tree, '<I', 0)
	elif field == 'value_size':
		_patch(path, chrom_tree + 12, '<I', 4)
	else:
		# The root becomes a non-leaf node whose one child is itself. The key
		# size is 4, so the child's offset follows a 4-byte key.
		_patch(path, chrom_tree + 32, '<BBH', 0, 0, 1)
		_patch(path, chrom_tree + 40, '<Q', chrom_tree + 32)

	with pytest.raises(ValueError, match=match):
		BigWig(path)


@pytest.mark.parametrize('field, match', [
	('offset', 'data index of .* cannot be read: unpack'),
	('magic', r'data index of .* cannot be read\.$'),
	('cycle', 'data index of .* cannot be read: it has a cycle'),
	('count', 'data index of .* cannot be read: the file ends inside it'),
	('two_chroms', 'a data block spans two chromosomes'),
])
def test_bad_data_index_raises(tmp_path, field, match):
	path, _, data_index = _two_chrom_bigwig(tmp_path)
	root = data_index + 48
	if field == 'offset':
		_patch(path, 24, '<Q', path.stat().st_size + 100)
	elif field == 'magic':
		_patch(path, data_index, '<I', 0)
	elif field == 'cycle':
		# A non-leaf entry is 16 bytes of range, then its child's offset.
		_patch(path, root, '<BBH', 0, 0, 1)
		_patch(path, root + 4 + 16, '<Q', root)
	elif field == 'count':
		_patch(path, root + 2, '<H', 65535)
	else:
		_patch(path, root + 4 + 8, '<I', 1)

	bw = BigWig(path)
	with pytest.raises(ValueError, match=match):
		bw.read('chr1', [0], 10)


@pytest.mark.parametrize('field, value', [
	('size', 2 ** 40),
	('size', 2 ** 63 + 5),
	('offset', 2 ** 40),
	('offset', 2 ** 64 - 1),
])
def test_data_index_past_end_of_file_raises(tmp_path, field, value):
	"""A block size or offset in the data index that points past the end of
	the file was read as given: a huge size was allocated, raising
	MemoryError, and an offset of 2**63 or more became negative and raised
	OSError."""

	path, _, data_index = _two_chrom_bigwig(tmp_path)
	entry = data_index + 48 + 4
	_patch(path, entry + (24 if field == 'size' else 16), '<Q', value)

	bw = BigWig(path)
	with pytest.raises(ValueError, match='data index of .* cannot be read: a '
			'data block lies past the end of the file'):
		bw.read('chr1', [95], 10)


@pytest.mark.parametrize('field, value', [
	('key_size', 2 ** 31),
	('count', 65535),
])
def test_chromosome_tree_past_end_of_file_raises(tmp_path, field, value):
	"""A chromosome tree node longer than the file was read as given: a huge
	key size was allocated, and a node cut short raised struct.error."""

	path, chrom_tree, _ = _two_chrom_bigwig(tmp_path)
	if field == 'key_size':
		_patch(path, chrom_tree + 8, '<I', value)
	else:
		_patch(path, chrom_tree + 34, '<H', value)

	with pytest.raises(ValueError, match='chromosome tree of .* cannot be '
			'read: the file ends inside it'):
		BigWig(path)


def test_version():
	assert re.fullmatch(r'\d+\.\d+\.\d+', figwig.__version__)
