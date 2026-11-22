# test_pipeline.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The whole path from a fragments file to the arrays a data loader reads,
by every route the commands offer, against counts made with numpy from the
fragments themselves.

Each command has tests of its own against the command before it; these
check what the commands hand each other: bigWigs written by `bam2bw` read
by `read_bigwig` and converted by `bw2fw`, per-cell tracks written by
`bam2fw --barcode-column` converted back by `fw2bw`, chromosomes listed in
the sizes file in an order that is not sorted, a cell with no fragments on
one chromosome, and windows of a model's width that run past a
chromosome's end.
"""

import os
import sys
import warnings

import numpy
import pytest

if sys.platform == 'win32':
	pytest.skip("pysam and pyBigWig do not build on Windows",
		allow_module_level=True)

import pysam  # noqa: E402
import pyBigWig  # noqa: E402

import figwig.bam2bw  # noqa: E402

from figwig import FigWigReader  # noqa: E402
from figwig import read_bigwig  # noqa: E402
from figwig import read_figwig  # noqa: E402
from figwig.__main__ import main  # noqa: E402

from .reads import write_chrom_sizes  # noqa: E402
from .reads import write_intervals  # noqa: E402


# In the sizes file's order, which is not the sorted order the fragments
# file is in.
CHROM_SIZES = [('chr2', 61_003), ('chr10', 40_000), ('chr1', 150_001),
	('chrX', 23_457)]
LENGTHS = dict(CHROM_SIZES)
BARCODES = ['AAAC-1', 'CCGT-1', 'GGTA-1', 'TTAC-1']
# TTAC-1 has no fragments on chrX.
ABSENT = ('TTAC-1', 'chrX')
WIDTH = 2114


def _fragments(rng, n=20_000):
	"""Random fragments as (chrom, start, end, barcode, 1) rows, plus some at
	each chromosome's first and last base, 1 bp long, and repeated."""

	names = [name for name, _ in CHROM_SIZES]
	lengths = numpy.array([length for _, length in CHROM_SIZES])
	rows = []
	for _ in range(n):
		chrom = names[rng.choice(len(names), p=lengths / lengths.sum())]
		barcode = BARCODES[rng.integers(0, len(BARCODES))]
		if (barcode, chrom) == ABSENT:
			continue
		length = int(rng.integers(1, 700))
		start = int(rng.integers(0, LENGTHS[chrom] - 1))
		rows.append((chrom, start, min(start + length, LENGTHS[chrom]), barcode, 1))

	for chrom, length in CHROM_SIZES:
		for barcode in BARCODES[:2]:
			rows += [(chrom, 0, 50, barcode, 1), (chrom, 0, 1, barcode, 1),
				(chrom, length - 40, length, barcode, 1), (chrom, length - 1, length,
				barcode, 1)]
	rows += [('chr1', 5000, 5100, 'GGTA-1', 1)] * 3

	# Sorted as a Cell Ranger fragments file is: chr1, chr10, chr2, chrX.
	return sorted(rows, key=lambda row: (row[0], row[1], row[2]))


def _expected(rows, coverage):
	"""{barcode: {chrom: float64 counts}}: both cut sites of each fragment,
	at its start and at its last base, or with `coverage` every base from its
	start to its end."""

	counts = {barcode: {chrom: numpy.zeros(length) for chrom, length in
		CHROM_SIZES} for barcode in BARCODES}
	for chrom, start, end, barcode, _ in rows:
		y = counts[barcode][chrom]
		if coverage:
			y[start:end] += 1
		else:
			y[start] += 1
			y[end - 1] += 1
	return counts


def _windows(rng, n=400):
	"""Windows of a model's width: random ones, some running past their
	chromosome's end, and one at each chromosome's first and last base."""

	names = numpy.array([name for name, _ in CHROM_SIZES])
	lengths = numpy.array([length for _, length in CHROM_SIZES])
	picked = rng.choice(len(names), size=n, p=lengths / lengths.sum())
	chroms = list(names[picked]) + list(names) + list(names)
	starts = list(rng.integers(0, lengths[picked])) + [0] * len(names) + list(
		lengths - 1)
	return numpy.array(chroms), numpy.array(starts, dtype=numpy.int64)


def _window_values(counts, chroms, starts, barcodes):
	"""(n, len(barcodes), WIDTH) float32, NaN past a chromosome's end."""

	out = numpy.full((len(starts), len(barcodes), WIDTH), numpy.nan,
		dtype=numpy.float32)
	for i, (chrom, start) in enumerate(zip(chroms, starts)):
		stop = min(int(start) + WIDTH, LENGTHS[chrom])
		for j, barcode in enumerate(barcodes):
			out[i, j, :stop - start] = counts[barcode][chrom][start:stop]
	return out


def assert_identical(a, b):
	assert a.shape == b.shape and a.dtype == b.dtype
	nan_a, nan_b = numpy.isnan(a), numpy.isnan(b)
	numpy.testing.assert_array_equal(nan_a, nan_b)
	numpy.testing.assert_array_equal(a[~nan_a], b[~nan_b])


def _read_bigwigs(paths, chroms, starts):
	"""`read_bigwig` over the files on threads. Every bigWig keeps every
	chromosome of the sizes file, TTAC-1's chrX too, so no window warns."""

	return read_bigwig(paths, chroms, starts, WIDTH, n_jobs=4)


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
	"""The sizes file, one BGZF fragments file of every cell, and one plain
	fragments file per cell, with the fragments they were written from."""

	root = tmp_path_factory.mktemp('pipeline')
	rows = _fragments(numpy.random.default_rng(31))
	sizes = str(write_chrom_sizes(root / 'genome.sizes', CHROM_SIZES))
	plain = str(write_intervals(root / 'fragments.tsv', rows))
	fragments = str(root / 'fragments.tsv.gz')
	pysam.tabix_compress(plain, fragments)
	cells = {}
	for barcode in BARCODES:
		cells[barcode] = str(write_intervals(root / (barcode + '.tsv'), [row for
			row in rows if row[3] == barcode]))
	return sizes, fragments, cells, rows


def _bam2bw(directory, filename, sizes, flags):
	os.makedirs(directory, exist_ok=True)
	name = os.path.join(directory, os.path.basename(filename).split('.')[0])
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		figwig.bam2bw.main([filename, '-s', sizes, '-n', name, '-u', '-p', '2'] +
			flags, prog='figwig bam2bw')
	return name + '.bw'


@pytest.mark.parametrize('coverage', [False, True], ids=['cuts', 'coverage'])
def test_fragments_reach_a_loader_by_every_path(tmp_path, inputs, coverage):
	sizes, fragments, cells, rows = inputs
	counts = _expected(rows, coverage)
	flags = ['-c'] if coverage else ['-f']
	chroms, starts = _windows(numpy.random.default_rng(32))
	expected = _window_values(counts, chroms, starts, BARCODES)

	# 1. bam2bw on each cell's fragments, read together as a loader reads a
	# list of bigWigs: channel i is cell i.
	bigwigs = [_bam2bw(tmp_path / 'bw', cells[barcode], sizes, flags) for
		barcode in BARCODES]
	assert_identical(_read_bigwigs(bigwigs, chroms, starts), expected)

	# The same bigWigs through pyBigWig, a reader figwig did not write.
	for barcode, path in zip(BARCODES[:2], bigwigs[:2]):
		bw = pyBigWig.open(path)
		try:
			for chrom, length in CHROM_SIZES:
				y = numpy.nan_to_num(bw.values(chrom, 0, length, numpy=True))
				numpy.testing.assert_array_equal(y, counts[barcode][chrom].astype(
					numpy.float32))
		finally:
			bw.close()

	# 2. Those bigWigs in one figwig file.
	converted = str(tmp_path / 'converted.fw')
	main(['bw2fw', '-o', converted, '-s', sizes, '-t', '1024', '-p', '4'] +
		bigwigs)
	reader = FigWigReader(converted)
	assert [track['name'] for track in reader.tracks] == BARCODES
	assert reader.chrom_sizes == LENGTHS
	assert all(numpy.dtype(track['dtype']).kind == 'u' for track in reader.tracks)
	assert_identical(read_figwig(converted, chroms, starts, WIDTH,
		dtype='float32', n_jobs=4), expected)

	# 3. bam2fw on each cell's file: a track per file, in the order given.
	per_file = str(tmp_path / 'per_file')
	main(['bam2fw'] + [cells[barcode] for barcode in BARCODES] + ['-s', sizes,
		'-n', per_file, '-u', '-p', '4', '-t', '1024'] + flags)
	reader = FigWigReader(per_file + '.fw')
	assert [track['name'] for track in reader.tracks] == BARCODES
	assert_identical(reader.read(chroms, starts, WIDTH, dtype='float32',
		n_jobs=4), expected)
	from_figwig = per_file + '.fw'

	# 4. bam2fw on the one fragments file of every cell, a track per barcode
	# in order of first appearance; read here by name in another order.
	if not coverage:
		per_cell = str(tmp_path / 'per_cell')
		main(['bam2fw', fragments, '-s', sizes, '-n', per_cell, '--barcode-column',
			'4', '-p', '4', '-t', '1024'] + flags)
		reader = FigWigReader(per_cell + '.fw')
		assert sorted(track['name'] for track in reader.tracks) == BARCODES
		order = BARCODES[::-1]
		assert_identical(reader.read(chroms, starts, WIDTH, tracks=order,
			dtype='float32', n_jobs=4), expected[:, ::-1])
		from_figwig = per_cell + '.fw'

	# 5. The figwig file of 4 (or 3, for coverage) back into bigWigs, read
	# as the bigWigs of 1 are.
	back = tmp_path / 'back'
	main(['fw2bw', from_figwig, '-o', str(back), '-p', '2'])
	assert_identical(_read_bigwigs([str(back / (barcode + '.bw')) for barcode in
		BARCODES], chroms, starts), expected)
