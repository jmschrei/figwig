# test_bw2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig bw2fw`, which writes bigWigs as the tracks of one
figwig file. The oracle is `read_bigwig` with `missing=0.0` on the same
windows."""

import os
import sys
import subprocess

import numpy
import pytest

import figwig.bw2fw

from figwig import FigWigReader
from figwig import read_bigwig
from figwig import read_figwig
from figwig import write_bigwig

from figwig.__main__ import main

ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
	MKL_NUM_THREADS="1")
CHROM_SIZES = {'chr1': 300_000, 'chr2': 120_000, 'chrM': 16_569}


def assert_identical(a, b):
	assert a.shape == b.shape and a.dtype == b.dtype
	nan_a, nan_b = numpy.isnan(a), numpy.isnan(b)
	numpy.testing.assert_array_equal(nan_a, nan_b)
	numpy.testing.assert_array_equal(a[~nan_a], b[~nan_b])


@pytest.fixture(scope='session')
def bigwigs(tmp_path_factory):
	"""Three bigWigs on the same chromosomes: per-base counts, a float signal
	in intervals, and a dense track of windows."""

	rng = numpy.random.default_rng(10)
	directory = tmp_path_factory.mktemp('bw')
	paths = [str(directory / name) for name in ('counts.bw', 'signal.bigWig',
		'dense.bigwig')]

	chroms, starts, values = [], [], []
	for chrom, length in CHROM_SIZES.items():
		positions = numpy.sort(rng.choice(length, length // 200, replace=False))
		chroms += [chrom] * len(positions)
		starts.append(positions)
		values.append(rng.integers(1, 300, len(positions)))
	write_bigwig(paths[0], CHROM_SIZES, chroms, numpy.concatenate(starts),
		numpy.concatenate(values)[:, None].astype(numpy.float32), zooms=0)

	chroms, starts, ends, values = [], [], [], []
	for chrom, length in CHROM_SIZES.items():
		bounds = numpy.sort(rng.choice(length, 400, replace=False))
		chroms += [chrom] * (len(bounds) - 1)
		starts.append(bounds[:-1])
		ends.append(bounds[1:])
		values.append(rng.normal(0, 2, len(bounds) - 1))
	write_bigwig(paths[1], CHROM_SIZES, chroms, numpy.concatenate(starts),
		numpy.concatenate(values).astype(numpy.float32),
		ends=numpy.concatenate(ends), zooms=0)

	chroms = ['chr1'] * 3 + ['chr2']
	starts = [0, 70_000, 299_000, 50_000]
	windows = rng.integers(0, 70000, (4, 1000)).astype(numpy.float32)
	windows[2, 300:] = 0
	write_bigwig(paths[2], CHROM_SIZES, chroms, starts, windows, zooms=0)
	return paths


def random_windows(rng, n, width):
	chroms = rng.choice(list(CHROM_SIZES), size=n)
	lengths = numpy.array([CHROM_SIZES[chrom] for chrom in chroms])
	starts = rng.integers(0, lengths)
	return chroms, starts.astype(numpy.int64)


def test_tracks_read_as_the_bigwigs_do(tmp_path, bigwigs):
	out = str(tmp_path / 'out.fw')
	main(['bw2fw', '-o', out, '-t', '1024', '-p', '2'] + bigwigs)

	reader = FigWigReader(out)
	assert [track['name'] for track in reader.tracks] == ['counts', 'signal',
		'dense']
	assert [track['dtype'] for track in reader.tracks] == ['uint16', 'float32',
		'uint32']
	assert reader.chrom_sizes == CHROM_SIZES
	assert reader.tile == 1024

	rng = numpy.random.default_rng(11)
	chroms, starts = random_windows(rng, 300, 500)
	expected = read_bigwig(bigwigs, chroms, starts, 500)
	assert_identical(read_figwig(out, chroms, starts, 500, dtype='float32'),
		expected)
	for chrom, length in CHROM_SIZES.items():
		assert_identical(reader.read(chrom, [0], length, dtype='float32'),
			read_bigwig(bigwigs, chrom, [0], length))


def test_metadata_table_names_and_describes_tracks(tmp_path, bigwigs):
	table = tmp_path / 'meta.tsv'
	table.write_text("file\tname\tassay\tstrand\tdtype\n"
		"counts.bw\tK562 ATAC\tATAC-seq\t+\tuint32\n"
		"{}\t\tDNase-seq\t\t\n"
		"dense.bigwig\tdense\tpredictions\t-\t\n".format(bigwigs[1]))
	out = str(tmp_path / 'out.fw')
	main(['bw2fw', '-o', out, '-m', str(table), '-g', 'hg38', '--groups', '1',
		'-l', '1'] + bigwigs)

	reader = FigWigReader(out)
	assert reader.genome == 'hg38'
	assert [track['name'] for track in reader.tracks] == ['K562 ATAC', 'signal',
		'dense']
	assert [track['dtype'] for track in reader.tracks] == ['uint32', 'float32',
		'uint32']
	assert [track['group'] for track in reader.tracks] == [0, 1, 2]
	assert reader.tracks[0]['assay'] == 'ATAC-seq' and reader.tracks[0][
		'strand'] == '+'
	assert reader.tracks[1]['assay'] == 'DNase-seq' and 'strand' not in \
		reader.tracks[1]
	assert 'file' not in reader.tracks[0]


def test_csv_table(tmp_path, bigwigs):
	table = tmp_path / 'meta.csv'
	table.write_text("path,cell\ncounts.bw,K562\nsignal.bigWig,HepG2\n")
	out = str(tmp_path / 'out.fw')
	main(['bw2fw', '-o', out, '-m', str(table)] + bigwigs[:2])
	assert [track['cell'] for track in FigWigReader(out).tracks] == ['K562',
		'HepG2']


def test_table_without_a_row_for_an_input_fails(tmp_path, bigwigs):
	table = tmp_path / 'meta.tsv'
	table.write_text("file\tcell\ncounts.bw\tK562\n")
	with pytest.raises(SystemExit, match="has no row for"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), '-m', str(table)] +
			bigwigs[:2])

	table.write_text("file\tcell\ncounts.bw\tK562\textra\n")
	with pytest.raises(SystemExit, match="expected 2 columns"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), '-m', str(table)] +
			bigwigs[:1])

	table.write_text("")
	with pytest.raises(SystemExit, match="needs a header row"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), '-m', str(table)] +
			bigwigs[:1])

	table.write_text("file\tfile\ncounts.bw\tK562\n")
	with pytest.raises(SystemExit, match="name each column once"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), '-m', str(table)] +
			bigwigs[:1])


def test_sizes_file_sets_the_chromosomes(tmp_path, bigwigs):
	sizes = tmp_path / 'sizes.txt'
	sizes.write_text("chr2\t120000\nchrZ\t5000\n")
	out = str(tmp_path / 'out.fw')
	with pytest.warns(UserWarning, match="not in the sizes file"):
		main(['bw2fw', '-o', out, '-s', str(sizes)] + bigwigs[:1])
	reader = FigWigReader(out)
	assert reader.chrom_sizes == {'chr2': 120_000, 'chrZ': 5000}
	assert_identical(reader.read('chr2', [0], 120_000, dtype='float32'),
		read_bigwig(bigwigs[0], 'chr2', [0], 120_000)[:, None])
	numpy.testing.assert_array_equal(reader.read('chrZ', [0], 5000), 0)


def test_bigwigs_must_agree_on_lengths(tmp_path, bigwigs, dense_bw):
	with pytest.raises(SystemExit, match="length of"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), bigwigs[0], dense_bw[0]])


def test_declared_dtype(tmp_path, bigwigs):
	out = str(tmp_path / 'out.fw')
	main(['bw2fw', '-o', out, '--dtype', 'uint16', bigwigs[0]])
	assert FigWigReader(out).tracks[0]['dtype'] == 'uint16'
	with pytest.raises(SystemExit, match="not an integer"):
		main(['bw2fw', '-o', out, '-f', '--dtype', 'uint16', bigwigs[1]])
	assert FigWigReader(out).tracks[0]['dtype'] == 'uint16'


def test_existing_output_is_refused_without_force(tmp_path, bigwigs):
	out = str(tmp_path / 'out.fw')
	main(['bw2fw', '-o', out] + bigwigs[:2])
	with pytest.raises(SystemExit, match="exists; --append adds tracks to it, "
			"and -f writes over it"):
		main(['bw2fw', '-o', out, bigwigs[2]])
	assert len(FigWigReader(out).tracks) == 2
	main(['bw2fw', '-o', out, '-f', bigwigs[2]])
	assert len(FigWigReader(out).tracks) == 1
	with pytest.raises(SystemExit, match="-p"):
		main(['bw2fw', '-o', out, '-f', '-p', '0', bigwigs[2]])


def test_chromosomes_absent_from_a_bigwig_are_zero(tmp_path, bigwigs):
	"""A bigWig without a chromosome that another has gives zeros there,
	without a warning escaping."""

	short = str(tmp_path / 'short.bw')
	write_bigwig(short, {'chr1': 300_000}, 'chr1', [10], [[3.0]], zooms=0)
	out = str(tmp_path / 'out.fw')
	main(['bw2fw', '-o', out, bigwigs[0], short])
	reader = FigWigReader(out)
	assert reader.chrom_sizes == CHROM_SIZES
	numpy.testing.assert_array_equal(reader.read('chr2', [0], 1000,
		tracks='short'), 0)
	assert reader.read('chr1', [10], 1, tracks='short')[0, 0] == 3


def test_missing_input_fails(tmp_path):
	with pytest.raises(SystemExit, match="is not a file"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), str(tmp_path / 'no.bw')])


def test_bad_arguments_fail_cleanly(tmp_path, bigwigs):
	with pytest.raises(SystemExit, match="power of two"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), '-t', '100', bigwigs[0]])
	with pytest.raises(SystemExit, match="level must be"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), '-l', '99', bigwigs[0]])


def test_helpers(bigwigs):
	assert figwig.bw2fw._track_name('/a/b/x.bw') == 'x'
	assert figwig.bw2fw._track_name('x.bigWig') == 'x'
	assert figwig.bw2fw._track_name('x.txt') == 'x.txt'
	tracks = figwig.bw2fw._tracks_from_table(bigwigs, None, 'uint8')
	assert tracks == [{'name': 'counts', 'dtype': 'uint8'}, {'name': 'signal',
		'dtype': 'uint8'}, {'name': 'dense', 'dtype': 'uint8'}]


def test_command_line(tmp_path, bigwigs):
	out = str(tmp_path / 'out.fw')
	result = subprocess.run([sys.executable, '-m', 'figwig', 'bw2fw', '-o', out,
		'-v'] + bigwigs, env=ENV, capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert 'group 1 of 1' in result.stderr
	assert len(FigWigReader(out).tracks) == 3


def test_sizes_must_agree_with_the_bigwigs(tmp_path, bigwigs):
	sizes = tmp_path / 'sizes.txt'
	sizes.write_text("chr1\t5000\nchr2\t130000\nchrM\t70\n")
	with pytest.raises(SystemExit, match="on one assembly"):
		main(['bw2fw', '-o', str(tmp_path / 'out.fw'), '-s', str(sizes)] +
			bigwigs[:1])


def test_nan_intervals_are_kept(tmp_path):
	"""An interval whose value is NaN is NaN in the figwig file, and with
	--missing nan the bases no interval covers are NaN too."""

	from .writers import write_bigwig
	bw = tmp_path / 'n.bw'
	write_bigwig(bw, {'chr1': 300}, [('chr1', 10, 20, float('nan')), ('chr1',
		100, 110, 2.0), ('chr1', 200, 210, 0.0)])
	bw = str(bw)
	out = str(tmp_path / 'n.fw')
	main(['bw2fw', '-o', out, bw])
	y = FigWigReader(out).read('chr1', [0], 300, tracks=0)[0]
	assert numpy.isnan(y[10:20]).all() and y[100] == 2 and y[0] == 0
	assert FigWigReader(out).tracks[0]['dtype'] == 'float32'
	main(['bw2fw', '-o', out, '-f', '--missing', 'nan', bw])
	y = FigWigReader(out).read('chr1', [0], 300, tracks=0)[0]
	assert numpy.isnan(y[:100]).all() and y[200] == 0 and y[100] == 2


def test_missing_is_recorded_for_fw2bw(tmp_path, bigwigs):
	"""The file's `missing` attribute says what bw2fw wrote for uncovered
	bases, which fw2bw follows by default."""

	out = str(tmp_path / 'm.fw')
	main(['bw2fw', '-o', out, bigwigs[0]])
	assert FigWigReader(out).attributes['missing'] == '0.0'
	main(['bw2fw', '-o', out, '-f', '--missing', 'nan', bigwigs[0]])
	assert FigWigReader(out).attributes['missing'] == 'nan'
	main(['bw2fw', '-o', out, '-f', '--missing', '-1', bigwigs[0]])
	assert FigWigReader(out).attributes['missing'] == '-1.0'


def test_append_adds_bigwigs_to_a_file(tmp_path, bigwigs):
	out = str(tmp_path / 'grow.fw')
	main(['bw2fw', '-o', out] + bigwigs[:2])
	main(['bw2fw', '-o', out, '--append'] + bigwigs[2:])
	reader = FigWigReader(out)
	assert len(reader.tracks) == 3 and [t['group'] for t in reader.tracks] == \
		[0, 0, 1]
	for i, bigwig in enumerate(bigwigs):
		assert_identical(reader.read('chr1', [0], 5000, tracks=i,
			dtype='float32'), read_bigwig(bigwig, 'chr1', [0], 5000))
	with pytest.raises(SystemExit, match="cannot be given with --append"):
		main(['bw2fw', '-o', out, '--append', '-t', '64', bigwigs[0]])
	with pytest.raises(SystemExit, match="does not exist"):
		main(['bw2fw', '-o', str(tmp_path / 'no.fw'), '--append', bigwigs[0]])


def test_missing_inputs_are_messages(tmp_path, bigwigs):
	out = str(tmp_path / 'out.fw')
	with pytest.raises(SystemExit, match="No such file|not a file|cannot"):
		main(['bw2fw', '-o', out, '-s', str(tmp_path / 'none.sizes'), bigwigs[0]])
	with pytest.raises(SystemExit, match="No such file|not a file|cannot"):
		main(['bw2fw', '-o', out, '-m', str(tmp_path / 'none.tsv'), bigwigs[0]])
	table = tmp_path / 'spaces.tsv'
	table.write_text("file name assay\n{} a ATAC\n".format(bigwigs[0]))
	with pytest.raises(SystemExit, match="one column; a table is tab-separated"):
		main(['bw2fw', '-o', out, '-m', str(table), bigwigs[0]])
