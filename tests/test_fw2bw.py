# test_fw2bw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig fw2bw`, which writes a figwig file's tracks as
bigWigs, and for the round trip back through `figwig bw2fw`."""

import os
import sys
import subprocess

import numpy
import pytest

import figwig.fw2bw

from figwig import BigWigReader
from figwig import FigWigReader
from figwig import read_bigwig

from figwig.__main__ import main

from .test_figwig import CHROM_SIZES
from .test_figwig import reference_tracks
from .test_figwig import write_reference

ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
	MKL_NUM_THREADS="1")


@pytest.fixture(scope='session')
def fw(tmp_path_factory):
	path = str(tmp_path_factory.mktemp('fw') / 'reference.fw')
	write_reference(path, reference_tracks())
	return path


def expected_bigwig(ref, chrom, track):
	"""What a bigWig of this track reads as: the values, with NaN as 0 and
	float32 rounding."""

	values = numpy.nan_to_num(ref[chrom][track]).astype(numpy.float32)
	return values


def test_every_track_becomes_a_bigwig(tmp_path, fw):
	ref = reference_tracks()
	main(['fw2bw', fw, '-o', str(tmp_path / 'bw'), '-p', '2'])
	names = ['atac_plus', 'atac_minus', 'signal', 'negative', 'empty']
	assert sorted(os.listdir(tmp_path / 'bw')) == sorted([name + '.bw' for name
		in names] + ['reference.tsv'])

	for i, name in enumerate(names[:4]):
		path = str(tmp_path / 'bw' / (name + '.bw'))
		for chrom, length in CHROM_SIZES.items():
			y = read_bigwig(path, chrom, [0], length)[0]
			numpy.testing.assert_array_equal(y, expected_bigwig(ref, chrom, i))

	# A track without values gives a bigWig without data blocks, whose header
	# and chromosomes read, as pyBigWig writes one.
	assert BigWigReader(str(tmp_path / 'bw' / 'empty.bw')).chrom_sizes == \
		CHROM_SIZES

	with open(tmp_path / 'bw' / 'reference.tsv') as handle:
		lines = handle.read().splitlines()
	assert lines[0] == 'file\tname\tdtype\tassay\tstrand'
	assert lines[1] == 'atac_plus.bw\tatac_plus\tuint16\tATAC-seq\t+'
	assert lines[3] == 'signal.bw\tsignal\tfloat32\tDNase-seq\t'


def test_round_trip_through_bw2fw(tmp_path, fw):
	main(['fw2bw', fw, '-o', str(tmp_path / 'bw')])
	back = str(tmp_path / 'back.fw')
	bigwigs = [str(tmp_path / 'bw' / (name + '.bw')) for name in ('atac_plus',
		'atac_minus', 'signal', 'negative')]
	main(['bw2fw', '-o', back, '-m', str(tmp_path / 'bw' / 'reference.tsv'),
		'-t', '64', '-g', 'hg38'] + bigwigs)

	original, copy = FigWigReader(fw), FigWigReader(back)
	assert copy.chrom_sizes == original.chrom_sizes
	assert len(copy.tracks) == 4
	for a, b in zip(original.tracks, copy.tracks):
		assert a['name'] == b['name'] and a['dtype'] == b['dtype']
		assert a.get('assay') == b.get('assay')
		assert a.get('strand', '') == b.get('strand')

	for chrom, length in CHROM_SIZES.items():
		a = original.read(chrom, [0], length, tracks=[0, 1, 2, 3], dtype='float32')
		b = copy.read(chrom, [0], length, dtype='float32')
		numpy.testing.assert_array_equal(numpy.nan_to_num(a), b)


def test_tracks_and_select(tmp_path, fw):
	main(['fw2bw', fw, '-o', str(tmp_path / 'a'), '-t', 'signal', 'empty'])
	assert sorted(os.listdir(tmp_path / 'a')) == ['empty.bw', 'reference.tsv',
		'signal.bw']

	main(['fw2bw', fw, '-o', str(tmp_path / 'b'), '--select', 'assay=ATAC-seq'])
	assert sorted(os.listdir(tmp_path / 'b')) == ['atac_minus.bw',
		'atac_plus.bw', 'reference.tsv']

	main(['fw2bw', fw, '-o', str(tmp_path / 'c'), '--select', 'assay=ATAC-seq',
		'strand=-'])
	assert sorted(os.listdir(tmp_path / 'c')) == ['atac_minus.bw',
		'reference.tsv']

	with pytest.raises(SystemExit, match="not both"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '--select', 'a=b', '-t',
			'signal'])
	with pytest.raises(SystemExit, match="no track has"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '--select', 'assay=none'])
	with pytest.raises(SystemExit, match="KEY=VALUE"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '--select', 'assay'])
	with pytest.raises(SystemExit, match="not one of the file's tracks"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '-t', 'nope'])
	with pytest.raises(SystemExit, match="is not a file"):
		main(['fw2bw', str(tmp_path / 'nope.fw'), '-o', str(tmp_path / 'd')])


def test_zooms_and_level(tmp_path, fw):
	main(['fw2bw', fw, '-o', str(tmp_path / 'z'), '-t', 'atac_plus', '-z', '2',
		'-l', '1'])
	import pybigtools
	bw = pybigtools.open(str(tmp_path / 'z' / 'atac_plus.bw'))
	assert len(bw.zooms()) >= 1


def test_bigwig_name_replaces_separators():
	assert figwig.fw2bw.bigwig_name({'name': 'a/b'}) == 'a_b.bw'
	assert figwig.fw2bw.bigwig_name({'name': 'plain'}) == 'plain.bw'


def test_text():
	assert figwig.fw2bw._text('x') == 'x'
	assert figwig.fw2bw._text(3) == '3'
	assert figwig.fw2bw._text(True) == 'true'
	assert figwig.fw2bw._text(None) == ''


def test_command_line(tmp_path, fw):
	result = subprocess.run([sys.executable, '-m', 'figwig', 'fw2bw', fw, '-o',
		str(tmp_path / 'cli'), '-v'], env=ENV, capture_output=True, text=True,
		timeout=600)
	assert result.returncode == 0, result.stderr
	assert 'chrM' in result.stderr
	assert len(os.listdir(tmp_path / 'cli')) == 6
