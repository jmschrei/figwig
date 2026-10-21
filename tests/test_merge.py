# test_merge.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig merge`, which writes the tracks of several figwig files
into one by copying their chunks."""

import os

import numpy
import pytest

from figwig import FigWigReader
from figwig import write_figwig

from figwig.__main__ import main

from .test_figwig import CHROM_SIZES
from .test_figwig import reference_tracks
from .test_figwig import write_reference


@pytest.fixture(scope='session')
def parts(tmp_path_factory):
	"""Two files of the reference tracks' values and a third of one more
	track, written with the same chromosomes and tile."""

	directory = tmp_path_factory.mktemp('merge')
	ref = reference_tracks()
	first = str(directory / 'first.fw')
	write_reference(first, ref)
	second = str(directory / 'second.fw')
	write_figwig(second, CHROM_SIZES, [{'name': 'extra', 'assay': 'x'}], 'chr2',
		[100], numpy.full((1, 1, 50), 2.0), tile=64)
	return ref, first, second


def test_merge_copies_every_track(tmp_path, parts):
	ref, first, second = parts
	out = str(tmp_path / 'all.fw')
	main(['merge', '-o', out, '-v', first, second])
	reader = FigWigReader(out, verify=True)
	assert [t['name'] for t in reader.tracks][-1] == 'extra'
	assert len(reader.tracks) == 6 and reader.tracks[5]['assay'] == 'x'
	assert reader.genome == 'hg38' and reader.tile == 64
	y = reader.read('chr2', [100], 50, dtype='float64')[0]
	numpy.testing.assert_array_equal(y[5], 2)
	numpy.testing.assert_array_equal(y[:5], numpy.nan_to_num(ref['chr2'][:,
		100:150]))
	for field in reader.stats.dtype.names:
		numpy.testing.assert_array_equal(reader.stats[field][:5], FigWigReader(
			first).stats[field])
	assert reader.tile_sums('extra', 'chr2').sum() == 100

	# The output may be one of the inputs, and the merge is a copy of the
	# chunks, so the file is their sizes less one set of tables.
	main(['merge', '-o', first, first, second])
	again = FigWigReader(first)
	assert len(again.tracks) == 6
	assert os.path.getsize(first) < os.path.getsize(out) + 1000


def test_merge_errors(tmp_path, parts):
	ref, first, second = parts
	other = str(tmp_path / 'other.fw')
	write_figwig(other, CHROM_SIZES, ['y'], 'chr1', [0], numpy.ones((1, 1, 10)),
		tile=128)
	with pytest.raises(SystemExit, match="was written with tile"):
		main(['merge', '-o', str(tmp_path / 'o.fw'), first, other])
	with pytest.raises(SystemExit, match="already one of the writer's"):
		main(['merge', '-o', str(tmp_path / 'o.fw'), first, first])
	with pytest.raises(SystemExit, match="is not a file"):
		main(['merge', '-o', str(tmp_path / 'o.fw'), str(tmp_path / 'no.fw')])
