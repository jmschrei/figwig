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


def test_merge_checks_the_genomes(tmp_path, parts):
	"""Files naming different genomes are refused unless -g says which to
	record; a file naming none takes the others'."""

	ref, first, second = parts
	mouse = str(tmp_path / 'mouse.fw')
	write_figwig(mouse, CHROM_SIZES, ['m'], 'chr1', [0], numpy.ones((1, 1, 10)),
		tile=FigWigReader(first).tile, genome='mm10')
	out = str(tmp_path / 'o.fw')
	with pytest.raises(SystemExit, match="different genomes, 'hg38', 'mm10'"):
		main(['merge', '-o', out, first, mouse])
	assert not os.path.exists(out)
	main(['merge', '-o', out, '-g', 'mm10', first, mouse])
	assert FigWigReader(out).genome == 'mm10'
	unnamed = str(tmp_path / 'unnamed.fw')
	write_figwig(unnamed, CHROM_SIZES, ['u'], 'chr1', [0], numpy.ones((1, 1,
		10)), tile=FigWigReader(first).tile)
	main(['merge', '-o', out, '-f', unnamed, first])
	assert FigWigReader(out).genome == 'hg38'


def test_existing_output_is_refused_without_force(tmp_path):
	ref = reference_tracks()
	a, b = str(tmp_path / 'a.fw'), str(tmp_path / 'b.fw')
	write_reference(a, ref)
	write_figwig(b, CHROM_SIZES, ['x'], 'chr1', [0], numpy.ones((1, 1, 10)),
		tile=64)
	out = str(tmp_path / 'out.fw')
	main(['merge', '-o', out, a, b])
	with pytest.raises(SystemExit, match="exists; -f writes over it"):
		main(['merge', '-o', out, a])
	assert len(FigWigReader(out).tracks) == 6
	main(['merge', '-o', out, '-f', a])
	assert len(FigWigReader(out).tracks) == 5
	main(['merge', '-o', out, out, b])
	assert len(FigWigReader(out).tracks) == 6

	other = str(tmp_path / 'l.fw')
	write_figwig(other, CHROM_SIZES, ['y'], 'chr1', [0], numpy.ones((1, 1, 10)),
		tile=64, level=1)
	with pytest.raises(SystemExit, match="level 1 where the first file has"):
		main(['merge', '-o', str(tmp_path / 'm.fw'), a, other])
