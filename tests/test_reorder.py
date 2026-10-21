# test_reorder.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig reorder`, which writes a figwig file in the other
order."""

import numpy
import pytest

from figwig import FigWigReader

from figwig.__main__ import main

from .test_figwig import CHROM_SIZES
from .test_figwig import reference_tracks
from .test_figwig import write_reference


@pytest.fixture(scope='session')
def fw(tmp_path_factory):
	path = str(tmp_path_factory.mktemp('fw') / 'reference.fw')
	write_reference(path, reference_tracks())
	return path


def assert_same_values(a, b):
	rng = numpy.random.default_rng(0)
	chroms = rng.choice(list(CHROM_SIZES), 50)
	starts = rng.integers(0, 5000, 50)
	x = a.read(chroms, starts, 300, dtype='float64')
	y = b.read(chroms, starts, 300, dtype='float64')
	numpy.testing.assert_array_equal(numpy.isnan(x), numpy.isnan(y))
	numpy.testing.assert_array_equal(numpy.nan_to_num(x), numpy.nan_to_num(y))


def test_round_trip_through_both_orders(tmp_path, fw):
	tracks = str(tmp_path / 'tracks.fw')
	back = str(tmp_path / 'back.fw')
	main(['reorder', fw, tracks, '--order', 'tracks', '-p', '2'])
	main(['reorder', tracks, back, '--order', 'csc', '-t', '64', '--groups', '2',
		'-l', '1', '-v'])
	original, by_track, again = FigWigReader(fw), FigWigReader(tracks), \
		FigWigReader(back)
	assert by_track.order == 'tracks' and again.order == 'positions'
	assert again.tile == 64 and [t['group'] for t in again.tracks] == [0, 0, 1,
		1, 2]
	assert by_track.genome == 'hg38' and again.attributes == original.attributes
	for a, b in zip(original.tracks, again.tracks):
		assert {k: v for k, v in a.items() if k != 'group'} == {k: v for k, v in
			b.items() if k != 'group'}
	assert_same_values(original, by_track)
	assert_same_values(original, again)
	numpy.testing.assert_array_equal(original.stats['nonzero'], again.stats[
		'nonzero'])


def test_errors(tmp_path, fw):
	with pytest.raises(SystemExit, match="is not a file"):
		main(['reorder', str(tmp_path / 'no.fw'), str(tmp_path / 'o.fw'),
			'--order', 'tracks'])
	with pytest.raises(SystemExit, match="power of two"):
		main(['reorder', fw, str(tmp_path / 'o.fw'), '--order', 'positions', '-t',
			'100'])


def test_layout_is_kept_and_a_file_can_be_written_over_itself(tmp_path, fw):
	"""The tile, groups and level default to the input's, and the output
	may be the input's own path."""

	path = str(tmp_path / 'same.fw')
	main(['reorder', fw, path, '--order', 'positions', '-t', '128', '--groups',
		'3', '-l', '1'])
	before = FigWigReader(path)
	assert before.tile == 128 and before._level == 1
	main(['reorder', path, path, '--order', 'positions'])
	after = FigWigReader(path)
	assert after.tile == 128 and after._level == 1 and [t['group'] for t in
		after.tracks] == [t['group'] for t in before.tracks]
	assert_same_values(FigWigReader(fw), after)
	main(['reorder', path, path, '--order', 'tracks'])
	by_track = FigWigReader(path)
	assert by_track.order == 'tracks'
	assert_same_values(FigWigReader(fw), by_track)
