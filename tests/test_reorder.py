# test_reorder.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig reorder`, which writes a figwig file in the other
order."""

import numpy
import pytest

import figwig.reorder
from figwig import FigWigReader
from figwig import FigWigWriter

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


@pytest.mark.parametrize('order', ['positions', 'tracks'])
def test_chromosomes_are_written_a_span_at_a_time(tmp_path, fw, monkeypatch,
	order):
	"""A chromosome is read and written in spans of whole tiles, so that a
	dense file's entries are not all held at once; the values and the
	statistics are those of a chromosome written in one call."""

	calls = []
	write_sparse = FigWigWriter.write_sparse
	monkeypatch.setattr(FigWigWriter, 'write_sparse', lambda self, *args:
		calls.append(args[0]) or write_sparse(self, *args))
	whole = str(tmp_path / 'whole.fw')
	main(['reorder', fw, whole, '--order', order, '-t', '64'])
	n_whole = len(calls)

	monkeypatch.setattr(figwig.reorder, '_SPAN_ENTRIES', 1)
	spans = str(tmp_path / 'spans.fw')
	main(['reorder', fw, spans, '--order', order, '-t', '64'])
	assert len(calls) - n_whole > 2 * n_whole

	original, a, b = FigWigReader(fw), FigWigReader(whole), FigWigReader(spans)
	assert_same_values(original, b)
	assert_same_values(a, b)
	for chrom, length in CHROM_SIZES.items():
		x = a.read(chrom, [0], length, dtype='float64')
		y = b.read(chrom, [0], length, dtype='float64')
		numpy.testing.assert_array_equal(x, y)
	numpy.testing.assert_array_equal(a.stats, b.stats)


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


def test_existing_output_is_refused_without_force(tmp_path, fw):
	out = str(tmp_path / 'o.fw')
	main(['reorder', fw, out, '--order', 'tracks'])
	with pytest.raises(SystemExit, match="exists; -f writes over it"):
		main(['reorder', fw, out, '--order', 'positions'])
	main(['reorder', fw, out, '--order', 'positions', '-f'])
	main(['reorder', out, out, '--order', 'tracks'])
	assert FigWigReader(out).order == 'tracks'
