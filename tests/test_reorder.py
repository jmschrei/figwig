# test_reorder.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig reorder`, which writes a figwig file in the other
order."""

import numpy
import pytest

from figwig import FigWigReader
from figwig import FigWigWriter

from figwig.__main__ import main

from . import format_reference
from .summary_files import read_regions
from .summary_reference import KEY
from .summary_reference import read_index
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


def test_existing_output_is_refused_without_force(tmp_path, fw):
	out = str(tmp_path / 'o.fw')
	main(['reorder', fw, out, '--order', 'tracks'])
	with pytest.raises(SystemExit, match="exists; -f writes over it"):
		main(['reorder', fw, out, '--order', 'positions'])
	main(['reorder', fw, out, '--order', 'positions', '-f'])
	main(['reorder', out, out, '--order', 'tracks'])
	assert FigWigReader(out).order == 'tracks'


##
# The regional summary index
##

def restarted(path):
	index = read_index(path)
	return index is not None and any(level.window_bins for group in index.groups for
		level in group.levels)


@pytest.mark.parametrize('flags, input_has_one, output_has_one', [
	([], True, True),
	([], False, False),
	(['--summary-index'], False, True),
	(['--no-summary-index'], True, False),
])
def test_reorder_writes_an_index_as_asked_or_as_its_input_has_one(tmp_path,
	summary_counts, watched, flags, input_has_one, output_has_one):
	source = summary_counts[1 if input_has_one else 0]
	other = 'tracks' if FigWigReader(source).order == 'positions' else 'positions'
	out = tmp_path / 'out.fw'
	# An input in track order has no tile of its own, so the tile is given.
	main(['reorder', str(source), str(out), '--order', other, '-t', '16'] + flags)
	assert restarted(out) == output_has_one
	assert FigWigReader(out).tile_sums().tobytes() == \
		FigWigReader(source).tile_sums().tobytes()
	# Each chromosome read alone, which goes through the index where there is one.
	regions, used = read_regions(out, watched)
	assert used == output_has_one and regions == read_regions(source, watched)[0]


def test_reorder_writes_a_new_index_where_the_header_announces_a_damaged_one(
	tmp_path, summary_counts, watched):
	"""The header says the input has an index, and reordering reads the values,
	not the index: a root that fails its checksum does not make the output lose
	the index it would have had."""

	damaged = tmp_path / 'damaged.fw'
	data = bytearray(summary_counts[1].read_bytes())
	data[read_index(summary_counts[1]).root + 40] ^= 1
	damaged.write_bytes(data)
	other = 'tracks' if FigWigReader(damaged).order == 'positions' else 'positions'
	out = tmp_path / 'out.fw'
	main(['reorder', str(damaged), str(out), '--order', other, '-t', '16'])
	assert restarted(out)
	assert FigWigReader(out).tile_sums().tobytes() == \
		FigWigReader(summary_counts[1]).tile_sums().tobytes()
	regions, used = read_regions(out, watched)
	assert used and regions == read_regions(summary_counts[1], watched)[0]


def test_reorder_of_an_appended_file_writes_no_index(tmp_path, summary_counts):
	"""An append clears optional bit 1 and keeps the metadata key: the header says
	the file has no index, and reorder follows the header, not the leftover key."""

	path = tmp_path / 'grown.fw'
	path.write_bytes(summary_counts[1].read_bytes())
	with FigWigWriter(path, None, ['extra'], append=True, engine='zlib', n_jobs=1):
		pass
	assert KEY in format_reference.File(path).metadata
	other = 'tracks' if FigWigReader(path).order == 'positions' else 'positions'
	out = tmp_path / 'out.fw'
	main(['reorder', str(path), str(out), '--order', other, '-t', '16'])
	assert format_reference.File(out).header.optional & 2 == 0 and not restarted(out)
	assert KEY not in format_reference.File(out).metadata


def test_reorder_in_place_keeps_the_index(tmp_path, summary_counts):
	path = tmp_path / 'same.fw'
	path.write_bytes(summary_counts[1].read_bytes())
	other = 'tracks' if FigWigReader(path).order == 'positions' else 'positions'
	main(['reorder', str(path), str(path), '--order', other, '-t', '16'])
	assert restarted(path) and FigWigReader(path).order == other
