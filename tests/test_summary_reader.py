# test_summary_reader.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""`FigWigReader.tile_sums` and `sums` on files with the regional summary
index: the values against the dense values the files were written from, what
the reads touch, and what a reader does with an index it cannot use. Files
are in summary_files.py; the damaged ones are in test_summary_corruption.py.
"""

import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy
import pytest

import figwig.figwig

from figwig import FigWigReader
from figwig import FigWigWriter
from figwig import _summary_index
from figwig._summary_index import _SummaryIndexError

from .summary_files import COUNTS_CHROMS
from .summary_files import SIGNAL_CHROMS
from .summary_files import SIGNAL_GROUPS
from .summary_files import SIGNAL_TRACKS
from .summary_files import counts_values
from .summary_files import expected_tile_sums
from .summary_files import patch
from .summary_files import repair
from .summary_files import signal_values
from .summary_files import with_locator
from .summary_files import write_signal
from .summary_reference import read_index


def dense_sums(values_of, chroms, names, starts, ends, ids, nan):
	"""`sums` as numpy gives it from the dense values: intervals clipped to
	their chromosome, NaN bases left out, or only counted."""

	out = numpy.zeros((len(names), len(ids)))
	for row, (name, a, b) in enumerate(zip(names, starts, ends)):
		b = min(int(b), chroms[name])
		values = values_of(name)[ids][:, a:b].astype(numpy.float64)
		out[row] = (b - a) if nan == 'count' else numpy.nansum(values, axis=1)
	return out


##
# tile_sums
##

SELECTIONS = [
	# Few windows of the level (the fragment path), and a chromosome whose
	# bins are most of the level (the whole stream).
	pytest.param(None, ['chrB'], id='short-chromosome'),
	pytest.param([3, 0, 3, 2, 4, 1], ['chrB', 'empty', 'chrB'], id='order-and-repeats'),
	pytest.param('counts', ['empty'], id='empty-chromosome'),
	pytest.param(None, ['chrA'], id='long-chromosome'),
	pytest.param([2], ['chrA', 'chrB'], id='two-chromosomes'),
]


@pytest.mark.parametrize('level', range(4))
@pytest.mark.parametrize('tracks, chroms', SELECTIONS)
def test_tile_sums_are_the_dense_tile_sums(summary_pair, tracks, chroms, level):
	ids = list(range(5)) if tracks is None else (
		[1] if tracks == 'counts' else tracks)
	want = expected_tile_sums(signal_values, SIGNAL_CHROMS, chroms, ids, level)
	if tracks == 'counts':
		want = want[0]		# one track by name is returned without its axis
	for path in summary_pair:
		got = FigWigReader(path, verify=True).tile_sums(tracks, chroms, level)
		assert got.dtype == want.dtype and got.shape == want.shape
		assert got.tobytes() == want.tobytes()


def test_tile_sums_of_a_short_chromosome_read_only_their_fragments(
	summary_pair, watched):
	reader = FigWigReader(summary_pair[1])
	got = reader.tile_sums(chroms='chrB')
	assert got.tobytes() == expected_tile_sums(signal_values, SIGNAL_CHROMS,
		['chrB'], list(range(5)), 0).tobytes()
	# Every inflate is a restart fragment of at most the 32 KiB window target,
	# but for the empty stream of the group with no rows, which a reader still
	# inflates to check its CRC-32 and Adler-32.
	assert any(raw for _, raw in watched.inflated)
	assert all(raw and size <= 32768 for size, raw in watched.inflated if size)
	assert [size for size, raw in watched.inflated if not size] and not any(
		raw for size, raw in watched.inflated if not size)
	assert not watched.whole_streams(summary_pair[1]) & set(watched.preads)
	# chrB is 38 of the summary's 25,055 bins: the pages and fragments it needs
	# are a small part of the groups' summary streams.
	whole = sum(level.source_bytes for group in read_index(summary_pair[1]).groups
		for level in group.levels[:1])
	assert sum(size for _, size in watched.preads) < whole / 10
	assert all(index is None for index in reader._index)
	assert all(levels is None for levels in reader._zoom_levels)


def test_a_levels_stream_header_is_checked_once_per_reader(summary_pair,
	watched):
	"""The version 1 header and zlib header of a level's stream are read the
	first time the level is used and not again by a second read."""

	reader = FigWigReader(summary_pair[1])
	reader.tile_sums(chroms='chrB')
	first = list(watched.preads)
	reader.tile_sums(chroms='empty')
	headers = {(level.source_offset - 12, 14) for group in read_index(
		summary_pair[1]).groups for level in group.levels[:1] if level.window_bins}
	assert headers
	for header in headers:
		assert first.count(header) == 1
		assert watched.preads.count(header) == 1


@pytest.mark.parametrize('level', range(4))
@pytest.mark.parametrize('chrom', ['chrA', 'chrB', 'empty'])
def test_tile_sums_through_many_pages_are_the_dense_tile_sums(summary_pages,
	chrom, level):
	"""chrB and 'empty' are the last windows of the last page, where a page
	holds 72 windows and not 102; the zoom levels have pages of their own."""

	want = expected_tile_sums(signal_values, SIGNAL_CHROMS, [chrom], list(range(5)),
		level)
	got = FigWigReader(summary_pages, verify=True).tile_sums(chroms=chrom, level=level)
	assert got.dtype == want.dtype and got.tobytes() == want.tobytes()


def test_integer_sums_through_many_pages_are_the_dense_sums(summary_pages,
	watched):
	"""A page of 102 windows of 8 tiles of 16 bases is 13,056 bases: these
	intervals lie in pages 0, 3, 3 to 4, 15, 29 and 30, and each reads fragments."""

	names = ['chrA', 'chrA', 'chrA', 'chrA', 'chrA', 'chrB']
	starts = [17, 40_000, 51_000, 200_001, 389_000, 3]
	ends = [9_000, 41_000, 59_000, 215_003, 399_990, 600]
	got = FigWigReader(summary_pages).sums(names, starts, ends, tracks=[1],
		dtype='float64')
	want = dense_sums(signal_values, SIGNAL_CHROMS, names, starts, ends, [1],
		'propagate')
	assert got.tobytes() == want.tobytes()
	assert any(raw for _, raw in watched.inflated)


def test_tile_sums_of_the_whole_genome_inflate_whole_streams(summary_pair,
	watched):
	reader = FigWigReader(summary_pair[1])
	got = reader.tile_sums()
	assert got.tobytes() == expected_tile_sums(signal_values, SIGNAL_CHROMS,
		list(SIGNAL_CHROMS), list(range(5)), 0).tobytes()
	assert reader._regional is None and not watched.inflated


def test_tile_sums_of_all_of_a_level_inflate_its_whole_stream(summary_pair,
	watched):
	"""chrA's bins are all 13 windows of the summary: whole streams inflate,
	with the stream's own checks, and no restart fragment does."""

	FigWigReader(summary_pair[1]).tile_sums(chroms='chrA')
	assert watched.inflated and not any(raw for _, raw in watched.inflated)


@pytest.mark.parametrize('windows, whole', [(1565, False), (1566, True)])
def test_a_read_of_exactly_half_of_a_levels_windows_inflates_it_whole(
	summary_pages, watched, windows, whole):
	"""The summary has 3,132 windows of 8 bins when written for this fixture:
	1,566 are exactly half, which reads whole, and 1,565 are fragments. Bins are
	tiles of 16 bases, so the interval from 0 of 128 bases a window ends there."""

	end = windows * 8 * 16
	got = FigWigReader(summary_pages).sums('chrA', [0], [end], tracks=[1],
		dtype='float64')
	want = dense_sums(signal_values, SIGNAL_CHROMS, ['chrA'], [0], [end], [1],
		'propagate')
	assert got.tobytes() == want.tobytes()
	assert any(raw for _, raw in watched.inflated) == (not whole)


def test_tile_sums_refuse_a_dense_array_past_the_limit(summary_pair,
	monkeypatch):
	monkeypatch.setattr(figwig.figwig, '_MAX_DENSE_BYTES', 1000)
	with pytest.raises(ValueError, match='dense allocation limit'):
		FigWigReader(summary_pair[1]).tile_sums(chroms='chrA')


##
# sums
##

INTERVALS = (['chrA', 'chrB', 'chrA', 'chrA', 'empty', 'chrB'],
	[17, 3, 31000 * 16 + 3, 16385 * 16 + 1, 7, 600],
	[16 * 51 + 7, 510, 31040 * 16 + 13, 16400 * 16 + 9, 230, 650])


@pytest.mark.parametrize('nan', ['propagate', 'skip', 'count'])
def test_sums_are_the_dense_sums_in_every_nan_mode(summary_counts, nan):
	names, starts, ends = INTERVALS
	ids = [1, 0, 1, 2]
	want = dense_sums(counts_values, COUNTS_CHROMS, names, starts, ends, ids, nan)
	for path in summary_counts:
		got = FigWigReader(path, verify=True).sums(names, starts, ends,
			tracks=ids, n_jobs=1, nan=nan, dtype='float64')
		assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize('nan', ['propagate', 'skip'])
def test_integer_sums_inflate_only_the_fragments_of_their_tiles(
	summary_counts, watched, nan):
	names, starts, ends = INTERVALS
	FigWigReader(summary_counts[1]).sums(names, starts, ends, tracks=[1, 0, 1, 2],
		n_jobs=1, nan=nan, dtype='float64')
	assert watched.inflated and all(raw for _, raw in watched.inflated)
	# Two intervals of chrA lie 15,000 tiles apart; the tiles between them are
	# not read: far less is inflated than the summary of chrA holds.
	assert sum(size for size, _ in watched.inflated) < 131072


def test_counting_nan_bases_does_not_use_the_summaries(summary_counts, watched):
	names, starts, ends = INTERVALS
	FigWigReader(summary_counts[1]).sums(names, starts, ends, tracks=[1, 0, 1, 2],
		n_jobs=1, nan='count', dtype='float64')
	assert not watched.inflated


def test_aligned_integer_sums_need_no_raw_read(summary_counts, monkeypatch):
	reader = FigWigReader(summary_counts[1])

	def unexpected(*args, **kwargs):
		raise AssertionError('whole tiles hold their sums; no chunk is read')

	monkeypatch.setattr(reader, '_read', unexpected)
	starts, ends = [16 * 17, 16 * 30001], [16 * 43, 16 * 30025]
	got = reader.sums('chrA', starts, ends, tracks=[0, 1], dtype='float64')
	want = dense_sums(counts_values, COUNTS_CHROMS, ['chrA'] * 2, starts, ends,
		[0, 1], 'propagate')
	assert got.tobytes() == want.tobytes()


def test_sums_of_tiles_that_could_have_rounded_are_read_from_the_values(
	summary_counts):
	"""A tile of the last track sums to 16 * (2**24 + 1), which f4 rounds, so
	an interval over such tiles is summed from the values, exactly."""

	starts, ends = [17, 16 * 41], [16 * 39 + 3, 16 * 65]
	got = FigWigReader(summary_counts[1]).sums('chrA', starts, ends, tracks=3,
		dtype='float64')
	want = dense_sums(counts_values, COUNTS_CHROMS, ['chrA'] * 2, starts, ends,
		[3], 'propagate')
	assert got.tobytes() == want[:, 0].tobytes()


def test_sums_over_a_gap_with_no_rows_read_no_fragment(summary_counts, watched):
	reader = FigWigReader(summary_counts[1])
	got = reader.sums('chrA', [16400 * 16], [16450 * 16], tracks=[0, 1])
	assert not got.any() and not watched.inflated
	assert all(index is None for index in reader._index)


def test_float_sums_are_summed_from_the_values(summary_pair, watched):
	"""A float32 and a float64 track: the summaries hold f4 tile sums, so the
	sums of these tracks are not taken from them."""

	names, starts, ends = ['chrA', 'chrA', 'chrB'], [100, 70000, 5], [50000,
		180001, 600]
	got = FigWigReader(summary_pair[1]).sums(names, starts, ends, tracks=[0, 2],
		dtype='float64')
	want = dense_sums(signal_values, SIGNAL_CHROMS, names, starts, ends, [0, 2],
		'propagate')
	assert got.tobytes() == want.tobytes() and not watched.inflated


@pytest.mark.parametrize('n_jobs', [0, 'bad', 1.5])
def test_sums_check_n_jobs_even_when_the_summaries_answer(summary_counts,
	n_jobs):
	with pytest.raises((TypeError, ValueError), match='n_jobs'):
		FigWigReader(summary_counts[1]).sums('chrA', [16 * 17], [16 * 43],
			tracks=0, n_jobs=n_jobs)


def test_tile_sums_of_a_level_with_one_window_of_rows(tmp_path, watched,
	small_levels_restart):
	"""Only chrM-like signal: one window of the level has rows, so its ids
	fragment ends exactly where its sums fragment starts, and a reader that
	needed a gap between them would refuse a file that is valid. 8,230 bins in
	windows of 2,048 are five windows, and the 38 bins of `small` are in the
	fifth."""

	chroms = {'big': 16 * 2048 * 4, 'small': 603}
	path = tmp_path / 'one.fw'
	values = (numpy.arange(603) % 7).astype(numpy.float64)
	with FigWigWriter(path, chroms, ['a'], tile=16, engine='zlib', n_jobs=1,
			summary_index=True) as writer:
		writer.write('small', [0], values[None, None])
	level = read_index(path).groups[0].levels[0]
	assert level.window_bins == 2048 and list(level.windows) == [4]
	got = FigWigReader(path).tile_sums(0, chroms='small')
	padded = numpy.zeros(38 * 16)
	padded[:603] = values
	assert got.tolist() == padded.reshape(38, 16).sum(1).tolist()
	assert any(raw for _, raw in watched.inflated)


def test_tile_sums_of_an_indexed_file_without_zoom_levels(tmp_path, watched):
	"""`zooms=0` is the default of the commands and of write_figwig: the file
	has the summary and no zoom levels, and its index has one level a group."""

	path = tmp_path / 'flat.fw'
	with FigWigWriter(path, SIGNAL_CHROMS, SIGNAL_TRACKS, tile=16, groups=SIGNAL_GROUPS,
			engine='zlib', zooms=0, n_jobs=1, summary_index=True) as writer:
		for chrom in SIGNAL_CHROMS:
			writer.write(chrom, [0], signal_values(chrom)[None])
	assert [len(group.levels) for group in read_index(path).groups] == [1, 1, 1]
	got = FigWigReader(path).tile_sums(chroms='chrB')
	want = expected_tile_sums(signal_values, SIGNAL_CHROMS, ['chrB'], list(range(5)), 0)
	assert got.tobytes() == want.tobytes()
	assert any(raw for _, raw in watched.inflated)


@pytest.mark.parametrize('which', [0, 1])
def test_tile_sums_past_the_levels_of_a_file_name_the_level(summary_pair, which):
	with pytest.raises(ValueError, match='keeps no zoom level 4'):
		FigWigReader(summary_pair[which]).tile_sums(chroms='chrB', level=4)


def test_the_dense_limit_counts_eight_bytes_a_bin_of_a_zoom_level(summary_pair,
	monkeypatch):
	"""chrB has 10 bins at level 1, float64: 80 bytes for one track, past a limit
	of 60 that 40 bytes (float32) would be under."""

	monkeypatch.setattr(figwig.figwig, '_MAX_DENSE_BYTES', 60)
	with pytest.raises(ValueError, match='dense allocation limit'):
		FigWigReader(summary_pair[1]).tile_sums(tracks=0, chroms='chrB', level=1)


def test_sums_refuse_a_file_replaced_since_the_reader_opened(summary_counts,
	tmp_path):
	path = tmp_path / 'owned.fw'
	shutil.copyfile(summary_counts[1], path)
	reader = FigWigReader(path)
	reader.sums('chrA', [16 * 17], [16 * 43], tracks=[0])
	other = tmp_path / 'other.fw'
	shutil.copyfile(summary_counts[1], other)
	other.replace(path)
	with pytest.raises(ValueError, match='replaced or rewritten'):
		reader.sums('chrA', [16 * 17], [16 * 43], tracks=[0])


##
# Reads of the file, bounded
##

@pytest.mark.parametrize('offset, size', [(127, 4), (10, 4), (-1, 4),
	(2 ** 40, 4), (200, -1)])
def test_the_index_is_never_read_from_outside_the_file_or_its_header(
	summary_pair, offset, size):
	reader = FigWigReader(summary_pair[1])
	fd = os.open(summary_pair[1], os.O_RDONLY)
	try:
		with pytest.raises(_SummaryIndexError, match='outside the file'):
			reader._summary_read(fd)(offset, size)
		last = reader._file_size
		with pytest.raises(_SummaryIndexError, match='outside the file'):
			reader._summary_read(fd)(last - 3, 4)
	finally:
		os.close(fd)


def test_a_short_read_of_the_index_is_a_truncation(summary_pair, monkeypatch):
	reader = FigWigReader(summary_pair[1])
	monkeypatch.setattr(figwig.figwig, '_pread', lambda fd, size, offset: b'')
	fd = os.open(summary_pair[1], os.O_RDONLY)
	try:
		with pytest.raises(_SummaryIndexError, match='truncated'):
			reader._summary_read(fd)(200, 4)
	finally:
		os.close(fd)


def test_the_last_bytes_of_the_file_can_be_read(summary_pair):
	reader = FigWigReader(summary_pair[1])
	fd = os.open(summary_pair[1], os.O_RDONLY)
	try:
		assert len(reader._summary_read(fd)(reader._file_size - 4, 4)) == 4
	finally:
		os.close(fd)


##
# An index the reader does not use
##

def test_an_index_whose_header_is_not_the_files_is_ignored(summary_pair,
	tmp_path):
	"""The root holds a copy of the header; one byte of it changed, with the
	root's CRC-32 repaired, is a stale index."""

	data = bytearray(summary_pair[1].read_bytes())
	root = read_index(summary_pair[1]).root
	data[root + 64 + 28] ^= 1
	repair(data, root, 256, 12)
	path = tmp_path / 'stale.fw'
	path.write_bytes(data)
	assert read_index(path) is None
	reader = FigWigReader(path)
	got = reader.tile_sums(chroms='chrB')
	assert got.tobytes() == expected_tile_sums(signal_values, SIGNAL_CHROMS,
		['chrB'], list(range(5)), 0).tobytes()
	assert reader._regional is False


def test_a_locator_that_is_not_an_object_is_ignored(summary_pair, tmp_path):
	"""The key is figwig's, but a value that is not a locator is not one the
	reader can follow: it reads the ordinary streams, as for a version it does not
	know, and so does the reference reader."""

	path = with_locator(summary_pair[1], tmp_path / 'text.fw', lambda old: b'"' +
		b'x' * (len(old) - 2) + b'"')
	assert read_index(path) is None
	reader = FigWigReader(path, verify=True)
	got = reader.tile_sums(chroms='chrB')
	assert got.tobytes() == expected_tile_sums(signal_values, SIGNAL_CHROMS,
		['chrB'], list(range(5)), 0).tobytes()
	assert reader._regional is False


@pytest.mark.parametrize('version', [1, 9])
def test_an_extension_version_the_reader_does_not_know_is_ignored(tmp_path,
	monkeypatch, version):
	with monkeypatch.context() as other:
		other.setattr(_summary_index, 'VERSION', version)
		path = write_signal(tmp_path / 'other.fw')
	assert read_index(path) is None
	reader = FigWigReader(path, verify=True)
	for level in range(4):
		got = reader.tile_sums(chroms='chrB', level=level)
		assert got.tobytes() == expected_tile_sums(signal_values, SIGNAL_CHROMS,
			['chrB'], list(range(5)), level).tobytes()
	assert reader._regional is False


def test_a_file_without_the_bit_is_read_whole(summary_pair, tmp_path, watched):
	"""The bit cleared, with the header's CRC-32 repaired, makes the key a
	leftover: the reader does not look at the root."""

	data = bytearray(summary_pair[1].read_bytes())
	patch(data, 100, '<I', 1)
	repair(data, 0, 128, 104)
	path = tmp_path / 'no-bit.fw'
	path.write_bytes(data)
	reader = FigWigReader(path)
	got = reader.tile_sums(chroms='chrB')
	assert got.tobytes() == expected_tile_sums(signal_values, SIGNAL_CHROMS,
		['chrB'], list(range(5)), 0).tobytes()
	assert reader._regional is None and not watched.inflated


##
# One reader, many callers
##

def test_a_reader_with_an_index_pickles_and_is_read_by_threads(summary_pair,
	tmp_path):
	path = tmp_path / 'owned.fw'
	shutil.copyfile(summary_pair[1], path)
	reader = FigWigReader(path)
	want = expected_tile_sums(signal_values, SIGNAL_CHROMS, ['chrB'], [0], 0)[0]
	with ThreadPoolExecutor(4) as pool:
		results = list(pool.map(lambda _: reader.tile_sums(0, 'chrB'), range(12)))
	assert all(result.tobytes() == want.tobytes() for result in results)
	results[0][:] = -1
	assert reader.tile_sums(0, 'chrB').tobytes() == want.tobytes()

	copy = pickle.loads(pickle.dumps(reader))
	assert copy._regional is None
	assert copy.tile_sums(0, 'chrB').tobytes() == want.tobytes()

	replacement = tmp_path / 'replacement.fw'
	shutil.copyfile(summary_pair[1], replacement)
	replacement.replace(path)
	with pytest.raises(ValueError, match='replaced or rewritten'):
		reader.tile_sums(0, 'chrB')


def test_an_index_of_another_version_is_not_opened_again_on_every_read(
	summary_pair, tmp_path, monkeypatch):
	"""A file whose index the reader cannot use is read as one without, and the
	reader remembers it: after the first read it opens the file as often as for
	a file with no index."""

	root = read_index(summary_pair[1]).root
	text = '{{"version": 3, "offset": {}, "length": 256}}'.format(root)
	path = with_locator(summary_pair[1], tmp_path / 'other.fw',
		lambda old: text.encode().ljust(len(old)))
	opened = []
	real = figwig.figwig._open_read
	monkeypatch.setattr(figwig.figwig, '_open_read', lambda path: (opened.append(
		path), real(path))[1])

	def opens(file):
		reader = FigWigReader(file)
		reader.tile_sums(chroms='chrB')
		opened.clear()
		reader.tile_sums(chroms='chrB')
		return len(opened)

	assert opens(path) == opens(summary_pair[0])
