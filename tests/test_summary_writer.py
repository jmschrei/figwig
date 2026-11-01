# test_summary_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""`summary_index=` on FigWigWriter and write_figwig: what the file holds
with it and without, read back by the independent readers of version 1
(format_reference.py) and of the index (summary_reference.py), and what an
append, a merge and a reorder do to it. The units are in
test_summary_index.py, the reads in test_summary_reader.py and the damaged
files in test_summary_corruption.py.
"""

import struct

import numpy
import pytest

import figwig.figwig

from figwig import FigWigReader
from figwig import FigWigWriter
from figwig import _summary_index
from figwig import write_figwig

from . import format_reference
from .summary_files import SIGNAL_CHROMS
from .summary_files import signal_values
from .summary_files import write_signal
from .summary_reference import KEY
from .summary_reference import read_index


##
# The flag
##

def test_summary_index_is_off_by_default(tmp_path):
	"""A writer given no flag writes the bytes it writes for False, with no
	optional bit 1 and no key."""

	default = write_signal(tmp_path / 'default.fw', indexed=None)
	off = write_signal(tmp_path / 'off.fw', indexed=False)
	assert default.read_bytes() == off.read_bytes()
	file = format_reference.File(default)
	assert file.header.optional == 1 and KEY not in file.metadata
	assert read_index(default) is None


@pytest.mark.parametrize('value', [None, 0, 1, 'yes', numpy.bool_(True)])
def test_summary_index_must_be_a_bool_before_any_file_is_made(tmp_path, value):
	path = tmp_path / 'never.fw'
	with pytest.raises(TypeError, match='summary_index must be a bool'):
		FigWigWriter(path, {'chr1': 16}, ['a'], summary_index=value)
	assert not path.exists()


def test_write_figwig_passes_the_flag_on(tmp_path):
	"""Eight tracks of ones in one group make summary rows of 36 bytes, and
	32,768 // 36 rows, 512 as a power of two, fit one window."""

	path = tmp_path / 'one-call.fw'
	write_figwig(path, {'chr1': 160_000}, [{'name': 'a{}'.format(i)} for i in
		range(8)], 'chr1', [0], numpy.ones((1, 8, 160_000), dtype=numpy.float32),
		tile=16, groups=8, n_jobs=1, engine='zlib', summary_index=True)
	assert read_index(path).groups[0].levels[0].window_bins == 512
	assert read_index(path).groups[0].levels[0].source_rows == 10_000


##
# What an indexed file holds
##

def test_indexed_file_announces_its_index_and_ends_it_at_the_tables(
	summary_pair):
	ordinary, indexed = summary_pair
	file, index = format_reference.File(indexed), read_index(indexed)
	assert file.header.version == 1 and file.header.optional == 3
	assert file.metadata[KEY] == {'version': 2, 'offset': index.root,
		'length': 256}
	assert 128 <= index.root and index.sections[0][0] == index.root
	assert index.sections[-1][1] == file.header.chroms
	assert format_reference.File(ordinary).header.optional == 1
	assert KEY not in format_reference.File(ordinary).metadata
	assert read_index(ordinary) is None


def test_indexed_file_keeps_every_version_1_summary_and_value(summary_pair):
	"""The streams the index restarts inflate to the bytes of the ordinary
	files', so the specification's reader finds the same summary and zoom
	levels, and the package the same values."""

	ordinary, indexed = map(format_reference.File, summary_pair)
	for a, b in zip(ordinary.groups, indexed.groups):
		assert a.summary[0].tobytes() == b.summary[0].tobytes()
		assert a.summary[1].tobytes() == b.summary[1].tobytes()
		assert len(a.zooms) == len(b.zooms) == 3
		for x, y in zip(a.zooms, b.zooms):
			assert x[0] == y[0] and x[1].tobytes() == y[1].tobytes()
			assert x[2].tobytes() == y[2].tobytes()
	reader = FigWigReader(summary_pair[1], verify=True)
	for chrom, length in SIGNAL_CHROMS.items():
		got = reader.read(chrom, [0], length, dtype='float64')[0]
		assert numpy.array_equal(got, signal_values(chrom), equal_nan=True)


# A level is restarted only past 64 KiB decoded: each group has 22,990 rows of
# the summary and 5,748 of the first zoom level (the tiles of chrA outside its
# 2,048-tile gap, and chrB's 38), 1,438 of the second and 360 of the third.
# Two tracks make a 12-byte summary row and a 20-byte zoom row, so windows are
# 2,048 and 1,024 bins; in track order one track makes rows of 8 and 12 bytes,
# and both are 2,048. A group with no signal lists nothing: an 8-byte stream.
WINDOWS = {
	'positions': {0: [2048, 1024, 0, 0], 1: [2048, 1024, 0, 0]},
	'tracks': {g: [2048, 2048, 0, 0] for g in range(3)},
}


def test_restarted_levels_have_the_windows_the_rule_gives(summary_pair):
	file = format_reference.File(summary_pair[1])
	index = read_index(summary_pair[1])
	for g, bins in WINDOWS[file.header.order].items():
		levels = index.groups[g].levels
		assert [level.window_bins for level in levels] == bins
		assert [level.source_rows for level in levels] == [22990, 5748, 1438, 360]
	silent = index.groups[len(index.groups) - 1].levels
	assert [(level.window_bins, level.source_rows, level.source_bytes) for level in silent] == \
		[(0, 0, 8)] * 4


def test_windows_cover_the_bins_and_leave_the_gap_empty(summary_pair):
	"""25,055 summary bins (chrA's 25,000 tiles, chrB's 38 and 'empty''s 17) in
	windows of 2,048 are 13 windows in one page; the gap's tiles, bins 4,096 to
	6,143, are exactly window 2, which has no rows and so no descriptor."""

	level = read_index(summary_pair[1]).groups[0].levels[0]
	assert level.total == 25055 and level.window_bins == 2048
	assert len(level.pages) == 1
	assert sorted(level.windows) == [0, 1] + list(range(3, 13))
	assert sum(w.n_rows for w in level.windows.values()) == level.source_rows
	# Window 1 ends at bin 4,095 and window 3 starts at bin 6,144.
	assert level.windows[1].ids[-1] == 4095 and level.windows[3].ids[0] == 6144
	# chrB's tiles, bins 25,000 to 25,037, share the last window with chrA.
	assert level.windows[12].ids[-1] == 25037


def test_a_level_of_many_windows_is_laid_out_in_many_pages(summary_pages):
	"""25,055 bins in windows of 8 are 3,132 windows: 31 pages, 102 windows to a
	page and 72 on the last, 4,096 bytes apart. The reference reader checks
	every descriptor and fragment of every page as it reads."""

	level = read_index(summary_pages).groups[0].levels[0]
	assert level.window_bins == 8 and level.total == 25055
	assert level.pages == [level.index_offset + 4096 * p for p in range(31)]
	data = summary_pages.read_bytes()
	counts = [struct.unpack_from('<I', data, page + 8)[0] for page in level.pages]
	assert counts == [102] * 30 + [72]
	# The first window of the second page, and of the last, have rows of their own.
	assert level.windows[102].ids[0] == 102 * 8 and level.windows[3060].ids[0] == 3060 * 8


def test_levels_are_restarted_exactly_when_they_pass_64_kib(summary_pair):
	"""Every level of every group: restarted if and only if its decoded rows,
	of 4 + 4 * tracks bytes for the summary and 4 + 8 * tracks for a zoom
	level, pass 65,536 bytes."""

	index = read_index(summary_pair[1])
	file = format_reference.File(summary_pair[1])
	for g, group in enumerate(index.groups):
		tracks = len(file.group_tracks[g])
		for level in group.levels:
			row = 4 + (8 if level.k else 4) * tracks
			assert (level.window_bins != 0) == (level.source_rows * row > 65536)


##
# The file around the index
##

def test_no_budget_leaves_the_ordinary_file_byte_for_byte(tmp_path,
	monkeypatch):
	"""With no pages to spare no level is restarted, and the writer writes no
	root, no key and no bit: the file is the one the flag's absence writes."""

	monkeypatch.setattr(_summary_index, 'INDEX_BYTES', 512)
	indexed = write_signal(tmp_path / 'indexed.fw')
	assert indexed.read_bytes() == write_signal(tmp_path / 'ordinary.fw',
		indexed=False).read_bytes()


def test_files_without_summaries_get_no_index(tmp_path, monkeypatch):
	monkeypatch.setattr(figwig.figwig, '_SUMMARY_BYTES', 1)
	file = format_reference.File(write_signal(tmp_path / 'none.fw'))
	assert all(group.summary is None and not group.zooms for group in
		file.groups)
	assert file.header.optional & 2 == 0 and KEY not in file.metadata


def test_indexed_bytes_do_not_depend_on_the_threads(tmp_path):
	one = write_signal(tmp_path / 'one.fw', n_jobs=1)
	three = write_signal(tmp_path / 'three.fw', n_jobs=3)
	assert one.read_bytes() == three.read_bytes()


def test_restarted_streams_do_not_depend_on_the_chunk_codec(tmp_path):
	"""A file whose chunks are stored (level 0, codec 0) still has zlib
	summaries, and they are restarted."""

	path = write_signal(tmp_path / 'stored.fw', level=0)
	assert format_reference.File(path).header.codec == 0
	assert read_index(path).groups[0].levels[0].window_bins == 2048


def test_failed_index_finalization_leaves_the_previous_path(tmp_path,
	monkeypatch):
	path = tmp_path / 'previous.fw'
	path.write_bytes(b'what was there before')

	def fail(*args):
		raise RuntimeError('serializing the index failed')

	monkeypatch.setattr(_summary_index._SummaryIndex, 'serialize', fail)
	with pytest.raises(RuntimeError, match='serializing'):
		write_signal(path)
	assert path.read_bytes() == b'what was there before'
	assert list(tmp_path.iterdir()) == [path]


##
# Append, merge and reorder
##

def test_summary_index_cannot_be_asked_of_an_append(summary_pair, tmp_path):
	path = tmp_path / 'append.fw'
	path.write_bytes(summary_pair[1].read_bytes())
	before = path.read_bytes()
	with pytest.raises(ValueError, match='summary_index=True is not supported '
			'with append=True'):
		FigWigWriter(path, None, ['extra'], append=True, summary_index=True)
	assert path.read_bytes() == before


def test_an_append_clears_the_bit_and_keeps_the_locator(summary_pair,
	tmp_path):
	"""The old index is unused bytes: the reader finds none, the key stays for
	whoever wrote it, and every earlier track reads as it did."""

	path = tmp_path / 'append.fw'
	path.write_bytes(summary_pair[1].read_bytes())
	locator = format_reference.File(path).metadata[KEY]
	with FigWigWriter(path, None, ['extra'], append=True, engine='zlib',
			n_jobs=1):
		pass
	file = format_reference.File(path)
	assert file.header.optional == 1 and file.metadata[KEY] == locator
	assert read_index(path) is None
	reader, before = FigWigReader(path, verify=True), FigWigReader(summary_pair[1])
	assert len(reader.tracks) == len(before.tracks) + 1
	assert numpy.array_equal(reader.read('chrB', [0], 603, tracks=['signal']),
		before.read('chrB', [0], 603, tracks=['signal']))
