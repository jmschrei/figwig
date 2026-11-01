# test_summary_index.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The units of figwig/_summary_index.py: merging bin ranges, bounded
inflating, the writer that decides which levels are restarted and how their
windows are sized, and the reader's checks of a descriptor page. The files
the writer produces are in test_summary_writer.py, the reads in
test_summary_reader.py and the damaged files in test_summary_corruption.py.
"""

import struct
import zlib

import numpy
import pytest

from figwig import _summary_index
from figwig._summary_index import _SummaryIndex
from figwig._summary_index import _SummaryIndexError
from figwig._summary_index import _SummaryIndexReader
from figwig._summary_index import _Level
from figwig._summary_index import Entry
from figwig._summary_index import _check_levels
from figwig._summary_index import _index_bytes
from figwig._summary_index import _inflate
from figwig._summary_index import _merge_ranges
from figwig._summary_index import _parse_page
from figwig._summary_index import _restart
from figwig._summary_index import _stream_offsets

from .summary_files import repair


##
# _merge_ranges
##

@pytest.mark.parametrize('ranges, expected', [
	([], []),
	([(2, 5)], [(2, 5)]),
	# Touching ranges are one; an empty range goes; order does not matter.
	([(5, 9), (1, 3), (3, 4), (9, 9), (12, 15)], [(1, 4), (5, 9), (12, 15)]),
	([(1, 10), (2, 3), (4, 12)], [(1, 12)]),
	([(7, 7), (3, 3)], []),
])
def test_merge_ranges_is_the_union_of_the_nonempty_ranges(ranges, expected):
	assert _merge_ranges(ranges) == expected


def test_merge_ranges_takes_an_array_and_a_long_unsorted_list():
	"""An (n, 2) array is taken as it is, and the union of many ranges in any
	order is the one a sweep gives: here 1,000 ranges of 3 bins every 5, which
	leave a gap of 2 between each, and the same in reverse and shuffled."""

	rng = numpy.random.default_rng(0)
	starts = numpy.arange(0, 5000, 5)
	pairs = numpy.column_stack([starts, starts + 3])
	want = [(int(a), int(a) + 3) for a in starts]
	assert _merge_ranges(pairs) == want
	assert _merge_ranges(pairs[::-1]) == want
	assert _merge_ranges(pairs[rng.permutation(len(pairs))]) == want
	# Ranges of 5 touch, and one that is inside another changes nothing.
	touching = numpy.column_stack([starts, starts + 5])
	assert _merge_ranges(numpy.vstack([touching, [[7, 9]]])) == [(0, 5000)]


def test_merge_ranges_takes_numpy_integers_and_returns_python_ones():
	merged = _merge_ranges((numpy.int64(a), numpy.uint32(b)) for a, b in
		[(4, 6), (0, 2)])
	assert merged == [(0, 2), (4, 6)]
	assert all(type(x) is int for pair in merged for x in pair)


##
# _inflate
##

def deflated(payload, raw=False):
	compressor = zlib.compressobj(wbits=-15 if raw else 15)
	return compressor.compress(payload) + compressor.flush()


@pytest.mark.parametrize('raw', [False, True])
def test_inflate_returns_the_bytes_of_a_stream(raw):
	payload = bytes(range(256)) * 4
	assert _inflate(deflated(payload, raw), len(payload), raw) == payload


@pytest.mark.parametrize('what', ['longer than asked', 'shorter than asked',
	'trailing bytes', 'cut off', 'not deflate'])
def test_inflate_refuses_what_is_not_exactly_the_stream(what):
	payload = bytes(range(200))
	data, size = deflated(payload), len(payload)
	if what == 'longer than asked':
		size += 1
	elif what == 'shorter than asked':
		size -= 1
	elif what == 'trailing bytes':
		data += b'\x00'
	elif what == 'cut off':
		data = data[:-5]
	else:
		data = b'\xff' * 40
	with pytest.raises(_SummaryIndexError, match='corrupt DEFLATE|wrong-sized or incomplete'):
		_inflate(data, size)


def test_inflate_never_asks_zlib_for_more_than_one_byte_past_the_size(
	monkeypatch):
	"""A fragment that expands without bound is stopped while it inflates,
	not after: zlib is given a limit of size + 1 and the output stays under
	it."""

	limits = []
	real = zlib.decompressobj

	class Recording:
		def __init__(self, *args):
			self.decoder = real(*args)

		def decompress(self, data, max_length=0):
			limits.append(max_length)
			return self.decoder.decompress(data, max_length)

		def __getattr__(self, name):
			return getattr(self.decoder, name)

	monkeypatch.setattr(_summary_index.zlib, 'decompressobj', Recording)
	with pytest.raises(_SummaryIndexError):
		_inflate(deflated(bytes(1 << 24)), 10)
	assert limits == [11]


def test_summary_index_error_is_a_value_error():
	assert issubclass(_SummaryIndexError, ValueError)


##
# _SummaryIndex: which levels are restarted, and how wide their windows are
##

def level_of(tracks, level, rows):
	"""One compressed level of `rows` rows over `tracks` tracks, every bin of
	the level listed: the writer's record of it and the stream it returned."""

	index = _SummaryIndex([tracks], [rows] * (level + 1))
	ids = numpy.arange(rows, dtype=numpy.uint32)
	sums = numpy.ones((rows, tracks), dtype=numpy.float64)
	stream = index.compress(0, level, ids, sums)
	return index.levels[0][0], stream, ids, sums


def rows_past_the_tiny_limit(tracks, level):
	row = 4 + (8 if level else 4) * tracks
	return 65536 // row + 1


@pytest.mark.parametrize('tracks, level, bins', [
	# bins is the largest power of two, at most 2,048, with bins * (4 + w *
	# tracks) <= 32,768, where w is 4 for the summary and 8 for a zoom level.
	(1, 0, 2048), (2, 0, 2048), (3, 0, 2048), (4, 0, 1024), (10, 0, 512),
	(100, 0, 64), (4000, 0, 2),
	(1, 1, 2048), (2, 1, 1024), (3, 1, 1024), (4, 1, 512), (1000, 3, 4),
])
def test_summary_index_sizes_windows_to_the_decoded_target(tracks, level, bins):
	record, _, _, _ = level_of(tracks, level, rows_past_the_tiny_limit(tracks,
		level))
	assert record[3] == bins and record[4] is not None


def test_summary_index_restarts_a_level_whose_row_is_exactly_the_target():
	# 8,191 tracks make a summary row of 4 + 4 * 8,191 = 32,768 bytes: one bin.
	record, _, _, _ = level_of(8191, 0, 3)
	assert record[3] == 1 and record[4] is not None


def test_summary_index_leaves_a_level_whose_row_is_past_the_target():
	# 8,192 tracks make a row of 32,772 bytes, though 3 rows are past 64 KiB.
	record, _, _, _ = level_of(8192, 0, 3)
	assert record[3] == 0 and record[4] is None


@pytest.mark.parametrize('rows, restarted', [(8191, False), (8192, False),
	(8193, True)])
def test_summary_index_restarts_a_level_only_past_64_kib(rows, restarted):
	"""One track makes a row of 8 bytes: 8,192 rows are exactly 65,536 bytes,
	which is not past 64 KiB, and 8,193 are."""

	record, _, _, _ = level_of(1, 0, rows)
	assert (record[4] is not None) == restarted


def test_summary_index_restarts_a_small_level_whose_stream_is_past_64_kib():
	"""Two rows of 8,191 tracks are 65,536 bytes, not past 64 KiB, but random
	sums deflate to a little more than they are: the level is restarted, since a
	reader that kept it ordinary would inflate a stream of 64 KiB and more."""

	rng = numpy.random.default_rng(0)
	sums = numpy.frombuffer(rng.bytes(2 * 8191 * 4), dtype='<f4').reshape(2, 8191)
	index = _SummaryIndex([8191], [2])
	stream = index.compress(0, 0, numpy.arange(2, dtype=numpy.uint32), sums)
	assert len(stream) > 65536 and index.levels[0][0][4] is not None


def test_summary_index_keeps_an_ordinary_level_as_the_ordinary_stream():
	record, stream, ids, sums = level_of(2, 0, 100)
	payload = ids.astype('<u4').tobytes() + sums.astype('<f4').tobytes()
	assert stream == zlib.compress(payload, 6)
	assert record == (100, len(stream), zlib.crc32(stream), 0, None)


def test_summary_index_writes_an_empty_level_as_an_empty_stream():
	index = _SummaryIndex([2], [1000])
	stream = index.compress(0, 0, numpy.zeros(0, dtype=numpy.uint32),
		numpy.zeros((0, 2)))
	assert stream == zlib.compress(b'', 6) and len(stream) == 8
	assert index.levels[0][0] == (0, 8, zlib.crc32(stream), 0, None)
	assert index.size() == 0


@pytest.mark.parametrize('level', [0, 2])
def test_summary_index_restarts_inside_one_stream_of_the_same_bytes(level):
	record, stream, ids, sums = level_of(2, level, 6000)
	dtype = '<f8' if level else '<f4'
	assert zlib.decompress(stream) == ids.astype('<u4').tobytes() + \
		sums.astype(dtype).tobytes()
	assert record[3] != 0 and record[0] == 6000 and record[1] == len(stream)
	assert record[2] == zlib.crc32(stream)


def test_summary_index_size_is_its_root_locators_catalogs_and_pages():
	index = _SummaryIndex([2], [6000])
	ids = numpy.arange(6000, dtype=numpy.uint32)
	index.compress(0, 0, ids, numpy.ones((6000, 2)))
	# 6,000 bins in windows of 2,048 are 3 windows, in one page of 16 + 3 * 40.
	assert index.size() == 256 + 32 + 32 + 3 * 40 + 16


##
# _SummaryIndex: the budget
##

def two_groups(monkeypatch, order):
	"""Two groups of two tracks with two levels over 20,480 and 5,120 bins,
	with a budget that holds the finest level's pages and 215 bytes more, one
	short of the next level's: level 0 needs 10 windows, 416 bytes, and level 1
	needs 5, 216. The root, the locators and the catalogs come out of it first."""

	base = 256 + 2 * (32 + 2 * 32)
	monkeypatch.setattr(_summary_index, 'INDEX_BYTES', base + 2 * (416 + 215))
	index = _SummaryIndex([2, 2], [20480, 5120])
	fine = numpy.arange(6000, dtype=numpy.uint32)
	coarse = numpy.arange(3300, dtype=numpy.uint32)
	for g in order:
		index.compress(g, 0, fine, numpy.ones((6000, 2)))
		index.compress(g, 1, coarse, numpy.ones((3300, 2)))
	return index


@pytest.mark.parametrize('order', [[0, 1], [1, 0]])
def test_summary_index_spends_a_groups_share_from_its_finest_level(
	monkeypatch, order):
	"""Each group has the same share, so which group compresses first does not
	change which levels are restarted: both keep level 0 and leave level 1."""

	index = two_groups(monkeypatch, order)
	for g in (0, 1):
		restarted = [record[4] is not None for record in index.levels[g]]
		assert restarted == [True, False]
	assert 0 < index.size() <= _summary_index.INDEX_BYTES


def test_summary_index_without_budget_restarts_nothing_and_writes_no_index(
	monkeypatch):
	monkeypatch.setattr(_summary_index, 'INDEX_BYTES', 512)
	index = _SummaryIndex([2], [20480])
	index.compress(0, 0, numpy.arange(6000, dtype=numpy.uint32),
		numpy.ones((6000, 2)))
	assert index.levels[0][0][4] is None and index.size() == 0


##
# _SummaryIndexReader: the checks of a descriptor page
##

# A level of 24 bins in windows of 8 has 3 windows, one page. Its 10 rows hold
# two tracks of f4 sums, so a window of n rows has ids of 4 * n bytes and sums
# of 8 * n, and its stream is 1,000 bytes with a 2-byte header and a 4-byte
# trailer around the fragments, which lie as a writer writes them: the ids
# fragments of every window with rows, in window order, then the sums fragments.
LEVEL = _Level(source_offset=0, index_offset=0, source_rows=10, source_bytes=1000,
	source_crc=0, window_bins=8, total=24)


def page_of(descriptors, first=0, count=None):
	"""The bytes of a page holding these descriptors, ten u4 each, with the
	CRC-32 the page's own rule asks for."""

	body = b''.join(struct.pack('<10I', *fields) for fields in descriptors)
	data = bytearray(b'FWRP' + struct.pack('<III', first, len(descriptors) if
		count is None else count, 0) + body)
	repair(data, 0, len(data), 12)
	return bytes(data)


# rows, rank, ids at, size, crc, raw crc, then the same for the sums.
WINDOW_0 = (3, 0, 2, 20, 0, 0, 52, 40, 0, 0)
EMPTY = (0,) * 10
WINDOW_2 = (4, 3, 22, 30, 0, 0, 92, 60, 0, 0)
GOOD = [WINDOW_0, EMPTY, WINDOW_2]


def read_page(descriptors, level=LEVEL, **kwargs):
	return _parse_page(page_of(descriptors, **kwargs), level, 0, 2, False)


def test_a_page_of_consistent_descriptors_is_read():
	rows = read_page(GOOD)
	assert [row[0] for row in rows] == [3, 0, 4]


@pytest.mark.parametrize('what, descriptors, message', [
	('more rows than bins in a window', [(9,) + WINDOW_0[1:]] + GOOD[1:],
		'invalid row count or rank'),
	# Alone in its page the window has nothing after it to disagree with, so
	# only the bound on its rank refuses it: 8 + 3 rows are past the 10.
	('a rank past the rows left', [(3, 8) + WINDOW_0[2:], EMPTY, EMPTY],
		'invalid row count or rank'),
	('an ids fragment before the stream header', [WINDOW_0[:2] + (1,) +
		WINDOW_0[3:]] + GOOD[1:], 'invalid fragment extent'),
	('an empty ids fragment', [WINDOW_0[:3] + (0,) + WINDOW_0[4:]] + GOOD[1:],
		'invalid fragment extent'),
	('an ids fragment longer than its rows allow', [WINDOW_0[:3] + (2 * 12 + 65,) +
		WINDOW_0[4:]] + GOOD[1:], 'invalid fragment extent'),
	('a sums fragment past the stream', [WINDOW_0[:6] + (990, 10) +
		WINDOW_0[8:]] + GOOD[1:], 'invalid fragment extent'),
	('ids and sums fragments that overlap', [WINDOW_0[:6] + (10,) +
		WINDOW_0[7:]] + GOOD[1:], 'overlapping ID/sum fragments'),
	('a second window that does not follow the first in rank', GOOD[:2] +
		[WINDOW_2[:1] + (4,) + WINDOW_2[2:]], 'out of order'),
	('a second window whose fragments start before the first ends', GOOD[:2] +
		[WINDOW_2[:2] + (10,) + WINDOW_2[3:]], 'out of order'),
	('a nonzero field in a window with no rows', [WINDOW_0, (0, 0, 0, 0, 1, 0,
		0, 0, 0, 0), WINDOW_2], 'nonzero empty-window descriptor'),
	# The ids of the second window, 30 to 60, lie inside the sums of the first, 22
	# to 62, though each window follows the one before in every offset.
	('the ids of a later window inside the sums of an earlier one', [(3, 0, 2, 20, 0,
		0, 22, 40, 0, 0), EMPTY, (4, 3, 30, 30, 0, 0, 92, 60, 0, 0)],
		'overlapping page fragments'),
])
def test_a_page_with_an_inconsistent_descriptor_is_refused(what, descriptors,
	message):
	with pytest.raises(_SummaryIndexError, match=message):
		read_page(descriptors)


def test_a_page_whose_last_window_has_fewer_bins_bounds_its_rows():
	"""24 bins in windows of 8, but a level of 21: the last window holds bins
	16 to 20, so a descriptor of 6 rows there is more than its window has."""

	with pytest.raises(_SummaryIndexError):
		read_page([WINDOW_0, EMPTY, (6,) + WINDOW_2[1:]], LEVEL._replace(total=21))


@pytest.mark.parametrize('what, kwargs', [
	('another first window', dict(first=1)),
	('another count', dict(count=2)),
])
def test_a_page_that_is_not_the_page_asked_for_is_refused(what, kwargs):
	with pytest.raises(_SummaryIndexError, match='descriptor page header'):
		read_page(GOOD, **kwargs)


def test_a_page_with_a_changed_byte_fails_its_checksum():
	data = bytearray(page_of(GOOD))
	data[20] ^= 1
	with pytest.raises(_SummaryIndexError, match='checksum'):
		_parse_page(bytes(data), LEVEL, 0, 2, False)


@pytest.mark.parametrize('what', ['random', 'zeros'])
def test_restarted_fragments_stay_within_their_bound_and_end_in_the_marker(what):
	"""A fragment of bytes that do not compress is a little longer than they are,
	and one of zeros is far shorter: both are at most 2 * decoded + 64 bytes and
	end with the flush marker, which the writer relies on and does not check."""

	rng = numpy.random.default_rng(0)
	ids = numpy.arange(1024, dtype=numpy.uint32)
	sums = rng.random((1024, 3), dtype=numpy.float32) if what == 'random' else \
		numpy.zeros((1024, 3), dtype=numpy.float32)
	stream, entries = _restart(ids, sums, 512, 2, '<f4')
	assert zlib.decompress(stream) == ids.tobytes() + sums.tobytes()
	for row in entries:
		for name, length in (('ids', 4 * row['n_rows']), ('sums', 12 * row['n_rows'])):
			start, size = int(row[name + '_offset']), int(row[name + '_bytes'])
			assert 0 < size <= 2 * length + 64
			assert stream[start + size - 4:start + size] == b'\x00\x00\xff\xff'


##
# _index_bytes and _check_levels
##

def test_index_bytes_counts_a_partial_last_window():
	"""Bins in windows of 2,048 are ceil(bins / 2,048) windows of 40 bytes, and a
	page of 16 bytes for each 102 of them."""

	assert _index_bytes(2048, 2048) == 40 + 16
	assert _index_bytes(2049, 2048) == 2 * 40 + 16
	assert _index_bytes(102 * 8, 8) == 102 * 40 + 16
	assert _index_bytes(102 * 8 + 1, 8) == 103 * 40 + 2 * 16


def levels_of(entries, sizes, zoomed=False, tracks=1):
	"""`_check_levels` over a version 1 record whose summary header starts at
	1,000 and whose streams have `sizes`, on a genome of 24 tiles."""

	end = 1000 + 12 + sum(sizes) + (4 if zoomed else 0) + 16 * (len(sizes) - 1)
	return _check_levels(entries, 1000, end, [24], tracks, zoomed, len(entries) - 1
		if zoomed else 0, 128, 1 << 20)


def test_check_levels_takes_a_level_where_the_record_has_it():
	levels, sections = levels_of([Entry(1012, 0, 5, 100, 0, 0)], [100])
	assert len(levels) == 1 and sections == [] and levels[0].total == 24


@pytest.mark.parametrize('rows, size', [(0, 100), (3, 5)])
def test_check_levels_refuses_a_restarted_level_without_rows_or_bytes(rows, size):
	entry = Entry(1012, 4096, rows, size, 0, 8)
	with pytest.raises(_SummaryIndexError, match='invalid window dimensions'):
		levels_of([entry], [size])


@pytest.mark.parametrize('what, entries, sizes, zoomed', [
	('a summary with rows and no stream', [Entry(1012, 0, 3, 0, 0, 0)], [0], False),
	('a zoom level with no stream', [Entry(1012, 0, 5, 100, 0, 0), Entry(1132, 0, 0,
		0, 0, 0)], [100, 0], True),
	('more rows than the level has bins', [Entry(1012, 0, 25, 100, 0, 0)], [100], False),
])
def test_check_levels_refuses_a_level_that_is_not_one(what, entries, sizes, zoomed):
	with pytest.raises(_SummaryIndexError, match='invalid source extent'):
		levels_of(entries, sizes, zoomed)


##
# _SummaryIndexReader._whole: a level read whole
##

def whole_level(ids, sums, total=100, **changes):
	"""The yield of `_whole` for a level of these rows, whose stream is built
	here with the CRC-32 the catalog would hold."""

	tracks = sums.shape[1]
	payload = numpy.asarray(ids, dtype='<u4').tobytes() + numpy.asarray(sums,
		dtype='<f4').tobytes()
	data = zlib.compress(payload, 6)
	level = _Level(source_offset=0, index_offset=0, source_rows=len(ids),
		source_bytes=len(data), source_crc=zlib.crc32(data), window_bins=0,
		total=total)._replace(**changes)
	return list(_SummaryIndexReader()._whole(lambda offset, size: data, level,
		tracks, 0))


def test_a_level_read_whole_yields_its_ids_and_sums():
	sums = numpy.arange(8, dtype=numpy.float32).reshape(4, 2)
	[(ids, got)] = whole_level([3, 8, 40, 99], sums)
	assert ids.tolist() == [3, 8, 40, 99] and got.tobytes() == sums.tobytes()


@pytest.mark.parametrize('ids, total', [
	([3, 3, 8], 100),		# not strictly ascending
	([8, 3, 40], 100),		# descending
	([3, 8, 100], 100),		# the last is not a bin of the level
])
def test_a_level_read_whole_must_have_ascending_ids_of_the_level(ids, total):
	with pytest.raises(_SummaryIndexError, match='out of order or bounds'):
		whole_level(ids, numpy.ones((3, 1)), total)


def test_a_level_read_whole_must_match_its_checksum():
	with pytest.raises(_SummaryIndexError, match='checksum'):
		whole_level([1, 2, 3], numpy.ones((3, 1)), source_crc=1)


def test_a_level_read_whole_is_refused_past_the_allocation_limit(monkeypatch):
	monkeypatch.setattr(_summary_index, 'MAX_LEVEL_BYTES', 10)
	with pytest.raises(_SummaryIndexError, match='allocation limit'):
		whole_level([1, 2, 3], numpy.ones((3, 1)))


def test_an_empty_level_read_whole_is_empty():
	[(ids, sums)] = whole_level([], numpy.zeros((0, 2)))
	assert len(ids) == 0 and sums.shape == (0, 2)


##
# The layout the two sides share
##

def test_the_structures_have_the_sizes_the_format_gives():
	assert (_summary_index.ROOT.size, _summary_index.LOCATOR.size,
		_summary_index.ENTRY.size, _summary_index.PAGE.size,
		_summary_index.DESCRIPTOR.itemsize) == (256, 32, 32, 16, 40)
	# A full page of 102 descriptors is exactly the stride between pages.
	assert _summary_index.PAGE.size + _summary_index.PAGE_ENTRIES * \
		_summary_index.DESCRIPTOR.itemsize == _summary_index.PAGE_STRIDE


@pytest.mark.parametrize('sizes, zoomed, offsets, end', [
	# A summary header of 12 bytes before the first stream; n_zooms (4) after it
	# when the record has zooms, even none; 16 for each zoom level's header.
	([100], False, [1012], 1112),
	([100], True, [1012], 1116),
	([100, 40], True, [1012, 1012 + 100 + 4 + 16], 1012 + 100 + 4 + 16 + 40),
	([100, 40, 8], True, [1012, 1132, 1132 + 40 + 16],
		1132 + 40 + 16 + 8),
])
def test_stream_offsets_follow_the_version_1_record(sizes, zoomed, offsets, end):
	assert _stream_offsets(1000, sizes, zoomed) == (offsets, end)
