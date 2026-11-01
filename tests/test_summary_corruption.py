# test_summary_corruption.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Files whose regional summary index is damaged. A reader that uses an index
it recognizes either reads the values the undamaged file holds or refuses the
file with a ValueError, and refuses before it reads or allocates by a count
nothing has checked. Three kinds of damage: a byte changed with every
checksum left as it was, a byte changed with the checksums over it repaired so
that only a rule of docs/format.rst can refuse it, and random bytes in either
way. The page rules have direct tests in test_summary_index.py.

Every case reads chrB of the first group with `tile_sums`, which takes the
catalog, one page and the fragments of one window.
"""

import struct
import zlib

import numpy
import pytest

from figwig import FigWigReader
from figwig import _summary_index
from figwig._summary_index import _SummaryIndexError

from .summary_files import patch
from .summary_files import put_fragment
from .summary_files import repair
from .summary_files import repair_page
from .summary_files import with_locator
from . import format_reference
from .format_reference import FormatError
from .summary_reference import read_index


def read(path):
	return FigWigReader(path).tile_sums(0, 'chrB')


def damaged(tmp_path, source, change):
	"""A copy of `source` that `change(data, index)` has changed, `index` being
	the reference reader's account of the undamaged file."""

	data = bytearray(source.read_bytes())
	change(data, read_index(source))
	path = tmp_path / 'damaged.fw'
	path.write_bytes(data)
	return path


def level_of(index, k=0):
	return index.groups[0].levels[k]


def window_of(index):
	"""chrB's window, the last of the summary: bins 24,576 to 25,054."""

	return level_of(index).windows[12]


def flip(at):
	"""A change that flips one bit at `at(index)`."""

	def change(data, index):
		data[at(index)] ^= 1
	return change


def fix_locator(data, index, g=0):
	repair(data, index.groups[g].locator, 32, 24)


def fix_catalog(data, index, g=0):
	group = index.groups[g]
	size = 32 * len(group.levels)
	patch(data, group.locator + 12, '<I', zlib.crc32(data[group.catalog:
		group.catalog + size]))
	fix_locator(data, index, g)


##
# A byte changed, no checksum repaired
##


##
# Rules refused with every checksum correct
##

def root_case(offset, fmt, value):
	def change(data, index):
		patch(data, index.root + offset, fmt, value(index) if callable(value)
			else value)
		repair(data, index.root, 256, 12)
	return change


def locator_case(offset, fmt, value):
	def change(data, index):
		patch(data, index.groups[0].locator + offset, fmt, value)
		fix_locator(data, index)
	return change


def catalog_case(offset, fmt, value, k=0):
	"""A field of level `k`'s catalog entry: source (0), index (8), rows (16),
	bytes (20), crc (24), bins (28)."""

	def change(data, index):
		patch(data, index.groups[0].catalog + 32 * k + offset, fmt,
			value(index) if callable(value) else value)
		fix_catalog(data, index)
	return change


def too_wide(data, index):
	"""Windows of 2,048 bins on a zoom level of two tracks: 20-byte rows, 40,960
	decoded bytes, past the 32 KiB target. With one track per group (track
	order) the rows are 12 bytes and 2,048 of them fit."""

	if level_of(index, 1).window_bins == 2048:
		pytest.skip('one track per group: a window of 2,048 bins is not too wide')
	catalog_case(28, '<I', 2048, k=1)(data, index)


def record_count(data, index):
	"""The run or block count that starts group 0's version 1 record, made a
	million: the raw index would run past the record."""

	at = struct.unpack_from('<Q', data, struct.unpack_from('<Q', data, 48)[0])[0]
	patch(data, at, '<I', 10 ** 6)


def ids_inside_the_sums(data, index):
	"""The ids of chrB's window moved onto the sums of the first window, so that
	each window still follows the one before it in every offset."""

	level, window = level_of(index), window_of(index)
	first = min(level.windows.values(), key=lambda w: w.sums_offset)
	patch(data, window.at + 8, '<I', first.sums_offset + 4)
	repair_page(data, level, window.at)


# What each case above is refused for. Several are refused by the first rule
# they break, which is why a few names read as another rule: the size of a stream
# and the offset of the next are one check of where the record ends.


def fragment_case(which, raw_of):
	def change(data, index):
		window = window_of(index)
		put_fragment(data, level_of(index), window, which, raw_of(window))
	return change


def garbage(which):
	def change(data, index):
		put_fragment(data, level_of(index), window_of(index), which,
			body=b'\xff' * 10 + b'\x00\x00\xff\xff')
	return change


def unflushed(data, index):
	compressor = zlib.compressobj(wbits=-15)
	body = compressor.compress(window_of(index).ids.astype('<u4').tobytes()) + \
		compressor.flush(zlib.Z_FINISH)
	put_fragment(data, level_of(index), window_of(index), 'ids',
		window_of(index).ids.astype('<u4').tobytes(), body)


def wrong_raw_crc(data, index):
	window = window_of(index)
	patch(data, window.at + 20, '<I', zlib.crc32(b'other bytes'))
	page = level_of(index).pages[0]
	repair(data, page, 16 + 40 * 13, 12)


def stream_header(first, second):
	def change(data, index):
		at = level_of(index).source_offset
		data[at:at + 2] = bytes([first, second])
	return change


def window_bits(bits):
	"""A zlib header for a window of 2 ** (bits + 8) bytes, no dictionary,
	with a valid check value."""

	cmf = (bits << 4) | 8
	flg = next(f for f in range(32) if (cmf * 256 + f) % 31 == 0)
	return stream_header(cmf, flg)


def with_dictionary():
	cmf = 0x78
	flg = next(f for f in range(256) if f & 32 and (cmf * 256 + f) % 31 == 0)
	return stream_header(cmf, flg)


# Every way the index is damaged that the tests make, with what it is refused for: the
# change to a copy of the indexed file, and a regex for the reason in the error.
# Each is refused by the reader, with the file and the feature in the message, and
# by the reference reader. Several are refused by the first rule they break, which is why
# a few names read as another rule: the size of a stream and the offset of the next are
# one check of where the record ends.
DAMAGE = {
	# A byte changed under a checksum, none repaired.
	'a byte of the root': (flip(lambda i: i.root + 40), 'checksum mismatch'),
	'a byte of the locator': (flip(lambda i: i.groups[0].locator + 16), 'checksum mismatch'),
	'a byte of the catalog': (flip(lambda i: i.groups[0].catalog + 16), 'checksum mismatch'),
	'a byte of the page checksum': (flip(lambda i: level_of(i).pages[0] + 12), 'checksum mismatch'),
	'a byte of the descriptor': (flip(lambda i: window_of(i).at + 8), 'checksum mismatch'),
	'a byte of the ids fragment': (flip(lambda i: level_of(i).source_offset + window_of(i).ids_offset + 3), 'checksum mismatch'),
	'a byte of the sums fragment': (flip(lambda i: level_of(i).source_offset + window_of(i).sums_offset + 3), 'checksum mismatch'),
	# A rule broken with every checksum over it repaired.
	'root magic': (root_case(0, '<B', ord('G')), 'invalid root magic, version or size'),
	'root extension version': (root_case(8, '<H', 3), 'invalid root magic, version or size'),
	'root size': (root_case(10, '<H', 128), 'invalid root magic, version or size'),
	'root reserved bytes': (root_case(40, '<B', 1), 'nonzero reserved bytes'),
	'root trailing reserved bytes': (root_case(200, '<B', 1), 'nonzero reserved bytes'),
	'root directory': (root_case(16, '<Q', lambda i: i.root + 264), 'invalid root dimensions'),
	'root group count': (root_case(24, '<I', 99), 'invalid root dimensions'),
	'root descriptors per page': (root_case(28, '<I', 101), 'invalid root dimensions'),
	'root window target': (root_case(32, '<I', 16384), 'invalid root dimensions'),
	'locator group number': (locator_case(16, '<I', 1), 'invalid group locator'),
	'locator track count': (locator_case(20, '<I', 99), 'invalid group locator'),
	'locator no levels': (locator_case(8, '<I', 0), 'invalid group locator'),
	'locator too many levels': (locator_case(8, '<I', 14), 'invalid group locator'),
	'locator reserved': (locator_case(28, '<I', 1), 'invalid group locator'),
	'locator catalog outside the file': (locator_case(0, '<Q', 2 ** 63), 'directory section outside data region'),
	'locator catalog checksum': (locator_case(12, '<I', 0), 'checksum mismatch'),
	# Level 0's stream offset is also where n_zooms is read, so the offset rule is
	# isolated on a zoom level, whose offset nothing else reads.
	'catalog stream offset': (catalog_case(0, '<Q', lambda i: level_of(i).source_offset + 1), 'catalog does not end with its version 1 record'),
	'catalog zoom stream offset': (catalog_case(0, '<Q', lambda i: level_of(i, 1).source_offset + 1, k=1), 'invalid source extent, count or order'),
	'catalog rows': (catalog_case(16, '<I', lambda i: level_of(i).source_rows + 1), 'version 1 stream header mismatch'),
	'catalog rows past the bins of the level': (catalog_case(16, '<I', 2 ** 30, k=2), 'invalid source extent, count or order'),
	'catalog stream bytes': (catalog_case(20, '<I', lambda i: level_of(i).source_bytes + 1), 'catalog does not end with its version 1 record'),
	'catalog stream bytes past the record': (catalog_case(20, '<I', 2 ** 31), 'version 1 zoom count differs from catalog'),
	'record count past its group': (record_count, 'raw index extends outside its group|summary starts outside its group'),
	'ids of a window inside the sums of another': (ids_inside_the_sums, 'overlapping page fragments'),
	'catalog zoom stream bytes': (catalog_case(20, '<I', lambda i: level_of(i, 1).source_bytes + 1, k=1), 'catalog does not end with its version 1 record'),
	'catalog stream checksum': (catalog_case(24, '<I', lambda i: level_of(i).source_crc ^ 1), 'version 1 stream header mismatch'),
	'catalog bins not a power of two': (catalog_case(28, '<I', 1000), 'invalid window dimensions'),
	'catalog bins past 2,048': (catalog_case(28, '<I', 4096), 'invalid window dimensions'),
	'catalog bins wider than the decoded target': (too_wide, 'invalid window dimensions'),
	'catalog ordinary level with pages': (catalog_case(28, '<I', 0), 'ordinary level has a descriptor index'),
	'catalog pages outside the data region': (catalog_case(8, '<Q', 2 ** 40), 'directory section outside data region'),
	'catalog pages over the catalog': (catalog_case(8, '<Q', lambda i: i.groups[0].catalog), 'overlapping directory sections'),
	'catalog ordinary level given pages': (catalog_case(8, '<Q', 1000, k=2), 'ordinary level has a descriptor index'),
	# The version 1 record's n_zooms, which no checksum covers on a regional read.
	'version 1 zoom count': (lambda data, i: patch(data, level_of(i).source_offset + level_of(i).source_bytes, '<I', 2), 'catalog does not end with its version 1 record'),
	# A fragment or a stream header whatever its checksums say.
	# A replacement has to compress to no more than the 689 bytes it replaces:
	# 462 ids moved down 256 or up 1,024 do.
	'ids below the window': (fragment_case('ids', lambda w: (w.ids - 256).astype('<u4').tobytes()), 'window IDs out of order or bounds'),
	'ids above the window': (fragment_case('ids', lambda w: (w.ids + 1024).astype('<u4').tobytes()), 'window IDs out of order or bounds'),
	'ids not ascending': (fragment_case('ids', lambda w: w.ids[::-1].astype('<u4').tobytes()), 'window IDs out of order or bounds'),
	'ids of fewer rows than the descriptor': (fragment_case('ids', lambda w: w.ids[:-1].astype('<u4').tobytes()), 'wrong-sized or incomplete DEFLATE'),
	'sums of fewer rows than the descriptor': (fragment_case('sums', lambda w: w.sums[:-1].tobytes()), 'wrong-sized or incomplete DEFLATE'),
	'ids that are not deflate': (garbage('ids'), 'corrupt DEFLATE stream'),
	'sums that are not deflate': (garbage('sums'), 'corrupt DEFLATE stream'),
	'sums that expand a thousandfold': (fragment_case('sums', lambda w: bytes(1 << 20)), 'wrong-sized or incomplete DEFLATE'),
	'ids without the flush marker': (unflushed, 'missing full-flush marker'),
	'a decoded checksum that is not the bytes': (wrong_raw_crc, 'checksum mismatch'),
	'a stream header with a dictionary': (with_dictionary(), 'invalid zlib wrapper'),
	'a stream header with a 4 KiB window': (window_bits(4), 'invalid zlib wrapper'),
	'a stream header that is not zlib': (stream_header(0x79, 0x9c), 'invalid zlib wrapper'),
}


@pytest.mark.parametrize('name', list(DAMAGE))
def test_a_damaged_index_is_refused_by_both_readers_with_the_reason(summary_pair,
	tmp_path, watched, name):
	"""The reader refuses with the file, the feature and the reason, and never
	reads past 64 KiB at a time before it has checked what it is reading; the
	reference reader, which checks every window and fragment and imports nothing
	from figwig, refuses the file too, so a rule the reader enforces is one the
	page states."""

	change, reason = DAMAGE[name]
	path = damaged(tmp_path, summary_pair[1], change)
	with pytest.raises(_SummaryIndexError, match=reason) as caught:
		read(path)
	assert str(caught.value).startswith('{}: the regional summary index cannot be '
		'read: '.format(path))
	assert all(size <= 65536 for _, size in watched.preads)
	with pytest.raises(FormatError):
		read_index(path)


@pytest.mark.parametrize('what, text', [
	('a length that is not 256', '{{"version": 2, "offset": {root}, "length": 128}}'),
	('an offset below the header', '{{"version": 2, "offset": 100, "length": 256}}'),
	('an offset with no room for the root before the tables', '{{"version": 2, "offset": {past}, "length": 256}}'),
	('an offset that is not an integer', '{{"version": 2, "offset": 1e5, "length": 256}}'),
])
def test_a_locator_that_cannot_be_followed_is_refused(summary_pair, tmp_path, what,
	text):
	"""The locator is an object of a version the reader knows, so the index is
	claimed; its root has to lie in the data region, 256 bytes long, or the file
	is refused, as it is by the reference reader. The replacement is padded with
	spaces to the length of the locator."""

	root = read_index(summary_pair[1]).root
	past = format_reference.File(summary_pair[1]).header.chroms - 255
	path = with_locator(summary_pair[1], tmp_path / 'locator.fw', lambda old:
		text.format(root=root, past=past).encode())
	with pytest.raises(_SummaryIndexError, match='invalid root locator'):
		read(path)
	with pytest.raises(FormatError):
		read_index(path)


def test_ids_that_end_at_the_next_windows_first_bin_are_refused(summary_pages,
	tmp_path):
	"""A window's bins end before the next window's first: an id equal to it, the
	last of window 3,128, is in the next window and not this one."""

	def change(data, index):
		level = index.groups[0].levels[0]
		window = level.windows[3128]
		ids = numpy.append(window.ids[:-1], 3129 * 8).astype('<u4')
		put_fragment(data, level, window, 'ids', ids.tobytes())

	path = damaged(tmp_path, summary_pages, change)
	with pytest.raises(_SummaryIndexError, match='window IDs out of order or bounds'):
		read(path)


def test_integer_sums_do_not_hide_a_corrupt_index(summary_pair, tmp_path):
	"""A damaged index raises in `sums` as it does in `tile_sums`, where an
	unusable version 1 summary falls back to the raw values, since the file
	claims an index it cannot trust."""

	def change(data, index):
		data[index.root + 40] ^= 1

	path = damaged(tmp_path, summary_pair[1], change)
	with pytest.raises(_SummaryIndexError, match='checksum mismatch'):
		FigWigReader(path).sums('chrA', [17], [16 * 40 + 3], tracks=[1])


def test_zoom_levels_in_a_file_without_optional_bit_0_are_refused(
	summary_pair, tmp_path):
	"""A group with four levels needs bit 0, which announces the zoom
	sections. Cleared, with the header's CRC-32 and the root's copy of the
	header repaired, the catalogs name levels the file has no sections for."""

	def change(data, index):
		patch(data, 100, '<I', 2)
		repair(data, 0, 128, 104)
		data[index.root + 64:index.root + 192] = data[:128]
		repair(data, index.root, 256, 12)

	with pytest.raises(_SummaryIndexError, match='invalid group locator'):
		read(damaged(tmp_path, summary_pair[1], change))


def test_a_whole_level_past_the_allocation_limit_is_refused(summary_pair,
	monkeypatch):
	"""A level read whole is inflated into memory, and a reader refuses one
	that would take more than the limit; here the limit is 1,000 bytes and
	chrA, most of the level, is read whole."""

	monkeypatch.setattr(_summary_index, 'MAX_LEVEL_BYTES', 1000)
	with pytest.raises(_SummaryIndexError, match='allocation limit'):
		FigWigReader(summary_pair[1]).tile_sums(chroms='chrA')


def test_an_index_past_the_size_limit_is_refused(summary_pair, monkeypatch):
	"""The structures of an index take at most 16 MiB from the root to the
	chromosome table; a reader with a limit of 1,000 bytes refuses this one,
	which takes more."""

	monkeypatch.setattr(_summary_index, 'INDEX_BYTES', 1000)
	with pytest.raises(_SummaryIndexError, match='invalid root dimensions'):
		read(summary_pair[1])


##
# Random damage
##

def structures(index):
	"""(offset, size, checksum field, whether a catalog) of everything the
	reads of the fuzz below can reach."""

	out = [(index.root, 256, 12, False)]
	for group in index.groups:
		out.append((group.locator, 32, 24, False))
		out.append((group.catalog, 32 * len(group.levels), None, True))
		for level in group.levels:
			windows = -(-level.total // level.window_bins) if level.window_bins else 0
			for p, page in enumerate(level.pages):
				out.append((page, 16 + 40 * min(102, windows - 102 * p), 12, False))
	return out


def test_random_damage_is_refused_or_changes_no_value(summary_pair, tmp_path):
	"""Sixty files with one to three random bytes changed in a random
	structure, the structure's checksums repaired in four of five: a ValueError
	or the undamaged values, never another exception and never other values."""

	source = summary_pair[1]
	index = read_index(source)
	reads = [lambda r: r.tile_sums(chroms='chrB'),
		lambda r: r.tile_sums(chroms='empty', level=1),
		lambda r: r.tile_sums(chroms='chrB', level=1)]
	want = [call(FigWigReader(source)) for call in reads]
	good = source.read_bytes()
	rng = numpy.random.default_rng(7)
	places = structures(index)
	path = tmp_path / 'fuzzed.fw'
	for _ in range(60):
		at, size, field, catalog = places[rng.integers(len(places))]
		data = bytearray(good)
		for _ in range(rng.integers(1, 4)):
			data[at + rng.integers(size)] = rng.integers(256)
		if rng.random() < 0.8:
			if catalog:
				fix_catalog(data, index, [g.catalog for g in index.groups].index(at))
			else:
				repair(data, at, size, field)
		path.write_bytes(data)
		for call, expected in zip(reads, want):
			try:
				got = call(FigWigReader(path))
			except ValueError:
				continue
			assert got.tobytes() == expected.tobytes()
