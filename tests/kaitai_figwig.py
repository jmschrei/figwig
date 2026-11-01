import kaitaistruct
from kaitaistruct import KaitaiStruct
from kaitaistruct import KaitaiStream
from kaitaistruct import BytesIO
import zlib


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
	raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Figwig(KaitaiStruct):
	"""A figwig file holds the per-base values of many tracks over one genome:
	a 128-byte header, the chunks or blocks of values, the chromosome
	table, the statistics table, the metadata (a UTF-8 JSON document) and
	the index. This definition is the one in docs/format.rst of the figwig
	repository, which is the normative specification; it parses the header,
	every table, the index with its runs, live tiles, block records,
	summaries and zoom levels, and the bytes of every chunk and block, and
	the optional regional summary index of a file that has one. The
	chunk and block payload types take the parameters the index does not
	hold by itself: a chunk's kind byte and its group's number of tracks,
	which come from the metadata's track records (JSON, outside Kaitai's
	reach), and the tile size from the header. The optional regional
	summary index is found through an offset in the metadata (JSON again),
	so summary_index takes the root's offset as a parameter, and
	summary_page takes the number of descriptors the page holds; docs/format.rst
	says how to find both. tests/test_kaitai.py parses
	the test files with the parser compiled from this file and compares
	them with the reference reader.
	"""
	def __init__(self, _io, _parent=None, _root=None):
		self._io = _io
		self._parent = _parent
		self._root = _root if _root else self
		self._read()

	def _read(self):
		self.header = Figwig.Header(self._io, self, self._root)

	class SummaryLocator(KaitaiStruct):
		"""A group's 32-byte locator in the index directory."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.levels_offset = self._io.read_u8le()
			self.n_levels = self._io.read_u4le()
			self.levels_crc = self._io.read_u4le()
			self.group = self._io.read_u4le()
			self.n_tracks = self._io.read_u4le()
			self.crc = self._io.read_u4le()
			self.reserved = self._io.read_u4le()

		@property
		def levels(self):
			if hasattr(self, '_m_levels'):
				return self._m_levels

			_pos = self._io.pos()
			self._io.seek(self.levels_offset)
			self._m_levels = []
			for i in range(self.n_levels):
				self._m_levels.append(Figwig.SummaryLevel(self._io, self, self._root))

			self._io.seek(_pos)
			return getattr(self, '_m_levels', None)


	class Chunk(KaitaiStruct):
		"""A chunk's decoded bytes, by its kind byte, for a group of n_tracks tracks."""
		def __init__(self, kind_byte, n_tracks, tile, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.kind_byte = kind_byte
			self.n_tracks = n_tracks
			self.tile = tile
			self._read()

		def _read(self):
			_on = self.kind
			if _on == 1:
				self.body = Figwig.Dense(self.code, self.n_tracks, self.tile, self._io, self, self._root)
			elif _on == 2:
				self.body = Figwig.Sparse(self.code, self.n_tracks, self._io, self, self._root)
			elif _on == 3:
				self.body = Figwig.Matrix(self.code, self._io, self, self._root)
			elif _on == 4:
				self.body = Figwig.Sparse(self.code, self.n_tracks, self._io, self, self._root)

		@property
		def kind(self):
			if hasattr(self, '_m_kind'):
				return self._m_kind

			self._m_kind = (self.kind_byte & 7)
			return getattr(self, '_m_kind', None)

		@property
		def code(self):
			if hasattr(self, '_m_code'):
				return self._m_code

			self._m_code = ((self.kind_byte >> 3) & 15)
			return getattr(self, '_m_code', None)

		@property
		def stored(self):
			if hasattr(self, '_m_stored'):
				return self._m_stored

			self._m_stored = (self.kind_byte & 128) != 0
			return getattr(self, '_m_stored', None)


	class Sparse(KaitaiStruct):
		"""A sparse or runs chunk: the entries in all, the bytes of each
		track's varints, each track's offsets as LEB128 deltas, padding to
		4 bytes, and the values in the element type.
		"""
		def __init__(self, code, n_tracks, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.code = code
			self.n_tracks = n_tracks
			self._read()

		def _read(self):
			self.total = self._io.read_u4le()
			self.lengths = []
			for i in range(self.n_tracks):
				self.lengths.append(self._io.read_u4le())

			self._raw_tracks = []
			self.tracks = []
			for i in range(self.n_tracks):
				self._raw_tracks.append(self._io.read_bytes(self.lengths[i]))
				_io__raw_tracks = KaitaiStream(BytesIO(self._raw_tracks[i]))
				self.tracks.append(Figwig.TrackOffsets(_io__raw_tracks, self, self._root))

			self.padding = self._io.read_bytes(((4 - (self._io.pos() % 4)) % 4))
			self.values = Figwig.ElementValues(self.code, self.total, self._io, self, self._root)


	class ChunkBytes(KaitaiStruct):
		"""The bytes of one chunk as they lie in the file: a zlib stream in a
		compressed file unless its kind byte's high bit says it is stored as
		it is. `inflated` holds the chunk's bytes in either case; decode
		them with `chunk`, given the group's number of tracks.
		"""
		def __init__(self, g, live_index, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.g = g
			self.live_index = live_index
			self._read()

		def _read(self):
			self.data = self._io.read_bytes_full()

		@property
		def stored(self):
			if hasattr(self, '_m_stored'):
				return self._m_stored

			self._m_stored =  (((self.kind_byte & 128) != 0) or (self._root.header.codec == 0)) 
			return getattr(self, '_m_stored', None)

		@property
		def inflated_stream(self):
			if hasattr(self, '_m_inflated_stream'):
				return self._m_inflated_stream

			if not (self.stored):
				_pos = self._io.pos()
				self._io.seek(0)
				self._raw__m_inflated_stream = self._io.read_bytes_full()
				self._m_inflated_stream = zlib.decompress(self._raw__m_inflated_stream)
				self._io.seek(_pos)

			return getattr(self, '_m_inflated_stream', None)

		@property
		def inflated(self):
			if hasattr(self, '_m_inflated'):
				return self._m_inflated

			self._m_inflated = (self.data if self.stored else self.inflated_stream)
			return getattr(self, '_m_inflated', None)

		@property
		def kind_byte(self):
			if hasattr(self, '_m_kind_byte'):
				return self._m_kind_byte

			self._m_kind_byte = self._root.index.records[self.g].live.kinds[self.live_index]
			return getattr(self, '_m_kind_byte', None)

		@property
		def tile(self):
			if hasattr(self, '_m_tile'):
				return self._m_tile

			self._m_tile = self._root.index.records[self.g].live.tiles[self.live_index]
			return getattr(self, '_m_tile', None)


	class ElementValues(KaitaiStruct):
		"""n values of the element type, as an empty list for the "one" type,
		7, which stores none and whose every value is 1.
		"""
		def __init__(self, code, n, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.code = code
			self.n = n
			self._read()

		def _read(self):
			self.values = []
			for i in range((0 if self.code == 7 else self.n)):
				_on = self.code
				if _on == 0:
					self.values.append(self._io.read_u1())
				elif _on == 4:
					self.values.append(self._io.read_s2le())
				elif _on == 6:
					self.values.append(self._io.read_f4le())
				elif _on == 7:
					self.values.append(self._io.read_u1())
				elif _on == 1:
					self.values.append(self._io.read_u2le())
				elif _on == 3:
					self.values.append(self._io.read_s1())
				elif _on == 5:
					self.values.append(self._io.read_s4le())
				elif _on == 8:
					self.values.append(self._io.read_f8le())
				elif _on == 2:
					self.values.append(self._io.read_u4le())



	class SummaryWindow(KaitaiStruct):
		"""One window's descriptor, all zero for a window with no rows."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.n_rows = self._io.read_u4le()
			self.first_source_row = self._io.read_u4le()
			self.ids_offset = self._io.read_u4le()
			self.ids_bytes = self._io.read_u4le()
			self.ids_crc = self._io.read_u4le()
			self.ids_raw_crc = self._io.read_u4le()
			self.sums_offset = self._io.read_u4le()
			self.sums_bytes = self._io.read_u4le()
			self.sums_crc = self._io.read_u4le()
			self.sums_raw_crc = self._io.read_u4le()


	class SummaryIndex(KaitaiStruct):
		"""The optional regional summary index, announced by header optional
		bit 1: the root and one locator per group. The root's offset is the
		"offset" of the metadata's org.figwig.summary-restarts object, JSON
		outside Kaitai's reach. A level catalog is a locator's levels; a
		descriptor page is parsed by summary_page.
		"""
		def __init__(self, root_offset, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.root_offset = root_offset
			self._read()

		def _read(self):
			pass

		@property
		def root(self):
			if hasattr(self, '_m_root'):
				return self._m_root

			_pos = self._io.pos()
			self._io.seek(self.root_offset)
			self._m_root = Figwig.SummaryRoot(self._io, self, self._root)
			self._io.seek(_pos)
			return getattr(self, '_m_root', None)

		@property
		def locators(self):
			if hasattr(self, '_m_locators'):
				return self._m_locators

			_pos = self._io.pos()
			self._io.seek(self.root.directory_offset)
			self._m_locators = []
			for i in range(self.root.n_groups):
				self._m_locators.append(Figwig.SummaryLocator(self._io, self, self._root))

			self._io.seek(_pos)
			return getattr(self, '_m_locators', None)


	class Dense(KaitaiStruct):
		"""Every value of the tile, track after track."""
		def __init__(self, code, n_tracks, tile, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.code = code
			self.n_tracks = n_tracks
			self.tile = tile
			self._read()

		def _read(self):
			self.values = Figwig.ElementValues(self.code, (self.n_tracks * self.tile), self._io, self, self._root)


	class Run(KaitaiStruct):
		"""The chunks of tiles first to last, back to back from offset for bytes bytes."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.first = self._io.read_u8le()
			self.last = self._io.read_u8le()
			self.offset = self._io.read_u8le()
			self.bytes = self._io.read_u8le()


	class ZoomSection(KaitaiStruct):
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.n_zooms = self._io.read_u4le()
			self.levels = []
			for i in range(self.n_zooms):
				self.levels.append(Figwig.ZoomLevel(self._io, self, self._root))



	class Matrix(KaitaiStruct):
		"""The tracks present, a pointer per present track into the entries
		plus one at the end, the entries' offsets within the tile, padding
		to 4 bytes, and the values.
		"""
		def __init__(self, code, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.code = code
			self._read()

		def _read(self):
			self.n_present = self._io.read_u4le()
			self.tracks = []
			for i in range(self.n_present):
				self.tracks.append(self._io.read_u4le())

			self.ptr = []
			for i in range((self.n_present + 1)):
				self.ptr.append(self._io.read_u4le())

			self.offsets = []
			for i in range(self.ptr[self.n_present]):
				self.offsets.append(self._io.read_u2le())

			self.padding = self._io.read_bytes(((4 - (self._io.pos() % 4)) % 4))
			self.values = Figwig.ElementValues(self.code, self.ptr[self.n_present], self._io, self, self._root)


	class LiveTiles(KaitaiStruct):
		"""The tiles that hold values, ascending, with each one's kind byte,
		compressed size, uncompressed size and the CRC-32 of its bytes as
		they lie in the file; laid out as five arrays, one per field.
		"""
		def __init__(self, num_tiles, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.num_tiles = num_tiles
			self._read()

		def _read(self):
			self.tiles = []
			for i in range(self.num_tiles):
				self.tiles.append(self._io.read_u4le())

			self.kinds = []
			for i in range(self.num_tiles):
				self.kinds.append(self._io.read_u1())

			self.sizes = []
			for i in range(self.num_tiles):
				self.sizes.append(self._io.read_u4le())

			self.raws = []
			for i in range(self.num_tiles):
				self.raws.append(self._io.read_u4le())

			self.crcs = []
			for i in range(self.num_tiles):
				self.crcs.append(self._io.read_u4le())



	class SummaryStream(KaitaiStruct):
		def __init__(self, num_tiles, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.num_tiles = num_tiles
			self._read()

		def _read(self):
			self.tiles = []
			for i in range(self.num_tiles):
				self.tiles.append(self._io.read_u4le())

			self.sums = []
			i = 0
			while not self._io.is_eof():
				self.sums.append(self._io.read_f4le())
				i += 1



	class Varint(KaitaiStruct):
		"""A LEB128 varint of at most three bytes, low seven bits first."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.groups = []
			i = 0
			while True:
				_ = Figwig.VarintGroup(self._io, self, self._root)
				self.groups.append(_)
				if not (_.has_next):
					break
				i += 1

		@property
		def value(self):
			if hasattr(self, '_m_value'):
				return self._m_value

			self._m_value = ((self.groups[0].low + ((self.groups[1].low << 7) if len(self.groups) >= 2 else 0)) + ((self.groups[2].low << 14) if len(self.groups) >= 3 else 0))
			return getattr(self, '_m_value', None)


	class SummaryLevel(KaitaiStruct):
		"""A level's entry in a catalog; its position gives the level, 0 the base summary."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.source_offset = self._io.read_u8le()
			self.index_offset = self._io.read_u8le()
			self.source_rows = self._io.read_u4le()
			self.source_bytes = self._io.read_u4le()
			self.source_crc = self._io.read_u4le()
			self.window_bins = self._io.read_u4le()


	class ZoomLevel(KaitaiStruct):
		"""Level k, from 1, sums the tile sums over bins of 4**k tiles."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.factor = self._io.read_u4le()
			self.n = self._io.read_u4le()
			self.len_stream = self._io.read_u4le()
			self.crc = self._io.read_u4le()
			self._raw__raw_stream = self._io.read_bytes(self.len_stream)
			self._raw_stream = zlib.decompress(self._raw__raw_stream)
			_io__raw_stream = KaitaiStream(BytesIO(self._raw_stream))
			self.stream = Figwig.ZoomStream(self.n, _io__raw_stream, self, self._root)


	class Block(KaitaiStruct):
		"""A track-order block's decoded bytes."""
		def __init__(self, num_positions, kind, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.num_positions = num_positions
			self.kind = kind
			self._read()

		def _read(self):
			self.positions = []
			for i in range(self.num_positions):
				self.positions.append(self._io.read_u4le())

			self.padding = self._io.read_bytes(((8 - ((4 * self.num_positions) % 8)) % 8))
			self.values = Figwig.ElementValues(self.code, self.num_positions, self._io, self, self._root)

		@property
		def code(self):
			if hasattr(self, '_m_code'):
				return self._m_code

			self._m_code = ((self.kind >> 3) & 15)
			return getattr(self, '_m_code', None)


	class RunList(KaitaiStruct):
		"""The runs of a record from run r on, each run's chunks sized by the
		live tiles in order: a run's chunks are the live entries from
		live_start on, as many as fill its bytes.
		"""
		def __init__(self, g, r, live_start, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.g = g
			self.r = r
			self.live_start = live_start
			self._read()

		def _read(self):
			pass

		@property
		def run(self):
			if hasattr(self, '_m_run'):
				return self._m_run

			self._m_run = self._root.index.records[self.g].runs[self.r]
			return getattr(self, '_m_run', None)

		@property
		def chunks(self):
			if hasattr(self, '_m_chunks'):
				return self._m_chunks

			_pos = self._io.pos()
			self._io.seek(self._root.index.records[self.g].runs[self.r].offset)
			self._raw__m_chunks = self._io.read_bytes(self._root.index.records[self.g].runs[self.r].bytes)
			_io__raw__m_chunks = KaitaiStream(BytesIO(self._raw__m_chunks))
			self._m_chunks = Figwig.RunChunks(self.g, self.live_start, _io__raw__m_chunks, self, self._root)
			self._io.seek(_pos)
			return getattr(self, '_m_chunks', None)

		@property
		def next(self):
			if hasattr(self, '_m_next'):
				return self._m_next

			if (self.r + 1) < self._root.index.records[self.g].num_entries:
				_pos = self._io.pos()
				self._io.seek(0)
				self._m_next = Figwig.RunList(self.g, (self.r + 1), (self.live_start + len(self.chunks.chunks)), self._io, self, self._root)
				self._io.seek(_pos)

			return getattr(self, '_m_next', None)


	class SummaryRoot(KaitaiStruct):
		"""The 256-byte root of the index."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.magic = self._io.read_bytes(8)
			if not self.magic == b"\x46\x57\x52\x53\x55\x4D\x31\x00":
				raise kaitaistruct.ValidationNotEqualError(b"\x46\x57\x52\x53\x55\x4D\x31\x00", self.magic, self._io, u"/types/summary_root/seq/0")
			self.version = self._io.read_u2le()
			self.len_root = self._io.read_u2le()
			self.crc = self._io.read_u4le()
			self.directory_offset = self._io.read_u8le()
			self.n_groups = self._io.read_u4le()
			self.descriptors_per_page = self._io.read_u4le()
			self.window_target = self._io.read_u4le()
			self.reserved_a = self._io.read_bytes(28)
			self.header_snapshot = self._io.read_bytes(128)
			self.reserved_b = self._io.read_bytes(64)


	class Chrom(KaitaiStruct):
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.len_name = self._io.read_u2le()
			self.name = (self._io.read_bytes(self.len_name)).decode(u"UTF-8")
			self.length = self._io.read_u8le()


	class ChromTable(KaitaiStruct):
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.entries = []
			for i in range(self._root.header.n_chroms):
				self.entries.append(Figwig.Chrom(self._io, self, self._root))



	class VarintGroup(KaitaiStruct):
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.b = self._io.read_u1()

		@property
		def has_next(self):
			if hasattr(self, '_m_has_next'):
				return self._m_has_next

			self._m_has_next = (self.b & 128) != 0
			return getattr(self, '_m_has_next', None)

		@property
		def low(self):
			if hasattr(self, '_m_low'):
				return self._m_low

			self._m_low = (self.b & 127)
			return getattr(self, '_m_low', None)


	class ZoomStream(KaitaiStruct):
		def __init__(self, num_bins, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.num_bins = num_bins
			self._read()

		def _read(self):
			self.bins = []
			for i in range(self.num_bins):
				self.bins.append(self._io.read_u4le())

			self.sums = []
			i = 0
			while not self._io.is_eof():
				self.sums.append(self._io.read_f8le())
				i += 1



	class Header(KaitaiStruct):
		"""The 128-byte header."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.magic = self._io.read_bytes(4)
			if not self.magic == b"\x46\x49\x47\x57":
				raise kaitaistruct.ValidationNotEqualError(b"\x46\x49\x47\x57", self.magic, self._io, u"/types/header/seq/0")
			self.version = self._io.read_u2le()
			self.flags = self._io.read_u2le()
			self.tile = self._io.read_u4le()
			self.codec = self._io.read_u2le()
			self.level = self._io.read_u2le()
			self.n_chroms = self._io.read_u4le()
			self.n_tracks = self._io.read_u4le()
			self.n_groups = self._io.read_u4le()
			self.tables_crc = self._io.read_u4le()
			self.chroms_offset = self._io.read_u8le()
			self.tracks_offset = self._io.read_u8le()
			self.index_offset = self._io.read_u8le()
			self.data_offset = self._io.read_u8le()
			self.metadata_offset = self._io.read_u8le()
			self.metadata_length = self._io.read_u8le()
			self.zooms_offset = self._io.read_u8le()
			self.length = self._io.read_u8le()
			self.required = self._io.read_u4le()
			self.optional = self._io.read_u4le()
			self.header_crc = self._io.read_u4le()
			self.padding = self._io.read_bytes(20)

		@property
		def track_order(self):
			if hasattr(self, '_m_track_order'):
				return self._m_track_order

			self._m_track_order = (self.flags & 1) != 0
			return getattr(self, '_m_track_order', None)

		@property
		def has_zooms(self):
			if hasattr(self, '_m_has_zooms'):
				return self._m_has_zooms

			self._m_has_zooms = (self.optional & 1) != 0
			return getattr(self, '_m_has_zooms', None)

		@property
		def has_summary_index(self):
			if hasattr(self, '_m_has_summary_index'):
				return self._m_has_summary_index

			self._m_has_summary_index = (self.optional & 2) != 0
			return getattr(self, '_m_has_summary_index', None)


	class SummaryPage(KaitaiStruct):
		"""A descriptor page, at index_offset + 4096 * p for page p. It holds
		min(102, windows - 102 * p) descriptors, where windows is the level's
		bins over the genome divided by window_bins, rounded up.
		"""
		def __init__(self, num_windows, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.num_windows = num_windows
			self._read()

		def _read(self):
			self.magic = self._io.read_bytes(4)
			if not self.magic == b"\x46\x57\x52\x50":
				raise kaitaistruct.ValidationNotEqualError(b"\x46\x57\x52\x50", self.magic, self._io, u"/types/summary_page/seq/0")
			self.first_window = self._io.read_u4le()
			self.count = self._io.read_u4le()
			self.crc = self._io.read_u4le()
			self.windows = []
			for i in range(self.num_windows):
				self.windows.append(Figwig.SummaryWindow(self._io, self, self._root))



	class TrackOffsets(KaitaiStruct):
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.deltas = []
			i = 0
			while not self._io.is_eof():
				self.deltas.append(Figwig.Varint(self._io, self, self._root))
				i += 1



	class RunChunks(KaitaiStruct):
		def __init__(self, g, live_start, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.g = g
			self.live_start = live_start
			self._read()

		def _read(self):
			self._raw_chunks = []
			self.chunks = []
			i = 0
			while not self._io.is_eof():
				self._raw_chunks.append(self._io.read_bytes(self._root.index.records[self.g].live.sizes[(self.live_start + i)]))
				_io__raw_chunks = KaitaiStream(BytesIO(self._raw_chunks[-1]))
				self.chunks.append(Figwig.ChunkBytes(self.g, (self.live_start + i), _io__raw_chunks, self, self._root))
				i += 1



	class BlockRecord(KaitaiStruct):
		"""One track-order block, its chromosome, first and last position, entries, place and kind."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.chrom = self._io.read_u4le()
			self.first = self._io.read_u4le()
			self.last = self._io.read_u4le()
			self.num_entries = self._io.read_u4le()
			self.offset = self._io.read_u8le()
			self.len_body = self._io.read_u4le()
			self.raw = self._io.read_u4le()
			self.kind = self._io.read_u4le()
			self.crc = self._io.read_u4le()

		@property
		def body(self):
			if hasattr(self, '_m_body'):
				return self._m_body

			if self._root.header.codec == 0:
				_pos = self._io.pos()
				self._io.seek(self.offset)
				self._raw__m_body = self._io.read_bytes(self.len_body)
				_io__raw__m_body = KaitaiStream(BytesIO(self._raw__m_body))
				self._m_body = Figwig.Block(self.num_entries, self.kind, _io__raw__m_body, self, self._root)
				self._io.seek(_pos)

			return getattr(self, '_m_body', None)

		@property
		def body_deflated(self):
			if hasattr(self, '_m_body_deflated'):
				return self._m_body_deflated

			if self._root.header.codec == 1:
				_pos = self._io.pos()
				self._io.seek(self.offset)
				self._raw__raw__m_body_deflated = self._io.read_bytes(self.len_body)
				self._raw__m_body_deflated = zlib.decompress(self._raw__raw__m_body_deflated)
				_io__raw__m_body_deflated = KaitaiStream(BytesIO(self._raw__m_body_deflated))
				self._m_body_deflated = Figwig.Block(self.num_entries, self.kind, _io__raw__m_body_deflated, self, self._root)
				self._io.seek(_pos)

			return getattr(self, '_m_body_deflated', None)


	class Summary(KaitaiStruct):
		"""The tiles of a group with a nonzero sum, and each track's float32 sum over them."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.n_sum = self._io.read_u4le()
			self.len_stream = self._io.read_u4le()
			self.crc = self._io.read_u4le()
			if self.len_stream > 0:
				self._raw__raw_stream = self._io.read_bytes(self.len_stream)
				self._raw_stream = zlib.decompress(self._raw__raw_stream)
				_io__raw_stream = KaitaiStream(BytesIO(self._raw_stream))
				self.stream = Figwig.SummaryStream(self.n_sum, _io__raw_stream, self, self._root)



	class Record(KaitaiStruct):
		"""One group's record: in position order its runs and live tiles, in
		track order its blocks, then its summary and, when the header's
		optional bit 0 is set, its zoom section. `g` is the group's number,
		its place in the directory.
		"""
		def __init__(self, g, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self.g = g
			self._read()

		def _read(self):
			self.num_entries = self._io.read_u4le()
			self.crc = self._io.read_u4le()
			if self._root.header.track_order:
				self.blocks = []
				for i in range(self.num_entries):
					self.blocks.append(Figwig.BlockRecord(self._io, self, self._root))


			if not (self._root.header.track_order):
				self.runs = []
				for i in range(self.num_entries):
					self.runs.append(Figwig.Run(self._io, self, self._root))


			if not (self._root.header.track_order):
				self.n_live = self._io.read_u4le()

			if not (self._root.header.track_order):
				self.len_live = self._io.read_u4le()

			if not (self._root.header.track_order):
				self._raw__raw_live = self._io.read_bytes(self.len_live)
				self._raw_live = zlib.decompress(self._raw__raw_live)
				_io__raw_live = KaitaiStream(BytesIO(self._raw_live))
				self.live = Figwig.LiveTiles(self.n_live, _io__raw_live, self, self._root)

			self.summary = Figwig.Summary(self._io, self, self._root)
			if self._root.header.has_zooms:
				self.zooms = Figwig.ZoomSection(self._io, self, self._root)


		@property
		def chunk_runs(self):
			"""The chunks of every run, in run order, each with its live index."""
			if hasattr(self, '_m_chunk_runs'):
				return self._m_chunk_runs

			if  ((not (self._root.header.track_order)) and (self.num_entries > 0)) :
				_pos = self._io.pos()
				self._io.seek(0)
				self._m_chunk_runs = Figwig.RunList(self.g, 0, 0, self._io, self, self._root)
				self._io.seek(_pos)

			return getattr(self, '_m_chunk_runs', None)


	class TrackStats(KaitaiStruct):
		"""One track's statistics over its non-NaN bases."""
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.sum = self._io.read_f8le()
			self.nonzero = self._io.read_u8le()
			self.min = self._io.read_f8le()
			self.max = self._io.read_f8le()


	class Index(KaitaiStruct):
		def __init__(self, _io, _parent=None, _root=None):
			self._io = _io
			self._parent = _parent
			self._root = _root if _root else self
			self._read()

		def _read(self):
			self.directory = []
			for i in range(self._root.header.n_groups):
				self.directory.append(self._io.read_u8le())

			self.records = []
			for i in range(self._root.header.n_groups):
				self.records.append(Figwig.Record(i, self._io, self, self._root))



	@property
	def chrom_table(self):
		"""The chromosome table, at the header's chroms offset."""
		if hasattr(self, '_m_chrom_table'):
			return self._m_chrom_table

		_pos = self._io.pos()
		self._io.seek(self.header.chroms_offset)
		self._m_chrom_table = Figwig.ChromTable(self._io, self, self._root)
		self._io.seek(_pos)
		return getattr(self, '_m_chrom_table', None)

	@property
	def stats_table(self):
		"""One statistics record per track, at the header's tracks offset."""
		if hasattr(self, '_m_stats_table'):
			return self._m_stats_table

		_pos = self._io.pos()
		self._io.seek(self.header.tracks_offset)
		self._m_stats_table = []
		for i in range(self.header.n_tracks):
			self._m_stats_table.append(Figwig.TrackStats(self._io, self, self._root))

		self._io.seek(_pos)
		return getattr(self, '_m_stats_table', None)

	@property
	def metadata(self):
		"""The metadata, one JSON object."""
		if hasattr(self, '_m_metadata'):
			return self._m_metadata

		_pos = self._io.pos()
		self._io.seek(self.header.metadata_offset)
		self._m_metadata = (self._io.read_bytes(self.header.metadata_length)).decode(u"UTF-8")
		self._io.seek(_pos)
		return getattr(self, '_m_metadata', None)

	@property
	def index(self):
		"""The directory and one record per group."""
		if hasattr(self, '_m_index'):
			return self._m_index

		_pos = self._io.pos()
		self._io.seek(self.header.index_offset)
		self._m_index = Figwig.Index(self._io, self, self._root)
		self._io.seek(_pos)
		return getattr(self, '_m_index', None)


