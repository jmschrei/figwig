The figwig file format, version 1
=================================

This page is the specification of the figwig file: every byte of a
version 1 file, what a reader must check, and what a writer must produce.
:doc:`figwig` says what the format is for and how to use it from Python;
this page is for writing another reader or writer, or for checking this
one. Two artifacts accompany it in the repository and are tested against
the files the package writes: ``tests/format_reference.py``, a reader
written from this page alone with numpy and zlib, whose every table,
record and value ``tests/test_format.py`` compares with the package's; and
``docs/figwig.ksy``, a `Kaitai Struct <https://kaitai.io>`_ definition of
the same layout, from which parsers in other languages can be compiled,
whose compiled Python parser ``tests/test_kaitai.py`` runs over the same
files.

The separate :doc:`regional summary index <summary_restarts>`
uses v1's optional metadata and unused data-region bytes while keeping this
core layout and its summary streams unchanged.
It supports interactive browser use without downloading and decompressing a
complete genome-wide summary level for a small region.

**Stability.** Version 1 is frozen: the layout on this page does not
change. A file that needs another layout carries another ``version`` in
its header, and a reader of version 1 refuses it. An addition that a
version 1 reader can ignore is announced by a bit of the header's
``optional`` field, and one it must know by a bit of ``required``, which a
reader refuses when it does not know the bit. No ``required`` bit is
defined, and one ``optional`` bit is.

The words *must* and *may* below are requirements on readers and writers;
*the writer* means the one in this package, whose choices among the
layouts the format allows are described where they matter, so that a file
can be rebuilt byte for byte.

Conventions
-----------

Every integer and float is little-endian. ``u1``, ``u2``, ``u4`` and
``u8`` are unsigned integers of that many bytes, ``i1``, ``i2`` and ``i4``
two's-complement signed ones, ``f4`` and ``f8`` IEEE 754 binary32 and
binary64. An *offset* counts bytes from the start of the file. *Padding to
n bytes* appends zero bytes until the length is a multiple of n; padding
bytes are zero when written and are ignored when read.

A *varint* is an unsigned integer in LEB128: seven bits per byte, the
least significant seven first, with the high bit of every byte set except
the last. The format uses varints only for offsets within a tile and
differences between them, which are below 2**16, so a varint takes at most
three bytes, and a reader may refuse a longer one.

A *CRC-32* is the cyclic redundancy check of gzip, PNG and
``zlib.crc32``: polynomial 0xEDB88320 in reflected form, initial value
0xFFFFFFFF, final complement. A *zlib stream* is RFC 1950: a two-byte
header, RFC 1951 DEFLATE data at any compression level, and the Adler-32
of the uncompressed bytes, which a reader checks by inflating to the end
of the stream. A stream with bytes after its end is invalid.

Multi-byte values inside a chunk are aligned to 4 bytes at most: a sparse
or matrix chunk's values follow padding to 4, so an ``f8`` or ``u8``
there may lie on an address that is not a multiple of 8, and a reader in
a language that cares must copy or use unaligned loads. A dense chunk's
values start at the chunk's first byte.

A *tile* is ``tile`` consecutive bases of one chromosome, where ``tile``
is the header's, a power of two from 16 to 65,536. Chromosome *c* of
length *L* has ``ceil(L / tile)`` tiles, and the last one extends past
the chromosome's end; every base past the end holds 0. Tiles are numbered
over the genome: tile 0 is the first tile of the first chromosome of the
chromosome table, and the first tile of chromosome *c* is the number of
tiles of the chromosomes before it. A genome has at most 2**26 tiles.

Layout
------

A file is, in this order:

1. the 128-byte header;
2. the data region: chunks, in position order, or blocks, in track order,
   in any order and with any unused bytes between them, from offset 128 to
   the chromosome table;
3. the chromosome table, at ``chroms``;
4. the statistics table, at ``tracks``;
5. the metadata, at ``metadata``, of ``metadata_length`` bytes;
6. the index, at ``index``: a directory of one offset per group, then one
   record per group, back to back, ending at ``length``.

Bytes at or after ``length`` are not part of the file and are ignored. The
four tables lie in the order above, each starting where the one before
ends as the writer lays them out, and ``tables_crc`` covers the bytes from
``chroms`` to the end of the directory, so a reader checks all four before
it reads any of them.

Every structure below is found through an offset in the header or the
index, never by scanning, and the data region may hold bytes nothing
points to (see "Appending and merging"). A reader must not assume that
the chunks are contiguous, that they are in tile order in the file, that
a chunk starts on any alignment, or that the data region ends where the
chromosome table starts.

The header
----------

.. list-table::
   :header-rows: 1
   :widths: 6 20 6 68

   * - offset
     - field
     - type
     - meaning
   * - 0
     - ``magic``
     - u4
     - the bytes ``FIGW``, which read as 0x57474946
   * - 4
     - ``version``
     - u2
     - 1
   * - 6
     - ``flags``
     - u2
     - bit 0 set for track order, clear for position order; every other bit is unknown in version 1 and a reader must refuse a file with one set
   * - 8
     - ``tile``
     - u4
     - bases per tile, a power of two from 16 to 65,536
   * - 12
     - ``codec``
     - u2
     - 0: every chunk and block is stored as its bytes; 1: each is a zlib stream unless its kind byte says it is stored
   * - 14
     - ``level``
     - u2
     - the compression level the chunks were written at, 0 with codec 0; it does not affect reading, and the writer only appends to or merges files of equal level
   * - 16
     - ``n_chroms``
     - u4
     - entries of the chromosome table, at least 1
   * - 20
     - ``n_tracks``
     - u4
     - tracks, at least 1
   * - 24
     - ``n_groups``
     - u4
     - groups, at least 1, and equal to ``n_tracks`` in track order
   * - 28
     - ``tables_crc``
     - u4
     - CRC-32 of the bytes from ``chroms`` to ``index + 8 * n_groups``
   * - 32
     - ``chroms``
     - u8
     - offset of the chromosome table
   * - 40
     - ``tracks``
     - u8
     - offset of the statistics table, which has one record per track
   * - 48
     - ``index``
     - u8
     - offset of the index
   * - 56
     - ``data``
     - u8
     - 128, where the data region starts
   * - 64
     - ``metadata``
     - u8
     - offset of the metadata
   * - 72
     - ``metadata_length``
     - u8
     - bytes of the metadata
   * - 80
     - ``zooms``
     - u8
     - 0, reserved; the zoom levels lie in the index records
   * - 88
     - ``length``
     - u8
     - the bytes of the file the format uses, from 128 to the file's size
   * - 96
     - ``required``
     - u4
     - feature bits a reader must know; none is defined, so a reader must refuse a file with any bit set
   * - 100
     - ``optional``
     - u4
     - feature bits a reader may ignore; bit 0 says each index record ends with a zoom section, and the writer sets it in every file; a reader ignores any other bit
   * - 104
     - ``header_crc``
     - u4
     - CRC-32 of these 128 bytes with this field as 0
   * - 108
     - padding
     - 20 bytes
     - 0

A reader checks, in this order: the magic; ``header_crc``; ``version``;
``required``; ``flags``; ``tile``; ``codec``; that ``length`` is at least
128 and at most the file's size; that ``128 <= chroms <= tracks <=
metadata <= index <= length``; that the three counts are nonzero and
``index + 8 * n_groups <= length``; then ``tables_crc``. Only then does it
read the tables.

The chromosome table
--------------------

``n_chroms`` entries, back to back, each: ``len`` (u2), the chromosome's
name in UTF-8 of ``len`` bytes, and its length in bases (u8). Names are
non-empty and distinct; a length is from 1 to 2**32 - 1. The entry's place
in the table is the chromosome's *id*, which the index uses, and the order
of the table is the file's chromosome order, which fixes the tile
numbering. The table ends at ``tracks``.

The statistics table
--------------------

``n_tracks`` records of 32 bytes, one per track in track order:

.. list-table::
   :header-rows: 1
   :widths: 14 6 80

   * - field
     - type
     - meaning
   * - ``sum``
     - f8
     - the sum of the track's values over every base that is not NaN
   * - ``nonzero``
     - u8
     - the bases whose value is neither 0 nor NaN
   * - ``min``
     - f8
     - the smallest value over the bases whose value is neither 0 nor NaN
   * - ``max``
     - f8
     - the largest, likewise

A track with no nonzero base has ``min`` and ``max`` 0, and a track whose
every nonzero value is positive has a positive ``min``: the 0 of a base
without a value, which the format cannot tell from a written 0, is no part
of either. The writer
accumulates ``sum`` with Neumaier's compensated summation, so its last
bits may differ between writers, and between writes of the same values in
different calls; a reader must not use it as an identity. The table ends
at ``metadata``.

The metadata
------------

``metadata_length`` bytes of UTF-8 encoding one JSON object. The format
defines five keys; a writer that appends to a file keeps every other key
of the file's metadata as it was, so other software may add its own.

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - key
     - value
   * - ``tracks``
     - a list of ``n_tracks`` objects in track order, each with ``name``, a non-empty string distinct among the tracks; ``dtype``, one of ``uint8``, ``uint16``, ``uint32``, ``int8``, ``int16``, ``int32``, ``float32``, ``float64``; ``group``, an integer from 0 to ``n_groups - 1``; and any other keys, the track's attributes
   * - ``attributes``
     - an object, the file's attributes
   * - ``genome``
     - a string naming the reference genome, or ``null``
   * - ``order``
     - ``"positions"`` or ``"tracks"``; informational, since the header's ``flags`` decide
   * - ``figwig``
     - a string, the version of the package that wrote the file; informational

The ``group`` keys define the groups: group *g* holds the tracks whose
``group`` is *g*, in track order, and that order is the order of a group's
tracks within its chunks (a track's *slot*). Every group holds at least
one track; in track order every group holds exactly one, and the writer
numbers them so that track *t* is group *t*. A reader refuses a metadata
that is not an object, whose ``attributes`` is not an object, or whose
``tracks`` does not describe ``n_tracks`` tracks as above.

A track's ``dtype`` is the type a reader presents its values in, and
every value stored for the track is exactly representable in it: an
integer type holds no fraction and no NaN, ``float32`` holds binary32
values. The element type of a chunk (below) is the storage type of one
tile for every track of the group, chosen per tile, and may be narrower
or wider than a track's ``dtype``: a ``uint8`` track's tile may lie in a
``u1`` chunk, in an ``f8`` chunk when a ``float64`` track of the group has
a value in the tile, or in a *one* chunk; the values are the same in any.

The writer serializes the object with Python's ``json.dumps(document,
allow_nan=False)``: keys in insertion order, the five keys above after any
carried over, in the order ``figwig``, ``genome``, ``attributes``,
``order``, ``tracks``; a track's record in the order its keys were given,
with ``dtype`` and ``group`` added at the end when they were not;
separators ``", "`` and ``": "``; non-ASCII characters as ``\uXXXX``
escapes; no NaN or infinity, which are not JSON. Any serialization of the
same object is a valid file; that one makes a rebuild byte-identical.

The index
---------

The index starts with the *directory*: ``n_groups`` offsets (u8), the
offset of each group's record. The records follow the directory back to
back, in group order, and the last ends at ``length``, but a reader finds
each record through the directory, which ``tables_crc`` covers. A record
holds the group's chunks or blocks, then its summary, then, when the
header's ``optional`` bit 0 is set, its zoom section.

A position-order record
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 10 72

   * - field
     - type
     - meaning
   * - ``n_runs``
     - u4
     - the runs that follow
   * - ``crc``
     - u4
     - CRC-32 of the bytes from the first run to the end of the live stream: the runs, ``n_live``, ``n_bytes`` and the stream
   * - runs
     - 32 bytes each
     - ``first`` (u8), ``last`` (u8), ``offset`` (u8), ``bytes`` (u8): the chunks of the live tiles ``first`` to ``last`` lie back to back in the file from ``offset`` for ``bytes`` bytes
   * - ``n_live``
     - u4
     - the live tiles: the tiles that hold a chunk
   * - ``n_bytes``
     - u4
     - bytes of the zlib stream that follows
   * - live stream
     - ``n_bytes``
     - a zlib stream inflating to ``17 * n_live`` bytes: five arrays, one after another, each with one entry per live tile: ``tile`` (u4), ``kind`` (u1), ``size`` (u4), ``raw`` (u4), ``crc`` (u4)

The live tiles are listed in ascending tile number, each at most once,
and every tile number is below the genome's tile count. A tile that is not
listed holds only zeros in every track of the group. For a listed tile,
``kind`` is its chunk's kind byte (below), whose kind is never 0; ``size``
the chunk's bytes in the file; ``raw`` its bytes once decoded, a multiple
of 8; and ``crc`` the CRC-32 of its ``size`` bytes as they lie in the
file. ``size`` and ``raw`` are nonzero, equal when the codec is 0 or the
kind byte's stored bit is set; ``raw`` is at most ``8 * tracks * tile +
16 * tracks + 64`` for a group of ``tracks`` tracks, the largest a chunk
of the group can be, and ``size`` at most that plus 64.

A run covers the tiles ``first`` to ``last`` inclusive: its chunks are
those of the live tiles in that range, in tile order, each starting where
the one before ends, so the offset of a live tile's chunk is its run's
``offset`` plus the ``size`` of the run's earlier live tiles, and a run's
``bytes`` is the sum of its chunks' sizes. Runs do not overlap in tile
range or in the file, and every live tile lies in exactly one run; a run
may cover tiles that are not live, which take no bytes. The writer makes a
new run each time a group's chunks stop being contiguous in the file,
which happens when another group's chunks are written in between, so a
group written in one stretch has one run.

To find the chunk of tile *T* of a group, then: search the live tiles for
*T* (they are sorted); if it is absent the tile is zeros; otherwise, with
*k* its index among the live tiles, find the run with ``first <= T <=
last``, let *k0* be the index of the first live tile at or above the run's
``first``, and the chunk lies at ``offset + sum(size[k0:k])`` for
``size[k]`` bytes. A reader that reads many tiles computes every live
tile's offset once by walking the runs in turn, as the package's does.

A track-order record
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 10 72

   * - field
     - type
     - meaning
   * - ``n_blocks``
     - u4
     - the block records that follow
   * - ``crc``
     - u4
     - CRC-32 of the block records' bytes
   * - block records
     - 40 bytes each
     - ``chrom`` (u4, a chromosome id), ``first`` and ``last`` (u4, the positions of the block's first and last entry on the chromosome), ``n`` (u4, entries, from 1 to 65,536), ``offset`` (u8), ``size`` (u4, bytes in the file), ``raw`` (u4, bytes once decoded, a multiple of 8), ``kind`` (u4, the kind byte, below 256), ``crc`` (u4, CRC-32 of the ``size`` bytes in the file)

The records are sorted by chromosome id then ``first``, and on one
chromosome each block's ``first`` is past the one before's ``last``, so a
track's entries on a chromosome ascend across its blocks. ``last`` is below
the chromosome's length. A block's kind byte has kind 1 and an element
code; its stored bit is never set, so in a codec 1 file every block is a
zlib stream, even one that compression made larger, and in a codec 0 file
``size`` equals ``raw``. ``raw`` is at most ``12 * n + 64``.

The summary
~~~~~~~~~~~

Either record continues with the group's summary: ``n_sum`` (u4),
``n_sum_bytes`` (u4), ``crc`` (u4, the CRC-32 of the stream's
``n_sum_bytes`` bytes), and a zlib stream of ``n_sum_bytes`` bytes inflating
to ``n_sum * (4 + 4 * tracks)`` bytes: the numbers of ``n_sum`` tiles (u4
each, ascending, all below the genome's tile count), then for each of those
tiles in turn the sum of each of the group's tracks over the tile (f4,
``tracks`` of them per tile). A tile's sum of a track is the sum of the
track's values over the tile's bases that are not NaN, rounded to f4; the
listed tiles are those where at least one track's sum is not 0. The
writer adds a tile's values in binary64 in base order and rounds once,
or, for a tile written in several calls, adds the calls' f4 sums in
binary64 and rounds again; another writer's sums may differ in the last
bit, and a reader must not expect to reproduce them. The writer keeps
summaries while ``n_tracks * n_tiles * 4`` is at most 2**28 bytes, and
otherwise writes ``n_sum`` 0, ``n_sum_bytes`` 0 and ``crc`` 0 for every
group, which a reader must accept as "no summary". A group with a summary
and no listed tile has ``n_sum`` 0 and a zlib stream of no bytes (8 bytes
long, with its CRC-32).

The zoom section
~~~~~~~~~~~~~~~~

When the header's ``optional`` bit 0 is set, which the writer always sets,
the summary is followed by ``n_zooms`` (u4, at most 12) and ``n_zooms``
levels, level *k* counted from 1:

.. list-table::
   :header-rows: 1
   :widths: 18 10 72

   * - field
     - type
     - meaning
   * - ``factor``
     - u4
     - ``4 ** k``, tiles per bin
   * - ``n``
     - u4
     - bins listed
   * - ``n_bytes``
     - u4
     - bytes of the stream
   * - ``crc``
     - u4
     - CRC-32 of the stream's ``n_bytes`` bytes
   * - stream
     - ``n_bytes``
     - a zlib stream inflating to ``n * (4 + 8 * tracks)`` bytes: ``n`` bin numbers (u4, ascending), then each bin's sum per track of the group (f8)

Bins are numbered over the genome as tiles are: a chromosome of *t* tiles
has ``ceil(t / factor)`` bins, each of ``factor`` consecutive tiles of that
chromosome, so no bin crosses a chromosome, and a chromosome's bins follow
the bins of the chromosomes before it. With chromosomes of 5 and 2 tiles,
level 1 (``factor`` 4) has bins 0 and 1 for tiles 0-3 and 4 of the first
chromosome and bin 2 for tiles 5-6 of the second; level 2 (``factor`` 16)
has bins 0 and 1. A bin's sum of a track is the sum of the f4 tile sums
of the summary over the bin's tiles, in f8; the listed bins are those that
hold a listed tile. A file written without zoom levels has ``n_zooms`` 0,
and a group without a summary has no levels either.

Chunks
------

The kind byte
~~~~~~~~~~~~~

A chunk's or block's *kind byte* has three fields:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - bits
     - field
   * - 0 to 2
     - the kind: 0 zero, 1 dense, 2 sparse, 3 matrix, 4 runs; a block is always 1
   * - 3 to 6
     - the element code: 0 ``u1``, 1 ``u2``, 2 ``u4``, 3 ``i1``, 4 ``i2``, 5 ``i4``, 6 ``f4``, 7 *one*, 8 ``f8``
   * - 7
     - stored: in a codec 1 file, the chunk's bytes are its decoded bytes, not a zlib stream; never set in a codec 0 file or on a block

Kind 0 is never written in a live entry: a tile of zeros is left out of
the live list, and the writer never writes a live chunk whose values are
all 0, though a reader accepts one. The element type is the type of every
stored value of the chunk, for all of its tracks; *one* stores no values,
every nonzero base having the value 1, and is only used by sparse and
matrix chunks. A reader refuses a kind above 4, a code above 8, a dense
or runs chunk with the one code, and a block whose kind is not 1.

Bytes in the file
~~~~~~~~~~~~~~~~~

A chunk's *decoded bytes* are its payload, as laid out below, padded to 8
bytes; their length is the live entry's ``raw``, which is therefore, with
``pad4`` and ``pad8`` rounding up to a multiple of 4 and 8:

- dense: ``pad8(tracks * tile * w)``;
- sparse and runs: ``pad8(pad4(4 + 4 * tracks + sum(lengths)) + total * w)``;
- matrix: ``pad8(pad4(8 + 8 * n_present + 2 * total) + total * w)``;
- a block: ``pad8(pad8(4 * n) + n * w)``.

In a codec 0 file the
chunk lies in the file as those bytes. In a codec 1 file it lies as one
zlib stream of them, unless the writer found the stream no shorter than
the bytes, in which case the chunk lies as its decoded bytes and its kind
byte carries the stored bit. ``size`` is the chunk's bytes in the file
either way, and ``crc`` is the CRC-32 of those bytes, which a reader
checks on every read of a stored chunk inside a codec 1 file, the one case
where no zlib checksum covers it, and on every chunk when asked to verify.

In the layouts that follow, ``tracks`` is the number of tracks of the
chunk's group, ``tile`` the header's, and an *offset* a base's position
within the tile, from 0 to ``tile - 1``. ``w`` is the element type's
width in bytes: 1, 2, 4, 1, 2, 4, 4, 0 and 8 for codes 0 to 8.

Dense, kind 1
~~~~~~~~~~~~~

Every value of the tile, track after track in slot order: ``tracks *
tile`` values of the element type, ``tracks * tile * w`` bytes. A track's
bases past the chromosome's end are 0.

Sparse, kind 2
~~~~~~~~~~~~~~

The nonzero bases of each track, with their values:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - field
     - meaning
   * - ``total`` (u4)
     - nonzero bases over every track of the group
   * - ``lengths`` (``tracks`` u4)
     - for each track in slot order, the bytes its offsets take below; 0 for a track with no nonzero base
   * - offsets
     - for each track in slot order, back to back with nothing between tracks, its nonzero offsets strictly ascending, the first as a varint, each later one as a varint of its difference from the one before; a track's offsets take exactly ``lengths[slot]`` bytes, and the number of its bases is the number of its bytes whose high bit is clear
   * - padding
     - to a multiple of 4 bytes, counted from the first byte of the chunk
   * - values
     - ``total`` values of the element type, each track's in turn, in the order of its offsets; none for the one code

The offsets of all tracks add up to ``total``. A NaN base counts as a
nonzero base and is stored with the value NaN, which makes the chunk
``f4`` or ``f8``.

Runs, kind 4
~~~~~~~~~~~~

The same layout as sparse, with another meaning: a track's offsets are the
bases where its value changes, strictly ascending, and its values the
value from each of those bases up to the next change, or to the end of
the tile. A track's value is 0 before its first change, so its first
change has a nonzero value, a track that starts the tile with a nonzero
value has its first offset at 0, and a run that returns to 0 is a change
whose value is 0, so a runs chunk's values include zeros and never use the
one code. ``total`` is the number of changes over every track, and two
consecutive changes of a track have different values, NaN counting as one
value. A reader need not check the values; a writer must not repeat one.

Matrix, kind 3
~~~~~~~~~~~~~~

For a group of many tracks of which few have a value in the tile:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - field
     - meaning
   * - ``n_present`` (u4)
     - tracks with a nonzero base in the tile, at most ``tracks``
   * - ``present`` (``n_present`` u4)
     - their slots, strictly ascending
   * - ``ptr`` (``n_present + 1`` u4)
     - ``ptr[j]`` is the index of present track *j*'s first entry below, ``ptr[0]`` is 0, ``ptr[n_present]`` is the number of entries in all, and the pointers do not decrease; the writer lists only tracks with an entry, and a reader accepts a present track with none
   * - offsets (u2 each)
     - ``ptr[n_present]`` offsets, those of each present track in turn, strictly ascending within a track
   * - padding
     - to a multiple of 4 bytes, counted from the first byte of the chunk
   * - values
     - ``ptr[n_present]`` values of the element type, in the order of the offsets; none for the one code

What a writer stores
~~~~~~~~~~~~~~~~~~~~

A value is a finite number or NaN; a writer refuses infinity, and stores
negative zero as 0. The values a chunk holds are the track's values *as
the track keeps them*: a track of ``dtype`` ``float32``, and any fraction of
a track of an integer ``dtype`` or of one whose ``dtype`` was inferred, is
rounded to binary32 before it is encoded or summed; an integer of an
integer track is kept exact, and the writer refuses an integer beyond
2**53; a ``float64`` track keeps every value. A value beyond binary32's
range is refused for every track but a ``float64`` one.

The writer stores a tile in whichever of the four kinds takes the fewest
decoded bytes (in a codec 1 file counting the per-track ``lengths`` of a
sparse or runs chunk at a sixty-fourth of their size, since mostly-zero
counts deflate to little), in the narrowest element type that holds every
stored value of the tile: ``u1``, ``u2`` or ``u4`` for integers from 0,
``i1``, ``i2`` or ``i4`` when one is negative, *one* when every nonzero
value is 1 (and the kind allows it), ``f4`` when a value is not an integer
or a NaN is present, and ``f8`` when a value of a ``float64`` track is
present, or an integer of an integer track lies beyond 2**24, which ``f4``
would round. A reader must accept any kind and any element type that
holds the values, whatever a writer's policy.

A tile written through ``write_sparse`` is stored as the kind ``write``
would choose for the rows its entries fill, but for how a runs chunk is
weighed: ``write`` counts the changes of the values it is given, before
they are rounded as the track keeps them, and each NaN base as one;
``write_sparse`` counts the changes the chunk would hold, so it may choose
runs where ``write`` does not.

Blocks
------

In track order, each track's nonzero bases on each chromosome lie in
blocks of 1 to 65,536 entries, in ascending position, each block's decoded
bytes being: ``n`` positions on the chromosome (u4 each, ascending), padding
to 8 bytes, ``n`` values of the element type, padding to 8 bytes. A value is
never 0; NaN is stored. The element code is chosen per block as for a
chunk, the one code included: a block whose every value is 1 holds its
positions and no values. The tile grid still applies to a track-order
file: the header's ``tile`` sets the tiles its summary and zoom levels sum
over.

Appending and merging
---------------------

A file is appended to in place: the new tracks' chunks are written from
the old ``length`` on, then new tables and a new index, and the header is
written last, after the rest has reached the disk, so that until the
header changes the file reads as it did, and a writer that fails leaves
the old header, whose ``length`` cuts the file back to what it was. The old
tables and index are then unused bytes inside the data region, between the
old chunks and the new, which no run, record or offset points to; a merge
into a new file copies each run's and block's bytes and drops them. A
reader that counts a file's bytes may report them, as ``figwig info -b``
does, but must not read them.

Limits
------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - what
     - limit
   * - chromosome length
     - 1 to 2**32 - 1 bases
   * - chromosome name
     - 1 to 65,535 bytes of UTF-8
   * - tiles over the genome
     - 2**26
   * - tile
     - a power of two from 16 to 65,536 bases
   * - tracks of a group
     - ``2**29 / tile``, so that a dense chunk of ``f8`` fits a u4 size: 32,768 at the default tile of 16,384
   * - values
     - finite binary64 or NaN; integers exact to 2**53
   * - entries per block
     - 65,536
   * - zoom levels
     - 12
   * - summary kept
     - while ``n_tracks * n_tiles * 4 <= 2**28`` bytes
   * - run bytes
     - the writer starts a new run past 2**40 bytes

Reader conformance
------------------

A reader of version 1:

- checks the header as "The header" says, and the tables' CRC-32, before
  reading a table;
- checks the chromosome table, the statistics table's extent, the
  metadata's JSON and its track records, and the directory's extent;
- checks each group's record against its CRC-32 before using its runs,
  live tiles or blocks, and each summary and zoom stream against its
  CRC-32;
- checks every count and size against ``length`` before reading or
  allocating by it, and that every zlib stream inflates to exactly the
  length the format implies, with no bytes left over;
- derives chunk offsets from the runs as "A position-order record" says,
  and refuses a record whose runs overlap, leave a live tile uncovered, or
  do not add up;
- checks the CRC-32 of a stored chunk inside a codec 1 file on every read;
- checks a chunk's payload as it decodes it: the kind and code, the length
  against the formula of "Bytes in the file" (at least that long), each
  track's varints filling exactly its ``lengths`` bytes and adding up to
  ``total``, offsets below ``tile`` (the package's reader checks that
  sparse offsets ascend only through their sum staying below ``tile``,
  and matrix offsets strictly), matrix slots ascending and in range with
  pointers in range, and a block's positions strictly ascending and
  matching its record's ``first`` and ``last``;
- treats every failure as a corrupt or unsupported file, never as a value.

A reader may skip the check of a track's varints it does not read, since
their byte count is all it needs to step over them, and may ignore the
``level``, ``order`` and ``figwig`` fields.

Appendix: a small file
----------------------

One chromosome of 20 bases, one track, a tile of 16, codec 0, no zoom
levels, and two nonzero bases: base 3 with the value 2 and base 17 with the
value 1. The first tile is a sparse ``u1`` chunk; the second is a sparse
chunk with the one code and no values. Offsets are decimal, bytes
hexadecimal.

.. code-block:: text

      0  46 49 47 57                                      magic FIGW
      4  01 00                                            version 1
      6  00 00                                            flags: position order
      8  10 00 00 00                                      tile 16
     12  00 00                                            codec 0
     14  00 00                                            level 0
     16  01 00 00 00                                      n_chroms 1
     20  01 00 00 00                                      n_tracks 1
     24  01 00 00 00                                      n_groups 1
     28  a5 32 d4 b7                                      tables_crc
     32  a0 00 00 00 00 00 00 00                          chroms 160
     40  ae 00 00 00 00 00 00 00                          tracks 174
     48  52 01 00 00 00 00 00 00                          index 338
     56  80 00 00 00 00 00 00 00                          data 128
     64  ce 00 00 00 00 00 00 00                          metadata 206
     72  84 00 00 00 00 00 00 00                          metadata_length 132
     80  00 00 00 00 00 00 00 00                          zooms 0
     88  cb 01 00 00 00 00 00 00                          length 459
     96  00 00 00 00                                      required 0
    100  01 00 00 00                                      optional 1
    104  68 84 ff c5                                      header_crc
    108  00 .. 00                                         20 bytes of padding
    128  01 00 00 00  01 00 00 00  03 00 00 00  02 00 00 00
           chunk of tile 0, kind byte 0x02 (sparse, u1): total 1; lengths [1];
           the varint 3; padding to 4; the value 2; padding to 8
    144  01 00 00 00  01 00 00 00  01 00 00 00  00 00 00 00
           chunk of tile 1, kind byte 0x3a (sparse, one): total 1; lengths [1];
           the varint 1 (base 17 is offset 1 of tile 1); padding; no values
    160  04 00  63 68 72 31  14 00 00 00 00 00 00 00      chromosome table: "chr1", 20
    174  00 00 00 00 00 00 08 40  02 00 00 00 00 00 00 00  statistics: sum 3.0, nonzero 2,
         00 00 00 00 00 00 f0 3f  00 00 00 00 00 00 00 40  min 1.0, max 2.0
    206  7b 22 66 69 67 77 69 67 22 3a 20 22 30 2e 31 2e 30 22 ...
           {"figwig": "0.1.0", "genome": null, "attributes": {}, "order":
           "positions", "tracks": [{"name": "a", "dtype": "uint8", "group": 0}]}
    338  5a 01 00 00 00 00 00 00                          directory: record of group 0 at 346
    346  01 00 00 00  f3 19 12 87                         n_runs 1, crc
    354  00 00 00 00 00 00 00 00  01 00 00 00 00 00 00 00  run: first 0, last 1,
         80 00 00 00 00 00 00 00  20 00 00 00 00 00 00 00  offset 128, bytes 32
    386  02 00 00 00  1d 00 00 00                         n_live 2, n_bytes 29
    394  78 9c 63 60 60 60 60 04 62 26 2b 01 20 89 8c 0f 47 7e f8 6b 3a ed 8e 28 00 22 19 05 43
           the live stream, inflating to:
           00 00 00 00  01 00 00 00    tiles 0, 1
           02  3a                      kind bytes
           10 00 00 00  10 00 00 00    sizes 16, 16
           10 00 00 00  10 00 00 00    raws 16, 16
           c3 59 f0 fd  35 96 dc 15    crcs
    423  02 00 00 00  14 00 00 00  59 8e 00 d9            summary: n_sum 2, n_sum_bytes 20, crc
    435  78 9c 63 60 60 60 60 64 00 03 07 06 86 06 7b 00 02 9b 01 01
           the summary stream, inflating to:
           00 00 00 00  01 00 00 00    tiles 0, 1
           00 00 00 40  00 00 80 3f    sums 2.0, 1.0
    455  00 00 00 00                                      zoom section: n_zooms 0
    459                                                   length

Appendix: three chunks
----------------------

The decoded bytes of one chunk of each remaining kind, from a tile of 64
bases (16 for the dense one), as ``figwig``'s writer produces them; the
sparse kind is in the file above.

A dense ``u2`` chunk, kind byte 0x09, of one track whose values are 0,
100, 200, ... 1500 over a tile of 16: the 16 values and nothing else,
32 bytes:

.. code-block:: text

    00 00 64 00 c8 00 2c 01 90 01 f4 01 58 02 bc 02 20 03 84 03 e8 03 4c 04 b0 04 14 05 78 05 dc 05

A runs ``f4`` chunk, kind byte 0x34, of two tracks over a tile of 64: the
first is 3 on every base, the second 2.5 on bases 10-19 and 40-49, 40
bytes:

.. code-block:: text

    05 00 00 00              total: 5 changes
    01 00 00 00 04 00 00 00  lengths: track 0's varints take 1 byte, track 1's 4
    00                       track 0: a change at 0
    0a 0a 14 0a              track 1: changes at 10, 20, 40, 50 (deltas 10, 10, 20, 10)
    00 00 00                 padding to 4
    00 00 40 40              3.0, the value of track 0 from base 0
    00 00 20 40 00 00 00 00  2.5 from base 10, 0 from base 20,
    00 00 20 40 00 00 00 00  2.5 from base 40, 0 from base 50
    00 00 00 00              padding to 8

A matrix ``u1`` chunk, kind byte 0x03, of a group of 32 tracks in a codec 0
file, of which slot 5 has the values 7 at base 3 and 9 at base 44, and
slot 20 the value 2 at base 63, 40 bytes:

.. code-block:: text

    02 00 00 00              n_present: 2
    05 00 00 00 14 00 00 00  present slots 5 and 20
    00 00 00 00 02 00 00 00 03 00 00 00
                             ptr: slot 5's entries are 0 and 1, slot 20's is 2, 3 in all
    03 00 2c 00 3f 00        offsets 3, 44, 63 as u2
    00 00                    padding to 4
    07 09 02                 the values
    00 00 00 00 00           padding to 8
