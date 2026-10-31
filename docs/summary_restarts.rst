Optional regional summary index
===============================

Indexed summaries are practically necessary for responsive interactive browser
use of large multitrack files. The `figwig-js <https://github.com/genome-spy/figwig-js/>`_
reader uses them to fetch and decompress windows for the requested region,
instead of complete genome-wide summary levels. Windows target 32 KiB decoded
bytes, capped at 2,048 bins. The :doc:`core format <format>` remains v1;
unmodified readers use the ordinary zlib summary streams.

Usage and limits
----------------

Enable the index with ``summary_index=True`` on ``FigWigWriter`` or
``write_figwig``; the default is ``False``.

.. code-block:: python

    with FigWigWriter("tracks.fw", chrom_sizes, tracks, tile=1024,
                      zooms=4, summary_index=True) as writer:
        writer.write("chr1", [0], values)

No reader flag is needed:

.. code-block:: python

    reader = FigWigReader("tracks.fw")
    chr1 = reader.tile_sums(chroms="chr1", level=1)
    totals = reader.sums("chr1", [100_001, 8_000_003],
                         [120_007, 8_020_009], tracks="counts")

Python reads local files. ``tile_sums`` allocates only selected bins, preserving
order, duplicates and float32/float64 values. Integer ``sums`` reads the union
of whole-tile interiors and raw boundary pieces, falling back to raw values when
a stored tile sum's magnitude reaches 2**24 or peak times interval length reaches
2**53. Floating-point ``sums``, ``read(..., bins=...)`` and NaN counting use raw
values. Only checked catalogs are cached; pickle drops them. No file descriptor
or decoded window is retained.

The writer keeps a level ordinary if both compressed and decoded sizes are at
most 64 KiB, one ID-plus-sum row exceeds 32 KiB, or its directory would exceed
its budget. The 16 MiB directory budget includes all structures; page allowance
is divided deterministically among groups, finest levels first, without reassigning unused
shares. The existing summary cutoff, ``n_tracks * n_tiles * 4 <= 2**28``, still
applies. If no level is indexed, no root, locator or optional bit is emitted.

``summary_index=True`` with ``append=True`` raises before opening the file.
Ordinary append clears the accelerator bit and preserves foreign metadata,
including the stale locator. Merge and reorder discard the index.

Discovery and checksums
-----------------------

All integers are little-endian. Offsets are absolute ``u8`` byte offsets except
fragment offsets, which are relative ``u4`` offsets in one legacy zlib stream.
The locator is::

    "org.figwig.summary-restarts": {"version": 2, "offset": ROOT, "length": 256}

Discovery requires optional bit ``0x2`` and this key. Missing keys and unknown
versions use ordinary v1 reads; recognized malformed structures raise errors.
A valid root with a different header snapshot is stale and may be ignored.
Discovery uses only the v1 header and tables, without a file-size probe or scan.
The optional bit is an extension convention, not an upstream-assigned feature.

The root, group directory, catalogs and descriptor pages occupy unused data
region bytes before the chromosome table. Root offset is at least 128; every
directory section lies within ``[ROOT + 256, chroms)`` and sections do not
overlap. They cannot extend the legacy group records or use the reserved zoom
offset. Existing group records remain back to back through declared ``length``.

The existing tables CRC protects the metadata locator and legacy group offsets.
The root and each group locator/page have their own CRC-32, and each group
locator protects its level catalog. A CRC field is zeroed when computing its
structure's checksum. A fragment stores CRCs of its compressed and decoded bytes.
All of these checks apply to a requested structure even when optional raw-data
verification is disabled.

Root, 256 bytes
---------------

========  ====================================================================
Offset    Field
========  ====================================================================
0         Eight bytes ``FWRSUM1\0``
8         Extension version ``u2 = 2``; root bytes ``u2 = 256``
12        Root CRC-32 ``u4``
16        Group-directory offset ``u8``, exactly ``ROOT + 256``
24        Group count ``u4``, equal to the core header's
28        Descriptors per page ``u4 = 102``
32        Decoded window target ``u4 = 32768``
36        28 reserved zero bytes
64        Exact 128-byte final v1 header
192       64 reserved zero bytes
========  ====================================================================

The header snapshot binds the index to the file generation. The writer reserves
the directory, writes tables and index, computes the final header, fills the
root and directory, then publishes the header last. Metadata holds only the
root locator, avoiding checksum cycles. The writer's temporary file preserves
the previous path on failure.

Group directory and level catalogs
----------------------------------

Group ``g`` has a 32-byte locator at ``ROOT + 256 + 32*g``::

    levels_offset u8
    level_count u4; levels_crc u4; group_id u4; track_count u4
    locator_crc u4; reserved u4 = 0

``group_id`` and ``track_count`` must match the v1 metadata. ``level_count`` is
from 1 to 13: base summary followed by the actual zooms in increasing level
order. Check the locator's CRC at byte 24 before following its catalog pointer.
The catalog occupies exactly ``32 * level_count`` bytes, checked against
``levels_crc``. Each 32-byte entry is::

    source_offset u8; index_offset u8
    source_rows u4; source_bytes u4; source_crc u4; window_bins u4

The entry's position gives its level and factor: level 0 has factor 1 and
``f4`` sums; level ``k > 0`` has factor ``4**k`` and ``f8`` sums. The source
offset/count/length/CRC match the existing legacy payload header, and its complete
extent lies in that group's v1 record. Source byte count zero means absent
summary; count zero with an actual empty zlib stream means available empty.
An empty available stream still needs its ordinary CRC/Adler checks before
returning zeros.

``index_offset = window_bins = 0`` means an ordinary level. A restarted level
has a nonzero index offset and ``window_bins`` equal to a power of two at most
2,048. Choose the largest such ``B`` satisfying
``B * (4 + width*track_count) <= 32768``, where width is 4 or 8. The writer
does not accelerate a level whose one row already exceeds that target.

Arithmetic windows and descriptor pages
---------------------------------------

Use the existing level-global bin IDs. The number of bins is the sum, over
chromosomes, of ``ceil(chromosome_tiles / factor)``. A storage window is
``global_bin_id // B``. A storage window can include bins from neighboring
chromosomes; summary bins themselves never cross a chromosome boundary. Regional
readers compute a chromosome's existing global-bin offset and filter the decoded
IDs using the requested interval. Windows group coordinates, not sparse row
counts, so a sparse gap cannot make a window span an unbounded number of bins.

The descriptor array contains ``ceil(total_bins / B)`` entries, including empty
windows. Page ``p`` is at ``index_offset + p*4096`` and starts with::

    magic: four bytes FWRP
    first_window u4; count u4; page_crc u4

The page holds up to 102 entries of 40 bytes. Its first window is ``p*102``;
its count is the smaller of 102 and the remaining window count. The last page
is trimmed to ``16 + 40*count`` bytes. Check the page CRC at byte 12 before
using a descriptor. Window ``j`` is slot ``j % 102`` in page ``j // 102``.

Each descriptor consists of ten ``u4`` values::

    n_rows; first_source_row
    ids_relative_offset; ids_compressed_bytes; ids_crc; ids_raw_crc
    sums_relative_offset; sums_compressed_bytes; sums_crc; sums_raw_crc

An empty descriptor has all ten fields zero and needs no fragment read. Otherwise
``n_rows`` is positive, at most the window's logical bin count, and
``first_source_row + n_rows <= source_rows``. IDs are strictly ascending and
lie in that global window and below the level's total bin count. Track values
retain v1 slot order. The ID fragment decodes to exactly ``4*n_rows`` bytes;
the sum fragment decodes to exactly ``width*track_count*n_rows`` bytes.

Fragments have positive compressed size, do not overlap, and stay inside their
source stream excluding its two-byte header and four-byte trailer. Each compressed
fragment has at most ``2*expected_decoded_bytes + 64`` bytes, a conservative bound
for this encoder. Check arithmetic and configured read/allocation limits before
requesting bytes or allocating output. The directory is dense but fetched in
bounded pages; it is never necessary to download a whole level's descriptor array.

Restart compression and decoding
--------------------------------

The complete legacy payload is still exactly all listed ``u4`` IDs, then all
row-major sums, wrapped by one RFC 1950 zlib stream with its normal Adler-32 and
legacy compressed-stream CRC. The stream uses a 32 KiB DEFLATE window, no preset
dictionary, and ``Z_FULL_FLUSH`` after each occupied window's ID bytes, then
after each occupied window's sum bytes. Empty windows emit no fragments. The
complete stream finishes normally. Do not re-round base sums or recompute zooms
from them; the writer compresses the existing level's actual bytes.

A fragment starts immediately after a full history reset (or after the initial
two-byte zlib header) and includes its ending full-flush marker ``00 00 ff ff``.
It has no independent zlib wrapper. Decode it with a fresh raw-DEFLATE context,
no dictionary, and an appended empty final block ``03 00``. Require the marker,
a true DEFLATE end, exact output length, complete input consumption, and both
CRCs. Limit output while inflating; checking the length only afterward is unsafe.
The artificial final block provides framing, not integrity. ``Z_SYNC_FLUSH``
does not reset history and cannot provide independent restart points.

Regional reads validate the selected directory pages and fragments. They cannot
check the original whole-stream CRC/Adler without reading the whole stream;
corruption in an unread fragment is deferred until that fragment or the complete
stream is requested. Python uses complete streams when a query touches at least
half of a level's windows, and retains its existing complete-stream path for
whole-genome queries. Complete-stream reads and unmodified v1 readers retain
their CRC/Adler checks.
