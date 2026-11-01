meta:
  id: figwig
  title: figwig file, version 1
  file-extension: fw
  license: MIT
  endian: le
  ks-version: '0.10'
doc: |
  A figwig file holds the per-base values of many tracks over one genome:
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
seq:
  - id: header
    type: header
instances:
  chrom_table:
    pos: header.chroms_offset
    type: chrom_table
    doc: The chromosome table, at the header's chroms offset.
  stats_table:
    pos: header.tracks_offset
    type: track_stats
    repeat: expr
    repeat-expr: header.n_tracks
    doc: One statistics record per track, at the header's tracks offset.
  metadata:
    pos: header.metadata_offset
    size: header.metadata_length
    type: str
    encoding: UTF-8
    doc: The metadata, one JSON object.
  index:
    pos: header.index_offset
    type: index
    doc: The directory and one record per group.
types:
  header:
    doc: The 128-byte header.
    seq:
      - id: magic
        contents: 'FIGW'
      - id: version
        type: u2
        doc: 1
      - id: flags
        type: u2
        doc: Bit 0 set for track order; other bits unknown to version 1.
      - id: tile
        type: u4
        doc: Bases per tile, a power of two from 16 to 65536.
      - id: codec
        type: u2
        doc: 0 for chunks stored as they are, 1 for zlib streams.
      - id: level
        type: u2
        doc: The compression level the chunks were written at.
      - id: n_chroms
        type: u4
      - id: n_tracks
        type: u4
      - id: n_groups
        type: u4
        doc: Equal to n_tracks in track order.
      - id: tables_crc
        type: u4
        doc: CRC-32 of the bytes from the chromosome table to the end of the index's directory.
      - id: chroms_offset
        type: u8
      - id: tracks_offset
        type: u8
        doc: Offset of the statistics table.
      - id: index_offset
        type: u8
      - id: data_offset
        type: u8
        doc: Always 128, where the first chunk may start.
      - id: metadata_offset
        type: u8
      - id: metadata_length
        type: u8
      - id: zooms_offset
        type: u8
        doc: Reserved, 0; zoom levels lie in the index records.
      - id: length
        type: u8
        doc: The bytes of the file that the format uses; later bytes are ignored.
      - id: required
        type: u4
        doc: Feature bits a reader must know; none is defined in version 1.
      - id: optional
        type: u4
        doc: |
          Feature bits a reader may ignore. Bit 0 says each record ends with
          a zoom section. Bit 1 says the file has a regional summary index,
          found through the metadata key org.figwig.summary-restarts.
      - id: header_crc
        type: u4
        doc: CRC-32 of these 128 bytes with this field as 0.
      - id: padding
        size: 20
    instances:
      track_order:
        value: flags & 1 != 0
      has_zooms:
        value: optional & 1 != 0
      has_summary_index:
        value: optional & 2 != 0
  chrom_table:
    seq:
      - id: entries
        type: chrom
        repeat: expr
        repeat-expr: _root.header.n_chroms
  chrom:
    seq:
      - id: len_name
        type: u2
      - id: name
        type: str
        size: len_name
        encoding: UTF-8
      - id: length
        type: u8
        doc: Bases, from 1 to 2**32 - 1.
  track_stats:
    doc: One track's statistics over its non-NaN bases.
    seq:
      - id: sum
        type: f8
      - id: nonzero
        type: u8
      - id: min
        type: f8
      - id: max
        type: f8
  index:
    seq:
      - id: directory
        type: u8
        repeat: expr
        repeat-expr: _root.header.n_groups
        doc: The offset of each group's record, which follow back to back.
      - id: records
        type: record(_index)
        repeat: expr
        repeat-expr: _root.header.n_groups
  record:
    doc: |
      One group's record: in position order its runs and live tiles, in
      track order its blocks, then its summary and, when the header's
      optional bit 0 is set, its zoom section. `g` is the group's number,
      its place in the directory.
    params:
      - id: g
        type: u4
    seq:
      - id: num_entries
        type: u4
        doc: n_runs in position order, n_blocks in track order.
      - id: crc
        type: u4
        doc: |
          CRC-32 of the record's bytes from here to the summary: the runs,
          n_live, len_live and the live stream in position order, the
          block records in track order.
      - id: blocks
        type: block_record
        repeat: expr
        repeat-expr: num_entries
        if: _root.header.track_order
      - id: runs
        type: run
        repeat: expr
        repeat-expr: num_entries
        if: not _root.header.track_order
      - id: n_live
        type: u4
        if: not _root.header.track_order
      - id: len_live
        type: u4
        if: not _root.header.track_order
      - id: live
        size: len_live
        process: zlib
        type: live_tiles(n_live)
        if: not _root.header.track_order
        doc: A zlib stream of the live tiles, field by field.
      - id: summary
        type: summary
      - id: zooms
        type: zoom_section
        if: _root.header.has_zooms
    instances:
      chunk_runs:
        pos: 0
        type: run_list(g, 0, 0)
        if: not _root.header.track_order and num_entries > 0
        doc: The chunks of every run, in run order, each with its live index.
  run:
    doc: The chunks of tiles first to last, back to back from offset for bytes bytes.
    seq:
      - id: first
        type: u8
      - id: last
        type: u8
      - id: offset
        type: u8
      - id: bytes
        type: u8
  live_tiles:
    doc: |
      The tiles that hold values, ascending, with each one's kind byte,
      compressed size, uncompressed size and the CRC-32 of its bytes as
      they lie in the file; laid out as five arrays, one per field.
    params:
      - id: num_tiles
        type: u4
    seq:
      - id: tiles
        type: u4
        repeat: expr
        repeat-expr: num_tiles
      - id: kinds
        type: u1
        repeat: expr
        repeat-expr: num_tiles
      - id: sizes
        type: u4
        repeat: expr
        repeat-expr: num_tiles
      - id: raws
        type: u4
        repeat: expr
        repeat-expr: num_tiles
      - id: crcs
        type: u4
        repeat: expr
        repeat-expr: num_tiles
  run_list:
    doc: |
      The runs of a record from run r on, each run's chunks sized by the
      live tiles in order: a run's chunks are the live entries from
      live_start on, as many as fill its bytes.
    params:
      - id: g
        type: u4
      - id: r
        type: u4
      - id: live_start
        type: u4
    instances:
      run:
        value: _root.index.records[g].runs[r]
      chunks:
        pos: _root.index.records[g].runs[r].offset
        size: _root.index.records[g].runs[r].bytes
        type: run_chunks(g, live_start)
      next:
        pos: 0
        type: run_list(g, r + 1, live_start + chunks.chunks.size)
        if: r + 1 < _root.index.records[g].num_entries
  run_chunks:
    params:
      - id: g
        type: u4
      - id: live_start
        type: u4
    seq:
      - id: chunks
        type: chunk_bytes(g, live_start + _index)
        size: _root.index.records[g].live.sizes[live_start + _index]
        repeat: eos
  chunk_bytes:
    doc: |
      The bytes of one chunk as they lie in the file: a zlib stream in a
      compressed file unless its kind byte's high bit says it is stored as
      it is. `inflated` holds the chunk's bytes in either case; decode
      them with `chunk`, given the group's number of tracks.
    params:
      - id: g
        type: u4
      - id: live_index
        type: u4
    seq:
      - id: data
        size-eos: true
    instances:
      kind_byte:
        value: _root.index.records[g].live.kinds[live_index]
      tile:
        value: _root.index.records[g].live.tiles[live_index]
      stored:
        value: kind_byte & 0x80 != 0 or _root.header.codec == 0
      inflated_stream:
        pos: 0
        size-eos: true
        process: zlib
        if: not stored
      inflated:
        value: "stored ? data : inflated_stream"
  block_record:
    doc: One track-order block, its chromosome, first and last position, entries, place and kind.
    seq:
      - id: chrom
        type: u4
      - id: first
        type: u4
      - id: last
        type: u4
      - id: num_entries
        type: u4
      - id: offset
        type: u8
      - id: len_body
        type: u4
        doc: The block's bytes in the file, compressed in a codec 1 file.
      - id: raw
        type: u4
        doc: The block's bytes once decoded.
      - id: kind
        type: u4
        doc: The kind byte, in a u4; kind 1 with an element code.
      - id: crc
        type: u4
    instances:
      body:
        pos: offset
        size: len_body
        type: block(num_entries, kind)
        if: _root.header.codec == 0
      body_deflated:
        pos: offset
        size: len_body
        process: zlib
        type: block(num_entries, kind)
        if: _root.header.codec == 1
  summary_index:
    doc: |
      The optional regional summary index, announced by header optional
      bit 1: the root and one locator per group. The root's offset is the
      "offset" of the metadata's org.figwig.summary-restarts object, JSON
      outside Kaitai's reach. A level catalog is a locator's levels; a
      descriptor page is parsed by summary_page.
    params:
      - id: root_offset
        type: u8
    instances:
      root:
        pos: root_offset
        type: summary_root
      locators:
        pos: root.directory_offset
        type: summary_locator
        repeat: expr
        repeat-expr: root.n_groups
  summary_root:
    doc: The 256-byte root of the index.
    seq:
      - id: magic
        contents: ['FWRSUM1', 0]
      - id: version
        type: u2
        doc: The extension's own version, 2.
      - id: len_root
        type: u2
        doc: 256.
      - id: crc
        type: u4
        doc: CRC-32 of these 256 bytes with this field as 0.
      - id: directory_offset
        type: u8
        doc: Offset of the group locators, this root's offset plus 256.
      - id: n_groups
        type: u4
        doc: The header's number of groups.
      - id: descriptors_per_page
        type: u4
        doc: 102.
      - id: window_target
        type: u4
        doc: Decoded bytes a window targets, 32768.
      - id: reserved_a
        size: 28
      - id: header_snapshot
        size: 128
        doc: The file's final 128-byte header; an index whose snapshot differs from the header is stale.
      - id: reserved_b
        size: 64
  summary_locator:
    doc: A group's 32-byte locator in the index directory.
    seq:
      - id: levels_offset
        type: u8
        doc: Offset of the group's level catalog.
      - id: n_levels
        type: u4
        doc: 1 to 13, the base summary then each zoom level.
      - id: levels_crc
        type: u4
        doc: CRC-32 of the catalog.
      - id: group
        type: u4
      - id: n_tracks
        type: u4
      - id: crc
        type: u4
        doc: CRC-32 of these 32 bytes with this field as 0.
      - id: reserved
        type: u4
    instances:
      levels:
        pos: levels_offset
        type: summary_level
        repeat: expr
        repeat-expr: n_levels
  summary_level:
    doc: A level's entry in a catalog; its position gives the level, 0 the base summary.
    seq:
      - id: source_offset
        type: u8
        doc: Offset of the level's version 1 zlib stream.
      - id: index_offset
        type: u8
        doc: Offset of the first descriptor page, 0 for an ordinary level.
      - id: source_rows
        type: u4
      - id: source_bytes
        type: u4
      - id: source_crc
        type: u4
      - id: window_bins
        type: u4
        doc: A power of two up to 2048, 0 for an ordinary level.
  summary_page:
    doc: |
      A descriptor page, at index_offset + 4096 * p for page p. It holds
      min(102, windows - 102 * p) descriptors, where windows is the level's
      bins over the genome divided by window_bins, rounded up.
    params:
      - id: num_windows
        type: u4
    seq:
      - id: magic
        contents: 'FWRP'
      - id: first_window
        type: u4
      - id: count
        type: u4
      - id: crc
        type: u4
        doc: CRC-32 of the page's bytes with this field as 0.
      - id: windows
        type: summary_window
        repeat: expr
        repeat-expr: num_windows
  summary_window:
    doc: One window's descriptor, all zero for a window with no rows.
    seq:
      - id: n_rows
        type: u4
      - id: first_source_row
        type: u4
      - id: ids_offset
        type: u4
        doc: Offset of the compressed ID fragment from the first byte of the level's zlib stream, at least 2 because of the stream's header.
      - id: ids_bytes
        type: u4
      - id: ids_crc
        type: u4
        doc: CRC-32 of the compressed fragment.
      - id: ids_raw_crc
        type: u4
        doc: CRC-32 of the decoded fragment.
      - id: sums_offset
        type: u4
      - id: sums_bytes
        type: u4
      - id: sums_crc
        type: u4
      - id: sums_raw_crc
        type: u4
  summary:
    doc: The tiles of a group with a nonzero sum, and each track's float32 sum over them.
    seq:
      - id: n_sum
        type: u4
      - id: len_stream
        type: u4
        doc: 0 when the file keeps no summary for the group.
      - id: crc
        type: u4
        doc: CRC-32 of the stream.
      - id: stream
        size: len_stream
        process: zlib
        type: summary_stream(n_sum)
        if: len_stream > 0
  summary_stream:
    params:
      - id: num_tiles
        type: u4
    seq:
      - id: tiles
        type: u4
        repeat: expr
        repeat-expr: num_tiles
      - id: sums
        type: f4
        repeat: eos
        doc: n times the group's tracks, tile by tile.
  zoom_section:
    seq:
      - id: n_zooms
        type: u4
      - id: levels
        type: zoom_level
        repeat: expr
        repeat-expr: n_zooms
  zoom_level:
    doc: Level k, from 1, sums the tile sums over bins of 4**k tiles.
    seq:
      - id: factor
        type: u4
      - id: n
        type: u4
      - id: len_stream
        type: u4
      - id: crc
        type: u4
        doc: CRC-32 of the stream.
      - id: stream
        size: len_stream
        process: zlib
        type: zoom_stream(n)
  zoom_stream:
    params:
      - id: num_bins
        type: u4
    seq:
      - id: bins
        type: u4
        repeat: expr
        repeat-expr: num_bins
      - id: sums
        type: f8
        repeat: eos
        doc: n times the group's tracks, bin by bin.
  block:
    doc: A track-order block's decoded bytes.
    params:
      - id: num_positions
        type: u4
      - id: kind
        type: u4
    seq:
      - id: positions
        type: u4
        repeat: expr
        repeat-expr: num_positions
      - id: padding
        size: (8 - 4 * num_positions % 8) % 8
      - id: values
        type: element_values(code, num_positions)
    instances:
      code:
        value: (kind >> 3) & 15
  chunk:
    doc: A chunk's decoded bytes, by its kind byte, for a group of n_tracks tracks.
    params:
      - id: kind_byte
        type: u1
      - id: n_tracks
        type: u4
      - id: tile
        type: u4
    seq:
      - id: body
        type:
          switch-on: kind
          cases:
            1: dense(code, n_tracks, tile)
            2: sparse(code, n_tracks)
            3: matrix(code)
            4: sparse(code, n_tracks)
    instances:
      kind:
        value: kind_byte & 7
      code:
        value: (kind_byte >> 3) & 15
      stored:
        value: kind_byte & 0x80 != 0
  dense:
    doc: Every value of the tile, track after track.
    params:
      - id: code
        type: u1
      - id: n_tracks
        type: u4
      - id: tile
        type: u4
    seq:
      - id: values
        type: element_values(code, n_tracks * tile)
  sparse:
    doc: |
      A sparse or runs chunk: the entries in all, the bytes of each
      track's varints, each track's offsets as LEB128 deltas, padding to
      4 bytes, and the values in the element type.
    params:
      - id: code
        type: u1
      - id: n_tracks
        type: u4
    seq:
      - id: total
        type: u4
      - id: lengths
        type: u4
        repeat: expr
        repeat-expr: n_tracks
      - id: tracks
        type: track_offsets
        size: lengths[_index]
        repeat: expr
        repeat-expr: n_tracks
      - id: padding
        size: (4 - _io.pos % 4) % 4
      - id: values
        type: element_values(code, total)
  track_offsets:
    seq:
      - id: deltas
        type: varint
        repeat: eos
  matrix:
    doc: |
      The tracks present, a pointer per present track into the entries
      plus one at the end, the entries' offsets within the tile, padding
      to 4 bytes, and the values.
    params:
      - id: code
        type: u1
    seq:
      - id: n_present
        type: u4
      - id: tracks
        type: u4
        repeat: expr
        repeat-expr: n_present
      - id: ptr
        type: u4
        repeat: expr
        repeat-expr: n_present + 1
      - id: offsets
        type: u2
        repeat: expr
        repeat-expr: ptr[n_present]
      - id: padding
        size: (4 - _io.pos % 4) % 4
      - id: values
        type: element_values(code, ptr[n_present])
  element_values:
    doc: |
      n values of the element type, as an empty list for the "one" type,
      7, which stores none and whose every value is 1.
    params:
      - id: code
        type: u1
      - id: n
        type: u4
    seq:
      - id: values
        type:
          switch-on: code
          cases:
            0: u1
            1: u2
            2: u4
            3: s1
            4: s2
            5: s4
            6: f4
            7: u1
            8: f8
        repeat: expr
        repeat-expr: "code == 7 ? 0 : n"
  varint:
    doc: A LEB128 varint of at most three bytes, low seven bits first.
    seq:
      - id: groups
        type: varint_group
        repeat: until
        repeat-until: not _.has_next
    instances:
      value:
        value: "groups[0].low + (groups.size >= 2 ? (groups[1].low << 7) : 0) + (groups.size >= 3 ? (groups[2].low << 14) : 0)"
  varint_group:
    seq:
      - id: b
        type: u1
    instances:
      has_next:
        value: b & 0x80 != 0
      low:
        value: b & 0x7f
