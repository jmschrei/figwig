# Writing figwig files

`FigWigWriter(path, chrom_sizes, tracks, genome=None, attributes=None, tile=None, groups=8, level=None, engine='auto', n_jobs=8, order=None, append=False, zooms=None, summary_index=False)`
with `write(chroms, starts, values, ends=None, tracks=None)` and
`write_sparse(chroms, positions, tracks, values)`; `write_figwig` writes one
call's worth. `tile` None is 16,384, `order` None is `'positions'`, `level`
None is 6 (0 in track order), `zooms` None is 0 (the file's with `append`).

Enable `summary_index=True` for large multitrack files intended for interactive
browser use. Indexed summaries are practically necessary to avoid downloading
and decompressing complete genome-wide levels for small viewports. Python also
uses the index automatically for selected-chromosome `tile_sums` and exact
integer `sums`. It preserves v1 values and old-reader access; the default is off.
Tiny levels or a full index budget retain ordinary streams.
It cannot be combined with `append=True`; ordinary append invalidates the index,
and merge/reorder do not propagate it. See `docs/summary_restarts.rst` and
`benchmarks/compare_summaries.py` in the source checkout.

```python
import numpy
from figwig import FigWigWriter

chrom_sizes = {"chr1": 248956422, "chr2": 242193529}
tracks = [{"name": "atac.+", "assay": "ATAC-seq", "strand": "+"},
	{"name": "atac.-", "assay": "ATAC-seq", "strand": "-"}]
chroms = numpy.array(["chr1", "chr1", "chr2"])
starts = numpy.array([1000, 5000, 2000])
y_hat = numpy.random.rand(3, 2, 2114).astype("float32")  # (n, tracks, width)
with FigWigWriter("k562.fw", chrom_sizes, tracks, genome="hg38") as writer:
	writer.write(chroms, starts, y_hat)        # whole groups at a time
```

## Tracks and values

- `tracks` are names or dicts with `name`, optional `dtype` (uint8, uint16,
  uint32, int8, int16, int32, float32, float64) and JSON-serializable
  attributes (no NaN; `group` is set by the writer). Left out, `dtype` is
  set at close to the narrowest type holding the track's values: an
  integer type, float32 for a fraction, float64 for an integer beyond
  uint32. Given, every value is checked against it. Storage is the same
  either way: each chunk is as narrow as its own values, so declaring
  `uint32` costs nothing.
- A float32 track keeps float32's rounding, so does an undeclared track
  that turns out to hold fractions; a value beyond float32's range raises
  unless the track is float64; -0.0 is 0; integers beyond 2**53 and
  infinite values are refused before anything is written; NaN is a value in
  a float track and an error in an integer one.
- Zeros are not stored, so a bigWig's "no data" and 0 are the same thing;
  `bw2fw --missing nan` keeps them apart (`references/converters.md`).
- Each `write` covers whole groups. Items may come in any order within a
  call but must not overlap; within a group later calls come after earlier
  ones on a chromosome, and a group cannot return to a chromosome it left.
  A window may run past its chromosome's end only where its values are NaN
  or 0. A call that raises writes nothing.
- `write` with `ends=` (intervals) encodes on the calling thread, whole-tile
  windows on the pool: write windows for speed.
- The file is written beside its path and moved there at `close`, so a
  failed write leaves the path as it was and a converter may write over its
  input. A writer never closed warns and writes nothing. The chunks depend
  on the values and on which calls wrote which groups, not on `n_jobs` or
  on how values were split between calls.

## Tile, groups and order

Tracks of one *group* are stored and decoded together: reading one track of
a group of 8 decodes all 8, and a chunk's element type is chosen for the
whole group, so a count track beside a float track is stored as floats.
`groups=` is a count of consecutive tracks or a list of lists of names.

| Tracks | Layout | Why |
|---|---|---|
| count tracks (the default case) | tile 16,384, groups of 8, or one group of all | a sparse chunk's other tracks are stepped over cheaply, so one group of 48 reads 8 of them faster than 48 groups of 1 |
| dense float signal read in short windows (conservation, fold change, p-value) | the defaults; `tile=1024, groups=1` (`bw2fw -t 1024 --groups 1`) only for one-thread reads | at the defaults three chr22 conservation tracks read 1.6 to 3.2x faster than their bigWigs on 1 to 32 threads; the small tile reads a third faster on one thread for a file a quarter larger and is slower from 8 threads on |
| cells | one group of all cells, `level=0` for the fastest reads or compressed for half the size | tiles with few cells present are matrix chunks either way; cut sites whose values are all 1 store no values |
| reads (Fiber-seq), one track per read and channel | position order, groups of about 1,000 consecutive reads sorted by start, per channel | a chunk holds at most 2**29 values (tracks × tile), so one group of every read is refused, and a group per read is hundreds of times slower. Position order beats track order for sums over reads by about 40x |
| a few tracks read over whole chromosomes (`read_entries`) | `order='tracks'` (`'csr'`), raw unless `level` is given | each track's nonzero bases in blocks per chromosome; a window read over thousands of such tracks decodes a block per track and is slow |

A read may also be its own "chromosome" of its length with one track per
channel, the genome coordinates in the attributes. Thousands of single-track
groups cost a group's index each (17 bytes per tile of the genome in
memory): use one group, or track order. `figwig reorder` rewrites a file in
the other order, or at another tile or groups. `level=0` stores every chunk
raw, the fastest to read, at 4 bytes per base of a dense float track;
compressed chunks inflate twice as fast with the `deflate` package.
`tile` is a power of two from 16 to 65,536 with at most 2**26 tiles over
the genome.

## Nonzero bases as entries

```python
import numpy
from figwig import FigWigWriter

chrom_sizes = {"chr1": 248956422}
cells = [("AAACGAAAGCGCAATG", "CD4 T"), ("AAACGAAAGGGTATCG", "B")]
tracks = [{"name": barcode, "dtype": "uint8", "cell_type": ct} for barcode, ct in cells]
# per chromosome: the cut sites' positions, the index of the cell (track) of each, the counts
positions = {"chr1": numpy.array([10_000, 10_000, 250_000])}
cell_ids = {"chr1": numpy.array([0, 1, 1])}
counts = {"chr1": numpy.array([1, 1, 2])}
with FigWigWriter("pbmc.fw", chrom_sizes, tracks, groups=len(tracks), level=0) as writer:
	for chrom in chrom_sizes:
		writer.write_sparse(chrom, positions[chrom], cell_ids[chrom], counts[chrom])
```

- Entries are (chrom, position, track, value), the form of fragment files,
  sparse-matrix triplets and `read(sparse=True)`. No two may name one base
  of one track: fragment files give two cut sites per fragment and cells
  repeat positions, so sum duplicates (`numpy.unique` on the key, then
  `bincount`) before writing.
- A call takes about 90 bytes of memory per entry, so write a chromosome
  at a time. Entries come after the group's earlier items on that
  chromosome, and in position order seal the tile they end in: the windows
  and entries of one tile must come in one call.

## Adding tracks, merging, checking

- `FigWigWriter(path, None, new_tracks, append=True)` adds tracks to a file
  in place, in new groups after its own: the new chunks go after the file's
  end and the header is written last, so a writer that fails or is killed
  leaves the file readable as it was. Its chromosomes, tile, level and
  order are taken (`chrom_sizes`, `tile`, `level`, `order` must agree or be
  None); `attributes=` adds to the file's; the old tables and index stay
  behind as dead bytes, which `figwig merge` into a new path drops. One
  appending writer at a time; a reader already open keeps reading until
  the append closes.
- `writer.add_file(path)` copies every track of another file written with
  the same chromosomes, tile, level and order, chunks and all; `figwig
  merge` is that from the shell. Tracks that came from a file cannot be
  written to.
- `FigWigReader(path, verify=True)` checks every chunk's CRC-32 as it is
  read; compressed chunks are checksummed by zlib on every read anyway, so
  it is for uncompressed files. The header, tables and index are checked
  at open either way.
