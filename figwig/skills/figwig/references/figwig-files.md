# figwig files: many tracks in one file

A figwig file (`.fw`) holds the per-base values of many tracks over one
genome, each with a JSON record of attributes, and reads them the way
`read_bigwig` reads bigWigs: many windows of one width in one call, from
any subset of the tracks, into `(windows, tracks, width)`. Values come back
in each track's own dtype, uint8 or uint16 for most count tracks, not
float32. Use one when a model trains on several tracks at once, or when a
directory of bigWigs is read in windows again and again.

## Signatures

```python
FigWigReader(path)
reader.read(chroms, starts, width, tracks=None, out=None, dtype=None, n_jobs=8,
	sparse=False, labels=None, bins=None)
reader.read_entries(tracks=None, chroms=None, n_jobs=8)   # -> SparseEntries
reader.tile_sums(tracks=None, chroms=None)  # -> float32 (tracks, tiles): sums per tile, no decoding
reader.select(**attributes)          # -> int64 indices, for `tracks`
reader.tracks                        # list of dicts: name, dtype, group, attributes
reader.stats                         # structured array: sum, nonzero, min, max per track
reader.chrom_sizes, reader.genome, reader.attributes, reader.tile, reader.order
read_figwig(path_or_reader, chroms, starts, width, tracks=None, out=None, dtype=None,
	n_jobs=8, sparse=False, labels=None, bins=None)

FigWigWriter(path, chrom_sizes, tracks, genome=None, attributes=None, tile=16384,
	groups=8, level=None, engine='auto', n_jobs=8, order='positions', append=False)
writer.write(chroms, starts, values, ends=None, tracks=None)
writer.write_sparse(chroms, positions, tracks, values)   # nonzero bases as entries
writer.add_file(path)                # copy another file's tracks in (same chroms, tile, level, order)
writer.close()                       # writes the tables and index; `with` calls it
FigWigReader(path, verify=True)      # check every chunk's CRC-32; for uncompressed files
write_figwig(path, chrom_sizes, tracks, chroms, starts, values, ends=None, genome=None,
	attributes=None, tile=16384, groups=8, level=None, engine='auto', n_jobs=8,
	order='positions')
```

| `read(...)` returns | when |
|---|---|
| dense `(n, tracks, width)` in the tracks' dtype | default; raises past 8 GiB unless `out=` is given |
| `SparseWindows(rows, channels, offsets, values, shape)` | `sparse=True`; sorted by row, channel, offset |
| dense `(n, labels, width)` float32 sums | `labels=` one int per track read, -1 skips |
| dense `(n, tracks, width // bins)` float32 sums per bin | `bins=` a divisor of `width`; `bins=width` is each window's total |

| `order=` | layout | fast | slow |
|---|---|---|---|
| `'positions'` / `'csc'` (default) | tiles across tracks, groups' tiles in one chunk | windows across many tracks, pseudobulks | one track over the genome |
| `'tracks'` / `'csr'` | each track's nonzero bases in blocks per chromosome, raw by default | `read_entries` of a few tracks | windows over thousands of tracks (one block per track per window) |

`figwig reorder in.fw out.fw --order tracks|positions` rewrites a file in the
other order, keeping its tile, groups and level; `out.fw` may be `in.fw`.
`figwig merge -o all.fw a.fw b.fw` copies several files' tracks into one
(same chromosomes, tile, level and order; `reorder` first otherwise).
`FigWigWriter(path, None, new_tracks, append=True)` adds tracks to a file;
`tile_sums()` is the file's coarse profile, in place of zoom levels, kept
while tracks x tiles x 4 bytes is at most 256 MB.

| `tracks` given to `read` | output shape | dtype |
|---|---|---|
| `None` | `(n, all tracks, width)` | the tracks' own if shared, else the narrowest type holding every one exactly (float32 for uint8 + float32, float64 for uint32 + float32) |
| a list of names or indices | `(n, len(tracks), width)`, channel `i` is `tracks[i]` | as above |
| one name or index | `(n, width)` | that track's |

`dtype=` asks for another type; an integer type that cannot hold a track's
values, or a sum, raises `ValueError`. A base never written is 0; a base
past the end of its chromosome is NaN in a float output and 0 in an integer
one. An unknown chromosome raises: a figwig file holds every chromosome it
was written with, so there is no warning as for bigWigs. A read uses one
thread per 64 chunks it touches, up to `n_jobs`, so small reads run on the
calling thread. A reader checks on every read that the file at its path is
the one it opened, and raises if a conversion replaced it.

## Reading for training

```python
import numpy
from figwig import FigWigReader

fw = FigWigReader("k562.fw")
tracks = fw.select(assay="ATAC-seq")           # e.g. [0 1]
y = fw.read(chroms, starts, width=2114, tracks=tracks)
# (n, 2, 2114) uint16 for count tracks
counts = fw.read(chroms, starts, width=2114, tracks=tracks, bins=2114)[:, :, 0]
binned = fw.read(chroms, starts, width=196_608, bins=128)   # (n, tracks, 1536)
```

- `select` matches equality on the keys of the track records: `name`,
  `dtype`, `group` and whatever attributes the file was written with (`assay`,
  `strand`, `biosample`, ...). Look at `fw.tracks` to see which exist;
  attributes from a metadata table are strings, so `select(units=1)` matches
  nothing where `select(units="1")` would.
- `out=` may be a numpy view of a pinned torch tensor of the output's shape
  and dtype; cast to float on the GPU, where a uint16 batch is half the bytes
  of float32. Integer outputs wrap in numpy and torch arithmetic the way any
  uint8 does; pass `dtype='float32'` for code ported from `read_bigwig`.
- A `FigWigReader` pickles without its index, which each worker reads
  again, and can be read from several threads, so it goes in a Dataset the
  way `references/reading.md` shows for `BigWigReader`; DataLoader workers
  do not make it faster, since one process reads about 70,000 windows per
  second, and under `spawn` each worker loads the compiled kernels.
- Tracks of one *group* are stored together and decoded together, so reading
  one track of a group of 8 decodes all 8. Files meant to be read a track at a
  time should be written with small groups, and groups should hold tracks of
  one kind: a count track grouped with a float track is stored as float
  values and decoded with it.
- The first dense read of a fresh install compiles for about 27 s, the first
  sparse read for 21 s more, a second output dtype for 12 s and track order
  for 12 s; later processes load the cache in a tenth of a second. Warm up
  before timing or spawning workers.

## Single cells and other sparse track sets

```python
tracks = [{"name": barcode, "dtype": "uint8", "cell_type": ct} for barcode, ct in cells]
with FigWigWriter("pbmc.fw", chrom_sizes, tracks, groups=len(tracks), level=0) as writer:
	for chrom in chrom_sizes:
		writer.write_sparse(chrom, positions[chrom], cell_ids[chrom], counts[chrom])

fw = FigWigReader("pbmc.fw")
labels = numpy.array([types.index(t["cell_type"]) for t in fw.tracks])
y = fw.read(chroms, starts, 2114, labels=labels)            # pseudobulks, (n, types, 2114)
found = fw.read(chroms, starts, 2114, tracks=fw.select(cell_type="B"), sparse=True)
```

- One group holding every cell, uncompressed (`level=0`) for the fastest
  reads or compressed for half the size; tiles with few cells present are
  matrix chunks either way.
- Never read thousands of cells densely; ask for `labels=` or `sparse=True`.
  A sparse read costs about 40 bytes of memory per entry, so it is for tracks
  that are mostly zero, not for dense tracks, whose dense read is smaller.
- For per-cell slices over the genome (`read_entries(tracks=[...])`), write or
  `reorder` the file in track order; position order scans every chunk for
  that.
- `write_sparse` entries are (chrom, position, track, value); no two may name
  one base of one track, and a call's entries come after the group's earlier
  items on that chromosome. One call takes about 90 bytes per entry, so
  write a chromosome at a time. Fragment files give two cut sites per
  fragment; merge duplicates before writing.
- Thousands of single-track groups cost a group's index each, 17 bytes per
  tile of the genome in memory once read; use one group, or track order.

## Writing

```python
from figwig import FigWigWriter

tracks = [{"name": "atac.+", "assay": "ATAC-seq", "strand": "+"},
	{"name": "atac.-", "assay": "ATAC-seq", "strand": "-"}]
with FigWigWriter("k562.fw", chrom_sizes, tracks, genome="hg38") as writer:
	writer.write(chroms, starts, y_hat)        # (n, 2, width), whole groups at a time
```

- `tracks` are names or dicts with `name`, optional `dtype` (uint8, uint16,
  uint32, int8, int16, int32, float32, float64) and JSON-serializable
  attributes (no NaN). Left out, `dtype` is set at close to the narrowest
  type holding the track's values: an integer type, float32 for a fraction,
  float64 for an integer beyond uint32. Given, every value is checked against
  it. Storage is the same either way: each chunk is as narrow as its own
  values, and an integer track grouped with a float track keeps its integers
  exact (float64 chunks where float32 would round them).
- A float32 track keeps float32's rounding; a value beyond float32's range
  raises unless the track is float64; -0.0 is 0.
- Each `write` covers whole groups (`groups=` is a count of consecutive
  tracks, 8 by default, or a list of lists of names, in any order). Items may
  come in any order within a call but must not overlap; within a group, later
  calls come after earlier ones on a chromosome, and a group cannot return to
  a chromosome it left. A window may run past its chromosome's end only where
  its values are NaN or 0. A call that raises writes nothing.
- NaN is a value in a float track and an error in an integer one. Zeros are
  not stored, so a bigWig's "no data" and 0 are the same thing here; see
  `bw2fw --missing nan` below.
- Chunks are dense, sparse (nonzero bases), runs (the bases where the value
  changes, for binned or piecewise-constant signal) or a matrix of the tracks
  present, whichever is smallest; a chunk that compression would grow is
  stored raw. `level=0` stores every chunk uncompressed, the fastest to read,
  at 4 bytes per base of a dense float track; `tile=` must be a power of two
  from 16 to 65536 with at most 2**26 tiles over the genome, and smaller
  tiles decode less per window. Compressed chunks are inflated by libdeflate
  when the `deflate` package is installed (`pip install figwig[fast]`), about
  twice as fast as zlib.
- A chunk whose nonzero values are all 1, as single-cell cut sites usually
  are, stores no values at all.
- The file is written beside its path and moved there at `close`, so a
  failed write leaves the path as it was, and `reorder` or `bw2fw` may write
  over their input. A writer never closed warns and writes nothing.
- The chunks depend on the values and on which calls wrote which groups, not
  on `n_jobs` or on how values were split between calls; the sum statistic
  is compensated.

## Converting

```bash
figwig bw2fw -o k562.fw -m tracks.tsv -g hg38 -p 8 atac.+.bw atac.-.bw dnase.bw
figwig bw2fw -o k562.fw --append -p 8 h3k27ac.bw            # add tracks to a file
figwig bw2fw -o wgbs.fw --missing nan -p 8 methylation.bw
figwig merge -o all.fw k562.fw hepg2.fw                     # copy files' tracks into one
figwig fw2bw k562.fw -o bigwigs -p 8 [--select assay=ATAC-seq | -t atac.+ atac.-]
figwig bam2fw -s hg38.chrom.sizes -n k562 -m files.tsv -p 8 a.bam b.bam
```

- The metadata table is tab-separated (or `.csv`), with a header; its first
  column names the input file by path or base name, a `name` column names the
  track, a `dtype` column sets its dtype, and every other column is an
  attribute, kept as a string. `fw2bw` writes such a table next to the
  bigWigs, named after the figwig file, so `bw2fw -m` reads it back.
- `bw2fw` takes the chromosomes from the bigWigs, which must agree on
  lengths; `-s sizes` sets them instead, must agree with every bigWig on the
  chromosomes they share (an hg19 bigWig with hg38 sizes is refused), warns
  about chromosomes it leaves out, and a bigWig lacking one is 0 there.
- Values read from a converted file equal `read_bigwig(..., missing=0.0,
  keep_nan=True)` on the bigWigs: an interval whose value is NaN is NaN.
  `--missing nan` makes the bases no interval covers NaN too, for methylation
  or conservation tracks whose zeros are measurements, at the cost of a dense
  float32 track; `fw2bw --missing nan` writes such a file's zeros back.
- `bam2fw` takes `figwig bam2bw`'s options and writes each input file as its
  own pair of tracks `<name>.+` and `<name>.-` (one track with `-u`), in one
  group; `-r` divides by each file's own read depth. Linux and macOS, with
  the `bam2bw` extra.
- Binned tracks (5 kb bins, 250 kb compartments) are stored as runs and read
  back exactly; a bigWig is still faster for reads of a few bases inside wide
  bins and for summaries over megabases, which the format has no zoom levels
  for.

## Errors

| Message | Cause |
|---|---|
| `tracks must make up whole groups` | a `write` named some but not all tracks of a group |
| `written again after the group of ... moved on from it` | a group returned to an earlier chromosome |
| `starts at ..., before the end of what was written on it before` | items out of order across calls |
| `a window may run past the end only where its values are NaN or 0` | real values past the chromosome's length |
| `track ... is uint16, and a value is ...` | a declared dtype the values do not fit; declare a wider one or none |
| `lies beyond float32's range` | a value above 3.4e38 in a track that is not float64 |
| `the dense output would take ... GiB` | a dense read of many tracks; use `sparse=True`, `labels=`, fewer windows or `out=` |
| `a sum lies outside what ... holds` | `labels=` or `bins=` into an integer dtype too narrow for the sums |
| `entries ... both name base ...` | duplicate (track, position) entries in `write_sparse`; merge them first |
| `which an earlier call already wrote`, `was begun by write` | `write` and `write_sparse` on one tile in two calls |
| `has values from ... to ..., which uint8 does not hold` | `dtype=` too narrow for a track on read |
| `is not one of the file's chromosomes` | a window on a chromosome the file was not written with |
| `has been replaced or rewritten since this reader opened it` | the file changed under a reader; open it again |
| `it is version 1 of the format`, `version 2` | a file written by a figwig before the format's release; write it again |
| `keeps no summary of the group of` | `tile_sums` on a file whose sums passed 256 MB, or written by an earlier figwig |
| `came from another file, whose tracks cannot be written to` | `write` to a track that `append` or `add_file` copied in |
| `was written with tile ..., level ... and ... order where this writer has` | `add_file` or `merge` of files with different layouts; `reorder` one first |
| `does not match its checksum` | `verify=True` found a corrupt chunk |
| `cannot be read as a figwig file`, `a chunk of ... cannot be read` | a truncated or corrupt file; every such case raises `ValueError` |
