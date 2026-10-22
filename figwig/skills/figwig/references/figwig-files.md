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
	sparse=False, labels=None, bins=None, nan='propagate')
reader.sums(chroms, starts, ends, tracks=None, dtype=None, n_jobs=8, nan='propagate',
	paired=False)
	# -> (n, tracks): totals over intervals of any lengths; nan='skip' leaves NaN
	# bases out, nan='count' gives the bases that are not NaN (covered-base means);
	# paired=True sums interval i over track tracks[i] alone, (n,)
reader.read_entries(tracks=None, chroms=None, n_jobs=8)   # -> SparseEntries
reader.tile_sums(tracks=None, chroms=None)  # -> float32 (tracks, tiles): sums per tile, no decoding
reader.select(**attributes)          # -> int64 indices, for `tracks`
reader.tracks                        # list of dicts: name, dtype, group, attributes
reader.stats                         # structured array: sum, nonzero, min, max per track
reader.chrom_sizes, reader.genome, reader.attributes, reader.tile, reader.order
read_figwig(path_or_reader, chroms, starts, width, tracks=None, out=None, dtype=None,
	n_jobs=8, sparse=False, labels=None, bins=None)

FigWigWriter(path, chrom_sizes, tracks, genome=None, attributes=None, tile=None,
	groups=8, level=None, engine='auto', n_jobs=8, order=None, append=False)
	# tile None is 16384, order None is 'positions', level None is 6 (0 in track order)
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
| dense `(n, labels, width)` sums, float64 for integer tracks and float32 otherwise | `labels=` one int per track read, -1 skips |
| dense `(n, tracks, width // bins)` sums per bin, the same dtypes | `bins=` a divisor of `width`; `bins=width` is each window's total |

| `order=` | layout | fast | slow |
|---|---|---|---|
| `'positions'` / `'csc'` (default) | tiles across tracks, groups' tiles in one chunk | windows across many tracks, pseudobulks | one track over the genome |
| `'tracks'` / `'csr'` | each track's nonzero bases in blocks per chromosome, raw by default | `read_entries` of a few tracks | windows over thousands of tracks (one block per track per window) |

`figwig reorder in.fw out.fw --order tracks|positions` rewrites a file in the
other order, keeping its tile and groups, and its level only within one
order (track order is level 0 unless `-l` is given, 3 to 4x larger);
`out.fw` may be `in.fw`.
`figwig merge -o all.fw a.fw b.fw` copies several files' tracks into one
(same chromosomes, tile, level and order; `reorder` first otherwise).
`FigWigWriter(path, None, new_tracks, append=True)` adds tracks to a file in
place (the new chunks are written after its end; its old index stays behind
as dead bytes, which `merge` into a new path drops);
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
the one it opened (inode and header), and raises if a conversion replaced
it or an append to it has closed; it keeps reading while the append runs.
`figwig info k562.fw` prints the file's genome, order, tile, chromosomes,
size and a table of its tracks; `-b` adds the bytes its chunks, tables
and nothing use.

## Reading for training

```python
import numpy
from figwig import FigWigReader

fw = FigWigReader("k562.fw")
tracks = fw.select(assay="ATAC-seq", biosample="K562")   # [0 1]: both strands; without biosample= the unstranded track too
chroms = numpy.array(["chr1", "chr1", "chr2"])            # one per window, from a peak table
starts = numpy.array([1_000_000, 2_500_000, 300_000])
y = fw.read(chroms, starts, width=2114, tracks=tracks, dtype="float32")
# (3, 2, 2114); the tracks' own dtype (uint8/uint16) without dtype=
counts = fw.read(chroms, starts, width=2114, tracks=tracks, bins=2114)[:, :, 0]
binned = fw.read(chroms, starts, width=2048, bins=128)    # (3, tracks, 16)
totals = fw.sums(chroms, starts, starts + [900, 2114, 410], tracks=tracks)  # peaks of any lengths, (3, 2)
```

- `select` matches equality on the keys of the track records: `name`,
  `dtype`, `group` and whatever attributes the file was written with (`assay`,
  `strand`, `biosample`, ...). Look at `fw.tracks` to see which exist;
  attributes from a metadata table are strings, so `select(units=1)` matches
  nothing where `select(units="1")` would, and a key no track has raises.
- `out=` may be a numpy view of a pinned torch tensor of the output's shape
  and dtype; cast to float on the GPU, where a uint16 batch is half the bytes
  of float32. Integer outputs wrap in numpy and torch arithmetic the way any
  uint8 does, and torch's uint16 lacks `add`, `mean` and `mse_loss`, so a
  training path passes `dtype="float32"` (10-20% slower than the native
  type) or casts at once; a torch dtype is accepted as `dtype=`. Which
  native type comes back depends on the tracks asked for (uint8 for one,
  uint16 for its neighbour), so a model should not rely on it.
- `bins=` is a bin width in bases, not a count as pyBigWig's `nBins` is,
  and must divide `width`; 2,114 has no divisor near 100, so bin a 2,048 or
  1,000 bp window. A mean from a figwig file is a sum over every base;
  pyBigWig `stats`, pybigtools and deepTools average over covered bases and
  give NaN for an empty region, so an analysis of count tracks ported from
  them changes numbers unless the file was written with `--missing nan`,
  where a bin or interval holding a NaN base sums to NaN.
- `sums(chroms, starts, ends, tracks)` is for intervals that differ in
  length, peaks, genes or cCREs: 172,732 peaks x 6 tracks took 0.25 s on 8
  threads, where a `read(bins=width)` per distinct width took 2.7 s and
  pyBigWig `stats` 46 s; 58,705 peaks x 3,000 cells took 1.0 s, equal to a
  sparse read's sums. A mean over covered bases, as pyBigWig and deepTools
  give it, is `sums(..., nan="skip") / sums(..., nan="count")` on a file
  written with `--missing nan`; a scaled-region profile (deepTools
  scale-regions) is `sums` over sub-intervals with edges `start + (j *
  length) // n_bins`, which matches `computeMatrix` to 2e-6.
- A `FigWigReader` pickles without its index, which each worker reads
  again, and can be read from several threads, so it goes in a Dataset the
  way `references/reading.md` shows for `BigWigReader`. DataLoader workers
  help once reading is the bottleneck: four fork or spawn workers halved an
  epoch of 8,192 windows from 0.15 s to 0.08 s, two spawn workers gained
  nothing, and under `spawn` each worker pays 2-3 s on its first epoch to
  load the compiled kernels.
- Tracks of one *group* are stored together and decoded together, so reading
  one track of a group of 8 decodes all 8. Files meant to be read a track at a
  time should be written with small groups, and groups should hold tracks of
  one kind: a count track grouped with a float track is stored as float
  values and decoded with it. For a *reads* dimension (Fiber-seq reads or
  cells as tracks, read all at once over a window), the rule is the other
  way: one group per channel over all the reads is 20-50x faster than a
  group per read, and position order beats track order. Reads as their own
  "chromosomes" (each read a chromosome of its length, with 2 tracks) also
  works for per-read access, with genome coordinates in the attributes. The
  defaults (tile 16,384, groups of 8) suit
  count tracks, for which one group of many tracks reads 8 of 48 tracks
  2-4x faster than 48 groups of one, since a sparse chunk's other tracks
  are stepped over cheaply; dense float tracks read in short windows
  (conservation, fold change, p-value) want `tile=1024, groups=1` (`bw2fw
  -t 1024 --groups 1`; `-l 1` gains nothing): 7x faster random 1 kb windows
  for a file 35-50% larger, and a dense track in a group of its own reads
  12x faster than beside two others.
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
- The file is written beside its path and moved there at `close`, with the
  mode the umask gives a new file, so a failed write leaves the path as it
  was, and `reorder` or `bw2fw` may write over their input. A writer never
  closed warns and writes nothing. `write` with `ends=` (intervals) encodes
  on the calling thread, where whole-tile windows go to the pool: write
  windows for speed. `append=True` with `attributes=` adds to the file's
  attributes.
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
figwig info k562.fw [-b]                                     # what the file holds
```

- `bw2fw`, `fw2bw`, `reorder` and `merge` refuse to write over an output
  that exists unless `-f` is given (`bam2fw` takes `--force`, since its
  `-f` means fragments; `--append` adds to a file instead); `-v` prints a
  line per group and a summary on stderr. `fw2bw` writes each bigWig beside
  its name and moves it into place when complete, as `bw2fw` writes its
  file, so a killed conversion leaves no half-written output; a killed
  `--append` leaves bytes past the file's end, which `info -b` reports and
  the next append writes over. `fw2bw -z N` writes N zoom levels (0 by
  default), each adding to the bigWig. There is no bedGraph export: `fw2bw`
  then `bigWigToBedGraph`, or `read_entries` and write the runs. Ctrl-C
  exits with 130, a missing input file with a message.

- The metadata table is tab-separated (or `.csv`), with a header; its first
  column names the input file by path or base name, a `name` column names the
  track, a `dtype` column sets its dtype, and every other column is an
  attribute, kept as a string (an empty cell is no attribute). `fw2bw`
  writes such a table next to the bigWigs, named after the figwig file, so
  `bw2fw -m` reads it back.
- `bw2fw` takes the chromosomes from the bigWigs, which must agree on
  lengths; `-s sizes` sets them instead, must agree with every bigWig on the
  chromosomes they share (an hg19 bigWig with hg38 sizes is refused), warns
  about chromosomes it leaves out, and a bigWig lacking one is 0 there.
- Values read from a converted file equal `read_bigwig(..., missing=0.0,
  keep_nan=True)` on the bigWigs: an interval whose value is NaN is NaN.
  `--missing nan` makes the bases no interval covers NaN too, for methylation
  or conservation tracks whose zeros are measurements, and for fold-change
  or p-value tracks whose explicit 0.0 intervals should survive a round
  trip (`bw2fw` then `fw2bw` with `--missing nan` on both is lossless; the
  defaults drop the 0.0 intervals); the track is then float32, which costs
  a compressed file under 1% and doubles an uncompressed one; `fw2bw
  --missing nan` writes such a file's zeros back.
  `bw2fw --append` of a bigWig with chromosomes the file lacks drops them
  with a warning; `bw2fw -s` fixes the chromosome set of files that will be
  merged.
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
| `is not one of the file's chromosomes, 'chr1', ...` | a window on a chromosome the file was not written with; the message names the file's |
| `has been replaced or rewritten since this reader opened it`, `has been truncated` | the file changed under a reader (a conversion, or an append that closed); open it again |
| `no track has an attribute` | `select` with a key no track record has; the message names the keys |
| `pass dtype= to read as another type` | `out=` of another dtype than the tracks' |
| `ends must hold one end per start`, `an interval's end is before its start` | `sums` with mismatched or reversed intervals |
| `exists; --append adds tracks to it, and -f writes over it`, `exists; -f writes over it`, `exists; --force writes over it` | `bw2fw`, `fw2bw`, `reorder`, `merge` or `bam2fw` onto an existing output |
| `nan must be 'propagate', 'skip' or 'count'`, `nan= applies to sums: give bins too` | `nan=` misspelt, or on a dense read |
| `the table has one column; a table is tab-separated` | a space-separated metadata table |
| `it is version 1 of the format`, `version 2`, `version 3` | a file written by a figwig before the format's release; write it again |
| `its tables do not match their checksum`, `The index of ... does not match its checksum` | a damaged table or index record; the file is corrupt |
| `keeps no summary of the group of` | `tile_sums` on a file whose sums passed 256 MB, or written by an earlier figwig |
| `came from another file, whose tracks cannot be written to` | `write` to a track that `append` or `add_file` copied in |
| `was written with order|tile|level ... where this writer has`, `was written with other chromosomes or lengths: it has ...` | `add_file` or `merge` of files with different layouts or chromosome sets; `reorder` one first, or `bw2fw -s` the same sizes |
| `does not match its checksum` | `verify=True` found a corrupt chunk |
| `its header does not match its checksum` | a damaged header (version 5 keeps a CRC-32 of it); the file is corrupt |
| `needs a feature this figwig does not know (bits ...)`, `its flags hold a bit this figwig does not know` | a newer figwig wrote it; upgrade figwig |
| `is being appended to by another writer` | two `append=True` writers on one file at once; wait for the first to close |
| `an integer lies beyond 2**53, which a figwig file cannot hold exactly` | a 64-bit integer the file's float64 cannot keep; scale or clip first |
| `is not a regular file` | the path is a directory, FIFO or device |
| `paired=True needs one track per interval` | `sums(paired=True)` without `tracks`, or with another count than the intervals |
| `cannot be read as a figwig file`, `a chunk of ... cannot be read` | a truncated or corrupt file; every such case raises `ValueError` |
