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
reader.read(chroms, starts, width, tracks=None, out=None, dtype=None, n_jobs=8)
reader.select(**attributes)          # -> int64 indices, for `tracks`
reader.tracks                        # list of dicts: name, dtype, group, attributes
reader.stats                         # structured array: sum, nonzero, min, max per track
reader.chrom_sizes, reader.genome, reader.attributes, reader.tile
read_figwig(path_or_reader, chroms, starts, width, tracks=None, out=None, dtype=None, n_jobs=8)

FigWigWriter(path, chrom_sizes, tracks, genome=None, attributes=None, tile=16384,
	groups=8, level=6, engine='auto', n_jobs=8)
writer.write(chroms, starts, values, ends=None, tracks=None)
writer.close()                       # writes the tables and index; `with` calls it
write_figwig(path, chrom_sizes, tracks, chroms, starts, values, ends=None, genome=None,
	attributes=None, tile=16384, groups=8, level=6, engine='auto', n_jobs=8)
```

| `tracks` given to `read` | output shape | dtype |
|---|---|---|
| `None` | `(n, all tracks, width)` | the tracks' own if shared, else float32 if any is float32, else the smallest integer holding all |
| a list of names or indices | `(n, len(tracks), width)`, channel `i` is `tracks[i]` | as above |
| one name or index | `(n, width)` | that track's |

`dtype=` asks for another type; an integer type that cannot hold a track's
values raises `ValueError`. A base never written is 0; a base past the end
of its chromosome is NaN in a float output and 0 in an integer one. An
unknown chromosome raises: a figwig file holds every chromosome it was
written with, so there is no warning as for bigWigs.

## Reading for training

```python
import numpy
from figwig import FigWigReader

fw = FigWigReader("k562.fw")
tracks = fw.select(assay="ATAC-seq")           # e.g. [0 1]
y = fw.read(chroms, starts, width=2114, tracks=tracks, n_jobs=1)
# (n, 2, 2114) uint16 for count tracks
```

- `select` matches equality on the keys of the track records: `name`,
  `dtype`, `group` and whatever attributes the file was written with (`assay`,
  `strand`, `biosample`, ...). Look at `fw.tracks` to see which exist.
- `out=` may be a numpy view of a pinned torch tensor of the output's shape
  and dtype; cast to float on the GPU, where a uint16 batch is half the bytes
  of float32.
- A `FigWigReader` pickles and can be read from several threads, so it goes
  in a Dataset the way `references/reading.md` shows for `BigWigReader`; in
  DataLoader workers use `n_jobs=1`.
- Tracks of one *group* are stored together and decoded together, so reading
  one track of a group of 8 decodes all 8. Files meant to be read a track at a
  time should be written with small groups.

## Writing

```python
from figwig import FigWigWriter

tracks = [{"name": "atac.+", "assay": "ATAC-seq", "strand": "+"},
	{"name": "atac.-", "assay": "ATAC-seq", "strand": "-"}]
with FigWigWriter("k562.fw", chrom_sizes, tracks, genome="hg38") as writer:
	writer.write(chroms, starts, y_hat)        # (n, 2, width), whole groups at a time
```

- `tracks` are names or dicts with `name`, optional `dtype` (uint8, uint16,
  uint32, int8, int16, int32, float32) and JSON-serializable attributes. Left
  out, `dtype` is set at close to the narrowest type holding the track's
  values; given, every value is checked against it. Storage is the same
  either way: each chunk is as narrow as its own values.
- Each `write` covers whole groups (`groups=` is a count of consecutive
  tracks, 8 by default, or a list of lists of names). Items may come in any
  order within a call but must not overlap; within a group, later calls come
  after earlier ones on a chromosome, and a group cannot return to a
  chromosome it left. A window may run past its chromosome's end only where
  its values are NaN or 0.
- NaN is a value in a float32 track and an error in an integer one. Zeros are
  not stored, so a bigWig's "no data" and 0 are the same thing here.
- `level=0` stores chunks uncompressed, the fastest to read; `tile=` must be
  a power of two from 16 to 65536, and smaller tiles decode less per window.
- The file depends on the values and on which calls wrote which groups, not
  on `n_jobs` or on how values were split between calls.

## Converting

```bash
figwig bw2fw -o k562.fw -m tracks.tsv -g hg38 -p 8 atac.+.bw atac.-.bw dnase.bw
figwig fw2bw k562.fw -o bigwigs -p 8 [--select assay=ATAC-seq | -t atac.+ atac.-]
figwig bam2fw -s hg38.chrom.sizes -n k562 -m files.tsv -p 8 a.bam b.bam
```

- The metadata table is tab-separated (or `.csv`), with a header; its first
  column names the input file by path or base name, a `name` column names the
  track, a `dtype` column sets its dtype, and every other column is an
  attribute, kept as a string. `fw2bw` writes such a table next to the
  bigWigs, named after the figwig file, so `bw2fw -m` reads it back.
- `bw2fw` takes the chromosomes from the bigWigs, which must agree on
  lengths; `-s sizes` sets them instead, and a bigWig lacking one is 0 there.
- `bam2fw` takes `figwig bam2bw`'s options and writes each input file as its
  own pair of tracks `<name>.+` and `<name>.-` (one track with `-u`), in one
  group; `-r` divides by each file's own read depth. Linux and macOS, with
  the `bam2bw` extra.
- Values read from a converted file equal `read_bigwig(..., missing=0.0)` on
  the bigWigs, and a track written back to a bigWig reads the same way, with
  NaN as 0.

## Errors

| Message | Cause |
|---|---|
| `tracks must make up whole groups` | a `write` named some but not all tracks of a group |
| `written again after the group of ... moved on from it` | a group returned to an earlier chromosome |
| `starts at ..., before the end of what was written on it before` | items out of order across calls |
| `a window may run past the end only where its values are NaN or 0` | real values past the chromosome's length |
| `track ... is uint16, and a value is ...` | a declared dtype the values do not fit; declare a wider one or none |
| `has values from ... to ..., which uint8 does not hold` | `dtype=` too narrow for a track on read |
| `is not one of the file's chromosomes` | a window on a chromosome the file was not written with |
| `cannot be read as a figwig file`, `a chunk of ... cannot be read` | a truncated or corrupt file; every such case raises `ValueError` |
