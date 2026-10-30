# Reading figwig files

A figwig file (`.fw`) holds many tracks over one genome, each with a JSON
record of attributes, read like a bigWig: many windows of one width in one
call, from any subset of the tracks, into `(n, tracks, width)` in the
tracks' own dtype (uint8 or uint16 for most count tracks). One
`FigWigReader(path)`; `read_figwig(path_or_reader, ...)` takes the same
arguments as `reader.read`. The byte layout is specified on the docs'
`format` page, with a Kaitai Struct definition, for reading from another
language.

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

## What a read returns

| Call | Returns |
|---|---|
| `read(chroms, starts, width)` | dense `(n, tracks, width)`; raises past 8 GiB unless `out=` is given |
| `tracks=` a list of names or indices | channel `i` is `tracks[i]`; one name gives `(n, width)` |
| `sparse=True` | `SparseWindows(rows, channels, offsets, values, shape)`, sorted by row, channel, offset; `scipy.sparse.coo_array((values, (rows * width + offsets, channels)), shape=(n * width, channels))` takes it as is |
| `labels=` one int per track read, -1 skips | `(n, labels, width)` sums: pseudobulks in the read |
| `bins=` a divisor of `width` | `(n, tracks, width // bins)` sums; `bins=width` is each window's total |
| `sums(chroms, starts, ends)` | `(n, tracks)` totals over intervals of any lengths |
| `tile_sums(tracks, chroms, level=0)` | `(tracks, tiles)` float32 sums per tile, no decoding; `level=k` the zoom level `zooms=` wrote, `(tracks, bins of 4**k tiles)` float64 |
| `read_entries(tracks, chroms)` | every nonzero base of tracks over chromosomes, `SparseEntries` |
| `channels="channel", items="read"` on `read`, `bins=` and `sums` | `(n, items, channels, width)`, the tracks arranged by two attributes, each item needing every channel once |

- Mixed tracks come back in the narrowest type holding every one exactly
  (float32 for uint8 + float32, float64 for uint32 + float32). `dtype=`
  asks for another and raises for an integer type that cannot hold a
  track; a torch dtype is accepted. Which native type comes back depends on
  the tracks asked for, so a training path passes `dtype="float32"` or
  casts at once: numpy and torch uint8/uint16 arithmetic wraps, and torch's
  uint16 lacks `mean` and `mse_loss`.
- Sums come back float64 for integer tracks (counts pass 2**24) and
  float32 for float32 tracks; an integer `dtype=` is checked for overflow.
- A base never written is 0. Past a chromosome's end is NaN in a float
  output and 0 in an integer one, except in `bins=` sums, which count those
  bases as 0 as `sums` does, so a chromosome's last, partial bin is its
  sum; `labels=` alone gives float64 per base, NaN there. An unknown
  chromosome raises; the file holds every chromosome it was written with,
  so there is no warning.
- `select(**attributes)` matches equality on `name`, `dtype`, `group` and
  the attributes the file was written with; look at `fw.tracks` for them.
  Attributes from a metadata table are strings: `select(units=1)` matches
  nothing where `select(units="1")` would. A key no track has raises. Its
  result is the index array to pass as `tracks=`.
- `fw.stats` is a structured array of `sum`, `nonzero`, `min`, `max` per
  track; `fw.zooms` the zoom levels' tiles per bin, `()` for none;
  `figwig info file.fw` prints all of this.
- `bins` is a width in bases, where pyBigWig's `nBins` is a count, and must
  divide `width`: 2,114 has no divisor near 100, so bin a 2,048 or 1,000 bp
  window.

## Means, and numbers that differ from other tools

A figwig sum counts every base, so a mean is `sum / length`. pyBigWig
`stats`, pybigtools and deepTools average over the bases an interval covers
and give NaN for an empty region; a figwig file cannot tell "no data" from
0 unless it was written with `bw2fw --missing nan`, where uncovered bases
are NaN. Then `nan=` on `sums` and binned reads says what a NaN base does:
`'propagate'` (default) makes the sum NaN, `'skip'` leaves it out,
`'count'` gives the number of non-NaN bases, and the covered-base mean is
`sums(..., nan="skip") / sums(..., nan="count")`. A deepTools
`computeMatrix` scale-regions profile is `sums` over sub-intervals with
edges `start + (j * length) // n_bins`. An analysis of count tracks ported
from those tools gives other numbers for regions with uncovered bases
unless the file was written so.

## Sparse track sets: cells and reads

Never read thousands of cells densely (64 windows × 2,114 bp × 10,000
cells is 1.4 GB of zeros): ask for `labels=` or `sparse=True`. A sparse
read costs about 40 bytes of memory per entry, so it is for tracks that are
mostly zero. `sums(..., paired=True)` takes one track per interval and sums
interval `i` over `tracks[i]` alone, for reads whose spans are their own
tracks; a sum of every read over every interval is quadratic. For per-cell
slices over the genome (`read_entries`), the file wants track order
(`references/figwig-writing.md`).

## Onto a GPU

`device="cuda"` (or `"cuda:1"`, or an index) on `read`, `sums` and
`read_figwig` returns a `cupy.ndarray` in that GPU's memory, complete when
the call returns; needs the `cuda` extra. Frameworks take it without a copy:

```python
y = fw.read(chroms, starts, 2048, dtype="float32", device="cuda")
t = torch.from_dlpack(y)                               # PyTorch
a = jax.dlpack.from_dlpack(y)                          # JAX
x = tf.experimental.dlpack.from_dlpack(y.__dlpack__())  # TensorFlow
```

Tracks under 0.2% nonzero (cells) are scattered on the device from their
entries, which beats copying the dense array several times over; denser
tracks are copied from a pinned host buffer kept on the reader. `out=` is
then a cupy array. Check once that `cupy.asnumpy(y)` equals the CPU read.
Without `device`, `out=` may be `buffer.numpy()` of a pinned torch tensor,
then `.to("cuda", non_blocking=True)`: the dense output is the large object
(3.2 GB for 8,192 × 48 × 2,048 float32), a pinned buffer moves at the bus's
rate, and a uint16 batch is half the bytes of float32. Do not reuse `out=`
while a tensor made from it is in use.

## DataLoader

A `FigWigReader` pickles without its index, which each worker reads again
(fast; it is small), and reads from several threads, so it goes in a
Dataset as `references/reading.md` shows for `BigWigReader`, with
`n_jobs=1` in workers. Workers help once reading is the bottleneck; under
`spawn` each pays a few seconds on its first batch to load the compiled
kernels. The reader checks on every read that the file at its path is the
one it opened (inode and header) and raises if a conversion replaced it or
an append to it has closed; it keeps reading while the append runs.
