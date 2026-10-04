# Writing bigWigs

## Signatures

```python
write_bigwig(paths, chrom_sizes, chroms, starts, values, ends=None, missing=0.0,
	zooms=10, level=6, engine='auto', n_jobs=8)
BigWigWriter(path, chrom_sizes, zooms=10, level=6, engine='auto', n_jobs=8)
writer.write(chroms, starts, values, ends=None, missing=0.0)
writer.close()   # writes the index, zoom levels and header; `with` calls it
```

| Items | `ends` | `values`, one path | `values`, a list of k paths |
|---|---|---|---|
| windows | `None` | `(n, width)` | `(n, k, width)`; channel `i` goes to `paths[i]` |
| intervals | `(n,)` | `(n,)` | `(n, k)` |
| single bases, such as per-base counts | `None` | `(n, 1)`: windows of width 1 | `(n, k, 1)` |

A list of one path still needs the channel axis: `(n, 1, width)`.

## Model predictions

`write_bigwig` takes the layout `read_bigwig` gives and models predict:

```python
import numpy
from figwig import read_bigwig
from figwig import write_bigwig

chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
chroms = numpy.array(["chr2", "chr1", "chr1"])
starts = numpy.array([300_000, 2_500_000, 1_000_000])
y_hat = numpy.random.default_rng(0).random((3, 2, 1000), dtype=numpy.float32)

write_bigwig(["plus.bw", "minus.bw"], chrom_sizes, chroms, starts, y_hat)
y = read_bigwig(["plus.bw", "minus.bw"], chroms, starts, width=1000)
print(numpy.array_equal(y, y_hat))
# True
```

To copy a file's chromosomes, pass `BigWigReader(path).chrom_sizes`.

## Rules

Each of these raises `ValueError`, and a call that raises writes nothing.

- Every chromosome written must be in `chrom_sizes`.
- Within one call, items may come in any order but must not overlap. A
  window's full width counts, even where its values are `missing` or NaN.
- Each call's items come after the last call's: on a chromosome later in
  `chrom_sizes`, or on the same one at or after the last call's last end.
  `BigWigReader.chrom_sizes` is in the order the file stores, which is name
  order (chr1, chr10, chr11, …, chr2) in ENCODE and UCSC files, so writing
  chr1, chr2, … one call each raises at chr10 with "chromosome 'chr10' is
  written after 'chr9', but comes before it in chrom_sizes".
  Loop over `writer.chrom_sizes`, or build `chrom_sizes` in the order you
  write.
- `starts` must be integers and at least 0. An interval must end after its
  start and at most at its chromosome's length.
- A window may run past the end of its chromosome only where its values are
  NaN. Predictions for windows near a chromosome's end raise with "runs past
  the end of its chromosome, at L, where its values must be NaN" unless
  `y[..., L - start:] = numpy.nan` first.
- Values must be finite or NaN, and within float32's range. They are written
  as float32.

## What is left out

A base or interval whose value is `missing` (0.0 unless given) or NaN is not
written. With `missing=0.0`, -0.0 is left out too. Reading with the same
`missing` gives the values back. To store zeros as data, write and read with
`missing=numpy.nan`.

## A batch at a time

`BigWigWriter` writes values as they come, so a track larger than memory,
such as genome-wide predictions, can be written one batch of windows at a
time:

```python
import numpy
from figwig import BigWigReader
from figwig import BigWigWriter

chrom_sizes = {"chr1": 100_000, "chr2": 50_000}
rng = numpy.random.default_rng(0)
with BigWigWriter("track.bw", chrom_sizes, zooms=0) as writer:
	for chrom, length in writer.chrom_sizes.items():
		for start in range(0, length, 10_000):
			batch = numpy.arange(start, min(start + 10_000, length), 1000)
			y = rng.random((len(batch), 1000), dtype=numpy.float32)
			writer.write(chrom, batch, y)

print(BigWigReader("track.bw").read("chr2", [49_000], width=1000).shape)
# (1, 1000)
```

- The file is not a bigWig until `close`. Reading it before raises
  "track.bw is not a bigWig file."
- An exception inside the `with` block abandons the file without a header,
  which raises the same way on a read. Write it again.
- `write_bigwig` with several paths writes them one after another. If one
  fails, it is left without a header and the later ones are not written.

## Options

| Option | Default | When to change it |
|---|---|---|
| `zooms` | 10 | 0 when no genome browser will draw the file. Building zoom levels takes most of a write's time. |
| `engine` | `'auto'`: libdeflate when the `deflate` package loads, else zlib | `'zlib'` for bytes identical to pyBigWig's (with `level=6`, `zooms=0`, `missing=numpy.nan`). `'isal'` is fastest at levels 1 to 3, but its files at levels 1 and 2 differ from run to run. |
| `level` | 6 | 1 to write faster. `figwig bam2bw` writes libdeflate level 1, into files 3% larger than pyBigWig's level 6. |
| `n_jobs` | 8 | threads that compress blocks, besides the calling thread. See `references/threads-and-memory.md`. |

The bytes written do not depend on `n_jobs` or on how values are split into
calls, except with `'isal'`. figwig picks fixedStep, varStep or bedGraph
sections for each window by size; there is nothing to set.

## Per-base counts

Single bases are windows of width 1, written as varStep sections:

```python
import numpy
from figwig import BigWigReader
from figwig import write_bigwig

positions = numpy.array([1000, 1003, 1004, 2050])
counts = numpy.array([3, 1, 1, 7], dtype=numpy.float32)
write_bigwig("counts.bw", {"chr1": 10_000}, "chr1", positions, counts[:, None])
print(BigWigReader("counts.bw").read("chr1", [1000], width=6))
# [[3. 0. 0. 1. 1. 0.]]
```

To make count tracks from reads, use `figwig bam2bw`
(`references/bam2bw.md`).
