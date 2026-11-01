# Reading bigWigs

`BigWigReader(path).read(chroms, starts, width, out=None, n_jobs=8, missing=0.0, keep_nan=False, *, oob=nan, engine='auto')`
and `read_bigwig(bigwigs, chroms, starts, width, ...)`, where `bigwigs` is
one path or reader (result `(n, width)`) or a list, even of one (result
`(n, len(bigwigs), width)`, channel `i` from `bigwigs[i]`). Always float32.
`engine='auto'` inflates with ISA-L when the `isal` extra (python-isal 1.8,
built on the ISA-L version the kernel knows, 2.31.1) is installed and with
zlib otherwise; `'zlib'` and `'isal'` choose one. The values are the same.

```python
import numpy
from figwig import BigWigReader
from figwig import read_bigwig

bw = BigWigReader("ENCFF830RWF.bigWig")
y = bw.read(["chr1", "chr1", "chr2"], [1_000_000, 2_500_000, 300_000], width=1000)
print(y.shape, y.dtype, y.sum(axis=1))
# (3, 1000) float32 [116.  43.  10.]

y = read_bigwig(["ENCFF830RWF.bigWig", "ENCFF989SAK.bigWig"], "chr1",
	[1_000_000, 2_500_000], width=1000)
print(y.shape)
# (2, 2, 1000)
```

## Arguments

- `chroms`: one name for every window, or one per window (list, array,
  pandas Series). `starts`: integers; a pandas column that once held a NaN
  is float64 and raises `TypeError`, so `.astype("int64")` it.
- A window starting before 0 raises: clip summit-centred windows with
  `numpy.maximum(summits - width // 2, 0)`. Past the end of a chromosome is
  allowed; those bases are NaN.
- One `width` per call. Regions of several lengths: read at the longest and
  mask, or one call per width. One call per region throws away the speed.

## Values

| Base | Value |
|---|---|
| covered by an interval | the interval's float32 value, copied, never computed |
| covered by no interval | `missing`, 0.0 unless given |
| covered by an interval whose value is NaN | `missing`, or NaN with `keep_nan=True` |
| past the end of its chromosome | `oob`, NaN unless given, whatever `missing` is |
| on a chromosome the file does not have | `missing`, with one `UserWarning` per read |

Defaults match pybigtools' `values()`; `missing=numpy.nan` matches pyBigWig's.
Pass NaN when a measured 0 must be told from no data. `oob=0.0` fills the
bases a window runs off a chromosome's end with 0 instead of NaN, and leaves NaN
for an interval whose value is NaN (`keep_nan=True`). No zoom levels are
used: binned values are a reshape, `y.reshape(len(y), -1, 32).mean(axis=-1)`
(`numpy.nanmean` if windows pass chromosome ends).

## Chromosome names

Many writers leave out chromosomes without data (figwig's keeps every one
of `chrom_sizes`), so a window on one is `missing` throughout, with a
warning naming the chromosomes and some of the file's own. The same
warning is what `1` against `chr1` looks like: check
`set(chroms) - set(reader.chrom_sizes)` before reading, and map names in the
call (`numpy.char.add("chr", chroms)`) rather than renaming your tables.
`reader.chrom_sizes` is in the file's order, which is name order (`chr1`,
`chr10`, `chr11`, …) in ENCODE and UCSC files. Under pytest's
`filterwarnings = error` the warning is an exception.

## Reuse the reader

A path given to `read_bigwig` opens the file and reads its index on every
call; a `BigWigReader` reads the index on its first `read` and keeps it, so
in a loop open the readers once and pass them. `out=` must be a writeable,
C-contiguous float32 array of the result's shape; a read that raises on an
unreadable block may already have written other rows into it. One reader
can be read from several threads and holds no open file between reads.

## PyTorch

Each item is a whole batch read in one call, from a `BatchSampler` over a
`RandomSampler` with `batch_size=None` on the loader. Read once in the main
process so each worker's pickled reader carries the index. The workers are
the parallelism: `n_jobs=1` inside them.

```python
import numpy
import torch
from figwig import BigWigReader

class Windows(torch.utils.data.Dataset):
	def __init__(self, path, chroms, starts, width):
		self.bw, self.chroms, self.starts = BigWigReader(path), chroms, starts
		self.width = width
		self.bw.read(chroms[:1], starts[:1], width)   # reads the index once

	def __len__(self):
		return len(self.starts)

	def __getitem__(self, idx):                        # a batch of indices
		return torch.from_numpy(self.bw.read(self.chroms[idx], self.starts[idx],
			self.width, n_jobs=1))

if __name__ == '__main__':
	starts = numpy.arange(1_000_000, 2_000_000, 1000)
	chroms = numpy.full(len(starts), "chr1")
	data = Windows("ENCFF830RWF.bigWig", chroms, starts, 1000)
	batches = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(data),
		batch_size=64, drop_last=False)
	loader = torch.utils.data.DataLoader(data, sampler=batches, batch_size=None,
		num_workers=2, persistent_workers=True, multiprocessing_context="spawn")
	y = torch.cat(list(loader))
	print(y.shape, float(y.sum()))
	# torch.Size([1000, 1000]) 28857.0
```

For several tracks keep a list of readers and call `read_bigwig(readers, ...)`.
For random shifts read windows wider by twice the largest shift and slice.
When every window fits in memory, read them all in one call and index the
array each step: one read on 8 threads beats an epoch streamed by workers.
Many tracks read every epoch want a figwig file (`references/figwig-reading.md`).

## Falling back to another reader

figwig refuses what it cannot read exactly (bigBed, big-endian, corrupt or
truncated indexes, overlapping intervals, unknown section types) with
`ValueError`. Its tests check it against pybigtools bit for bit, so:

```python
import numpy
import pybigtools
from figwig import BigWigReader

def read(path, chroms, starts, width):
	try:
		return BigWigReader(path).read(chroms, starts, width)
	except ValueError:
		bw = pybigtools.open(path)
		return numpy.array([bw.values(chrom, start, start + width) for chrom,
			start in zip(chroms, starts)], dtype=numpy.float32)
```

pybigtools sums overlapping intervals and panics on a bigBed with an
exception that is not an `Exception`, so check the magic first where a path
may be either. A missing file or a URL raises `FileNotFoundError`, which
this does not catch.
