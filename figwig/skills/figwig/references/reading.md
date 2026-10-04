# Reading bigWigs

## Signatures

```python
BigWigReader(path)        # reads the header and chromosome tree now
reader.chrom_sizes        # {name: length} in the file's order: chr1, chr10, chr11, ...
reader.read(chroms, starts, width, out=None, n_jobs=8, missing=0.0)
read_bigwig(bigwigs, chroms, starts, width, out=None, n_jobs=8, missing=0.0)
```

| `bigwigs` given as | Result, float32 |
|---|---|
| one path or one `BigWigReader` | `(n, width)` |
| a list or tuple, even of one file | `(n, len(bigwigs), width)`; channel `i` is `bigwigs[i]` |

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

- `chroms` is one name for every window, or one per window as a list, numpy
  array or pandas Series.
- `starts` must be integers. Floats raise `TypeError: starts must be
  integers, not float64.` A pandas column that once held a NaN is float64, so
  cast it with `.astype("int64")`.
- A window that starts before 0 raises `ValueError`. Clip windows centred on
  summits: `numpy.maximum(summits - width // 2, 0)`. A window may run past
  the end of its chromosome; those bases are NaN.
- `width` is one width for every window. For regions of several lengths,
  read the longest width and mask, or make one call per width. One call per
  region gives up the speed.
- Windows may overlap, repeat and come in any order.

## Values

| Base | Value |
|---|---|
| covered by an interval | the interval's value, copied from the file's float32 |
| covered by no interval, or by an interval whose value is NaN | `missing`, 0.0 unless given |
| past the end of its chromosome | NaN, whatever `missing` is |
| on a chromosome the file does not have | `missing`, with one `UserWarning` per read |

`missing=numpy.nan` gives pyBigWig's `values()`, and the default gives
pybigtools' `values()`. Pass NaN when a measured 0 must be told apart from no
data.

figwig does not use zoom levels. For binned values, average bins of bases:
`y.reshape(len(y), -1, 32).mean(axis=-1)`, or `numpy.nanmean` when windows
run past chromosome ends.

## Reuse the reader

`read_bigwig` given paths opens every file and reads its data index on every
call. A `BigWigReader` reads its index on its first `read` and keeps it. In a
loop, open the readers once and pass them:

```python
import numpy
from figwig import BigWigReader
from figwig import read_bigwig

readers = [BigWigReader("ENCFF830RWF.bigWig"), BigWigReader("ENCFF989SAK.bigWig")]
starts = numpy.arange(1_000_000, 2_000_000, 1000)
out = numpy.empty((100, 2, 1000), dtype=numpy.float32)
totals = numpy.zeros(2)
for i in range(0, len(starts), 100):
	read_bigwig(readers, "chr1", starts[i:i + 100], 1000, out=out)
	totals += out.sum(axis=(0, 2))
print(totals)
# [ 28857. 195167.]
```

`out` must be a writeable, C-contiguous float32 array of the result's shape.
A read that raises on an unreadable data block may already have written other
windows into it. One reader can be read from several threads at once, and
holds no open file between reads.

## Chromosome names

```
UserWarning: 2 windows are on chromosomes not in ENCFF830RWF.bigWig, and are
0.0 throughout: '1', '2'. Its chromosomes include 'chr1', 'chr10', 'chr11'.
```

Either the sample has no data there, since writers leave out chromosomes
without data, or the names differ (`1` against `chr1`). Check before reading
with `set(chroms) - set(reader.chrom_sizes)`. Under pytest's
`filterwarnings = error`, the warning is an exception.

## PyTorch

Make each dataset item a whole batch, read in one call. A `BigWigReader`
pickles, carrying its index if it has read it, so it works in DataLoader
workers under any start method. The workers already run in parallel, so read
on one thread in each:

```python
import numpy
import torch
from figwig import BigWigReader

class Windows(torch.utils.data.Dataset):
	def __init__(self, path, chroms, starts, width, batch_size):
		self.bw, self.chroms, self.starts = BigWigReader(path), chroms, starts
		self.width, self.batch_size = width, batch_size

	def __len__(self):
		return -(-len(self.starts) // self.batch_size)

	def __getitem__(self, i):
		batch = slice(i * self.batch_size, (i + 1) * self.batch_size)
		return torch.from_numpy(self.bw.read(self.chroms[batch],
			self.starts[batch], self.width, n_jobs=1))

if __name__ == '__main__':
	starts = numpy.arange(1_000_000, 2_000_000, 1000)
	chroms = numpy.full(len(starts), "chr1")
	data = Windows("ENCFF830RWF.bigWig", chroms, starts, 1000, batch_size=64)
	loader = torch.utils.data.DataLoader(data, batch_size=None, num_workers=2,
		multiprocessing_context="spawn")
	y = torch.cat(list(loader))
	print(y.shape, float(y.sum()))
	# torch.Size([1000, 1000]) 28857.0
```

When every window fits in memory, read them all once with `n_jobs=-1` and
wrap the result with `torch.from_numpy`, which does not copy.

## Falling back to another reader

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

figwig's tests check it against pybigtools' `values()` bit for bit, except
that pybigtools sums overlapping intervals where figwig raises. A missing file
or a URL raises `FileNotFoundError`, which this does not catch.
