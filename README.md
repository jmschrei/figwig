# figwig

[![Unit Tests](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml) [![Documentation Status](https://readthedocs.org/projects/figwig/badge/?version=latest)](https://figwig.readthedocs.io/en/latest/?badge=latest)

[[docs](https://figwig.readthedocs.io/en/latest/index.html)][[release notes](https://figwig.readthedocs.io/en/latest/whats_new.html)]

A fast, multithreaded reader of many bigWig windows at once, into numpy.

Loading training data for a genomics model means reading the signal under tens
or hundreds of thousands of windows: 1,000 bp around every peak and background
region of an experiment, say. figwig reads all of them in one call, straight
into a float32 array, from one bigWig or from several at once, on as many
threads as you give it.

The common readers make one call per window, and each call holds Python's GIL,
so adding threads does not help them. figwig decompresses with zlib's own
`uncompress()` and decodes with numba, both without the GIL, so its threads
run in parallel. It sorts the windows by position itself, so they can be given
in any order.

figwig depends only on numpy and numba. A file it cannot read with certainty
raises a `ValueError` that says what it found, rather than being guessed at,
so that a pipeline can fall back to another reader.

## Installation

figwig is not on PyPI yet. Install it from a clone:

```bash
git clone https://github.com/jmschrei/figwig.git
cd figwig
pip install .
```

It needs Python 3.10 or later, numpy 1.23 or later, and numba 0.58 or later.

### Development install

```bash
git clone https://github.com/jmschrei/figwig.git
cd figwig
uv sync --extra dev
uv run pytest
```

## Usage

The examples read two public ENCODE bigWigs: the ATAC-seq signal of HG02943
(ENCFF830RWF, 238 MB) and the DNase-seq signal of dorsolateral prefrontal
cortex (ENCFF989SAK, 645 MB).

```bash
wget https://www.encodeproject.org/files/ENCFF830RWF/@@download/ENCFF830RWF.bigWig
wget https://www.encodeproject.org/files/ENCFF989SAK/@@download/ENCFF989SAK.bigWig
```

#### Reading windows

```python
import numpy
from figwig import BigWig

bw = BigWig("ENCFF830RWF.bigWig")
print(len(bw.chroms), bw.chroms["chr1"])
# 149 248956422

chroms = numpy.array(["chr1", "chr1", "chr2"])
starts = numpy.array([1_000_000, 2_500_000, 300_000])
y = bw.read(chroms, starts, width=1000, n_jobs=8)
print(y.shape, y.dtype)
# (3, 1000) float32
print(y.sum(axis=1))
# [116.  43.  10.]
```

Window `j` covers `[starts[j], starts[j] + width)` on `chroms[j]` and is
written into row `j`, whatever order the windows are given in. `chroms` can
also be a single name for every window. Both can be lists, numpy arrays or
pandas Series.

`BigWig` reads the file's data index on its first read and keeps it, so
reading one file many times, as a data loader does, pays for the index once.
`read` also takes `out=`, a float32 array to fill rather than allocate a new
one each time.

#### Reading several bigWigs into one array

```python
import numpy
from figwig import read_windows

chroms = numpy.array(["chr1", "chr1", "chr2"])
starts = numpy.array([1_000_000, 2_500_000, 300_000])
y = read_windows(["ENCFF830RWF.bigWig", "ENCFF989SAK.bigWig"], chroms, starts,
	width=1000)
print(y.shape)
# (3, 2, 1000)
print(y.sum(axis=2))
# [[116. 351.]
#  [ 43. 240.]
#  [ 10.  63.]]
```

Channel `i` holds the values from the `i`-th file, in the
`(batch, channels, length)` layout that sequence models take: the plus and
minus strands of a stranded assay, say, or one track per task. Reading into
that array directly avoids stacking one array per file, which takes time and
twice the memory. Every file's work shares one pool of threads. A path is
opened, and its index read, on every call; to read the same files repeatedly,
pass `BigWig` objects instead, which keep their indexes. With one file,
`read_windows` gives `(n, width)`, as `BigWig.read` does.

#### Bases without data

```python
import numpy
from figwig import BigWig

bw = BigWig("ENCFF830RWF.bigWig")
y = bw.read("chr1", [1_000_000], width=1000, missing=numpy.nan)
print(numpy.isnan(y).sum(), (y == 0).sum())
# 907 0

end = bw.chroms["chr1"]
print(bw.read("chr1", [end - 3], width=6))
# [[ 0.  0.  0. nan nan nan]]

y = bw.read(["1", "2"], [1_000_000, 300_000], width=1000)
# UserWarning: 2 windows are on chromosomes not in ENCFF830RWF.bigWig, and are
# 0.0 throughout: '1', '2'. Its chromosomes include 'chr1', 'chr10', 'chr11'.
```

A base that no interval covers is `missing`, 0 unless given. A base past the
end of its chromosome is NaN. A window on a chromosome the file does not have
is `missing` throughout, with a warning: writers leave out chromosomes without
data, and a different naming scheme (`1` against `chr1`) looks the same. See
[Values and coordinates](#values-and-coordinates).

#### Training with PyTorch

```python
import numpy
import torch
from figwig import BigWig

class Windows(torch.utils.data.Dataset):
	def __init__(self, path, chroms, starts, width, batch_size):
		self.bw, self.chroms, self.starts = BigWig(path), chroms, starts
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

Each item is a whole batch, read in one call. A `BigWig` can be pickled, so
it can go to DataLoader workers under any start method. When every window fits
in memory, reading all of them once is simpler still, and `torch.from_numpy`
wraps the result without a copy.

#### Binned values

```python
from figwig import BigWig

bw = BigWig("ENCFF830RWF.bigWig")
y = bw.read(["chr1", "chr1", "chr2"], [1_000_000, 2_500_000, 300_000],
	width=1024)
binned = y.reshape(len(y), -1, 32).mean(axis=-1)
print(binned.shape)
# (3, 32)
```

figwig reads every base and does not use a file's zoom levels. A mean over
bins of bases, as above, gives binned targets; use `numpy.nanmean` if windows
run past the end of a chromosome.

#### Falling back to another reader

```python
import numpy
import pybigtools
from figwig import BigWig

def read(path, chroms, starts, width):
	try:
		return BigWig(path).read(chroms, starts, width)
	except ValueError:
		bw = pybigtools.open(path)
		return numpy.array([bw.values(chrom, start, start + width) for chrom,
			start in zip(chroms, starts)], dtype=numpy.float32)
```

figwig's tests compare it against pybigtools' `values()`, bit for bit, which
makes pybigtools a natural fallback for the files figwig refuses. Where
intervals overlap, pybigtools sums them.

## Values and coordinates

Coordinates are 0-based and half-open, as in BED files: window `j` covers the
bases `starts[j]` to `starts[j] + width - 1`. Every value is copied from the
file's float32, not computed.

| base | value |
|---|---|
| covered by an interval | the interval's value |
| covered by no interval | `missing`, 0.0 unless given |
| covered by an interval whose value is NaN | `missing` |
| past the end of its chromosome | NaN |
| on a chromosome the file does not have | `missing`, with a warning |

The first four rows are what pybigtools' `values(chrom, start, end,
missing=missing)` gives, cast to float32, and the tests check figwig against
it bit for bit. pybigtools raises for a chromosome the file does not have.
pyBigWig's `values()` gives NaN for a base that no interval covers, which
`missing=numpy.nan` matches, and raises for a window that runs past the end
of its chromosome or is on a chromosome the file does not have.

## What it reads, and what it refuses

figwig reads bigWig files with bedGraph, varStep or fixedStep sections,
compressed or not, at base-pair resolution. Everything else raises a
`ValueError` saying what was found, rather than being guessed at:

- a file that is not a little-endian bigWig, such as a bigBed or a big-endian
  bigWig;
- a chromosome tree or data index that is corrupt, or cut short by a
  truncated file;
- a data index whose entries are unsorted or span two chromosomes;
- a window that starts before 0 or ends past 2**32 - 1;
- a window overlapping a data block that cannot be decompressed, holds a
  section of another type, or has intervals that are unsorted, overlap each
  other or those of a neighbouring block, or lie outside the block's index
  entry. Overlapping intervals give a base two values, and readers disagree
  on which to report: pybigtools sums them.

A read that raises because a data block cannot be read may already have
written other windows into `out`. figwig does not read bigWigs over HTTP,
return binned summaries, use the zoom levels, or write files.

## Threads, memory and the first call

`n_jobs` is the most threads a read uses, 8 by default, and -1 gives one per
CPU the process may run on. A read is split into batches by the file's data
blocks, 256 blocks to a batch, and each batch runs on one thread, so a read
uses more threads only when its windows span more batches. In a DataLoader
the workers already read in parallel, so each can read its batch on one
thread, as above.

The output takes `n * width * 4` bytes for one file, times the number of
files for several. Each thread also holds the decompressed blocks of the
batch it is reading: 256 blocks and any more that its windows run into, each
at most the file's `uncompressBufSize`, which is 32 KB in ENCODE's files.

The first read in a new environment compiles figwig's numba kernels, which
took 0.7 s on the machine below. numba caches them on disk, next to the
installed package, or in the user's cache directory when the package
directory cannot be written, and later processes load them in well under a
second. One `BigWig` can be read from several threads at once, and holds no
open file between reads.

figwig needs a little-endian machine, which every common one is, and Python
3.10 to 3.14. It has been run on Linux. It has code paths for macOS and for
Windows, where Python has no `os.pread`, and the CI workflow runs the tests on
both, but neither has been run yet. Where zlib's shared library cannot be
found, blocks are decompressed by Python's zlib module instead, with the same
values.

## Speed

The test read 167,750 windows of 1,000 bp, centred on the fold-0 training
peaks and negatives of ENCODE ATAC-seq experiment ENCSR123WME, from the two
bigWigs above, held on tmpfs. Of the 167.75 million bases, 11.9 million are
nonzero in the ATAC file and 19.0 million in the DNase file.

Each configuration ran in its own process, three times, and the table gives
the median. To give the other libraries more threads, the windows were split
into 8 chunks read on a thread pool, each chunk with its own file handle.
The table gives every library the windows sorted by position, which helps
the others' caching. figwig sorts them itself: given unsorted, it took
0.144 s and 0.292 s on 8 threads.

The machine was 2x AMD EPYC 9575F, with Python 3.13.5, numpy 2.5.3 and numba
0.68. Every library returned the same values on every window, once pyBigWig's
NaN for bases without an interval was set to 0. `benchmarks/compare_readers.py`
runs the comparison on any bigWigs and BED files.

| library | ATAC, 1 thread | ATAC, 8 threads | DNase, 1 thread | DNase, 8 threads |
|---|---|---|---|---|
| pybigtools 0.2.5, `values()` per window | 1.25 s | 1.42 s | 1.98 s | 2.21 s |
| pyBigWig 0.3.26, `values()` per window | 3.95 s | 4.08 s | 4.50 s | 4.64 s |
| pybbi 0.4.2, `stackup` | 9.61 s | 9.48 s | 10.75 s | 10.63 s |
| figwig 0.1.0 | 0.81 s | **0.135 s** | 1.85 s | **0.277 s** |

On one thread figwig took 0.81 s to pybigtools' 1.25 s on the ATAC file, and
1.85 s to 1.98 s on the denser DNase file. Its advantage is that its threads
run in parallel.

## Origin

figwig began as the bigWig reader inside
[tangermeme](https://github.com/jmschrei/tangermeme)'s `extract_loci`
(PR #107). It was checked there against pybigtools, bit for bit, on the
training and validation sets of 92 Cherimoya models and on 3,131 synthetic
cases.
