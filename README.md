# figwig

[![Unit Tests](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml) [![Documentation Status](https://readthedocs.org/projects/figwig/badge/?version=latest)](https://figwig.readthedocs.io/en/latest/?badge=latest)

[[docs](https://figwig.readthedocs.io/en/latest/index.html)][[release notes](https://figwig.readthedocs.io/en/latest/whats_new.html)]

A fast, multithreaded reader and writer of bigWig files, into and out of
numpy.

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

figwig also writes bigWigs, from intervals, single bases or dense arrays such
as a model's predictions. It lays the file out the way pyBigWig does, and
compresses its data blocks on several threads, with zlib or, when the optional
`deflate` package is installed, with libdeflate, both without the GIL. On the
two tracks timed below, on 8 threads with libdeflate, it wrote the same files
23 and 26 times faster than pyBigWig without zoom levels, and 16 and 17 times
faster with them.

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
To write faster with libdeflate, install the `fast` extra, which adds the
`deflate` package:

```bash
pip install ".[fast]"
```

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
print(len(bw.chrom_sizes), bw.chrom_sizes["chr1"])
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
from figwig import read_bigwig

chroms = numpy.array(["chr1", "chr1", "chr2"])
starts = numpy.array([1_000_000, 2_500_000, 300_000])
y = read_bigwig(["ENCFF830RWF.bigWig", "ENCFF989SAK.bigWig"], chroms, starts,
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
`read_bigwig` gives `(n, width)`, as `BigWig.read` does.

#### Bases without data

```python
import numpy
from figwig import BigWig

bw = BigWig("ENCFF830RWF.bigWig")
y = bw.read("chr1", [1_000_000], width=1000, missing=numpy.nan)
print(numpy.isnan(y).sum(), (y == 0).sum())
# 907 0

end = bw.chrom_sizes["chr1"]
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

#### Writing a bigWig

```python
from figwig import BigWig
from figwig import BigWigWriter

chroms = {"chr1": 248_956_422, "chr2": 242_193_529}
with BigWigWriter("example.bw", chroms) as bw:
	bw.add("chr1", [1000, 2000], [1500, 2100], [0.5, 2.0])
	bw.add("chr1", [5000, 5003, 5004], values=[3, 1, 1])
	bw.add("chr2", 10_000, values=[0.0, 1.5, 1.5, 0.0, 2.5])

y = BigWig("example.bw").read(["chr1", "chr2"], [5000, 10_000], width=6)
print(y)
# [[3.  0.  0.  1.  1.  0. ]
#  [0.  1.5 1.5 0.  2.5 0. ]]
```

`add` takes values three ways: intervals, as starts, ends and values; single
bases, as positions and values, which is what per-base counts are; and a dense
array of the values of every base from one start, which is what a model's
predictions along a region are. In a dense array, the bases equal to
`missing`, 0.0 unless given, and NaN are left out of the file, so that
`BigWig.read` with the same `missing` reads the array back. Chromosomes are
added in the order of `chroms`, and within one, each call starts at or after
the end of the last, so a long chromosome can be written a part at a time.
The header, the index and the zoom levels are written when the writer is
closed, at the end of the `with` block.

#### Writing values held in memory

```python
import numpy
from figwig import BigWig
from figwig import write_bigwig

rng = numpy.random.default_rng(0)
predictions = rng.random(1_000_000, dtype=numpy.float32)
positions = numpy.array([150, 870, 871, 2_000_000])
counts = numpy.array([2, 1, 4, 1])

write_bigwig("tracks.bw", {"chr1": 248_956_422, "chr2": 242_193_529},
	{"chr2": (positions, counts), "chr1": predictions}, n_jobs=8)

bw = BigWig("tracks.bw")
print(numpy.array_equal(bw.read("chr1", [0], width=1_000_000)[0], predictions))
# True
print(bw.read("chr2", [868], width=5))
# [[0. 0. 1. 4. 0.]]
```

`write_bigwig` takes each chromosome's values as a tuple of (starts, ends,
values) for intervals, a tuple of (positions, values) for single bases, or an
array for a dense array from the start of the chromosome, and writes the
chromosomes in the order of `chroms`.

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
return binned summaries, or use the zoom levels when it reads.

## What it writes

figwig lays a bigWig out the way libBigWig, the C library inside pyBigWig,
does: the header, the chromosome tree, data blocks of at most 32,768 bytes
before compression, the R-tree index over them, and the zoom levels with their
indexes. Intervals become bedGraph sections, single bases varStep sections of
span 1, and dense arrays fixedStep sections of span 1. A file of intervals or
single bases written without zoom levels, with `engine='zlib'` at level 6, is
byte for byte the file pyBigWig writes from the same calls, and the tests
check that it is.

Where libBigWig writes a wrong value, figwig writes the right one, so those
files differ from pyBigWig's there:

- the maximum in the header, which libBigWig misses when the first value is
  the largest, and leaves at the smallest positive double, about 2.2e-308,
  when no value is positive;
- the end of the last data block of each fixedStep call, which libBigWig puts
  6 bases past its last item;
- the sum and sum of squares of the last zoom record of each zoom block,
  which libBigWig leaves at 0. pyBigWig's mean over a whole chromosome, which
  it takes from the zoom levels, then drifts from the exact one: by 2.2% on
  one chromosome of a test file.

libBigWig also writes an empty data block where it changes section type,
which figwig leaves out, and stops making zoom levels at the first level that
needs as many zoom blocks as the one before it, which can leave a sparse track
with only its finest level. figwig skips a level that has no fewer records
than the last one it kept, and goes on to the coarser ones. The levels'
sizes are libBigWig's: 16 times the mean width of an item, or 10 bases if that
is more, then 4 times larger at each level, up to the longest chromosome, for
at most `zooms` levels, 10 by default.

Blocks are compressed with zlib, or with libdeflate when the `deflate`
package is installed and `engine` is `'auto'`, its default, except on
Windows, where libdeflate's functions cannot be loaded. Both write zlib
streams that any bigWig reader can read, and both write the same file
whatever `n_jobs` is. libdeflate is faster: on the data blocks of the counts
track below, at level 6 on one thread, it compressed 142 MB/s to zlib's
32 MB/s, into blocks 0.4% smaller. `engine='isal'` uses ISA-L, faster still
at levels 1 to 3, but at levels 1 and 2 its output can differ from one run to
the next on the same input, so its files are not reproducible byte for byte;
their values are.

Values are laid out in batches of about a million items, which are compressed
on up to `n_jobs` threads while the calling thread lays out the next batch.
The zoom levels are built when the writer is closed, from the data blocks read
back from the file. The finest level's blocks are written as they are
compressed. The other levels' are kept compressed in memory until every level
is built, since whether a level is written depends on how many records it
has, and so are the finest level's where Python has no `os.pread`, as on
Windows.

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
3.10 to 3.14. Its tests run on Linux under each of those versions, and on
macOS and Windows under 3.10 and 3.13. On Windows, Python has no `os.pread`,
so blocks are read by seeking the file under a lock. Where zlib's library
cannot be loaded, as on Windows, blocks are decompressed and compressed by
Python's zlib module instead, with the same values. libdeflate's functions
cannot be loaded from the `deflate` package on Windows either, so there
`engine='auto'` is zlib and `engine='libdeflate'` raises a `ValueError`.
pyBigWig, which the writer's tests compare files with byte for byte, does not
build on Windows, so those comparisons run on Linux and macOS.

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

### Writing

The writers wrote the values of two tracks again, held in memory as numpy
arrays: the 5' ends of the reads of ENCODE ATAC-seq BAM ENCFF877LRY on the
plus strand, as counts at 15,875,960 single bases, and the fold-change signal
of ENCODE snATAC-seq pseudobulk ENCSR206UWN, ENCFF932UQM, as 21,212,477
intervals. figwig and pyBigWig wrote one call per
chromosome, and pybigtools one `write()` from an iterator of tuples. Every
write, opening the file and closing it included, ran in its own process
after a warm-up write of the first 1,000 values of each chromosome, three
times, onto tmpfs, and the table gives the median, at compression level 6. pybigtools 0.2.5's `write()` takes no option for zoom levels and writes
them, so it was timed with them only.

The machine, Python and numpy were those above, with pyBigWig 0.3.26,
pybigtools 0.2.5 and deflate 0.9.0. Every writer's file gave the same values
on 2,000 windows of 1,000 bases read back with figwig.
`benchmarks/compare_writers.py` runs the comparison on any bigWigs.

| writer | counts | counts, zoom levels | signal | signal, zoom levels |
|---|---|---|---|---|
| pyBigWig 0.3.26 | 4.20 s | 12.81 s | 7.42 s | 10.41 s |
| pybigtools 0.2.5 | | 4.25 s | | 5.27 s |
| figwig, zlib, 1 thread | 4.22 s | 12.20 s | 6.17 s | 8.24 s |
| figwig, zlib, 8 threads | 0.53 s | 1.68 s | 0.79 s | 1.16 s |
| figwig, libdeflate, 1 thread | 1.07 s | 4.93 s | 1.75 s | 3.37 s |
| figwig, libdeflate, 8 threads | **0.18 s** | **0.81 s** | **0.29 s** | **0.62 s** |

figwig's files were the size of pyBigWig's: 35.7 and 107.1 MB without zoom
levels, against 35.8 and 108.0 MB, and 214.0 and 126.5 MB with them, against
213.9 and 127.4 MB. pybigtools chooses its zoom levels differently, and wrote
96.4 and 162.2 MB. With zoom levels most of figwig's time is spent building
them. Its peak memory was above pyBigWig's: 565 MB against 355 MB for the
counts without zoom levels on one thread, of which the arrays of values held
318 MB, and 1,502 MB against 1,090 MB with zoom levels on 8 threads.

## Origin

figwig began as the bigWig reader inside
[tangermeme](https://github.com/jmschrei/tangermeme)'s `extract_loci`
(PR #107). It was checked there against pybigtools, bit for bit, on the
training and validation sets of 92 Cherimoya models and on 3,131 synthetic
cases.
