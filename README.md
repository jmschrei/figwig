# figwig

[![Unit Tests](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml) [![Documentation Status](https://readthedocs.org/projects/figwig/badge/?version=latest)](https://figwig.readthedocs.io/en/latest/?badge=latest)

[[docs](https://figwig.readthedocs.io/en/latest/index.html)][[release notes](https://figwig.readthedocs.io/en/latest/whats_new.html)]

A fast, multithreaded reader of many bigWig windows at once, into numpy.

Loading training data for a genomics model means reading the signal under tens
or hundreds of thousands of windows: 1,000 bp around every peak and background
region of an experiment, say. figwig reads all of them in one call, straight
into a float32 array, on as many threads as you give it.

The common readers make one call per window, and each call holds Python's GIL,
so adding threads does not help them. figwig decompresses with zlib's own
`uncompress()` and decodes with numba, both without the GIL, so its threads
run in parallel. It sorts the windows by position itself, so they can be given
in any order, and a data block that several windows share is decompressed
once.

figwig depends only on numpy and numba.

## Installation

figwig is not on PyPI yet. Install it from a clone:

```bash
git clone https://github.com/jmschrei/figwig.git
cd figwig
pip install .
```

### Development install

```bash
git clone https://github.com/jmschrei/figwig.git
cd figwig
uv sync --extra dev
uv run pytest
```

## Usage

```python
import numpy
from figwig import BigWig

bw = BigWig("ENCFF830RWF.bigWig")
bw.chroms                      # {'chr1': 248956422, 'chr10': 133797422, ...}

chroms = numpy.array(["chr1", "chr1", "chr2"])
starts = numpy.array([1_000_000, 2_500_000, 300_000])
y = bw.read(chroms, starts, width=1000, n_jobs=8)
y.shape, y.dtype               # ((3, 1000), dtype('float32'))
```

Window `j` covers `[starts[j], starts[j] + width)` on `chroms[j]` and is
written into row `j`. `chroms` can also be a single name for every window.

`BigWig` reads the file's data index on its first read and keeps it, so
reading one file many times, as a data loader does, pays for the index once.
`read` also takes `out=`, a float32 array of shape `(n, width)` to fill rather
than allocate a new one each batch. For a single read there is
`read_windows(path, chroms, starts, width)`.

The values are those pybigtools' `values(chrom, start, end, missing=missing)`
gives, cast to float32:
- each base gets the value of the interval covering it;
- a base no interval covers is `missing`, 0 unless given;
- a base past the end of its chromosome is NaN;
- an interval whose value is NaN is treated as covering nothing.

A window on a chromosome the file does not have, which writers leave out when
it has no data, is `missing` throughout, and a warning names the chromosomes.
pyBigWig's `values()` gives NaN where no interval covers a base, which
`missing=numpy.nan` matches.

To use the result in torch, `torch.from_numpy(y)` wraps it without a copy.

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

A program that has to read such files can catch the error and fall back to
another reader. figwig does not read bigWigs over HTTP, return binned
summaries, use the zoom levels, or write files.

## Speed

The test was 167,750 windows of 1,000 bp: the training peaks and negatives of
ENCODE ATAC-seq experiment ENCSR123WME, read from two ENCODE bigWigs on tmpfs.

Each configuration ran in its own process, three times, and the table gives
the median. To give the other libraries more threads, the windows were split
into 8 chunks read on a thread pool, each chunk with its own file handle.
"Sorted" means the windows were given sorted by position, which helps the
others' caching. figwig sorts them itself.

The machine was 2x AMD EPYC 9575F, with Python 3.13 and numba 0.65. Every
library returned identical values on every window.

| library | ATAC, 1 thread | ATAC, 8 threads | DNase, 1 thread | DNase, 8 threads |
|---|---|---|---|---|
| pybigtools 0.2.5, `values()` per window, sorted | 1.24 s | 1.46 s | 1.96 s | 2.25 s |
| pyBigWig 0.3.26, `values()` per window, sorted | 3.96 s | 4.11 s | 4.49 s | 4.68 s |
| pybbi 0.4.2, `stackup`, sorted | 9.59 s | 9.46 s | 10.75 s | 10.63 s |
| figwig | 0.80 s | **0.135 s** | 1.85 s | **0.276 s** |

The ATAC file (ENCFF830RWF, 238 MB) holds sparse counts. The DNase file
(ENCFF989SAK, 645 MB) is denser. On one thread figwig is level with
pybigtools on the denser file. Its advantage is that its threads run in
parallel.

## Origin

figwig began as the bigWig reader inside
[tangermeme](https://github.com/jmschrei/tangermeme)'s `extract_loci`
(PR #107). It was checked there against pybigtools, bit for bit, on the
training and validation sets of 92 Cherimoya models and on 3,131 synthetic
cases.
