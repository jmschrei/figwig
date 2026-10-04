---
name: figwig
description: >-
  Read and write bigWig files fast with figwig. Use it to read per-base
  signal under many genomic windows, such as peaks, training regions or a
  model's inputs, from one bigWig or several into a float32 numpy array on
  several threads. Use it to write bigWigs from numpy arrays, model
  predictions, intervals or per-base counts, and to convert BAM, SAM,
  BED or fragment files to bigWig coverage tracks with `figwig bam2bw`.
  It fires whenever code reads values from or creates a .bw/.bigWig file,
  replaces pyBigWig, pybigtools or pybbi loops, builds a data loader over
  bigWig tracks, or turns reads into 5'-end count tracks. This file is a
  router: read the matching reference file before writing code.
---

# figwig

figwig reads and writes bigWig files into and out of numpy. Its threads run
in parallel, because zlib, libdeflate and its numba kernels run without the
GIL. It depends only on numpy and numba.

```python
from figwig import BigWigReader
from figwig import read_bigwig
from figwig import BigWigWriter
from figwig import write_bigwig
```

## When it is the wrong tool

| Need | Use instead |
|---|---|
| a bigBed or a big-endian bigWig | pybigtools or pyBigWig; figwig raises `ValueError` |
| a bigWig at a URL | download it first, or use pyBigWig; figwig takes a URL for a path and raises `FileNotFoundError` |
| summaries from zoom levels, such as a whole-chromosome mean or `stats()` | pyBigWig or pybigtools; figwig reads every base and ignores zoom levels |
| a bigWig whose intervals overlap | pybigtools, which sums them; figwig raises |

## Conventions that hold everywhere

- Coordinates are 0-based and half-open, as in BED. Window `j` is
  `[starts[j], starts[j] + width)` on `chroms[j]`, and its values go to row `j`
  whatever order the windows are given in.
- Reads return float32 arrays. One file gives `(n, width)`. A list of files
  gives `(n, len(files), width)`, which is `(batch, channels, length)`.
- `missing=0.0` by default: a base no interval covers reads as 0, where
  pyBigWig gives NaN. A base past the end of its chromosome is always NaN.
- A write leaves out every base whose value is `missing` or NaN, so reading
  with the same `missing` gives the values back.
- `n_jobs=8` by default on every reader and writer call. `-1` uses every CPU
  this process may run on. `figwig bam2bw -p` defaults to 1 core.
- A file figwig cannot read exactly raises `ValueError`. Catch it to fall
  back to another reader.

## Install

figwig is not on PyPI yet. Install it from a clone of
<https://github.com/jmschrei/figwig> with `pip install ".[fast]"`. The
`fast` extra adds libdeflate, which compresses several times faster than
zlib. `figwig bam2bw` needs the `bam2bw` extra, which adds pysam, pyfaidx,
biopython, tqdm, isal and deflate, and runs on Linux and macOS only, since
pysam does not support Windows. Python 3.10 or later.

## Task → reference

| The task is… | Read |
|---|---|
| reading windows from one bigWig or several: shapes, `missing`, chromosome names, reusing a reader | `references/reading.md` |
| a PyTorch Dataset or DataLoader over bigWig tracks | `references/reading.md`, then `references/threads-and-memory.md` |
| writing model predictions, intervals or per-base counts to a bigWig | `references/writing.md` |
| writing a genome-wide track a batch at a time | `references/writing.md` |
| choosing `n_jobs` or `-p`, the expected gain from threads, memory, the first call's compile time | `references/threads-and-memory.md` |
| converting BAM, SAM, BED or fragment files to bigWig | `references/bam2bw.md` |

## Symptom → reference

| You see… | Read |
|---|---|
| any `ValueError`, `TypeError` or `UserWarning` raised by figwig | `references/errors.md` |
| a file written by figwig that no reader opens ("is not a bigWig file") | `references/errors.md` |
| values that are 0 where you expected NaN, or the reverse | `references/reading.md` |
| `figwig bam2bw` running on one core, or a minus-strand bigWig that is empty | `references/bam2bw.md` |
| threads that do not make a read faster | `references/threads-and-memory.md` |

The full documentation is at <https://figwig.readthedocs.io>.
