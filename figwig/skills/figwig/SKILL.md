---
name: figwig
description: >-
  Read and write bigWig files fast with figwig. Use it to read per-base
  signal under many genomic windows, such as peaks, training regions or a
  model's inputs, from one bigWig or several into a float32 numpy array on
  several threads. Use it to write bigWigs from numpy arrays, model
  predictions, intervals or per-base counts, and to convert BAM, SAM,
  BED or fragment files to bigWig coverage tracks with `figwig bam2bw`.
  Use its own figwig file (.fw) to hold many tracks in one file with
  attributes per track, read in windows from any subset of them as
  integers, and convert bigWigs or reads to and from it with `figwig
  bw2fw`, `fw2bw` and `bam2fw`. It fires whenever code reads values from
  or creates a .bw/.bigWig/.fw file, replaces pyBigWig, pybigtools or
  pybbi loops, builds a data loader over bigWig or figwig tracks, or turns
  reads into 5'-end count tracks. This file is a router: read the matching
  reference file before writing code.
---

# figwig

figwig reads and writes bigWig files into and out of numpy, and its own
figwig file (`.fw`) holds many tracks in one file. Its threads run in
parallel, because zlib, libdeflate and its numba kernels run without the
GIL. It depends only on numpy and numba.

```python
from figwig import BigWigReader
from figwig import read_bigwig
from figwig import BigWigWriter
from figwig import write_bigwig
from figwig import FigWigReader
from figwig import read_figwig
from figwig import FigWigWriter
from figwig import write_figwig
```

## When it is the wrong tool

| Need | Use instead |
|---|---|
| a bigBed or a big-endian bigWig | pybigtools or pyBigWig; figwig raises `ValueError` |
| a bigWig at a URL | download it first, or use pyBigWig; figwig takes a URL for a path and raises `FileNotFoundError` |
| summaries from a bigWig's zoom levels, such as a whole-chromosome mean or `stats()` | pyBigWig or pybigtools; figwig reads every base of a bigWig and ignores its zoom levels (a figwig file's `tile_sums` serves this) |
| a bigWig whose intervals overlap | pybigtools, which sums them; figwig raises |

## Conventions that hold everywhere

- Coordinates are 0-based and half-open, as in BED. Window `j` is
  `[starts[j], starts[j] + width)` on `chroms[j]`, and its values go to row `j`
  whatever order the windows are given in.
- A bigWig read returns float32: one file gives `(n, width)`, and a list of
  files `(n, len(files), width)`, which is `(batch, channels, length)`. A
  figwig file's read gives `(n, tracks, width)` in the tracks' own dtype.
- `missing=0.0` by default on bigWigs: a base no interval covers reads as 0,
  where pyBigWig gives NaN, and a bigWig write leaves out every base whose
  value is `missing` or NaN, so reading with the same `missing` gives the
  values back. A figwig file has no `missing`: a base never written is 0.
- A base past the end of its chromosome is NaN in a float output.
- `n_jobs=8` by default on every reader and writer call. `-1` uses every CPU
  this process may run on. `figwig bam2bw -p` defaults to 1 core.
- A file figwig cannot read exactly raises `ValueError`. Catch it to fall
  back to another reader.

## Install

`pip install "figwig[fast]"`, or `uv add "figwig[fast]"`. The `fast` extra
adds libdeflate, which compresses several times faster than zlib and
inflates figwig chunks about twice as fast. `figwig bam2bw` needs the
`bam2bw` extra (`pip install "figwig[bam2bw]"`), which adds pysam, pyfaidx,
biopython, tqdm, isal and deflate, and runs on Linux and macOS only, since
pysam does not support Windows. `device="cuda"` reads of figwig files need
the `cuda` extra, cupy for CUDA 12 (`cupy-cuda13x` or `cupy-cuda11x`
otherwise). Python 3.10 or later.

## Task → reference

| The task is… | Read |
|---|---|
| reading windows from one bigWig or several: shapes, `missing`, chromosome names, reusing a reader | `references/reading.md` |
| a PyTorch Dataset or DataLoader over bigWig tracks | `references/reading.md`, then `references/threads-and-memory.md` |
| writing model predictions, intervals or per-base counts to a bigWig | `references/writing.md` |
| writing a genome-wide track a batch at a time | `references/writing.md` |
| choosing `n_jobs` or `-p`, the expected gain from threads, memory, the first call's compile time | `references/threads-and-memory.md` |
| converting BAM, SAM, BED or fragment files to bigWig | `references/bam2bw.md` |
| a figwig file (`.fw`): many tracks in one file, read in windows by attribute, summed per label, per bin or over intervals, or straight onto a GPU (`device="cuda"`) | `references/figwig-files.md` |
| single cells, reads or any thousands of sparse tracks as a figwig file: `write_sparse`, `sparse=True`, `labels=`, the two orders, tile and groups | `references/figwig-files.md` |
| adding tracks to a figwig file, tile sums and zoom levels, `verify=True`, or converting bigWigs or reads to and from one (`figwig bw2fw`, `fw2bw`, `bam2fw`, `reorder`, `merge`, `info`) | `references/figwig-files.md` |

## Symptom → reference

| You see… | Read |
|---|---|
| any `ValueError`, `TypeError` or `UserWarning` raised by figwig's bigWig reader, writer or `bam2bw` | `references/errors.md` |
| any error from a figwig file (`.fw`): its `FigWigReader`, `FigWigWriter` or converters | the Errors table of `references/figwig-files.md` |
| a bigWig written by figwig that no reader opens ("is not a bigWig file") | `references/errors.md` |
| values that are 0 where you expected NaN, or the reverse | `references/reading.md` |
| `figwig bam2bw` running on one core, or a minus-strand bigWig that is empty | `references/bam2bw.md` |
| threads that do not make a read faster | `references/threads-and-memory.md` |
| `tracks must make up whole groups`, or a figwig read decoding more tracks than asked for | `references/figwig-files.md` |

The full documentation is at <https://figwig.readthedocs.io>.
