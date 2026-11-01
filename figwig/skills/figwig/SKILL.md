---
name: figwig
description: >-
  Read and write bigWig files fast with figwig, and its own figwig file
  (.fw) that holds many tracks in one file. Use it to read per-base signal
  under many genomic windows (peaks, training regions, a model's inputs)
  from one bigWig or several into a numpy array on threads; to write
  bigWigs from numpy arrays, model predictions, intervals or per-base
  counts; to convert BAM, SAM, BED or fragment files to count tracks with
  `figwig bam2bw` or `bam2fw`; and to build a figwig file of many tracks,
  cells or reads with attributes per track, read in windows from any subset
  of them, summed per label, per bin or over intervals, or straight onto a
  GPU. It fires whenever code reads values from or creates a
  .bw/.bigWig/.fw file, replaces pyBigWig, pybigtools, pybbi or deepTools
  loops, builds a data loader over genomic tracks, or turns reads into
  count tracks. This file is a router: read the matching reference file
  before writing code.
---

# figwig

A multithreaded bigWig reader and writer, into and out of numpy, and the
figwig file (`.fw`): many tracks in one file, a JSON record per track.
Depends on numpy and numba only; docs at <https://figwig.readthedocs.io>.
Public API: `BigWigReader`/`read_bigwig`, `BigWigWriter`/`write_bigwig`,
`FigWigReader`/`read_figwig`, `FigWigWriter`/`write_figwig`; commands
`figwig bam2bw`, `bam2fw`, `bw2fw`, `fw2bw`, `reorder`, `merge`, `info`.
Read signatures from the code; the files here hold what the code does not
say.

## When it is the wrong tool

| Need | Use instead |
|---|---|
| a bigBed or a big-endian bigWig | pybigtools or pyBigWig; figwig raises `ValueError` |
| a bigWig at a URL | download it first; figwig takes a URL for a path and raises `FileNotFoundError` |
| a bigWig's zoom-level summaries (`stats()`, a whole-chromosome mean) | pyBigWig or pybigtools; figwig reads every base. A figwig file's `tile_sums` serves this |
| a bigWig whose intervals overlap | pybigtools, which sums them; figwig raises |
| a genome browser track from a figwig file | `figwig fw2bw -z 10`; browsers do not read `.fw` |

## Conventions that hold everywhere

- Coordinates are 0-based, half-open. Window `j` is `[starts[j], starts[j] + width)`
  on `chroms[j]` and lands in row `j`, whatever order the windows are given in.
  Windows may overlap and repeat; `starts` must be integers, at least 0.
- A bigWig read is float32, `(n, width)` for one file and `(n, files, width)`
  for a list, even of one. A figwig read is `(n, tracks, width)` in the
  tracks' own dtype.
- bigWigs have `missing` (0.0 by default; pyBigWig gives NaN): a base no
  interval covers reads as it, and a write leaves out bases whose value is
  it or NaN. A figwig file has no `missing`: a base never written is 0.
- A base past the end of its chromosome is NaN in a float output.
- `n_jobs=8` by default on every reader and writer, `-1` for every CPU;
  `-p` on the commands defaults to 1.
- Anything figwig cannot read exactly raises `ValueError` with the cause.

## Install

`pip install "figwig[fast]"` (libdeflate: faster writes, faster `.fw` reads);
`"figwig[bam2bw]"` for `figwig bam2bw`/`bam2fw` (Linux and macOS);
`"figwig[cuda]"` for `device="cuda"` (cupy for CUDA 12; `cupy-cuda13x` or
`cupy-cuda11x` otherwise). Python 3.10+.

## Task → reference

| The task is… | Read |
|---|---|
| read windows from one bigWig or several; reuse readers; a Dataset or DataLoader over bigWigs | `references/reading.md` |
| write model predictions, intervals or per-base counts to bigWigs; a genome-wide track a batch at a time; files identical to pyBigWig's | `references/writing.md` |
| BAM, SAM, BED or fragment files → bigWig count tracks | `references/bam2bw.md` |
| bigWigs ↔ figwig file (`bw2fw`, `fw2bw`), reads → figwig tracks (`bam2fw`), `reorder`, `merge`, `info`, the metadata table, `--missing nan` | `references/converters.md` |
| read a figwig file: tracks by attribute, dtype, `sparse=`, `labels=` pseudobulks, `bins=`, `sums` over peaks or genes, `tile_sums`, zoom levels, reads or cells as `channels=`/`items=`, GPU output, DataLoader | `references/figwig-reading.md` |
| write a figwig file: tracks and dtypes, tile, groups and order, `write_sparse` for cells or reads, `append=True`, `add_file`, `verify=True`, `summary_index=` | `references/figwig-writing.md` |
| a whole job: a training set from many bigWigs or BAMs, genome-wide predictions, a single-cell file, summaries over peaks, a lossless signal round trip, Fiber-seq reads as tracks | `references/recipes.md` |
| choose `n_jobs` or `-p`, estimate memory, the first call's compile time, measure on the target machine | `references/threads-and-memory.md` |

## Symptom → reference

| You see… | Read |
|---|---|
| any `ValueError`, `TypeError` or `UserWarning` from figwig, or a command's message | `references/errors.md` |
| "is not a bigWig file" on a file figwig wrote | `references/errors.md`: it was never closed |
| 0 where you expected NaN, or the reverse; numbers that differ from pyBigWig or deepTools | `references/reading.md` for bigWigs, `references/figwig-reading.md` for `.fw` |
| `figwig bam2bw` on one core, or an empty minus-strand bigWig | `references/bam2bw.md` |
| threads not making a read faster; a first call that takes 30 s | `references/threads-and-memory.md` |
| `tile_sums`, or `sums` of integer tracks, over one chromosome slow on a big multitrack file | `references/figwig-writing.md`: `summary_index=True` |
| a figwig read slower than expected, or decoding more tracks than asked | `references/threads-and-memory.md` for threads and batches, then `references/figwig-writing.md`: tile, groups and order |
| "has been replaced or rewritten since this reader opened it" | `references/errors.md`: open the reader again |
