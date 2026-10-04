# Threads, memory and speed

The numbers below were measured on one machine, 2x AMD EPYC 9575F (256
cores) at a load of about 20, with Python 3.13 and numba 0.68, as medians of
3 to 5 runs. Speedups move with the CPU, the file and the load, so measure
on the target machine before promising one (last section).

## Reading

`n_jobs` is the most threads one read uses, counting every thread that does
work while the calling thread waits: 8 by default, and -1 for every CPU the
process may run on. A read sorts its windows, finds the data blocks
they overlap, and splits those blocks into batches of 256, one batch per
thread. A read whose windows touch few blocks gets few threads, whatever
`n_jobs` is: it starts at most one thread per batch, and runs a single batch
on the calling thread, so a large `n_jobs` costs nothing on a small read.

One reader, opened and warmed up once, reading random 1,000 bp windows from a
238 MB ATAC-seq bigWig:

| Windows per read | 1 thread | 4 threads | 8 threads | 32 threads |
|---|---|---|---|---|
| 100 | 3.5 ms | 3.0 ms | 3.0 ms | 3.0 ms |
| 1,000 | 29 ms | 19 ms | 19 ms | 18 ms |
| 10,000 | 235 ms | 74 ms | 52 ms | 53 ms |
| 100,000 | 756 ms | 221 ms | 128 ms | 74 ms |

Reading 167,750 windows sorted by position, file opening included, took
0.82 s on 1 thread, 0.13 s on 8 (6.1 times faster) and 0.074 s on 32 (11
times). On a denser 645 MB DNase-seq bigWig it took 1.85, 0.27 and 0.16 s. pybigtools, the
fastest of the other readers, took 1.25 s on 1 thread and was no faster on 8.

- Batches of a thousand windows or fewer gain little from threads. In
  DataLoader workers pass `n_jobs=1` and get parallelism from `num_workers`.
- The cost per window falls as a read takes more windows: 35 µs per window
  for 100 windows and 7.6 µs for 100,000 on one thread, 0.74 µs for 100,000
  on 32. A training loop reading 128 random peak windows per call took
  4.1 ms per call with a reused reader (32 µs per window), and 5.2 ms given a
  path, which re-reads the index. When the windows fit in memory, read them
  all in one call and index the array each step.
- To read every window once, use `n_jobs` from 8 to 32.

## Writing

`n_jobs` is the most threads compressing data blocks, besides the calling
thread, which lays out the values. Writing 15.9 million per-base counts:

| Engine, zoom levels | 1 | 2 | 4 | 8 | 16 | 32 threads |
|---|---|---|---|---|---|---|
| libdeflate, none | 1.07 s | 0.47 s | 0.25 s | 0.18 s | 0.19 s | 0.23 s |
| zlib, none | 4.22 s | 2.04 s | 1.04 s | 0.53 s | 0.28 s | 0.20 s |
| libdeflate, 10 | 4.85 s | 2.50 s | 1.37 s | 0.75 s | 0.53 s | 0.55 s |

pyBigWig took 4.20 s without zoom levels and 12.80 s with them.

- libdeflate, the default when the `deflate` package is installed, gains
  nothing past about 8 threads. zlib keeps gaining up to 32.
- Zoom levels quadrupled the time here. Pass `zooms=0` when no genome
  browser will draw the file.
- With zoom levels, peak memory grew with threads: 580 MB on 1 thread and
  1,028 MB on 32. Without them it stayed near 580 MB.

## figwig bam2bw

| Input | `-p 1` | `-p 2` | `-p 4` | `-p 8` | `-p 16` | `-p 32` |
|---|---|---|---|---|---|---|
| 2.4 GB ATAC-seq BAM, default flags | 9.35 s | 4.65 s | 2.71 s | 1.74 s | 1.75 s | 1.82 s |
| 18.9 M scATAC fragments, `-f -u` | 2.91 s | 1.74 s | 1.41 s | 1.25 s | 1.30 s | 1.41 s |

Gains end near `-p 8`. bam2bw 0.5.1 took 58.5 s and 28.5 s on these inputs
at `-p 2`.
Peak memory grew from 831 MB at `-p 1` to 1,032 MB at `-p 32` on the BAM.
On a 10 GB BAM at `-p 1` the peak was 2.9 GB, and 4.9 GB with `-u -f`.

## Memory

- A read's output takes `n * width * 4` bytes per file: 671 MB for 167,750
  windows of 1,000 bp. Pass `out=` to reuse one array.
- Each reading thread also holds its batch's decompressed blocks, each at
  most the file's `uncompressBufSize` (32 KB in ENCODE files).
- A write's working memory does not grow with the size of a call: 166 MB
  above the values for 25 million values, 173 MB for 100 million, on 8
  threads without zoom levels.

## The first call

The first call in a new environment compiles figwig's numba kernels: 0.7 s
for the reader, 1.1 s for `figwig bam2bw`. numba caches them next to the
installed package, or in the user's cache directory, and later processes
load them in well under a second. Warm up with a small read before timing.
On inputs bam2bw finishes in under half a second, `figwig bam2bw` took a
median 0.12 s longer, mostly importing numba.

## Oversubscription

Threads multiply across processes: 8 DataLoader workers at the default
`n_jobs=8` start 64 threads. Keep processes times `n_jobs` within the free
cores. On a shared machine, check the load first. numpy's OpenBLAS starts a
thread per CPU when it is imported; set `OPENBLAS_NUM_THREADS=1` for many
short processes, such as one `figwig bam2bw` per file, which do no linear
algebra.

## Measuring on the target machine

```python
import time
import numpy
from figwig import BigWigReader

bw = BigWigReader("signal.bw")
chroms = numpy.full(10_000, "chr1")
starts = numpy.random.default_rng(0).integers(0, 200_000_000, 10_000)
bw.read(chroms[:100], starts[:100], 1000)   # loads the kernels and the index
for n_jobs in (1, 2, 4, 8, 16):
	start = time.perf_counter()
	bw.read(chroms, starts, 1000, n_jobs=n_jobs)
	print(n_jobs, round(time.perf_counter() - start, 3))
```

Use the real window count and width. figwig's repository has
`benchmarks/compare_readers.py` and `benchmarks/compare_writers.py`, which
time figwig against pybigtools, pyBigWig and pybbi on any bigWigs.
