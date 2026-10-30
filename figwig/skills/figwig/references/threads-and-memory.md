# Threads, memory and the first call

Speedups move with the CPU, the file and the load; the rules below are
stable, and the last section measures the numbers where they matter.

## Reading

- bigWigs: a read sorts its windows, finds the data blocks they touch and
  splits them into batches of 256, one per thread, up to `n_jobs`. A read
  of a few hundred windows gets one or two threads whatever `n_jobs` is;
  reading every window in one call gains up to 8 to 32 threads. A reused
  reader beats a path, which re-reads the index on every call.
- figwig files: one thread per 64 chunks touched, up to `n_jobs`, with the
  pool kept between reads, and batches cut so that each thread gets at
  least two, however large the chunks. Gains end near 8 threads for sparse
  reads and continue to 32 for dense float tracks, which read at the
  defaults 1.6 to 3.2x faster than their bigWigs on 1 to 32 threads. Sums
  in position order take no copy of the output per thread; track order
  takes one per thread, which the 8 GiB guard counts.
- In DataLoader workers pass `n_jobs=1`; `num_workers` is the parallelism.

## Writing

- bigWigs: with `n_jobs` above 1, a planning thread lays each batch out, the
  `n_jobs` pool threads encode and compress it, and a summary thread folds
  it in, so the calling thread only takes the next batch; the writer runs
  `n_jobs + 2` threads. With libdeflate gains end near 8 threads; zlib keeps
  gaining to 32. Zoom levels add a pass over the data, 2 to 4x
  the time of a write without them, and their memory grows with threads.
- figwig files: whole tiles encode on the pool, in up to one task per
  thread of at least 4 MB of tiles per call, partial tiles and
  intervals on the calling thread, so write windows for speed; the writer
  holds about 0.5 GB plus two to three times one call's values, so aim for
  calls of a few hundred MB. `bw2fw` stops gaining near `-p 8`; the
  track-order writer does not gain.
- `figwig bam2bw`: gains end near `-p 8`; peak memory is a few GB for a
  10 GB BAM, nearly double with `-u -f`. `-c` on a sorted, indexed BAM of
  128 MiB and 5 M mapped reads or more (`bam2fw -c`: 128 MiB and 50,000
  reads per file) is read region by region at `-p 8` or more and keeps
  gaining to `-p 32`, holding only the regions in flight, which grow with
  `-p`.

## Memory

- A dense output is `n × channels × width × itemsize`: 8,192 windows × 48
  tracks × 2,048 bases is 3.2 GB as float32. `out=` reuses one array.
- A sparse figwig read costs about 40 bytes per entry while sorting; a
  `write_sparse` call about 90 bytes per entry (write a chromosome at a time).
- A bigWig write's working memory does not grow with the size of a call.
- Threads multiply across processes: 8 workers at `n_jobs=8` is 64
  threads. Keep processes × `n_jobs` within the free cores, and set
  `OPENBLAS_NUM_THREADS=1` for many short processes (numpy's OpenBLAS
  otherwise starts a thread per CPU at import).

## The first call

The first call in a fresh environment compiles numba kernels: about a
second for the bigWig reader and `figwig bam2bw`, tens of seconds for the
first dense figwig read and more for each further kind (sparse, another
output dtype, sums, track order, a write). numba caches them next to the
installed package or in the user's cache directory; later processes load
them in a fraction of a second. Warm up with a small call before timing and
before spawning DataLoader workers, each of which loads the cache.

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

Use the real window count and width, and `FigWigReader` the same way. The
repository's `benchmarks/compare_readers.py`, `compare_writers.py` and
`compare_formats.py` time figwig against the other libraries and formats.
