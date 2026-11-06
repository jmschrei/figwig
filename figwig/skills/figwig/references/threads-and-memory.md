# Threads, memory and the first call

Speedups move with the CPU, the file and the load; the rules below are
stable, and the last section measures the numbers where they matter.

## Reading

- bigWigs: a read sorts its windows, finds the data blocks they touch and
  splits them into batches of 256, one per thread, up to `n_jobs`. A read
  of a few hundred windows gets one or two threads whatever `n_jobs` is;
  reading every window in one call gains up to 8 to 32 threads. A reused
  reader beats a path, which re-reads the index on every call.
- figwig files: one thread per 64 chunks touched, up to `n_jobs`, but no
  more than one per 256 KiB of chunks decoded and per 2 MiB of dense output
  (at least 4 where asked for; a read summed per bin over fewer windows
  than threads keeps them all), with the pool kept between reads, and
  batches cut so that each thread gets at least two, however large the
  chunks, and a read of many small groups gets no batches of a chunk or
  two; on a read of several groups each thread takes the next batch as it
  ends one. Gains end near 8 threads for sparse
  reads and continue to 32 for dense float tracks, which read at the
  defaults 1.6 to 3.3x faster than their bigWigs on 1 to 32 threads. `sums`
  run every group at once; `paired=True` over integer tracks and short
  intervals is one read, and otherwise sums its batches of integer tracks
  side by side in position order; `labels=` over several groups of integer
  tracks runs in bands of windows side by side; a few long windows summed
  per bin (a browser's view) are read as parts of whole bins, whose float
  sums differ from one thread's only in rounding. A read of fewer than 1 in
  16 of a group's tracks (one cell of thousands) passes over chunks that hold
  none of them; each batch loads in one GIL-free call where libc's `pread`
  and the `deflate` package both load (not on Windows). The indexes a read needs
  load on the calling thread while each spans fewer than 131,072 tiles, and
  from the first that spans more on up to 8 threads, each holding the one
  it lays out (a browser session of six genome-wide groups peaks at 169 MB
  on 8 threads, against 146 MB one at a time). Sums in position order take no copy of the output per
  thread; track order takes one per thread, which the 8 GiB guard counts.
- In DataLoader workers pass `n_jobs=1`; `num_workers` is the parallelism.

## Writing

- bigWigs: with `n_jobs` above 1, a planning thread lays each batch out, the
  `n_jobs` pool threads encode and compress it, and a summary thread folds
  it in, so the calling thread only takes the next batch; the writer runs
  `n_jobs + 2` threads. With libdeflate gains end near 8 threads; zlib keeps
  gaining to 32. Zoom levels add a pass over the data, 1.3 to 5x
  the time of a write without them, and their memory grows with threads.
- figwig files: whole tiles encode on the pool, in up to one task per
  thread of at least 4 MB of tiles per call, and intervals and partial
  tiles there too, but a tile a call continues or leaves open; the writer
  holds about 0.5 GB plus two to three times one call's values, so aim for
  calls of a few hundred MB. `write_sparse` sorts and checks a call's
  entries on the pool, and `write` checks and copies its values there, in
  waves of 4 MB per thread; with threads a writer writes the file on a
  thread of its own (64 MB queued at most). The track-order writer and
  `bw2fw` stop gaining near 8 threads.
- `figwig bam2bw`: gains end near `-p 8`; peak memory is a few GB for a
  10 GB BAM, nearly double with `-u -f`. `-c` on a sorted, indexed BAM of
  128 MiB and 5 M mapped reads or more (`bam2fw -c`: 128 MiB and 50,000
  reads per file) is read region by region at `-p 8` or more and keeps
  gaining to `-p 32`, holding only the regions in flight, which grow with
  `-p`.

## Memory

- A dense output is `n × channels × width × itemsize`: 8,192 windows × 48
  tracks × 2,048 bases is 3.2 GB as float32. `out=` reuses one array.
- A sparse figwig read costs about 40 bytes per entry while sorting, besides
  the 25 of the returned arrays; a `write_sparse` call about 60 bytes per
  entry (write a chromosome at a time).
- A bigWig write's working memory does not grow with the size of a call.
- Threads multiply across processes: 8 workers at `n_jobs=8` is 64
  threads. Keep processes × `n_jobs` within the free cores, and set
  `OPENBLAS_NUM_THREADS=1` for many short processes (numpy's OpenBLAS
  otherwise starts a thread per CPU at import).

## The first call

The first call in a fresh environment compiles numba kernels: a second or
two for the bigWig reader and `figwig bam2bw`, about 20 s for the first
dense figwig read and up to 20 s more for each further kind (sparse,
another output dtype, sums, `read_entries`), a few seconds for a
track-order read or a write. numba caches them next to the
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
