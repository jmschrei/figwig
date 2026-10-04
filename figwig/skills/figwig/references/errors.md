# Errors and warnings

Messages are quoted up to the first value they fill in.

## Reading

| Message | Cause | Fix |
|---|---|---|
| `X is not a bigWig file.` | not a bigWig; or a figwig file that was never closed, or whose `with` block raised | check the path; write the file again and let `close` run |
| `X is a bigBed file; figwig reads bigWig files.` | bigBed | pybigtools or pyBigWig |
| `X is a big-endian file, which figwig does not read.` | big-endian bigWig | pyBigWig |
| `The chromosome tree of X cannot be read…`, `The data index of X cannot be read…` | corrupt or truncated file | download it again; or fall back (`references/reading.md`) |
| `N windows overlap data blocks that figwig cannot read in X: …` | a block that does not decompress, an unknown section type, or intervals that overlap or are unsorted | fall back to pybigtools, which sums overlapping intervals |
| `N windows start before 0 or end past 2**32 - 1, such as chr1:-5-5.` | windows centred near a chromosome's start | `numpy.maximum(starts, 0)` |
| `starts must be integers, not float64.` | float starts, often a pandas column that held a NaN | `.astype("int64")` |
| `chroms must have one name per start, or be a single name.` | lengths differ | pass one name, or one per window |
| `out must be a writeable, C-contiguous float32 array of shape (n, width).` | wrong `out` | allocate `numpy.empty(shape, dtype=numpy.float32)`; a list of files needs `(n, len(files), width)` |
| `n_jobs must be at least 1, or -1 for every CPU.` | `n_jobs=0` or below -1 | |
| `FileNotFoundError` on an `https://` path | figwig does not read URLs | download the file first |
| `UserWarning: N windows are on chromosomes not in X, and are 0.0 throughout: …` | no data on those chromosomes, or `1` against `chr1` naming | `set(chroms) - set(reader.chrom_sizes)`; rename, or expect `missing` there |

## Writing

Every call that raises writes nothing; the writer stays usable.

| Message | Cause | Fix |
|---|---|---|
| `chromosome 'chr10' is written after 'chr9', but comes before it in chrom_sizes.` | calls not in `chrom_sizes` order; `BigWigReader.chrom_sizes` is in name order | loop over `writer.chrom_sizes`, or build `chrom_sizes` in writing order |
| `window 0 on 'chr1' starts at 50, before the end of what was written on it before, 101.` | a call's items start before the last call's end | sort the batches by position before writing |
| `window 1 on 'chr1', at 5, starts before window 0 ends, at 10; they must not overlap.` | overlapping windows or intervals in one call | merge or trim them; for overlapping predictions, average them first |
| `window 0 on 'chr2' runs past the end of its chromosome, at 5000, where its values must be NaN.` | a window over a chromosome's end with values there | set the values past the end to NaN |
| `chromosome 'chrX' is not one of the writer's chromosomes.` | missing from `chrom_sizes` | add it, or drop those items |
| `values must be finite or NaN, but … holds an infinite value.` | `inf` | replace it, or set it to NaN to leave it out |
| `… holds values outside float32's range.` | float64 values too large for float32 | rescale |
| `values must have one row per start, of shape (n, width), not (n,); pass ends to write intervals.` | 1-D values without `ends` | `values[:, None]` for single bases, or pass `ends` |
| `values must have one channel per path, of shape (n, 1, width), not (n, width).` | a list of paths needs a channel axis | pass one path, or `values[:, None]` |
| `values must hold one value per interval, of shape (n,), not …` | 2-D values with `ends` | one value per interval |
| `engine='libdeflate' needs the deflate package: pip install deflate.` | `deflate` missing | install figwig's `fast` extra, or `engine='zlib'` |
| `engine='libdeflate' needs libdeflate's functions, which could not be loaded…` | Windows | `engine='zlib'` or `'auto'` |
| `level must be from 0 to 9 with engine 'zlib', not 12.` | level out of range | zlib 0-9, libdeflate 0-12, isal 0-3 |
| `this BigWigWriter has been closed.` | `write` after `close` or after the `with` block | write inside the block |

## figwig bam2bw

| Message | Cause | Fix |
|---|---|---|
| `Filenames must end in one of .bam, .sam, .tsv, .tsv.gz, .bed, .bed.gz.` | another extension | rename, such as `.txt.gz` to `.tsv.gz` |
| `-p/--parallel must be a number of cores, or negative to count back from the number of CPUs, not 0.` | `-p 0` | `-p 8`, or `-p -1` for every CPU |
| `--mate_pairs only supports BAM/SAM input files.` | `-mp` with BED/tsv | drop `-mp` |
| the minus-strand bigWig is empty | BED/tsv input has no strand | `-u` |
| `X encountered in input but not in FASTA/chrom sizes.` (with `-v`) | chromosome missing from `-s` | expected for decoy and unplaced contigs; otherwise a naming mismatch or the wrong assembly |
