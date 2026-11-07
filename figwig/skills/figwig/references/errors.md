# Errors and warnings

Messages are quoted up to the first value they fill in. A call that raises
writes nothing; writers stay usable.

## Any call

| Message | Cause → fix |
|---|---|
| numba `TypingError: … Invalid use of type(CPUDispatcher(<function …>)) with parameters …`, `TypeError: can't unbox array from PyObject into native value` | processes that compiled at the same time into one numba cache saved one kernel version over another → delete the cache's `.nbi`/`.nbc` files (`figwig/__pycache__`, or `NUMBA_CACHE_DIR`), then warm up in one process before starting workers (`references/threads-and-memory.md`) |

## Reading bigWigs

| Message | Cause → fix |
|---|---|
| `X is not a bigWig file.` | not a bigWig, or one figwig wrote whose writer never reached `close` (an exception in the `with` block) → write it again |
| `X is a bigBed file; figwig reads bigWig files.`, `X is a big-endian file, which figwig does not read.` | → pybigtools or pyBigWig |
| `The chromosome tree of X cannot be read…`, `The data index of X cannot be read…` | corrupt or truncated → download again, or fall back (`references/reading.md`) |
| `N windows overlap data blocks that figwig cannot read in X: …` | a block that does not decompress, an unknown section type, overlapping or unsorted intervals → pybigtools, which sums overlaps |
| `N windows start before 0 or end past 2**32 - 1, such as chr1:-5-5.` | → `numpy.maximum(starts, 0)` |
| `starts must be integers, not float64.` | a pandas column that held a NaN → `.astype("int64")` |
| `chroms must have one name per start, or be a single name.` | |
| `out must be a writeable, C-contiguous float32 array of shape (n, width).` | a list of files needs `(n, len(files), width)` |
| `FileNotFoundError` on an `https://` path | → download first |
| `UserWarning: N windows are on chromosomes not in X, and are 0.0 throughout: …` | no data there, or `1` against `chr1` → `set(chroms) - set(reader.chrom_sizes)` |

## Writing bigWigs

| Message | Cause → fix |
|---|---|
| `chromosome 'chr10' is written after 'chr9', but comes before it in chrom_sizes.` | calls not in `chrom_sizes` order (ENCODE's is name order) → loop over `writer.chrom_sizes`, or build `chrom_sizes` in writing order |
| `window 0 on 'chr1' starts at 50, before the end of what was written on it before, 101.` | → sort batches by position |
| `window 1 on 'chr1', at 5, starts before window 0 ends, at 10; they must not overlap.` | the full width counts even where values are `missing` or NaN → average overlaps, write single bases (`(n, 1)` values) or intervals |
| `window 0 on 'chr2' runs past the end of its chromosome, at 5000, where its values must be NaN.` | → set the values past the end to NaN |
| `chromosome 'chrX' is not one of the writer's chromosomes.` | → add it to `chrom_sizes`, or drop the items |
| `values must be finite or NaN, but … holds an infinite value.`, `… holds values outside float32's range.` | → replace or rescale; NaN leaves a base out |
| `values must have one row per start, of shape (n, width), not (n,); pass ends to write intervals.` | → `values[:, None]` for single bases, or `ends` |
| `values must have one channel per path, of shape (n, 1, width), not (n, width).` | a list of paths → one path, or `values[:, None]` |
| `values must hold one value per interval, of shape (n,), not …` | 2-D values with `ends` |
| `engine='libdeflate' needs the deflate package: pip install deflate.`, `engine='libdeflate' needs libdeflate's functions, which could not be loaded…` (Windows) | → the `fast` extra, or `engine='zlib'` |
| `level must be from 0 to 9 with engine 'zlib', not 12.` | zlib 0-9, libdeflate 0-12, isal 0-3 |
| `engine='isal' requires figwig[isal].` (reading) | → the `isal` extra, or `engine='auto'`, which reads with zlib without it |
| `engine must be 'auto', 'zlib' or 'isal', not 'libdeflate'.` (reading) | the reader's engines are the inflaters; libdeflate is a writer engine |
| `this BigWigWriter has been closed.` | `write` after `close` |

## figwig bam2bw and bam2fw

| Message | Cause → fix |
|---|---|
| `Filenames must end in one of .bam, .sam, .tsv, .tsv.gz, .bed, .bed.gz.` | → rename, `.txt.gz` to `.tsv.gz` |
| `-p/--parallel must be a number of cores, or negative to count back from the number of CPUs, not 0.` | → `-p 8`, or `-p -1` |
| `--mate_pairs only supports BAM/SAM input files.` | → drop `-mp` on BED/tsv |
| `--coverage does not take --mate_pairs`, `argument -f/--fragments: not allowed with argument -c/--coverage`, `--barcode-column counts each cell's fragment ends; --coverage is not taken with it` | `-c` with a mode that fixes one position per read or pair, or with per-cell counting → one of `-c`, `-f`, `-3p`, `-mp` |
| the minus-strand bigWig is empty | BED/tsv has no strand → `-u` |
| `X encountered in input but not in FASTA/chrom sizes.` (with `-v`) | expected for decoys and unplaced contigs; otherwise a naming mismatch or the wrong assembly |
| `--barcode-column is a column number from 4`, `--barcode-column reads BED/tsv files of fragments, not X` | columns 1-3 are coordinates; a BAM has no barcode column |
| `the line '...' does not hold a chromosome, a start, an end and a barcode in column N` | too few columns, or a coordinate that is not an integer; there is no per-line fallback in cell mode |
| `no fragment of X is on a chromosome of SIZES; the file's chromosomes include ...` | `22` against `chr22`, or the wrong assembly |
| `X is cut short, or is not a gzip file: ...`, `X ends without the empty block that ends a BGZF file, so it was cut short` | a truncated download; cell mode refuses it where bam2bw's own loop raises `EOFError` |
| `no fragment of X was counted: the file holds none` | an empty file, or only `#` lines |
| `lists a barcode more than once`, `--cells lists N barcodes, more than the M ...`, `holds more than N barcodes, which is the most a file with M chromosomes can keep apart` | a duplicate row in `--cells`; too many cells for the key layout → give `--cells` the cells to keep, or fewer chromosomes in `-s` |

## figwig files

| Message | Cause → fix |
|---|---|
| `tracks must make up whole groups` | a `write` named some tracks of a group |
| `written again after the group of ... moved on from it`, `starts at ..., before the end of what was written on it before` | a group returned to an earlier chromosome, or items out of order across calls |
| `chromosome 'chr1' is written after 'chr2', but comes before it in chrom_sizes; in track order …` | track order takes each track's chromosomes in `chrom_sizes` order across calls → loop over `writer.chrom_sizes` |
| `a window may run past the end only where its values are NaN or 0` | |
| `track ... is uint16, and a value is ...`, `lies beyond float32's range` | a declared dtype the values do not fit → a wider one, or none |
| `values must be finite or NaN, but one is infinite`, `an integer lies beyond 2**53, which a figwig file cannot hold exactly` | in `write`, `write_sparse` or a bigWig given to `bw2fw` → clip or scale first |
| `entries ... both name base ...` | duplicate (track, position) entries in `write_sparse` → sum them first |
| `which an earlier call already wrote`, `was begun by write` | `write` and `write_sparse` on one tile in two calls |
| `came from another file, whose tracks cannot be written to` | `write` to a track `append` or `add_file` copied in |
| `falls in tile N of 'chr1', which the group of 'x' holds from the file it came from` | `write_sparse` to a tile a copied-in group already holds → put those entries in a new file, or write the file again with them |
| `is being appended to by another writer` | two `append=True` writers → wait for the first to close |
| `FigWigWriter(...) was never closed, and its values were not written.` (a `ResourceWarning`; the path is left as it was) | a `write` without `close` or a `with` block → use `with` |
| `was written with order\|tile\|level ... where this writer has`, `was written with other chromosomes or lengths: it has ...` | `add_file` or `merge` of unlike files → `reorder` one, or `bw2fw -s` the same sizes |
| `none of the chromosomes of x.bw is in the sizes file: it has chr1, chr2, ... and the sizes have 1, 2, ...` | `bw2fw -s` with another assembly's sizes, or names without `chr` → a sizes file naming the bigWig's chromosomes |
| `figwig bam2fw: file does not contain alignment data`, `no BGZF EOF marker; file may be truncated` (pysam's words, passed on), `x.bam is not a file.` | the input is not a BAM, is cut short, or the path is wrong |
| `x.bam aligns to a chr1 of 249,250,621 bases, but the sizes give it 248,956,422: the reads and the sizes must be on one assembly`, `none of the chromosomes of x.bam is in the sizes` (`bam2fw` refuses; `figwig bam2bw: warning: ...` goes on, leaving the reads outside the sizes out) | `-s` is another assembly's sizes (hg19 reads, hg38 sizes) or another naming's → the assembly the BAM was aligned to |
| `the dense output would take ... GiB` | → `sparse=True`, `labels=`, fewer windows or `out=` |
| `a sum lies outside what ... holds`, `has values from ... to ..., which uint8 does not hold`, `pass dtype= to read as another type` | `dtype=` or `out=` too narrow or of another type |
| `is not one of the file's chromosomes, 'chr1', ...` | the file holds only the chromosomes it was written with |
| `no track has an attribute` | `select` with a key no track has; the message names the keys |
| `nan must be 'propagate', 'skip' or 'count'`, `nan= applies to sums: give bins too` | |
| `ends must hold one end per start`, `an interval's end is before its start`, `paired=True needs one track per interval` | `sums` arguments |
| `has no attribute 'read'`, `has no track with channel=...`, `are both channel=... of read=...` | `channels=`/`items=`: a track without the attribute, a missing item × channel, or two tracks for one |
| `keeps no summary of the group of`, `keeps no zoom level k of the group of` | the group's tile sums passed 256 MiB (in a file from figwig 0.2.0 or earlier, the whole file's); the file was written without `zooms=k` (`reader.zooms`) |
| `has been replaced or rewritten since this reader opened it`, `has been truncated` | a conversion or a closed append changed the file → open the reader again |
| `it is version ... of the format, and this figwig reads version 1`, `needs a feature this figwig does not know (bits ...)`, `its flags hold a bit this figwig does not know` | a newer figwig wrote it → upgrade |
| `... does not match its checksum` (tables, index, header, or a chunk with `verify=True`), `cannot be read as a figwig file`, `a chunk of ... cannot be read`, `is not a regular file` | damaged, truncated, crafted, or not a file; always `ValueError` |
| `exists; --append adds tracks to it, and -f writes over it`, `exists; -f writes over it`, `exists; --force writes over it` | a command onto an existing output |
| `... has no row for ... in its first column ... The table has one column; a table is tab-separated, or comma-separated with a .csv extension.` | a space-separated metadata table |
