# Converting: bw2fw, fw2bw, bam2fw, reorder, merge, info

```bash
figwig bw2fw -o k562.fw -m tracks.tsv -g hg38 -p 8 atac.+.bw atac.-.bw dnase.bw   # bigWigs -> one file
figwig bw2fw -o k562.fw --append -p 8 h3k27ac.bw                                  # add tracks in place
figwig bw2fw -o wgbs.fw --missing nan -p 8 methylation.bw                         # keep 0 apart from no data
figwig fw2bw k562.fw -o bigwigs -p 8 [--select assay=ATAC-seq | -t atac.+ atac.-]  # tracks -> bigwigs/<name>.bw + k562.tsv
figwig bam2fw -s hg38.chrom.sizes -n k562 -m files.tsv -p 8 a.bam b.bam           # reads -> k562.fw, a pair of tracks per file
figwig bam2fw -s hg38.chrom.sizes -n pbmc -f -p 8 --barcode-column 4 --cells cells.tsv fragments.tsv.gz   # a track per cell
figwig reorder k562.fw by_track.fw --order tracks                                 # or positions, -t, --groups, -l
figwig merge -o all.fw k562.fw hepg2.fw                                           # copy files' tracks into one
figwig info k562.fw [-b]                                                          # genome, order, tile, chromosomes, size, a track table
```

## Common to every command

- An existing output is refused unless `-f` (`--force` for `bam2fw`, whose
  `-f` means fragments); `--append` adds to one instead. Outputs are
  written beside their path and moved into place when complete, so a
  killed conversion leaves no half-written file. `-p` (1 by default) is
  threads; `merge` takes none. `-v` prints a line per group and a summary
  on stderr. A missing input is a message, Ctrl-C exits 130.
- `-m` is a metadata table: tab-separated (or `.csv`), a header, one row
  per input, matched on its first column by the input's path or base name.
  A `name` column names the track, `dtype` sets its dtype, and every other
  column is an attribute kept as a string (an empty cell is no attribute).
  A space-separated table fails with "the table has one column".
- `fw2bw` writes the same table beside its bigWigs, named after the file,
  so `bw2fw -m` reads it back: names and attributes round-trip.

## bw2fw

- Chromosomes come from the bigWigs, which must agree on lengths; `-s
  sizes` sets them instead, must agree with every bigWig on the
  chromosomes they share (an hg19 bigWig with hg38 sizes is refused), must
  share at least one with every bigWig (`1` against `chr1` is refused rather
  than written as zeros), warns about chromosomes it drops, and a bigWig
  lacking one is 0 there. Use `-s`
  to give files that will be merged the same chromosome set.
- `-t` tile, `--groups` tracks per group, `--dtype` one dtype for all,
  `-l` level (0 for raw chunks), `-z` zoom levels beside the tile sums
  (`bam2fw` takes it too; `reorder` and `merge` keep the input's).
  `--summary-index` writes the regional summary index (`bam2fw` takes it
  too; `reorder` and `merge` keep an input's unless `--no-summary-index`;
  not with `--append`). `--append` keeps the file's chromosomes, tile and
  level and refuses `-s`,
  `-t`, `-l`; a chromosome the file lacks is dropped with a warning.
- Values equal `read_bigwig(..., missing=0.0, keep_nan=True)` on the
  bigWig: uncovered bases 0, NaN intervals NaN (the track is then float32).
- `--missing nan` makes uncovered bases NaN too, for tracks whose zeros are
  measurements (methylation, conservation) or whose explicit 0.0 intervals
  must survive (fold change, p-value): `bw2fw` then `fw2bw` with `--missing
  nan` on both is lossless, where the defaults drop 0.0 intervals. The
  convention is recorded as the file's `missing` attribute, which `fw2bw`
  follows by default; a float32 track costs a compressed file under 1% and
  doubles an uncompressed one.
- A bigWig with an infinite value (some ENCODE p-value tracks) is refused;
  clip it first.

## fw2bw

`<track name>.bw` per track in `-o`; a name another track already has
gets the track's index (`atac.5.bw`) and the table says which is whose.
Zeros and a float track's NaN are left out, as `BigWigWriter` leaves them
out with `missing=0.0`; `--missing nan` leaves NaN out and writes the
zeros, for a file made with `bw2fw --missing
nan`. `-z N` writes N zoom levels (0 by default; a browser wants 10). There
is no bedGraph export: `fw2bw` then `bigWigToBedGraph`, or `read_entries`.

## bam2fw

`figwig bam2bw`'s options, read the same way on `-p` cores, but each input
file becomes its own pair of tracks `<name>.+` and `<name>.-` (one with
`-u`) in one group, named by the table's `name` column or the file name;
the table's other columns are the tracks' attributes, plus `strand`. `-r`
divides by each file's own depth. Counts are written as entries at level 1
(`-l` otherwise); with `-c` (aligned-base coverage) as runs, each track its
own group. Each BAM's header is checked against `-s` first: a chromosome
both name must have one length and the BAM must share a chromosome with
the sizes, or the file is refused as another assembly's or naming's
(`figwig bam2bw` warns and goes on, as bam2bw does). Needs the `bam2bw`
extra; Linux and macOS.

`--barcode-column 4` (the column, from 1; 4 in a Cell Ranger fragments
file) writes a fragments file as one track per cell, named by its barcode,
every cell of a file in one group, or the fewest groups a chunk's 2**29
values allow (32,768 cells at the default tile; `--groups N` for smaller
ones), no strand. `--cells cells.tsv` (tab-separated or `.csv`, a header, barcodes in
the first column) keeps only those cells, in that order, with a `name`
column as track name, `dtype` as dtype and every other column (a cell type)
as an attribute; a raw 10x file holds hundreds of thousands of non-cell
barcodes, so pass the called cells. Fragments of unlisted barcodes are left
out (`-v` counts them). Use `-f` to count both ends, as for any fragments
file. Memory is 8 bytes per fragment end during the scan and about three
times that at the peak. Several files give `<name>.<barcode>` tracks, a
group per file.

## reorder and merge

- `reorder` reads every nonzero base and writes it in the order asked:
  `tracks`/`csr` or `positions`/`csc`. `-t` and `--groups` default to the
  input's (16,384 and 8 from track order); `-l` to the input's from
  position order to position order, else 6 in position order and 0 in
  track order, so a reordering to track order is raw and 3 to 4x larger
  unless `-l` is given. The output
  may be the input itself.
- `merge` copies chunks, no decoding, from files written with the same
  chromosomes, tile, level and order (`reorder` one first otherwise) and
  distinct track names; it drops the dead bytes an append left. `-g`
  records a genome where the files disagree; the output may be an input.
- `info -b` counts chunk bytes, table and index bytes, a regional summary
  index's bytes, and the bytes nothing uses (an append's leftovers, which
  `merge` drops, an appended file's old summary index among them) and any past
  the header's length (a killed append, written over by the next).
