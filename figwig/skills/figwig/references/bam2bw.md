# figwig bam2bw

bam2bw 0.5.1's command with the same flags, outputs and messages, and
faster readers. Needs the `bam2bw` extra; Linux and macOS.

```bash
figwig bam2bw my.bam -s hg38.chrom.sizes -n sample -p 8                   # sample.+.bw, sample.-.bw
figwig bam2bw my.bam -s hg38.chrom.sizes -n sample -p 8 -u                # sample.bw
figwig bam2bw fragments.tsv.gz -s hg38.chrom.sizes -n sample -p 8 -f -u   # sample.bw
```

## Defaults to override

| Flag | Default | Why change it |
|---|---|---|
| `-p` | 1 | one core. Gains end near 8, except `-c` on a large sorted, indexed BAM (below), which keeps gaining to 32. |
| `-z` | 0 | no zoom levels. Pass `-z 10` only for a browser track: on a sparse count track it multiplied the file size by 10. It is a maximum; levels no smaller than the last are dropped. |
| `-u` | off | two files, one per strand. BED/tsv input has no strand and lands entirely in `+.bw`, so pass `-u` for it. |
| what counts | every mapped record's 5' end | both mates, duplicates, secondary, supplementary, QC-fail, any MAPQ. Filter first if they should not count: `samtools view -b -F 0xF04 -q 30`. |

## Flags

| Flag | Meaning |
|---|---|
| `-s` | chrom sizes, `.fai` or FASTA (a compressed FASTA must be BGZF) |
| `-n` | output prefix |
| `-f` | count both ends of each read or interval (start and end - 1); not with `-3p` |
| `-3p` | 3' ends instead of 5' |
| `-c` | every aligned base of each read (CIGAR M, = and X runs; a deletion or skipped intron is not covered), or every base of a BED/tsv interval: deepTools `bamCoverage --binSize 1`, which it matched at every base on eleven experiments (ChIP-seq, ATAC, DNase, PRO-cap, RNA-seq, CUT&RUN, MNase, ChIP-nexus, PacBio RNA); written as intervals of one value; not with `-f`, `-3p`, `-mp`. At `-p 8` or more a coordinate-sorted BAM with a `.bai` no older than it is read region by region on every core, holding only the regions in flight, from 128 MiB and 5 M mapped reads in the indexes (`bam2bw`) or 128 MiB and 50,000 mapped reads per file (`bam2fw`); the output is the same |
| `-ps`, `-ns` | added to each start (`-ps`) and end (`-ns`); counting 5' ends that shifts plus-strand reads by `-ps` and minus-strand by `-ns`, with `-3p` the two swap. ATAC-seq Tn5 correction: `-ps 4 -ns -5` |
| `-mp` | one count per read pair at a position both mates fix (PRO-seq, PRO-cap); BAM/SAM only; `--rna5 read1\|read2` picks the mate with the RNA's 5' end, `--opposite_strand` reports the other mate's strand |
| `-sf` | multiply every count |
| `-r` | divide by total read depth before `-sf` |
| `-v` | progress bars and the chromosomes not in `-s` |

| Assay | Flags |
|---|---|
| ATAC-seq BAM | defaults, plus `-ps 4 -ns -5` for the Tn5 shift |
| DNase-seq, ChIP-seq BAM | defaults |
| fragments file (bulk or 10x single-cell) | `-f -u`; 10x fragments are usually Tn5-shifted already |
| PRO-seq, PRO-cap | `-mp`, with `--rna5` and `-3p` as the protocol needs |
| read-span coverage for a browser or a pileup-style target (ChIP-seq, RNA-seq) | `-c`, with `-r` for depth normalization; a spliced read covers its exons only |

## Inputs

- Names must end in `.bam`, `.sam`, `.tsv`, `.tsv.gz`, `.bed` or `.bed.gz`
  (rename `.txt.gz` to `.tsv.gz`). Several inputs are pooled into one track.
- Reads on chromosomes missing from `-s` are dropped silently unless `-v`.
  A BAM whose header gives a chromosome another length than `-s`, or
  shares no chromosome with it, gets a warning on stderr first (another
  assembly's or naming's sizes); `bam2fw` refuses it.
- BAM, and BED/tsv that is uncompressed or BGZF, are read on the `-p`
  threads. A plain-gzip `.tsv.gz` is inflated in order on one core:
  recompress it with `bgzip` under the same name. SAM, `-mp` and URLs go
  through bam2bw's pysam loop on one core, as slowly as bam2bw.
- `-p` is cores, not processes: fast files are read one after another on
  every core, slow ones in a pool of processes beside them. `-p -1` is every
  CPU; `-p 0` raises.

## Many files

One command per output track. Run several processes at a low `-p` rather
than one file at a time at a high `-p`, keeping processes × `-p` within the
free cores and processes × peak memory within RAM (a few GB for a 10 GB
BAM). Set `OPENBLAS_NUM_THREADS=1` on each: numpy's OpenBLAS otherwise
starts a thread per CPU at import, which costs CPU-seconds per process on a
many-core machine.

## Outputs

The same entries as bam2bw 0.5.1, not the same bytes (libdeflate level 1,
figwig's own zoom levels): compare decoded values, never checksums. For
many files as tracks of one file, or per-file tracks, use `figwig bam2fw`
(`references/converters.md`).
