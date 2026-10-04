# figwig bam2bw

`figwig bam2bw` is bam2bw 0.5.1's command, with the same flags, output
files and messages, and faster readers. It needs the `bam2bw` extra.

```bash
figwig bam2bw my.bam -s hg38.chrom.sizes -n sample -p 8                   # sample.+.bw, sample.-.bw
figwig bam2bw my.bam -s hg38.chrom.sizes -n sample -p 8 -u                # sample.bw
figwig bam2bw fragments.tsv.gz -s hg38.chrom.sizes -n sample -p 8 -f -u   # sample.bw
```

## Defaults to override

| Flag | Default | Effect |
|---|---|---|
| `-p` | 1 | One core. Pass a number of cores; see `references/threads-and-memory.md`. |
| `-z` | 0 | No zoom levels, so genome browsers draw zoomed-out views slowly. Pass `-z 10` for a file to browse. |
| `-u` | off | Two files, one per strand. |
| ends counted | each mapped record's 5' end | Both mates of a pair count, each at its own 5' end. |

## Flags

| Flag | Meaning |
|---|---|
| `-s` | chrom sizes, `.fai` or FASTA; a compressed FASTA must be BGZF |
| `-n` | output prefix: `<n>.+.bw` and `<n>.-.bw`, or `<n>.bw` with `-u` |
| `-u` | one unstranded bigWig |
| `-f` | count both ends of each BAM/SAM read, or of each BED/tsv interval; not with `-3p` |
| `-3p` | count 3' ends instead of 5' ends |
| `-ps`, `-ns` | shift plus-strand and minus-strand positions by this many bases |
| `-mp` | count each read pair once, at a position both mates determine (PRO-seq, PRO-cap); BAM/SAM only |
| `--rna5 read1\|read2` | with `-mp`, the mate carrying the RNA's 5' end; default `read1` |
| `--opposite_strand` | with `-mp`, report the other mate's strand |
| `-sf` | multiply every count by this |
| `-r` | divide by the total read depth before `-sf`, so both strands sum to 1 (or to `-sf`) |
| `-v` | progress bars, and the names of chromosomes not in `-s` |

## Inputs

- Filenames must end in `.bam`, `.sam`, `.tsv`, `.tsv.gz`, `.bed` or `.bed.gz`.
- Several input files are pooled into one track.
- Reads on chromosomes missing from `-s` are dropped. Only `-v` names them.
- A BED or tsv file is read as chromosome, start and end, with no strand, so
  every count goes to the plus file. Pass `-u`, or the minus file is empty.

## Which inputs are fast

| Input | Read by |
|---|---|
| BAM | libdeflate and a numba kernel on the `-p` threads |
| BED/tsv, uncompressed or BGZF | a numba kernel on the `-p` threads |
| SAM, any input with `-mp`, a URL, a `.tsv.gz`/`.bed.gz` that is plain gzip | bam2bw's pysam or line loop, on one core per file, as slowly as bam2bw |

Recompressing a plain-gzip fragments file with `bgzip`, under the same
`.tsv.gz` name, lets the threaded kernel read it.

`-p` is a number of cores, not processes. Fast files are read one after
another, each on every core. Slow files go to a pool of processes, one per
file up to `-p`, alongside them. A negative `-p` counts back from the number
of CPUs (`-1` is all of them), and `-p 0` raises.

## Which flags for which assay

| Assay | Flags |
|---|---|
| ATAC-seq, DNase-seq, ChIP-seq BAM, paired or single end | defaults: each mate's 5' end is an event of its own |
| scATAC-seq fragments file | `-f -u` |
| PRO-seq, PRO-cap and other paired-end protocols where a pair is one tag | `-mp`, with `--rna5` and `-3p` as the protocol needs; read on one core |

## Outputs

The bigWigs hold the same values as bam2bw 0.5.1's, but not the same bytes,
since figwig compresses with libdeflate at level 1 and writes its own zoom
levels. Compare decoded values, never checksums. The first call in a new
environment compiles figwig's kernels, which takes about a second; later
calls load them from numba's cache.
