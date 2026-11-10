# figwig

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/figwig?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=ORANGE&left_text=downloads)](https://pepy.tech/projects/figwig) [![Unit Tests](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml/badge.svg)](https://github.com/jmschrei/figwig/actions/workflows/python-package.yml) [![Documentation Status](https://readthedocs.org/projects/figwig/badge/?version=latest)](https://figwig.readthedocs.io/en/latest/?badge=latest)

[[docs](https://figwig.readthedocs.io/en/latest/index.html)][[release notes](https://figwig.readthedocs.io/en/latest/whats_new.html)]

A fast, multithreaded reader and writer of bigWig and figwig files, into and out of
numpy. Also contains a formal specification of the figwig file format, which holds
any number of per-base tracks over one genome with a JSON record of attributes for
each, and reads windows from any subset of them in one call, into
`(windows, tracks, width)`. `figwig bw2fw`, `fw2bw` and `bam2fw` convert bigWigs
and reads to and from a figwig file.

Built on work from Nezar Abdennur and Jack Huey.

Loading training data for a genomics model means reading the signal under tens
or hundreds of thousands of windows: 1,000 bp around every peak and background
region of an experiment, say. figwig reads all of them in one call, straight
into a float32 array, from one bigWig or from several at once, and writes
bigWigs from the same windows, such as a model's predictions, or from
intervals. The common readers make one call per window, and each call holds
Python's GIL, so adding threads does not help them. figwig decompresses and
compresses with zlib or libdeflate and decodes with numba, all without the
GIL, so its threads run in parallel. It depends only on numpy and numba.

Its command, `figwig bam2bw`, turns SAM/BAM files of reads, or BED/tsv files
of fragments, into bigWigs of per-base counts. It is
[bam2bw](https://github.com/jmschrei/bam2bw) with faster readers, the same
outputs, and `-c` added.

| | figwig | fastest other tool |
|---|---|---|
| Read 167,750 windows of 1,000 bp from an ATAC-seq bigWig | **0.103 s** on 8 threads | 1.25 s, pybigtools |
| Write 15.9 million per-base counts | **0.14 s** on 8 threads | 4.20 s, pyBigWig |
| Write 21.2 million intervals, with zoom levels | **0.45 s** on 8 threads | 5.47 s, pybigtools |
| Convert a 2.4 GB ATAC-seq BAM to two stranded bigWigs | **4.65 s**, 910 MB peak, at `-p 2` | 58.48 s, 3,039 MB peak, bam2bw 0.5.1 |
| The same BAM's per-base coverage, like deepTools | **1.6 s** at `-p 8` | 14.9 s on 8 processes, `bamCoverage --binSize 1`, the same value at every base |
| Read 8,192 windows of 2,048 bp from 48 count tracks on chr1 | **0.071 s** on 8 threads, **0.040 s** on 32, from one figwig file | 0.19 s and 0.083 s from 48 bigWigs, figwig; 5.9 s pybigtools; 4.0 s zarr |
| A DataLoader batch of 64 windows x 48 tracks as a CUDA tensor | **4 ms** from one figwig file | 30 ms from the bigWigs, figwig; 120 ms pybigtools |
| 10 pseudobulks of 10,246 cells over 8,192 windows | **0.23 s** on 8 threads | 5.1 s from an in-memory scipy CSR; 95 s from one bigWig per cell |
| Convert 257.6 million 10x fragments to 10,246 per-cell tracks | **8 s** at `-p 8`, **3.9 s** at `-p 32`, 19 GB peak | 140 s, 24 GB, a pandas recipe; 217 s, 8 GB, SnapATAC2 (fragments, not counts) |

Measured on a 2x AMD EPYC 9575F with Python 3.13 and numba 0.68. The
[benchmarks page](https://figwig.readthedocs.io/en/latest/benchmarks.html) has
every setup, the other tables (figwig files against bigWigs, zarr and HDF5 at
6, 48 and 512 tracks; writers; threads) and the plots.

## Installation

```bash
pip install figwig
```

Or, with [uv](https://docs.astral.sh/uv/):

```bash
uv add figwig
```

It needs Python 3.10 or later, numpy 1.23 or later, and numba 0.58 or later.
Optional extras:

- `figwig[fast]` adds the `deflate` package, which writes bigWigs several
  times faster and reads figwig files about twice as fast.
- `figwig[isal]` adds the `isal` package, whose ISA-L reads bigWigs 1.3 to 1.6
  times faster, with the same values.
- `figwig[bam2bw]` adds pysam, pyfaidx, biopython, tqdm, isal and deflate,
  which `figwig bam2bw` needs. `figwig bam2bw` runs on Linux and macOS, since
  pysam does not support Windows.
- `figwig[cuda]` adds cupy (for CUDA 12; install `cupy-cuda13x` or
  `cupy-cuda11x` otherwise), with which `FigWigReader.read(..., device="cuda")`
  lands windows in GPU memory, for `torch.from_dlpack` and its JAX and
  TensorFlow equivalents.

```bash
pip install "figwig[fast]"
pip install "figwig[isal]"
pip install "figwig[bam2bw]"
pip install "figwig[cuda]"
```

### Development install

```bash
git clone https://github.com/jmschrei/figwig.git
cd figwig
uv sync --extra dev
uv run pytest -n 8    # the tests on 8 cores; the first run compiles the kernels
```

## Claude Code Skill

figwig ships a [Claude Code](https://claude.com/claude-code) skill that
teaches a coding agent to use figwig in any project: reading windows,
writing predictions, intervals and counts, converting reads with
`figwig bam2bw`, figwig files and their converters, the rules a write
follows, what each error means, and how much threads gain. Install it into
`~/.claude/skills/figwig` with:

```bash
figwig install-skill
```

After upgrading figwig, run `figwig install-skill --force` to replace the
installed copy, which it otherwise leaves, with a message. `-d DIRECTORY`
installs into another skills directory, and `--symlink` links to the copy
inside the installed package instead of copying it.

## Python API

The examples read two public ENCODE bigWigs: the ATAC-seq signal of HG02943
(ENCFF830RWF, 238 MB) and the DNase-seq signal of dorsolateral prefrontal
cortex (ENCFF989SAK, 645 MB).

```bash
wget https://www.encodeproject.org/files/ENCFF830RWF/@@download/ENCFF830RWF.bigWig
wget https://www.encodeproject.org/files/ENCFF989SAK/@@download/ENCFF989SAK.bigWig
```

#### Reading windows

```python
import numpy
from figwig import BigWigReader

bw = BigWigReader("ENCFF830RWF.bigWig")
chroms = numpy.array(["chr1", "chr1", "chr2"])
starts = numpy.array([1_000_000, 2_500_000, 300_000])
y = bw.read(chroms, starts, width=1000, n_jobs=8)
print(y.shape, y.dtype)
# (3, 1000) float32
print(y.sum(axis=1))
# [116.  43.  10.]
```

Window `j` covers `[starts[j], starts[j] + width)` on `chroms[j]`, 0-based as
in BED files, and is written into row `j`, whatever order the windows are
given in. A base that no interval covers is `missing`, 0 unless given, and a
base past the end of its chromosome is NaN, or `oob` if given. A
`BigWigReader` reads the file's index once and keeps it, and can be pickled to
DataLoader workers.

#### Reading several bigWigs into one array

```python
import numpy
from figwig import read_bigwig

chroms = numpy.array(["chr1", "chr1", "chr2"])
starts = numpy.array([1_000_000, 2_500_000, 300_000])
y = read_bigwig(["ENCFF830RWF.bigWig", "ENCFF989SAK.bigWig"], chroms, starts,
	width=1000)
print(y.shape)
# (3, 2, 1000)
```

Channel `i` holds the values from the `i`-th file, in the
`(batch, channels, length)` layout that sequence models take, and every
file's work shares one pool of threads.

#### Writing windows

```python
import numpy
from figwig import read_bigwig
from figwig import write_bigwig

chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
chroms = numpy.array(["chr2", "chr1", "chr1"])
starts = numpy.array([300_000, 2_500_000, 1_000_000])
y_hat = numpy.random.default_rng(0).random((3, 2, 1000), dtype=numpy.float32)

write_bigwig(["plus.bw", "minus.bw"], chrom_sizes, chroms, starts, y_hat)
y = read_bigwig(["plus.bw", "minus.bw"], chroms, starts, width=1000)
print(numpy.array_equal(y, y_hat))
# True
```

`write_bigwig` takes what `read_bigwig` gives, such as a model's predictions
for the same windows. A base whose value is `missing`, 0.0 unless given, or
NaN is left out of the file, so that reading with the same `missing` gives
the values back.

#### Writing a batch at a time

```python
from figwig import BigWigReader
from figwig import BigWigWriter

chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
with BigWigWriter("example.bw", chrom_sizes) as writer:
	writer.write("chr1", [1000, 2000], [0.5, 2.0], ends=[1500, 2100])
	writer.write("chr1", [5000, 5003, 5004], [[3], [1], [1]])
	writer.write("chr2", [10_000], [[0.0, 1.5, 1.5, 0.0, 2.5]])

y = BigWigReader("example.bw").read(["chr1", "chr2"], [5000, 10_000], width=6)
print(y)
# [[3.  0.  0.  1.  1.  0. ]
#  [0.  1.5 1.5 0.  2.5 0. ]]
```

`BigWigWriter` writes intervals (with `ends`), single bases (windows of width
1) and windows as they come, so a track larger than memory can be written a
batch at a time. Each call's items come after the last call's, in the order
of `chrom_sizes`. The header, index and zoom levels are written when the
writer is closed.

#### Reading and writing figwig files

```python
import numpy
from figwig import FigWigReader
from figwig import FigWigWriter

chrom_sizes = {"chr1": 248_956_422, "chr2": 242_193_529}
tracks = [
	{"name": "atac.+", "assay": "ATAC-seq", "strand": "+"},
	{"name": "atac.-", "assay": "ATAC-seq", "strand": "-"},
]
counts = numpy.random.default_rng(0).poisson(0.1, (3, 2, 2114))
chroms = numpy.array(["chr2", "chr1", "chr1"])
starts = numpy.array([300_000, 2_500_000, 1_000_000])

with FigWigWriter("example.fw", chrom_sizes, tracks, genome="hg38") as fw:
	fw.write(chroms, starts, counts)

fw = FigWigReader("example.fw")
y = fw.read(chroms, starts, width=2114, tracks=fw.select(strand="+"))
print(y.shape, y.dtype, numpy.array_equal(y[:, 0], counts[:, 0]))
# (3, 1, 2114) uint8 True
```

A figwig file holds many tracks over one genome, each with attributes that
`select` matches, and is read like a bigWig: many windows of one width in
one call, from any subset of the tracks, into `(windows, tracks, width)` in
the tracks' own dtype. The same read gives the nonzero bases as a sparse
list with `sparse=True`, the tracks summed per label with `labels=`, or
each window summed per bin with `bins=`; `sums` gives each track's total
over intervals of any lengths; `tile_sums` gives each track's sum per tile,
or per bin of a zoom level, without decoding a chunk; and `device="cuda"`
lands the output of `read` and `sums` in GPU memory, where
`torch.from_dlpack` takes it without a copy. A base no track wrote is 0, as
is a bigWig's no-data unless `bw2fw --missing nan` keeps them apart.

A file is written in position order, for reading windows across many
tracks, or in track order, for reading a few tracks over the genome, and
`figwig reorder` rewrites it in the other; `write_sparse` takes nonzero
bases as entries, such as the cut sites of single cells. `fw.tracks`,
`fw.stats` and `fw.chrom_sizes` say what a file holds, as `figwig info`
does from the shell. The first read of each kind in a new environment
compiles its kernels, with nothing printed meanwhile; numba caches them
beside the package, or in the user's cache directory when the package
directory is read-only, and `NUMBA_CACHE_DIR` chooses another.

The [figwig files](https://figwig.readthedocs.io/en/latest/figwig.html) guide
covers the layout, the API and the converters, and
[the format page](https://figwig.readthedocs.io/en/latest/format.html)
specifies the format byte by byte, with a Kaitai Struct definition and a
reference reader. The
[reading](https://figwig.readthedocs.io/en/latest/reading.html) and
[writing](https://figwig.readthedocs.io/en/latest/writing.html) guides cover
values and coordinates, chromosomes a file does not have, PyTorch data
loaders, binned values, the files figwig refuses and falling back to another
reader, compression, zoom levels, and files identical to pyBigWig's.

## Command line

#### figwig bam2bw

```bash
figwig bam2bw my.bam -s hg38.chrom.sizes -n test-run -p 8                  # test-run.+.bw, test-run.-.bw
figwig bam2bw fragments.tsv.gz -s hg38.chrom.sizes -n test-run -f -u -p 8  # test-run.bw
```

`figwig bam2bw` is bam2bw 0.5.1 with the same arguments, output files and
messages, except that `-p` is a number of cores, 1 unless given, `-c` is new,
and a BAM whose header disagrees with the sizes is reported on stderr first.
By default it counts the 5' end of every mapped read at each base and writes
the two strands to two bigWigs; `-u` writes one, `-f` counts both ends of each
fragment, `-3p` the 3' ends, `-c` every aligned base of each read (the
coverage deepTools' `bamCoverage` gives), `-ps`, `-ns`, `-sf` and `-r` shift
and scale the counts, and `-z` writes zoom levels. SAM files, `--mate_pairs`,
remote files and malformed records go through bam2bw's own pysam or line
loop, so that their result, or their error, is bam2bw's. The bigWigs hold
bam2bw's entries but not its bytes. `figwig bam2bw -h` lists the arguments,
and the [documentation](https://figwig.readthedocs.io/en/latest/bam2bw.html)
says how files are read and shared across cores.

#### figwig bw2fw, fw2bw, bam2fw, fiberbam2fw, reorder, merge and info

```bash
figwig bw2fw -o k562.fw -m tracks.tsv -g hg38 -p 8 atac.+.bw atac.-.bw dnase.bw   # bigWigs -> k562.fw
figwig fw2bw k562.fw -o bigwigs -p 8 --select assay=ATAC-seq                      # tracks -> bigwigs/<name>.bw
figwig bam2fw a.bam b.bam -s hg38.chrom.sizes -n k562 -m files.tsv -p 8          # reads -> k562.fw
figwig bam2fw fragments.tsv.gz -s hg38.chrom.sizes -n pbmc --barcode-column 4 --cells cells.tsv -f -p 8   # a track per cell
figwig fiberbam2fw fibers.bam -n fibers -g hg38 -p 8                               # Fiber-seq reads -> a track per read and channel
figwig reorder k562.fw by_track.fw --order tracks                                # the other order
figwig merge -o all.fw k562.fw hepg2.fw                                          # files -> one file
figwig info k562.fw                                                              # what the file holds
```

`figwig bw2fw` writes bigWigs as the tracks of one figwig file, named after
their files or by a metadata table whose other columns become the tracks'
attributes, or adds them to a file with `--append`; `figwig fw2bw` writes a
figwig file's tracks back as bigWigs with such a table; `figwig bam2fw`
takes `figwig bam2bw`'s arguments and writes each input file as its own pair
of stranded tracks, or a fragments file as one track per cell with
`--barcode-column`; `figwig fiberbam2fw` writes a Fiber-seq BAM as five tracks per
read (m6A and 5mC probabilities, nucleosomes, MSPs, FIRE quality) with the
read's alignment and tags as attributes; `figwig reorder` rewrites a file in the other order;
`figwig merge` copies several files' tracks into one; and `figwig info`
prints a file's tracks, attributes and statistics. The
[converters](https://figwig.readthedocs.io/en/latest/converters.html) page
has the options.

#### figwig install-skill

Installs the Claude Code skill; see [above](#claude-code-skill).

## Origin

figwig began as the bigWig reader inside
[tangermeme](https://github.com/jmschrei/tangermeme)'s `extract_loci`
(PR #107). It was checked there against pybigtools, bit for bit, on the
training and validation sets of 92 Cherimoya models and on 3,131 synthetic
cases.
