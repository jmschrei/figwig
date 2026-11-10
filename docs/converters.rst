figwig bw2fw, fw2bw, bam2fw, fiberbam2fw, reorder, merge and info
=================================================================

Six commands move tracks between bigWigs, reads and :doc:`figwig files
<figwig>`, between a figwig file's two orders, and between figwig files,
and ``figwig info`` prints what a file holds. Each converter but ``merge``
takes ``-p``, the number of threads to read and compress on (1 by
default), and every converter takes ``-v``, which prints a line per group
of tracks and a summary of the output on stderr; ``-t`` is the tile in
``bw2fw``, ``bam2fw``, ``fiberbam2fw`` and ``reorder`` and the track list in ``fw2bw``, and
``-m`` takes a tab-separated table, or a comma-separated one when its name
ends in ``.csv``. Every converter refuses to write over an output that
exists unless ``-f`` is given (``--force`` for ``bam2fw``, whose ``-f``
means fragments, and ``fiberbam2fw``), every converter writes its output beside its path and
moves it into place when complete, so a conversion that fails or is
killed leaves no half-written output, a missing input is a message rather
than a traceback, and Ctrl-C exits with 130.

figwig bw2fw
------------

.. code-block:: bash

    figwig bw2fw -o k562.fw -g hg38 -p 8 atac.+.bw atac.-.bw dnase.bw
    figwig bw2fw -o k562.fw -m tracks.tsv -p 8 atac.+.bw atac.-.bw dnase.bw
    figwig bw2fw -o k562.fw --append -p 8 h3k27ac.bw

``figwig bw2fw`` writes bigWigs as the tracks of one figwig file, in the
order given, each named after its file without the ``.bw``, ``.bigWig`` or
``.bigwig`` extension. The chromosomes are every chromosome of the bigWigs,
in order of first appearance, and the bigWigs must agree on each one's
length; ``-s`` gives a chrom_sizes, ``.fai`` or FASTA file to take them
from instead, which must agree with the bigWigs on the length of every
chromosome they share, so that a bigWig on another assembly is refused: a
bigWig without one of the file's chromosomes is zero there, and a
chromosome of a bigWig that the file leaves out is named in a warning,
since its values are dropped. ``-g`` records the reference genome, as a
name or a URL. ``-t`` sets the tile size, ``--groups`` the consecutive
tracks stored together (8 by default; group tracks of one kind together,
since a count track grouped with a float track is stored and decoded with
it), ``--dtype`` one dtype for every track, ``-l`` the compression
level (6 by default), with 0 for uncompressed chunks, which costs four bytes per base of
a dense float track, and ``-z`` the number of zoom levels to keep beside
the tile sums (none by default; ``bam2fw`` takes it too, and ``reorder``
and ``merge`` keep the input's), and ``--summary-index`` to write the
optional regional summary index (:ref:`regional summaries <regional-summaries>`; off by default,
not with ``--append``; ``bam2fw`` takes it too). Each piece of a chromosome is read on one thread of
its own while the piece before it is encoded and compressed on ``-p``
threads, and the arrays read are handed to the writer without a copy.
``--append`` adds the
bigWigs as tracks to the figwig file at the output, in place, in new
groups after its tracks; its chromosomes, tile and level are kept, and
``-s``, ``-t`` and ``-l`` are refused with it; a chromosome of a bigWig
that the file does not have is left out with a warning. The defaults
suit count tracks, which read fastest in one group of many; for dense
float tracks read in short windows, such as conservation or fold-change
signal, ``-t 1024 --groups 1`` reads short windows 1.4 to 1.7 times faster
on 1 to 32 threads, for a file 29% larger and a whole chromosome read more
slowly on 1 and 8 threads (:doc:`benchmarks`).

``-m`` takes a metadata table: tab-separated, or comma-separated with a
``.csv`` extension, with a header and one row per bigWig, matched on its
first column by the path given on the command line or its base name. A
``name`` column names the track, a ``dtype`` column sets its dtype, and
every other column becomes an attribute of the track, kept as the string it
was:

.. code-block:: text

    file        name      assay     biosample  strand
    atac.+.bw   atac.+    ATAC-seq  K562       +
    atac.-.bw   atac.-    ATAC-seq  K562       -
    dnase.bw    dnase     DNase-seq K562

A bigWig's values read from the figwig file as ``read_bigwig`` gives them
with ``missing=0.0`` and ``keep_nan=True``: a base no interval covers is
0, and an interval whose value is NaN, as the gaps of a Hi-C insulation or
compartment track are, is NaN, which makes the track float32. ``--missing
nan`` writes the bases no interval covers as NaN instead, for tracks whose
zeros mean something other than their gaps, such as the 0% sites of a
methylation track against its unsequenced sites, the unaligned bases of
a conservation track, or the explicit 0.0 intervals of a fold-change or
p-value track, which the default drops (``bw2fw`` then ``fw2bw`` with
``--missing nan`` on both is lossless, except that an interval whose value is
NaN comes back as a gap); the track is then float32, which
costs a compressed
file little (0.4% on nine chr22 annotation tracks, nothing measurable on
a genome-wide fold-change track) and doubles an uncompressed one. The
convention is recorded as the file's ``missing`` attribute (``'0.0'``,
``'nan'`` or the value given), which ``fw2bw`` follows by default. A value
is equal as a number: -0.0 reads as 0. A bigWig holding an infinite value,
as some ENCODE signal p-value tracks do, is refused with a message, since
no figwig file holds one; clip it first. An empty cell of the table is no
attribute.

figwig fw2bw
------------

.. code-block:: bash

    figwig fw2bw k562.fw -o bigwigs -p 8
    figwig fw2bw k562.fw -o bigwigs --select assay=ATAC-seq
    figwig fw2bw k562.fw -o bigwigs -t atac.+ atac.-

``figwig fw2bw`` writes a figwig file's tracks as bigWigs, ``<track
name>.bw`` in the directory ``-o`` names, with a metadata table beside them
named after the figwig file, ``k562.tsv`` here, whose columns are ``file``,
``name``, ``dtype`` and every attribute the tracks have. ``figwig bw2fw -m``
reads that table, so the two commands round-trip a file's tracks, names and
attributes. ``-t`` names the tracks to write and ``--select`` takes
``KEY=VALUE`` pairs that their attributes must have. ``-z N`` writes N
zoom levels into the bigWigs, for a genome browser, each level adding to
the file (none by default), and ``-l`` sets the bigWigs' compression level (6
by default). A
track's bases that are 0, and a float track's NaN, are left out of its
bigWig, as ``BigWigWriter.write`` leaves them out with
``missing=0.0``, so reading the bigWig with ``read_bigwig`` gives the same
numbers as reading the track, with NaN as 0; ``--missing nan`` leaves the
NaN bases out and writes the zeros, for a file ``bw2fw --missing nan``
made. Without ``--missing``, ``fw2bw`` follows the file's ``missing``
attribute when ``bw2fw`` recorded one, so a round trip through the two
needs the flag on neither. Two tracks whose names make the same file name
(``a/b`` and ``a_b``) get one file each, the second with its track index
appended (``a_b.1.bw``), and the table says which file is whose.

figwig bam2fw
-------------

.. code-block:: bash

    figwig bam2fw -s hg38.chrom.sizes -n k562 -p 8 atac.bam dnase.bam
    figwig bam2fw -s hg38.chrom.sizes -n k562 -m files.tsv -g hg38 -p 8 atac.bam dnase.bam
    figwig bam2fw -s hg38.chrom.sizes -n k562 -f -u -p 8 fragments.tsv.gz
    figwig bam2fw -s hg38.chrom.sizes -n pbmc -f -p 8 --barcode-column 4 --cells cells.tsv fragments.tsv.gz

``figwig bam2fw`` takes what :doc:`figwig bam2bw <bam2bw>` takes, SAM/BAM
files of reads or BED/tsv files of fragments and the same options for what
to count, and reads them the same way, on ``-p`` cores. Each BAM's header is checked against ``-s`` first: a chromosome both
name must have one length, and the BAM must share a chromosome with the
sizes, or the file is refused as another assembly's or another naming's
(``figwig bam2bw`` warns and goes on, as bam2bw does). Where ``figwig
bam2bw`` pools every file into one pair of bigWigs, ``figwig bam2fw``
writes each file as its own pair of tracks, ``<name>.+`` and ``<name>.-``,
in one group, or one track with ``-u``, into ``<-n>.fw``. ``<name>`` is the
file's name without its extension unless the metadata table's ``name``
column gives another; the table's other columns become attributes of the
file's tracks, and stranded tracks carry a ``strand`` attribute. ``-r``
divides each file's counts by that file's read depth. Counts are written
as entries with ``write_sparse``, which encodes them on the ``-p``
threads, at compression level 1, like ``figwig bam2bw``'s, unless ``-l``
says otherwise. A file's tracks read from the figwig file exactly as the bigWigs
``figwig bam2bw`` writes from that file alone read with ``read_bigwig``. It
needs the ``bam2bw`` extra and runs on Linux and macOS. With ``-c`` each
file's tracks hold the coverage of its reads' aligned bases, as
:doc:`figwig bam2bw -c <bam2bw>` counts it, written as runs, each track its
own group, since a stretch of one value costs a runs chunk a few bytes
where entries would cost one per base; ``-c`` is not taken with
``--barcode-column``. Sorted, indexed BAM files of 128 MiB and 50,000
mapped reads or more are read a region at a time on every core when ``-p``
is 8 or more, as :doc:`bam2bw` describes, into tracks of the same values.

``--barcode-column N`` writes a fragments file as one track per cell
instead: column ``N`` (from 1; 4 in a Cell Ranger fragments file) holds
the barcode, each barcode becomes a track named by it, in order of first
appearance, every cell of a file goes in one group, or in as few groups as
keep each chunk under 2**29 values (32,767 cells at the default tile;
``--groups`` for smaller ones), and there is one track per cell rather
than a pair, since fragments have no strand. ``--cells`` gives a table, tab-separated or
``.csv`` with a header, whose first column lists the barcodes to keep, in
the order their tracks take: a ``name`` column names a cell's track,
``dtype`` sets its dtype, every other column, such as a cell type, becomes
an attribute of it, and fragments of barcodes not in the table are left
out (``-v`` counts them). A Cell Ranger fragments file holds the
fragments of every barcode, hundreds of thousands of which are not cells,
so ``--cells`` with the called cells is the usual call. The file is read
in BGZF units on the ``-p`` threads, as ``figwig bam2bw`` reads it, and
each cell's fragment ends (``-f``, the usual choice; ``-ps`` and ``-ns``
shift them) go in as entries, holding 8 bytes per end while the file is
scanned and four to six times that at the peak. With several files each file's cells are its own group, named
``<name>.<barcode>``, with the file's attributes from ``-m`` and the
cell's from ``--cells``. :doc:`benchmarks` times the conversion against
the other ways of making a per-cell file.

.. code-block:: text

    usage: figwig bam2fw [-h] -s SIZES [-u] [-f | -3p | -c] [-ps POS_SHIFT]
                         [-ns NEG_SHIFT] [-mp] [--rna5 {read1,read2}]
                         [--opposite_strand] [-sf SCALE_FACTOR] [-r] [-p PARALLEL]
                         -n NAME [-m METADATA] [-g GENOME]
                         [--barcode-column BARCODE_COLUMN] [--cells CELLS]
                         [--groups GROUPS] [-t TILE] [--dtype DTYPE] [-z ZOOMS]
                         [-l LEVEL] [--summary-index] [--force] [-v]
                         filename [filename ...]

figwig fiberbam2fw
------------------

.. code-block:: bash

    figwig fiberbam2fw -n fibers -g hg38 -p 4 fibers.bam
    figwig fiberbam2fw -n fibers -m files.tsv -p 4 --threshold 125 a.bam b.bam
    figwig fiberbam2fw -n fibers -c m6a,nuc --primary-only fibers.bam

``figwig fiberbam2fw`` writes the reads of a coordinate-sorted Fiber-seq
BAM as five tracks each, on reference coordinates, into ``<-n>.fw``. A
Fiber-seq read is one chromatin fiber, and its signal is per read and per
base. The channels are

- ``m6a``: the probability (the ML tag, 0 to 255) that each A of the
  molecule, and each T whose opposite A is methylated, carries an m6A, from
  the ``A+a`` and ``T-a`` blocks of MM;
- ``cpg``: the probability of 5mC at each CpG, from ``C+m``;
- ``nuc``: 1 over each nucleosome of the ``ns`` and ``nl`` tags;
- ``msp``: 1 over each methyltransferase-sensitive patch, the stretches
  between nucleosomes, of ``as`` and ``al``;
- ``fire``: the FIRE quality ``aq`` over each MSP that has one above 0.

Tracks are named ``<read>.<channel>`` and carry the attributes ``read``
(the track's alignment: the read's name, with ``@<chrom>:<start>`` added to
a second alignment of the same read, such as a supplementary one),
``read_name``, ``channel``, ``chrom``, ``start`` and ``end`` (the reference
span, base 0, end exclusive), ``strand``, ``mapq``, ``flag``, ``length``
(the molecule's), the tags named by ``--tags`` (``HP``, ``RG``, ``rq`` and
``ec`` by default) and the ``sample`` of the read group's ``SM``, and the
per-channel counts: ``m6a_calls`` and ``cpg_calls`` (every call in MM,
including those of probability 0), ``m6a_unplaced`` and ``cpg_unplaced``
(calls on a base with no reference base, an insertion or a soft clip),
``at_bases`` (A and T in the molecule, the denominator of an m6A rate),
``nuc_calls``, ``msp_calls`` and ``fire_calls`` (intervals), and
``n_values`` (nonzero bases written). The file's own attributes record the
files, channels, threshold, filters and how many alignments each filter left
out. A table given to ``-m``, as for ``bam2fw``, puts a ``name`` prefix
before a file's tracks and its other columns on them. Several files go one
after another into one file and must agree on their chromosomes, which are
those of the header; with several files, each file's tracks are prefixed
by its name.

Every track is ``uint8``. The tracks of a channel over 1,024 consecutive
reads (``--group-size``) are one group, in position order, the layout
:ref:`reads, cells and channels as tracks <reads-cells-channels>` describes,
so ``read(..., channels="channel", items="read")`` gives (windows, reads,
channels, width) and ``sums(..., paired=True)`` a total per read over its own
span. A call of probability 0 is not written, which a base without a call
shares, and ``m6a_calls`` and ``n_values`` tell them apart.
``--threshold T`` writes 1 for each call of probability at least ``T``
instead (fibertools calls m6A at 125) and leaves the others out. ``-c`` picks the
channels, ``--primary-only`` leaves out supplementary alignments, which carry
the whole read's MM and are placed by their own alignment, so that a read
split across two loci has a track on each, ``--min-mapq`` the alignments
below a quality; secondary alignments, unmapped ones and those with none of
the tags the channels need are always left out. The tags follow
fibertools: ``MM`` and ``ML``, and ``ns``, ``nl``, ``as``, ``al`` and
``aq`` in molecule coordinates, or an ``MA`` tag with ``AQ``, whose starts
count from 1.

MM counts bases along the original molecule, which for a reverse-strand
alignment is the reverse complement of SEQ, and ``ns`` and ``as`` are
offsets along that molecule too; both are mapped to the reference through
the alignment, a footprint covering the reference from its first to its last
base that has one. The file's values equal ``ft extract --all``'s for every
call it lists and every footprint, and htslib's reading of MM and ML at
every probability (see the checks in :doc:`benchmarks`). An alignment whose MM names
more bases than the read holds, whose ML is shorter than its MM, whose MM is
malformed, that carries 6mA on a base and strand other than ``A+`` or
``T-``, or that has no sequence or is hard-clipped (so that the bases MM
counts are not there), has the calls it cannot place left out, with one
warning that names the first such read; its footprints are still written,
a hard clip counted into the molecule's length. A file that is not coordinate sorted, or has a read past the
end of its header's chromosome, is refused. Blocks of reads are inflated
and parsed on up to 8 of the ``-p`` threads, and the file does not depend
on ``-p``. It needs the ``bam2bw`` extra and runs on Linux and macOS.
:doc:`benchmarks` gives the time, memory and size.

.. code-block:: text

    usage: figwig fiberbam2fw [-h] -n NAME [-p PARALLEL] [-m METADATA] [-g GENOME]
                              [-c CHANNELS] [--threshold THRESHOLD] [--tags TAGS]
                              [--primary-only] [--min-mapq MIN_MAPQ]
                              [--group-size GROUP_SIZE] [-t TILE] [-z ZOOMS]
                              [-l LEVEL] [--force] [-v]
                              filename [filename ...]

figwig reorder
--------------

.. code-block:: bash

    figwig reorder k562.fw k562_by_track.fw --order tracks -p 8
    figwig reorder pbmc_by_cell.fw pbmc.fw --order positions --groups 10246 -l 0 -p 8

``figwig reorder`` reads every nonzero base of a figwig file, chromosome by
chromosome, and writes it again in the order asked for, with the same
tracks, attributes and genome: ``tracks`` (or ``csr``) for reading a few
tracks over the genome, ``positions`` (or ``csc``) for reading windows
across many tracks. In position order ``-t`` sets the tile size and
``--groups`` the consecutive tracks per group, which default to the
input's when it is in position order and to 16,384 and 8 otherwise; ``-l``
sets the compression level, which defaults to the input's when the output
is in the input's order, and otherwise to 6 in position order and 0 in
track order, so a file reordered to track order is uncompressed, and
1.6 to 4.7 times larger, unless ``-l`` says otherwise. The output has a
regional summary index when the header of the input announces one, unless
``--no-summary-index`` says otherwise, and ``--summary-index`` writes one for
an input that has none. The output may
be the input's own path, and no other existing file unless ``-f`` is
given: the input is read
whole before the output takes its place. The values are read in the
narrowest type that holds every track exactly, so a uint32 track beside a
float track keeps its integers.

figwig merge
------------

.. code-block:: bash

    figwig merge -o all.fw k562.fw hepg2.fw gm12878.fw

``figwig merge`` writes the tracks of several figwig files into one, in
the order given, each file's groups after the one before's, with their
attributes, dtypes, statistics and summaries. The files must have been
written with the same chromosomes, tile, compression level and order, and
their track names must differ; ``figwig reorder`` writes a file again with
another tile, level or order first. The chunks are copied as they are, so
a merge costs a read and a write of the bytes and no decoding, and
leaves out the tables and indexes a file appended to in place left
behind. The output has a regional summary index when the header of the first
file announces one, unless ``--no-summary-index`` says otherwise, and
``--summary-index`` writes one when it does not; it is written anew over the
merged tracks, not copied.
The output may be one of the inputs, and no other existing file
unless ``-f`` is given. ``-g`` records a reference genome; without it the
merged file takes the genome the files name, and files that name different
genomes are refused. The first file's attributes are kept.

figwig info
-----------

.. code-block:: bash

    figwig info k562.fw
    figwig info -b k562.fw

``figwig info`` prints a file's path, format version, the figwig that
wrote it, zoom levels, genome, order, tile, compression level, chromosomes, groups,
tracks, size in bytes and attributes, one per line, then a tab-separated
table of its tracks with their name, dtype, group, nonzero bases (NaN
bases are not counted), sum and every attribute. A control character in a
name or attribute is escaped and a nested attribute is printed as JSON, so
every value takes one line. ``-b`` also counts the
bytes its chunks take, its tables and index take, those of its regional summary
index, the bytes nothing uses, which an append in place leaves behind (an
index it could not use, stale or damaged, among them) and ``merge`` drops, and any bytes
past the end the header names, which a killed append leaves and the next
append writes over; it reads every group's index, a few milliseconds per
group. A file of thousands of tracks prints a line per track.
