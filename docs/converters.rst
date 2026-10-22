figwig bw2fw, fw2bw, bam2fw, reorder, merge and info
====================================================

Five commands move tracks between bigWigs, reads and :doc:`figwig files
<figwig>`, between a figwig file's two orders, and between figwig files,
and ``figwig info`` prints what a file holds. Each converter but ``merge``
takes ``-p``, the number of threads to read and compress on, and all take
``-v``, which prints a line per group of tracks and a summary of the
output; ``bw2fw`` and ``fw2bw`` refuse to write over an output that
exists unless ``-f`` is given.

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
it), ``--dtype`` one dtype for every track, and ``-l`` the compression
level, with 0 for uncompressed chunks, which costs four bytes per base of
a dense float track. Each piece of a chromosome is read on one thread of
its own while the piece before it is encoded and compressed on ``-p``
threads, and the arrays read are handed to the writer without a copy. The
output is written beside its path and moved there when it is complete, so
a failed conversion leaves what the path held, and an output that exists
is refused unless ``-f`` says to write over it. ``--append`` adds the
bigWigs as tracks to the figwig file at the output, in place, in new
groups after its tracks; its chromosomes, tile and level are kept, and
``-s``, ``-t`` and ``-l`` are refused with it; a chromosome of a bigWig
that the file does not have is left out with a warning. The defaults
suit count tracks; for dense float tracks read in short windows, such as
conservation or fold-change signal, ``-t 1024 --groups 1`` reads seven
times faster for a file half again as large (:doc:`benchmarks`).

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
methylation track against its unsequenced sites, or the unaligned bases of
a conservation track; the track is then float32, which costs a compressed
file little (0.4% on nine chr22 annotation tracks, nothing measurable on
a genome-wide fold-change track) and doubles an uncompressed one. A value
is equal as a number: -0.0 reads as 0. An empty cell of the table is no
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
``KEY=VALUE`` pairs that their attributes must have. ``-z`` writes zoom
levels into the bigWigs, for a genome browser, each level adding to the
file, and ``-l`` sets their compression level; a bigWig that exists in
the directory is refused unless ``-f`` is given. A track's bases that
are 0, and a float track's NaN,
are left out of its bigWig, as ``BigWigWriter.write`` leaves them out with
``missing=0.0``, so reading the bigWig with ``read_bigwig`` gives the same
numbers as reading the track, with NaN as 0; ``--missing nan`` leaves the
NaN bases out and writes the zeros, for a file ``bw2fw --missing nan``
made.

figwig bam2fw
-------------

.. code-block:: bash

    figwig bam2fw -s hg38.chrom.sizes -n k562 -p 8 atac.bam dnase.bam
    figwig bam2fw -s hg38.chrom.sizes -n k562 -m files.tsv -g hg38 -p 8 atac.bam dnase.bam
    figwig bam2fw -s hg38.chrom.sizes -n k562 -f -u -p 8 fragments.tsv.gz

``figwig bam2fw`` takes what :doc:`figwig bam2bw <bam2bw>` takes, SAM/BAM
files of reads or BED/tsv files of fragments and the same options for what
to count, and reads them the same way, on ``-p`` cores. Where ``figwig
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
needs the ``bam2bw`` extra and runs on Linux and macOS.

.. code-block:: text

    usage: figwig bam2fw [-h] -s SIZES [-u] [-f | -3p] [-ps POS_SHIFT]
                         [-ns NEG_SHIFT] [-mp] [--rna5 {read1,read2}]
                         [--opposite_strand] [-sf SCALE_FACTOR] [-r] [-p PARALLEL]
                         -n NAME [-m METADATA] [-g GENOME] [-t TILE]
                         [--dtype DTYPE] [-l LEVEL] [-v]
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
about four times larger, unless ``-l`` says otherwise. The output may be
the input's own path: the input is read
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
behind. The output may be one of the inputs. ``-g`` records a reference
genome; the first file's is kept otherwise.

figwig info
-----------

.. code-block:: bash

    figwig info k562.fw
    figwig info -b k562.fw

``figwig info`` prints a file's path, genome, order, tile, compression
level, chromosomes, groups, tracks, size in bytes and attributes, one per
line, then a tab-separated table of its tracks with their name, dtype,
group, nonzero bases, sum and every attribute. ``-b`` also counts the
bytes its chunks take, its tables and index take, and the bytes nothing
uses, which an append in place leaves behind and ``merge`` drops; it reads
every group's index, a few milliseconds per group.
