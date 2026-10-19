figwig bw2fw, fw2bw and bam2fw
==============================

Three commands move tracks between bigWigs, reads and :doc:`figwig files
<figwig>`. Each takes ``-p``, the number of threads to read and compress
on, and ``-v`` for progress.

figwig bw2fw
------------

.. code-block:: bash

    figwig bw2fw -o k562.fw -g hg38 -p 8 atac.+.bw atac.-.bw dnase.bw
    figwig bw2fw -o k562.fw -m tracks.tsv -p 8 atac.+.bw atac.-.bw dnase.bw

``figwig bw2fw`` writes bigWigs as the tracks of one figwig file, in the
order given, each named after its file without the ``.bw``, ``.bigWig`` or
``.bigwig`` extension. The chromosomes are every chromosome of the bigWigs,
in order of first appearance, and the bigWigs must agree on each one's
length; ``-s`` gives a chrom_sizes, ``.fai`` or FASTA file to take them
from instead, and a bigWig without one of them is zero there. ``-g``
records the reference genome, as a name or a URL. ``-t`` sets the tile
size, ``--groups`` the consecutive tracks stored together (8 by default),
``--dtype`` one dtype for every track, and ``-l`` the compression level,
with 0 for uncompressed chunks.

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

A bigWig's values read from the figwig file exactly as ``read_bigwig``
gives them with ``missing=0.0``: a base no interval covers is 0.

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
levels into the bigWigs, for a genome browser, and ``-l`` sets their
compression level. A track's bases that are 0, and a float32 track's NaN,
are left out of its bigWig, as ``BigWigWriter.write`` leaves them out with
``missing=0.0``, so reading the bigWig with ``read_bigwig`` gives the same
numbers as reading the track, with NaN as 0.

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
at compression level 1, like ``figwig bam2bw``'s, unless ``-l`` says
otherwise. A file's tracks read from the figwig file exactly as the bigWigs
``figwig bam2bw`` writes from that file alone read with ``read_bigwig``. It
needs the ``bam2bw`` extra and runs on Linux and macOS.

.. code-block:: text

    usage: figwig bam2fw [-h] -s SIZES [-u] [-f | -3p] [-ps POS_SHIFT]
                         [-ns NEG_SHIFT] [-mp] [--rna5 {read1,read2}]
                         [--opposite_strand] [-sf SCALE_FACTOR] [-r] [-p PARALLEL]
                         -n NAME [-m METADATA] [-g GENOME] [-t TILE]
                         [--dtype DTYPE] [-l LEVEL] [-v]
                         filename [filename ...]
