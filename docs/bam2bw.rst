figwig bam2bw
=============

``figwig bam2bw`` turns SAM/BAM files of reads, or BED/tsv files of fragments,
into bigWigs of per-base counts. It is `bam2bw
<https://github.com/jmschrei/bam2bw>`_ 0.5.1 with the same arguments, the same
output files and the same messages, reading files the way the winner of a
speed search over bam2bw's code reads them, and writing them with
:class:`~figwig.BigWigWriter`. It needs the ``bam2bw`` extra, which adds
pysam, pyfaidx, biopython, tqdm, joblib, isal and deflate:

.. code-block:: bash

    pip install ".[bam2bw]"

    figwig bam2bw my.bam -s hg38.chrom.sizes -n test-run -p 2                  # test-run.+.bw, test-run.-.bw
    figwig bam2bw fragments.tsv.gz -s hg38.chrom.sizes -n test-run -f -u -p 2  # test-run.bw

By default it counts the 5' end of every mapped read at each base, and writes
the counts of the two strands to two bigWigs, ``<name>.+.bw`` and
``<name>.-.bw``. Several input files are pooled. Reads on chromosomes that
the sizes file does not list are left out, and counts that fall outside a
chromosome, after a shift, are left out and reported.

.. code-block:: text

    usage: figwig bam2bw [-h] -s SIZES [-u] [-f | -3p] [-ps POS_SHIFT]
                         [-ns NEG_SHIFT] [-mp] [--rna5 {read1,read2}]
                         [--opposite_strand] [-sf SCALE_FACTOR] [-r] [-p PARALLEL]
                         -n NAME [-z ZOOMS] [-v]
                         filename [filename ...]

    positional arguments:
      filename              The SAM/BAM or tsv/tsv.gz file to be processed.

    options:
      -h, --help            show this help message and exit
      -s, --sizes SIZES     A chrom_sizes, .fai, or FASTA file. Only the first two
                            columns of a chrom_sizes/.fai file are read. A
                            compressed FASTA must be BGZF, not gzip.
      -u, --unstranded      Have only one, unstranded, output.
      -f, --fragments       The data is fragments and so both ends should be
                            recorded.
      -3p, --three_prime    Record the 3' end of each read instead of the 5' end.
      -ps, --pos_shift POS_SHIFT
                            A shift to apply to positive strand reads.
      -ns, --neg_shift NEG_SHIFT
                            A shift to apply to negative strand reads.
      -mp, --mate_pairs     Treat paired-end reads as a single RNA/fragment tag
                            instead of counting each mate independently: buffer
                            reads by name, and once both mates of a pair are seen,
                            record one jointly-determined position (see --rna5 and
                            --opposite_strand). BAM/SAM input only.
      --rna5 {read1,read2}  Which mate carries the 5' end of the RNA/fragment. The
                            other mate's own 5' end is used as the RNA's 3' end.
                            Only used with --mate_pairs.
      --opposite_strand     Report the strand of the mate opposite the one chosen
                            by --rna5, instead of that mate's own strand. Only
                            used with --mate_pairs.
      -sf, --scale_factor SCALE_FACTOR
                            A scaling factor to multiply each position by.
      -r, --read_depth      Whether to divide through by total (pre-scaled) read
                            depth.
      -p, --parallel PARALLEL
                            The number of jobs to use, max of one per input file.
      -n, --name NAME
      -z, --zooms ZOOMS     The number of zooms to store in the bigwig.
      -v, --verbose

How it reads
------------

A BAM file's BGZF blocks are inflated by libdeflate on the ``-p`` threads, and
its records are walked by a numba kernel rather than through pysam's objects;
the kernel writes one integer key per counted position, and sorting the keys
and counting the runs of equal keys gives each chromosome's counts. A BED/tsv
file is scanned by a numba kernel, on the ``-p`` threads when it is BGZF or not
compressed. bam2bw gives each file one process, and at most one process per
file; here the threads ``-p`` leaves over go to each file.

A file these readers would not read exactly as htslib, or bam2bw's own loop
over lines, would read it is read by bam2bw's pysam or line loop instead, so
that its result, or its error, is bam2bw's: a SAM file, ``--mate_pairs``, a
remote file, or a malformed record or line.

How it writes
-------------

The counts are written by :class:`~figwig.BigWigWriter`, as single bases
(varStep sections), with libdeflate at level 1 on up to ``-p`` threads, so the
bigWigs hold bam2bw's entries but not its bytes, and are the same whatever
``-p`` is. ``-z`` writes figwig's zoom levels rather than pyBigWig's. The
entries and messages were checked against bam2bw 0.5.1's on the speed search's
own cases: a 2.4 GB ATAC-seq BAM and 18.9 million scATAC fragments, four
synthetic BAM and SAM files, nine other real invocations covering
``--mate_pairs``, ``-3p``, ``-z``, shifts, scaling, three BAMs at once and a
FASTA as the sizes, and 51 malformed BAMs, each at six settings of ``-p``,
``-v``, ``-f`` and ``-u``.
