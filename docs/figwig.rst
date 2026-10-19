figwig files
============

A figwig file holds the per-base values of many tracks over one genome: the
two strands of an assay, every experiment of a cell type, replicates, or a
model's predictions. It is read the way a bigWig is read with figwig, many
windows of one width in one call, but from any subset of its tracks at
once, into an array of shape ``(windows, tracks, width)``, and the values
come back as the integers they were written as. Each track carries a JSON
record of attributes, such as its assay, biosample and strand, which
``select`` matches, and the file names the reference genome it is on.

The layout is built for that read. Every track is cut on one grid of tiles
of ``tile`` bases, 16,384 by default, so a window's tiles are found by
division rather than a tree search. The tiles of a *group* of tracks at
one position are one *chunk*: a chunk whose values are all zero takes no
space; the others hold their values densely or as nonzero bases, whichever
is smaller, in the narrowest integer type that holds them, or float32 where
they are not integers; and the chunks are compressed with zlib or
libdeflate, as bigWig blocks are. Six genome-wide count tracks that take
971 MB as bigWigs take 266 MB as one figwig file (:doc:`benchmarks`).

Reading
-------

.. code-block:: python

    import numpy
    from figwig import FigWigReader

    fw = FigWigReader("k562.fw")
    print(fw.genome, len(fw.chrom_sizes), fw.tile)
    # hg38 25 16384
    print([track["name"] for track in fw.tracks])
    # ['atac.+', 'atac.-', 'procap.+', 'procap.-', 'dnase', 'cage']

    chroms = numpy.array(["chr1", "chr1", "chr2"])
    starts = numpy.array([1_000_000, 2_500_000, 300_000])
    y = fw.read(chroms, starts, width=2114)
    print(y.shape, y.dtype)
    # (3, 6, 2114) uint16

    y = fw.read(chroms, starts, width=2114, tracks=["atac.+", "atac.-"])
    print(y.shape, y.dtype)
    # (3, 2, 2114) uint16

    y = fw.read(chroms, starts, width=2114, tracks="dnase")
    print(y.shape, y.dtype)
    # (3, 2114) uint16

Coordinates are 0-based and half-open, as in BED files, and the windows may
come in any order and overlap, as with :class:`~figwig.BigWigReader`.
``tracks`` is a list of names or indices, in the order the output's
channels take, or one name or index for an output of shape ``(n, width)``.
Every chromosome must be one of the file's: a figwig file holds every
chromosome it was written with, so an unknown name is an error rather than a
warning. A base that was never written is 0, and a base past the end of its
chromosome is NaN in a float output and 0 in an integer one.

The output's dtype is the tracks' own when they share one, float32 when any
of them is float32, and otherwise the smallest integer type that holds all
of them; ``dtype=`` asks for another, and raises for an integer type that
cannot hold a track's values. ``out=`` fills an array rather than
allocating one, such as a numpy view of a pinned torch tensor:

.. code-block:: python

    import numpy
    import torch
    from figwig import FigWigReader

    fw = FigWigReader("k562.fw")
    buffer = torch.empty((64, 6, 2114), dtype=torch.int32, pin_memory=True)
    chroms = numpy.full(64, "chr1")
    starts = numpy.arange(1_000_000, 1_064_000, 1000)
    fw.read(chroms, starts, width=2114, out=buffer.numpy(), dtype="int32")
    y = buffer.to("cuda", non_blocking=True).float()

A ``FigWigReader`` reads the file's tables and metadata when it is opened,
and each group's index on the first read that touches it, after which the
index is kept. It can be read from several threads at once and pickled, so
it can go to DataLoader workers under any start method, as
:doc:`reading` shows for bigWigs. ``read_figwig`` reads from a path or a
reader in one call, as ``read_bigwig`` does.

Selecting tracks by their attributes
------------------------------------

.. code-block:: python

    from figwig import FigWigReader

    fw = FigWigReader("k562.fw")
    print(fw.tracks[0])
    # {'name': 'atac.+', 'dtype': 'uint16', 'group': 0, 'assay': 'ATAC-seq', 'strand': '+'}
    print(fw.select(assay="ATAC-seq"))
    # [0 1]
    print(fw.select(strand="-"))
    # [1 3]
    y = fw.read("chr1", [1_000_000], width=2114, tracks=fw.select(assay="ATAC-seq"))

``select`` matches every keyword against the track records and returns the
indices of the tracks that have them all, in the file's order, to pass as
``tracks``. The records hold whatever attributes the tracks were written
with, besides ``name``, ``dtype`` and ``group``. ``fw.stats`` is a
structured array with each track's ``sum``, ``nonzero`` bases, ``min`` and
``max``, which a count model normalizes by:

.. code-block:: python

    print(fw.stats["sum"][fw.select(assay="ATAC-seq")])
    # [15859293. 15860363.]

Writing
-------

.. code-block:: python

    import numpy
    from figwig import FigWigWriter
    from figwig import read_figwig

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

    y = read_figwig("example.fw", chroms, starts, width=2114)
    print(y.dtype, numpy.array_equal(y, counts))
    # uint8 True

``write`` takes windows as ``read`` gives them, ``(n, tracks, width)``, or
intervals with ``ends``, ``(n, tracks)``, for whole groups of tracks at a
time. Items may come in any order within a call but must not overlap, and
within each group a call's items come after earlier calls' on the same
chromosome; a group may move to another chromosome in any order but cannot
return to one it has left. A window may run past the end of its chromosome
only where its values are NaN or 0. A base that is not written is 0.

``tracks`` are names, or dicts with a ``name``, an optional ``dtype`` and
any JSON-serializable attributes. A track's ``dtype`` is what ``read``
returns it as: given, it is checked against every value written; left out,
it is set when the writer closes to the narrowest type that holds the
track's values, uint8 here. Either way each chunk is stored as narrowly as
its own values allow, so declaring ``uint32`` costs nothing on disk.
``genome`` is a name or a URL, and ``attributes`` a dict for the file as a
whole.

``groups`` says which tracks' tiles are stored together: a number of
consecutive tracks, 8 by default, or a list of groups of names. A read of
any track in a group decodes the whole group's chunk, so the tracks of a
group should be those read together, such as the strands of one assay, and
a file for hundreds of tracks read a few at a time wants small groups. A
``write`` call covers whole groups; a track that is read alone belongs in a
group of its own. ``write_figwig`` writes one call's worth to a new file:

.. code-block:: python

    from figwig import write_figwig

    write_figwig("example.fw", chrom_sizes, ["plus", "minus"], chroms, starts, counts)

Each tile is encoded as the writer reaches it, and a tile a call fills only
part of waits for the next call, so predictions can be written a batch at a
time. Chunks are compressed on ``n_jobs`` threads, with libdeflate when the
``deflate`` package is installed, as :doc:`writing` describes for bigWigs,
and ``level=0`` stores them uncompressed, which decodes fastest. The file
depends on the values and on which calls wrote which groups, not on the
number of threads or on how the values were split between calls.

Converting
----------

:doc:`converters` describes the commands: ``figwig bw2fw`` writes bigWigs as
the tracks of one figwig file, with a metadata table for their names and
attributes; ``figwig fw2bw`` writes a figwig file's tracks back as bigWigs,
with that table; and ``figwig bam2fw`` counts reads or fragments as
``figwig bam2bw`` does, one pair of tracks per input file. A bigWig's
values read from the figwig file exactly as ``read_bigwig`` gives them with
``missing=0.0``, and a track written back to a bigWig reads the same way,
with its zeros and a float32 track's NaN left out of the bigWig as
``BigWigWriter.write`` leaves them out.

The layout
----------

A figwig file is, in order: a 128-byte header; the chunks, as they were
written; the chromosome table; a table of each track's sum, nonzero bases,
minimum and maximum; the metadata, a JSON document with the genome, the
file's attributes and every track's record; and the index, one record per
group holding the kind byte and compressed size of every tile and the runs
of tiles that lie back to back in the file. The header names the file's
tile size, codec and every table's offset, and a reader checks each table
and each chunk it decodes against the file's length and the format, so a
truncated or corrupt file raises a ``ValueError`` that says what it found.

The index is a flat array per group of 5 bytes per tile, read on a group's
first use: for hg38 at a tile of 16,384 bases, 0.95 MB per group on disk,
and about 13 bytes per tile in memory once the chunk offsets are laid out.
A file of hundreds of tracks in groups of one, read in a few groups at a
time, keeps only those groups' indexes in memory; read all at once, it
wants larger groups or a tile of 65,536.

Reserved fields hold a place for zoom levels, which the format does not
write yet.
