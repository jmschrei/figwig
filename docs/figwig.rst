figwig files
============

A figwig file holds the per-base values of many tracks over one genome: the
two strands of an assay, every experiment of a cell type, replicates, a
model's predictions, or the cells of a single-cell experiment. It is read
the way a bigWig is read with figwig, many windows of one width in one
call, but from any subset of its tracks at once, into an array of shape
``(windows, tracks, width)``, and the values come back as the integers they
were written as; the same read gives the nonzero bases as a sparse list, or
the tracks summed per label. Each track carries a JSON record of
attributes, such as its assay, biosample and strand, which ``select``
matches, and the file names the reference genome it is on.

A file is written in one of two orders, chosen for how it will be read, as
a sparse matrix is stored by rows or by columns. In *position order*, the
default, every track is cut on one grid of tiles of ``tile`` bases, 16,384
by default, so a window's tiles are found by division rather than a tree
search. The tiles of a *group* of tracks at one position are one *chunk*: a
chunk whose values are all zero takes no space; the others hold their
values densely, as nonzero bases per track with their offsets as varints,
or as a sparse matrix of the tracks present, whichever is smaller, in the
narrowest integer type that holds them, float32 where they are not
integers, or no values at all where every nonzero value is 1, as in a tile
of single-cell cut sites; and the chunks are compressed with libdeflate or
zlib, as bigWig blocks are. That is the order for reading windows across
many tracks. In *track order*, each track's
nonzero bases lie in blocks per chromosome, so one track over the genome
is a few slices and a window read searches each requested track's blocks.
That is the order for reading a few of many tracks. Six genome-wide count
tracks that take 971 MB as bigWigs take 226 MB as one figwig file
(:doc:`benchmarks`).

The examples on this page read ``k562.fw``, six genome-wide count tracks
converted from bigWigs with ``figwig bw2fw`` and a metadata table: the two
strands of the 5' ends of ENCODE ATAC-seq BAM ENCFF877LRY, the two strands
of PRO-cap ENCFF994CSC and ENCFF253HUA, and the unstranded cut counts of
ATAC-seq ENCFF830RWF and DNase-seq ENCFF989SAK.

Reading
-------

.. code-block:: python

    import numpy
    from figwig import FigWigReader

    fw = FigWigReader("k562.fw")
    print(fw.genome, len(fw.chrom_sizes), fw.tile)
    # hg38 195 16384
    print([track["name"] for track in fw.tracks])
    # ['atac.+', 'atac.-', 'procap.+', 'procap.-', 'atac_cuts', 'dnase']

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
    # {'name': 'atac.+', 'dtype': 'uint8', 'assay': 'ATAC-seq', 'biosample': 'K562', 'strand': '+', 'group': 0}
    print(fw.select(assay="ATAC-seq"))
    # [0 1 4]
    print(fw.select(assay="ATAC-seq", biosample="K562"))
    # [0 1]
    y = fw.read("chr1", [1_000_000], width=2114, tracks=fw.select(strand="-"))

``select`` matches every keyword against the track records and returns the
indices of the tracks that have them all, in the file's order, to pass as
``tracks``. The records hold whatever attributes the tracks were written
with, besides ``name``, ``dtype`` and ``group``. ``fw.stats`` is a
structured array with each track's ``sum``, ``nonzero`` bases, ``min`` and
``max``, which a count model normalizes by:

.. code-block:: python

    print(fw.stats["sum"][fw.select(biosample="K562", assay="ATAC-seq")])
    # [22410840. 22410840.]

Sparse reads and sums per label
-------------------------------

A read of thousands of tracks that are mostly zero, such as cells, should
not come back dense: a batch of 64 windows of 2,114 bp over 10,000 cells
is 1.4 GB of mostly zeros. ``sparse=True`` returns the nonzero bases as a
``SparseWindows`` of entries, and ``labels`` sums the tracks of each label
into one channel, which turns cells into pseudobulks in the read itself:

.. code-block:: python

    import numpy
    from figwig import FigWigReader

    fw = FigWigReader("pbmc_cells.fw")              # one track per cell
    cells = fw.select(cell_type="CD14 Mono")
    found = fw.read(chroms, starts, width=2114, tracks=cells, sparse=True)
    print(found.shape, len(found.values))
    # (64, 1543, 2114) 9631
    print(found.rows[:3], found.channels[:3], found.offsets[:3], found.values[:3])
    # [0 0 0] [4 4 9] [1207 1208 86] [1 1 2]

    labels = numpy.array([types.index(track["cell_type"]) for track in fw.tracks])
    y = fw.read(chroms, starts, width=2114, labels=labels)
    print(y.shape, y.dtype)
    # (64, 7, 2114) float32

An entry is a window's row, the track's channel, the base's offset in the
window and its value, sorted by row, channel and offset; ``shape`` is what
the dense array would have been, and ``scipy.sparse.coo_array`` or
``torch.sparse_coo_tensor`` take the arrays as they are. ``labels`` gives
one integer per track read, from 0, with -1 leaving a track out, and the
sums come back as float32 unless ``dtype`` says otherwise. A dense read
whose output would pass 8 GiB raises unless ``out`` is given, with a
message pointing here.

``read_entries(tracks, chroms)`` gives every nonzero base of some tracks
over whole chromosomes as a ``SparseEntries`` of chromosome index,
position, track index and value, which is the read a file in track order
is laid out for.

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
group should be those read together, such as the strands of one assay; a
file of cells read as pseudobulks wants every cell in one group, and a
file read a few tracks at a time wants track order. A ``write`` call
covers whole groups. ``write_figwig`` writes one call's worth to a new
file:

.. code-block:: python

    from figwig import write_figwig

    write_figwig("example.fw", chrom_sizes, ["plus", "minus"], chroms, starts, counts)

Each tile is encoded as the writer reaches it, and a tile a call fills only
part of waits for the next call, so predictions can be written a batch at a
time. Chunks are compressed on ``n_jobs`` threads, with libdeflate when the
``deflate`` package is installed, as :doc:`writing` describes for bigWigs,
and ``level=0`` stores them uncompressed, which decodes fastest. Reads
inflate chunks with libdeflate as well where it is installed, about twice
as fast as zlib, and with zlib otherwise. The file depends on the values
and on which calls wrote which groups, not on the number of threads or on
how the values were split between calls.

Writing nonzero bases, and the two orders
-----------------------------------------

.. code-block:: python

    from figwig import FigWigWriter

    tracks = [{"name": barcode, "dtype": "uint8", "cell_type": cell_type}
        for barcode, cell_type in zip(barcodes, cell_types)]
    with FigWigWriter("pbmc_cells.fw", chrom_sizes, tracks, genome="hg38",
            groups=len(tracks), level=0) as fw:
        for chrom in chrom_sizes:                    # cut sites of every cell
            fw.write_sparse(chrom, positions[chrom], cells[chrom], counts[chrom])

    with FigWigWriter("pbmc_by_cell.fw", chrom_sizes, tracks, genome="hg38",
            order="tracks") as fw:
        for chrom in chrom_sizes:
            fw.write_sparse(chrom, positions[chrom], cells[chrom], counts[chrom])

``write_sparse`` takes entries, one nonzero base each: a chromosome, a
position, a track and a value, which is the form of fragment files, of a
sparse matrix's triplets and of what ``read`` gives with ``sparse=True``.
The groups of the tracks named are written over the tiles the entries fall
in, with every other base 0, and no two entries may name one base of one
track. Entries come after a group's earlier items on a chromosome, and in
position order they seal the tile they end in, so that the windows and
nonzero bases of one tile come in one call.

``order`` fixes the layout when the writer is made, since it decides the
index and the chunks. ``'positions'`` (or ``'csc'``) is the tile layout
above; ``'tracks'`` (or ``'csr'``) stores each track's nonzero bases in
blocks of up to 65,536 per chromosome, uncompressed unless ``level`` is
given, since a window read over many tracks decodes a block of each. Both
orders take the same ``write`` and ``write_sparse``, return the same arrays
from ``read``, and differ in what is fast: ``figwig reorder`` rewrites a
file in the other order. In an uncompressed position-order file a tile
whose few tracks have values is stored as a matrix chunk, whose size grows
with the values rather than with the tracks; in a compressed file the
per-track counts compress to almost nothing and the plain sparse chunk is
used.

Converting
----------

:doc:`converters` describes the commands: ``figwig bw2fw`` writes bigWigs as
the tracks of one figwig file, with a metadata table for their names and
attributes; ``figwig fw2bw`` writes a figwig file's tracks back as bigWigs,
with that table; ``figwig bam2fw`` counts reads or fragments as ``figwig
bam2bw`` does, one pair of tracks per input file; and ``figwig reorder``
writes a figwig file in the other order. A bigWig's
values read from the figwig file exactly as ``read_bigwig`` gives them with
``missing=0.0``, and a track written back to a bigWig reads the same way,
with its zeros and a float32 track's NaN left out of the bigWig as
``BigWigWriter.write`` leaves them out.

The layout
----------

A figwig file is, in order: a 128-byte header; the chunks or blocks, as
they were written; the chromosome table; a table of each track's sum,
nonzero bases, minimum and maximum; the metadata, a JSON document with the
genome, the file's attributes, its order and every track's record; and the
index. In position order the index holds one record per group with the
kind byte and the compressed and uncompressed sizes of every tile and the
runs of tiles that lie back to back in the file; in track order, one record
per track with its blocks' chromosome, first and last position, entries,
offset and size. The header names the file's order, tile size, codec and
every table's offset, and a reader checks each table and each chunk it
decodes against the file's length and the format, so a truncated or
corrupt file raises a ``ValueError`` that says what it found.

The position-order index is a flat array per group of 9 bytes per tile,
read on a group's first use: for hg38 at a tile of 16,384 bases, 1.7 MB per
group on disk, and about 17 bytes per tile in memory once the chunk offsets
are laid out. A file of hundreds of tracks in groups of one, read in a few
groups at a time, keeps only those groups' indexes in memory; read all at
once, it wants larger groups, a tile of 65,536, or track order.

Reserved fields hold a place for zoom levels, which the format does not
write yet.
