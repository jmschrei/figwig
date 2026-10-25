figwig files
============

A figwig file holds the per-base values of many tracks over one genome: the
two strands of an assay, every experiment of a cell type, replicates, a
model's predictions, or the cells of a single-cell experiment. It is read
the way a bigWig is read with figwig, many windows of one width in one
call, but from any subset of its tracks at once, into an array of shape
``(windows, tracks, width)``, and the values come back as the integers they
were written as; the same read gives the nonzero bases as a sparse list,
the tracks summed per label, or each window summed per bin. Each track
carries a JSON record of attributes, such as its assay, biosample and
strand, which ``select`` matches, and the file names the reference genome
it is on.

A file is written in one of two orders, chosen for how it will be read, as
a sparse matrix is stored by rows or by columns. In *position order*, the
default, every track is cut on one grid of tiles of ``tile`` bases, 16,384
by default, so a window's tiles are found by division rather than a tree
search. The tiles of a *group* of tracks at one position are one *chunk*: a
chunk whose values are all zero takes no space; the others hold their
values densely, as nonzero bases per track with their offsets as varints,
as runs of one value, or as a sparse matrix of the tracks present,
whichever is smallest, in the narrowest integer type that holds them,
float32 where they are not integers, float64 where float32 would change an
integer that must be kept, or no values at all where every nonzero value
is 1, as in a tile of single-cell cut sites; and the chunks are compressed
with libdeflate or zlib, as bigWig blocks are, or stored as they are when
compression would grow them. That is the order for reading windows across
many tracks.

In *track order*, each track's nonzero bases lie in blocks per
chromosome, so one track over the genome is a few slices and a window read
searches each requested track's blocks. That is the order for reading a
few of many tracks. Six genome-wide count tracks that take 971 MB as
bigWigs take 227 MB as one figwig file (:doc:`benchmarks`).

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

The output's dtype is the tracks' own when they share one, and otherwise
the narrowest type that holds every one of them exactly: float32 for a
uint8 or int16 track read with a float32 one, float64 for a uint32 or int32
track read with a float32 one. ``dtype=`` asks for another, and raises for
an integer type that cannot hold a track's values. ``out=`` fills an array
rather than allocating one, such as a numpy view of a pinned torch tensor:

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

The output, not the file, is the large object on the way to a GPU: 8,192
windows of 2,048 bases over 48 tracks are 3.2 GB as float32, which a copy
from ordinary memory moves at 5 to 10 GB/s while the read itself takes
under 0.1 s on 8 threads. A read into pinned memory skips the copy into
a staging buffer, the transfer then runs at the bus's rate (20 to 25 GB/s
on PCIe 4 or 5) and overlaps with the next read, and reading as ``uint8``
or ``uint16`` where the tracks allow it moves a quarter or half the
bytes, cast to float on the device. A ``DataLoader`` with
``pin_memory=True`` makes one more copy from the worker's array; passing
the pinned buffer as ``out`` is the shorter path.

A ``FigWigReader`` reads the file's tables and metadata when it is opened,
and each group's index on the first read that touches it, after which the
index is kept, 17 bytes per tile of the genome per group. A read uses one
thread per 64 chunks it touches, up to ``n_jobs``, and keeps its threads
between reads, so a read of 64 windows runs on the calling thread and a
read of 8,192 on 8. A reader checks on every read that the file at its
path is the one it opened, by its inode and header, and raises if a
conversion has replaced it; it can be read from several threads at once
and pickled, so it can go to DataLoader workers under any start method,
as :doc:`reading` shows for bigWigs, and a pickled reader carries no
index, which each worker reads again. ``read_figwig`` reads from a path
or a reader in one call, as ``read_bigwig`` does. The first read in a new
environment compiles the kernels for that kind of read, about 27 s for a
dense read; numba caches them on disk, and later processes load them in a
tenth of a second (:doc:`threads` lists every kind of first call).

Reading onto a GPU
^^^^^^^^^^^^^^^^^^

``device="cuda"`` (or ``"cuda:1"``, or a device index) lands a read's
output in that GPU's memory, as a ``cupy.ndarray``, and needs the
``cuda`` extra (``pip install "figwig[cuda]"``, which is ``cupy-cuda12x``;
install ``cupy-cuda13x`` or ``cupy-cuda11x`` for CUDA 13 or 11). Every
framework takes the array without a copy through DLPack:

.. code-block:: python

    y = fw.read(chroms, starts, 2048, dtype="float32", device="cuda")
    torch.from_dlpack(y)                                 # PyTorch
    jax.dlpack.from_dlpack(y)                            # JAX
    tf.experimental.dlpack.from_dlpack(y.__dlpack__())   # TensorFlow
    y                                                    # CuPy, as it is

The read itself runs on the CPU. Tracks whose nonzero bases are under
0.2 percent of the genome by the file's statistics, as single cells are,
are listed as entries, moved and scattered into a zeroed device array,
so the bus carries the entries rather than the dense windows (128 of
10,246 cells over 8,192 windows: 0.05 s against 0.25 s through a pinned
copy on an H200); denser tracks, such as bulk counts at 1 percent, are
decoded into a pinned host buffer, kept on the reader for the next read
of the same shape, and copied, which is the faster route for them. Sums per
label, per bin and over intervals go through the pinned buffer, and
``sparse=True`` moves the entries themselves. The device's stream is
synchronized before the array is returned, so it is complete whatever
stream the framework then uses; ``out`` may be a cupy array of the
output's shape and dtype. ``read_figwig`` and ``sums`` take ``device``
too. A reader with pinned buffers pickles without them, so it can go to
``DataLoader`` workers as before.

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
``tracks``; it returns an empty array, which ``read`` refuses, when no
track matches, as when a value is given as a number where the table it was
read from held a string, and raises when no track has the key at all,
naming the keys there are. The records hold whatever attributes the tracks
were written with, besides ``name``, ``dtype`` and ``group``; ``figwig
info k562.fw`` prints them as a table, with the file's genome, order,
tile, chromosomes and size. ``fw.stats``
is a structured array with each track's ``sum``, ``nonzero`` bases, ``min``
and ``max``, which a count model normalizes by:

.. code-block:: python

    print(fw.stats["sum"][fw.select(biosample="K562", assay="ATAC-seq")])
    # [22410840. 22410840.]

Sparse reads, sums per label and sums per bin
---------------------------------------------

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
the dense array would have been, and ``scipy.sparse.coo_array((values,
(rows * width + offsets, channels)), shape=(n * width, channels))`` or
``torch.sparse_coo_tensor`` take the arrays as they are. A sparse read
takes about 40 bytes of memory per entry while it sorts them, so it is for
tracks that are mostly zero, not for dense ones, whose dense read is
smaller. ``labels`` gives one integer per track read, from 0, with -1
leaving a track out, and the sums come back as float64 where a track is
an integer type, since counts sum past float32's 2**24 in a few windows,
and as float32 for float32 tracks, unless ``dtype`` says otherwise; a sum asked
for in an integer type is taken in int64 and checked against that type,
since a sum can overflow a type that each track fits. A dense read whose
output would pass 8 GiB raises unless ``out`` is given, with a message
pointing here.

``bins`` sums each window over bins of that many bases, which is the
target of a model of binned coverage, and ``bins=width`` gives each
window's total per track, which a count loss takes:

.. code-block:: python

    y = fw.read(chroms, starts, width=196_608, bins=128)
    print(y.shape, y.dtype)
    # (64, 6, 1536) float32
    counts = fw.read(chroms, starts, width=2114, bins=2114)[:, :, 0]

``bins`` is a width in bases, where pyBigWig's ``nBins`` is a count, and
must divide ``width``; the sums are taken without laying the
window's bases out, from a scratch tile each chunk is decoded into once,
and bases past the end of a chromosome count as 0. A bin that straddles
two tiles is summed from both, so a binned read's batches of chunks are
not cut inside a window, and the threads sum into the one output; a
window longer than a batch may grow, such as a whole chromosome, is cut,
and the read then sums into a copy of its output per thread, which the
memory guard counts. The output is ``width / bins`` times smaller than a
dense one. ``bins`` and ``labels`` can be given together.

``sums`` takes intervals that differ in length, such as peaks, genes or
cCREs, and gives each track's total over each, the way ``bins=width``
does for windows of one width, from the tiles the intervals overlap:

.. code-block:: python

    totals = fw.sums(chroms, starts, ends, tracks=["atac.+", "atac.-"])
    print(totals.shape, totals.dtype)
    # (172732, 2) float64
    means = totals / (ends - starts)[:, None]

These sums count every base of an interval, so a mean is a sum over the
interval's length. pyBigWig's ``stats(type="mean")``, pybigtools and
deepTools average over the bases an interval covers and give NaN for a
region no interval covers, which a figwig file cannot tell from 0 unless
it was written with ``--missing nan``; there, ``nan`` says what a NaN base
does: ``'propagate'`` makes the sum NaN, ``'skip'`` leaves the base out,
and ``'count'`` gives the number of bases that are not NaN, so that the
mean over covered bases is one over the other:

.. code-block:: python

    covered = fw.sums(chroms, starts, ends, nan="skip") / fw.sums(chroms, starts, ends, nan="count")

``read`` takes ``nan`` with ``bins`` the same way. ``sums(..., paired=True)``
takes one track per interval and sums interval ``i`` over track ``i``
alone, in reads of a few tracks of one group at a time, for a reads
dimension where each read's span is its own track; without it, a sum of
every track over every interval is quadratic in the reads. It costs about
0.03 ms per interval on 90,543 Fiber-seq reads. An analysis of count
tracks ported from those tools gives other means for regions with
uncovered bases unless the file was written so. ``sums`` takes any number
of tracks: over 3,000 cells, 58,705 peaks take a second on 8 threads, the
same as a sparse read's sums; a chunk that fewer than four windows fall
on, or a read of more tracks than a scratch tile holds (256 MB), is binned
straight from its entries, and a float32 sum is taken in float64.

``read_entries(tracks, chroms)`` gives every nonzero base of some tracks
over whole chromosomes as a ``SparseEntries`` of chromosome index,
position, track index and value, in the narrowest type that holds every
track exactly, which is the read a file in track order is laid out for.

Summaries
---------

The file keeps each track's sum over every tile, and ``tile_sums`` gives
them without decoding a chunk:

.. code-block:: python

    sums = fw.tile_sums()                          # (6, tiles over the genome) float32
    chr1 = fw.tile_sums("atac.+", "chr1")          # (15196,) at 16,384 bases per tile
    per_mb = chr1[:15168].reshape(-1, 64).sum(-1)  # 1 Mb bins

This is what a bigWig's zoom levels serve: a genome-wide profile at the
resolution of the tiles, a per-chromosome total, or a coarser binning by
a reshape, in the time it takes to inflate a few hundred kilobytes.
Chromosome by chromosome, tile ``k`` holds bases ``[k * tile, (k + 1) *
tile)`` and a chromosome of length ``L`` has ``ceil(L / tile)`` tiles. A
NaN base is left out of its tile's sum, as it is left out of the track's
sum and its count of nonzero bases, so a file written with its uncovered
bases NaN keeps the statistics of its covered ones. The sums are
kept while every track's take at most 256 MB together, which is 350 tracks
on hg38 at the default tile; a file of thousands of cells keeps none, and
``tile_sums`` says so.

A file written with ``zooms=k`` also keeps ``k`` zoom levels, level ``j``
the sum of each track over bins of ``4**j`` tiles, in float64, which
``tile_sums(level=j)`` returns the same way (``fw.zooms`` lists the levels,
``(4, 16, 64)`` for three) and ``figwig info`` prints; ``bw2fw -z`` and
``bam2fw -z`` write them, and ``reorder``, ``merge`` and ``append=True``
keep the file's count. A bin never crosses a chromosome: a chromosome of
``n`` tiles has ``ceil(n / 4**j)`` bins at level ``j``. The levels are
computed from the tile sums at close, so they are kept only while the
tile sums are, and the default is none; they cost a few hundred kilobytes
per level for six tracks and are announced by a header bit that a reader
without them ignores.

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
only where its values are NaN or 0. A base that is not written is 0, and a
call that raises writes nothing.

``tracks`` are names, or dicts with a ``name``, an optional ``dtype`` and
any JSON-serializable attributes; NaN is not JSON and is refused, and
``group`` is set by the writer. ``genome`` is a name or a URL, and
``attributes`` a dict for the file as a whole. A track's ``dtype`` is what
``read`` returns it as: given, it is checked against every value written;
left out, it is set when the writer closes to the narrowest type that
holds the track's values, uint8 here, float32 where a value is not an
integer, and float64 where an integer lies beyond int32 and uint32, which
float32 would round. Either way each chunk is stored as narrowly as its
own values allow, so declaring ``uint32`` costs nothing on disk. The
values a track keeps are those its dtype holds: a float32 track keeps
float32's rounding of what it is given, as does a track left undeclared
that turns out to hold fractions, so a value too small for float32 is 0
and not counted as nonzero, and a value beyond float32's range is an error
unless the track is declared ``float64``; -0.0 is 0. A chunk of an
undeclared track written before its first fraction was seen may hold an
integer exactly where float32 would round it, so a read with
``dtype='float64'`` can differ between a position-order and a track-order
file of the same writes above 2**24: read such a track in its own dtype,
or declare it ``float64``. No figwig file holds an infinite value: a base
is a finite value or NaN, and ``write``, ``write_sparse`` and ``bw2fw``
refuse ``inf`` before writing anything, as ``BigWigWriter`` does (ENCODE's
signal p-value bigWigs can hold it; clip them first).

``groups`` says which tracks' tiles are stored together: a number of
consecutive tracks, 8 by default, or a list of groups of names, each of
which may list its tracks in any order, since a group's tracks lie in the
file's track order within its chunks. A read of any track in a group
decodes the whole group's chunk, so the tracks of a group should be those
read together, such as the strands of one assay, and tracks of one kind,
since a chunk's element type is chosen for the whole group: a count track
grouped with a conservation track costs float32 bytes per count before
compression, and its reads decode the conservation values too. A file of
cells read as pseudobulks wants every cell in one group, and a file read a
few tracks at a time wants track order. The defaults, a tile of 16,384
and groups of 8, suit count tracks, whose chunks are mostly zero; dense
float tracks read in short windows, such as conservation, fold-change or
p-value signal, want ``tile=1024`` and ``groups=1``, which
:doc:`benchmarks` measures at seven times the speed of the defaults for a
file half again as large. A ``write`` call covers whole groups.
``write_figwig`` writes one call's worth to a new file:

.. code-block:: python

    from figwig import write_figwig

    write_figwig("example.fw", chrom_sizes, ["plus", "minus"], chroms, starts, counts)

Each tile is encoded as the writer reaches it, and a tile a call fills only
part of waits for the next call, so predictions can be written a batch at a
time. Chunks are compressed on ``n_jobs`` threads, with libdeflate when the
``deflate`` package is installed, as :doc:`writing` describes for bigWigs,
and a chunk that compression would grow is stored as it is. ``level=0``
stores every chunk uncompressed, which decodes fastest: the six count
tracks of :doc:`benchmarks` are 412 MB at level 0 against 227 MB at level
6, and a dense float track costs four bytes per base. Reads inflate chunks
with libdeflate as well where it
is installed, about twice as fast as zlib, and with zlib otherwise; the
level barely changes the file's size or its reads, and :doc:`benchmarks`
measures the levels and other codecs. The
chunks depend on the values and on which calls wrote which groups, not on
the number of threads or on how the values were split between calls; the
sum in the statistics table is a compensated float64, which splits may move
in its last bits.

The writer writes beside the path it was given, with the mode the umask
gives a new file, and moves the file there when it closes, so a reader of that path never sees part of a file, a
writer that raises or is never closed leaves the path as it was, and a
converter may write over its own input. A writer that is garbage collected
with values written and no ``close`` warns. Every chunk's CRC-32 goes into
the index, and ``FigWigReader(path, verify=True)`` checks each chunk it
reads against it; a compressed chunk's zlib checksum is checked on every
read either way, as are the CRC-32 of a chunk a compressed file stores as
it is, of the tables and directory (in the header) at open and of each
group's index record when it is read, so ``verify`` is for uncompressed
files, at the cost of a
pass over the bytes read, and for the rare damage zlib's Adler-32 misses:
of 2,715 single-bit flips in one compressed chunk, 2,714 were refused and
one read as four wrong bases, which ``verify`` caught.

Adding tracks to a file
-----------------------

.. code-block:: python

    with FigWigWriter("k562.fw", None, [{"name": "h3k27ac", "assay": "ChIP-seq"}],
            append=True) as fw:
        for chrom, starts, values in batches:
            fw.write(chrom, starts, values)

    with FigWigWriter("all.fw", chrom_sizes, None, tile=16384) as fw:
        fw.add_file("k562.fw")
        fw.add_file("hepg2.fw")

``append=True`` adds the tracks given to the file at the path, in place,
in new groups after its own. The file's chunks stay where they are, the
new chunks are written after its end, and the tables, the index and last
the header are written when the writer closes. Until then the file reads
as it did: a reader already open on it keeps reading and is told to open
it again once the append has closed, a writer that fails cuts the file
back to what it was, and a process killed part way leaves a file whose
header still describes the old one. The writer holds the file's lock
while it appends, so a second writer appending to the same file at the
same time is refused, since its chunks would land on the first's. The
file's chromosomes, tile, level and order are taken, ``attributes`` adds
to the file's, and ``chrom_sizes``, ``tile``, ``level`` and ``order`` must
agree with them or be left at None. Appending costs the new chunks alone,
and leaves the old tables and index behind as dead bytes between the
chunks, about the size of the index, which ``figwig merge`` of the file
into a new path drops.

``add_file`` adds every track of another file written with the same
chromosomes, tile, level and order, by copying its chunks, with its
attributes, dtypes, statistics and summaries; ``figwig merge`` does that
from the command line. Tracks that came from a file cannot be written to,
and ``write`` with ``tracks=None`` means every track given to the writer
itself.

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
nonzero bases of one tile come in one call. A call takes about 90 bytes of
memory per entry besides the arrays given, so a genome's entries are best
written a chromosome at a time, as above.

``order`` fixes the layout when the writer is made, since it decides the
index and the chunks. ``'positions'`` (or ``'csc'``) is the tile layout
above; ``'tracks'`` (or ``'csr'``) stores each track's nonzero bases in
blocks of up to 65,536 per chromosome, uncompressed unless ``level`` is
given, since a window read over many tracks decodes a block of each. Both
orders take the same ``write`` and ``write_sparse``, return the same arrays
from ``read``, and differ in what is fast: ``figwig reorder`` rewrites a
file in the other order. A tile whose few tracks have values is stored as
a matrix chunk, whose size grows with the values rather than with the
tracks, in an uncompressed file, and in a compressed one when the sparse
chunk's count per track, mostly zero, would not deflate to less, as with
100,000 cells in one group.

Converting
----------

:doc:`converters` describes the commands: ``figwig bw2fw`` writes bigWigs
as the tracks of one figwig file, with a metadata table for their names
and attributes, or adds them to a file with ``--append``; ``figwig fw2bw``
writes a figwig file's tracks back as bigWigs, with that table; ``figwig
bam2fw`` counts reads or fragments as ``figwig bam2bw`` does, one pair of
tracks per input file; ``figwig reorder`` writes a figwig file in the
other order; ``figwig merge`` writes the tracks of several files into one,
copying their chunks; and ``figwig info`` prints what a file holds. A
bigWig's values read from the figwig file as ``read_bigwig`` gives them
with ``missing=0.0`` and ``keep_nan=True``, and ``bw2fw --missing nan``
keeps the bases no interval covers apart from 0, for tracks where the two
differ.

Reads, cells and channels as tracks
-----------------------------------

A dimension that is not a track in the bigWig sense, such as the reads of
a Fiber-seq experiment with a methylation call and a base quality per
base, or the cells of a single-cell experiment with a count and a
fragment-end flag, is stored as tracks with attributes that name the
dimension: a record per read and channel, ``{"name": "r17.m6a", "read":
"r17", "channel": "m6a", "strand": "+"}``, so that ``select(channel="m6a")``
or ``select(read="r17")`` picks a slice. ``read(..., channels="channel",
items="read")`` arranges the tracks read by item and channel into
(windows, reads, channels, width), items and channels in the order of
their first appearance among the tracks read, each item needing every
channel once; ``bins`` and ``sums`` arrange the same way, into (windows,
reads, channels, bins) and (intervals, reads, channels). The arrangement
is a reshape of the ordinary read and changes no bytes of a file: a
channel is an attribute like any other.

Group the tracks by channel over runs of consecutive reads, not by read:
the reads sorted by start and cut into groups of about a thousand, so that
each group's chunks cover one stretch of the genome and a window decodes
the few chunks of the groups whose reads overlap it. A chunk may hold at
most 2**29 values (tracks times tile), which rules out one group of every
read, and a group per read costs a planning step per group on every read.
On the m6A calls of 90,543 Fiber-seq reads in groups of 1,024 (position
order, tile 16,384, 62 MB), summing every read over 8,192 windows of
2,048 bases took 0.42 s and listing the calls in 1,024 windows 0.16 s,
against 3.8 s and 3.5 s at best from a scipy CSR matrix per chromosome
held in memory, and 15.7 s and 7.1 s from the same file in track order;
``sums(..., paired=True)`` gives each read's total over its own span,
2.7 s for the 90,543 reads against 5.8 s from the matrix
(:doc:`benchmarks`). A read may also be its own "chromosome", of its
length, with one track per channel, which gives (reads, channels, length)
in one call, with the genome coordinates kept in the file's attributes.

Versions
--------

The header names the format's version, 1 in this first release, and a
reader refuses a file of a later version with a message that says a newer
figwig wrote it. The version changes only when the layout changes in a way
an older reader would misread; an addition an older reader could ignore is
marked instead in the header's ``optional`` bits, and one it must know in
its ``required`` bits, which a reader refuses when it does not know them.
Within a version, a file written by any figwig reads in any later figwig,
and a rebuild from the same inputs with the same options is
byte-identical, which is why the file carries no timestamp.

What the format is for, and what it is not
------------------------------------------

The format is built for per-base signal read in windows across many tracks
at once: counts and pileups, fold change and p-values, conservation and
mappability, predictions, and cells. Measured on 60 bigWigs of the kinds
the genomics test collection holds, a figwig file of one track is smaller
than the bigWig figwig's own writer makes of it for dense per-base floats
(a tenth to four fifths of the size), deep counts and coverage, and tracks
of a few runs of one value, and larger for sparse low-depth counts, where a
chunk of a few bases has a fixed cost that a bigWig's varStep item does
not. The gain in size over a directory of bigWigs comes mostly from the
tracks being in one file and in the narrowest type: six count tracks that
are 971 MB as ENCODE's bigWigs are 408 MB as bigWigs figwig writes and 227
MB as one figwig file. The gain in reading comes from the tile grid: a
window's chunks are found by division, each is decoded once for every track
in it, and the values arrive as uint8 or uint16, half or a quarter of the
bytes of float32 (:doc:`benchmarks`).

A figwig file has no "no data". A base that was not written, a base an
interval of 0 covers and a base outside every interval are the same 0, so
a bigWig whose zeros mean something other than its gaps, as a methylation
track's 0% sites do against its unsequenced sites, or a conservation
track's unaligned bases, loses that distinction through ``bw2fw`` unless
``--missing nan`` is given, which writes the uncovered bases as NaN and the
track as float32. The tile sums, and the zoom levels a file is written
with, stand in for a bigWig's zoom levels down to the tile's resolution:
a summary at 1 Mb bins is a reshape of them, and a summary finer than a
tile decodes the chunks under it, through ``bins``, which took 0.09 s for
100 bp bins over chr22 where pyBigWig's ``stats`` took 9.8 s
(:doc:`benchmarks`); a genome browser needs the tracks written back with
``fw2bw -z``. Every value is at
most float64, and there are no string or categorical tracks. A read from a
remote file would need the group's index, a few tens of KB for a
genome-wide track, before its first chunk; the reader does not read over
HTTP. The format is little-endian, and read and written by this package
alone; the layout below is complete enough to read it without it.

The layout
----------

A figwig file is, in order: a 128-byte header; the chunks or blocks, as
they were written; the chromosome table; a table of each track's sum,
nonzero bases, minimum and maximum; the metadata, a JSON document with the
writing figwig's version (``figwig``), the ``genome``, the file's
``attributes``, its ``order`` and every track's record (``tracks``), in
which any other key is kept through an append for whoever put it there;
and the index. Every integer is little-endian, and every chunk starts on an 8-byte
boundary of the file. The index finds every chunk by its offset, so a
file appended to holds its old tables and index between its chunks, and
bytes past the header's ``length`` are ignored. A reader checks each
table and each chunk it decodes against the file's length and the format
before asking for memory, so a truncated or corrupt file raises a
``ValueError`` that says what it found.

The header, 128 bytes:

=================  ====  ==================================================
field              type  meaning
=================  ====  ==================================================
magic              u4    ``FIGW`` as little-endian bytes, 0x57474946
version            u2    1; a reader refuses a later version
flags              u2    bit 0 set for track order
tile               u4    bases per tile, a power of two from 16 to 65,536
codec              u2    0 for uncompressed chunks, 1 for zlib streams
level              u2    the compression level the chunks were written at
n_chroms           u4    chromosomes in the chromosome table
n_tracks           u4    tracks
n_groups           u4    groups, which is n_tracks in track order
tables_crc         u4    CRC-32 of the bytes from ``chroms`` to the end of the index's directory
chroms             u8    offset of the chromosome table
tracks             u8    offset of the statistics table
index              u8    offset of the index
data               u8    offset of the first chunk, 128
metadata           u8    offset of the metadata
metadata_length    u8    bytes of the metadata
zooms              u8    0; the zoom levels lie in the index, after each group's summary
length             u8    bytes of the file
required           u4    feature bits a reader must know; none defined, so a reader refuses any set bit
optional           u4    feature bits a reader may ignore; bit 0 says each index record ends with a zoom section, and is set in every file this release writes
header_crc         u4    CRC-32 of these 128 bytes with this field as 0
padding            20    0
=================  ====  ==================================================

The chromosome table is, for each chromosome in the file's order, the
length of its UTF-8 name as a u2, the name, and its length in bases as a
u8, from 1 to 2**32 - 1. The statistics table is one 32-byte record per
track: the sum (f8), the nonzero bases (u8), the minimum (f8) and the
maximum (f8) of the values written, the minimum and maximum 0 for a track
with no nonzero base. The metadata is UTF-8 JSON with the keys ``genome``
(a string or null), ``attributes`` (an object), ``order`` (``positions``
or ``tracks``) and ``tracks``, a list with one object per track in order,
each with its ``name``, its ``dtype`` (one of ``uint8``, ``uint16``,
``uint32``, ``int8``, ``int16``, ``int32``, ``float32``, ``float64``), its
``group`` (an integer from 0) and its attributes; a reader takes the
records' order as the track order and, within a group, as the order of the
group's tracks in its chunks.

The index starts with a directory of ``n_groups`` u8 offsets, one per
group's record. In position order a record is ``n_runs`` (u4) and the
CRC-32 (u4) of the record's bytes up to the summary, then that
many 32-byte runs of ``first`` tile, ``last`` tile, ``offset`` in the file
and ``bytes`` (all u8), then ``n_live`` (u4), ``n_bytes`` (u4) and a zlib
stream of ``n_bytes`` that inflates to ``17 * n_live`` bytes: the live
tiles' numbers (u4 each, ascending), then their kind bytes (u1 each), then
their compressed sizes (u4 each), then their uncompressed sizes (u4 each),
then the CRC-32 of each chunk's bytes as they lie in the file (u4 each).
Tiles are numbered over the genome in order, each chromosome's tiles after
the one before's, and a tile not listed holds only zeros. A run holds the
chunks of tiles ``first`` to ``last``, back to back from ``offset`` for
``bytes`` bytes, so a chunk's offset is its run's plus the compressed
sizes of the run's earlier tiles. A group written chromosome by chromosome
has one run; groups written in turn have a run per switch. In track order
a record is ``n_blocks`` (u4) and the CRC-32 (u4) of the blocks' bytes,
then that many 40-byte blocks of
``chrom`` (u4, an index into the chromosome table), ``first`` and
``last`` position (u4), ``n`` entries (u4), ``offset`` (u8), compressed
``size`` and uncompressed ``raw`` size (u4), ``kind`` (u4, the kind byte)
and the block's CRC-32 (u4). Either record ends with the group's summary:
``n_sum`` (u4), ``n_sum_bytes`` (u4) and a zlib stream of that many bytes
that inflates to ``n_sum * (4 + 4 * tracks)`` bytes, the numbers of the
tiles with a nonzero sum (u4 each, ascending) then each of those tiles'
sums per track of the group (f4 each, tile by tile); a group whose
summary was not kept has ``n_sum_bytes`` of 0. When the header's
``optional`` bit 0 is set, the summary is followed by the group's zoom
section: ``n_zooms`` (u4, at most 12), then for level ``k`` from 1 its
``factor`` (u4, ``4 ** k``), ``n`` bins (u4), ``n_bytes`` (u4), the CRC-32
(u4) of the stream that follows, and a zlib stream of ``n_bytes`` that
inflates to ``n * (4 + 8 * tracks)`` bytes: the numbers of the bins with a
nonzero sum (u4 each, ascending) then each of those bins' sums per track
(f8 each, bin by bin). Bins are numbered over the genome as tiles are,
each chromosome of ``n`` tiles taking ``ceil(n / factor)`` bins, so a bin
never crosses a chromosome. A file written without zoom levels has
``n_zooms`` of 0, and a group whose summary was not kept keeps no levels.

A chunk's kind byte holds its kind in bits 0-2: 0 for a tile that holds
only zeros and has no chunk, 1 dense, 2 sparse, 3 matrix, 4 runs; its
element code in bits 3-6: 0 uint8, 1 uint16, 2 uint32, 3 int8, 4 int16, 5
int32, 6 float32, 7 "one", 8 float64; and in bit 7 whether the chunk is
stored as it is in a compressed file, because deflating it would have made
it larger, in which case its compressed and uncompressed sizes are equal.
A compressed chunk is one zlib stream. A dense chunk holds every value of
the tile, track after track in the group's order, in the element type. A
sparse chunk holds the number of nonzero bases in all (u4), then for each
track the bytes its offsets take (u4), then each track's nonzero offsets
within the tile, the first as it is and the rest as differences from the
one before, each as a LEB128 varint (seven bits per byte, low bits first,
the high bit set on every byte but the last; at most three bytes), then
padding to a multiple of 4 bytes, then each track's nonzero values in the
element type; a track's number of bases is the number of its bytes whose
high bit is clear. A runs chunk has the same layout, but its offsets are
the bases where a track's value changes, and its values the value from
each change up to the next, or to the end of the tile, with 0 before a
track's first change. A matrix chunk holds the number of tracks present
(u4), their positions in the group (u4 each, ascending), a u4 pointer per
present track into the entries plus one at the end, the entries' offsets
within the tile (u2 each, ascending within a track), padding to 4 bytes,
and the values. The "one" element type stores no values, every nonzero
value being 1; a dense or runs chunk never uses it. A chunk's bytes are
padded to a multiple of 8. A track-order block holds its entries'
positions on the chromosome (u4 each, ascending), padding to 8 bytes, and
their values in the block's element type, padded to 8 bytes.

A reader that follows this can read a figwig file with numpy and zlib
alone, as one of the tests of the format does.

Limits
------

=================================  ===================================================================
what                               limit
=================================  ===================================================================
chromosome length                  2**32 - 1 bases
chromosome name                    65,535 bytes of UTF-8
tiles over the genome              2**26; the writer refuses a tile too small for the genome
tile sums                          kept while tracks x tiles x 4 bytes is at most 256 MB
tracks per group                   2**29 / tile: 32,768 at the default tile, 131,072 at 4,096
nonzero bases per track-order      65,536 per block, any number of blocks
block
dense read                         8 GiB unless ``out`` is given
scratch tile for sums              256 MB per thread, channels times tile times 8 bytes; a read of more channels bins each window from its chunks without one
values                             integers to 2**53 exactly; a 64-bit integer beyond it is refused; float32 or float64 otherwise; NaN allowed, infinity refused
=================================  ===================================================================

The position-order index on disk holds the tiles with values, so a group
with little data costs a few bytes; in memory a group's index is 17 bytes
per tile of the genome, 3.2 MB for hg38 at the default tile, kept from a
group's first read on, so a file of thousands of groups read all at once
holds thousands of those, and wants larger groups or track order.
