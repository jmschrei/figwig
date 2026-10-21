# bw2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig bw2fw` command: bigWigs into one figwig file.

Each bigWig becomes one track, named after its file unless a metadata table
names it, and the table's other columns become the track's attributes. The
tracks are read a group at a time, each chromosome in pieces, with
`read_bigwig`, and written with `FigWigWriter`, so that a bigWig's values
read back from the figwig file as `read_bigwig` gives them with
`missing=0.0` and `keep_nan=True`: a base no interval covers is 0, or the
value `--missing` gives, and an interval whose value is NaN is NaN.
"""

import argparse
import os
import sys
import warnings

from concurrent.futures import ThreadPoolExecutor

import numpy

from ._tables import _match_rows
from ._tables import _read_table
from ._tables import _track_name
from .bigwig import BigWigReader
from .bigwig import read_bigwig
from .figwig import _DEFAULT_GROUPS
from .figwig import _DEFAULT_TILE
from .figwig import FigWigWriter

# A chromosome is read and written this many bases at a time, as windows of
# this many bases.
_PIECE = 1 << 22
_WINDOW = 1 << 16


def _parser(prog):
	parser = argparse.ArgumentParser(prog=prog, description="Convert bigWig "
		"files into one figwig file, one track per bigWig.")
	parser.add_argument('bigwig', nargs='+', help="The bigWig files, one per "
		"track, in the order the tracks take.")
	parser.add_argument('-o', '--output', required=True, help="The figwig file "
		"to write.")
	parser.add_argument('-m', '--metadata', default=None, help="A tab- or "
		"comma-separated table with a header and one row per bigWig, matched "
		"on its first column by the path given here or its base name. A "
		"'name' column names the track, a 'dtype' column sets its dtype, and "
		"every other column is an attribute of the track.")
	parser.add_argument('-s', '--sizes', default=None, help="A chrom_sizes, "
		".fai or FASTA file giving the chromosomes and their order. Default is "
		"every chromosome of the bigWigs, in order of first appearance.")
	parser.add_argument('-g', '--genome', default=None, help="The reference "
		"genome, as a name or a URL, recorded in the file.")
	parser.add_argument('-t', '--tile', default=_DEFAULT_TILE, type=int,
		help="The bases per tile, a power of two from 16 to 65536. Default is "
		"{}.".format(_DEFAULT_TILE))
	parser.add_argument('--groups', default=_DEFAULT_GROUPS, type=int,
		help="The consecutive tracks whose tiles are stored together. Default "
		"is {}.".format(_DEFAULT_GROUPS))
	parser.add_argument('--dtype', default=None, help="The dtype of every "
		"track, such as uint16. Default is the narrowest type that holds each "
		"track's values.")
	parser.add_argument('-l', '--level', default=6, type=int, help="The "
		"compression level, or 0 to store the chunks uncompressed. Default is "
		"6.")
	parser.add_argument('--missing', default=0.0, type=float, help="The value "
		"of a base that no interval of a bigWig covers: 0, or nan to keep it "
		"apart from a measured 0, as for methylation or conservation, at the "
		"cost of a float32 track. An interval whose value is NaN is NaN either "
		"way. Default is 0.")
	parser.add_argument('-p', '--parallel', default=1, type=int, help="The "
		"number of threads to read and compress on. Default is 1.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


def _tracks_from_table(filenames, metadata, dtype=None):
	"""One track record per file: its name, dtype and attributes, from the
	metadata table when there is one."""

	tracks = []
	rows = [None] * len(filenames)
	if metadata is not None:
		header, records = _read_table(metadata)
		rows = _match_rows(metadata, records, header, filenames)

	for filename, row in zip(filenames, rows):
		track = {'name': _track_name(filename), 'dtype': dtype}
		if row is not None:
			for column, value in row.items():
				if column == 'name' and len(value) > 0:
					track['name'] = value
				elif column == 'dtype' and len(value) > 0:
					track['dtype'] = value
				elif column not in ('name', 'dtype') and column != list(row)[0]:
					track[column] = value
		tracks.append(track)

	return tracks


def _shared_chrom_sizes(readers, sizes=None):
	"""The chromosomes of the output: those of the sizes file, which must
	agree with the bigWigs on the length of every chromosome they share,
	or every chromosome of the bigWigs in order of first appearance, which
	must agree on each one's length. A chromosome of a bigWig that the sizes
	file leaves out is named in a warning, since its values are dropped."""

	if sizes is not None:
		from .bam2bw import read_chrom_sizes
		chrom_sizes = read_chrom_sizes(sizes)
		lengths = dict(chrom_sizes)
		dropped = []
		for reader in readers:
			for name, length in reader.chrom_sizes.items():
				if name not in lengths:
					dropped.append(name)
				elif lengths[name] != length:
					raise ValueError("{} gives {} a length of {}, but the sizes "
						"file gives it {}; the bigWigs and the sizes must be on one "
						"assembly.".format(reader.path, name, length, lengths[name]))
		if dropped:
			dropped = sorted(set(dropped))
			warnings.warn("{} chromosome{} of the bigWigs {} not in the sizes "
				"file, and {} values are left out: {}{}".format(len(dropped), "s"
				if len(dropped) > 1 else "", "are" if len(dropped) > 1 else "is",
				"their" if len(dropped) > 1 else "its", ", ".join(dropped[:5]),
				", ..." if len(dropped) > 5 else "."), stacklevel=2)
		return chrom_sizes

	chrom_sizes = {}
	for reader in readers:
		for name, length in reader.chrom_sizes.items():
			if chrom_sizes.setdefault(name, length) != length:
				raise ValueError("{} gives {} a length of {}, but another bigWig "
					"gives it {}.".format(reader.path, name, length,
					chrom_sizes[name]))

	return list(chrom_sizes.items())


def _convert(filenames, output, tracks, chrom_sizes, genome=None, tile=_DEFAULT_TILE,
	groups=_DEFAULT_GROUPS, level=6, n_jobs=1, verbose=False, missing=0.0):
	"""Write the bigWigs `filenames` as the tracks of `output`."""

	readers = [BigWigReader(filename) for filename in filenames]

	def pieces(writer):
		for g, members in enumerate(writer._groups):
			group = [readers[i] for i in members.tolist()]
			names = [writer.tracks[i]['name'] for i in members.tolist()]
			for chrom, length in chrom_sizes:
				# A piece is read as windows of a tile's width, so that the
				# bigWig reader's batches spread over the threads.
				width = int(min(_WINDOW, length))
				for a in range(0, length, _PIECE):
					starts = numpy.arange(a, min(length, a + _PIECE), width)
					yield g, group, names, chrom, starts, width

	def read_piece(piece):
		g, group, names, chrom, starts, width = piece
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			return read_bigwig(group, [chrom] * len(starts), starts, width,
				n_jobs=n_jobs, missing=missing, keep_nan=True)

	# Each piece is read while the one before it is written: the writer
	# encodes and compresses on its own threads, and the arrays read here
	# are handed to it without a copy, since nothing here touches them
	# again.
	with FigWigWriter(output, chrom_sizes, tracks, genome=genome, tile=tile,
			groups=groups, level=level, n_jobs=n_jobs) as writer, \
			ThreadPoolExecutor(1) as reader:
		writer._copies = False
		ahead = None
		for piece in pieces(writer):
			future = reader.submit(read_piece, piece)
			if ahead is not None:
				_write_piece(writer, output, ahead, verbose)
			ahead = (piece, future)
		if ahead is not None:
			_write_piece(writer, output, ahead, verbose)


def _write_piece(writer, output, ahead, verbose):
	(g, group, names, chrom, starts, width), future = ahead
	writer.write(chrom, starts, future.result(), tracks=names)
	if verbose and int(starts[-1]) + width >= writer.chrom_sizes[chrom]:
		print("{}: group {} of {}, {}".format(output, g + 1, len(
			writer._groups), chrom), file=sys.stderr)


def main(argv=None, prog='figwig bw2fw'):
	args = _parser(prog).parse_args(argv)
	for filename in args.bigwig:
		if not os.path.isfile(filename):
			raise SystemExit("{}: {} is not a file.".format(prog, filename))

	try:
		tracks = _tracks_from_table(args.bigwig, args.metadata, args.dtype)
		readers = [BigWigReader(filename) for filename in args.bigwig]
		chrom_sizes = _shared_chrom_sizes(readers, args.sizes)
		_convert(args.bigwig, args.output, tracks, chrom_sizes, genome=args.genome,
			tile=args.tile, groups=args.groups, level=args.level,
			n_jobs=args.parallel, verbose=args.verbose, missing=args.missing)
	except (ValueError, TypeError) as e:
		raise SystemExit("{}: {}".format(prog, e))
