# bam2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig bam2fw` command: reads or fragments into one figwig file of
per-base counts.

It takes what `figwig bam2bw` takes, SAM/BAM files of reads or BED/tsv
files of fragments and the same options for what to count, and reads them
the same way. Where `figwig bam2bw` pools every file into one pair of
bigWigs, this writes each file as its own pair of tracks, `<name>.+` and
`<name>.-`, or one track with `-u`, in one group, so that a figwig file
holds a whole set of experiments. A metadata table gives the tracks their
names and attributes.
"""

import argparse
import collections
import importlib.util
import os

import numpy

from ._tables import _match_rows
from ._tables import _read_table
from .figwig import _DEFAULT_TILE
from .figwig import FigWigWriter

_EXTENSIONS = ('.bam', '.sam', '.tsv.gz', '.tsv', '.bed.gz', '.bed')


def _parser(prog):
	from .bam2bw import _input_arguments

	parser = argparse.ArgumentParser(prog=prog, description="Convert SAM/BAM "
		"files of reads, or BED/tsv files of fragments, into one figwig file of "
		"per-base counts, with a track per strand of each file.")
	_input_arguments(parser)
	parser.add_argument('-p', '--parallel', default=1, type=int, help="The "
		"number of cores to use, or a negative number to count back from the "
		"number of CPUs: -1 is all of them.")
	parser.add_argument('-n', '--name', required=True, help="The figwig file "
		"to write, without its .fw extension.")
	parser.add_argument('-m', '--metadata', default=None, help="A tab- or "
		"comma-separated table with a header and one row per input file, "
		"matched on its first column by the path given here or its base name. "
		"A 'name' column names the file's tracks, and every other column is an "
		"attribute of them.")
	parser.add_argument('-g', '--genome', default=None, help="The reference "
		"genome, as a name or a URL, recorded in the file.")
	parser.add_argument('-t', '--tile', default=_DEFAULT_TILE, type=int,
		help="The bases per tile, a power of two from 16 to 65536. Default is "
		"{}.".format(_DEFAULT_TILE))
	parser.add_argument('--dtype', default=None, help="The dtype of every "
		"track, such as uint16. Default is the narrowest type that holds each "
		"track's values.")
	parser.add_argument('-l', '--level', default=1, type=int, help="The "
		"compression level, or 0 to store the chunks uncompressed. Default is "
		"1.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


def _base_name(filename):
	"""A file's name without its directory and its extension."""

	name = os.path.basename(filename)
	for extension in _EXTENSIONS:
		if name.endswith(extension):
			return name[:-len(extension)]

	return name


def _tracks_for_files(filenames, metadata, unstranded, dtype=None):
	"""The track records of every file, as (tracks, groups): a track per
	strand of each file, or one with `unstranded`, and the files' rows of the
	metadata table as their attributes."""

	rows = [None] * len(filenames)
	if metadata is not None:
		header, records = _read_table(metadata)
		rows = _match_rows(metadata, records, header, filenames)

	tracks, groups = [], []
	for filename, row in zip(filenames, rows):
		name = _base_name(filename)
		attributes = {}
		if row is not None:
			first = list(row)[0]
			for column, value in row.items():
				if column == 'name' and len(value) > 0:
					name = value
				elif column not in ('name', 'dtype', first):
					attributes[column] = value

		if unstranded:
			names = [name]
			strands = [None]
		else:
			names = [name + '.+', name + '.-']
			strands = ['+', '-']

		group = []
		for track_name, strand in zip(names, strands):
			track = {'name': track_name, 'dtype': dtype}
			if strand is not None:
				track['strand'] = strand
			track.update(attributes)
			tracks.append(track)
			group.append(track_name)
		groups.append(group)

	return tracks, groups


def _trim_counts(reads, chrom_sizes, unstranded):
	"""Each file's counts as sorted arrays within their chromosomes, with
	the entries that fall outside a chromosome dropped and counted."""

	from .bam2bw import as_arrays

	trimmed, discarded = [], collections.defaultdict(int)
	for pos_reads, neg_reads in reads:
		strands = {}
		for key, counts in (('+', pos_reads), ('-', neg_reads)):
			if unstranded and key == '-':
				continue
			strands[key] = {}
			for chrom, size in chrom_sizes:
				positions, values = as_arrays(counts.pop(chrom))
				lo, hi = numpy.searchsorted(positions, [0, size])
				if lo > 0 or hi < len(positions):
					discarded[chrom] += int(values[:lo].sum()) + int(values[hi:]
						.sum())
				strands[key][chrom] = positions[lo:hi], values[lo:hi]
		trimmed.append(strands)

	return trimmed, discarded


def _write_counts(writer, names, strands, chrom_sizes, scale_factor, read_depth):
	"""Write one file's counts, strands['+'] and strands['-'] or strands['+']
	alone, as the tracks `names`, chromosome by chromosome, each count times
	`scale_factor` and divided by `read_depth` unless it is None."""

	for chrom, _ in chrom_sizes:
		parts = [strands[key][chrom] for key in ('+', '-') if key in strands]
		# Each strand's positions are sorted, so a stable sort of the two
		# together merges them; numpy.unique hashes, which is far slower here.
		positions = numpy.concatenate([p for p, _ in parts])
		if len(positions) == 0:
			continue
		positions.sort(kind='stable')
		positions = positions[numpy.concatenate([[True], positions[1:] !=
			positions[:-1]])]

		values = numpy.zeros((len(positions), len(parts), 1), dtype=numpy.float64)
		for k, (p, counts) in enumerate(parts):
			values[numpy.searchsorted(positions, p), k, 0] = counts

		values *= scale_factor
		if read_depth is not None:
			values /= read_depth

		writer.write(chrom, positions, values, tracks=names)


def main(argv=None, prog='figwig bam2fw'):
	from .bam2bw import _check_filenames
	from .bam2bw import _cores
	from .bam2bw import _REQUIRED
	from .bam2bw import _tqdm
	from .bam2bw import read_chrom_sizes
	from .bam2bw import read_files

	args = _parser(prog).parse_args(argv)

	missing = [name for name in _REQUIRED if importlib.util.find_spec(name) is
		None]
	if len(missing) > 0:
		raise SystemExit(("{} needs {}, from figwig's bam2bw extra: pip install "
			"'figwig[bam2bw]'").format(prog, ', '.join(missing)))

	try:
		_check_filenames(args)
		chrom_sizes = read_chrom_sizes(args.sizes)
		tracks, groups = _tracks_for_files(args.filename, args.metadata,
			args.unstranded, args.dtype)
	except (ValueError, TypeError) as e:
		raise SystemExit("{}: {}".format(prog, e))

	tqdm = _tqdm(args.verbose)
	if args.verbose:
		import multiprocessing
		tqdm.set_lock(multiprocessing.RLock())

	cores = _cores(args.parallel)
	reads = read_files(args, chrom_sizes, cores)
	if args.verbose:
		print("\n" * len(args.filename))

	files, discarded = _trim_counts(reads, chrom_sizes, args.unstranded)
	del reads
	if len(discarded) > 0:
		print("{} entries fell outside the provided chromosome sizes and were "
			"discarded ({}).".format(sum(discarded.values()), ", ".join(
			"{}: {}".format(chrom, count) for chrom, count in discarded.items())))

	try:
		with FigWigWriter(args.name + '.fw', chrom_sizes, tracks,
				genome=args.genome, tile=args.tile, groups=groups, level=args.level,
				n_jobs=cores) as writer:
			for strands, group in zip(files, groups):
				read_depth = None
				if args.read_depth:
					read_depth = sum(int(values.sum()) for counts in strands.values()
						for _, values in counts.values())
					if args.verbose:
						print("Dividing {} through by a read depth of {}.".format(
							group[0], read_depth))

				_write_counts(writer, group, strands, chrom_sizes, args.scale_factor,
					read_depth)
	except (ValueError, TypeError) as e:
		raise SystemExit("{}: {}".format(prog, e))
