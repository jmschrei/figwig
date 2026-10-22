# merge.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig merge` command: the tracks of several figwig files in one.

The files must have been written with the same chromosomes, tile,
compression level and order, and their track names must differ; their
chunks are copied as they are, so merging costs a read and a write of the
bytes and no decoding. `figwig reorder` writes a file again with another
tile, level or order first.
"""

import argparse
import os
import sys

from .figwig import FigWigReader
from .figwig import FigWigWriter


def _parser(prog):
	parser = argparse.ArgumentParser(prog=prog, description="Write the tracks "
		"of several figwig files, written with the same chromosomes, tile, "
		"level and order, into one file, copying their chunks.")
	parser.add_argument('figwig', nargs='+', help="The figwig files, in the "
		"order their tracks take.")
	parser.add_argument('-o', '--output', required=True, help="The figwig file "
		"to write, which may be one of the inputs.")
	parser.add_argument('-g', '--genome', default=None, help="The reference "
		"genome to record. Default is the first file's.")
	parser.add_argument('-f', '--force', action='store_true', help="Write "
		"over a figwig file at the output other than one of the inputs, which "
		"is refused without it.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


def _merge(paths, output, genome=None, verbose=False):
	"""Write the tracks of the figwig files `paths` into `output`."""

	first = FigWigReader(paths[0])
	with FigWigWriter(output, first.chrom_sizes, None, genome=first.genome if
			genome is None else genome, attributes=first.attributes,
			tile=first.tile, level=first._level, order=first.order) as writer:
		for path in paths:
			writer.add_file(path)
			if verbose:
				print("{}: added {} tracks of {}".format(output, len(FigWigReader(
						path).tracks), path), file=sys.stderr)


def main(argv=None, prog='figwig merge'):
	args = _parser(prog).parse_args(argv)
	for path in args.figwig:
		if not os.path.isfile(path):
			raise SystemExit("{}: {} is not a file.".format(prog, path))
	if os.path.exists(args.output) and not args.force and not any(
			os.path.samefile(args.output, path) for path in args.figwig):
		raise SystemExit("{}: {} exists; -f writes over it.".format(prog,
			args.output))

	try:
		_merge(args.figwig, args.output, genome=args.genome,
			verbose=args.verbose)
	except (ValueError, TypeError, OSError) as e:
		raise SystemExit("{}: {}".format(prog, str(e).replace('n_jobs', '-p')
			.replace('this writer has', 'the first file has').replace(
			"the writer lacks", "the first file lacks").replace("the writer has",
			"the first file has").replace("in the writer", "in the first file")))
