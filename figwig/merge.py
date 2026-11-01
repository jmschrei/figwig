# merge.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig merge` command: the tracks of several figwig files in one.

The files must have been written with the same chromosomes, tile,
compression level and order, and their track names must differ; their
chunks are copied as they are, so merging costs a read and a write of the
bytes and no decoding. `figwig reorder` writes a file again with another
tile, level or order first. The merged file has a regional summary index,
built over it, when the header of the first file announces one.
"""

import argparse
import os
import sys

from . import _summary_index
from ._tables import _exit
from ._tables import _refuse_existing
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
	parser.add_argument('--summary-index', action=argparse.BooleanOptionalAction,
		default=None, help="Write the optional regional summary index, which "
		"lets a reader decode the tile sums of one chromosome or interval "
		"without inflating a whole level (--no-summary-index leaves it out). "
		"Default is to write one when the first file's header announces one.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


def _merge(paths, output, genome=None, verbose=False, summary_index=None):
	"""Write the tracks of the figwig files `paths` into `output`, with a
	regional summary index when `summary_index` is True, or None and the first
	file's header announces one."""

	first = FigWigReader(paths[0])
	if genome is None:
		# The files that name a genome must agree; one that names none
		# takes the others'.
		genomes = {FigWigReader(path).genome for path in paths} - {None}
		if len(genomes) > 1:
			raise ValueError("the files name different genomes, {}; give -g to "
				"say which the merged file is on.".format(', '.join(repr(g) for g in
				sorted(genomes))))
		genome = genomes.pop() if genomes else None
	if summary_index is None:
		summary_index = bool(first._optional & _summary_index.OPTIONAL_BIT)
	with FigWigWriter(output, first.chrom_sizes, None, genome=genome,
			attributes=first.attributes, tile=first.tile, level=first._level,
			order=first.order, zooms=len(first.zooms),
			summary_index=summary_index) as writer:
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

	try:
		if not args.force:
			_refuse_existing(args.output, '-f', besides=args.figwig)
		_merge(args.figwig, args.output, genome=args.genome,
			verbose=args.verbose, summary_index=args.summary_index)
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e, **{'this writer has': 'the first file has',
			'the writer lacks': 'the first file lacks', 'the writer has':
			'the first file has', 'in the writer': 'in the first file'})
