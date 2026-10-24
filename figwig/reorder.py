# reorder.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig reorder` command: a figwig file written again in the other
order.

A file in position order reads windows across many tracks fast and a file
in track order reads a few tracks over the genome fast; this reads every
nonzero base of the input, chromosome by chromosome, and writes it in the
order asked for, with the same tracks, attributes and genome.
"""

import argparse
import os
import sys

from .figwig import _DEFAULT_GROUPS
from .figwig import _DEFAULT_TILE
from ._tables import _exit
from ._tables import _refuse_existing
from .figwig import FigWigReader
from .figwig import FigWigWriter


def _parser(prog):
	parser = argparse.ArgumentParser(prog=prog, description="Write a figwig "
		"file again in position order or in track order.")
	parser.add_argument('figwig', help="The figwig file to read.")
	parser.add_argument('output', help="The figwig file to write.")
	parser.add_argument('--order', required=True, choices=['positions', 'tracks',
		'csc', 'csr'], help="The order of the output: 'positions' (or 'csc') "
		"for reading windows across many tracks, 'tracks' (or 'csr') for "
		"reading a few tracks over the genome.")
	parser.add_argument('-t', '--tile', default=None, type=int, help="The "
		"bases per tile in position order. Default is the input's tile, or {} "
		"for an input in track order.".format(_DEFAULT_TILE))
	parser.add_argument('--groups', default=None, type=int, help="The "
		"consecutive tracks per group in position order. Default is the "
		"input's groups, or {} for an input in track order.".format(
		_DEFAULT_GROUPS))
	parser.add_argument('-l', '--level', default=None, type=int, help="The "
		"compression level, or 0 for uncompressed chunks. Default is the "
		"input's level when the output is in the same order, and otherwise 6 "
		"in position order and 0 in track order.")
	parser.add_argument('-p', '--parallel', default=1, type=int, help="The "
		"number of threads. Default is 1.")
	parser.add_argument('-f', '--force', action='store_true', help="Write "
		"over a figwig file at the output other than the input itself, which "
		"is refused without it.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


def _reorder(path, output, order, tile=None, groups=None, level=None, n_jobs=1,
	verbose=False):
	"""Write the figwig file `path` as `output` in `order`. The tile, the
	groups and, for an output in the input's order, the level default to the
	input's, so that a file written again keeps its layout; an input in
	track order gives the position-order defaults. The output may be the
	input's own path, which holds the input until the output is complete."""

	reader = FigWigReader(path)
	tracks = [{key: value for key, value in track.items() if key != 'group'}
		for track in reader.tracks]
	if reader.order == 'positions':
		if tile is None:
			tile = reader.tile
		if groups is None:
			groups = [[reader.tracks[i]['name'] for i in group.tolist()] for
				group in reader._groups]
		if level is None and order in ('positions', 'csc'):
			level = reader._level
	if tile is None:
		tile = _DEFAULT_TILE
	if groups is None:
		groups = _DEFAULT_GROUPS
	total = 0
	with FigWigWriter(output, reader.chrom_sizes, tracks, genome=reader.genome,
			attributes=reader.attributes, tile=tile, groups=groups, level=level,
			n_jobs=n_jobs, order=order, zooms=len(reader.zooms)) as writer:
		for chrom in reader.chrom_sizes:
			entries = reader.read_entries(chroms=[chrom], n_jobs=n_jobs)
			if len(entries.values) > 0:
				writer.write_sparse(chrom, entries.positions, entries.tracks,
					entries.values)
			total += len(entries.values)
	if verbose:
		print("{}: {} tracks in {} order, {:,} nonzero bases, {:.1f} MB.".format(
			output, len(tracks), order, total, os.path.getsize(output) / 1e6),
			file=sys.stderr)


def main(argv=None, prog='figwig reorder'):
	args = _parser(prog).parse_args(argv)
	if not os.path.isfile(args.figwig):
		raise SystemExit("{}: {} is not a file.".format(prog, args.figwig))

	try:
		if not args.force:
			_refuse_existing(args.output, '-f', besides=[args.figwig])
		_reorder(args.figwig, args.output, args.order, tile=args.tile,
			groups=args.groups, level=args.level, n_jobs=args.parallel,
			verbose=args.verbose)
	except (ValueError, TypeError, OSError) as e:
		raise _exit(prog, e)
