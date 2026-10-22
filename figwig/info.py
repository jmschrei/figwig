# info.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig info` command: what a figwig file holds.

It prints the file's genome, order, tile, compression level, chromosomes
and size, then one line per track with its dtype, group, nonzero bases,
sum and attributes, as a tab-separated table.
"""

import argparse
import os
import sys

from .figwig import FigWigReader


def _parser(prog):
	parser = argparse.ArgumentParser(prog=prog, description="Print what a "
		"figwig file holds: its genome, order, tile, level, chromosomes and "
		"size, and a tab-separated table of its tracks.")
	parser.add_argument('figwig', help="The figwig file.")
	parser.add_argument('-b', '--bytes', action='store_true', help="Also "
		"count the bytes its chunks take and the bytes nothing uses, which "
		"reads every group's index.")
	return parser


def _describe(reader, count_bytes=False):
	"""The lines `figwig info` prints for `reader`."""

	size = os.path.getsize(reader.path)
	lines = [
		"{}".format(reader.path),
		"genome\t{}".format(reader.genome if reader.genome is not None else ''),
		"order\t{}".format(reader.order),
		"tile\t{}".format(reader.tile),
		"level\t{}".format(reader._level),
		"chromosomes\t{}\t{}".format(len(reader.chrom_sizes), ', '.join(
			list(reader.chrom_sizes)[:6]) + (', ...' if len(reader.chrom_sizes) >
			6 else '')),
		"groups\t{}".format(len(reader._groups)),
		"tracks\t{}".format(len(reader.tracks)),
		"bytes\t{}".format(size),
	]
	if count_bytes:
		live = 0
		for g in range(len(reader._groups)):
			index = reader._group_index(g)
			if reader.order == 'tracks':
				live += int(index['size'].astype('int64').sum())
			else:
				live += int(reader._runs[g]['bytes'].astype('int64').sum())
		tables = int(reader._file_size) - int(reader._tables_at)
		lines.append("chunk bytes\t{}".format(live))
		lines.append("table and index bytes\t{}".format(tables))
		lines.append("unused bytes\t{}".format(int(reader._file_size) - 128 - live
			- tables))
		if size > reader._file_size:
			lines.append("bytes past the end\t{}".format(size - int(
				reader._file_size)))
	for key, value in reader.attributes.items():
		lines.append("attribute\t{}\t{}".format(key, value))

	keys = []
	for track in reader.tracks:
		for key in track:
			if key not in ('name', 'dtype', 'group') and key not in keys:
				keys.append(key)
	lines.append('')
	lines.append('\t'.join(['name', 'dtype', 'group', 'nonzero', 'sum'] + keys))
	for i, track in enumerate(reader.tracks):
		stat = reader.stats[i]
		lines.append('\t'.join([track['name'], track['dtype'], str(track['group']),
			str(int(stat['nonzero'])), repr(float(stat['sum']))] + [str(track.get(
			key, '')) for key in keys]))
	return lines


def main(argv=None, prog='figwig info'):
	args = _parser(prog).parse_args(argv)
	if not os.path.isfile(args.figwig):
		raise SystemExit("{}: {} is not a file.".format(prog, args.figwig))

	try:
		reader = FigWigReader(args.figwig)
		for line in _describe(reader, args.bytes):
			print(line)
		sys.stdout.flush()
	except (ValueError, TypeError, OSError) as e:
		raise SystemExit("{}: {}".format(prog, e))
	except BrokenPipeError:
		# The reader of the output, such as `head`, has gone; say nothing.
		os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
		raise SystemExit(0)
