# fw2bw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The `figwig fw2bw` command: a figwig file's tracks into bigWigs.

Each track, or each track selected by its attributes, becomes one bigWig
named after the track, and a metadata table beside them records each
track's name, dtype and attributes, so that `figwig bw2fw` can make the
figwig file again. Tracks are read a group at a time, each chromosome in
pieces, and their bases that are 0 are left out of the bigWigs, as
`BigWigWriter.write` leaves them out with `missing=0.0`; a NaN in a float32
track is left out too.
"""

import argparse
import os
import sys

from concurrent.futures import ThreadPoolExecutor

import numpy

from ._tables import write_table
from .figwig import FigWigReader
from .writer import BigWigWriter

_PIECE = 1 << 22
_WINDOW = 1 << 16


def _parser(prog):
	parser = argparse.ArgumentParser(prog=prog, description="Convert the "
		"tracks of a figwig file into bigWig files, one per track.")
	parser.add_argument('figwig', help="The figwig file.")
	parser.add_argument('-o', '--output', default='.', help="The directory to "
		"write the bigWigs into, as <track name>.bw, with a metadata table "
		"named after the figwig file. Default is the current directory.")
	parser.add_argument('-t', '--tracks', nargs='+', default=None, help="The "
		"tracks to write, by name. Default is every track.")
	parser.add_argument('--select', nargs='+', default=None, metavar='KEY=VALUE',
		help="Write the tracks whose attributes have these values, such as "
		"assay=ATAC-seq strand=+.")
	parser.add_argument('-z', '--zooms', default=0, type=int, help="The zoom "
		"levels to write in each bigWig. Default is 0.")
	parser.add_argument('-l', '--level', default=6, type=int, help="The "
		"compression level of the bigWigs. Default is 6.")
	parser.add_argument('-p', '--parallel', default=1, type=int, help="The "
		"number of threads to read and compress on. Default is 1.")
	parser.add_argument('-v', '--verbose', action='store_true')
	return parser


def selected_tracks(reader, names=None, select=None):
	"""The indices of the tracks to write: those named, those whose
	attributes match, or all."""

	if names is not None and select is not None:
		raise ValueError("give --tracks or --select, not both.")

	if names is not None:
		ids, _ = reader._select(list(names))
		return ids

	if select is not None:
		attributes = {}
		for item in select:
			if '=' not in item:
				raise ValueError("--select takes KEY=VALUE pairs, not {!r}.".format(
					item))
			key, value = item.split('=', 1)
			attributes[key] = value

		ids = reader.select(**attributes)
		if len(ids) == 0:
			raise ValueError("no track has {}.".format(' '.join(select)))
		return ids

	return numpy.arange(len(reader.tracks))


def bigwig_name(track):
	"""The file name of a track's bigWig: its name with path separators
	replaced, plus .bw."""

	name = track['name'].replace(os.sep, '_')
	if os.altsep is not None:
		name = name.replace(os.altsep, '_')
	return name + '.bw'


def convert(path, directory, ids, zooms=0, level=6, n_jobs=1, verbose=False):
	"""Write the tracks `ids` of the figwig file `path` as bigWigs in
	`directory`, and the metadata table. Returns the table's path."""

	reader = FigWigReader(path)
	os.makedirs(directory, exist_ok=True)
	chrom_sizes = list(reader.chrom_sizes.items())

	by_group = {}
	for i in ids.tolist():
		by_group.setdefault(int(reader._group_of[i]), []).append(i)

	# A group's tracks are read together and their bigWigs written at once,
	# one track per thread, each writer compressing on its share of the
	# threads.
	paths = {}
	pool = ThreadPoolExecutor(n_jobs) if n_jobs > 1 else None
	for g, members in by_group.items():
		writers = {}
		try:
			for i in members:
				paths[i] = os.path.join(directory, bigwig_name(reader.tracks[i]))
				writers[i] = BigWigWriter(paths[i], chrom_sizes, zooms=zooms,
					level=level, n_jobs=max(1, n_jobs // len(members)))

			for chrom, length in chrom_sizes:
				width = int(min(_WINDOW, length))
				for a in range(0, length, _PIECE):
					starts = numpy.arange(a, min(length, a + _PIECE), width)
					values = reader.read(chrom, starts, width, tracks=members,
						dtype='float32', n_jobs=n_jobs)
					if pool is None:
						for k, i in enumerate(members):
							writers[i].write(chrom, starts, values[:, k], missing=0.0)
					else:
						list(pool.map(lambda k: writers[members[k]].write(chrom,
							starts, values[:, k], missing=0.0), range(len(members))))
				if verbose:
					print("{}: group {}, {}".format(path, g, chrom), file=sys.stderr)

			for writer in writers.values():
				writer.close()
		except BaseException as error:
			for writer in writers.values():
				writer.__exit__(type(error), error, error.__traceback__)
			raise
	if pool is not None:
		pool.shutdown()

	keys = ['file', 'name', 'dtype']
	for i in ids.tolist():
		for key in reader.tracks[i]:
			if key not in keys and key != 'group':
				keys.append(key)

	records = []
	for i in ids.tolist():
		record = {key: _text(value) for key, value in reader.tracks[i].items()
			if key != 'group'}
		record['file'] = os.path.basename(paths[i])
		records.append(record)

	table = os.path.join(directory, os.path.splitext(os.path.basename(path))[0]
		+ '.tsv')
	write_table(table, keys, records)
	return table


def _text(value):
	"""An attribute as the table holds it: strings as they are, everything
	else as JSON would print it."""

	if isinstance(value, str):
		return value
	if isinstance(value, bool):
		return 'true' if value else 'false'
	if value is None:
		return ''
	return str(value)


def main(argv=None, prog='figwig fw2bw'):
	args = _parser(prog).parse_args(argv)
	if not os.path.isfile(args.figwig):
		raise SystemExit("{}: {} is not a file.".format(prog, args.figwig))

	try:
		reader = FigWigReader(args.figwig)
		ids = selected_tracks(reader, args.tracks, args.select)
		convert(args.figwig, args.output, ids, zooms=args.zooms, level=args.level,
			n_jobs=args.parallel, verbose=args.verbose)
	except (ValueError, TypeError) as e:
		raise SystemExit("{}: {}".format(prog, e))
