# _tables.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The metadata tables that `figwig bw2fw`, `figwig fw2bw` and `figwig
bam2fw` read and write: one row per input file, with a header, as tab- or
comma-separated text by the file's extension.

The first column names the file a row describes, as the path given on the
command line or its base name, and every other column is an attribute of
that file's tracks. A `name` column gives the track its name instead of the
file's base name, and a `dtype` column its dtype. Values are kept as the
strings they were.
"""

import csv
import os


def _delimiter(path):
	return ',' if str(path).endswith('.csv') else '\t'


def read_table(path):
	"""The rows of a metadata table, as (header, list of dicts), with the
	first column's header first."""

	with open(path, newline='') as handle:
		rows = list(csv.reader(handle, delimiter=_delimiter(path)))

	rows = [row for row in rows if len(row) > 0]
	if len(rows) == 0:
		raise ValueError("{} is empty; a metadata table needs a header row."
			.format(path))

	header = [column.strip() for column in rows[0]]
	if len(header) == 0 or len(set(header)) != len(header) or any(len(column)
			== 0 for column in header):
		raise ValueError("{}: the header must name each column once.".format(
			path))

	records = []
	for i, row in enumerate(rows[1:]):
		if len(row) != len(header):
			raise ValueError("{}, line {}: expected {} columns, found {}.".format(
				path, i + 2, len(header), len(row)))
		records.append({column: value.strip() for column, value in zip(header,
			row)})

	return header, records


def match_rows(path, records, header, filenames):
	"""The row of each input file, matched on the first column by the path
	given or its base name."""

	by_key = {}
	for record in records:
		key = record[header[0]]
		by_key[key] = record
		by_key.setdefault(os.path.basename(key), record)

	rows = []
	for filename in filenames:
		record = by_key.get(filename, by_key.get(os.path.basename(filename)))
		if record is None:
			raise ValueError("{} has no row for {} in its first column, {!r}."
				.format(path, filename, header[0]))
		rows.append(record)

	return rows


def write_table(path, header, records):
	"""Write rows as a metadata table, with the columns in `header`."""

	with open(path, 'w', newline='') as handle:
		writer = csv.writer(handle, delimiter=_delimiter(path),
			lineterminator='\n')
		writer.writerow(header)
		for record in records:
			writer.writerow([record.get(column, '') for column in header])


def track_name(filename):
	"""A track's name from its bigWig's path: the base name without a
	.bw, .bigWig or .bigwig extension."""

	name = os.path.basename(filename)
	for extension in ('.bw', '.bigWig', '.bigwig'):
		if name.endswith(extension):
			return name[:-len(extension)]

	return name
