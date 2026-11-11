# render.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The README's table from the rows' measurements, and the rewrite of the
README between its markers."""

import re

START = '<!-- readme-table:start -->'
END = '<!-- readme-table:end -->'
ORDER = ['bulk-1', 'bulk-48', 'bulk-1024', 'scatac', 'fiber', 'hic', 'schic']


def significant(value, digits=3):
	"""`value` to `digits` significant figures without an exponent."""

	if value == 0:
		return '0'
	text = '{:.{}g}'.format(value, digits)
	if 'e' in text:
		text = '{:f}'.format(float(text)).rstrip('0').rstrip('.')
	if abs(value) >= 10 ** digits and '.' not in text:
		return '{:,}'.format(int(float(text)))
	return text


def seconds(value):
	if value < 0.1:
		return '{} ms'.format(significant(value * 1000))
	return '{} s'.format(significant(value))


def megabytes(value):
	if value >= 1000:
		return '{} GB'.format(significant(value / 1000))
	return '{} MB'.format(significant(value))


def quantity(value, unit):
	return seconds(value) if unit == 's' else megabytes(value)


def cell(items):
	"""A table cell: each item as `figwig vs baseline`, figwig in bold where it
	is smaller, labelled where the item has a label, joined by semicolons."""

	parts = []
	for it in items:
		mine = quantity(it['figwig'], it['unit'])
		if it['baseline'] is None:
			theirs = it['baseline_text'] or 'not run'
			wins = False
		else:
			theirs = quantity(it['baseline'], it['unit'])
			wins = it['figwig'] < it['baseline']
		text = '{} vs {}'.format('**{}**'.format(mine) if wins else mine, theirs)
		if it['baseline_name']:
			text += ' ({})'.format(it['baseline_name'])
		if it['label']:
			text = '{}: {}'.format(it['label'], text)
		parts.append(text)
	return '; '.join(parts)


def table(rows):
	lines = ['| Setting | Baseline | Read | Write | Size |', '|---|---|---|---|---|']
	for key in ORDER:
		if key not in rows:
			continue
		r = rows[key]
		lines.append('| {} | {} | {} | {} | {} |'.format(r['setting'], r['baseline'],
			cell(r['read']), cell(r['write']), cell(r['size'])))
	return '\n'.join(lines)


def block(results):
	"""The whole `<details>` block, between the markers."""

	rows = results['rows']
	meta = results['meta']
	machine = meta['machine']
	notes = [n for key in ORDER if key in rows for n in rows[key].get('notes', [])]
	date = max(r['date'] for r in rows.values())[:10]
	lines = [START, '<details>',
		'<summary>Read time, write time and file size by assay, against what most people use today</summary>',
		'',
		'Each cell is figwig against the baseline, figwig first, in bold where it is smaller. figwig ran '
		'on {threads} threads unless a note says otherwise; each baseline is given its fastest setting (the other tools mostly do not '
		'gain from threads). Reads are windows of 2,048 bp unless noted.'.format(threads=meta['threads']),
		'', table(rows), '']
	if notes:
		lines += [' '.join(notes), '']
	lines += ['Measured by `benchmarks/readme_table/run.py` on {date} on {cpu} ({cores} cores), figwig {fw} '
		'at {commit}; the numbers are in `benchmarks/readme_table/results.json`.'.format(date=date,
		cpu=machine['cpu'], cores=machine['cores'], fw=meta['versions'].get('figwig', '?'),
		commit=meta.get('commit') or '?'), '', '</details>', END]
	return '\n'.join(lines)


def update(readme, results):
	"""`readme` (text) with the block between its markers replaced."""

	pattern = re.compile(re.escape(START) + '.*?' + re.escape(END), re.S)
	if not pattern.search(readme):
		raise ValueError('the README has no {} ... {} markers'.format(START, END))
	return pattern.sub(lambda m: block(results), readme, count=1)
