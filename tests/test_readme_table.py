# test_readme_table.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""The renderer of benchmarks/readme_table: number formats, the table, and the
rewrite of the README between its markers."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	'benchmarks', 'readme_table'))

import render  # noqa: E402
from common import item  # noqa: E402


@pytest.mark.parametrize('value,text', [(0.103, '0.103 s'), (0.00108, '1.08 ms'), (143.2, '143 s'),
	(26.93, '26.9 s'), (0.0991, '99.1 ms'), (0.5, '0.5 s'), (8.0, '8 s'), (1234.0, '1,230 s'), (0.0712, '71.2 ms')])
def test_seconds(value, text):
	assert render.seconds(value) == text


@pytest.mark.parametrize('value,text', [(35.74, '35.7 MB'), (148.4, '148 MB'), (792.0, '792 MB'),
	(26600.0, '26.6 GB'), (1110.0, '1.11 GB')])
def test_megabytes(value, text):
	assert render.megabytes(value) == text


def test_cell_bolds_the_smaller_and_labels():
	text = render.cell([item(0.5, 1.25, 's', baseline_name='pybigtools'),
		item(13.6, 0.015, 's', label='one cell')])
	assert text == '**0.5 s** vs 1.25 s (pybigtools); one cell: 13.6 s vs 15 ms'


def test_cell_without_a_baseline():
	text = render.cell([item(76.0, None, 's', baseline_text='334 s, written by figwig')])
	assert text == '76 s vs 334 s, written by figwig'


def results():
	row = {'setting': 'one track', 'baseline': 'bigWig', 'read': [item(0.1, 1.0)],
		'write': [item(0.2, 4.0)], 'size': [item(35.7, 35.8, 'MB')], 'notes': ['A note.'],
		'date': '2026-11-11T10:00:00'}
	return {'meta': {'threads': 8, 'machine': {'cpu': 'CPU', 'cores': 256},
		'versions': {'figwig': '0.3.0'}, 'commit': 'abc1234'}, 'rows': {'bulk-1': row}}


def test_update_replaces_only_between_the_markers():
	readme = 'top\n{}\nOLDTEXT\n{}\nbottom\n'.format(render.START, render.END)
	new = render.update(readme, results())
	assert new.startswith('top\n') and new.endswith('\nbottom\n')
	assert 'OLDTEXT' not in new and '| one track | bigWig |' in new and 'A note.' in new
	assert render.update(new, results()) == new


def test_update_needs_the_markers():
	with pytest.raises(ValueError):
		render.update('no markers', results())
