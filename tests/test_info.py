# test_info.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests of `figwig info`."""

import os
import sys
import subprocess

import numpy

import pytest

from figwig import __version__
from figwig import FigWigReader
from figwig import FigWigWriter
from figwig.__main__ import main
from figwig.figwig import _VERSION
from figwig.info import _describe
from figwig.info import _shown

from .test_figwig import TRACKS
from .test_figwig import reference_tracks
from .test_figwig import write_reference

ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1')


@pytest.fixture(scope='module')
def ref():
	return reference_tracks()


@pytest.fixture(scope='module')
def fw(tmp_path_factory, ref):
	path = str(tmp_path_factory.mktemp('info') / 'reference.fw')
	write_reference(path, ref)
	return path


def test_describe_lists_the_file_and_its_tracks(fw):
	reader = FigWigReader(fw)
	lines = _describe(reader, count_bytes=True)
	assert lines[0] == reader.path
	fields = {line.split('\t')[0]: line.split('\t')[1:] for line in lines[1:]
		if line and not line.startswith('name\t')}
	assert fields['genome'] == ['hg38'] and fields['order'] == ['positions']
	assert fields['version'] == [str(_VERSION)]
	assert fields['written by'] == ['figwig ' + __version__]
	assert fields['tile'] == [str(reader.tile)] and fields['level'] == ['6']
	assert fields['chromosomes'] == ['3', 'chr1, chr2, chrM']
	assert fields['tracks'] == [str(len(TRACKS))]
	assert int(fields['bytes'][0]) == os.path.getsize(fw)
	assert int(fields['chunk bytes'][0]) + int(fields['table and index bytes'][
		0]) + int(fields['unused bytes'][0]) + 128 == os.path.getsize(fw)
	assert int(fields['unused bytes'][0]) == 0
	header = lines[lines.index('') + 1].split('\t')
	assert header[:5] == ['name', 'dtype', 'group', 'nonzero', 'sum']
	assert 'assay' in header and 'strand' in header
	rows = [line.split('\t') for line in lines[lines.index('') + 2:]]
	assert [row[0] for row in rows] == [t['name'] for t in TRACKS]
	assert rows[0][header.index('assay')] == 'ATAC-seq'
	assert int(rows[0][3]) == int(reader.stats['nonzero'][0])
	assert float(rows[0][4]) == float(reader.stats['sum'][0])


def test_values_are_shown_on_one_printable_line(tmp_path):
	"""Control characters in a name or attribute are escaped, and a nested
	attribute is shown as JSON."""

	assert _shown('a\tb\nc\x1b[0m') == 'a\\x09b\\x0ac\\x1b[0m'
	assert _shown('caf\u00e9 \u2028') == 'caf\u00e9 \\u2028'
	assert _shown({'k': [1, 'x']}) == '{"k": [1, "x"]}'
	assert _shown(None) == 'null' and _shown(2.5) == '2.5'
	path = str(tmp_path / 'esc.fw')
	with FigWigWriter(path, {'chr1': 100}, [{'name': 'bad\nname', 'note': {
			'depth': 2}}], attributes={'nested': {'a': [1]}, 'line': 'x\ry'},
			tile=64) as writer:
		writer.write('chr1', [0], numpy.ones((1, 1, 10)))
	lines = _describe(FigWigReader(path), count_bytes=False)
	assert all('\n' not in line and '\r' not in line for line in lines)
	assert 'attribute\tnested\t{"a": [1]}' in lines
	assert 'attribute\tline\tx\\x0dy' in lines
	assert any(line.startswith('bad\\x0aname\t') and line.endswith(
		'{"depth": 2}') for line in lines)


def test_unused_bytes_after_an_append(tmp_path, ref):
	path = str(tmp_path / 'grown.fw')
	write_reference(path, ref)
	from figwig import FigWigWriter
	with FigWigWriter(path, None, ['more'], append=True) as writer:
		writer.write('chr1', [0], [[[1.0] * 10]])
	lines = _describe(FigWigReader(path), count_bytes=True)
	unused = [line for line in lines if line.startswith('unused bytes')][0]
	assert int(unused.split('\t')[1]) > 0
	lines = _describe(FigWigReader(path))
	assert not any(line.startswith('unused bytes') for line in lines)

	# Bytes past the end the header names, as a killed append leaves.
	with open(path, 'ab') as handle:
		handle.write(bytes(1000))
	lines = _describe(FigWigReader(path), count_bytes=True)
	past = [line for line in lines if line.startswith('bytes past the end')][0]
	assert past.split('\t')[1] == '1000'


def test_command_line(tmp_path, fw, capsys):
	main(['info', fw])
	out = capsys.readouterr().out
	assert out.startswith(FigWigReader(fw).path) and 'atac_plus\t' in out
	with pytest.raises(SystemExit, match="is not a file"):
		main(['info', str(tmp_path / 'none.fw')])
	with pytest.raises(SystemExit, match="is not a figwig file"):
		main(['info', __file__])
	result = subprocess.run([sys.executable, '-m', 'figwig', 'info', '-b', fw],
		env=ENV, capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert 'unused bytes\t0' in result.stdout
