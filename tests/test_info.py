# test_info.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests of `figwig info`."""

import os
import sys
import subprocess

import pytest

from figwig import FigWigReader
from figwig.__main__ import main
from figwig.info import _describe

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
