# test_fw2bw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig fw2bw`, which writes a figwig file's tracks as
bigWigs, and for the round trip back through `figwig bw2fw`."""

import os
import sys
import time
import threading
import subprocess

import numpy
import pytest

import figwig.fw2bw

from figwig import BigWigReader
from figwig import FigWigReader
from figwig import read_bigwig

from figwig.__main__ import main

from .test_figwig import CHROM_SIZES
from .test_figwig import reference_tracks
from .test_figwig import write_reference

ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
	MKL_NUM_THREADS="1")


@pytest.fixture(scope='session')
def fw(tmp_path_factory):
	path = str(tmp_path_factory.mktemp('fw') / 'reference.fw')
	write_reference(path, reference_tracks())
	return path


def expected_bigwig(ref, chrom, track):
	"""What a bigWig of this track reads as: the values, with NaN as 0 and
	float32 rounding."""

	values = numpy.nan_to_num(ref[chrom][track]).astype(numpy.float32)
	return values


def test_every_track_becomes_a_bigwig(tmp_path, fw):
	ref = reference_tracks()
	main(['fw2bw', fw, '-o', str(tmp_path / 'bw'), '-p', '2'])
	names = ['atac_plus', 'atac_minus', 'signal', 'negative', 'empty']
	assert sorted(os.listdir(tmp_path / 'bw')) == sorted([name + '.bw' for name
		in names] + ['reference.tsv'])

	for i, name in enumerate(names[:4]):
		path = str(tmp_path / 'bw' / (name + '.bw'))
		for chrom, length in CHROM_SIZES.items():
			y = read_bigwig(path, chrom, [0], length)[0]
			numpy.testing.assert_array_equal(y, expected_bigwig(ref, chrom, i))

	# A track without values gives a bigWig without data blocks, as pyBigWig
	# writes one, which reads as missing everywhere and converts back to an
	# empty track.
	empty = str(tmp_path / 'bw' / 'empty.bw')
	assert BigWigReader(empty).chrom_sizes == CHROM_SIZES
	assert (read_bigwig(empty, 'chr1', [0, 50], 100) == 0).all()
	main(['bw2fw', empty, '-o', str(tmp_path / 'empty.fw')])
	back = FigWigReader(str(tmp_path / 'empty.fw'))
	assert back.stats['nonzero'].tolist() == [0] and back.chrom_sizes == \
		CHROM_SIZES
	assert (back.read('chr1', [0], CHROM_SIZES['chr1']) == 0).all()

	with open(tmp_path / 'bw' / 'reference.tsv') as handle:
		lines = handle.read().splitlines()
	assert lines[0] == 'file\tname\tdtype\tassay\tstrand'
	assert lines[1] == 'atac_plus.bw\tatac_plus\tuint16\tATAC-seq\t+'
	assert lines[3] == 'signal.bw\tsignal\tfloat32\tDNase-seq\t'


def test_round_trip_through_bw2fw(tmp_path, fw):
	main(['fw2bw', fw, '-o', str(tmp_path / 'bw')])
	back = str(tmp_path / 'back.fw')
	bigwigs = [str(tmp_path / 'bw' / (name + '.bw')) for name in ('atac_plus',
		'atac_minus', 'signal', 'negative')]
	main(['bw2fw', '-o', back, '-m', str(tmp_path / 'bw' / 'reference.tsv'),
		'-t', '64', '-g', 'hg38'] + bigwigs)

	original, copy = FigWigReader(fw), FigWigReader(back)
	assert copy.chrom_sizes == original.chrom_sizes
	assert len(copy.tracks) == 4
	for a, b in zip(original.tracks, copy.tracks):
		assert a['name'] == b['name'] and a['dtype'] == b['dtype']
		assert a.get('assay') == b.get('assay')
		assert a.get('strand') == b.get('strand')

	for chrom, length in CHROM_SIZES.items():
		a = original.read(chrom, [0], length, tracks=[0, 1, 2, 3], dtype='float32')
		b = copy.read(chrom, [0], length, dtype='float32')
		numpy.testing.assert_array_equal(numpy.nan_to_num(a), b)


def test_tracks_and_select(tmp_path, fw):
	main(['fw2bw', fw, '-o', str(tmp_path / 'a'), '-t', 'signal', 'empty'])
	assert sorted(os.listdir(tmp_path / 'a')) == ['empty.bw', 'reference.tsv',
		'signal.bw']

	main(['fw2bw', fw, '-o', str(tmp_path / 'b'), '--select', 'assay=ATAC-seq'])
	assert sorted(os.listdir(tmp_path / 'b')) == ['atac_minus.bw',
		'atac_plus.bw', 'reference.tsv']

	main(['fw2bw', fw, '-o', str(tmp_path / 'c'), '--select', 'assay=ATAC-seq',
		'strand=-'])
	assert sorted(os.listdir(tmp_path / 'c')) == ['atac_minus.bw',
		'reference.tsv']

	with pytest.raises(SystemExit, match="not both"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '--select', 'a=b', '-t',
			'signal'])
	with pytest.raises(SystemExit, match="no track has"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '--select', 'assay=none'])
	with pytest.raises(SystemExit, match="KEY=VALUE"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '--select', 'assay'])
	with pytest.raises(SystemExit, match="not one of the file's tracks"):
		main(['fw2bw', fw, '-o', str(tmp_path / 'd'), '-t', 'nope'])
	with pytest.raises(SystemExit, match="is not a file"):
		main(['fw2bw', str(tmp_path / 'nope.fw'), '-o', str(tmp_path / 'd')])


def test_zooms_and_level(tmp_path, fw):
	main(['fw2bw', fw, '-o', str(tmp_path / 'z'), '-t', 'atac_plus', '-z', '2',
		'-l', '1'])
	import pybigtools
	bw = pybigtools.open(str(tmp_path / 'z' / 'atac_plus.bw'))
	assert len(bw.zooms()) >= 1


def test_bigwig_name_replaces_separators():
	assert figwig.fw2bw._bigwig_name({'name': 'a/b'}) == 'a_b.bw'
	assert figwig.fw2bw._bigwig_name({'name': 'plain'}) == 'plain.bw'


def test_text():
	assert figwig.fw2bw._text('x') == 'x'
	assert figwig.fw2bw._text(3) == '3'
	assert figwig.fw2bw._text(True) == 'true'
	assert figwig.fw2bw._text(None) == ''


def test_command_line(tmp_path, fw):
	result = subprocess.run([sys.executable, '-m', 'figwig', 'fw2bw', fw, '-o',
		str(tmp_path / 'cli'), '-v'], env=ENV, capture_output=True, text=True,
		timeout=600)
	assert result.returncode == 0, result.stderr
	assert 'group 0: ' in result.stderr and 'chrM' not in result.stderr
	assert len(os.listdir(tmp_path / 'cli')) == 6
	result = subprocess.run([sys.executable, '-m', 'figwig', 'fw2bw', fw, '-o',
		str(tmp_path / 'cli')], env=ENV, capture_output=True, text=True,
		timeout=600)
	assert result.returncode != 0 and 'exists; -f writes over it' in \
		result.stderr
	result = subprocess.run([sys.executable, '-m', 'figwig', 'fw2bw', fw, '-o',
		str(tmp_path / 'cli'), '-f'], env=ENV, capture_output=True, text=True,
		timeout=600)
	assert result.returncode == 0, result.stderr


def test_failed_conversion_leaves_no_bigwig(tmp_path, fw, monkeypatch):
	"""Each bigWig is written beside its name and moved there when
	complete, so a conversion that fails leaves neither it nor its partial
	file behind."""

	import figwig.writer

	def fail(self, *args, **kwargs):
		raise ValueError("stopped part way")

	monkeypatch.setattr(figwig.writer.BigWigWriter, 'write', fail)
	out = tmp_path / 'failed'
	with pytest.raises(SystemExit, match="stopped part way"):
		main(['fw2bw', fw, '-o', str(out)])
	assert sorted(os.listdir(out)) == []


@pytest.mark.parametrize('ahead', [1, 3])
def test_bigwigs_written_on_their_own_threads_are_one_threads(tmp_path, fw,
		monkeypatch, ahead):
	"""Each track's bigWig, written on a thread of its own with the pieces of
	its group in order, and up to `ahead` pieces at once, has the bytes one
	thread writes, over chromosomes of many pieces: no bigWig takes two
	writes at once, and none starts a piece `ahead` past the last its
	group's slowest bigWig finished, though that one is slowed."""

	monkeypatch.setattr(figwig.fw2bw, '_PIECE', 1 << 9)
	monkeypatch.setattr(figwig.fw2bw, '_WINDOW', 1 << 7)
	monkeypatch.setattr(figwig.fw2bw, '_AHEAD', ahead)
	lock = threading.Lock()
	active, leads = [], []

	class Watched(figwig.fw2bw.BigWigWriter):
		"""A bigWig writer that records its writes, its group's first one
		slowed."""

		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.started = self.finished = 0
			self.busy = False
			with lock:
				active.append(self)

		def write(self, *args, **kwargs):
			with lock:
				assert not self.busy, 'two writes at once'
				self.busy = True
				self.started += 1
				leads.append(self.started - min(w.finished for w in active))
				slow = self is active[0]
			if slow:
				time.sleep(0.01)
			try:
				return super().write(*args, **kwargs)
			finally:
				with lock:
					self.busy = False
					self.finished += 1

		def close(self):
			with lock:
				active.remove(self)
			return super().close()

	written = []
	for threads in ('1', '4'):
		out = tmp_path / ('bw' + threads)
		with monkeypatch.context() as patch:
			if threads != '1':
				patch.setattr(figwig.fw2bw, 'BigWigWriter', Watched)
			main(['fw2bw', fw, '-o', str(out), '-p', threads])
		written.append({name: (out / name).read_bytes() for name in os.listdir(
			out)})
	assert written[0] == written[1]
	assert len(written[0]) == 6
	assert 0 < max(leads) <= ahead


def test_bigwigs_of_a_group_of_more_tracks_than_threads_take_no_more(
		tmp_path, monkeypatch):
	"""A group of more tracks than -p has its bigWigs written at most -p at a
	time, and still more than one at a time, none taking two writes at once,
	with the bytes one thread writes."""

	path = str(tmp_path / 'one_group.fw')
	write_reference(path, reference_tracks(), groups=5)
	monkeypatch.setattr(figwig.fw2bw, '_PIECE', 1 << 9)
	monkeypatch.setattr(figwig.fw2bw, '_WINDOW', 1 << 7)
	lock = threading.Lock()
	writing, most = [0], [0]

	class Counted(figwig.fw2bw.BigWigWriter):
		"""A bigWig writer that counts the writes running at once."""

		busy = False

		def write(self, *args, **kwargs):
			with lock:
				assert not self.busy, 'two writes at once'
				self.busy = True
				writing[0] += 1
				most[0] = max(most[0], writing[0])
			time.sleep(0.005)
			try:
				return super().write(*args, **kwargs)
			finally:
				with lock:
					self.busy = False
					writing[0] -= 1

	written = []
	for threads in ('1', '2'):
		out = tmp_path / ('bw' + threads)
		with monkeypatch.context() as patch:
			if threads != '1':
				patch.setattr(figwig.fw2bw, 'BigWigWriter', Counted)
			main(['fw2bw', path, '-o', str(out), '-p', threads])
		written.append({name: (out / name).read_bytes() for name in os.listdir(
			out)})
	assert written[0] == written[1]
	assert len(written[0]) == 6
	assert most[0] == 2


def test_bigwigs_of_a_group_of_as_many_tracks_as_threads_run_ahead(tmp_path,
		monkeypatch):
	"""A group of as many tracks as -p writes each track's bigWig on a
	thread of its own: the others start a second piece while a slowed one
	is still on the first."""

	path = str(tmp_path / 'one_group.fw')
	write_reference(path, reference_tracks(), groups=5)
	monkeypatch.setattr(figwig.fw2bw, '_PIECE', 1 << 9)
	monkeypatch.setattr(figwig.fw2bw, '_WINDOW', 1 << 7)
	monkeypatch.setattr(figwig.fw2bw, '_AHEAD', 2)
	lock = threading.Lock()
	active, leads = [], []

	class Slowed(figwig.fw2bw.BigWigWriter):
		"""A bigWig writer that records how far it is ahead of the group's
		last, the group's first one slowed."""

		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)
			self.started = self.finished = 0
			with lock:
				active.append(self)

		def write(self, *args, **kwargs):
			with lock:
				self.started += 1
				leads.append(self.started - min(w.finished for w in active))
				slow = self is active[0]
			if slow:
				time.sleep(0.01)
			try:
				return super().write(*args, **kwargs)
			finally:
				with lock:
					self.finished += 1

	monkeypatch.setattr(figwig.fw2bw, 'BigWigWriter', Slowed)
	main(['fw2bw', path, '-o', str(tmp_path / 'bw'), '-p', '5'])
	assert max(leads) == 2
