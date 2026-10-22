# test_isal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import sys
import zlib

import numpy
import pytest

import figwig
import figwig._isal

from figwig.bigwig import _ZlibDecoder
from figwig.bigwig import _decoder


@pytest.fixture
def fresh_functions():
	"""Look ISA-L's functions up again in this test, and again after it."""

	figwig._isal._isal_functions.cache_clear()
	yield
	figwig._isal._isal_functions.cache_clear()


@pytest.fixture
def small_bw(tmp_path):
	path = tmp_path / 'a.bw'
	values = numpy.arange(20000, dtype=numpy.float32).reshape(20, 1000)
	values[0, :2] = [numpy.nan, -0.0]
	figwig.write_bigwig(path, {'chr1': 20000}, 'chr1', numpy.arange(20) * 1000,
		values, zooms=0, engine='zlib')
	return path


@pytest.mark.parametrize('n_jobs', [1, 3])
@pytest.mark.parametrize('from_numba', [True, False])
def test_isal_values_and_output(small_bw, monkeypatch, n_jobs, from_numba):
	"""ISA-L gives zlib's values, called from numba or, where it cannot be,
	through python-isal's Python API."""

	pytest.importorskip('isal')
	if from_numba and figwig._isal._isal_functions() is None:
		pytest.skip('ISA-L cannot be called from numba here')
	if not from_numba:
		monkeypatch.setattr(figwig._isal, '_isal_functions', lambda: None)

	reader = figwig.BigWigReader(small_bw)
	starts = [15000, 0, 999, 0, 19950]
	expected = reader.read('chr1', starts, 100, n_jobs=n_jobs, engine='zlib')
	out = numpy.empty_like(expected)
	assert reader.read('chr1', starts, 100, out=out, n_jobs=n_jobs,
		engine='isal') is out
	numpy.testing.assert_array_equal(expected.view('u4'), out.view('u4'))
	both = figwig.read_bigwig([reader, small_bw], 'chr1', starts, 100,
		n_jobs=n_jobs, engine='isal')
	numpy.testing.assert_array_equal(both[:, 0].view('u4'), expected.view('u4'))
	numpy.testing.assert_array_equal(both[:, 1].view('u4'), expected.view('u4'))


def test_auto_uses_isal_where_it_can_be_called():
	pytest.importorskip('isal')
	if figwig._isal._isal_functions() is None:
		pytest.skip('ISA-L cannot be called from numba here')

	decoder = _decoder('auto')
	assert isinstance(decoder, figwig._isal._IsalDecoder)
	assert decoder.functions is not None


def test_auto_without_isal_uses_zlib(monkeypatch, fresh_functions):
	monkeypatch.setitem(sys.modules, 'isal', None)
	assert isinstance(_decoder('auto'), _ZlibDecoder)
	with pytest.raises(ImportError, match=r'figwig\[isal\]'):
		_decoder('isal')


def test_unchecked_isal_version(monkeypatch, fresh_functions):
	"""An ISA-L whose layout has not been checked is not called from numba:
	the default uses zlib, and engine='isal' python-isal's Python API, which
	falls back to zlib for what ISA-L cannot inflate."""

	isal = pytest.importorskip('isal')
	monkeypatch.setattr(isal, 'ISAL_VERSION', 'unknown')
	assert figwig._isal._isal_functions() is None
	assert isinstance(_decoder('auto'), _ZlibDecoder)

	decoder = _decoder('isal')
	assert decoder.functions is None
	assert decoder.decompress(zlib.compress(b'ACGT')) == b'ACGT'
	with pytest.raises(zlib.error):
		decoder.decompress(b'corrupt')


def test_unknown_reader_engine(small_bw):
	with pytest.raises(ValueError, match="engine must be 'auto', 'zlib' or 'isal'"):
		figwig.read_bigwig(small_bw, 'chr1', [0], 1, engine='unknown')
