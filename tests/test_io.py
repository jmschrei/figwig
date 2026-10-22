import errno
import os

import numpy
import pytest

from figwig._io import _native_reads, _pread_blocks, _read_block_runs


def test_positional_runs_and_eof(tmp_path):
	path = tmp_path / 'bytes'
	path.write_bytes(bytes(range(100)))
	fd = os.open(path, os.O_RDONLY)
	try:
		data = numpy.zeros(20, dtype=numpy.uint8)
		got = _pread_blocks(fd, data, numpy.array([0, 10]), numpy.array([10, 20]),
			numpy.array([30, 95], dtype=numpy.int64))
		assert got.tolist() == [1, 0]
		assert data[:10].tolist() == list(range(30, 40))
		assert data[10:15].tolist() == list(range(95, 100))
	finally:
		os.close(fd)


def test_native_read_error():
	if _native_reads() is None:
		pytest.skip('native positional reads are unavailable')
	with pytest.raises(OSError) as caught:
		_pread_blocks(-1, numpy.empty(1, dtype=numpy.uint8), numpy.array([0]),
			numpy.array([1]), numpy.array([0]))
	assert caught.value.errno == errno.EBADF


def test_native_loop_retries_interruptions_and_short_reads():
	"""Exercise the loop independently of the OS's usual full reads."""

	calls = []
	def read(fd, destination, size, offset):
		calls.append((size, offset))
		return -1 if len(calls) == 1 else min(3, size)

	got = numpy.zeros(1, dtype=numpy.int64)
	function = getattr(_read_block_runs, 'py_func', _read_block_runs)
	assert function(read, lambda: [errno.EINTR], 1,
		numpy.empty(10, dtype=numpy.uint8), numpy.array([0]), numpy.array([10]),
		numpy.array([20]), got) == 0
	assert got.tolist() == [1]
	assert calls == [(10, 20), (10, 20), (7, 23), (4, 26), (1, 29)]
