import numpy
import pytest
import zlib

from figwig._cuda_inflate import _uncompress2, _validate
from .test_cuda_validation import block


def test_native_validation_checks_checksums_sections_and_consumed_bytes():
	native = _uncompress2()
	if native is None:
		pytest.skip('zlib uncompress2 is unavailable')
	function, ulong = native
	good = zlib.compress(block())
	raws = [good, good + b'trailing bytes', good[:-1],
		good[:-1] + bytes([good[-1] ^ 1]), zlib.compress(block(4)), b'']
	sizes = numpy.array([len(raw) for raw in raws], dtype=numpy.uint64)
	starts = numpy.cumsum(sizes) - sizes
	data = numpy.frombuffer(b''.join(raws), dtype=numpy.uint8)
	output = numpy.empty(len(raws) * 64, dtype=numpy.uint8)
	metadata = numpy.zeros((len(raws), 4), dtype=numpy.int64)
	_validate(function, data, starts, sizes, numpy.zeros(len(raws), dtype='i8'),
		numpy.tile([100, 110], (len(raws), 1)), output, 64, metadata,
		numpy.zeros(1, dtype=ulong), numpy.zeros(1, dtype=ulong))
	assert metadata[:2, 0].tolist() == [len(block()), len(block())]
	assert numpy.all(metadata[2:, 0] <= 0)
	assert metadata[:2, 3].tolist() == [len(good)-4] * 2
	assert output[:len(block())].tobytes() == block()
	assert output[64:64+len(block())].tobytes() == block()


@pytest.mark.parametrize('native', [True, False])
def test_native_payload_alignment(native):
	from figwig._cuda_inflate import _pack, _pack_loop
	data = numpy.arange(30, dtype=numpy.uint8)
	out = numpy.zeros(32, dtype=numpy.uint8)
	(_pack if native else _pack_loop)(data, numpy.array([2, 14], dtype='i8'),
		numpy.array([7, 5], dtype='u8'), numpy.array([0, 16], dtype='u8'), out)
	numpy.testing.assert_array_equal(out[:7], data[2:9])
	numpy.testing.assert_array_equal(out[16:21], data[14:19])
	assert numpy.all(out[7:16] == 0)
