# writers.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""bigWig writers for the tests.

`write_raw_bigwig` lays a bigWig out exactly as given, including layouts
that pybigtools' writer refuses: varStep and fixedStep sections,
uncompressed blocks, overlapping or unsorted intervals, and raw bytes as a
block. `write_bigwig` writes an ordinary bigWig with pybigtools, which the
tests also use as the reference reader.
"""

import zlib
import struct

import pybigtools


def write_bigwig(path, chroms, entries):
	"""Write `entries`, (chrom, start, end, value) tuples, with pybigtools."""

	bw = pybigtools.open(str(path), 'w')
	bw.write(chroms, iter(entries))


def write_raw_bigwig(path, chroms, sections, compress=True,
	chrom_block_size=None):
	# A bigWig laid out exactly as given, for the sections pybigtools does not
	# write. Each section is one data block: (chrom, kind, step, span, items),
	# where bedGraph (kind 1) items are (start, end, value), varStep (2) are
	# (start, value), and fixedStep (3) is (start, [values]). A block given as
	# bytes is written as it is, under the index entry (chrom, start, end).
	# With chrom_block_size smaller than the number of chromosomes, the
	# chromosome tree is a root over leaves of that many chromosomes, as UCSC
	# writes it for a genome with more than 256 chromosomes.
	names = list(chroms)
	ids = {name: i for i, name in enumerate(names)}
	key_size = max(len(name) for name in names)

	blocks = []
	for section in sections:
		if isinstance(section[-1], bytes):
			chrom, start, end, raw = section
			blocks.append((ids[chrom], start, end, raw, False))
			continue

		chrom, kind, step, span, items = section
		if kind == 1:
			body = b''.join(struct.pack('<IIf', s, e, v) for s, e, v in items)
			n, start = len(items), min(s for s, _, _ in items)
			end = max(e for _, e, _ in items)
		elif kind == 2:
			body = b''.join(struct.pack('<If', s, v) for s, v in items)
			n, start = len(items), min(s for s, _ in items)
			end = max(s for s, _ in items) + span
		else:
			first, values = items
			body = b''.join(struct.pack('<f', v) for v in values)
			n, start, end = len(values), first, first + step * (len(values) -
				1) + span

		header = struct.pack('<IIIIIBBH', ids[chrom], start, end, step, span,
			kind, 0, n)
		blocks.append((ids[chrom], start, end, header + body, compress))

	ctree_offset = 64 + 40
	block = chrom_block_size or len(names)
	leaves = []
	for k in range(0, len(names), block):
		leaf = struct.pack('<BBH', 1, 0, len(names[k:k + block]))
		for name in names[k:k + block]:
			leaf += name.encode().ljust(key_size, b'\0') + struct.pack('<II',
				ids[name], chroms[name])
		leaves.append((names[k], leaf))

	ctree = struct.pack('<IIIIQQ', 0x78CA8C91, block, key_size, 8, len(names), 0)
	if len(leaves) == 1:
		ctree += leaves[0][1]
	else:
		position = ctree_offset + len(ctree) + 4 + len(leaves) * (key_size + 8)
		ctree += struct.pack('<BBH', 0, 0, len(leaves))
		for first, leaf in leaves:
			ctree += first.encode().ljust(key_size, b'\0') + struct.pack('<Q',
				position)
			position += len(leaf)
		ctree += b''.join(leaf for _, leaf in leaves)

	data_offset = ctree_offset + len(ctree)
	data, leaves, largest = struct.pack('<Q', len(blocks)), [], 0
	for chrom, start, end, raw, packed in blocks:
		payload = zlib.compress(raw) if packed else raw
		largest = max(largest, len(raw))
		leaves.append((chrom, start, chrom, end, data_offset + len(data),
			len(payload)))
		data += payload

	index_offset = data_offset + len(data)
	rtree = struct.pack('<IIQIIIIQII', 0x2468ACE0, len(leaves), len(leaves),
		leaves[0][0], leaves[0][1], leaves[-1][2], leaves[-1][3],
		index_offset, 1, 0) + struct.pack('<BBH', 1, 0, len(leaves))
	for leaf in leaves:
		rtree += struct.pack('<IIIIQQ', *leaf)

	header = struct.pack('<IHHQQQHHQQIQ', 0x888FFC26, 4, 0, ctree_offset,
		data_offset, index_offset, 0, 0, 0, 64, largest if compress else 0, 0)
	with open(path, 'wb') as handle:
		handle.write(header + struct.pack('<Qdddd', 0, 0, 0, 0, 0) + ctree +
			data + rtree)


