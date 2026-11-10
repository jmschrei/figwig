# fiber_reads.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Fiber-seq reads built from an explicit layout, and what each should write.

A read is a molecule of random A, C, G and T, aligned by a CIGAR that may
carry soft clips, insertions and deletions on either strand. The calls are
chosen on the molecule, written into MM and ML by an encoder here, and the
bases they should land on are found by walking the molecule base by base,
which shares nothing with how `figwig fiberbam2fw` counts skips. The
nucleosomes, MSPs and FIREs are intervals of the molecule written as ns,
nl, as, al and aq.
"""

import array

import numpy
import pysam

CHROM_SIZES = [("chr1", 6000), ("chr2", 4000)]

_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def reverse_complement(sequence):
	return "".join(_COMPLEMENT[base] for base in reversed(sequence))


def header(sample="K562", groups=("rg1",)):
	return pysam.AlignmentHeader.from_dict({
		"HD": {"VN": "1.6", "SO": "coordinate"},
		"SQ": [{"SN": name, "LN": size} for name, size in CHROM_SIZES],
		"RG": [{"ID": group, "SM": sample} for group in groups]})


def query_length(cigar):
	"""The bases of SEQ a CIGAR consumes (hard clips excluded) and the
	hard-clipped bases on its left and right."""

	n, left, right = 0, 0, 0
	for k, (length, op) in enumerate(cigar):
		if op in "MIS=X":
			n += length
		elif op == "H":
			if k == 0:
				left += length
			else:
				right += length
	return n, left, right


def cigar_string(cigar):
	return "".join("{}{}".format(length, op) for length, op in cigar)


def reference_positions(start, cigar):
	"""The reference base of each base of SEQ, None for an insertion or a
	soft clip, found by walking the CIGAR."""

	out, r = [], start
	for length, op in cigar:
		for _ in range(length):
			if op in "M=X":
				out.append(r)
				r += 1
			elif op in "IS":
				out.append(None)
			elif op in "DN":
				r += 1
	return out


def encode_mm(molecule, calls):
	"""MM and ML for calls on a molecule, as the tag's blocks.

	`calls` is a list of (base, strand, code, mode, {index: probability})
	blocks; each index is a position of the molecule holding `base`. The
	skip counts are the bases of that kind passed over since the last
	call, and ML lists the probabilities block by block.
	"""

	blocks, ml = [], []
	for base, strand, code, mode, probabilities in calls:
		places = [j for j, b in enumerate(molecule) if b == base]
		skips, last = [], 0
		for j in sorted(probabilities):
			here = places.index(j)
			skips.append(here - last)
			last = here + 1
			ml.append(probabilities[j])
		blocks.append("{}{}{}{},{}".format(base, strand, code, mode,
			",".join(str(s) for s in skips)).rstrip(","))
	return ";".join(blocks) + ";", ml


class Read:
	"""One alignment: where it is, its molecule and its tags."""

	def __init__(self, name, tid, start, cigar, reverse=False, flag=None,
			mapq=60, seed=0, mods=None, features=None, tags=None, molecule=None,
			mm=None, ml=None, rg="rg1", write_mm=True):
		rng = numpy.random.default_rng(seed)
		self.name, self.tid, self.start, self.cigar = name, tid, start, cigar
		self.reverse, self.mapq, self.rg = reverse, mapq, rg
		self.flag = (16 if reverse else 0) if flag is None else flag
		n, self.left, self.right = query_length(cigar)
		self.full = n + self.left + self.right
		self.molecule = molecule or "".join(rng.choice(list("ACGT"), self.full))
		self.tags = dict(tags or {})
		self.features = features or {}
		self.calls = []
		if mm is not None:
			self.mm, self.ml = mm, ml
		elif write_mm:
			self.calls = self._draw(rng, mods or ("a", "t", "m"))
			self.mm, self.ml = encode_mm(self.molecule, self.calls)
		else:
			self.mm = self.ml = None
		self.sequence = self.molecule
		if reverse:
			self.sequence = reverse_complement(self.molecule)
		if self.left or self.right:
			self.sequence = self.sequence[self.left:len(self.sequence) - self.right]

	def _draw(self, rng, which):
		blocks = []
		for key, (base, strand, code) in (("a", ("A", "+", "a")),
				("t", ("T", "-", "a")), ("m", ("C", "+", "m"))):
			if key not in which:
				continue
			places = [j for j, b in enumerate(self.molecule) if b == base]
			chosen = rng.choice(places, size=max(len(places) * 3 // 4, 1), replace=False)
			values = rng.integers(0, 256, size=len(chosen))
			values[: min(3, len(values))] = (0, 255, 125)[: min(3, len(values))]
			blocks.append((base, strand, code, "?" if key == "m" else ".", {
				int(j): int(v) for j, v in zip(chosen, values)}))
		return blocks

	def segment(self, hdr):
		segment = pysam.AlignedSegment(hdr)
		segment.query_name = self.name
		segment.flag = self.flag
		segment.reference_id = self.tid
		segment.reference_start = self.start
		segment.mapping_quality = self.mapq
		segment.cigarstring = cigar_string(self.cigar)
		segment.query_sequence = self.sequence
		if self.mm is not None:
			segment.set_tag("MM", self.mm, "Z")
		if self.ml is not None:
			segment.set_tag("ML", array.array("B", self.ml))
		for key, (start, length) in self.features.items():
			if key == "nuc":
				segment.set_tag("ns", array.array("I", start))
				segment.set_tag("nl", array.array("I", length))
			elif key == "msp":
				segment.set_tag("as", array.array("I", start))
				segment.set_tag("al", array.array("I", length))
			elif key == "aq":
				segment.set_tag("aq", array.array("B", start))
		segment.set_tag("RG", self.rg, "Z")
		for key, value in self.tags.items():
			segment.set_tag(key, value)
		return segment

	def counts(self):
		"""{channel: (calls in MM, calls on a base with no reference base)}."""

		refs = reference_positions(self.start, self.cigar)
		length = len(self.sequence)
		out = {"m6a": [0, 0], "cpg": [0, 0]}
		for base, strand, code, mode, probabilities in self.calls:
			for j in probabilities:
				s = (self.full - 1 - j if self.reverse else j) - self.left
				out["m6a" if code == "a" else "cpg"][0] += 1
				if not 0 <= s < length or refs[s] is None:
					out["m6a" if code == "a" else "cpg"][1] += 1
		return out

	def expected(self):
		"""{channel: {reference base: value}} from the molecule: each call at
		its base of the molecule and each footprint base by base."""

		refs = reference_positions(self.start, self.cigar)
		length = len(self.sequence)

		def place(m):
			s = (self.full - 1 - m if self.reverse else m) - self.left
			if 0 <= s < length:
				return refs[s]
			return None

		out = {"m6a": {}, "cpg": {}, "nuc": {}, "msp": {}, "fire": {}}
		for base, strand, code, mode, probabilities in self.calls:
			channel = "m6a" if code == "a" else "cpg"
			for j, p in probabilities.items():
				r = place(j)
				if r is not None and p > 0:
					out[channel][r] = p
		quality = self.features.get("aq", (None, None))[0]
		for key in ("nuc", "msp"):
			starts, lengths = self.features.get(key, ([], []))
			for k, (st, ln) in enumerate(zip(starts, lengths)):
				placed = [r for r in (place(m) for m in range(st, st + ln)) if r is not None]
				if not placed:
					continue
				for r in range(min(placed), max(placed) + 1):
					out[key][r] = 1
				if key == "msp" and quality is not None and quality[k] > 0:
					for r in range(min(placed), max(placed) + 1):
						out["fire"][r] = min(int(quality[k]), 255)
		return out


def write_reads(path, reads, hdr=None, sort=True):
	"""Write reads to a BAM, in coordinate order unless `sort` is False."""

	hdr = hdr or header()
	order = sorted(reads, key=lambda r: (r.tid, r.start)) if sort else reads
	with pysam.AlignmentFile(str(path), "wb", header=hdr) as handle:
		for read in order:
			handle.write(read.segment(hdr))
	return order


def intervals(rng, length, n, minimum=15, maximum=60, gap=10):
	"""Disjoint (start, length) intervals along a molecule of `length`."""

	starts, lengths, at = [], [], int(rng.integers(0, 40))
	for _ in range(n):
		ln = int(rng.integers(minimum, maximum))
		if at + ln > length:
			break
		starts.append(at)
		lengths.append(ln)
		at += ln + int(rng.integers(0, gap))
	return starts, lengths


def standard_reads():
	"""Reads on both strands with soft clips, insertions and deletions, that
	carry every channel."""

	cigars = [
		[(300, "M")],
		[(20, "S"), (150, "M"), (3, "D"), (200, "M"), (2, "I"), (100, "M"), (10, "S")],
		[(120, "M"), (4, "I"), (80, "M"), (6, "D"), (150, "M")],
		[(30, "S"), (250, "M"), (1, "D"), (120, "M")],
	]
	reads = []
	rng = numpy.random.default_rng(7)
	for k, (tid, start, reverse) in enumerate([
			(0, 100, False), (0, 160, True), (0, 160, False), (0, 700, True),
			(0, 2900, False), (1, 50, True), (1, 400, False), (1, 3000, True)]):
		cigar = cigars[k % len(cigars)]
		n, left, right = query_length(cigar)
		full = n + left + right
		nuc = intervals(rng, full, 8)
		msp = intervals(rng, full, 8, 5, 40, 30)
		quality = [int(q) for q in rng.choice([0, 0, 60, 243, 255], size=len(msp[0]))]
		reads.append(Read("read{}".format(k), tid, start, cigar, reverse=reverse,
			seed=k, features={"nuc": nuc, "msp": msp, "aq": (quality, None)},
			tags={"HP": 1 + k % 2, "rq": 0.99 - k / 1000}))
	return reads
