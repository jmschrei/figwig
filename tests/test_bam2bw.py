# test_bam2bw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import gzip
import shutil
import subprocess
import sys
import importlib.util

import numpy
import pytest

from numpy.testing import assert_array_almost_equal

# pysam, which writes the inputs, and pyBigWig, which reads the bigWigs back,
# do not build on Windows.
if sys.platform == 'win32':
	pytest.skip("pysam and pyBigWig do not build on Windows",
		allow_module_level=True)

import pysam  # noqa: E402
import pyBigWig  # noqa: E402

import figwig  # noqa: E402
import figwig.bam2bw  # noqa: E402

from .reads import write_bam  # noqa: E402
from .reads import write_chrom_sizes  # noqa: E402
from .reads import write_fasta  # noqa: E402
from .reads import write_intervals  # noqa: E402
from .reads import write_paired_bam  # noqa: E402


BAM2BW = [sys.executable, "-m", "figwig", "bam2bw"]

# OpenBLAS starts a thread per core when numpy is imported, which on a large
# machine costs each run seconds of CPU; figwig bam2bw uses no BLAS.
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
	MKL_NUM_THREADS="1")


# Every test here runs the real command line, `figwig bam2bw`, in a
# subprocess and reads the bigWigs back. That is the contract users depend on:
# the flags, the two-or-one output files, and the numbers inside them. These
# are bam2bw's own tests, run against figwig's version of it.
#
# The read layout and the expected values it implies are documented below.
# Expected positions and counts are hardcoded below rather than
# recomputed, so that a change in the recording logic shows up as a failing
# test instead of as two matching re-derivations of the same mistake.


# The one read layout that almost every test in the suite runs against. It is
# deliberately small enough to compute expected bigWig values by hand, while
# still covering each branch the main loop can take:
#
#    - both strands, on more than one chromosome,
#    - positions hit by more than one read, so that counts have to accumulate
#      rather than overwrite,
#    - a chromosome in the sizes file that no read touches (chr3),
#    - a chromosome the reads touch that is not in the sizes file (chrUN),
#    - an unmapped read.
#
# The 5' end of a forward read is its reference start and its 3' end is
# reference_end - 1; for a reverse read the two are swapped. Working those out
# for the reads below gives, with no flags at all:
#
#    plus    chr1 {100: 2, 200: 1}    chr2 {50: 1}
#    minus   chr1 {339: 2, 419: 1}    chr2 {89: 1}
#
# for a total of eight counted reads, which is a power of two so that the
# read-depth normalized values stay exactly representable as float32 and can
# still be compared to four decimal places.

CHROM_SIZES = [("chr1", 1000), ("chr2", 500), ("chr3", 200)]
BAM_CHROM_SIZES = CHROM_SIZES + [("chrUN", 100)]

READS = [
	("chr1", 100, 50, False),
	("chr1", 100, 50, False),
	("chr1", 200, 30, False),
	("chr1", 300, 40, True),
	("chr1", 300, 40, True),
	("chr1", 400, 20, True),
	("chr2", 50, 25, False),
	("chr2", 80, 10, True),
	("chrUN", 10, 10, False)
]

# The unmapped read is placed on chr1, the way a read whose mate aligned is
# placed in a real BAM. It must be dropped by the unmapped check rather than
# by the missing-chromosome check, so that removing either one shows up.

UNMAPPED = [("chr1", 500, 20)]

# The interval files carry no strand, so `figwig bam2bw` treats every entry as being
# on the forward strand. The layout mirrors the forward-strand reads above so
# that the two input paths can be compared against each other.

INTERVALS = [
	("chr1", 100, 150),
	("chr1", 100, 150),
	("chr1", 200, 230),
	("chr2", 50, 75),
	("chrUN", 10, 20)
]

FASTA_EXTENSIONS = ".fa", ".fasta", ".fna", ".fas"


## Input files, built once for the whole session


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
	return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sizes(data_dir):
	return write_chrom_sizes(data_dir / "test.chrom.sizes", CHROM_SIZES)


@pytest.fixture(scope="session")
def fastas(data_dir):
	"""A FASTA of the same chromosomes under each accepted extension."""

	return {extension: write_fasta(data_dir / ("test" + extension), CHROM_SIZES)
		for extension in FASTA_EXTENSIONS}


@pytest.fixture(scope="session")
def bam(data_dir):
	return write_bam(data_dir / "test.bam", BAM_CHROM_SIZES, READS,
		unmapped=UNMAPPED)


@pytest.fixture(scope="session")
def bed(data_dir):
	return write_intervals(data_dir / "test.bed", INTERVALS)


@pytest.fixture(scope="session")
def bed_gz(data_dir):
	return write_intervals(data_dir / "test.bed.gz", INTERVALS)


@pytest.fixture(scope="session")
def tsv(data_dir):
	return write_intervals(data_dir / "test.tsv", INTERVALS)


@pytest.fixture(scope="session")
def tsv_gz(data_dir):
	return write_intervals(data_dir / "test.tsv.gz", INTERVALS)


## Running the tool


@pytest.fixture
def run(tmp_path):
	"""Run `figwig bam2bw` in a scratch directory and hand back the finished process.

	The `-n` prefix is supplied automatically and points into this test's own
	`tmp_path`, so tests never name output files and never collide. Nothing is
	asserted about the return code here -- use this fixture directly only when
	the failure itself is what is being tested.
	"""

	def _run(*args, name="out"):
		command = BAM2BW + [str(arg) for arg in args]
		command += ["-n", str(tmp_path / name)]

		return subprocess.run(command, capture_output=True, text=True, env=ENV)

	return _run


@pytest.fixture
def stranded(tmp_path, run):
	"""Run `figwig bam2bw` and read back the plus and minus bigWigs it wrote."""

	def _stranded(*args, name="out"):
		process = run(*args, name=name)

		assert process.returncode == 0, process.stderr

		return (read_counts(tmp_path / (name + ".+.bw")),
			read_counts(tmp_path / (name + ".-.bw")))

	return _stranded


@pytest.fixture
def unstranded(tmp_path, run):
	"""Run `figwig bam2bw -u` and read back the single bigWig it wrote."""

	def _unstranded(*args, name="out"):
		process = run(*args, "-u", name=name)

		assert process.returncode == 0, process.stderr

		return read_counts(tmp_path / (name + ".bw"))

	return _unstranded


## Reading the bigWigs back


# `figwig bam2bw` writes one entry per non-zero position with span=1, so reading a
# bigWig back gives a sparse mapping of position to value. Everything the tests
# assert on is expressed through that mapping rather than through the file
# bytes, because the byte layout is pyBigWig's business and can change between
# versions without the counts being wrong.


def read_counts(path):
	"""Read a bigWig into a nested dictionary of positions and values.

	Every chromosome in the header appears as a key even when it carries no
	entries, so that a test can assert a chromosome is empty rather than
	having to assert it is absent.

	Parameters
	----------
	path: str or pathlib.Path
		The bigWig file to read.

	Returns
	-------
	values: dict of str to dict of int to float
		A mapping from chromosome name, to a mapping from position to the
		value recorded there. Positions with no signal are not present.
	"""

	values = {}

	bw = pyBigWig.open(str(path))
	for chrom in bw.chroms():
		values[chrom] = {}

		for start, end, value in (bw.intervals(chrom) or ()):
			for position in range(start, end):
				values[chrom][position] = value

	bw.close()

	return values


def entries(values, chrom):
	"""Split one chromosome of a read bigWig into sorted positions and values.

	This exists so that assertions can be written as two flat lists compared
	with `assert_array_almost_equal`, which is both easier to read than a dict
	literal and reports the first differing position when it fails.

	Parameters
	----------
	values: dict of str to dict of int to float
		The return value of `read_counts`.

	chrom: str
		The chromosome to pull out.

	Returns
	-------
	positions: list of int
		The positions carrying signal, in ascending order.

	counts: list of float
		The value at each of those positions, in the same order.
	"""

	positions = sorted(values[chrom])
	counts = [values[chrom][position] for position in positions]

	return positions, counts


def total(values):
	"""Sum every value across every chromosome of a read bigWig.

	Parameters
	----------
	values: dict of str to dict of int to float
		The return value of `read_counts`.

	Returns
	-------
	total: float
		The sum of all recorded values.
	"""

	return sum(sum(chrom.values()) for chrom in values.values())


def chrom_lengths(path):
	"""Read the chromosome lengths out of a bigWig header.

	The lengths never appear in the values, so a sizes file or FASTA that
	gives the wrong length produces a correct-looking track with a wrong
	header. This is the only way to see that.

	Parameters
	----------
	path: str or pathlib.Path
		The bigWig file to read.

	Returns
	-------
	lengths: dict of str to int
		A mapping from chromosome name to the length in the header.
	"""

	bw = pyBigWig.open(str(path))
	lengths = dict(bw.chroms())
	bw.close()

	return lengths


## Input validation


@pytest.mark.parametrize("extension", [".txt", ".fastq", ".bw", ".bigWig",
	".bam.bai", ".fa", ""])
def test_rejects_unknown_extension(run, sizes, tmp_path, extension):
	path = tmp_path / ("reads" + extension)
	path.write_text("")

	process = run(path, "-s", sizes)

	assert process.returncode != 0
	assert "Filenames must end in one of" in process.stderr


def test_rejects_unknown_extension_among_valid_ones(run, sizes, bam, tmp_path):
	path = tmp_path / "reads.txt"
	path.write_text("")

	process = run(bam, path, "-s", sizes)

	assert process.returncode != 0
	assert "Filenames must end in one of" in process.stderr


def test_requires_sizes(run, bam):
	process = run(bam)

	assert process.returncode == 2
	assert "-s/--sizes" in process.stderr


def test_requires_name(bam, sizes):
	# The -n prefix is supplied by the `run` fixture, so this one test has to
	# build the command itself in order to leave it out.
	process = subprocess.run(BAM2BW + [str(bam), "-s", str(sizes)],
		capture_output=True, text=True, env=ENV)

	assert process.returncode == 2
	assert "-n/--name" in process.stderr


def test_requires_an_input_file(run, sizes):
	process = run("-s", sizes)

	assert process.returncode == 2


def test_missing_input_file(run, sizes, tmp_path):
	process = run(tmp_path / "does_not_exist.bam", "-s", sizes)

	assert process.returncode != 0


def test_missing_sizes_file(run, bam, tmp_path):
	process = run(bam, "-s", tmp_path / "does_not_exist.chrom.sizes")

	assert process.returncode != 0


## Chromosome sizes and FASTA input


def test_chrom_sizes_sets_the_header(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes)

	assert list(pos.keys()) == ["chr1", "chr2", "chr3"]
	assert list(neg.keys()) == ["chr1", "chr2", "chr3"]


def test_chrom_not_in_sizes_is_absent_from_output(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes)

	assert "chrUN" not in pos
	assert "chrUN" not in neg


@pytest.mark.parametrize("extension", FASTA_EXTENSIONS)
def test_fasta_is_accepted(stranded, bam, fastas, extension):
	pos, neg = stranded(bam, "-s", fastas[extension])

	assert list(pos.keys()) == ["chr1", "chr2", "chr3"]

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [2, 1], 4)


@pytest.mark.parametrize("extension", FASTA_EXTENSIONS)
def test_fasta_lengths_match_chrom_sizes(run, bam, fastas, tmp_path,
	extension):
	"""The lengths only reach the bigWig header, so a FASTA read wrongly
	produces a correct-looking track over a wrong chromosome."""

	process = run(bam, "-s", fastas[extension])
	assert process.returncode == 0

	assert chrom_lengths(tmp_path / "out.+.bw") == dict(CHROM_SIZES)


@pytest.mark.parametrize("extension", FASTA_EXTENSIONS)
def test_fasta_matches_chrom_sizes(stranded, bam, sizes, fastas, extension):
	from_sizes, _ = stranded(bam, "-s", sizes, name="sizes")
	from_fasta, _ = stranded(bam, "-s", fastas[extension], name="fasta")

	assert from_sizes == from_fasta


def test_chrom_sizes_order_is_preserved(stranded, bam, tmp_path):
	reordered = write_chrom_sizes(tmp_path / "reordered.chrom.sizes",
		[("chr3", 200), ("chr1", 1000), ("chr2", 500)])

	pos, _ = stranded(bam, "-s", reordered)

	assert list(pos.keys()) == ["chr3", "chr1", "chr2"]


## Output file naming


def test_stranded_writes_two_files(run, bam, sizes, tmp_path):
	process = run(bam, "-s", sizes)

	assert process.returncode == 0
	assert (tmp_path / "out.+.bw").exists()
	assert (tmp_path / "out.-.bw").exists()
	assert not (tmp_path / "out.bw").exists()


def test_unstranded_writes_one_file(run, bam, sizes, tmp_path):
	process = run(bam, "-s", sizes, "-u")

	assert process.returncode == 0
	assert (tmp_path / "out.bw").exists()
	assert not (tmp_path / "out.+.bw").exists()
	assert not (tmp_path / "out.-.bw").exists()


## BAM: 5' ends, the default behaviour


def test_bam_forward_reads(stranded, bam, sizes):
	pos, _ = stranded(bam, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(pos, "chr2")
	assert_array_almost_equal(positions, [50])
	assert_array_almost_equal(counts, [1], 4)


def test_bam_reverse_reads(stranded, bam, sizes):
	_, neg = stranded(bam, "-s", sizes)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [339, 419])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(neg, "chr2")
	assert_array_almost_equal(positions, [89])
	assert_array_almost_equal(counts, [1], 4)


def test_bam_chromosome_without_reads_is_empty(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes)

	assert pos["chr3"] == {}
	assert neg["chr3"] == {}


def test_bam_drops_unmapped_and_filtered_reads(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes)

	# Nine reads plus one unmapped read go in; the unmapped read and the one
	# on chrUN are dropped, leaving eight.
	assert total(pos) + total(neg) == 8


## -f/--fragments


def test_bam_fragments_forward(stranded, bam, sizes):
	pos, _ = stranded(bam, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 149, 200, 229])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)

	positions, counts = entries(pos, "chr2")
	assert_array_almost_equal(positions, [50, 74])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_bam_fragments_reverse(stranded, bam, sizes):
	_, neg = stranded(bam, "-s", sizes, "-f")

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [300, 339, 400, 419])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)

	positions, counts = entries(neg, "chr2")
	assert_array_almost_equal(positions, [80, 89])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_bam_fragments_double_the_signal(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes)
	pos_f, neg_f = stranded(bam, "-s", sizes, "-f", name="frag")

	assert total(pos_f) + total(neg_f) == 2 * (total(pos) + total(neg))


## -u/--unstranded


def test_unstranded_merges_both_strands(unstranded, bam, sizes):
	values = unstranded(bam, "-s", sizes)

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [100, 200, 339, 419])
	assert_array_almost_equal(counts, [2, 1, 2, 1], 4)

	positions, counts = entries(values, "chr2")
	assert_array_almost_equal(positions, [50, 89])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_unstranded_equals_sum_of_strands(stranded, unstranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, name="stranded")
	values = unstranded(bam, "-s", sizes, name="unstranded")

	for chrom, _ in CHROM_SIZES:
		merged = dict(pos[chrom])
		for position, count in neg[chrom].items():
			merged[position] = merged.get(position, 0) + count

		assert values[chrom] == merged


def test_unstranded_fragments(unstranded, bam, sizes):
	values = unstranded(bam, "-s", sizes, "-f")

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [100, 149, 200, 229, 300, 339, 400,
		419])
	assert_array_almost_equal(counts, [2, 2, 1, 1, 2, 2, 1, 1], 4)

	positions, counts = entries(values, "chr2")
	assert_array_almost_equal(positions, [50, 74, 80, 89])
	assert_array_almost_equal(counts, [1, 1, 1, 1], 4)


## -ps/--pos_shift and -ns/--neg_shift


def test_zero_shifts_are_the_default(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, name="default")
	pos_z, neg_z = stranded(bam, "-s", sizes, "-ps", 0, "-ns", 0, name="zero")

	assert pos == pos_z
	assert neg == neg_z


def test_pos_shift_moves_only_the_recorded_start(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, "-ps", 4)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [104, 204])
	assert_array_almost_equal(counts, [2, 1], 4)

	# Reverse reads record reference_end - 1, which -ps does not touch.
	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [339, 419])
	assert_array_almost_equal(counts, [2, 1], 4)


def test_neg_shift_moves_only_the_recorded_end(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, "-ns", 5)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [344, 424])
	assert_array_almost_equal(counts, [2, 1], 4)


def test_negative_shifts_are_accepted(stranded, bam, sizes):
	"""The Tn5 correction for ATAC-seq is +4 on one strand and -5 on the other.

	This is the single most common way `figwig bam2bw` is invoked on ATAC-seq, and a
	negative value is easy to break at the argparse level without breaking
	anything else, so it gets its own test rather than only appearing inside a
	combination.
	"""

	pos, neg = stranded(bam, "-s", sizes, "-ps", 4, "-ns", -5)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [104, 204])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [334, 414])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(pos, "chr2")
	assert_array_almost_equal(positions, [54])
	assert_array_almost_equal(counts, [1], 4)

	positions, counts = entries(neg, "chr2")
	assert_array_almost_equal(positions, [84])
	assert_array_almost_equal(counts, [1], 4)


@pytest.mark.parametrize("flag", ["-ps", "-ns", "--pos_shift", "--neg_shift"])
def test_negative_shift_parses_on_every_spelling(run, bam, sizes, flag):
	process = run(bam, "-s", sizes, flag, -5)

	assert process.returncode == 0, process.stderr


def test_shifts_with_fragments(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, "-f", "-ps", 4, "-ns", -5)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [104, 144, 204, 224])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [304, 334, 404, 414])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)

	# The chr2 reverse read is only 10bp long, so shifting both of its ends
	# inward collapses them onto the same position, where the counts add
	# rather than overwrite.
	positions, counts = entries(neg, "chr2")
	assert_array_almost_equal(positions, [84])
	assert_array_almost_equal(counts, [2], 4)


def test_shifts_preserve_total_signal(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, name="default")
	pos_s, neg_s = stranded(bam, "-s", sizes, "-ps", 4, "-ns", -5,
		name="shifted")

	assert total(pos_s) + total(neg_s) == total(pos) + total(neg)


## -sf/--scale_factor


def test_scale_factor_one_is_the_default(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, name="default")
	pos_s, neg_s = stranded(bam, "-s", sizes, "-sf", 1, name="scaled")

	assert pos == pos_s
	assert neg == neg_s


def test_scale_factor_multiplies_every_value(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, "-sf", 10)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [20, 10], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [339, 419])
	assert_array_almost_equal(counts, [20, 10], 4)


def test_fractional_scale_factor(stranded, bam, sizes):
	pos, _ = stranded(bam, "-s", sizes, "-sf", 0.5)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [1, 0.5], 4)


def test_scale_factor_does_not_move_positions(stranded, bam, sizes):
	pos, _ = stranded(bam, "-s", sizes, name="default")
	pos_s, _ = stranded(bam, "-s", sizes, "-sf", 7.5, name="scaled")

	assert sorted(pos["chr1"]) == sorted(pos_s["chr1"])


## -r/--read_depth


def test_read_depth_normalizes_to_one(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, "-r")

	assert_array_almost_equal(total(pos) + total(neg), 1.0, 4)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [0.25, 0.125], 4)


def test_read_depth_counts_both_strands_once(unstranded, stranded, bam, sizes):
	"""Read depth is summed over both strands when stranded and over the one
	merged dictionary when not, so the two must agree rather than the
	unstranded run halving or doubling."""

	pos, neg = stranded(bam, "-s", sizes, "-r", name="stranded")
	values = unstranded(bam, "-s", sizes, "-r", name="unstranded")

	assert_array_almost_equal(total(pos) + total(neg), 1.0, 4)
	assert_array_almost_equal(total(values), 1.0, 4)

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [100, 200, 339, 419])
	assert_array_almost_equal(counts, [0.25, 0.125, 0.25, 0.125], 4)


def test_read_depth_with_scale_factor(stranded, bam, sizes):
	"""The documented way to make the bigWigs sum to a chosen value."""

	pos, neg = stranded(bam, "-s", sizes, "-r", "-sf", 1000000)

	assert_array_almost_equal(total(pos) + total(neg), 1000000, 4)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [250000, 125000], 4)


def test_read_depth_with_fragments_normalizes_to_one(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, "-f", "-r")

	assert_array_almost_equal(total(pos) + total(neg), 1.0, 4)


## Interval input: .bed, .bed.gz, .tsv, .tsv.gz


def test_bed_records_starts(stranded, bed, sizes):
	pos, _ = stranded(bed, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(pos, "chr2")
	assert_array_almost_equal(positions, [50])
	assert_array_almost_equal(counts, [1], 4)


def test_bed_has_no_reverse_strand(stranded, bed, sizes):
	"""Interval files carry no strand, so everything lands on the plus file."""

	_, neg = stranded(bed, "-s", sizes)

	assert total(neg) == 0


def test_bed_fragments(stranded, bed, sizes):
	pos, _ = stranded(bed, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 149, 200, 229])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)

	positions, counts = entries(pos, "chr2")
	assert_array_almost_equal(positions, [50, 74])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_bed_drops_chroms_not_in_sizes(stranded, bed, sizes):
	pos, _ = stranded(bed, "-s", sizes)

	assert "chrUN" not in pos
	assert total(pos) == 4


def test_bed_shifts(stranded, bed, sizes):
	pos, _ = stranded(bed, "-s", sizes, "-ps", 4, "-ns", -5, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [104, 144, 204, 224])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)


@pytest.mark.parametrize("other", ["bed_gz", "tsv", "tsv_gz"])
def test_interval_formats_agree(stranded, bed, sizes, other, request):
	"""The four interval extensions are the same parser and must agree."""

	other_path = request.getfixturevalue(other)

	from_bed, _ = stranded(bed, "-s", sizes, "-f", name="bed")
	from_other, _ = stranded(other_path, "-s", sizes, "-f", name="other")

	assert from_bed == from_other


def test_bed_float_coordinates(stranded, sizes, tmp_path):
	"""Fragment files exported from some pipelines carry float coordinates."""

	path = write_intervals(tmp_path / "floats.bed", [
		("chr1", "100.0", "150.0"),
		("chr1", "200.0", "230.0")
	])

	pos, _ = stranded(path, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 149, 200, 229])
	assert_array_almost_equal(counts, [1, 1, 1, 1], 4)


def test_bed_extra_columns_are_ignored(stranded, sizes, tmp_path):
	path = write_intervals(tmp_path / "wide.bed", [
		("chr1", 100, 150, "peak1", 960, "+"),
		("chr1", 200, 230, "peak2", 12, "-")
	])

	pos, _ = stranded(path, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_empty_bed_produces_empty_bigwigs(run, sizes, tmp_path):
	path = write_intervals(tmp_path / "empty.bed", [])

	process = run(path, "-s", sizes)

	assert process.returncode == 0
	assert total(read_counts(tmp_path / "out.+.bw")) == 0


## Multiple input files and -p/--parallel


def test_two_files_are_concatenated(stranded, bam, sizes):
	pos, neg = stranded(bam, bam, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [4, 2], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [339, 419])
	assert_array_almost_equal(counts, [4, 2], 4)


@pytest.mark.parametrize("n_jobs", [1, 2, 3])
def test_parallel_matches_serial(stranded, bam, sizes, n_jobs):
	pos, neg = stranded(bam, bam, bam, "-s", sizes, "-p", 1, name="serial")
	pos_p, neg_p = stranded(bam, bam, bam, "-s", sizes, "-p", n_jobs,
		name="parallel")

	assert pos == pos_p
	assert neg == neg_p


def test_parallel_unstranded_matches_serial(unstranded, bam, sizes):
	"""When unstranded, the two per-file dictionaries are the same object, so
	the merge across files has to avoid counting each file twice."""

	serial = unstranded(bam, bam, "-s", sizes, "-p", 1, name="serial")
	parallel = unstranded(bam, bam, "-s", sizes, "-p", 2, name="parallel")

	assert serial == parallel

	positions, counts = entries(serial, "chr1")
	assert_array_almost_equal(positions, [100, 200, 339, 419])
	assert_array_almost_equal(counts, [4, 2, 4, 2], 4)


def test_more_jobs_than_files(stranded, bam, sizes):
	pos, _ = stranded(bam, "-s", sizes, "-p", 4)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [2, 1], 4)


def test_file_order_does_not_matter(stranded, bam, bed, sizes):
	first, _ = stranded(bam, bed, "-s", sizes, name="first")
	second, _ = stranded(bed, bam, "-s", sizes, name="second")

	assert first == second


def test_mixed_bam_and_bed(stranded, bam, bed, sizes):
	pos, neg = stranded(bam, bed, "-s", sizes)

	# The BED entries land on the plus strand alongside the forward reads.
	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [4, 2], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [339, 419])
	assert_array_almost_equal(counts, [2, 1], 4)


def test_read_depth_spans_all_files(stranded, bam, sizes):
	pos, neg = stranded(bam, bam, "-s", sizes, "-r")

	assert_array_almost_equal(total(pos) + total(neg), 1.0, 4)


## -z/--zooms and -v/--verbose


@pytest.mark.parametrize("zooms", [0, 1, 4])
def test_zooms_do_not_change_values(stranded, bam, sizes, zooms):
	pos, neg = stranded(bam, "-s", sizes, name="default")
	pos_z, neg_z = stranded(bam, "-s", sizes, "-z", zooms, name="zoomed")

	assert pos == pos_z
	assert neg == neg_z


def test_verbose_reports_missing_chromosomes(run, bam, sizes):
	process = run(bam, "-s", sizes, "-v")

	assert process.returncode == 0
	assert "chrUN" in process.stdout + process.stderr


def test_quiet_by_default(run, bam, sizes):
	process = run(bam, "-s", sizes)

	assert process.returncode == 0
	assert "chrUN" not in process.stdout


## Combinations that occur in practice


def test_atac_bam_with_tn5_shift(stranded, bam, sizes):
	"""ATAC-seq reads, 5' cut sites, Tn5 offset applied per strand."""

	pos, neg = stranded(bam, "-s", sizes, "-ps", 4, "-ns", -5)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [104, 204])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [334, 414])
	assert_array_almost_equal(counts, [2, 1], 4)


def test_atac_fragments_unstranded(unstranded, tsv_gz, sizes):
	"""A 10x-style fragments.tsv.gz, both cut sites, one unstranded track."""

	values = unstranded(tsv_gz, "-s", sizes, "-f")

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [100, 149, 200, 229])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)

	positions, counts = entries(values, "chr2")
	assert_array_almost_equal(positions, [50, 74])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_atac_fragments_unstranded_shifted(unstranded, tsv_gz, sizes):
	values = unstranded(tsv_gz, "-s", sizes, "-f", "-ps", 4, "-ns", -5)

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [104, 144, 204, 224])
	assert_array_almost_equal(counts, [2, 2, 1, 1], 4)


def test_chip_seq_unstranded_cpm(unstranded, bam, sizes):
	"""ChIP-seq 5' ends, strand collapsed, normalized to counts per million."""

	values = unstranded(bam, "-s", sizes, "-r", "-sf", 1000000)

	assert_array_almost_equal(total(values), 1000000, 4)

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [100, 200, 339, 419])
	assert_array_almost_equal(counts, [250000, 125000, 250000, 125000], 4)


def test_merged_replicates_in_parallel(stranded, bam, sizes):
	"""Three replicate files pooled into one normalized pair of tracks."""

	pos, neg = stranded(bam, bam, bam, "-s", sizes, "-p", 3, "-r", "-sf",
		1000000)

	assert_array_almost_equal(total(pos) + total(neg), 1000000, 4)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [250000, 125000], 4)


def test_every_flag_together(unstranded, bam, sizes):
	"""Two files, both ends, shifted, pooled in parallel, CPM, zoomed."""

	values = unstranded(bam, bam, "-s", sizes, "-f", "-ps", 4, "-ns", -5,
		"-r", "-sf", 1000000, "-p", 2, "-z", 2)

	assert_array_almost_equal(total(values), 1000000, 4)

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [104, 144, 204, 224, 304, 334, 404,
		414])
	assert_array_almost_equal(counts, [125000, 125000, 62500, 62500, 125000,
		125000, 62500, 62500], 4)

	positions, counts = entries(values, "chr2")
	assert_array_almost_equal(positions, [54, 69, 84])
	assert_array_almost_equal(counts, [62500, 62500, 125000], 4)


@pytest.mark.parametrize("is_unstranded", [False, True])
@pytest.mark.parametrize("fragments", [False, True])
@pytest.mark.parametrize("shifts", [(0, 0), (4, -5)])
def test_total_signal_is_conserved(run, tmp_path, bam, sizes, is_unstranded,
	fragments, shifts):
	"""Across the whole flag cube, the amount of signal written depends only
	on how many reads were counted and whether both ends were recorded.

	Shifting moves positions and can collapse two ends onto one another, and
	going unstranded merges the two dictionaries, but neither may create or
	destroy signal.
	"""

	args = [bam, "-s", sizes, "-ps", shifts[0], "-ns", shifts[1]]
	if fragments:
		args.append("-f")
	if is_unstranded:
		args.append("-u")

	process = run(*args)
	assert process.returncode == 0, process.stderr

	if is_unstranded:
		written = total(read_counts(tmp_path / "out.bw"))
	else:
		written = (total(read_counts(tmp_path / "out.+.bw"))
			+ total(read_counts(tmp_path / "out.-.bw")))

	assert written == 8 * (2 if fragments else 1)


## CIGAR handling
#
# The recorded positions come from reference_start and reference_end, so what
# a read covers on the reference -- not how long the sequenced fragment was --
# decides where its ends land. These pin the arithmetic for the CIGAR
# operations that show up in real alignments.


@pytest.mark.parametrize("label,cigar,five_prime,three_prime", [
	("50M exact match", [(0, 50)], 100, 149),
	("10S30M10S soft clipped", [(4, 10), (0, 30), (4, 10)], 100, 129),
	("10H30M hard clipped", [(5, 10), (0, 30)], 100, 129),
	("20M100N20M spliced", [(0, 20), (3, 100), (0, 20)], 100, 239),
	("20M5D20M deletion", [(0, 20), (2, 5), (0, 20)], 100, 144),
	("20M5I20M insertion", [(0, 20), (1, 5), (0, 20)], 100, 139)
])
def test_cigar_sets_the_reference_span(stranded, sizes, tmp_path, label, cigar,
	five_prime, three_prime):
	path = write_bam(tmp_path / "cigar.bam", CHROM_SIZES,
		[("chr1", 100, 0, False, cigar)])

	pos, _ = stranded(path, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, sorted([five_prime, three_prime]))
	assert_array_almost_equal(counts, [1, 1], 4)


def test_soft_clips_do_not_move_the_five_prime_end(stranded, sizes, tmp_path):
	"""Adapter bases left on the read must not shift the recorded cut site."""

	clipped = write_bam(tmp_path / "clipped.bam", CHROM_SIZES,
		[("chr1", 100, 0, False, [(4, 10), (0, 30), (4, 10)])])
	plain = write_bam(tmp_path / "plain.bam", CHROM_SIZES,
		[("chr1", 100, 30, False)])

	from_clipped, _ = stranded(clipped, "-s", sizes, "-f", name="clipped")
	from_plain, _ = stranded(plain, "-s", sizes, "-f", name="plain")

	assert from_clipped == from_plain


def test_spliced_reverse_read(stranded, sizes, tmp_path):
	"""On the reverse strand the 5' end of a spliced read is on the far side
	of the intron, which is the case most likely to be got backwards."""

	path = write_bam(tmp_path / "spliced.bam", CHROM_SIZES,
		[("chr1", 100, 0, True, [(0, 20), (3, 100), (0, 20)])])

	_, neg = stranded(path, "-s", sizes, "-f")

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [100, 239])
	assert_array_almost_equal(counts, [1, 1], 4)


## SAM flags
#
# Beyond the unmapped bit, `figwig bam2bw` does no flag filtering at all. Every
# remaining record is counted, so a PCR duplicate, a secondary alignment of a
# multi-mapping read, and a QC-failed record each contribute a count. These
# tests exist to make that explicit, so that adding a filter later is a
# deliberate change to a documented behaviour rather than a silent one.


@pytest.mark.parametrize("label,flag", [
	("PCR duplicate", 1024),
	("secondary alignment", 256),
	("supplementary alignment", 2048),
	("QC fail", 512)
])
def test_flagged_records_are_still_counted(stranded, sizes, tmp_path, label,
	flag):
	path = write_bam(tmp_path / "flagged.bam", CHROM_SIZES,
		[("chr1", 100, 50, False, [(0, 50)], flag)])

	pos, _ = stranded(path, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100])
	assert_array_almost_equal(counts, [1], 4)


@pytest.mark.parametrize("mapping_quality", [0, 1, 60])
def test_low_mapping_quality_is_not_filtered(stranded, sizes, tmp_path,
	mapping_quality):
	"""There is no -q option, so a multi-mapping read at MAPQ 0 counts the
	same as a uniquely mapping one."""

	path = write_bam(tmp_path / "mapq.bam", CHROM_SIZES,
		[("chr1", 100, 50, False, [(0, 50)], 0, mapping_quality)])

	pos, _ = stranded(path, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100])
	assert_array_almost_equal(counts, [1], 4)


## Chromosome boundaries


def test_read_at_position_zero(stranded, sizes, tmp_path):
	path = write_bam(tmp_path / "zero.bam", CHROM_SIZES,
		[("chr1", 0, 50, False)])

	pos, _ = stranded(path, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [0, 49])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_read_ending_on_the_last_base(stranded, sizes, tmp_path):
	"""chr1 is 1000bp, so 999 is the last position a bigWig entry may use."""

	path = write_bam(tmp_path / "last.bam", CHROM_SIZES,
		[("chr1", 950, 50, False), ("chr1", 950, 50, True)])

	pos, neg = stranded(path, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [950, 999])
	assert_array_almost_equal(counts, [1, 1], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [950, 999])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_negative_position_from_a_shift_is_reported(run, sizes, tmp_path):
	"""A shift large enough to push a read off the front of the chromosome
	cannot be written, so it is discarded and said out loud rather than
	vanishing."""

	path = write_bam(tmp_path / "near_zero.bam", CHROM_SIZES,
		[("chr1", 10, 20, False)])

	process = run(path, "-s", sizes, "--pos_shift=-100")

	assert process.returncode == 0
	assert "discarded" in process.stdout
	assert "chr1" in process.stdout
	assert total(read_counts(tmp_path / "out.+.bw")) == 0


## Degenerate inputs


def test_empty_bam(run, sizes, tmp_path):
	path = write_bam(tmp_path / "empty.bam", CHROM_SIZES, [])

	process = run(path, "-s", sizes)

	assert process.returncode == 0
	assert total(read_counts(tmp_path / "out.+.bw")) == 0
	assert total(read_counts(tmp_path / "out.-.bw")) == 0


def test_empty_bam_with_read_depth(run, sizes, tmp_path):
	"""Read-depth normalization divides by the total count, which is zero
	here, so this is the input that would raise ZeroDivisionError."""

	path = write_bam(tmp_path / "empty.bam", CHROM_SIZES, [])

	process = run(path, "-s", sizes, "-r")

	assert process.returncode == 0
	assert total(read_counts(tmp_path / "out.+.bw")) == 0


def test_bam_with_no_reads_on_any_listed_chromosome(run, bam, tmp_path):
	"""Every read is on a chromosome the sizes file does not mention."""

	other = write_chrom_sizes(tmp_path / "other.chrom.sizes",
		[("chrOther", 100)])

	process = run(bam, "-s", other)

	assert process.returncode == 0
	assert total(read_counts(tmp_path / "out.+.bw")) == 0


def test_scale_factor_of_zero(stranded, bam, sizes):
	"""Positions are still written, carrying a value of zero."""

	pos, _ = stranded(bam, "-s", sizes, "-sf", 0)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [0, 0], 4)


def test_single_base_read(stranded, sizes, tmp_path):
	"""A 1bp read has the same 5' and 3' end, so -f records two counts at one
	position rather than one count at each of two."""

	path = write_bam(tmp_path / "one.bam", CHROM_SIZES,
		[("chr1", 100, 1, False)])

	pos, _ = stranded(path, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100])
	assert_array_almost_equal(counts, [2], 4)


def test_zero_length_interval(stranded, sizes, tmp_path):
	"""Some fragment files contain start == end."""

	path = write_intervals(tmp_path / "zero.bed", [("chr1", 100, 100)])

	pos, _ = stranded(path, "-s", sizes, "-f")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [99, 100])
	assert_array_almost_equal(counts, [1, 1], 4)


## chrom_sizes and FASTA parsing robustness


def test_crlf_line_endings(stranded, bam, sizes, tmp_path):
	"""A chrom_sizes file written on Windows."""

	path = tmp_path / "crlf.chrom.sizes"
	path.write_bytes(b"chr1\t1000\r\nchr2\t500\r\nchr3\t200\r\n")

	expected, _ = stranded(bam, "-s", sizes, name="unix")
	actual, _ = stranded(bam, "-s", path, name="crlf")

	assert actual == expected


def test_space_separated_sizes(stranded, bam, sizes, tmp_path):
	path = tmp_path / "spaces.chrom.sizes"
	path.write_text("chr1 1000\nchr2 500\nchr3 200\n")

	expected, _ = stranded(bam, "-s", sizes, name="tabs")
	actual, _ = stranded(bam, "-s", path, name="spaces")

	assert actual == expected


def test_fasta_header_with_a_description(stranded, bam, sizes, tmp_path):
	"""Reference FASTAs carry an accession and length after the chromosome
	name. Only the first token names the chromosome in a BAM, so the two have
	to be matched on that."""

	path = tmp_path / "described.fa"
	with open(path, "w") as outfile:
		for chrom, size in CHROM_SIZES:
			outfile.write(">{} AC:CM000663.2 LN:{}\n".format(chrom, size))
			outfile.write(("ACGT" * (size // 4 + 1))[:size] + "\n")

	expected, _ = stranded(bam, "-s", sizes, name="sizes")
	actual, _ = stranded(bam, "-s", path, name="described")

	assert actual == expected


def test_bgzipped_fasta(stranded, bam, sizes, fastas, tmp_path):
	"""pyfaidx reads a BGZF-compressed FASTA, which is what samtools produces
	and what the .gz half of the accepted extensions means in practice."""

	path = str(tmp_path / "compressed.fa.gz")
	pysam.tabix_compress(str(fastas[".fa"]), path, force=True)

	expected, _ = stranded(bam, "-s", sizes, name="plain")
	actual, _ = stranded(bam, "-s", path, name="bgzipped")

	assert actual == expected


## Progress reporting under parallelism


def test_verbose_with_parallel_files(run, bam, sizes, tmp_path):
	"""Two files under -v, each with its own progress bar, read one after
	another in the main process."""

	process = run(bam, bam, "-s", sizes, "-v", "-p", 2)

	assert process.returncode == 0

	pos = read_counts(tmp_path / "out.+.bw")
	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [4, 2], 4)


## Malformed input is rejected rather than silently mis-parsed
#
# These assert only that the run fails, not what it says. The current messages
# are bare tracebacks; the point of the tests is that a future change cannot
# quietly start accepting one of these and writing a track built from a
# misreading of the file.


def test_fai_file_is_accepted(stranded, bam, sizes, fastas, tmp_path):
	"""A samtools .fai index starts with the two chrom_sizes columns and then
	carries three more, and is the file most likely to be reached for when a
	chrom_sizes is not already lying around."""

	pysam.faidx(str(fastas[".fa"]))

	expected, _ = stranded(bam, "-s", sizes, name="sizes")
	actual, _ = stranded(bam, "-s", str(fastas[".fa"]) + ".fai", name="fai")

	assert actual == expected


@pytest.mark.parametrize("label,content", [
	("extra columns", "chr1\t1000\tfoo\tbar\nchr2\t500\tx\ty\nchr3\t200\ta\tb\n"),
	("blank lines", "chr1\t1000\n\nchr2\t500\n\nchr3\t200\n\n"),
	("comment header", "#genome hg38\nchr1\t1000\nchr2\t500\nchr3\t200\n"),
	("leading whitespace", "  chr1\t1000\nchr2\t500\nchr3\t200\n")
])
def test_tolerated_chrom_sizes_variants(stranded, bam, sizes, tmp_path, label,
	content):
	"""Only the first two fields are read, and blank and # lines are skipped,
	so these all describe the same three chromosomes."""

	path = tmp_path / "variant.chrom.sizes"
	path.write_text(content)

	expected, _ = stranded(bam, "-s", sizes, name="plain")
	actual, _ = stranded(bam, "-s", path, name="variant")

	assert actual == expected


@pytest.mark.parametrize("label,content", [
	("one column", "chr1\n"),
	("non-numeric length", "chr1\tlong\n")
])
def test_malformed_chrom_sizes_is_rejected(run, bam, tmp_path, label, content):
	path = tmp_path / "bad.chrom.sizes"
	path.write_text(content)

	process = run(bam, "-s", path)

	assert process.returncode != 0
	assert "bad.chrom.sizes, line 1" in process.stderr


def test_duplicate_chromosome_in_sizes_is_rejected(run, bam, tmp_path):
	path = write_chrom_sizes(tmp_path / "duplicate.chrom.sizes",
		[("chr1", 1000), ("chr1", 1000)])

	process = run(bam, "-s", path)

	assert process.returncode != 0


def test_plain_gzipped_fasta_is_rejected(run, bam, fastas, tmp_path):
	"""pyfaidx reads BGZF but not plain gzip, so a FASTA compressed with gzip
	fails even though the extension is accepted."""

	path = tmp_path / "plain.fa.gz"
	with open(fastas[".fa"], "rb") as infile:
		with gzip.open(path, "wb") as outfile:
			shutil.copyfileobj(infile, outfile)

	process = run(bam, "-s", path)

	assert process.returncode != 0
	assert "BGZF" in process.stderr

	# pyfaidx already explains BGZF on its own. Naming the file is what bam2bw
	# adds, and is what matters when -s was filled in by a script.
	assert "plain.fa.gz" in process.stderr


@pytest.mark.parametrize("label,content", [
	("comment header", "# id=sample\n# description=\nchr1\t100\t150\nchr1\t200\t230\n"),
	("comment between entries", "chr1\t100\t150\n# a note\nchr1\t200\t230\n"),
	("blank lines", "chr1\t100\t150\n\nchr1\t200\t230\n\n"),
	("trailing blank line", "chr1\t100\t150\nchr1\t200\t230\n\n")
])
def test_tolerated_bed_variants(stranded, sizes, tmp_path, label, content):
	"""Blank and # lines are skipped, so these all describe the same two
	intervals. A 10x CellRanger fragments file always opens with a # header,
	and is the reason this matters."""

	path = tmp_path / "variant.bed"
	path.write_text(content)

	pos, _ = stranded(path, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [1, 1], 4)


@pytest.mark.parametrize("label,content", [
	("two columns", "chr1\t100\n"),
	("track header", 'track name="peaks"\nchr1\t100\t150\n'),
	("non-numeric coordinates", "chr1\tstart\tend\n")
])
def test_malformed_bed_is_rejected(run, sizes, tmp_path, label, content):
	path = tmp_path / "bad.bed"
	path.write_text(content)

	process = run(path, "-s", sizes)

	assert process.returncode != 0
	assert "bad.bed, line" in process.stderr


def test_mapped_read_without_a_cigar_is_rejected(run, sizes, tmp_path):
	"""A mapped record whose CIGAR is '*' has no reference end, so there is no
	3' position for it. It must fail rather than write a track missing it."""

	path = write_bam(tmp_path / "no_cigar.bam", CHROM_SIZES,
		[("chr1", 100, 50, False, [])])

	process = run(path, "-s", sizes)

	assert process.returncode != 0
	assert "no_cigar.bam" in process.stderr
	assert "CIGAR" in process.stderr


def test_missing_output_directory_is_rejected(run, bam, sizes, tmp_path):
	process = run(bam, "-s", sizes, name="no_such_directory/out")

	assert process.returncode != 0


## -3p/--three_prime
#
# The mirror of the default: record the far end of the read rather than the
# near one. For a forward read that is reference_end - 1 and for a reverse read
# it is reference_start, so the expected positions are the default ones with
# the two strands' roles swapped.


def test_three_prime_bam(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, "-3p")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [149, 229])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(pos, "chr2")
	assert_array_almost_equal(positions, [74])
	assert_array_almost_equal(counts, [1], 4)

	positions, counts = entries(neg, "chr1")
	assert_array_almost_equal(positions, [300, 400])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(neg, "chr2")
	assert_array_almost_equal(positions, [80])
	assert_array_almost_equal(counts, [1], 4)


def test_three_prime_is_the_other_end_of_fragments(stranded, bam, sizes):
	"""-f records both ends, so every position -3p records must be one of
	them, and the two together must be the -f set."""

	both, both_neg = stranded(bam, "-s", sizes, "-f", name="frag")
	five, five_neg = stranded(bam, "-s", sizes, name="five")
	three, three_neg = stranded(bam, "-s", sizes, "-3p", name="three")

	for chrom, _ in CHROM_SIZES:
		assert set(five[chrom]) | set(three[chrom]) == set(both[chrom])
		assert set(five_neg[chrom]) | set(three_neg[chrom]) == set(both_neg[chrom])


def test_three_prime_bed(stranded, bed, sizes):
	pos, _ = stranded(bed, "-s", sizes, "-3p")

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [149, 229])
	assert_array_almost_equal(counts, [2, 1], 4)

	positions, counts = entries(pos, "chr2")
	assert_array_almost_equal(positions, [74])
	assert_array_almost_equal(counts, [1], 4)


def test_three_prime_unstranded(unstranded, bam, sizes):
	values = unstranded(bam, "-s", sizes, "-3p")

	positions, counts = entries(values, "chr1")
	assert_array_almost_equal(positions, [149, 229, 300, 400])
	assert_array_almost_equal(counts, [2, 1, 2, 1], 4)


def test_three_prime_preserves_total_signal(stranded, bam, sizes):
	pos, neg = stranded(bam, "-s", sizes, name="five")
	pos_3, neg_3 = stranded(bam, "-s", sizes, "-3p", name="three")

	assert total(pos_3) + total(neg_3) == total(pos) + total(neg)


def test_three_prime_and_fragments_are_mutually_exclusive(run, bam, sizes):
	process = run(bam, "-s", sizes, "-f", "-3p")

	assert process.returncode == 2
	assert "not allowed with" in process.stderr


## -mp/--mate_pairs
#
# A pair carries one tag, not two. The RNA's 5' end is the 5' end of whichever
# mate --rna5 names, and its 3' end is the *other* mate's own 5' end, since a
# read can only ever reveal its own 5'-most base. The layout below is chosen so
# that every one of those four positions is distinct:
#
#   pairA  read1 fwd chr1:100-150   read2 rev chr1:300-350
#   pairB  read1 rev chr1:600-650   read2 fwd chr1:400-450
#   pairC  read1 fwd chr2:100-140   read2 rev chr2:200-240
#
# so read1's own 5' ends are 100 / 649 / 100 and read2's are 349 / 400 / 239.


@pytest.fixture
def paired(data_dir):
	return write_paired_bam(data_dir / "paired.bam", CHROM_SIZES, [
		("pairA", ("chr1", 100, 50, False), ("chr1", 300, 50, True)),
		("pairB", ("chr1", 600, 50, True), ("chr1", 400, 50, False)),
		("pairC", ("chr2", 100, 40, False), ("chr2", 200, 40, True))
	])


@pytest.mark.parametrize("flags,pos_chr1,pos_chr2,neg_chr1,neg_chr2", [
	([], [100], [100], [649], []),
	(["-3p"], [349], [239], [400], []),
	(["--opposite_strand"], [649], [], [100], [100]),
	(["-3p", "--opposite_strand"], [400], [], [349], [239]),
	(["--rna5", "read2"], [400], [], [349], [239]),
	(["--rna5", "read2", "-3p"], [649], [], [100], [100]),
	(["--rna5", "read2", "--opposite_strand"], [349], [239], [400], []),
	(["--rna5", "read2", "-3p", "--opposite_strand"], [100], [100], [649], [])
])
def test_mate_pairs_records_one_position_per_pair(stranded, paired, sizes,
	flags, pos_chr1, pos_chr2, neg_chr1, neg_chr2):
	pos, neg = stranded(paired, "-s", sizes, "-mp", *flags)

	assert_array_almost_equal(entries(pos, "chr1")[0], pos_chr1)
	assert_array_almost_equal(entries(pos, "chr2")[0], pos_chr2)
	assert_array_almost_equal(entries(neg, "chr1")[0], neg_chr1)
	assert_array_almost_equal(entries(neg, "chr2")[0], neg_chr2)

	# Three pairs in, three counts out, wherever they land.
	assert total(pos) + total(neg) == 3


def test_mate_pairs_halves_the_count(stranded, paired, sizes):
	"""Without -mp each mate is an independent event, which is the doubling
	the flag exists to remove."""

	pos, neg = stranded(paired, "-s", sizes, name="plain")
	pos_mp, neg_mp = stranded(paired, "-s", sizes, "-mp", name="paired")

	assert total(pos) + total(neg) == 6
	assert total(pos_mp) + total(neg_mp) == 3


def test_without_mate_pairs_each_mate_counts_separately(stranded, paired,
	sizes):
	pos, neg = stranded(paired, "-s", sizes)

	assert_array_almost_equal(entries(pos, "chr1")[0], [100, 400])
	assert_array_almost_equal(entries(pos, "chr2")[0], [100])
	assert_array_almost_equal(entries(neg, "chr1")[0], [349, 649])
	assert_array_almost_equal(entries(neg, "chr2")[0], [239])


def test_mate_pairs_fragments_records_both_ends(stranded, paired, sizes):
	"""With -f the pair contributes both of its jointly-determined ends, on
	the strand --rna5 selects."""

	pos, neg = stranded(paired, "-s", sizes, "-mp", "-f")

	assert_array_almost_equal(entries(pos, "chr1")[0], [100, 349])
	assert_array_almost_equal(entries(pos, "chr2")[0], [100, 239])
	assert_array_almost_equal(entries(neg, "chr1")[0], [400, 649])

	assert total(pos) + total(neg) == 6


def test_mate_pairs_does_not_need_a_name_sorted_bam(stranded, sizes, data_dir,
	tmp_path):
	"""Mates are buffered by name as the file streams, so the order records
	appear in must not change the result."""

	forward = write_paired_bam(tmp_path / "forward.bam", CHROM_SIZES, [
		("pairA", ("chr1", 100, 50, False), ("chr1", 300, 50, True)),
		("pairB", ("chr1", 600, 50, True), ("chr1", 400, 50, False))
	])
	interleaved = write_paired_bam(tmp_path / "interleaved.bam", CHROM_SIZES, [
		("pairA", ("chr1", 100, 50, False), None),
		("pairB", ("chr1", 600, 50, True), None),
		("pairA", None, ("chr1", 300, 50, True)),
		("pairB", None, ("chr1", 400, 50, False))
	])

	first, first_neg = stranded(forward, "-s", sizes, "-mp", name="forward")
	second, second_neg = stranded(interleaved, "-s", sizes, "-mp",
		name="interleaved")

	assert first == second
	assert first_neg == second_neg


@pytest.mark.parametrize("label,extra", [
	("improper pair", 0),
	("secondary", 0x2 | 0x100),
	("supplementary", 0x2 | 0x800)
])
def test_mate_pairs_skips_flagged_records(stranded, sizes, tmp_path, label,
	extra):
	"""Only one primary alignment per mate can be matched up unambiguously by
	name, so the rest are dropped."""

	path = write_paired_bam(tmp_path / "flagged.bam", CHROM_SIZES, [
		("good", ("chr1", 100, 50, False), ("chr1", 300, 50, True)),
		("bad", ("chr1", 500, 50, False, extra), ("chr1", 700, 50, True, extra))
	])

	pos, neg = stranded(path, "-s", sizes, "-mp")

	assert_array_almost_equal(entries(pos, "chr1")[0], [100])
	assert total(pos) + total(neg) == 1


def test_mate_pairs_skips_orphans(stranded, sizes, tmp_path):
	"""A mate whose partner never appears contributes nothing."""

	path = write_paired_bam(tmp_path / "orphan.bam", CHROM_SIZES, [
		("good", ("chr1", 100, 50, False), ("chr1", 300, 50, True)),
		("orphan", ("chr1", 500, 50, False), None)
	])

	pos, neg = stranded(path, "-s", sizes, "-mp")

	assert_array_almost_equal(entries(pos, "chr1")[0], [100])
	assert total(pos) + total(neg) == 1


@pytest.mark.parametrize("other", ["bed", "bed_gz", "tsv", "tsv_gz"])
def test_mate_pairs_rejects_interval_input(run, sizes, request, other):
	"""Interval files carry no mate information."""

	process = run(request.getfixturevalue(other), "-s", sizes, "-mp")

	assert process.returncode != 0
	assert "--mate_pairs only supports BAM/SAM" in process.stderr


def test_opposite_strand_duplicates_flipping_rna5_and_three_prime(stranded,
	paired, sizes):
	"""--opposite_strand adds no track that the other two flags cannot already
	produce: the eight combinations collapse to four distinct outputs."""

	tracks = {}
	for rna5 in ("read1", "read2"):
		for three_prime in ([], ["-3p"]):
			for opposite in ([], ["--opposite_strand"]):
				name = "{}{}{}".format(rna5, "_3p" if three_prime else "",
					"_opp" if opposite else "")
				pos, neg = stranded(paired, "-s", sizes, "-mp", "--rna5", rna5,
					*(three_prime + opposite), name=name)
				tracks[name] = (sorted(entries(pos, "chr1")[0]),
					sorted(entries(neg, "chr1")[0]))

	assert tracks["read1"] == tracks["read2_3p_opp"]
	assert tracks["read1_3p"] == tracks["read2_opp"]
	assert tracks["read1_opp"] == tracks["read2_3p"]
	assert tracks["read1_3p_opp"] == tracks["read2"]

	assert len(set(map(str, tracks.values()))) == 4


def test_sam_input_is_processed(stranded, sizes, tmp_path, bam):
	path = tmp_path / "test.sam"
	with pysam.AlignmentFile(str(bam), "rb") as infile:
		with pysam.AlignmentFile(str(path), "w", header=infile.header) as out:
			for alignment in infile:
				out.write(alignment)

	pos, neg = stranded(path, "-s", sizes)

	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [2, 1], 4)


def test_missing_sam_file_errors(run, sizes, tmp_path):
	process = run(tmp_path / "does_not_exist.sam", "-s", sizes)

	assert process.returncode != 0


def test_out_of_range_positions_are_reported(run, tmp_path):
	"""A chrom_sizes file shorter than the BAM header -- the wrong assembly,
	or a truncated file -- makes reads near the end of a chromosome vanish."""

	path = write_bam(tmp_path / "far.bam", [("chr1", 3000)],
		[("chr1", 100, 50, False), ("chr1", 2000, 50, False)])
	short = write_chrom_sizes(tmp_path / "short.chrom.sizes", [("chr1", 1000)])

	process = run(path, "-s", short, "-v")

	reported = (process.returncode != 0
		or "chr1" in process.stdout + process.stderr)

	assert reported


def test_read_depth_covers_only_what_is_written(run, tmp_path):
	path = write_bam(tmp_path / "far.bam", [("chr1", 3000)],
		[("chr1", 100, 50, False), ("chr1", 2000, 50, False)])
	short = write_chrom_sizes(tmp_path / "short.chrom.sizes", [("chr1", 1000)])

	process = run(path, "-s", short, "-r")
	assert process.returncode == 0

	written = (total(read_counts(tmp_path / "out.+.bw"))
		+ total(read_counts(tmp_path / "out.-.bw")))

	assert_array_almost_equal(written, 1.0, 4)


def test_in_range_reads_survive_alongside_discarded_ones(run, tmp_path):
	"""Only the offending entries go; the rest of the chromosome is written."""

	path = write_bam(tmp_path / "mixed.bam", [("chr1", 3000)],
		[("chr1", 100, 50, False), ("chr1", 200, 50, False),
		 ("chr1", 2000, 50, False)])
	short = write_chrom_sizes(tmp_path / "short.chrom.sizes", [("chr1", 1000)])

	process = run(path, "-s", short)

	assert process.returncode == 0

	pos = read_counts(tmp_path / "out.+.bw")
	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [1, 1], 4)


def test_discard_message_counts_the_entries(run, tmp_path):
	path = write_bam(tmp_path / "far.bam", [("chr1", 3000)],
		[("chr1", 2000, 50, False), ("chr1", 2100, 50, False),
		 ("chr1", 2200, 50, True)])
	short = write_chrom_sizes(tmp_path / "short.chrom.sizes", [("chr1", 1000)])

	process = run(path, "-s", short)

	assert process.returncode == 0
	assert "3 entries" in process.stdout
	assert "chr1: 3" in process.stdout


def test_nothing_is_reported_when_everything_fits(run, bam, sizes):
	process = run(bam, "-s", sizes)

	assert process.returncode == 0
	assert "discarded" not in process.stdout


def test_read_depth_is_reported_under_verbose(run, bam, sizes):
	process = run(bam, "-s", sizes, "-r", "-v")

	assert process.returncode == 0
	assert "read depth of 8" in process.stdout


def test_read_depth_is_not_reported_without_verbose(run, bam, sizes):
	process = run(bam, "-s", sizes, "-r")

	assert process.returncode == 0
	assert "read depth" not in process.stdout


def test_discarded_entries_are_excluded_from_read_depth_unstranded(run,
	tmp_path):
	path = write_bam(tmp_path / "far.bam", [("chr1", 3000)],
		[("chr1", 100, 50, False), ("chr1", 2000, 50, True)])
	short = write_chrom_sizes(tmp_path / "short.chrom.sizes", [("chr1", 1000)])

	process = run(path, "-s", short, "-r", "-u")

	assert process.returncode == 0
	assert_array_almost_equal(total(read_counts(tmp_path / "out.bw")), 1.0, 4)


## figwig
#
# What figwig's version adds to bam2bw: the `figwig` command it runs under,
# BigWigWriter as the writer, and the fast readers of the bam2bw speed search,
# which must take ordinary files rather than leave them to pysam's loop.


def test_figwig_lists_its_commands():
	process = subprocess.run([sys.executable, "-m", "figwig", "-h"],
		capture_output=True, text=True, env=ENV)

	assert process.returncode == 0
	assert "bam2bw" in process.stdout


def test_figwig_requires_a_command():
	process = subprocess.run([sys.executable, "-m", "figwig"],
		capture_output=True, text=True, env=ENV)

	assert process.returncode == 2
	assert "command" in process.stderr


def test_figwig_version():
	process = subprocess.run([sys.executable, "-m", "figwig", "--version"],
		capture_output=True, text=True, env=ENV)

	assert process.returncode == 0
	assert process.stdout.strip() == "figwig {}".format(figwig.__version__)


def test_usage_names_figwig_bam2bw():
	process = subprocess.run(BAM2BW, capture_output=True, text=True, env=ENV)

	assert process.returncode == 2
	assert process.stderr.startswith("usage: figwig bam2bw ")


def test_missing_packages_are_named(bam, sizes, tmp_path, monkeypatch):
	"""Without the bam2bw extra, the error names the packages and the extra,
	before any file is read or written."""

	find_spec = importlib.util.find_spec
	monkeypatch.setattr(importlib.util, "find_spec",
		lambda name, *args: None if name in ("pysam", "deflate") else find_spec(
			name, *args))

	with pytest.raises(SystemExit, match=r"pysam, deflate.*figwig\[bam2bw\]"):
		figwig.bam2bw.main([str(bam), "-s", str(sizes), "-n",
			str(tmp_path / "out")])

	assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("kind", ["bam", "bed", "tsv_gz", "bgzf"])
def test_fast_readers_take_ordinary_files(sizes, data_dir, tmp_path, kind, p,
	request):
	"""A file the fast readers take gives each chromosome's counts as
	(positions, counts, start, end) rather than as the dictionary pysam's or
	the per-line loop fills, which would give the same values far slower."""

	if kind == "bgzf":
		path = str(tmp_path / "test.tsv.gz")
		pysam.tabix_compress(str(request.getfixturevalue("tsv")), path)
	else:
		path = str(request.getfixturevalue(kind))

	args = figwig.bam2bw._parser("figwig bam2bw").parse_args([path, "-s",
		str(sizes), "-n", "out", "-p", str(p)])
	pos_reads, _ = figwig.bam2bw.extract_reads(args, CHROM_SIZES, 0, threads=p)

	assert isinstance(pos_reads["chr1"], tuple)
	positions, counts, start, end = pos_reads["chr1"]
	assert positions[start:end].tolist() == [100, 200]
	assert counts[start:end].tolist() == [2, 1]


@pytest.mark.parametrize("flags", [[], ["-f", "-u"], ["-3p", "-r", "-z", "2"]])
def test_bytes_do_not_depend_on_parallel(run, bam, bed, sizes, tmp_path, flags):
	"""The readers count the same reads at any -p, and BigWigWriter compresses
	with libdeflate, whose bytes do not depend on the number of threads."""

	names = []
	for p in (1, 2, 5):
		process = run(bam, bed, "-s", sizes, "-p", p, *flags, name="p{}".format(p))
		assert process.returncode == 0, process.stderr
		names.append("p{}".format(p))

	suffixes = [".bw"] if "-u" in flags else [".+.bw", ".-.bw"]
	for suffix in suffixes:
		data = [(tmp_path / (name + suffix)).read_bytes() for name in names]
		assert data[0] == data[1] == data[2]


@pytest.mark.parametrize("zooms", [0, 3])
def test_zooms_are_written_by_figwig(run, bam, sizes, tmp_path, zooms):
	process = run(bam, "-s", sizes, "-z", zooms)
	assert process.returncode == 0, process.stderr

	bw = pyBigWig.open(str(tmp_path / "out.+.bw"))
	levels = bw.header()["nLevels"]
	bw.close()

	assert (levels == 0) if zooms == 0 else (1 <= levels <= zooms)


@pytest.mark.parametrize("batch", [1, 2, 3, 1 << 20])
def test_write_counts_in_batches(tmp_path, monkeypatch, batch):
	"""Chromosomes go to the writer in batches of at least _WRITE_ITEMS
	positions; the file is the same whatever the batch, and holds each count
	times the scale factor over the read depth, as a float32."""

	chrom_sizes = [("chrA", 100), ("chrB", 50), ("chrC", 80), ("chrD", 30)]
	reads = {
		"chrA": (numpy.array([3, 7, 50]), numpy.array([1, 4, 2])),
		"chrB": (numpy.empty(0, dtype=int), numpy.empty(0, dtype=int)),
		"chrC": (numpy.array([0, 79]), numpy.array([5, 1])),
		"chrD": (numpy.array([12]), numpy.array([3])),
	}

	def write(name, batch):
		monkeypatch.setattr(figwig.bam2bw, "_WRITE_ITEMS", batch)
		path = tmp_path / name
		with figwig.BigWigWriter(path, chrom_sizes, zooms=0) as writer:
			figwig.bam2bw.write_counts(writer, chrom_sizes, reads, 2.5, 7)
		return path

	path = write("batched.bw", batch)
	assert path.read_bytes() == write("whole.bw", 1 << 20).read_bytes()

	values = read_counts(path)
	for chrom, (positions, counts) in reads.items():
		expected = numpy.float32(counts * 2.5 / 7).tolist()
		assert values[chrom] == dict(zip(positions.tolist(), expected))


## Which files are read where
#
# Files the fast readers take are read one after another in the main process,
# each on every core -p gives; files read on one thread whatever they are
# given go to a pool of processes.


@pytest.fixture(scope="session")
def sam(data_dir, bam):
	path = data_dir / "test.sam"
	with pysam.AlignmentFile(str(bam), "rb") as infile:
		with pysam.AlignmentFile(str(path), "w", header=infile.header) as out:
			for alignment in infile:
				out.write(alignment)

	return path


@pytest.fixture(scope="session")
def bgzf_tsv(data_dir, tsv):
	path = str(data_dir / "test.bgzf.tsv.gz")
	pysam.tabix_compress(str(tsv), path)
	return path


def parse(*args):
	return figwig.bam2bw._parser("figwig bam2bw").parse_args([str(arg)
		for arg in args] + ["-n", "out"])


def test_cores(monkeypatch):
	monkeypatch.setattr(figwig.bam2bw, "_cpu_count", lambda: 16)

	assert figwig.bam2bw._cores(1) == 1
	assert figwig.bam2bw._cores(5) == 5
	assert figwig.bam2bw._cores(-1) == 16
	assert figwig.bam2bw._cores(-4) == 13
	assert figwig.bam2bw._cores(-40) == 1

	with pytest.raises(ValueError, match="not 0"):
		figwig.bam2bw._cores(0)


@pytest.mark.parametrize("kind,one_thread", [("bam", False), ("bed", False),
	("tsv", False), ("bgzf_tsv", False), ("sam", True), ("bed_gz", True),
	("tsv_gz", True)])
def test_reads_on_one_thread(sizes, request, kind, one_thread):
	path = str(request.getfixturevalue(kind))
	args = parse(path, "-s", sizes)

	assert figwig.bam2bw._reads_on_one_thread(path, args) == one_thread


def test_reads_on_one_thread_mate_pairs_and_remote(bam, sizes, tmp_path):
	assert figwig.bam2bw._reads_on_one_thread(str(bam), parse(bam, "-s",
		sizes, "-mp"))

	for path in ("https://example.org/test.bam", str(tmp_path / "missing.bam")):
		assert figwig.bam2bw._reads_on_one_thread(path, parse(path, "-s", sizes))


def test_plan(bam, sam, bed, bed_gz, sizes):
	"""Files the fast readers take get every core the pool leaves; the pool has
	a process per file read on one thread, up to the cores there are."""

	plan = figwig.bam2bw._plan

	assert plan(parse(sam, "-s", sizes), 4) == ([0], [], 0, 4)
	assert plan(parse(bam, bed, bam, "-s", sizes), 6) == ([0, 1, 2], [], 0, 6)
	assert plan(parse(sam, bam, bed_gz, "-s", sizes), 4) == ([1], [0, 2], 2, 2)
	assert plan(parse(sam, bam, bed_gz, "-s", sizes), 2) == ([1], [0, 2], 2, 1)
	assert plan(parse(sam, sam, sam, "-s", sizes), 2) == ([], [0, 1, 2], 2, 1)
	assert plan(parse(sam, bam, bed_gz, "-s", sizes), 1) == ([0, 1, 2], [], 0, 1)


def test_each_bam_gets_the_threads_it_is_given(bam, sizes, monkeypatch):
	"""The BAM reader takes the threads extract_reads is given, not -p divided
	by the number of files."""

	given = []
	counter = figwig.bam2bw._KeyCounter

	class Recorder(counter):
		def __init__(self, arrays, stranded, threads):
			given.append(threads)
			super().__init__(arrays, stranded, threads)

	monkeypatch.setattr(figwig.bam2bw, "_KeyCounter", Recorder)

	args = parse(bam, bam, bam, "-s", sizes, "-p", 6)
	pos_reads, _ = figwig.bam2bw.extract_reads(args, CHROM_SIZES, 1, threads=6)

	assert given == [6]
	assert isinstance(pos_reads["chr1"], tuple)


@pytest.mark.parametrize("p", [2, 3, 8, -1])
def test_pool_and_threads_write_what_one_core_does(run, bam, sam, bed, bed_gz,
	bgzf_tsv, sizes, tmp_path, p):
	"""SAM and plain-gzip files read by the pool, at the same time as BAM and
	BED files read here on the cores left, give the bytes of every file read
	here on one core. The pool's processes exit as usual, so that a spawned
	or forkserver pool (macOS, Python 3.14) reports no leaked semaphore."""

	files = [sam, bam, bed_gz, bgzf_tsv, sam, bed]
	for name, cores in (("one", 1), ("many", p)):
		process = run(*files, "-s", sizes, "-p", cores, name=name)
		assert process.returncode == 0, process.stderr
		assert "leaked" not in process.stderr

	for suffix in (".+.bw", ".-.bw"):
		assert ((tmp_path / ("one" + suffix)).read_bytes() ==
			(tmp_path / ("many" + suffix)).read_bytes())


def test_verbose_with_pooled_files(run, sam, sizes, tmp_path):
	"""The progress bars of the pool's processes share this process's lock."""

	process = run(sam, sam, "-s", sizes, "-v", "-p", 2)

	assert process.returncode == 0, process.stderr
	assert "leaked" not in process.stderr

	pos = read_counts(tmp_path / "out.+.bw")
	positions, counts = entries(pos, "chr1")
	assert_array_almost_equal(positions, [100, 200])
	assert_array_almost_equal(counts, [4, 2], 4)


def test_zero_parallel_is_rejected(run, bam, sizes):
	process = run(bam, "-s", sizes, "-p", 0)

	assert process.returncode != 0
	assert "-p/--parallel" in process.stderr


## Known bugs
#
# These describe how the tool should behave. They are skipped rather than
# asserting the current output, so that fixing the bug turns them green
# instead of requiring the test to be rewritten.


@pytest.mark.skip(reason="BUG: -mp on a single-end BAM drops every read, since "
	"none is a proper pair, and writes empty bigWigs with exit 0 instead of "
	"saying the flag does not apply to the input")
def test_mate_pairs_on_a_single_end_bam_is_reported(run, bam, sizes, tmp_path):
	process = run(bam, "-s", sizes, "-mp")

	reported = (process.returncode != 0
		or "pair" in (process.stdout + process.stderr).lower())

	assert reported
