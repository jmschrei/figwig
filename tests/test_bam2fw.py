# test_bam2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig bam2fw`, against `figwig bam2bw` on the same inputs.

`figwig bam2fw` counts what `figwig bam2bw` counts, so each file's tracks in
the figwig file must read as the bigWigs `figwig bam2bw` writes from that
file alone, under the same options.
"""

import os
import sys
import subprocess

from pathlib import Path

import numpy
import pytest

if sys.platform == 'win32':
	pytest.skip("pysam does not build on Windows", allow_module_level=True)

import figwig.bam2bw  # noqa: E402
import figwig.bam2fw  # noqa: E402

from figwig import BigWigReader  # noqa: E402
from figwig import FigWigReader  # noqa: E402

from figwig.__main__ import main  # noqa: E402

from .reads import write_bam  # noqa: E402
from .reads import write_chrom_sizes  # noqa: E402
from .reads import write_intervals  # noqa: E402
from .summary_files import read_regions  # noqa: E402
from .summary_reference import read_index  # noqa: E402

ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
	MKL_NUM_THREADS="1")

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
	("chrUN", 10, 10, False),
]
OTHER_READS = [
	("chr1", 120, 50, False),
	("chr1", 980, 30, True),
	("chr2", 5, 25, False),
	("chr2", 490, 10, True),
	("chr3", 0, 10, False),
]
UNMAPPED = [("chr1", 500, 20)]
INTERVALS = [
	("chr1", 100, 150),
	("chr1", 100, 150),
	("chr1", 200, 230),
	("chr2", 50, 75),
	("chrUN", 10, 20),
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
	return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def sizes(data_dir):
	return str(write_chrom_sizes(data_dir / "test.chrom.sizes", CHROM_SIZES))


@pytest.fixture(scope="session")
def bam(data_dir):
	return str(write_bam(data_dir / "test.bam", BAM_CHROM_SIZES, READS,
		unmapped=UNMAPPED))


@pytest.fixture(scope="session")
def other_bam(data_dir):
	return str(write_bam(data_dir / "other.bam", BAM_CHROM_SIZES, OTHER_READS))


@pytest.fixture(scope="session")
def tsv_gz(data_dir):
	return str(write_intervals(data_dir / "test.tsv.gz", INTERVALS))


def bam2bw(tmp_path, filename, sizes, *flags):
	"""`figwig bam2bw` on one file, as the bigWigs it wrote."""

	os.makedirs(tmp_path, exist_ok=True)
	name = str(tmp_path / ("bw_" + os.path.basename(filename)))
	figwig.bam2bw.main([filename, "-s", sizes, "-n", name] + list(flags),
		prog="figwig bam2bw")
	if "-u" in flags:
		return [name + ".bw"]
	return [name + ".+.bw", name + ".-.bw"]


def bam2fw(tmp_path, filenames, sizes, *flags):
	os.makedirs(tmp_path, exist_ok=True)
	name = str(tmp_path / "out")
	main(["bam2fw"] + list(filenames) + ["-s", sizes, "-n", name] + list(flags))
	return name + ".fw"


def bigwig_values(path, chrom, length):
	"""A bigWig's values over a chromosome, as zeros for a bigWig without
	data blocks, which `figwig bam2bw` writes for a strand with no counts
	and `BigWigReader` does not open."""

	reader = BigWigReader(path)
	if reader._data_index == 0:
		return numpy.zeros(length, dtype=numpy.float32)
	return reader.read(chrom, [0], length)[0]


def assert_tracks_match_bigwigs(path, names, bigwigs):
	reader = FigWigReader(path)
	for chrom, length in CHROM_SIZES:
		expected = numpy.stack([bigwig_values(bigwig, chrom, length) for bigwig
			in bigwigs])[None]
		y = reader.read(chrom, [0], length, tracks=names, dtype="float32")
		numpy.testing.assert_array_equal(y, expected)


@pytest.mark.parametrize("flags", [(), ("-u",), ("-3p",), ("-ps", "4", "-ns",
	"-5"), ("-sf", "0.5",), ("-r",), ("-sf", "2", "-r"), ("-f", "-u")])
def test_counts_what_bam2bw_counts(tmp_path, bam, sizes, flags):
	bigwigs = bam2bw(tmp_path, bam, sizes, *flags)
	path = bam2fw(tmp_path, [bam], sizes, *flags)
	names = ["test"] if "-u" in flags else ["test.+", "test.-"]
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == names
	assert reader.chrom_sizes == dict(CHROM_SIZES)
	assert_tracks_match_bigwigs(path, names, bigwigs)

	if "-sf" in flags or "-r" in flags:
		assert reader.tracks[0]["dtype"] == "float32"
	else:
		assert reader.tracks[0]["dtype"] == "uint8"
	if "-u" not in flags:
		assert reader.tracks[0]["strand"] == "+" and reader.tracks[1]["strand"] \
			== "-"
		assert [track["group"] for track in reader.tracks] == [0, 0]


def test_each_file_is_its_own_tracks(tmp_path, bam, other_bam, sizes):
	path = bam2fw(tmp_path, [bam, other_bam], sizes)
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == ["test.+", "test.-",
		"other.+", "other.-"]
	assert [track["group"] for track in reader.tracks] == [0, 0, 1, 1]
	assert_tracks_match_bigwigs(path, ["test.+", "test.-"], bam2bw(tmp_path, bam,
		sizes))
	assert_tracks_match_bigwigs(path, ["other.+", "other.-"], bam2bw(tmp_path,
		other_bam, sizes))


def test_read_depth_is_per_file(tmp_path, bam, other_bam, sizes):
	path = bam2fw(tmp_path, [bam, other_bam], sizes, "-r")
	assert_tracks_match_bigwigs(path, ["test.+", "test.-"], bam2bw(tmp_path, bam,
		sizes, "-r"))
	assert_tracks_match_bigwigs(path, ["other.+", "other.-"], bam2bw(tmp_path,
		other_bam, sizes, "-r"))


def test_fragments_file(tmp_path, tsv_gz, sizes):
	path = bam2fw(tmp_path, [tsv_gz], sizes, "-f")
	assert_tracks_match_bigwigs(path, ["test.+", "test.-"], bam2bw(tmp_path,
		tsv_gz, sizes, "-f"))


def test_metadata_table(tmp_path, bam, other_bam, sizes):
	table = tmp_path / "meta.tsv"
	table.write_text("file\tname\tassay\tbiosample\n"
		"test.bam\tK562_ATAC\tATAC-seq\tK562\n"
		"{}\t\tDNase-seq\tHepG2\n".format(other_bam))
	path = bam2fw(tmp_path, [bam, other_bam], sizes, "-m", str(table), "-g",
		"hg38", "-t", "256", "--dtype", "uint16", "-l", "0")
	reader = FigWigReader(path)
	assert reader.genome == "hg38" and reader.tile == 256
	assert [track["name"] for track in reader.tracks] == ["K562_ATAC.+",
		"K562_ATAC.-", "other.+", "other.-"]
	assert reader.tracks[0]["assay"] == "ATAC-seq"
	assert reader.tracks[1]["biosample"] == "K562"
	assert reader.tracks[2]["biosample"] == "HepG2"
	assert [track["dtype"] for track in reader.tracks] == ["uint16"] * 4
	assert_tracks_match_bigwigs(path, ["K562_ATAC.+", "K562_ATAC.-"], bam2bw(
		tmp_path, bam, sizes))


def test_discarded_entries_are_reported(tmp_path, bam, sizes, capsys):
	bam2fw(tmp_path, [bam], sizes, "-ps", "700")
	out = capsys.readouterr().out
	assert "fell outside the provided chromosome sizes" in out


def test_errors(tmp_path, bam, sizes):
	with pytest.raises(SystemExit, match="Filenames must end in"):
		bam2fw(tmp_path, [str(tmp_path / "x.txt")], sizes)
	table = tmp_path / "meta.tsv"
	table.write_text("file\tcell\nnope.bam\tK562\n")
	with pytest.raises(SystemExit, match="has no row for"):
		bam2fw(tmp_path, [bam], sizes, "-m", str(table))
	with pytest.raises(SystemExit, match="outside 0 to 255"):
		bam2fw(tmp_path, [bam], sizes, "--dtype", "uint8", "-sf", "200")
	with pytest.raises(SystemExit, match="No such file|not a file|cannot"):
		bam2fw(tmp_path, [bam], str(tmp_path / "none.sizes"))
	# An input that is not a BAM, or is cut short, is pysam's message, not
	# its traceback; a path that does not exist is said to be none.
	fake = tmp_path / "fake.bam"
	fake.write_text("not a bam\n")
	with pytest.raises(SystemExit, match="does not contain alignment data"):
		bam2fw(tmp_path, [str(fake)], sizes)
	with open(bam, "rb") as handle:
		fake.write_bytes(handle.read(os.path.getsize(bam) - 40))
	with pytest.raises(SystemExit, match="truncated|cut short|EOF"):
		bam2fw(tmp_path, [str(fake)], sizes)
	with pytest.raises(SystemExit, match="none.bam is not a file"):
		bam2fw(tmp_path, [str(tmp_path / "none.bam")], sizes)
	path = bam2fw(tmp_path, [bam], sizes)
	with pytest.raises(SystemExit, match="exists; --force writes over it"):
		bam2fw(tmp_path, [bam], sizes)
	assert bam2fw(tmp_path, [bam], sizes, "--force") == path


def test_bam_header_is_checked_against_the_sizes(tmp_path, bam):
	"""A chromosome the BAM's header and the sizes both name must have one
	length, and the BAM must share a chromosome with the sizes: an hg19 BAM
	against hg38 sizes placed every read at its hg19 position, and sizes of
	other names dropped every read, both with exit 0. ``figwig bam2bw``
	warns instead, keeping bam2bw 0.5.1's behaviour."""

	other = write_chrom_sizes(tmp_path / "other.sizes", [("chr1", 1001),
		("chr2", 500)])
	with pytest.raises(SystemExit, match="aligns to a chr1 of 1,000 bases, but "
			"the sizes give it 1,001: the reads and the sizes must be on one "
			"assembly"):
		bam2fw(tmp_path, [bam], str(other))
	renamed = write_chrom_sizes(tmp_path / "renamed.sizes", [("1", 1000),
		("2", 500)])
	with pytest.raises(SystemExit, match="none of the chromosomes of .* is in "
			"the sizes: it has chr1, chr2, chr3, ... and the sizes have 1, 2"):
		bam2fw(tmp_path, [bam], str(renamed))
	# A sizes file that names some of the BAM's chromosomes, at their
	# lengths, still converts; the others' reads are left out and reported.
	subset = write_chrom_sizes(tmp_path / "subset.sizes", [("chr2", 500)])
	path = bam2fw(tmp_path, [bam], str(subset))
	assert list(FigWigReader(path).chrom_sizes) == ["chr2"]



def test_helpers():
	assert figwig.bam2fw._base_name("/a/b/x.tsv.gz") == "x"
	assert figwig.bam2fw._base_name("x.bam") == "x"
	assert figwig.bam2fw._base_name("x.other") == "x.other"
	tracks, groups = figwig.bam2fw._tracks_for_files(["a.bam", "b.bam"], None,
		True, "uint32")
	assert tracks == [{"name": "a", "dtype": "uint32"}, {"name": "b", "dtype":
		"uint32"}]
	assert groups == [["a"], ["b"]]


def test_command_line(tmp_path, bam, sizes):
	name = str(tmp_path / "cli")
	result = subprocess.run([sys.executable, "-m", "figwig", "bam2fw", bam, "-s",
		sizes, "-n", name, "-p", "2"], env=ENV, capture_output=True, text=True,
		timeout=600)
	assert result.returncode == 0, result.stderr
	assert len(FigWigReader(name + ".fw").tracks) == 2


# --barcode-column: a fragments file as one track per cell.

FRAGMENTS = [
	("chr1", 100, 150, "AAAC-1", 2),
	("chr1", 100, 150, "AAAG-1", 1),
	("chr1", 120, 160, "AAAC-1", 1),
	("chr1", 200, 230, "CCCT-1", 1),
	("chr2", 50, 75, "AAAG-1", 3),
	("chr2", 60, 75, "AAAC-1", 1),
	("chr2", 480, 510, "CCCT-1", 1),
	("chrUN", 10, 20, "AAAC-1", 1),
]
BARCODES = ["AAAC-1", "AAAG-1", "CCCT-1"]


@pytest.fixture(scope="session")
def fragments(data_dir):
	path = data_dir / "cells.tsv"
	with open(path, "w") as handle:
		handle.write("# id=test\n#\n")
		for fields in FRAGMENTS:
			handle.write("\t".join(map(str, fields)) + "\n")
	return str(path)


@pytest.fixture(scope="session")
def fragments_gz(data_dir):
	return str(write_intervals(data_dir / "cells.tsv.gz", FRAGMENTS))


@pytest.fixture(scope="session")
def per_cell_bigwigs(tmp_path_factory, data_dir, sizes):
	"""`figwig bam2bw -f -u` on each barcode's fragments alone, per flag
	set: the oracle for a track of the per-cell file."""

	import functools

	@functools.lru_cache(maxsize=None)
	def bigwigs(flags):
		out = tmp_path_factory.mktemp("percell")
		paths = {}
		for barcode in BARCODES:
			path = write_intervals(out / (barcode + ".tsv"), [fields for fields in
				FRAGMENTS if fields[3] == barcode])
			paths[barcode] = bam2bw(out, str(path), sizes, "-u", *flags)[0]
		return paths

	return bigwigs


@pytest.mark.parametrize("flags", [("-f",), (), ("-3p",), ("-f", "-ps", "4",
	"-ns", "-5"), ("-f", "-sf", "0.5")])
def test_barcode_column_counts_each_cell(tmp_path, fragments, sizes,
		per_cell_bigwigs, flags):
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", *flags)
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == BARCODES
	assert [track["group"] for track in reader.tracks] == [0, 0, 0]
	assert "strand" not in reader.tracks[0]
	bigwigs = per_cell_bigwigs(flags)
	for barcode in BARCODES:
		assert_tracks_match_bigwigs(path, [barcode], [bigwigs[barcode]])


def test_barcode_column_read_depth_is_the_files(tmp_path, fragments, sizes):
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f")
	scaled = bam2fw(tmp_path / "r", [fragments], sizes, "--barcode-column", "4",
		"-f", "-r")
	counts, reader = FigWigReader(path), FigWigReader(scaled)
	depth = counts.stats["sum"].sum()
	for chrom, length in CHROM_SIZES:
		expected = counts.read(chrom, [0], length, dtype="float64") / depth
		numpy.testing.assert_allclose(reader.read(chrom, [0], length,
			dtype="float64"), expected, rtol=1e-6)


def test_cells_table_picks_names_and_attributes(tmp_path, fragments, sizes,
		per_cell_bigwigs, capsys):
	table = tmp_path / "cells.csv"
	table.write_text("barcode,name,cell_type,dtype\n"
		"CCCT-1,,B,\nAAAC-1,mono_1,CD14 Mono,uint16\nGGGG-1,,T,\n")
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f",
		"--cells", str(table), "-v")
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == ["CCCT-1", "mono_1",
		"GGGG-1"]
	assert [track["cell_type"] for track in reader.tracks] == ["B", "CD14 Mono",
		"T"]
	assert [track["dtype"] for track in reader.tracks] == ["uint8", "uint16",
		"uint8"]
	assert reader.stats["sum"][2] == 0
	bigwigs = per_cell_bigwigs(("-f",))
	assert_tracks_match_bigwigs(path, ["CCCT-1"], [bigwigs["CCCT-1"]])
	assert_tracks_match_bigwigs(path, ["mono_1"], [bigwigs["AAAC-1"]])
	out = capsys.readouterr().out
	assert "2 fragments of {} have barcodes not in --cells".format(fragments) in out
	assert "chrUN encountered in input but not in FASTA/chrom sizes." in out


def test_groups_and_several_files(tmp_path, fragments, fragments_gz, sizes):
	table = tmp_path / "files.tsv"
	table.write_text("file\tname\tsample\ncells.tsv\tA\tpbmc\ncells.tsv.gz\t\t"
		"pbmc2\n")
	path = bam2fw(tmp_path, [fragments, fragments_gz], sizes, "--barcode-column",
		"4", "-f", "--groups", "2", "-m", str(table))
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == ["A." + b for b in
		BARCODES] + ["cells." + b for b in BARCODES]
	assert [track["group"] for track in reader.tracks] == [0, 0, 1, 2, 2, 3]
	assert [track["sample"] for track in reader.tracks] == ["pbmc"] * 3 + [
		"pbmc2"] * 3
	y = reader.read("chr1", [0], 1000, dtype="float32")
	numpy.testing.assert_array_equal(y[0, :3], y[0, 3:])


@pytest.mark.parametrize("threads", ["1", "3"])
def test_several_files_in_a_group_each(tmp_path, fragments, fragments_gz, sizes,
		threads):
	# Each file's cells are one group, written as already sorted entries,
	# so the second file's tracks lie at slots other than their ids.
	table = tmp_path / "files.tsv"
	table.write_text("file\tname\ncells.tsv\tA\ncells.tsv.gz\tB\n")
	path = bam2fw(tmp_path, [fragments, fragments_gz], sizes, "--barcode-column",
		"4", "-f", "-p", threads, "-m", str(table))
	reader = FigWigReader(path)
	assert [track["group"] for track in reader.tracks] == [0, 0, 0, 1, 1, 1]
	y = reader.read("chr1", [0], 1000, dtype="float32")
	assert y.sum() > 0
	numpy.testing.assert_array_equal(y[0, :3], y[0, 3:])
	numpy.testing.assert_array_equal(reader.stats["sum"][:3], reader.stats[
		"sum"][3:])


def test_threads_and_units_give_the_same_file(tmp_path, fragments, fragments_gz,
		sizes, monkeypatch):
	one = bam2fw(tmp_path / "one", [fragments], sizes, "--barcode-column", "4",
		"-f", "-p", "1")
	# Units of 64 bytes of text, so that barcodes are found across units
	# and numbered afterwards.
	monkeypatch.setattr(figwig.bam2bw, "_IV_UNIT_BYTES", 64)
	units = bam2fw(tmp_path / "units", [fragments], sizes, "--barcode-column",
		"4", "-f", "-p", "3")
	# A plain-gzip file is read as a stream whatever -p is.
	stream = bam2fw(tmp_path / "gz", [fragments_gz], sizes, "--barcode-column",
		"4", "-f", "-p", "3")
	with open(one, "rb") as a, open(units, "rb") as b:
		assert a.read() == b.read()
	assert [t["name"] for t in FigWigReader(stream).tracks] == BARCODES
	reader, other = FigWigReader(one), FigWigReader(stream)
	numpy.testing.assert_array_equal(reader.stats["sum"], other.stats["sum"])


def test_bgzf_units(tmp_path, fragments, sizes):
	import pysam

	bgzf = str(tmp_path / "cells.bed.gz")
	pysam.tabix_compress(fragments, bgzf)
	one = bam2fw(tmp_path / "one", [fragments], sizes, "--barcode-column", "4",
		"-f")
	path = bam2fw(tmp_path / "bgzf", [bgzf], sizes, "--barcode-column", "4", "-f",
		"-p", "2")
	reader, other = FigWigReader(one), FigWigReader(path)
	numpy.testing.assert_array_equal(reader.stats["sum"], other.stats["sum"])
	assert [t["name"] for t in other.tracks] == BARCODES


def test_barcode_column_errors(tmp_path, fragments, bam, sizes):
	with pytest.raises(SystemExit, match="column number from 4"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "2")
	with pytest.raises(SystemExit, match="reads BED/tsv files of fragments"):
		bam2fw(tmp_path, [bam], sizes, "--barcode-column", "4")
	with pytest.raises(SystemExit, match="does not hold a chromosome, a start, "
			"an end and a barcode in column 6"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "6")
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\nAAAC-1\nAAAC-1\n")
	with pytest.raises(SystemExit, match="lists a barcode more than once"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "--cells",
			str(table))
	elsewhere = tmp_path / "elsewhere.tsv"
	elsewhere.write_text("chrZ\t1\t5\tAAAC-1\n")
	with pytest.raises(SystemExit, match="no fragment of .* is on a chromosome "
			"of .*; the file's chromosomes include 'chrZ'"):
		bam2fw(tmp_path, [str(elsewhere)], sizes, "--barcode-column", "4")
	table.write_text("barcode\nTTTT-1\n")
	path = bam2fw(tmp_path / "listed", [fragments], sizes, "--barcode-column",
		"4", "--cells", str(table))
	assert FigWigReader(path).stats["sum"].tolist() == [0]
	bad = tmp_path / "bad.tsv"
	bad.write_text("chr1\t100\tx\tAAAC-1\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		bam2fw(tmp_path, [str(bad)], sizes, "--barcode-column", "4")
	bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-ps", "700",
		"-f")


def test_barcode_column_discards_and_reports(tmp_path, fragments, sizes, capsys):
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f")
	out = capsys.readouterr().out
	assert "1 entries fell outside the provided chromosome sizes" in out
	assert "chr2: 1" in out
	reader = FigWigReader(path)
	assert reader.read("chr2", [480], 20, tracks="CCCT-1")[0, 0] == 1


def test_key_packing_with_many_chromosomes(tmp_path):
	"""The key holds the chromosome above the cell above the position; 3,000
	chromosomes take 12 bits of it, and the counts must still land on the
	right chromosome, cell and base."""

	chrom_sizes = [("c{}".format(k), 50_000) for k in range(3000)]
	sizes = str(write_chrom_sizes(tmp_path / "many.chrom.sizes", chrom_sizes))
	rng = numpy.random.default_rng(0)
	rows = []
	for _ in range(2000):
		chrom = "c{}".format(rng.integers(3000))
		start = int(rng.integers(49_000))
		rows.append((chrom, start, start + 100, "B{}".format(rng.integers(50))))
	path = write_intervals(tmp_path / "many.tsv", rows)
	out = bam2fw(tmp_path, [str(path)], sizes, "--barcode-column", "4", "-f")
	reader = FigWigReader(out)
	names = [track["name"] for track in reader.tracks]
	assert names == list(dict.fromkeys(barcode for *_, barcode in rows))
	chosen = set(rng.choice(3000, 20, replace=False).tolist()) | {int(
		rows[0][0][1:])}
	expected = {k: numpy.zeros((50, 50_000), dtype=numpy.int64) for k in chosen}
	totals = numpy.zeros(50, dtype=numpy.int64)
	for chrom, start, end, barcode in rows:
		totals[names.index(barcode)] += 2
		if int(chrom[1:]) in chosen:
			expected[int(chrom[1:])][names.index(barcode), [start, end - 1]] += 1
	for k in chosen:
		y = reader.read("c{}".format(k), [0], 50_000, dtype="int64")[0]
		numpy.testing.assert_array_equal(y, expected[k])
	numpy.testing.assert_array_equal(reader.stats["sum"], totals)


def test_too_many_cells_for_the_key(tmp_path, fragments, sizes, monkeypatch):
	"""With 3 chromosomes the key keeps 2 bits for them; 37 key bits leave 3
	for the cell, so 8 cells."""

	import figwig._cells

	monkeypatch.setattr(figwig._cells, "_KEY_BITS", 37)
	rows = [("chr1", 10 * k, 10 * k + 5, "B{}".format(k)) for k in range(9)]
	path = str(write_intervals(tmp_path / "nine.tsv", rows))
	with pytest.raises(SystemExit, match="holds more than 8 barcodes"):
		bam2fw(tmp_path, [path], sizes, "--barcode-column", "4")
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\n" + "".join("B{}\n".format(k) for k in range(9)))
	with pytest.raises(SystemExit, match="--cells lists 9 barcodes, more than "
			"the 8"):
		bam2fw(tmp_path, [path], sizes, "--barcode-column", "4", "--cells",
			str(table))
	out = bam2fw(tmp_path, [str(write_intervals(tmp_path / "eight.tsv",
		rows[:8]))], sizes, "--barcode-column", "4")
	assert len(FigWigReader(out).tracks) == 8


def test_malformed_fragments_files(tmp_path, sizes):
	import pysam

	def convert(name, text, *flags, binary=False):
		path = tmp_path / name
		if binary:
			path.write_bytes(text)
		else:
			path.write_text(text)
		return bam2fw(tmp_path / name.replace(".", "_"), [str(path)], sizes,
			"--barcode-column", "4", "-f", *flags)

	with pytest.raises(SystemExit, match="no fragment of .* was counted: the "
			"file holds none"):
		convert("empty.tsv", "")
	with pytest.raises(SystemExit, match="the file holds none"):
		convert("comments.tsv", "# id=x\n#\n\n")
	out = convert("crlf.tsv", "chr1\t100\t150\tAAAC-1\t1\r\nchr1\t200\t250\t"
		"AAAG-1\t2\r\n")
	assert FigWigReader(out).stats["sum"].tolist() == [2, 2]
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("utf8.tsv", b"chr1\t100\t150\tAAAC\xc3\xa9-1\t1\n", binary=True)
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("huge.tsv", "chr1\t1000000000000000000\t150\tAAAC-1\t1\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("float.tsv", "chr1\t100.0\t150\tAAAC-1\t1\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("short.tsv", "chr1\t100\t150\n")
	with pytest.raises(SystemExit, match="does not hold a chromosome"):
		convert("bare_cr.tsv", "chr1\t100\t150\tAAAC-1\t1\rchr1\t200\t250\t"
			"AAAC-1\t1\n")
	out = convert("negative.tsv", "chr1\t-5\t3\tAAAC-1\t1\nchr1\t5\t3\tAAAC-1\t1\n")
	assert FigWigReader(out).stats["sum"].tolist() == [3]

	# A gzip file cut inside a member, plain or BGZF, and a BGZF file cut at
	# a block boundary, which lacks the empty block that ends one.
	lines = "".join("chr1\t{}\t{}\tB{}\t1\n".format(k, k + 50, k % 7) for k in
		range(20_000))
	(tmp_path / "whole.tsv").write_text(lines)
	bgzf = str(tmp_path / "whole.tsv.gz")
	pysam.tabix_compress(str(tmp_path / "whole.tsv"), bgzf)
	with open(bgzf, "rb") as handle:
		data = handle.read()
	assert len(data) > 2 * 28
	# Without its end block, a BGZF file is refused before it is read; the
	# cut inside a member is what the gzip stream finds otherwise.
	with pytest.raises(SystemExit, match="ends without the empty block that "
			"ends a BGZF file, so it was cut short"):
		convert("half.tsv.gz", data[:len(data) // 2], binary=True)
	with pytest.raises(SystemExit, match="ends without the empty block"):
		convert("blocks.tsv.gz", data[:-28], "-p", "2", binary=True)
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("half_marked.tsv.gz", data[:len(data) // 2] + data[-28:],
			binary=True)
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("half_marked.tsv.gz", data[:len(data) // 2] + data[-28:], "-p",
			"2", binary=True)
	import gzip
	plain = gzip.compress(lines.encode())
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("plain.tsv.gz", plain[:len(plain) // 2], binary=True)
	with pytest.raises(SystemExit, match="is cut short, or is not a gzip file"):
		convert("garbage.tsv.gz", b"\x1f\x8b\x08\x00" + bytes(100), binary=True)


def test_bgzf_blocks_as_units(tmp_path, sizes, monkeypatch):
	"""A BGZF file of several blocks, scanned a block per unit, with lines
	cut by block ends and barcodes found in every unit."""

	import pysam

	rng = numpy.random.default_rng(1)
	rows = [("chr1", int(k), int(k) + 10, "B{}".format(rng.integers(40))) for k
		in sorted(rng.integers(0, 990, 30_000))]
	path = write_intervals(tmp_path / "blocks.tsv", rows)
	bgzf = str(tmp_path / "blocks.tsv.gz")
	pysam.tabix_compress(str(path), bgzf)
	assert os.path.getsize(bgzf) > 3 * 65_536 // 4
	one = bam2fw(tmp_path / "one", [str(path)], sizes, "--barcode-column", "4",
		"-f")
	monkeypatch.setattr(figwig.bam2bw, "_IV_UNIT_BLOCKS", 1)
	units = bam2fw(tmp_path / "units", [bgzf], sizes, "--barcode-column", "4",
		"-f", "-p", "3")
	with open(one, "rb") as a, open(units, "rb") as b:
		assert a.read() == b.read()
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\n" + "".join("B{}\n".format(k) for k in
		range(0, 40, 2)))
	listed = bam2fw(tmp_path / "listed", [bgzf], sizes, "--barcode-column", "4",
		"-f", "-p", "3", "--cells", str(table))
	reader, other = FigWigReader(one), FigWigReader(listed)
	names = [track["name"] for track in reader.tracks]
	numpy.testing.assert_array_equal(other.stats["sum"], reader.stats["sum"][[
		names.index("B{}".format(k)) for k in range(0, 40, 2)]])


def test_barcode_column_other_options(tmp_path, fragments, sizes):
	table = tmp_path / "cells.tsv"
	table.write_text("barcode\tname\nAAAC-1\tx\nAAAG-1\tx\n")
	with pytest.raises(SystemExit, match="x"):
		bam2fw(tmp_path / "dup", [fragments], sizes, "--barcode-column", "4",
			"--cells", str(table))
	table.write_text("barcode\nCCCT-1\nAAAC-1\n")
	path = bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f",
		"-u", "-r", "--cells", str(table), "-z", "2", "-t", "256", "-l", "0",
		"--dtype", "float32", "-g", "hg38")
	reader = FigWigReader(path)
	assert reader.tile == 256 and reader.genome == "hg38" and reader.zooms == (
		4, 16)
	assert [track["dtype"] for track in reader.tracks] == ["float32"] * 2
	# -r divides by the depth of the cells kept: CCCT-1 and AAAC-1 together.
	plain = FigWigReader(bam2fw(tmp_path / "plain", [fragments], sizes,
		"--barcode-column", "4", "-f", "--cells", str(table)))
	depth = plain.stats["sum"].sum()
	numpy.testing.assert_allclose(reader.stats["sum"], plain.stats["sum"] / depth,
		rtol=1e-6)
	with pytest.raises(SystemExit, match="--mate_pairs only supports BAM/SAM"):
		bam2fw(tmp_path / "mp", [fragments], sizes, "--barcode-column", "4", "-mp")


def test_barcode_column_command_line(tmp_path, fragments, sizes):
	name = str(tmp_path / "cli")
	result = subprocess.run([sys.executable, "-m", "figwig", "bam2fw", fragments,
		"-s", sizes, "-n", name, "-p", "2", "--barcode-column", "4", "-f", "-v"],
		env=ENV, capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert [track["name"] for track in FigWigReader(name + ".fw").tracks] == \
		BARCODES
	assert "chrUN encountered in input but not in FASTA/chrom sizes." in \
		result.stdout


def test_default_groups_stay_within_a_chunk(tmp_path, sizes, monkeypatch):
	"""Every cell of a file goes in one group unless a chunk could not hold
	the group's values, tracks times tile; then the cells are cut into the
	fewest groups that fit (64 values at a tile of 16: 4 cells each)."""

	monkeypatch.setattr(figwig.bam2fw, "_CHUNK_VALUES", 64)
	rows = [("chr1", 10 * k, 10 * k + 5, "B{}".format(k)) for k in range(10)]
	path = str(write_intervals(tmp_path / "ten.tsv", rows))
	out = bam2fw(tmp_path, [path], sizes, "--barcode-column", "4", "-t", "16")
	assert [track["group"] for track in FigWigReader(out).tracks] == [0, 0, 0, 0,
		1, 1, 1, 1, 2, 2]
	out = bam2fw(tmp_path / "three", [path], sizes, "--barcode-column", "4",
		"-t", "16", "--groups", "3")
	assert [track["group"] for track in FigWigReader(out).tracks] == [0, 0, 0,
		1, 1, 1, 2, 2, 2, 3]


def test_entries_arrive_in_the_writers_order(tmp_path, sizes):
	"""The scanner's keys sort into the order the writer encodes in, tile by
	tile and cell by cell, so bam2fw hands them over presorted; the file must
	be the one the same entries make through the sorting path, given in any
	order, and a duplicate must still be named by the position given."""

	from figwig import FigWigWriter

	rng = numpy.random.default_rng(2)
	rows = [("chr1", int(k), int(k) + 7, "B{}".format(rng.integers(30))) for k
		in sorted(rng.integers(0, 990, 5000))]
	rows += [("chr2", int(k), int(k) + 7, "B{}".format(rng.integers(30))) for k
		in sorted(rng.integers(0, 490, 2000))]
	path = str(write_intervals(tmp_path / "many.tsv", rows))
	out = bam2fw(tmp_path, [path], sizes, "--barcode-column", "4", "-f", "-t",
		"256")
	reader = FigWigReader(out)
	names = [track["name"] for track in reader.tracks]

	entries = {}
	for chrom, start, end, barcode in rows:
		for position in (start, end - 1):
			entries[(chrom, position, barcode)] = entries.get((chrom, position,
				barcode), 0) + 1
	keys = list(entries)
	rng.shuffle(keys)
	other = str(tmp_path / "shuffled.fw")
	with FigWigWriter(other, CHROM_SIZES, names, tile=256, level=1,
			groups=len(names)) as writer:
		for chrom in ("chr1", "chr2"):
			these = [key for key in keys if key[0] == chrom]
			writer.write_sparse(chrom, [key[1] for key in these], [key[2] for
				key in these], [entries[key] for key in these])
	with open(out, "rb") as a, open(other, "rb") as b:
		assert a.read() == b.read()

	with FigWigWriter(str(tmp_path / "dup.fw"), CHROM_SIZES, ["a", "b"],
			groups=2) as writer:
		with pytest.raises(ValueError, match="entries 1 and 2 both name base 5"):
			writer.write_sparse("chr1", [3, 5, 5], [0, 1, 1], [1, 1, 1])
		writer._sorted_entries = True
		with pytest.raises(ValueError, match="entry 2 on 'chr1' is at 2000, "
				"outside the chromosome"):
			writer.write_sparse("chr1", [3, 5, 2000], [0, 1, 1], [1, 1, 1])
		writer.write_sparse("chr1", [3, 5], [0, 1], [1, 1])


def test_infinite_scale_factor_is_refused(tmp_path, fragments, sizes):
	"""The presorted hand-off promises finite values, so it is not used
	when the scale factor is not finite, and the writer refuses as it would
	any infinite value."""

	with pytest.raises(SystemExit, match="values must be finite or NaN"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-f", "-sf",
			"inf")
	path = bam2fw(tmp_path / "zero", [fragments], sizes, "--barcode-column", "4",
		"-f", "-sf", "0")
	assert FigWigReader(path).stats["sum"].tolist() == [0, 0, 0]


# --coverage: every base from a read's start to its end.

COVERAGE_READS = [
	("chr1", 100, 50, False),
	("chr1", 100, 50, False),
	("chr1", 120, 30, True),
	("chr1", 140, 40, False),
	("chr1", 980, 30, True),
	("chr2", 0, 25, False),
	("chr2", 490, 20, True),
	("chr3", 0, 10, False),
	("chrUN", 10, 10, False),
]


@pytest.fixture(scope="session")
def coverage_bam(data_dir):
	return str(write_bam(data_dir / "coverage.bam", BAM_CHROM_SIZES,
		COVERAGE_READS))


@pytest.fixture(scope="session")
def coverage_bed(data_dir):
	return str(write_intervals(data_dir / "coverage.bed", [(chrom, start, start +
		length) for chrom, start, length, _ in COVERAGE_READS]))


def pileup(pos_shift=0, neg_shift=0, stranded=True):
	"""Every base from each read's shifted start to its shifted end, per
	strand, clipped to the chromosome: the oracle for --coverage."""

	expected = {chrom: {"+": numpy.zeros(n), "-": numpy.zeros(n)} for chrom, n
		in CHROM_SIZES}
	for chrom, start, length, reverse in COVERAGE_READS:
		if chrom not in expected:
			continue
		a, b = start + pos_shift, start + length + neg_shift
		expected[chrom]["-" if reverse and stranded else "+"][max(a, 0):max(
			b, 0)] += 1
	return expected


def assert_coverage(bigwigs, expected, strands, scale=1.0):
	for chrom, length in CHROM_SIZES:
		for path, strand in zip(bigwigs, strands):
			numpy.testing.assert_array_equal(bigwig_values(path, chrom, length),
				(expected[chrom][strand] * scale).astype(numpy.float32))


@pytest.mark.parametrize("flags, shifts", [((), (0, 0)), (("-u",), (0, 0)),
	(("-ps", "4", "-ns", "-5"), (4, -5)), (("-p", "2"), (0, 0))])
def test_coverage_is_the_pileup_of_read_spans(tmp_path, coverage_bam, sizes,
		flags, shifts):
	stranded = "-u" not in flags
	bigwigs = bam2bw(tmp_path, coverage_bam, sizes, "-c", *flags)
	expected = pileup(*shifts, stranded=stranded)
	strands = ["+", "-"] if stranded else ["+"]
	assert_coverage(bigwigs, expected, strands)

	path = bam2fw(tmp_path, [coverage_bam], sizes, "-c", *flags)
	reader = FigWigReader(path)
	names = ["coverage"] if "-u" in flags else ["coverage.+", "coverage.-"]
	assert [track["name"] for track in reader.tracks] == names
	assert [track["group"] for track in reader.tracks] == list(range(len(names)))
	assert reader.tracks[0]["dtype"] == "uint8"
	assert_tracks_match_bigwigs(path, names, bigwigs)


def test_coverage_of_a_bed_file_and_of_a_sam_file(tmp_path, coverage_bam,
		coverage_bed, sizes):
	import pysam

	expected = pileup(stranded=False)
	assert_coverage(bam2bw(tmp_path, coverage_bed, sizes, "-c", "-u"), expected,
		["+"])
	assert_coverage(bam2bw(tmp_path / "threads", coverage_bed, sizes, "-c", "-u",
		"-p", "2"), expected, ["+"])
	sam = str(tmp_path / "coverage.sam")
	pysam.view("-h", "-o", sam, coverage_bam, catch_stdout=False)
	assert_coverage(bam2bw(tmp_path / "sam", sam, sizes, "-c"), pileup(),
		["+", "-"])


def test_coverage_scaling_and_depth(tmp_path, coverage_bam, sizes):
	expected = pileup()
	assert_coverage(bam2bw(tmp_path, coverage_bam, sizes, "-c", "-sf", "0.5"),
		expected, ["+", "-"], scale=0.5)
	# The depth is the reads that start inside their chromosome: every read
	# but the one on chrUN, 8.
	assert_coverage(bam2bw(tmp_path / "r", coverage_bam, sizes, "-c", "-r"),
		expected, ["+", "-"], scale=1 / 8)
	path = bam2fw(tmp_path / "fw", [coverage_bam], sizes, "-c", "-r", "-sf",
		"2")
	reader = FigWigReader(path)
	for chrom, length in CHROM_SIZES:
		y = reader.read(chrom, [0], length, dtype="float64")
		numpy.testing.assert_allclose(y[0, 0], expected[chrom]["+"] * 2 / 8)


def chunks_of(path, group):
	"""A group's kind bytes and chunks, tile by tile, wherever in the file
	they lie."""

	kinds, sizes, _, offsets, _ = FigWigReader(path)._group_index(group)
	with open(path, "rb") as handle:
		data = handle.read()
	return kinds.tolist(), [data[o:o + n] for o, n in zip(offsets.tolist(),
		sizes.tolist())]


@pytest.mark.parametrize("spacing", [0, 8, 16, 1 << 30])
@pytest.mark.parametrize("flags, shifts", [((), (0, 0)), (("-u", "-ps", "4",
	"-ns", "-5"), (4, -5)), (("-sf", "0.5"), (0, 0))])
def test_coverage_in_pieces_is_the_file_of_one_piece(tmp_path, coverage_bam,
		sizes, monkeypatch, flags, shifts, spacing):
	"""bam2fw -c on threads writes a chromosome's coverage a piece at a
	time, a whole number of tiles each, as a dense window or, for a stretch
	of pieces of few runs (here every piece, some, or none), as its runs;
	on one thread as intervals. Pieces of 64 bases over tiles of 16, which runs cross and
	some of which hold no run, give the file one piece per chromosome
	gives, byte for byte, every tile the chunk the intervals give, though
	the two groups' chunks may lie in another order, and the pileup."""

	monkeypatch.setattr(figwig.bam2fw, "_RUN_SPACING", spacing)
	flags = ("-c", "-t", "16") + flags
	intervals = bam2fw(tmp_path / "intervals", [coverage_bam], sizes, *flags)
	whole = bam2fw(tmp_path / "whole", [coverage_bam], sizes, "-p", "2", *flags)
	monkeypatch.setattr(figwig.bam2fw, "_COVERAGE_PIECE", 64)
	pieces = bam2fw(tmp_path / "pieces", [coverage_bam], sizes, "-p", "2",
		*flags)
	with open(whole, "rb") as a, open(pieces, "rb") as b:
		assert a.read() == b.read()
	for g in range(len(FigWigReader(pieces).tracks)):
		assert chunks_of(intervals, g) == chunks_of(pieces, g)

	expected = pileup(*shifts, stranded="-u" not in flags)
	scale = 0.5 if "-sf" in flags else 1.0
	reader = FigWigReader(pieces)
	for chrom, length in CHROM_SIZES:
		y = reader.read(chrom, [0], length, dtype="float64")[0]
		for k, strand in enumerate(["+"] if "-u" in flags else ["+", "-"]):
			numpy.testing.assert_array_equal(y[k], expected[chrom][strand] * scale)


def test_coverage_mixes_runs_and_windows(tmp_path, sizes, monkeypatch):
	"""On threads, stretches of pieces of few runs between pieces of many go
	in as runs, each stretch in one call, and the others as windows; the
	file is the one a single thread writes as intervals, byte for byte."""

	reads = []
	for piece in range(15):
		if piece % 4 == 2:
			reads += [("chr1", 64 * piece + 3 * k, 2, False) for k in range(20)]
		else:
			reads.append(("chr1", 64 * piece + 5, 10, False))
	bam = str(write_bam(tmp_path / "mix.bam", BAM_CHROM_SIZES, reads))
	monkeypatch.setattr(figwig.bam2fw, "_COVERAGE_PIECE", 64)
	monkeypatch.setattr(figwig.bam2fw, "_RUN_SPACING", 32)
	calls = []
	original = figwig.bam2fw._dense_runs

	def counted(*args):
		calls.append(1)
		return original(*args)

	monkeypatch.setattr(figwig.bam2fw, "_dense_runs", counted)
	one = bam2fw(tmp_path / "one", [bam], sizes, "-c", "-u", "-t", "16")
	assert calls == []
	threaded = bam2fw(tmp_path / "threaded", [bam], sizes, "-c", "-u", "-t", "16",
		"-p", "2")
	assert len(calls) == 4
	with open(one, "rb") as a, open(threaded, "rb") as b:
		assert a.read() == b.read()


def test_coverage_reports_the_bases_clipped_away(tmp_path, coverage_bam, sizes,
		capsys):
	bam2bw(tmp_path, coverage_bam, sizes, "-c")
	out = capsys.readouterr().out
	# chr1:980+30 runs 10 bases past 1000, chr2:490+20 runs 10 past 500.
	assert "20 entries fell outside the provided chromosome sizes" in out
	assert "chr1: 10" in out and "chr2: 10" in out


def test_coverage_refuses_what_it_cannot_mean(tmp_path, coverage_bam, fragments,
		sizes):
	# argparse refuses -c with -f or -3p, with its own message on stderr.
	with pytest.raises(SystemExit):
		bam2bw(tmp_path, coverage_bam, sizes, "-c", "-f")
	with pytest.raises(SystemExit):
		bam2bw(tmp_path, coverage_bam, sizes, "-c", "-3p")
	with pytest.raises(ValueError, match="--coverage does not take --mate_pairs"):
		bam2bw(tmp_path, coverage_bam, sizes, "-c", "-mp")
	with pytest.raises(SystemExit, match="--coverage is not taken with it"):
		bam2fw(tmp_path, [fragments], sizes, "--barcode-column", "4", "-c")


def test_coverage_counts_aligned_blocks(tmp_path, sizes):
	"""A spliced read covers its two exons and not the intron, a deletion is
	not covered, an insertion and soft clips consume no reference, and the
	shifts move the first base and the end: pysam's blocks, as deepTools
	counts them, from the kernel and from the pysam loop alike."""

	import pysam

	header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": chrom,
		"LN": length} for chrom, length in BAM_CHROM_SIZES]}
	records = [("chr1", 100, "20M30N20M", False), ("chr1", 200, "10M5D10M", True),
		("chr1", 300, "5S10M3I10M5S", False), ("chr1", 400, "30M", False),
		("chr2", 10, "40I", False)]
	path = str(tmp_path / "blocks.bam")
	with pysam.AlignmentFile(path, "wb", header=header) as handle:
		for k, (chrom, start, cigar, reverse) in enumerate(records):
			read = pysam.AlignedSegment(handle.header)
			read.query_name = "r{}".format(k)
			read.reference_id = handle.get_tid(chrom)
			read.reference_start = start
			read.cigarstring = cigar
			read.flag = 16 if reverse else 0
			read.mapping_quality = 60
			length = sum(n for n, op in [(int(x), y) for x, y in __import__(
				"re").findall(r"(\d+)([MIDNSHP=X])", cigar)] if op in "MIS=X")
			read.query_sequence = "A" * length
			handle.write(read)
	pysam.index(path)
	sam = str(tmp_path / "blocks.sam")
	pysam.view("-h", "-o", sam, path, catch_stdout=False)

	expected = {"+": numpy.zeros(1000), "-": numpy.zeros(1000)}
	expected["+"][100:120] += 1
	expected["+"][150:170] += 1
	expected["-"][200:210] += 1
	expected["-"][215:225] += 1
	expected["+"][300:320] += 1
	expected["+"][400:430] += 1
	for source in (path, sam):
		for strand, bigwig in zip("+-", bam2bw(tmp_path / ("out_" + os.path.basename(
				source)[-3:]), source, sizes, "-c")):
			numpy.testing.assert_array_equal(bigwig_values(bigwig, "chr1", 1000),
				expected[strand].astype(numpy.float32))
	shifted = {"+": numpy.zeros(1000), "-": numpy.zeros(1000)}
	shifted["+"][104:120] += 1
	shifted["+"][150:165] += 1
	shifted["-"][204:210] += 1
	shifted["-"][215:220] += 1
	shifted["+"][304:315] += 1
	shifted["+"][404:425] += 1
	for strand, bigwig in zip("+-", bam2bw(tmp_path / "shift", path, sizes, "-c",
			"-ps", "4", "-ns", "-5")):
		numpy.testing.assert_array_equal(bigwig_values(bigwig, "chr1", 1000),
			shifted[strand].astype(numpy.float32))
	# A shift that empties a read leaves it uncovered rather than negative.
	for strand, bigwig in zip("+-", bam2bw(tmp_path / "empty", path, sizes, "-c",
			"-ps", "40")):
		values = bigwig_values(bigwig, "chr1", 1000)
		assert values.min() >= 0 and values[440:470].sum() == 0


# --coverage, the edges: the run decoder itself, inputs the other paths
# take, batches, clipping at both ends, several files, the options.

def test_coverage_runs_from_events():
	"""Overlapping reads, coverage that returns to 0 in the middle, runs
	clipped at both ends of the chromosome, a read wholly outside it, and
	the number of bases clipped away, straight from the counted events."""

	from figwig.bam2bw import _coverage_runs
	from figwig.bam2bw import _read_count

	def events(spans):
		keys = []
		for start, end in spans:
			keys += [2 * start, 2 * end + 1]
		positions, counts = numpy.unique(numpy.array(keys), return_counts=True)
		return positions, counts

	positions, counts = events([(10, 20), (15, 25), (15, 25), (40, 50)])
	starts, ends, values, lost = _coverage_runs(positions, counts, 100)
	assert starts.tolist() == [10, 15, 20, 40]
	assert ends.tolist() == [15, 20, 25, 50]
	assert values.tolist() == [1, 3, 2, 1] and lost == 0

	positions, counts = events([(-5, 5), (95, 105), (200, 210), (-20, -10)])
	starts, ends, values, lost = _coverage_runs(positions, counts, 100)
	assert starts.tolist() == [0, 95] and ends.tolist() == [5, 100]
	assert values.tolist() == [1, 1]
	assert lost == 5 + 5 + 10 + 10
	assert _read_count(positions, counts, 100) == 1

	empty = numpy.empty(0, dtype=numpy.int64)
	assert _coverage_runs(empty, empty, 100)[3] == 0
	assert len(_coverage_runs(empty, empty, 100)[0]) == 0


def fixed_regions(width):
	"""A stand-in for figwig._indexed._regions: regions of `width` bases,
	finer than the 16 kb windows the real one cuts on."""

	def regions(files, chrom, size, longest=None):
		bounds = list(range(0, max(size, 1), width)) + [max(size, 1)]
		return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
	return regions


@pytest.fixture(autouse=True)
def indexed_at_any_size(monkeypatch):
	"""The test files are far smaller than the inputs -c reads through the
	indexed reader, which takes every size here, as it takes a large file."""

	import figwig._indexed

	monkeypatch.setattr(figwig._indexed, "_MIN_CORES", 1)
	monkeypatch.setattr(figwig._indexed, "_MIN_MAPPED", 0)
	monkeypatch.setattr(figwig._indexed, "_MIN_BYTES", 0)
	monkeypatch.setattr(figwig._indexed, "_MIN_FILE_MAPPED", 0)


def test_indexed_coverage_only_from_its_sizes(tmp_path, indexed_bams, sizes,
		monkeypatch, capsys):
	"""-c reads through the indexed reader on `_MIN_CORES` cores or more:
	bam2bw from `_MIN_BYTES` and `_MIN_MAPPED` mapped reads in its files'
	indexes, both summed, and bam2fw when each of its files has `_MIN_BYTES`
	and `_MIN_FILE_MAPPED` mapped reads; otherwise the whole-file reader
	gives the output."""

	import pysam
	import figwig._indexed

	_, indexed = indexed_bams
	small = sorted_indexed(tmp_path / "small.bam", BAM_CHROM_SIZES, [("chr1", 10 *
		i, 30, bool(i % 2)) for i in range(40)])
	mapped, few = (sum(s.mapped for s in pysam.AlignmentFile(path)
		.get_index_statistics()) for path in (indexed, small))
	calls = []
	original = figwig._indexed._indexed_coverage

	def counted(*args, **kwargs):
		calls.append(1)
		return original(*args, **kwargs)

	monkeypatch.setattr(figwig._indexed, "_indexed_coverage", counted)

	def took(command, files, cores=1, **minimums):
		for name, value in minimums.items():
			monkeypatch.setattr(figwig._indexed, name, value)
		calls.clear()
		if command == "bam2bw":
			figwig.bam2bw.main(files + ["-s", sizes, "-n", str(tmp_path / "out"),
				"-c", "-p", str(cores)])
		else:
			bam2fw(tmp_path, files, sizes, "-c", "--force", "-p", str(cores))
		capsys.readouterr()
		return len(calls) > 0

	assert took("bam2bw", [indexed], _MIN_MAPPED=mapped)
	assert not took("bam2bw", [indexed], _MIN_MAPPED=mapped + 1)
	assert took("bam2bw", [indexed, indexed], _MIN_MAPPED=2 * mapped)
	assert not took("bam2bw", [indexed, indexed], _MIN_MAPPED=2 * mapped + 1)
	size = os.path.getsize(indexed)
	assert took("bam2bw", [indexed, indexed], _MIN_MAPPED=0, _MIN_BYTES=2 * size)
	assert not took("bam2bw", [indexed, indexed], _MIN_BYTES=2 * size + 1)
	monkeypatch.setattr(figwig._indexed, "_MIN_BYTES", 0)

	size, least = os.path.getsize(indexed), os.path.getsize(small)
	assert took("bam2fw", [indexed], _MIN_BYTES=size)
	assert not took("bam2fw", [indexed], _MIN_BYTES=size + 1)
	assert took("bam2fw", [indexed, small], _MIN_BYTES=least)
	assert not took("bam2fw", [indexed, small], _MIN_BYTES=least + 1)
	assert took("bam2fw", [indexed, small], _MIN_BYTES=0, _MIN_FILE_MAPPED=few)
	assert not took("bam2fw", [indexed, small], _MIN_FILE_MAPPED=few + 1)

	monkeypatch.setattr(figwig._indexed, "_MIN_FILE_MAPPED", 0)
	monkeypatch.setattr(figwig._indexed, "_MIN_MAPPED", 0)
	for command in ("bam2bw", "bam2fw"):
		assert not took(command, [indexed], cores=1, _MIN_CORES=2)
		assert took(command, [indexed], cores=2)


@pytest.fixture(scope="session")
def indexed_bams(data_dir):
	"""Random reads of every CIGAR shape and flag, several reaching past a
	chromosome's end or on a chromosome the sizes leave out, as written (no
	index, unsorted) and as a coordinate-sorted, indexed copy."""

	import pysam

	rng = numpy.random.default_rng(23)
	reads = []
	for i in range(400):
		chrom, size = BAM_CHROM_SIZES[rng.integers(0, len(BAM_CHROM_SIZES))]
		start = int(rng.integers(0, size))
		a, b = int(rng.integers(1, 40)), int(rng.integers(1, 40))
		shape = rng.integers(0, 4)
		cigar = [[(0, a)], [(0, a), (3, int(rng.integers(1, 300))), (0, b)],
			[(0, a), (2, int(rng.integers(1, 20))), (0, b)], [(4, 3), (0, a), (1, 2),
			(0, b), (4, 5)]][shape]
		flag = int(rng.choice([0, 0, 0, 1024, 256, 2048, 512]))
		reads.append((chrom, start, a, bool(rng.integers(0, 2)), cigar, flag, int(
			rng.integers(0, 61))))
	unmapped = [("chr1", 500, 30), ("chr2", 10, 20)]
	written = write_bam(data_dir / "random.bam", BAM_CHROM_SIZES, reads,
		unmapped=unmapped)
	indexed = str(data_dir / "random_sorted.bam")
	pysam.sort("-o", indexed, str(written))
	pysam.index(indexed)
	return str(written), indexed


def assert_same_figwig_values(a, b):
	ra, rb = FigWigReader(a), FigWigReader(b)
	assert len(ra.tracks) == len(rb.tracks)
	for chrom, length in CHROM_SIZES:
		numpy.testing.assert_array_equal(ra.read(chrom, [0], length), rb.read(chrom,
			[0], length))


@pytest.mark.parametrize("flags", [(), ("-u",), ("-ps", "3", "-ns", "-2"), ("-ps",
	"-7", "-ns", "9", "-u"), ("-r",), ("-sf", "0.5", "-p", "4"), ("-u", "-r",
	"-p", "3"), ("-u", "-p", "8"), ("-ps", "-7", "-ns", "9", "-sf", "2", "-p",
	"12")])
def test_indexed_coverage_is_the_whole_file_coverage(tmp_path, indexed_bams,
		sizes, flags, monkeypatch, capsys):
	"""-c of a sorted, indexed BAM is read region by region (here regions of
	64 bases, which most reads cross) and gives the bigWigs and figwig file
	the whole-file reader gives the same reads unsorted, and the same lines."""

	import figwig._indexed

	unsorted, indexed = indexed_bams
	monkeypatch.setattr(figwig._indexed, "_regions", fixed_regions(64))
	calls = []
	original = figwig._indexed._indexed_coverage

	def counted(*args, **kwargs):
		calls.append(1)
		return original(*args, **kwargs)

	monkeypatch.setattr(figwig._indexed, "_indexed_coverage", counted)
	flags = ("-c",) + flags
	capsys.readouterr()
	whole = bam2bw(tmp_path / "whole", unsorted, sizes, *flags)
	whole_lines = capsys.readouterr().out
	assert calls == []
	by_region = bam2bw(tmp_path / "indexed", indexed, sizes, *flags)
	assert calls and capsys.readouterr().out == whole_lines
	assert "entries fell outside" in whole_lines
	for a, b in zip(whole, by_region):
		with open(a, "rb") as fa, open(b, "rb") as fb:
			assert fa.read() == fb.read()

	calls.clear()
	whole = bam2fw(tmp_path / "fw_whole", [unsorted], sizes, *flags)
	whole_lines = capsys.readouterr().out
	assert calls == []
	by_region = bam2fw(tmp_path / "fw_indexed", [indexed], sizes, *flags)
	assert calls and capsys.readouterr().out == whole_lines
	assert_same_figwig_values(whole, by_region)


@pytest.mark.parametrize("flags", [("-u",), ("-ps", "-40", "-ns", "300"), ("-u",
	"-ps", "-30000", "-ns", "40000")])
def test_indexed_coverage_over_index_windows(tmp_path, monkeypatch, flags):
	"""On chromosomes of many 16 kb index windows and many BGZF blocks, with
	regions of at most 32 kb cut sooner every 2 kB of the BAM and reads
	whose splices span tens of kb, each region reads from the index's offset
	for its first base to the first read of a 16 kb bin past its end, and
	gives what the whole file does."""

	import pysam
	import figwig._indexed

	monkeypatch.setattr(figwig._indexed, "_REGION", 1 << 15)
	monkeypatch.setattr(figwig._indexed, "_REGION_BYTES", 1 << 11)
	monkeypatch.setattr(figwig.bam2fw, "_COVERAGE_PIECE", 32768)
	chrom_sizes = [("big", 200000), ("small", 30000)]
	rng = numpy.random.default_rng(29)
	reads = []
	for _ in range(40000):
		chrom, size = chrom_sizes[int(rng.integers(0, 2))]
		start = int(rng.integers(0, size - 100))
		a, b = int(rng.integers(5, 60)), int(rng.integers(5, 60))
		splice = int(rng.integers(1, 60000)) if rng.random() < 0.05 else 0
		cigar = [(0, a), (3, splice), (0, b)] if splice else [(0, a + b)]
		reads.append((chrom, start, a + b, bool(rng.integers(0, 2)), cigar))
	written = write_bam(tmp_path / "long.bam", chrom_sizes, reads, seed=31)
	os.makedirs(tmp_path / "sorted")
	indexed = str(tmp_path / "sorted" / "long.bam")
	pysam.sort("-o", indexed, str(written))
	pysam.index(indexed)
	sizes_path = str(write_chrom_sizes(tmp_path / "long.sizes", chrom_sizes))

	whole = bam2bw(tmp_path / "whole", str(written), sizes_path, "-c", *flags)
	by_region = bam2bw(tmp_path / "indexed", indexed, sizes_path, "-c", *flags)
	for a, b in zip(whole, by_region):
		with open(a, "rb") as fa, open(b, "rb") as fb:
			assert fa.read() == fb.read()

	# Pieces of 32 kb, which runs of the shifted reads cross, are written
	# as the whole-file path writes them: the same bytes for one track.
	whole = bam2fw(tmp_path / "fw_whole", [str(written)], sizes_path, "-c", *flags)
	by_region = bam2fw(tmp_path / "fw_indexed", [indexed], sizes_path, "-c",
		*flags)
	ra, rb = FigWigReader(whole), FigWigReader(by_region)
	for chrom, length in chrom_sizes:
		numpy.testing.assert_array_equal(ra.read(chrom, [0], length), rb.read(chrom,
			[0], length))
	if "-u" in flags:
		with open(whole, "rb") as fa, open(by_region, "rb") as fb:
			assert fa.read() == fb.read()


def test_indexed_coverage_keeps_to_its_cores(tmp_path, indexed_bams, sizes,
		monkeypatch):
	"""On one core every region is read on the main thread, between writes,
	and bam2fw writes runs; on two, regions are read on another thread; on
	eight, bam2fw writes each region's depth."""

	import threading
	import figwig._indexed

	unsorted, indexed = indexed_bams
	monkeypatch.setattr(figwig._indexed, "_regions", fixed_regions(64))
	calls = []
	for name in ("_region_runs", "_region_dense"):
		original = getattr(figwig._indexed, name)

		def recorded(*args, _name=name, _original=original):
			calls.append((_name, threading.current_thread() is
				threading.main_thread()))
			return _original(*args)

		monkeypatch.setattr(figwig._indexed, name, recorded)

	bam2bw(tmp_path / "bw1", indexed, sizes, "-c")
	bam2fw(tmp_path / "fw1", [indexed], sizes, "-c")
	assert calls and set(calls) == {("_region_runs", True)}
	calls.clear()
	bam2bw(tmp_path / "bw2", indexed, sizes, "-c", "-p", "2")
	bam2fw(tmp_path / "fw2", [indexed], sizes, "-c", "-p", "2")
	assert calls and set(calls) == {("_region_runs", False)}
	calls.clear()
	bam2bw(tmp_path / "bw8", indexed, sizes, "-c", "-p", "8")
	bam2fw(tmp_path / "fw8", [indexed], sizes, "-c", "-p", "8")
	assert {name for name, _ in calls} == {"_region_runs", "_region_dense"}
	assert not any(main for _, main in calls)


@pytest.mark.parametrize("spacing", [1, 8, 16, 1 << 30])
@pytest.mark.parametrize("flags", [("-u",), ("-ps", "-7", "-ns", "9", "-sf",
	"2")])
def test_indexed_depth_goes_to_the_writer_as_runs_or_windows(tmp_path,
		indexed_bams, sizes, monkeypatch, spacing, flags):
	"""On threads, a region's depth goes to the writer as one window or, when
	it changes value less often than once per `_RUN_SPACING` bases, as runs;
	either gives the whole file's values, and for one track, with regions of
	whole tiles as the index's 16 kb windows are of the default tile, the
	same chunks."""

	import figwig._indexed

	unsorted, indexed = indexed_bams
	monkeypatch.setattr(figwig._indexed, "_regions", fixed_regions(64))
	monkeypatch.setattr(figwig._indexed, "_RUN_SPACING", spacing)
	flags = ("-c", "-t", "16") + flags
	whole = bam2fw(tmp_path / "whole", [unsorted], sizes, *flags)
	got = bam2fw(tmp_path / "got", [indexed], sizes, "-p", "8", *flags)
	assert_same_figwig_values(whole, got)
	if "-u" in flags:
		assert chunks_of(whole, 0) == chunks_of(got, 0)


@pytest.mark.parametrize("seed", range(4))
def test_sparse_regions_are_the_runs_of_their_depth(seed):
	"""The runs of a strand's depth over a region, found without its window,
	are the window's runs of one nonzero value, and count its changes of
	value: reads that touch, overlap, start before the chromosome or end
	past it, with the start events and the coverage lost of the first and
	last regions."""

	from figwig._indexed import _changes, _dense_kernel, _sparse_kernel

	rng = numpy.random.default_rng(seed)
	size = 1000
	starts = rng.integers(-50, size + 20, 60)
	ends = starts + rng.integers(1, 80, 60)
	starts[:10] = ends[10:20]
	bias = 1 << 33
	keys = numpy.sort(numpy.concatenate([bias + 2 * starts, bias + 2 * ends +
		1])).astype(numpy.int64)
	for beg, end in [(0, 300), (250, 700), (650, size), (0, size), (400, 400),
			(990, size)]:
		for first in (beg == 0, False):
			for last in (end == size, False):
				count_lo, count_hi = 0 if first else beg, size if last else end
				out = numpy.empty(end - beg, dtype=numpy.int32)
				counted, lost = _dense_kernel()(keys, 0, len(keys), beg, size, first,
					last, count_lo, count_hi, out)
				runs = [numpy.empty(len(keys), dtype=dtype) for dtype in ("int64",
					"int64", "int32")]
				n, c, gone = _sparse_kernel()(keys, 0, len(keys), beg, end, size,
					first, last, count_lo, count_hi, *runs)
				assert (c, gone) == (counted, lost)
				if end == beg:
					assert n == 0
					continue

				edges = numpy.flatnonzero(out[1:] != out[:-1]) + 1
				lefts = numpy.concatenate([[0], edges])
				rights = numpy.concatenate([edges, [len(out)]])
				kept = out[lefts] != 0
				assert runs[0][:n].tolist() == (beg + lefts[kept]).tolist()
				assert runs[1][:n].tolist() == (beg + rights[kept]).tolist()
				assert runs[2][:n].tolist() == out[lefts[kept]].tolist()
				assert _changes(runs[0][:n], runs[1][:n], beg, end) == len(edges) + 1
				assert n > 0


def test_mapped_reads_from_the_pseudo_bins(tmp_path, indexed_bams):
	"""The mapped reads read from the pseudo-bins alone are pysam's; a BAM
	without a .bai, with one older than it, or with one cut short inside
	its references has 0 (the full parse then refuses a .bai cut later)."""

	import shutil
	import pysam
	from figwig._indexed import _mapped_reads

	unsorted, indexed = indexed_bams
	small = sorted_indexed(tmp_path / "small.bam", BAM_CHROM_SIZES, [("chr1", 10 *
		i, 30, bool(i % 2)) for i in range(40)])
	for path in (indexed, small):
		assert _mapped_reads(path) == sum(s.mapped for s in pysam.AlignmentFile(
			path).get_index_statistics()) > 0
	assert _mapped_reads(unsorted) == 0

	copy = str(tmp_path / "copy.bam")
	shutil.copy(indexed, copy)
	index = Path(indexed + ".bai").read_bytes()
	for cut in (len(index) - 9, len(index) // 2, 12):
		with open(copy + ".bai", "wb") as handle:
			handle.write(index[:cut])
		assert _mapped_reads(copy) == 0
	with open(copy + ".bai", "wb") as handle:
		handle.write(index)
	assert _mapped_reads(copy) == _mapped_reads(indexed)
	os.utime(copy + ".bai", (1, 1))
	assert _mapped_reads(copy) == 0


def test_bam2bw_reads_on_every_core(tmp_path, indexed_bams, sizes, monkeypatch):
	"""bam2bw -c reads regions on all its cores, its bigWig writer taking
	little of the time; bam2fw -c leaves the figwig writer its share."""

	import figwig._indexed

	_, indexed = indexed_bams
	threads = []
	original = figwig._indexed._indexed_coverage

	def recorded(files, chrom_sizes, args, n, *rest, **kwargs):
		threads.append(n)
		return original(files, chrom_sizes, args, n, *rest, **kwargs)

	monkeypatch.setattr(figwig._indexed, "_indexed_coverage", recorded)
	bam2bw(tmp_path / "bw", indexed, sizes, "-c", "-p", "8")
	bam2fw(tmp_path / "fw", [indexed], sizes, "-c", "-p", "8")
	assert threads == [8, figwig._indexed._shares(8)[0]] == [8, 6]


def test_reading_and_writing_share_the_cores():
	"""While regions are read the writer takes a quarter of the cores, at
	least one, and the regions the rest; one core does both in turn on the
	main thread."""

	from figwig._indexed import _shares

	assert _shares(1) == (0, 1)
	for cores in range(2, 70):
		readers, writer = _shares(cores)
		assert readers + writer == cores and readers >= 1
		assert writer == max(1, cores // 4)


def test_regions_follow_the_bam_bytes(tmp_path, monkeypatch):
	"""Regions cover each chromosome from 0 to its end on 16 kb windows,
	at most _REGION long, and are cut finer where the reads are dense; a
	chromosome without reads, or not in the BAM, is one region."""

	import pysam
	import figwig._indexed

	monkeypatch.setattr(figwig._indexed, "_REGION", 1 << 16)
	monkeypatch.setattr(figwig._indexed, "_REGION_BYTES", 1 << 13)
	chrom_sizes = [("dense", 300000), ("empty", 200000)]
	reads = [("dense", int(p), 50, False) for p in numpy.linspace(0, 299900,
		600).astype(int)]
	reads += [("dense", int(p), 50, False) for p in numpy.linspace(20000, 40000,
		6000).astype(int)]
	written = write_bam(tmp_path / "r.bam", chrom_sizes, reads)
	indexed = str(tmp_path / "r_sorted.bam")
	pysam.sort("-o", indexed, str(written))
	pysam.index(indexed)
	sizes_path = str(write_chrom_sizes(tmp_path / "r.sizes", chrom_sizes))
	args = figwig.bam2bw._parser("t").parse_args([indexed, "-s", sizes_path, "-n",
		"x", "-c"])
	files = figwig._indexed._open_files([indexed], chrom_sizes, args)
	dense = figwig._indexed._regions(files, "dense", 300000)
	assert dense[0][0] == 0 and dense[-1][1] == 300000
	assert all(b == c for (_, b), (c, _) in zip(dense, dense[1:]))
	assert all(a % (1 << 14) == 0 and b - a <= 1 << 16 for a, b in dense)
	within = [b - a for a, b in dense if 16384 <= a < 40000]
	assert within and max(within) < 1 << 16
	assert figwig._indexed._regions(files, "empty", 200000) == [(0, 200000)]
	assert figwig._indexed._regions(files, "absent", 200000) == [(0, 200000)]


def sorted_indexed(path, chrom_sizes, reads, **kwargs):
	"""`write_bam`'s file, coordinate-sorted and indexed beside it; the file
	as written is unsorted/<its name>, under the same name so that bam2fw
	names its tracks alike."""

	import pysam

	os.makedirs(path.parent / "unsorted", exist_ok=True)
	written = write_bam(path.parent / "unsorted" / path.name, chrom_sizes, reads,
		**kwargs)
	pysam.sort("-o", str(path), str(written))
	pysam.index(str(path))
	return str(path)


def test_indexed_coverage_keeps_earlier_output_when_it_fails(tmp_path, sizes):
	"""bam2bw -c writes its bigWigs beside their paths and moves them there
	only once done, so a read the indexed path hands back that the
	whole-file reader then refuses leaves an earlier output as it was."""

	reads = [("chr1", 10 * i, 30, bool(i % 2)) for i in range(40)]
	good = sorted_indexed(tmp_path / "good.bam", BAM_CHROM_SIZES, reads)
	bad = sorted_indexed(tmp_path / "bad.bam", BAM_CHROM_SIZES, reads + [("chr1",
		500, 30, False, [])])
	name = str(tmp_path / "out")
	figwig.bam2bw.main([good, "-s", sizes, "-n", name, "-c"])
	before = [Path(name + suffix).read_bytes() for suffix in (".+.bw", ".-.bw")]
	with pytest.raises(Exception):
		figwig.bam2bw.main([bad, "-s", sizes, "-n", name, "-c"])
	assert [Path(name + suffix).read_bytes() for suffix in (".+.bw", ".-.bw")] == \
		before
	assert sorted(os.listdir(tmp_path)) == sorted(["good.bam", "good.bam.bai",
		"bad.bam", "bad.bam.bai", "unsorted", "out.+.bw", "out.-.bw"])


@pytest.mark.parametrize("command", ["bam2bw", "bam2fw"])
@pytest.mark.parametrize("flags", [(), ("-r",)])
def test_indexed_coverage_reports_once_when_it_falls_back(tmp_path, sizes,
		command, flags, capsys):
	"""With -v, a read the record kernel hands to pysam sends the whole file
	back to the old reader, which prints the chromosomes missing from the
	sizes; the indexed path has printed nothing before then."""

	reads = [("chr1", 10 * i, 30, False) for i in range(40)]
	reads += [("chrUN", 5, 30, False), ("chr2", 50, 30, False, [(4, 30), (3,
		10)])]
	path = sorted_indexed(tmp_path / "r.bam", BAM_CHROM_SIZES, reads)
	unsorted = str(tmp_path / "unsorted" / "r.bam")
	capsys.readouterr()
	run = bam2bw if command == "bam2bw" else (lambda t, f, s, *a: bam2fw(t, [f],
		s, *a))
	run(tmp_path / "whole", unsorted, sizes, "-c", "-v", *flags)
	whole = capsys.readouterr().out
	run(tmp_path / "indexed", path, sizes, "-c", "-v", *flags)
	indexed = capsys.readouterr().out
	assert whole.count("chrUN encountered") == 1
	assert indexed == whole


def test_indexed_coverage_takes_indexes_of_empty_references(tmp_path, sizes):
	"""A reference without reads has no pseudo-bin in a .bai, and counts as
	none mapped, so -v keeps the indexed path."""

	import figwig._indexed
	reads = [("chr1", 10 * i, 30, False) for i in range(40)]
	path = sorted_indexed(tmp_path / "r.bam", BAM_CHROM_SIZES + [("chrEmpty",
		1000)], reads)
	args = figwig.bam2bw._parser("t").parse_args([path, "-s", sizes, "-n", "x",
		"-c", "-v"])
	files = figwig._indexed._open_files([path], CHROM_SIZES, args)
	assert files is not None
	assert [mapped for _, _, mapped, _, _ in files[0].index][-1] == 0


def test_indexed_coverage_refuses_another_bams_index(tmp_path, sizes):
	"""An index made for another BAM of the same header, newer than this
	one, points past its end, and the coverage is the whole file's."""

	import figwig._indexed
	import shutil
	import time

	rng = numpy.random.default_rng(5)
	big = [("chr1", int(p), 30, bool(p % 2)) for p in rng.integers(0, 960,
		20000)]
	small = [("chr2", int(p), 30, bool(p % 2)) for p in rng.integers(0, 460,
		300)]
	other = sorted_indexed(tmp_path / "big.bam", BAM_CHROM_SIZES, big, seed=3)
	path = sorted_indexed(tmp_path / "small.bam", BAM_CHROM_SIZES, small)
	time.sleep(0.01)
	shutil.copy(other + ".bai", path + ".bai")
	os.utime(path + ".bai", (os.path.getmtime(path) + 1,) * 2)
	args = figwig.bam2bw._parser("t").parse_args([path, "-s", sizes, "-n", "x",
		"-c"])
	assert figwig._indexed._open_files([path], CHROM_SIZES, args) is None
	whole = bam2bw(tmp_path / "whole", str(tmp_path / "unsorted" / "small.bam"),
		sizes, "-c", "-u")
	got = bam2bw(tmp_path / "got", path, sizes, "-c", "-u")
	assert Path(whole[0]).read_bytes() == Path(got[0]).read_bytes()


def test_indexed_coverage_scales_only_covered_bases(tmp_path, indexed_bams,
		sizes):
	"""bam2fw -c on threads scales each region's depth where it is nonzero,
	as the runs are scaled, so -sf nan leaves the uncovered bases 0."""

	unsorted, indexed = indexed_bams
	whole = bam2fw(tmp_path / "whole", [unsorted], sizes, "-c", "-u", "-sf",
		"nan")
	got = bam2fw(tmp_path / "got", [indexed], sizes, "-c", "-u", "-sf", "nan",
		"-p", "8")
	assert_same_figwig_values(whole, got)


def test_indexed_coverage_falls_back(tmp_path, indexed_bams, sizes, monkeypatch):
	"""A BAM whose index is older than it, and one the record kernel hands
	to pysam part of the way through, are read as before, with the same
	result; an index cut short is refused."""

	import shutil
	import figwig._indexed

	unsorted, indexed = indexed_bams
	monkeypatch.setattr(figwig._indexed, "_regions", fixed_regions(64))
	whole = bam2bw(tmp_path / "whole", unsorted, sizes, "-c")

	stale = str(tmp_path / "stale.bam")
	shutil.copy(indexed + ".bai", stale + ".bai")
	shutil.copy(indexed, stale)
	os.utime(stale + ".bai", (1, 1))
	assert figwig._indexed._index_path(stale) is None

	# An index cut short is not taken; pysam crashes opening some of them,
	# so the parser is called alone. The last 8 bytes are optional.
	with open(indexed + ".bai", "rb") as handle:
		index = handle.read()
	cut = str(tmp_path / "cut.bai")
	for n in range(0, len(index) - 8):
		with open(cut, "wb") as handle:
			handle.write(index[:n])
		assert figwig._indexed._read_index(cut, len(BAM_CHROM_SIZES)) is None
	assert figwig._indexed._read_index(indexed + ".bai", len(BAM_CHROM_SIZES)
		) is not None
	# A negative count of chunks in the first reference's first bin.
	with open(cut, "wb") as handle:
		handle.write(index[:16] + (-1).to_bytes(4, "little", signed=True) +
			index[20:])
	assert figwig._indexed._read_index(cut, len(BAM_CHROM_SIZES)) is None
	# A chunk that begins past 2**63, which int64 offsets cannot hold.
	with open(cut, "wb") as handle:
		handle.write(index[:20] + (1 << 63).to_bytes(8, "little") + index[28:])
	assert figwig._indexed._read_index(cut, len(BAM_CHROM_SIZES)) is None

	original = figwig._indexed._region_keys
	seen = []

	def fails_later(*args):
		seen.append(1)
		if len(seen) == 5:
			raise figwig._indexed._Fallback()
		return original(*args)

	monkeypatch.setattr(figwig._indexed, "_region_keys", fails_later)
	for path in (stale, indexed):
		got = bam2bw(tmp_path / ("out_" + os.path.basename(path)), path, sizes,
			"-c")
		for a, b in zip(whole, got):
			for chrom, length in CHROM_SIZES:
				numpy.testing.assert_array_equal(bigwig_values(a, chrom, length),
					bigwig_values(b, chrom, length))
	assert len(seen) >= 5
	fw_whole = bam2fw(tmp_path / "fw_whole", [unsorted], sizes, "-c")
	seen.clear()
	assert_same_figwig_values(fw_whole, bam2fw(tmp_path / "fw_fallback",
		[indexed], sizes, "-c"))
	assert len(seen) >= 5


def test_coverage_of_bgzf_and_gzip_beds_and_of_several_files(tmp_path,
		coverage_bed, coverage_bam, sizes):
	import gzip
	import pysam

	expected = pileup(stranded=False)
	bgzf = str(tmp_path / "coverage.bed.gz")
	pysam.tabix_compress(coverage_bed, bgzf)
	plain = str(tmp_path / "plain.bed.gz")
	with open(coverage_bed, "rb") as src, gzip.open(plain, "wb") as dst:
		dst.write(src.read())
	for name, path, flags in [("bgzf", bgzf, ("-p", "3")), ("plain", plain, ("-p",
			"3")), ("bgzf1", bgzf, ())]:
		assert_coverage(bam2bw(tmp_path / name, path, sizes, "-c", "-u", *flags),
			expected, ["+"])

	# Several inputs are pooled by bam2bw and kept apart by bam2fw.
	bigwigs = bam2bw(tmp_path / "pooled", coverage_bam, sizes, "-c")
	two = bam2bw(tmp_path / "two", coverage_bam, sizes, "-c")
	doubled = {chrom: {s: 2 * v for s, v in strands.items()} for chrom, strands
		in pileup().items()}
	main(["bam2bw", coverage_bam, coverage_bam, "-s", sizes, "-n", str(tmp_path /
		"two" / "both"), "-c"])
	assert_coverage([str(tmp_path / "two" / "both.+.bw"), str(tmp_path / "two" /
		"both.-.bw")], doubled, ["+", "-"])
	table = tmp_path / "files.tsv"
	table.write_text("file\tname\tassay\ncoverage.bam\tA\tChIP-seq\ncoverage.bed\t\t"
		"ATAC-seq\n")
	path = bam2fw(tmp_path / "fw", [coverage_bam, coverage_bed], sizes, "-c",
		"-m", str(table), "--dtype", "uint16", "-z", "1", "-t", "256")
	reader = FigWigReader(path)
	assert [track["name"] for track in reader.tracks] == ["A.+", "A.-",
		"coverage.+", "coverage.-"]
	assert [track["group"] for track in reader.tracks] == [0, 1, 2, 3]
	assert reader.tracks[0]["assay"] == "ChIP-seq"
	assert [track["dtype"] for track in reader.tracks] == ["uint16"] * 4
	assert reader.zooms == (4,)
	assert_tracks_match_bigwigs(path, ["A.+", "A.-"], bigwigs)
	assert_tracks_match_bigwigs(path, ["coverage.+"], bam2bw(tmp_path / "bed",
		coverage_bed, sizes, "-c", "-u"))
	del two


def test_coverage_shifts_clip_and_skip(tmp_path, coverage_bam, sizes, capsys):
	"""A shift that pushes a read before 0 is clipped there and reported; a
	read it empties is skipped; and the depth counts the reads whose first
	base lies inside the chromosome, from the kernel and the pysam loop."""

	import pysam

	sam = str(tmp_path / "coverage.sam")
	pysam.view("-h", "-o", sam, coverage_bam, catch_stdout=False)
	expected = pileup(-150, 0)
	for name, source in (("bam", coverage_bam), ("sam", sam)):
		bigwigs = bam2bw(tmp_path / name, source, sizes, "-c", "-ps", "-150")
		assert_coverage(bigwigs, expected, ["+", "-"])
		out = capsys.readouterr().out
		# chr1's reads at 100 (two) and 120 start 50 and 30 before 0, chr2's
		# at 0 starts 150 before it, chr1:980 runs 10 past 1000 and chr2:490
		# 10 past 500.
		assert "entries fell outside" in out
	# With -r the depth is the reads that start inside: chr1 at 980
	# (shifted to 830) and chr2 at 490 (to 340); the others are shifted
	# before 0, so 2 reads.
	bigwigs = bam2bw(tmp_path / "r", coverage_bam, sizes, "-c", "-ps", "-150",
		"-r")
	assert_coverage(bigwigs, expected, ["+", "-"], scale=1 / 2)
	# A shift past every read's end empties them all: no data at all.
	bigwigs = bam2bw(tmp_path / "none", coverage_bam, sizes, "-c", "-ps", "2000")
	for chrom, length in CHROM_SIZES:
		for bigwig in bigwigs:
			assert bigwig_values(bigwig, chrom, length).sum() == 0


def test_coverage_through_small_write_batches(tmp_path, coverage_bam, sizes,
		monkeypatch):
	"""The runs are written in batches of _WRITE_ITEMS items that may end
	inside a chromosome; batches of three items give the same file as one."""

	import figwig.bam2bw

	whole = bam2bw(tmp_path / "whole", coverage_bam, sizes, "-c")
	monkeypatch.setattr(figwig.bam2bw, "_WRITE_ITEMS", 3)
	small = bam2bw(tmp_path / "small", coverage_bam, sizes, "-c")
	for a, b in zip(whole, small):
		with open(a, "rb") as x, open(b, "rb") as y:
			assert x.read() == y.read()


def test_coverage_of_an_empty_input_and_a_zero_scale(tmp_path, sizes):
	bam = str(write_bam(tmp_path / "empty.bam", BAM_CHROM_SIZES, [], unmapped=[(
		"chr1", 5, 20)]))
	bigwigs = bam2bw(tmp_path / "bw", bam, sizes, "-c")
	for bigwig in bigwigs:
		assert BigWigReader(bigwig)._data_index == 0
	path = bam2fw(tmp_path / "fw", [bam], sizes, "-c")
	reader = FigWigReader(path)
	assert reader.stats["sum"].tolist() == [0, 0]
	bed = str(write_intervals(tmp_path / "zero.bed", [("chr1", 10, 10), ("chr1",
		20, 15), ("chr1", 30, 40)]))
	bigwig, = bam2bw(tmp_path / "zero", bed, sizes, "-c", "-u")
	values = bigwig_values(bigwig, "chr1", 1000)
	assert values.sum() == 10 and values[30:40].min() == 1


def test_coverage_command_line(tmp_path, coverage_bam, sizes):
	name = str(tmp_path / "cli")
	result = subprocess.run([sys.executable, "-m", "figwig", "bam2bw", coverage_bam,
		"-s", sizes, "-n", name, "-c", "-u", "-v", "-z", "2", "-p", "2"], env=ENV,
		capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert "20 entries fell outside" in result.stdout
	assert_coverage([name + ".bw"], pileup(stranded=False), ["+"])
	result = subprocess.run([sys.executable, "-m", "figwig", "bam2fw", coverage_bam,
		"-s", sizes, "-n", name, "-c", "-v", "-p", "2"], env=ENV,
		capture_output=True, text=True, timeout=600)
	assert result.returncode == 0, result.stderr
	assert len(FigWigReader(name + ".fw").tracks) == 2


def test_summary_index_is_written_when_asked(tmp_path, bam, sizes, small_windows,
	watched):
	"""With -t 16 the 1,900 bases are 115 bins, 15 windows of 8, and a read of
	chr2 alone is 5 of them, through the index."""

	indexed = bam2fw(tmp_path / "indexed", [bam], sizes, "-u", "-t", "16",
		"--summary-index")
	plain = bam2fw(tmp_path / "plain", [bam], sizes, "-u", "-t", "16")
	assert any(level.window_bins for group in read_index(indexed).groups for level in
		group.levels)
	assert read_index(plain) is None
	assert FigWigReader(indexed).tile_sums().tobytes() == \
		FigWigReader(plain).tile_sums().tobytes()
	regions, used = read_regions(indexed, watched)
	assert used and regions == read_regions(plain, watched)[0]


def test_summary_index_is_written_for_cells_too(tmp_path, fragments, sizes,
	small_windows, watched):
	"""The cells' writer is a second call of FigWigWriter in bam2fw."""

	path = bam2fw(tmp_path / "indexed", [fragments], sizes, "--barcode-column", "4",
		"-f", "-t", "16", "--summary-index")
	plain = bam2fw(tmp_path / "plain", [fragments], sizes, "--barcode-column", "4",
		"-f", "-t", "16")
	assert any(level.window_bins for group in read_index(path).groups for level in
		group.levels)
	regions, used = read_regions(path, watched)
	assert used and regions == read_regions(plain, watched)[0]
