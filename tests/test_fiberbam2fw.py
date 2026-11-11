# test_fiberbam2fw.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig fiberbam2fw`, on Fiber-seq reads built from an explicit
layout (`tests/fiber_reads.py`): every call and footprint of a read is
placed by walking the molecule base by base, and the file must hold exactly
those values, on both strands and across soft clips, insertions and
deletions. htslib's reading of MM and ML (pysam's `modified_bases`) is the
second oracle for the m6A and 5mC channels.
"""

import array
import subprocess
import sys

import numpy
import pytest

if sys.platform == "win32":
	pytest.skip("pysam does not build on Windows", allow_module_level=True)

import pysam  # noqa: E402

import figwig.fiberbam2fw  # noqa: E402

from figwig import FigWigReader  # noqa: E402

from .fiber_reads import CHROM_SIZES  # noqa: E402
from .fiber_reads import Read  # noqa: E402
from .fiber_reads import header  # noqa: E402
from .fiber_reads import standard_reads  # noqa: E402
from .fiber_reads import write_reads  # noqa: E402

CHANNELS = ["m6a", "cpg", "nuc", "msp", "fire"]


def convert(tmp_path, bams, *flags, name="out"):
	"""Run the command's main on BAMs and return the path of the file."""

	bams = [bams] if isinstance(bams, (str, bytes)) or hasattr(bams, "joinpath") else bams
	path = tmp_path / name
	figwig.fiberbam2fw.main(["-n", str(path), "--force"] + list(flags) +
		[str(b) for b in bams])
	return str(path) + ".fw"


def track_values(reader, read, channel):
	"""One alignment's channel over its reference span, from its track."""

	index = reader.select(read=read, channel=channel)
	assert len(index) == 1
	record = reader.tracks[int(index[0])]
	values = reader.read(record["chrom"], [record["start"]], record["end"] -
		record["start"], tracks=int(index[0]))[0]
	return {record["start"] + int(i): int(values[i]) for i in numpy.flatnonzero(values)}


@pytest.fixture(scope="module")
def reads():
	return standard_reads()


@pytest.fixture(scope="module")
def bam(tmp_path_factory, reads):
	path = tmp_path_factory.mktemp("fiber") / "reads.bam"
	write_reads(path, reads)
	return path


@pytest.fixture(scope="module")
def fw(tmp_path_factory, bam):
	return convert(tmp_path_factory.mktemp("fiber_fw"), bam)


def test_every_channel_holds_what_the_molecule_gives(fw, reads):
	reader = FigWigReader(fw)
	for read in reads:
		expected = read.expected()
		for channel in CHANNELS:
			assert track_values(reader, read.name, channel) == expected[channel], (
				read.name, channel)


def test_the_calls_have_values_to_check():
	# The oracle is only a check when the reads put calls and footprints on
	# the reference, on both strands.
	reads = standard_reads()
	assert sum(len(r.expected()["m6a"]) for r in reads) > 500
	assert sum(len(r.expected()["cpg"]) for r in reads) > 50
	assert sum(len(r.expected()["nuc"]) for r in reads) > 500
	assert sum(len(r.expected()["fire"]) for r in reads) > 50
	assert {r.reverse for r in reads} == {True, False}


def test_m6a_and_5mc_agree_with_htslib(bam, fw):
	reader = FigWigReader(fw)
	with pysam.AlignmentFile(str(bam)) as handle:
		for record in handle:
			refs = record.get_reference_positions(full_length=True)
			for channel, code in (("m6a", "a"), ("cpg", "m")):
				want = {}
				for (_, _, c), items in record.modified_bases.items():
					if c != code:
						continue
					for q, p in items:
						if refs[q] is not None and p > 0:
							want[refs[q]] = p
				assert track_values(reader, record.query_name, channel) == want


def test_probabilities_are_kept_and_a_call_of_zero_is_not_written(fw, reads):
	reader = FigWigReader(fw)
	values = track_values(reader, reads[0].name, "m6a")
	assert len(set(values.values())) > 50
	assert 0 not in values.values()
	assert 255 in values.values() and 125 in values.values()


def test_threshold_writes_ones(tmp_path, bam, reads):
	path = convert(tmp_path, bam, "--threshold", "125")
	reader = FigWigReader(path)
	for read in reads:
		expected = read.expected()
		for channel in ("m6a", "cpg"):
			assert track_values(reader, read.name, channel) == {
				r: 1 for r, p in expected[channel].items() if p >= 125}


def test_channels_option_writes_the_ones_given(tmp_path, bam, reads):
	reader = FigWigReader(convert(tmp_path, bam, "-c", "nuc,m6a"))
	assert sorted({t["channel"] for t in reader.tracks}) == ["m6a", "nuc"]
	assert len(reader.tracks) == 2 * len(reads)
	assert track_values(reader, reads[1].name, "nuc") == reads[1].expected()["nuc"]


def test_tracks_and_their_attributes(fw, reads):
	reader = FigWigReader(fw)
	assert len(reader.tracks) == 5 * len(reads)
	assert all(t["dtype"] == "uint8" for t in reader.tracks)
	for read in reads:
		for channel in CHANNELS:
			index = reader.select(read=read.name, channel=channel)
			record = reader.tracks[int(index[0])]
			assert record["name"] == "{}.{}".format(read.name, channel)
			assert record["read_name"] == read.name
			assert record["chrom"] == CHROM_SIZES[read.tid][0]
			assert record["start"] == read.start
			assert record["strand"] == ("-" if read.reverse else "+")
			assert record["mapq"] == 60 and record["flag"] == read.flag
			assert record["length"] == len(read.sequence)
			assert record["HP"] == read.tags["HP"]
			assert record["rq"] == pytest.approx(read.tags["rq"])
			assert record["RG"] == "rg1" and record["sample"] == "K562"
		counts = read.counts()
		m6a = reader.tracks[int(reader.select(read=read.name, channel="m6a")[0])]
		assert (m6a["m6a_calls"], m6a["m6a_unplaced"]) == tuple(counts["m6a"])
		assert m6a["n_values"] == len(read.expected()["m6a"])
		assert m6a["at_bases"] == sum(1 for b in read.molecule if b in "AT")
		cpg = reader.tracks[int(reader.select(read=read.name, channel="cpg")[0])]
		assert (cpg["cpg_calls"], cpg["cpg_unplaced"]) == tuple(counts["cpg"])
		nuc = reader.tracks[int(reader.select(read=read.name, channel="nuc")[0])]
		assert nuc["nuc_calls"] == len(read.features["nuc"][0])
		fire = reader.tracks[int(reader.select(read=read.name, channel="fire")[0])]
		assert fire["fire_calls"] == sum(1 for q in read.features["aq"][0] if q > 0)
	attributes = reader.attributes["fiberbam2fw"]
	assert attributes["channels"] == CHANNELS and attributes["reads"] == [len(reads)]


def test_reads_arrange_as_windows_reads_channels_width(fw, reads):
	reader = FigWigReader(fw)
	read = reads[0]
	index = reader.select(read=read.name)
	out = reader.read("chr1", [read.start, read.start + 50], 200, tracks=index,
		channels="channel", items="read")
	assert out.shape == (2, 1, 5, 200)
	assert [reader.tracks[int(i)]["channel"] for i in index] == CHANNELS
	expected = read.expected()["m6a"]
	for offset in range(200):
		assert out[0, 0, 0, offset] == expected.get(read.start + offset, 0)


def test_groups_hold_consecutive_reads_of_one_channel(tmp_path, bam, reads):
	path = convert(tmp_path, bam, "--group-size", "3")
	reader = FigWigReader(path)
	names = [t["name"] for t in reader.tracks]
	assert names[:3] == ["{}.m6a".format(r.name) for r in reads[:3]]
	assert names[3:6] == ["{}.cpg".format(r.name) for r in reads[:3]]
	groups = [t["group"] for t in reader.tracks]
	assert groups[:3] == [0] * 3 and groups[3:6] == [1] * 3
	assert len(set(groups)) == 5 * 3
	for read in reads:
		assert track_values(reader, read.name, "m6a") == read.expected()["m6a"]
	# a block that moves to the next chromosome is two calls of its group
	assert {t["chrom"] for t in reader.tracks[:3 * 5]} == {"chr1"}
	assert any(t["chrom"] != reader.tracks[0]["chrom"] for t in reader.tracks)


def test_group_size_and_threads_do_not_change_the_values(tmp_path, bam, reads):
	one = convert(tmp_path, bam, "-p", "1", name="one")
	four = convert(tmp_path, bam, "-p", "4", name="four")
	with open(one, "rb") as a, open(four, "rb") as b:
		assert a.read() == b.read()
	small = FigWigReader(convert(tmp_path, bam, "--group-size", "2", "-p", "4", name="s"))
	for read in reads:
		for channel in CHANNELS:
			assert track_values(small, read.name, channel) == read.expected()[channel]


def test_secondary_unmapped_and_untagged_alignments_are_left_out(tmp_path, reads):
	extra = [
		Read("secondary", 0, 120, [(100, "M")], flag=256, seed=11),
		Read("untagged", 0, 130, [(100, "M")], seed=12, write_mm=False),
		Read("low", 0, 140, [(100, "M")], seed=13, mapq=3),
	]
	path = tmp_path / "mixed.bam"
	write_reads(path, list(reads) + extra)
	out = convert(tmp_path, path)
	reader = FigWigReader(out)
	read_names = {t["read_name"] for t in reader.tracks}
	assert read_names == {r.name for r in reads} | {"low"}
	skipped = reader.attributes["fiberbam2fw"]["skipped"][0]
	assert skipped["secondary"] == 1 and skipped["no_tags"] == 1

	filtered = FigWigReader(convert(tmp_path, path, "--min-mapq", "10", name="q"))
	assert "low" not in {t["read_name"] for t in filtered.tracks}
	assert filtered.attributes["fiberbam2fw"]["skipped"][0]["low_mapq"] == 1


def test_supplementary_alignments_have_tracks_of_their_own(tmp_path, reads):
	supplementary = Read("read0", 0, 2000, [(60, "S"), (200, "M"), (40, "S")],
		flag=2048, seed=0, molecule=reads[0].molecule[:300])
	supplementary.calls, supplementary.mm, supplementary.ml = (reads[0].calls,
		reads[0].mm, reads[0].ml)
	path = tmp_path / "sup.bam"
	write_reads(path, list(reads) + [supplementary])
	reader = FigWigReader(convert(tmp_path, path))
	both = reader.select(read_name="read0", channel="m6a")
	assert len(both) == 2
	names = [reader.tracks[int(i)]["read"] for i in both]
	assert names[0] == "read0" and names[1] == "read0@chr1:2000"
	assert reader.tracks[int(both[1])]["flag"] == 2048
	primary = FigWigReader(convert(tmp_path, path, "--primary-only", name="p"))
	assert len(primary.select(read_name="read0", channel="m6a")) == 1
	assert primary.attributes["fiberbam2fw"]["skipped"][0]["supplementary"] == 1


def test_ma_tags_give_the_same_footprints(tmp_path, reads):
	ma_reads = []
	for read in reads:
		clone = Read(read.name, read.tid, read.start, read.cigar,
			reverse=read.reverse, seed=0, molecule=read.molecule, mm=read.mm,
			ml=read.ml, tags=read.tags)
		clone.calls = read.calls
		starts, lengths = read.features["nuc"]
		msp_starts, msp_lengths = read.features["msp"]
		quality = read.features["aq"][0]
		fire = [(s, n) for s, n, q in zip(msp_starts, msp_lengths, quality) if q > 0]
		text = "{};nuc.:{};msp.:{};fire.Q:{}".format(read.full, ",".join(
			"{}-{}".format(s + 1, n) for s, n in zip(starts, lengths)), ",".join(
			"{}-{}".format(s + 1, n) for s, n in zip(msp_starts, msp_lengths)),
			",".join("{}-{}".format(s + 1, n) for s, n in fire))
		clone.tags["MA"] = text
		clone.tags["AQ"] = array.array("B", [q for q in quality if q > 0])
		ma_reads.append(clone)
	path = tmp_path / "ma.bam"
	write_reads(path, ma_reads)
	reader = FigWigReader(convert(tmp_path, path))
	for read in reads:
		expected = read.expected()
		for channel in ("nuc", "msp", "fire"):
			assert track_values(reader, read.name, channel) == expected[channel]


def test_hard_clipped_alignments_place_footprints_and_say_the_calls_are_left_out(
		tmp_path, capsys):
	cigar = [(25, "H"), (200, "M"), (2, "D"), (100, "M"), (15, "H")]
	read = Read("clipped", 0, 500, cigar, reverse=True, seed=5, features={
		"nuc": ([30, 120, 260], [60, 70, 40]), "msp": ([95, 200], [25, 50]),
		"aq": ([0, 243], None)}, write_mm=False)
	read.mm, read.ml = "A+a.,0;", [200]
	path = tmp_path / "hard.bam"
	write_reads(path, [read])
	reader = FigWigReader(convert(tmp_path, path))
	expected = read.expected()
	for channel in ("nuc", "msp", "fire"):
		assert track_values(reader, "clipped", channel) == expected[channel]
	assert track_values(reader, "clipped", "m6a") == {}
	assert "hard-clipped" in capsys.readouterr().err


def test_a_molecule_without_calls_has_empty_tracks(tmp_path):
	read = Read("bare", 0, 300, [(100, "M")], seed=1, mods=(), features={
		"nuc": ([10], [30])})
	read.mm, read.ml = "A+a.;", []
	path = tmp_path / "bare.bam"
	write_reads(path, [read])
	reader = FigWigReader(convert(tmp_path, path))
	assert track_values(reader, "bare", "m6a") == {}
	assert track_values(reader, "bare", "nuc") == {r: 1 for r in range(310, 340)}


@pytest.mark.parametrize("mm, ml, text", [
	("A+a.,100000;", [200], "runs past the end of the read"),
	("A+a.,0,0,0;", [200], "fewer values"),
	("A+a.,x;", [200], "malformed"),
	("T+a.,0;", [200], "6mA on a base and strand other than A+ or T-"),
	("A-a.,0;", [200], "6mA on a base and strand other than A+ or T-"),
])
def test_a_tag_that_cannot_be_placed_is_reported(tmp_path, capsys, mm, ml, text):
	read = Read("odd", 0, 300, [(200, "M")], seed=2, molecule="A" * 100 + "T" * 100,
		write_mm=False)
	read.mm, read.ml = mm, ml
	path = tmp_path / "odd.bam"
	write_reads(path, [read])
	convert(tmp_path, path)
	err = capsys.readouterr().err
	assert text in err and "odd" in err


def test_a_base_named_twice_is_placed_once(tmp_path, capsys):
	read = Read("twice", 0, 300, [(100, "M")], seed=2, molecule="A" * 100,
		write_mm=False)
	read.mm, read.ml = "A+a.,0,1;A+a.,0;", [200, 100, 50]
	path = tmp_path / "twice.bam"
	write_reads(path, [read])
	reader = FigWigReader(convert(tmp_path, path))
	assert track_values(reader, "twice", "m6a") == {300: 200, 302: 100}
	assert "names a base twice" in capsys.readouterr().err


def test_several_files_and_a_metadata_table(tmp_path, reads):
	first, second = tmp_path / "a.bam", tmp_path / "b.bam"
	write_reads(first, reads[:4])
	write_reads(second, reads[4:], hdr=header(sample="HG002"))
	table = tmp_path / "files.tsv"
	table.write_text("file\tname\tcondition\n{}\tsampleA\ttreated\n{}\tsampleB\tcontrol\n".format(
		first, second))
	reader = FigWigReader(convert(tmp_path, [first, second], "-m", str(table), "-g", "hg38"))
	assert len(reader.tracks) == 5 * len(reads)
	assert reader.tracks[0]["name"] == "sampleA.read0.m6a"
	assert reader.tracks[-1]["name"] == "sampleB.read7.fire"
	assert reader.tracks[0]["condition"] == "treated"
	assert reader.tracks[-1]["condition"] == "control" and reader.tracks[-1]["sample"] == "HG002"
	assert reader.genome == "hg38"
	assert reader.attributes["fiberbam2fw"]["reads"] == [4, 4]
	assert track_values(reader, "sampleB.read5", "nuc") == reads[5].expected()["nuc"]

	unnamed = FigWigReader(convert(tmp_path, [first, second], name="u"))
	assert unnamed.tracks[0]["name"] == "a.read0.m6a"


def test_tags_option(tmp_path, bam):
	reader = FigWigReader(convert(tmp_path, bam, "--tags", "HP"))
	assert "HP" in reader.tracks[0] and "rq" not in reader.tracks[0]
	assert "RG" not in reader.tracks[0] and "sample" not in reader.tracks[0]
	reader = FigWigReader(convert(tmp_path, bam, "--tags", "none", name="n"))
	assert "HP" not in reader.tracks[0]


def test_non_finite_tags_are_left_out(tmp_path):
	read = Read("nan", 0, 300, [(100, "M")], seed=2, tags={"rq": float("nan"),
		"ec": float("inf"), "HP": 2})
	path = tmp_path / "nan.bam"
	write_reads(path, [read])
	record = FigWigReader(convert(tmp_path, path)).tracks[0]
	assert "rq" not in record and "ec" not in record and record["HP"] == 2


def test_errors(tmp_path, bam, reads, capsys):
	def refused(*args, match):
		with pytest.raises(SystemExit) as info:
			figwig.fiberbam2fw.main(["-n", str(tmp_path / "e")] + [str(a) for a in args])
		assert match in str(info.value)

	refused(bam, "-c", "m6a,foo", match="--channels takes")
	refused(bam, "-c", "m6a,m6a", match="each channel once")
	refused(bam, "--threshold", "300", match="--threshold")
	refused(bam, "--group-size", "0", match="--group-size")
	refused(bam, "--group-size", str(2 ** 20), "-t", "65536", match="--group-size")
	refused(tmp_path / "missing.bam", match="is not a file")

	unsorted = tmp_path / "unsorted.bam"
	write_reads(unsorted, list(reversed(reads)), sort=False)
	refused(unsorted, match="not coordinate sorted")

	past = tmp_path / "past.bam"
	write_reads(past, [Read("far", 1, 3950, [(100, "M")], seed=1)])
	refused(past, match="past the end of chr2")

	untagged = tmp_path / "untagged.bam"
	write_reads(untagged, [Read("a", 0, 10, [(50, "M")], write_mm=False)])
	refused(untagged, match="has no alignment with a")

	other = tmp_path / "other.bam"
	write_reads(other, [Read("a", 0, 10, [(50, "M")], seed=1)], hdr=pysam.AlignmentHeader.from_dict({
		"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 6000}]}))
	refused(bam, other, match="other chromosomes")

	figwig.fiberbam2fw.main(["-n", str(tmp_path / "e"), str(bam)])
	refused(bam, match="--force")
	capsys.readouterr()


def test_command_line(tmp_path, bam, reads):
	result = subprocess.run([sys.executable, "-m", "figwig", "fiberbam2fw", "-n",
		str(tmp_path / "cli"), "-p", "2", "-v", str(bam)], capture_output=True,
		text=True)
	assert result.returncode == 0, result.stderr
	assert "{} alignments written".format(len(reads)) in result.stderr
	reader = FigWigReader(str(tmp_path / "cli.fw"))
	assert track_values(reader, reads[0].name, "m6a") == reads[0].expected()["m6a"]

	result = subprocess.run([sys.executable, "-m", "figwig", "fiberbam2fw", "-h"],
		capture_output=True, text=True)
	assert result.returncode == 0 and "--threshold" in result.stdout
