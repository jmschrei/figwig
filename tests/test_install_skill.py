# test_install_skill.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Tests for `figwig install-skill` and the skill it installs.

The command copies the bundled skill, `figwig/skills/figwig`, into a skills
directory as `<directory>/figwig`. An existing destination is an error unless
--force is given, which replaces it; --symlink links to the bundled copy
instead. The skill's own files are checked too: the frontmatter Claude Code
reads before loading it, and the reference files the router names.
"""

import os
import re

import pytest

import figwig.install_skill

from figwig.__main__ import main


SKILL_SOURCE = os.path.join(os.path.dirname(os.path.abspath(
	figwig.install_skill.__file__)), "skills", "figwig")


def install(directory, *flags):
	main(["install-skill", "-d", str(directory), *flags])
	return directory / "figwig"


def test_copy_install_writes_skill_and_references(tmp_path):
	dest = install(tmp_path)
	assert not os.path.islink(dest)
	assert (dest / "SKILL.md").is_file()
	assert sorted(os.listdir(dest / "references")) == sorted(os.listdir(
		os.path.join(SKILL_SOURCE, "references")))


def test_copy_install_leaves_out_checkpoints_and_caches(tmp_path, monkeypatch):
	source = tmp_path / "pkg" / "skills" / "figwig"
	(source / ".ipynb_checkpoints").mkdir(parents=True)
	(source / "__pycache__").mkdir()
	(source / "SKILL.md").write_text("skill")
	monkeypatch.setattr(figwig.install_skill, "__file__", str(tmp_path / "pkg" /
		"install_skill.py"))

	dest = install(tmp_path / "skills")
	assert sorted(os.listdir(dest)) == ["SKILL.md"]


def test_installing_twice_without_force_raises(tmp_path):
	install(tmp_path)
	with pytest.raises(FileExistsError, match="Re-run with --force"):
		install(tmp_path)


def test_force_replaces_an_existing_skill(tmp_path):
	dest = install(tmp_path)
	(dest / "stale.md").write_text("from an older figwig")

	install(tmp_path, "--force")
	assert not (dest / "stale.md").exists()
	assert (dest / "SKILL.md").is_file()


def test_force_replaces_a_dangling_symlink(tmp_path):
	(tmp_path / "figwig").symlink_to(tmp_path / "moved")
	with pytest.raises(FileExistsError):
		install(tmp_path)

	dest = install(tmp_path, "-f")
	assert not os.path.islink(dest)
	assert (dest / "SKILL.md").is_file()


def test_symlink_install_points_at_the_bundled_skill(tmp_path):
	dest = install(tmp_path, "--symlink")
	assert os.path.islink(dest)
	assert os.path.realpath(dest) == os.path.realpath(SKILL_SOURCE)


def test_default_directory_is_claude_skills_in_home(tmp_path, monkeypatch):
	# expanduser reads HOME, and on Windows USERPROFILE.
	monkeypatch.setenv("HOME", str(tmp_path))
	monkeypatch.setenv("USERPROFILE", str(tmp_path))
	main(["install-skill"])
	assert (tmp_path / ".claude" / "skills" / "figwig" / "SKILL.md").is_file()


def test_force_into_the_bundled_location_leaves_the_skill_in_place(tmp_path,
	monkeypatch):
	"""--directory naming the package's own skills folder, with --force, would
	delete the bundled skill before copying it. A stand-in package keeps the
	real one out of reach."""

	source = tmp_path / "pkg" / "skills" / "figwig"
	source.mkdir(parents=True)
	(source / "SKILL.md").write_text("skill")
	monkeypatch.setattr(figwig.install_skill, "__file__", str(tmp_path / "pkg" /
		"install_skill.py"))

	with pytest.raises(ValueError, match="is the bundled skill itself"):
		install(tmp_path / "pkg" / "skills", "--force")
	assert (source / "SKILL.md").read_text() == "skill"


def skill_documents():
	"""Every Markdown file of the bundled skill, as (name, text) pairs."""

	names = ["SKILL.md"] + sorted("references/" + name for name in os.listdir(
		os.path.join(SKILL_SOURCE, "references")))
	documents = []
	for name in names:
		with open(os.path.join(SKILL_SOURCE, name)) as handle:
			documents.append((name, handle.read()))
	return documents


def frontmatter(text):
	"""The SKILL.md frontmatter as {key: value}, reading `key: value` lines and
	folded `key: >-` blocks, the two shapes it uses, since PyYAML is not a
	dependency."""

	lines = text.split("\n")
	assert lines[0] == "---", "SKILL.md must open with a --- line"
	end = lines.index("---", 1)

	fields, key = {}, None
	for line in lines[1:end]:
		if line.startswith((" ", "\t")) and key is not None:
			fields[key] = (fields[key] + " " + line.strip()).strip()
		elif ":" in line:
			key, value = line.split(":", 1)
			value = value.strip()
			fields[key] = "" if value in (">", ">-", "|", "|-") else value
	return fields


def test_frontmatter_names_the_skill_and_describes_it():
	"""Claude Code reads only the frontmatter before deciding to load a skill,
	so a missing name or description, a name other than the directory's, or
	a description over 1024 characters stops it loading or triggering."""

	fields = frontmatter(skill_documents()[0][1])
	assert fields.get("name") == "figwig"
	assert fields.get("description")
	assert len(fields["description"]) <= 1024


def test_every_named_file_is_a_skill_path_that_exists():
	"""A file named as `references/x.md` from the skill's root can be opened
	at once; a bare `x.md` has to be searched for."""

	for name, text in skill_documents():
		for target in re.findall(r"`([A-Za-z0-9_/.-]*\.md)`", text):
			assert target == "SKILL.md" or target.startswith("references/"), (
				"{} names `{}` without its path from the skill's root".format(
				name, target))
			assert os.path.isfile(os.path.join(SKILL_SOURCE, target)), (
				"{} names `{}`, which does not exist".format(name, target))


def test_every_reference_is_named_by_the_router():
	"""A reference file that SKILL.md does not name is never opened."""

	router = skill_documents()[0][1]
	for name in os.listdir(os.path.join(SKILL_SOURCE, "references")):
		assert "`references/{}`".format(name) in router, name
