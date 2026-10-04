# install_skill.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""
The `figwig install-skill` command. It copies figwig's Claude Code skill,
which ships as package data in `figwig/skills/figwig`, into a skills
directory, `~/.claude/skills` unless given. Claude Code does not look for
skills inside site-packages, and nothing is written into a home directory on
import or install, so the skill is installed only by this command.
"""

import os
import shutil


def run(args):
	"""Install the skill as `args` ask: into `args.directory`, or
	`~/.claude/skills` if it is None, as `<directory>/figwig`; as a symlink to
	the bundled copy if `args.symlink`; and replacing whatever is there if
	`args.force`, which is otherwise an error."""

	source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills",
		"figwig")
	if not os.path.isdir(source):
		raise FileNotFoundError("Bundled skill not found at {}; the package may "
			"be installed without its data files.".format(source))

	if args.directory is not None:
		skills_dir = os.path.expanduser(args.directory)
	else:
		skills_dir = os.path.expanduser(os.path.join("~", ".claude", "skills"))

	os.makedirs(skills_dir, exist_ok=True)
	dest = os.path.join(skills_dir, "figwig")

	# With --force this would delete the bundled skill before copying it.
	if os.path.realpath(dest) == os.path.realpath(source):
		raise ValueError("{} is the bundled skill itself; choose another "
			"directory.".format(dest))

	# lexists, so that a dangling symlink from an earlier --symlink install,
	# which exists() does not see, is found and replaced too.
	if os.path.lexists(dest):
		if not args.force:
			raise FileExistsError("A skill already exists at {}. Re-run with "
				"--force to overwrite it.".format(dest))

		if os.path.islink(dest) or os.path.isfile(dest):
			os.remove(dest)
		else:
			shutil.rmtree(dest)

	if args.symlink:
		os.symlink(source, dest)
		print("Symlinked the figwig skill:\n  {} -> {}".format(dest, source))
	else:
		shutil.copytree(source, dest,
			ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__"))
		print("Installed the figwig skill to:\n  {}".format(dest))

	print("Restart Claude Code, or reload its skills, to pick it up.")
