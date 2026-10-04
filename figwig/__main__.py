# __main__.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""
The `figwig` command, also run as `python -m figwig`. `figwig bam2bw`
converts SAM/BAM files or BED/tsv files of fragments into per-base bigWig
tracks (see `figwig.bam2bw`), and `figwig install-skill` installs figwig's
Claude Code skill (see `figwig.install_skill`).
"""

import sys
import argparse


def _parser():
	from . import __version__

	parser = argparse.ArgumentParser(prog='figwig',
		description='The commands of figwig, a reader and writer of bigWig files.')
	parser.add_argument('--version', action='version',
		version='figwig {}'.format(__version__))

	commands = parser.add_subparsers(dest='command', metavar='command',
		required=True)
	commands.add_parser('bam2bw', help="Convert SAM/BAM or BED/tsv files into "
		"per-base bigWig tracks of reads' 5' ends; see figwig bam2bw -h.")

	install = commands.add_parser('install-skill', help="Install figwig's "
		"Claude Code skill into a skills directory.", description="Copy "
		"figwig's Claude Code skill, which teaches a coding agent how to read "
		"and write bigWigs with figwig, into DIRECTORY/figwig.")
	install.add_argument('-d', '--directory', default=None, help="The skills "
		"directory to install into. Default is ~/.claude/skills.")
	install.add_argument('--symlink', action='store_true', help="Link to the "
		"skill inside the installed package instead of copying it, so that "
		"edits to the package show up at once; the link breaks if the package "
		"moves or is uninstalled.")
	install.add_argument('-f', '--force', action='store_true', help="Remove "
		"whatever is at the destination first, such as the skill of an older "
		"figwig.")
	return parser


def main(argv=None):
	argv = sys.argv[1:] if argv is None else list(argv)

	# A command's own arguments are parsed by the command, so that its usage
	# and errors are its own.
	if len(argv) > 0 and argv[0] == 'bam2bw':
		from . import bam2bw
		bam2bw.main(argv[1:], prog='figwig bam2bw', fast_exit=True)
		return

	args = _parser().parse_args(argv)
	if args.command == 'install-skill':
		from . import install_skill
		install_skill.run(args)


if __name__ == '__main__':
	main()
