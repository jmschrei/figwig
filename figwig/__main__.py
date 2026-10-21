# __main__.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""
The `figwig` command, also run as `python -m figwig`. `figwig bam2bw`
converts SAM/BAM files or BED/tsv files of fragments into per-base bigWig
tracks (see `figwig.bam2bw`), `figwig bam2fw` converts them into one figwig
file (see `figwig.bam2fw`), `figwig bw2fw` and `figwig fw2bw` convert
between bigWigs and figwig files (see `figwig.bw2fw` and `figwig.fw2bw`),
`figwig reorder` writes a figwig file in the other order (see
`figwig.reorder`), `figwig merge` writes the tracks of several figwig files
into one (see `figwig.merge`), and `figwig install-skill` installs figwig's
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
	commands.add_parser('bam2fw', help="Convert SAM/BAM or BED/tsv files into "
		"one figwig file of per-base counts, a track per strand of each file; "
		"see figwig bam2fw -h.")
	commands.add_parser('bw2fw', help="Convert bigWig files into one figwig "
		"file, one track per bigWig; see figwig bw2fw -h.")
	commands.add_parser('fw2bw', help="Convert a figwig file's tracks into "
		"bigWig files; see figwig fw2bw -h.")
	commands.add_parser('reorder', help="Write a figwig file again in "
		"position order or in track order; see figwig reorder -h.")
	commands.add_parser('merge', help="Write the tracks of several figwig "
		"files into one, copying their chunks; see figwig merge -h.")

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
	if len(argv) > 0 and argv[0] == 'bam2fw':
		from . import bam2fw
		bam2fw.main(argv[1:], prog='figwig bam2fw')
		return
	if len(argv) > 0 and argv[0] == 'bw2fw':
		from . import bw2fw
		bw2fw.main(argv[1:], prog='figwig bw2fw')
		return
	if len(argv) > 0 and argv[0] == 'fw2bw':
		from . import fw2bw
		fw2bw.main(argv[1:], prog='figwig fw2bw')
		return
	if len(argv) > 0 and argv[0] == 'reorder':
		from . import reorder
		reorder.main(argv[1:], prog='figwig reorder')
		return
	if len(argv) > 0 and argv[0] == 'merge':
		from . import merge
		merge.main(argv[1:], prog='figwig merge')
		return

	args = _parser().parse_args(argv)
	if args.command == 'install-skill':
		from . import install_skill
		install_skill.run(args)


if __name__ == '__main__':
	main()
