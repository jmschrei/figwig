# __main__.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""
The `figwig` command, also run as `python -m figwig`. Its one command,
`figwig bam2bw`, converts SAM/BAM files or BED/tsv files of fragments into
per-base bigWig tracks (see `figwig.bam2bw`).
"""

import sys
import argparse


def _parser():
	from . import __version__

	parser = argparse.ArgumentParser(prog='figwig',
		description='Commands that write bigWig files with figwig.')
	parser.add_argument('--version', action='version',
		version='figwig {}'.format(__version__))

	commands = parser.add_subparsers(dest='command', metavar='command',
		required=True)
	commands.add_parser('bam2bw', help="Convert SAM/BAM or BED/tsv files into "
		"per-base bigWig tracks of reads' 5' ends; see figwig bam2bw -h.")
	return parser


def main(argv=None):
	argv = sys.argv[1:] if argv is None else list(argv)

	# A command's own arguments are parsed by the command, so that its usage
	# and errors are its own.
	if len(argv) > 0 and argv[0] == 'bam2bw':
		from . import bam2bw
		bam2bw.main(argv[1:], prog='figwig bam2bw', fast_exit=True)
		return

	_parser().parse_args(argv)


if __name__ == '__main__':
	main()
