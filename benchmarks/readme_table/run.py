# run.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Measure the rows of the README's table and rewrite the table.

	python benchmarks/readme_table/run.py                       # every row
	python benchmarks/readme_table/run.py bulk-1 fiber          # some rows
	python benchmarks/readme_table/run.py --render              # print the table
	python benchmarks/readme_table/run.py --update-readme       # rewrite README.md

Rows: bulk-1, bulk-48, bulk-1024 (`bulk.py`), scatac (`scatac.py`), fiber
(`fiber.py`), hic (`hic.py`) and schic (`schic.py`). Each row compares figwig
with what most people use for that assay, as its docstring says, on inputs
from the test-data collection ($TESTDATA_ROOT, default ~/datasets/testdata),
and returns read times, write times and sizes (see `common.py`). A row run
again replaces its entry in `results.json`; `--render` and `--update-readme`
only read that file, so a row can be rerun without the others.

Run it in an environment that has figwig and the baselines (the repository's
`benchmarks/readme_table/requirements.txt`; `uv venv --python 3.12 .bench-venv`,
then `uv pip install -e ".[bam2bw,fast,isal]" -r benchmarks/readme_table/requirements.txt`),
on a machine with most of its cores idle: it refuses to start below
`--min-idle` percent idle CPU. Every measurement is its own process, the
median of `--repeats`. `--quick` runs every row on the smallest input tiers, once, and saves nothing (to test a row). Scratch files go in `--scratch` (default
/dev/shm/figwig-readme-table-$USER, which must hold about 100 GB for the
Hi-C rows) and are removed at the end unless `--keep` is given.
"""

import os
import sys
import json
import time
import shutil
import getpass
import argparse
import datetime
import importlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common  # noqa: E402
import render  # noqa: E402

MODULES = {'bulk-1': 'bulk', 'bulk-48': 'bulk', 'bulk-1024': 'bulk', 'scatac': 'scatac',
	'fiber': 'fiber', 'hic': 'hic', 'schic': 'schic'}
RESULTS = os.path.join(HERE, 'results.json')
README = os.path.join(common.REPO, 'README.md')


def load():
	if os.path.exists(RESULTS):
		with open(RESULTS) as handle:
			return json.load(handle)
	return {'meta': {}, 'rows': {}}


def save(results):
	with open(RESULTS, 'w') as handle:
		json.dump(results, handle, indent='\t', sort_keys=True)
		handle.write('\n')


def run_worker(args):
	module, function = args.worker.split(':')
	function = getattr(importlib.import_module(module), function)
	out = function(json.loads(args.ctx), **json.loads(args.kwargs))
	print(json.dumps(out))


def main():
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('rows', nargs='*', help='one or more of: ' + ', '.join(MODULES))
	parser.add_argument('--threads', type=int, default=8, help="figwig's threads")
	parser.add_argument('--repeats', type=int, default=3)
	parser.add_argument('--scratch', default='/dev/shm/figwig-readme-table-{}'.format(getpass.getuser()))
	parser.add_argument('--quick', action='store_true', help='the smallest input tiers and one repeat, to test a row; nothing is saved')
	parser.add_argument('--keep', action='store_true')
	parser.add_argument('--min-idle', type=float, default=70.0)
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--update-readme', action='store_true')
	parser.add_argument('--worker', help=argparse.SUPPRESS)
	parser.add_argument('--ctx', help=argparse.SUPPRESS)
	parser.add_argument('--kwargs', help=argparse.SUPPRESS)
	args = parser.parse_args()
	for key in args.rows:
		if key not in MODULES:
			parser.error('unknown row {}'.format(key))

	if args.worker:
		return run_worker(args)

	if args.render or args.update_readme:
		results = load()
		if args.render:
			print(render.block(results))
		if args.update_readme:
			with open(README) as handle:
				text = handle.read()
			with open(README, 'w') as handle:
				handle.write(render.update(text, results))
		return

	idle = common.idle_percent()
	if idle < args.min_idle:
		sys.exit('the machine is {:.0f}% idle, below --min-idle {:.0f}'.format(idle, args.min_idle))

	wanted = args.rows or [k for k in render.ORDER]
	results = load()
	scratch = os.path.join(args.scratch, 'run-{}'.format(os.getpid()))
	os.makedirs(scratch)
	ctx = {'scratch': scratch, 'threads': args.threads, 'repeats': args.repeats,
		'testdata': common.TESTDATA, 'python': sys.executable, 'quick': args.quick}
	if args.quick:
		ctx['repeats'] = 1
	results['meta'] = {'threads': args.threads, 'repeats': args.repeats,
		'versions': common.versions(), 'machine': common.machine(), 'commit': common.commit(),
		'idle_percent_at_start': round(idle, 1)}
	try:
		for key in wanted:
			module = importlib.import_module(MODULES[key])
			start = time.time()
			print('=== {}'.format(key), flush=True)
			row = module.ROWS[key](ctx)
			row['date'] = datetime.datetime.now().isoformat(timespec='seconds')
			row['elapsed_seconds'] = round(time.time() - start, 1)
			results['rows'][key] = row
			if not args.quick:
				save(results)
			print('{} done in {:.0f} s'.format(key, time.time() - start), flush=True)
	finally:
		if not args.keep:
			shutil.rmtree(scratch, ignore_errors=True)
	print(render.table(results['rows']))


if __name__ == '__main__':
	main()
