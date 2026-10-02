# compare_readers.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Time figwig against pybigtools, pyBigWig and pybbi on the same windows.

	uv run --with pybigtools==0.2.5 --with pyBigWig==0.3.26 --with pybbi==0.4.2 \
		python benchmarks/compare_readers.py --bigwig signal.bw \
		--bed peaks.bed --bed negatives.bed --out results.jsonl

Each window is `--width` bases centred on one region of the BED files, which
are read in the order given: mid = start + (end - start) // 2, and the window
is [mid - width // 2, mid - width // 2 + width).

Every library writes window j into row j of a float32 array of shape
(n, width), except that figwig writes the windows in the order it is given
them, and a sorted read is put back in order after the timing stops:

	- figwig: one BigWig.read over every window, on `threads` threads.
	- pybigtools: one values(arr=) per window, into a float64 row.
	- pyBigWig: one values(numpy=True) per window.
	- pybbi: one stackup over the windows.

At more than one thread, pybigtools, pyBigWig and pybbi split the windows into
`threads` contiguous chunks read on a thread pool, each chunk with its own
file handle. "sorted" gives the windows sorted by chromosome and start, which
helps the others' caching; figwig sorts them itself.

Each configuration of library, threads, order and bigWig runs in its own
process, which warms up on the first 1,000 windows, then opens the file again
and times one full read, opening included. Every repetition runs every
configuration once, in an order rotated from the last repetition's, and the
table gives the median and range over repetitions. Threads are pinned through
OMP, MKL, OpenBLAS, NUMBA and RAYON.

Each run records the SHA-256 of numpy.nan_to_num of its output. pyBigWig gives
NaN for bases no interval covers, where the others give 0, and nan_to_num
makes the two agree. The table reports how many distinct outputs each bigWig
produced across every run, which is 1 when all the libraries agree.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import itertools
import statistics
import subprocess

from concurrent.futures import ThreadPoolExecutor

import numpy


LIBRARIES = ['figwig', 'pybigtools', 'pybigwig', 'pybbi']


def load_windows(beds, width):
	"""Chromosome names, starts, and the order that sorts the windows."""

	names, starts = [], []
	for bed in beds:
		with open(bed) as handle:
			for line in handle:
				if line.startswith(('#', 'track', 'browser')) or not line.strip():
					continue

				chrom, start, end = line.split('\t')[:3]
				mid = int(start) + (int(end) - int(start)) // 2
				names.append(chrom)
				starts.append(mid - width // 2)

	names, starts = numpy.array(names), numpy.array(starts, dtype=numpy.int64)
	return names, starts, numpy.lexsort((starts, names))


def chunks(idx, k):
	return [c for c in numpy.array_split(idx, k) if len(c)]


def read_figwig(path, names, starts, idx, out, threads, width):
	from figwig import BigWig

	return BigWig(path).read(names[idx], starts[idx], width, n_jobs=threads)


def read_pybigtools(path, names, starts, idx, out, threads, width):
	import pybigtools

	def work(part):
		bw = pybigtools.open(path)
		scratch = numpy.empty(width, dtype=numpy.float64)
		for j in part:
			bw.values(names[j], int(starts[j]), int(starts[j]) + width, arr=scratch)
			out[j] = scratch
		bw.close()

	with ThreadPoolExecutor(threads) as pool:
		list(pool.map(work, chunks(idx, threads)))


def read_pybigwig(path, names, starts, idx, out, threads, width):
	import pyBigWig

	def work(part):
		bw = pyBigWig.open(path)
		for j in part:
			out[j] = bw.values(names[j], int(starts[j]), int(starts[j]) + width,
				numpy=True)
		bw.close()

	with ThreadPoolExecutor(threads) as pool:
		list(pool.map(work, chunks(idx, threads)))


def read_pybbi(path, names, starts, idx, out, threads, width):
	import bbi

	def work(part):
		out[part] = bbi.stackup(path, names[part], starts[part], starts[part] +
			width)

	with ThreadPoolExecutor(threads) as pool:
		list(pool.map(work, chunks(idx, threads)))


READERS = {'figwig': read_figwig, 'pybigtools': read_pybigtools,
	'pybigwig': read_pybigwig, 'pybbi': read_pybbi}

DISTRIBUTIONS = {'figwig': 'figwig', 'pybigtools': 'pybigtools',
	'pybigwig': 'pyBigWig', 'pybbi': 'pybbi'}


def time_one(args):
	"""Time one configuration and print it as a JSON line."""

	names, starts, sorted_order = load_windows(args.bed, args.width)
	order = sorted_order if args.order == 'sorted' else numpy.arange(len(starts))
	read = READERS[args.lib]

	warm = numpy.empty((len(starts), args.width), dtype=numpy.float32)
	read(args.bigwig[0], names, starts, order[:1000], warm, args.threads,
		args.width)
	del warm

	out = numpy.empty((len(starts), args.width), dtype=numpy.float32)
	start = time.perf_counter()
	given = read(args.bigwig[0], names, starts, order, out, args.threads,
		args.width)
	seconds = time.perf_counter() - start
	if given is not None:
		out[order] = given

	from importlib.metadata import version
	out = numpy.nan_to_num(out)
	print(json.dumps({'lib': args.lib, 'version': version(DISTRIBUTIONS[
		args.lib]), 'threads': args.threads,
		'order': args.order, 'bigwig': os.path.basename(args.bigwig[0]),
		'seconds': seconds, 'windows': len(starts),
		'sha256': hashlib.sha256(out.view(numpy.uint8)).hexdigest(),
		'nonzero': int((out != 0).sum()), 'python': sys.version.split()[0],
		'numpy': numpy.__version__}))


def table(records):
	"""A markdown table of median seconds per configuration, per bigWig."""

	lines = []
	for bigwig in sorted({r['bigwig'] for r in records}):
		rows = [r for r in records if r['bigwig'] == bigwig]
		lines.append('**{}**: {} windows of {} runs; distinct outputs: {}\n'.format(
			bigwig, rows[0]['windows'], len(rows), len({r['sha256'] for r in
			rows})))

		configs = sorted({(r['threads'], r['order']) for r in rows})
		lines.append('| library | ' + ' | '.join('{} thread{}, {}'.format(t,
			's' if t > 1 else '', o) for t, o in configs) + ' |')
		lines.append('|---' * (len(configs) + 1) + '|')
		for lib in LIBRARIES:
			cells = []
			for t, o in configs:
				ts = [r['seconds'] for r in rows if (r['lib'], r['threads'],
					r['order']) == (lib, t, o)]
				cells.append('{:.3f} s ({:.3f}-{:.3f})'.format(statistics.median(
					ts), min(ts), max(ts)) if ts else '')

			versions = {r['version'] for r in rows if r['lib'] == lib}
			lines.append('| {} {} | '.format(lib, ', '.join(sorted(versions))) +
				' | '.join(cells) + ' |')

		lines.append('')

	return '\n'.join(lines)


def run(args):
	"""Run every configuration, `args.repeats` times, each in its own process."""

	configs = list(itertools.product(args.libs, args.threads_list, ['given',
		'sorted'], args.bigwig))
	for rep in range(args.repeats):
		shift = rep * 11 % len(configs)
		for lib, threads, order, bigwig in configs[shift:] + configs[:shift]:
			t = str(threads)
			env = dict(os.environ, OMP_NUM_THREADS=t, MKL_NUM_THREADS=t,
				OPENBLAS_NUM_THREADS=t, NUMBA_NUM_THREADS=t, RAYON_NUM_THREADS=t)
			command = [sys.executable, os.path.abspath(__file__), '--one', '--lib',
				lib, '--threads', t, '--order', order, '--bigwig', bigwig,
				'--width', str(args.width)]
			for bed in args.bed:
				command += ['--bed', bed]

			result = subprocess.run(command, env=env, capture_output=True,
				text=True)
			if result.returncode != 0:
				print('FAILED', lib, threads, order, bigwig, result.stderr[-1500:],
					flush=True)
				continue

			record = json.loads(result.stdout.strip().splitlines()[-1])
			record['rep'] = rep
			with open(args.out, 'a') as handle:
				handle.write(json.dumps(record) + '\n')

			print(rep, lib, threads, order, record['bigwig'], '{:.3f} s'.format(
				record['seconds']), record['sha256'][:10], flush=True)

	with open(args.out) as handle:
		print(table([json.loads(line) for line in handle]))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('--bigwig', action='append', required=True)
	parser.add_argument('--bed', action='append', required=True)
	parser.add_argument('--width', type=int, default=1000)
	parser.add_argument('--out', default='results.jsonl')
	parser.add_argument('--repeats', type=int, default=3)
	parser.add_argument('--libs', nargs='+', default=LIBRARIES,
		choices=LIBRARIES)
	parser.add_argument('--threads-list', type=int, nargs='+', default=[1, 8])
	parser.add_argument('--one', action='store_true', help=argparse.SUPPRESS)
	parser.add_argument('--lib', help=argparse.SUPPRESS)
	parser.add_argument('--threads', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--order', help=argparse.SUPPRESS)
	args = parser.parse_args()

	if args.one:
		time_one(args)
	else:
		run(args)
