# compare_writers.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Time figwig's writer against pyBigWig and pybigtools on the same values.

	uv run --with pybigtools==0.2.5 --with pyBigWig==0.3.26 --with deflate \
		python benchmarks/compare_writers.py --bigwig counts.bw \
		--bigwig signal.bw --scratch /dev/shm/writers --out results.jsonl

The values come from existing bigWigs: every interval of every chromosome,
read once with pyBigWig's intervals() and cached as a .npz in --scratch.
A bigWig whose intervals are all one base long is written as single bases
(varStep), and any other as intervals (bedGraph):

	- figwig: one BigWigWriter.write per chromosome, single bases as windows
	  of width 1 and intervals with ends, with missing=NaN so that every
	  item is written, at level 6, with zlib or libdeflate, on `threads`
	  threads.
	- pyBigWig: one addEntries per chromosome, which compresses with zlib at
	  level 6, on one thread.
	- pybigtools: one write() from an iterator of (chrom, start, end, value)
	  tuples. It always writes zoom levels, so it is only run with them.

Each configuration of library, engine, threads, zoom levels and bigWig runs
in its own process, which loads the cached values, imports the library and
warms up on a write of the first 1,000 values of each chromosome, then times
one write, opening the file and closing it included. Every repetition runs every
configuration once, in an order rotated from the last repetition's, and the
table gives the median and range over repetitions, the size of the file and
the process's peak resident memory. Threads are pinned through OMP, MKL,
OpenBLAS, NUMBA and RAYON.

The peak memory is Linux's VmHWM, which counts only the process's own
memory. Its ru_maxrss would not do: on Linux it carries over the memory of
the process that started it, here the parent, which holds every interval of
a bigWig while it caches them. Elsewhere ru_maxrss is all there is.

Each run then reads 2,000 windows of 1,000 bases, at positions drawn from a
fixed seed on the chromosomes with values that are longer than 1,000 bases,
from the file it wrote with
figwig's reader, and records the SHA-256 of the result. The table reports
how many distinct results each bigWig gave across every run, which is 1
when every writer wrote the same values.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import resource
import itertools
import statistics
import subprocess

import numpy


def entries_path(scratch, bigwig):
	return os.path.join(scratch, os.path.basename(bigwig) + '.entries.npz')


def cache_entries(scratch, bigwig):
	"""Every interval of `bigwig`, chromosome by chromosome, as a .npz."""

	path = entries_path(scratch, bigwig)
	if os.path.exists(path):
		return path

	import pyBigWig

	bw = pyBigWig.open(bigwig)
	chroms = bw.chroms()
	arrays = {'names': numpy.array(list(chroms)), 'lengths': numpy.array(list(
		chroms.values()), dtype=numpy.int64)}
	for i, chrom in enumerate(chroms):
		intervals = bw.intervals(chrom)
		if intervals:
			table = numpy.array(intervals, dtype=numpy.float64)
			arrays['starts_{}'.format(i)] = table[:, 0].astype(numpy.int64)
			arrays['ends_{}'.format(i)] = table[:, 1].astype(numpy.int64)
			arrays['values_{}'.format(i)] = table[:, 2].astype(numpy.float32)

	numpy.savez(path, **arrays)
	return path


def load_entries(path):
	with numpy.load(path) as data:
		chroms = dict(zip(data['names'].tolist(), data['lengths'].tolist()))
		entries = {}
		for i, chrom in enumerate(chroms):
			if 'starts_{}'.format(i) in data:
				entries[chrom] = (data['starts_{}'.format(i)], data['ends_{}'.format(
					i)], data['values_{}'.format(i)])

	single = all(numpy.all(e - s == 1) for s, e, _ in entries.values())
	return chroms, entries, single


def write_figwig(path, chroms, entries, single, engine, threads, zooms):
	from figwig import BigWigWriter

	with BigWigWriter(path, chroms, zooms=zooms, level=6, engine=engine,
			n_jobs=threads) as writer:
		for chrom, (starts, ends, values) in entries.items():
			if single:
				writer.write(chrom, starts, values[:, None], missing=numpy.nan)
			else:
				writer.write(chrom, starts, values, ends=ends, missing=numpy.nan)


def write_pybigwig(path, chroms, entries, single, engine, threads, zooms):
	import pyBigWig

	bw = pyBigWig.open(path, 'w')
	bw.addHeader(list(chroms.items()), maxZooms=zooms)
	for chrom, (starts, ends, values) in entries.items():
		if single:
			bw.addEntries(chrom, starts, values=values.astype(numpy.float64),
				span=1)
		else:
			bw.addEntries([chrom] * len(starts), starts, ends=ends,
				values=values.astype(numpy.float64))
	bw.close()


def write_pybigtools(path, chroms, entries, single, engine, threads, zooms):
	import pybigtools

	def items():
		for chrom, (starts, ends, values) in entries.items():
			yield from zip(itertools.repeat(chrom), starts.tolist(), ends.tolist(),
				values.tolist())

	bw = pybigtools.open(path, 'w')
	bw.write(chroms, items())


WRITERS = {'figwig': write_figwig, 'pybigwig': write_pybigwig,
	'pybigtools': write_pybigtools}


def windows_hash(path, chroms, entries):
	"""The SHA-256 of 2,000 windows of 1,000 bases read back with figwig."""

	from figwig import BigWigReader

	rng = numpy.random.default_rng(0)
	names = sorted(name for name in entries if chroms[name] > 1000)
	picks = rng.choice(len(names), 2000)
	chosen = numpy.array([names[i] for i in picks])
	starts = numpy.array([rng.integers(0, chroms[name] - 1000) for name in
		chosen], dtype=numpy.int64)
	y = BigWigReader(path).read(chosen, starts, width=1000)
	return hashlib.sha256(y.tobytes()).hexdigest()


def peak_mb():
	"""This process's peak resident memory, in MB: VmHWM where /proc has
	it, and otherwise ru_maxrss, in bytes on macOS and in KB elsewhere."""

	try:
		with open('/proc/self/status') as handle:
			for line in handle:
				if line.startswith('VmHWM:'):
					return int(line.split()[1]) / 1024
	except OSError:
		pass

	rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	return rss / 2**20 if sys.platform == 'darwin' else rss / 1024


def time_one(args):
	chroms, entries, single = load_entries(entries_path(args.scratch,
		args.bigwig))
	path = os.path.join(args.scratch, 'out-{}-{}-{}-{}.bw'.format(args.lib,
		args.engine, args.threads, args.zooms))

	warm = {chrom: tuple(x[:1000] for x in arrays) for chrom, arrays in
		entries.items()}
	WRITERS[args.lib](path, chroms, warm, single, args.engine, args.threads,
		args.zooms)

	start = time.perf_counter()
	WRITERS[args.lib](path, chroms, entries, single, args.engine, args.threads,
		args.zooms)
	seconds = time.perf_counter() - start

	record = {'bigwig': args.bigwig, 'lib': args.lib, 'engine': args.engine,
		'threads': args.threads, 'zooms': args.zooms, 'seconds': seconds,
		'bytes': os.path.getsize(path), 'peak_mb': peak_mb(), 'hash':
		windows_hash(path, chroms, entries), 'items': int(sum(len(s) for s, _, _
		in entries.values())), 'single': single}
	print(json.dumps(record), flush=True)


def configurations(args):
	configs = []
	for bigwig in args.bigwig:
		for zooms in args.zooms_list:
			for engine in args.engines:
				for threads in args.threads_list:
					configs.append((bigwig, 'figwig', engine, threads, zooms))
			configs.append((bigwig, 'pybigwig', 'zlib', 1, zooms))
			if zooms > 0 and 'pybigtools' in args.libs:
				configs.append((bigwig, 'pybigtools', '-', 1, zooms))

	return [c for c in configs if c[1] in args.libs]


def table(records):
	rows = {}
	for record in records:
		key = (record['bigwig'], record['zooms'], record['lib'], record['engine'],
			record['threads'])
		rows.setdefault(key, []).append(record)

	print('| bigWig | zooms | writer | threads | seconds, median (range) | size | '
		'peak memory |')
	print('|---|---|---|---|---|---|---|')
	for key in sorted(rows):
		times = [r['seconds'] for r in rows[key]]
		r = rows[key][0]
		name = r['lib'] if r['lib'] != 'figwig' else 'figwig, ' + r['engine']
		print('| {} | {} | {} | {} | {:.2f} ({:.2f}-{:.2f}) | {:.1f} MB | {:.0f} '
			'MB |'.format(os.path.basename(key[0]), key[1], name, key[4],
			statistics.median(times), min(times), max(times), r['bytes'] / 1e6,
			statistics.median(x['peak_mb'] for x in rows[key])))

	for bigwig in sorted({r['bigwig'] for r in records}):
		hashes = {r['hash'] for r in records if r['bigwig'] == bigwig}
		items = {r['items'] for r in records if r['bigwig'] == bigwig}
		print('{}: {} items; {} distinct read-back result(s)'.format(
			os.path.basename(bigwig), items.pop(), len(hashes)))


def run(args):
	os.makedirs(args.scratch, exist_ok=True)
	for bigwig in args.bigwig:
		cache_entries(args.scratch, bigwig)

	configs = configurations(args)
	records = []
	for repeat in range(args.repeats):
		shift = repeat % len(configs)
		for bigwig, lib, engine, threads, zooms in configs[shift:] + configs[:shift]:
			env = dict(os.environ)
			for name in ('OMP', 'MKL', 'OPENBLAS', 'NUMBA', 'RAYON'):
				env[name + '_NUM_THREADS'] = str(threads)

			out = subprocess.run([sys.executable, __file__, '--one', '--bigwig',
				bigwig, '--lib', lib, '--engine', engine, '--threads', str(threads),
				'--zooms', str(zooms), '--scratch', args.scratch], env=env,
				capture_output=True, text=True, check=True)
			record = json.loads(out.stdout.strip().splitlines()[-1])
			records.append(record)
			with open(args.out, 'a') as handle:
				handle.write(json.dumps(record) + '\n')

	table(records)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('--bigwig', action='append', required=True)
	parser.add_argument('--scratch', required=True)
	parser.add_argument('--out', default='results.jsonl')
	parser.add_argument('--repeats', type=int, default=3)
	parser.add_argument('--libs', nargs='+', default=list(WRITERS),
		choices=list(WRITERS))
	parser.add_argument('--engines', nargs='+', default=['zlib', 'libdeflate'])
	parser.add_argument('--threads-list', type=int, nargs='+', default=[1, 8])
	parser.add_argument('--zooms-list', type=int, nargs='+', default=[0, 10])
	parser.add_argument('--one', action='store_true', help=argparse.SUPPRESS)
	parser.add_argument('--lib', help=argparse.SUPPRESS)
	parser.add_argument('--engine', help=argparse.SUPPRESS)
	parser.add_argument('--threads', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--zooms', type=int, help=argparse.SUPPRESS)
	args = parser.parse_args()

	if args.one:
		args.bigwig = args.bigwig[0]
		time_one(args)
	else:
		run(args)
