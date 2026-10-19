# compare_formats.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Time figwig files against bigWigs holding the same tracks, for a range of
numbers of tracks and of threads, reading and writing.

	uv run python benchmarks/compare_formats.py --bigwig a.bw --bigwig b.bw \
		--bed peaks.narrowPeak --workdir /dev/shm/formats --out results.jsonl \
		--tracks 1 2 6 24 96 --threads-list 1 2 4 8 16 32

The tracks are the bigWigs given, in order, and past them tracks derived
from those bigWigs: track i is bigWig i % k with every chromosome rolled by
a random offset, so that it has the statistics of a real track but shares
no position with it. Each derived track is written as a bigWig too, so
that N bigWigs and the figwig file of the same N tracks hold the same
values. The figwig files are written by `figwig bw2fw`, with groups of 8
tracks, and that conversion is the write benchmark, along with `figwig
fw2bw` back and, with --bam and --sizes, `figwig bam2fw` against `figwig
bam2bw` on one BAM.

The read benchmark is `--windows` windows of `--width` bases centred on
regions of the BED files, drawn at random with a fixed seed, read in one
call: with `read_bigwig` over the N bigWigs into float32, with
`FigWigReader.read` into the tracks' own dtype, and into float32. The
DataLoader pattern is timed too: the first 2,048 of those windows in
batches of 64 on one thread. Each configuration runs in its own process,
which warms up on 1,000 windows, opens the files again and times one read,
opening included; every repetition runs every configuration once, in an
order rotated from the last, and the table gives medians. Conversions run
`--write-repeats` times, once by default, since each is a pass over every
track. Threads are pinned through OMP, MKL, OpenBLAS and NUMBA.
"""

import os
import sys
import json
import time
import argparse
import statistics
import subprocess

import numpy


def load_windows(beds, width, n, random_state=0):
	names, starts = [], []
	for bed in beds:
		opener = __import__('gzip').open if bed.endswith('.gz') else open
		with opener(bed, 'rt') as handle:
			for line in handle:
				if line.startswith(('#', 'track', 'browser')) or not line.strip():
					continue
				fields = line.split('\t')
				chrom, start, end = fields[:3]
				mid = int(start) + (int(end) - int(start)) // 2
				if len(fields) >= 10 and fields[9].strip() not in ('-1', '.'):
					mid = int(start) + int(fields[9])
				names.append(chrom)
				starts.append(mid - width // 2)

	names, starts = numpy.array(names), numpy.array(starts, dtype=numpy.int64)
	keep = starts >= 0
	names, starts = names[keep], starts[keep]
	rng = numpy.random.default_rng(random_state)
	picked = rng.choice(len(starts), size=min(n, len(starts)), replace=False)
	return names[picked], starts[picked]


def track_path(workdir, i):
	return os.path.join(workdir, 'tracks', 'track_{:03d}.bw'.format(i))


def figwig_path(workdir, n):
	return os.path.join(workdir, 'tracks_{}.fw'.format(n))


def prepare_tracks(args, n):
	"""Make sure bigWigs 0 to n - 1 exist in the work directory."""

	import figwig

	os.makedirs(os.path.join(args.workdir, 'tracks'), exist_ok=True)
	sources = args.bigwig
	for i in range(n):
		path = track_path(args.workdir, i)
		if os.path.exists(path):
			continue

		source = sources[i % len(sources)]
		if i < len(sources):
			os.symlink(os.path.abspath(source), path)
			continue

		reader = figwig.BigWigReader(source)
		rng = numpy.random.default_rng(i)
		chrom_sizes = {name: length for name, length in reader.chrom_sizes.items()
			if '_' not in name and name != 'chrEBV'}
		with figwig.BigWigWriter(path, chrom_sizes, zooms=0, level=6,
				n_jobs=args.prepare_threads) as writer:
			for chrom, length in chrom_sizes.items():
				piece = 1 << 22
				starts = numpy.arange(0, length, piece)
				values = reader.read([chrom] * len(starts), starts, piece,
					n_jobs=args.prepare_threads).ravel()[:length]
				values = numpy.roll(values, int(rng.integers(0, length)))
				for a in range(0, length, piece):
					writer.write(chrom, [a], values[None, a:a + piece])
		print('derived track {} from {}'.format(i, os.path.basename(source)),
			file=sys.stderr, flush=True)


def shell(command, env):
	t0 = time.perf_counter()
	result = subprocess.run(command, env=env, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError('{} failed: {}'.format(' '.join(command),
			result.stderr[-2000:]))
	return time.perf_counter() - t0


def pinned_env(threads):
	env = dict(os.environ)
	for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
			'NUMBA_NUM_THREADS'):
		env[key] = str(threads)
	return env


def convert(args, n, threads):
	"""Write the figwig file of n tracks with `figwig bw2fw`, timed."""

	path = figwig_path(args.workdir, n)
	if os.path.exists(path):
		os.remove(path)
	bigwigs = [track_path(args.workdir, i) for i in range(n)]
	seconds = shell([sys.executable, '-m', 'figwig', 'bw2fw', '-o', path, '-p',
		str(threads), '--groups', '8', '-l', str(args.level)] + bigwigs,
		pinned_env(threads))
	return seconds, os.path.getsize(path), sum(os.path.getsize(b) for b in
		bigwigs)


def convert_back(args, n, threads):
	directory = os.path.join(args.workdir, 'back_{}'.format(n))
	seconds = shell([sys.executable, '-m', 'figwig', 'fw2bw', figwig_path(
		args.workdir, n), '-o', directory, '-p', str(threads), '-l', str(
		args.level)], pinned_env(threads))
	size = sum(os.path.getsize(os.path.join(directory, f)) for f in os.listdir(
		directory))
	subprocess.run(['rm', '-rf', directory])
	return seconds, size


def time_one(args):
	"""One read configuration, run in its own process: prints JSON."""

	import figwig

	names, starts = load_windows(args.bed, args.width, args.windows)
	n, threads, mode = args.n, args.threads, args.mode
	if mode == 'bigwig':
		paths = [track_path(args.workdir, i) for i in range(n)]

		def open_():
			return [figwig.BigWigReader(path) for path in paths]

		def read(readers, chroms, starts_, out=None):
			return figwig.read_bigwig(readers, chroms, starts_, args.width,
				n_jobs=threads, out=out)
	else:
		def open_():
			return figwig.FigWigReader(figwig_path(args.workdir, n))

		dtype = 'float32' if mode == 'figwig_float32' else None

		def read(reader, chroms, starts_, out=None):
			return reader.read(chroms, starts_, args.width, n_jobs=threads,
				dtype=dtype, out=out)

	handle = open_()
	y = read(handle, names[:1000], starts[:1000])
	del y, handle

	if args.batches:
		handle = open_()
		t0 = time.perf_counter()
		for a in range(0, 2048, 64):
			read(handle, names[a:a + 64], starts[a:a + 64])
		seconds = time.perf_counter() - t0
		digest = None
	else:
		t0 = time.perf_counter()
		handle = open_()
		y = read(handle, names, starts)
		seconds = time.perf_counter() - t0
		digest = __import__('hashlib').sha256(numpy.ascontiguousarray(
			numpy.nan_to_num(y.astype(numpy.float32))).tobytes()).hexdigest()[:12]

	print(json.dumps({'kind': 'read', 'n': n, 'threads': threads, 'mode': mode,
		'batches': args.batches, 'seconds': seconds, 'digest': digest}))


def run(args):
	records = []
	out = open(args.out, 'a')

	def record(entry):
		records.append(entry)
		out.write(json.dumps(entry) + '\n')
		out.flush()

	for n in args.tracks:
		prepare_tracks(args, n)
		for threads in args.write_threads:
			for repeat in range(args.write_repeats):
				seconds, fw_size, bw_size = convert(args, n, threads)
				record({'kind': 'bw2fw', 'n': n, 'threads': threads, 'seconds':
					seconds, 'figwig_bytes': fw_size, 'bigwig_bytes': bw_size})
				seconds, back_size = convert_back(args, n, threads)
				record({'kind': 'fw2bw', 'n': n, 'threads': threads, 'seconds':
					seconds, 'bigwig_bytes': back_size})
			print('converted {} tracks on {} threads'.format(n, threads),
				file=sys.stderr, flush=True)

	configurations = [(n, t, mode, False) for n in args.tracks for t in
		args.threads_list for mode in ('bigwig', 'figwig', 'figwig_float32')]
	configurations += [(n, 1, mode, True) for n in args.tracks for mode in
		('bigwig', 'figwig')]
	for repeat in range(args.repeats):
		order = configurations[repeat:] + configurations[:repeat]
		for n, threads, mode, batches in order:
			command = [sys.executable, os.path.abspath(__file__), '--one',
				'--workdir', args.workdir, '--n', str(n), '--threads',
				str(threads), '--mode', mode, '--width', str(args.width),
				'--windows', str(args.windows)]
			command += ['--batches'] if batches else []
			for bed in args.bed:
				command += ['--bed', bed]
			result = subprocess.run(command, env=pinned_env(threads),
				capture_output=True, text=True)
			if result.returncode != 0:
				print(result.stderr[-2000:], file=sys.stderr)
				continue
			record(json.loads(result.stdout.strip().splitlines()[-1]))
		print('repetition {} done'.format(repeat + 1), file=sys.stderr,
			flush=True)

	if args.bam is not None:
		for threads in args.write_threads:
			for repeat in range(args.write_repeats):
				name = os.path.join(args.workdir, 'bam')
				seconds = shell([sys.executable, '-m', 'figwig', 'bam2bw', args.bam,
					'-s', args.sizes, '-n', name, '-p', str(threads)],
					pinned_env(threads))
				size = sum(os.path.getsize(name + s) for s in ('.+.bw', '.-.bw'))
				record({'kind': 'bam2bw', 'threads': threads, 'seconds': seconds,
					'bytes': size})
				seconds = shell([sys.executable, '-m', 'figwig', 'bam2fw', args.bam,
					'-s', args.sizes, '-n', name, '-p', str(threads)],
					pinned_env(threads))
				record({'kind': 'bam2fw', 'threads': threads, 'seconds': seconds,
					'bytes': os.path.getsize(name + '.fw')})

	out.close()
	print(table(records))


def table(records):
	lines = []

	def median(rows):
		return statistics.median(row['seconds'] for row in rows)

	by_key = {}
	for row in records:
		if row['kind'] == 'read':
			by_key.setdefault((row['n'], row['threads'], row['mode'],
				row['batches']), []).append(row)
	if by_key:
		lines.append('reading: median seconds')
		lines.append('{:>6} {:>7} {:>14} {:>14} {:>14}'.format('tracks', 'threads',
			'bigWigs f32', 'figwig own', 'figwig f32'))
		for (n, t) in sorted({(k[0], k[1]) for k in by_key if not k[3]}):
			cells = []
			for mode in ('bigwig', 'figwig', 'figwig_float32'):
				rows = by_key.get((n, t, mode, False))
				cells.append('{:14.3f}'.format(median(rows)) if rows else ' ' * 14)
			lines.append('{:>6} {:>7} {}'.format(n, t, ' '.join(cells)))
		lines.append('')
		lines.append('batches of 64 windows, 32 batches, 1 thread: median seconds')
		for n in sorted({k[0] for k in by_key if k[3]}):
			cells = []
			for mode in ('bigwig', 'figwig'):
				rows = by_key.get((n, 1, mode, True))
				cells.append('{:14.3f}'.format(median(rows)) if rows else ' ' * 14)
			lines.append('{:>6} {}'.format(n, ' '.join(cells)))
		lines.append('')

	for kind in ('bw2fw', 'fw2bw', 'bam2bw', 'bam2fw'):
		rows = [row for row in records if row['kind'] == kind]
		if not rows:
			continue
		lines.append('{}: median seconds'.format(kind))
		keys = sorted({(row.get('n', 0), row['threads']) for row in rows})
		for n, t in keys:
			these = [row for row in rows if row.get('n', 0) == n and row['threads']
				== t]
			sizes = ' '.join('{}={:.1f}MB'.format(key, these[0][key] / 1e6) for key
				in ('figwig_bytes', 'bigwig_bytes', 'bytes') if key in these[0])
			lines.append('{:>6} {:>7} {:10.2f}  {}'.format(n, t, median(these),
				sizes))
		lines.append('')

	return '\n'.join(lines)


def main():
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('--bigwig', action='append', default=[])
	parser.add_argument('--bed', action='append', default=[])
	parser.add_argument('--bam', default=None)
	parser.add_argument('--sizes', default=None)
	parser.add_argument('--workdir', required=True)
	parser.add_argument('--out', default='results.jsonl')
	parser.add_argument('--width', type=int, default=2114)
	parser.add_argument('--windows', type=int, default=8192)
	parser.add_argument('--tracks', type=int, nargs='+', default=[1, 2, 6, 24,
		96])
	parser.add_argument('--threads-list', type=int, nargs='+', default=[1, 2, 4,
		8, 16, 32])
	parser.add_argument('--write-threads', type=int, nargs='+', default=[1, 8,
		32])
	parser.add_argument('--prepare-threads', type=int, default=16)
	parser.add_argument('--level', type=int, default=6)
	parser.add_argument('--repeats', type=int, default=3, help="Repetitions of "
		"each read configuration.")
	parser.add_argument('--write-repeats', type=int, default=1, help="Repetitions "
		"of each conversion.")
	parser.add_argument('--table', default=None, help="Print the table of an "
		"existing results file and exit.")
	parser.add_argument('--one', action='store_true', help=argparse.SUPPRESS)
	parser.add_argument('--n', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--threads', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--mode', help=argparse.SUPPRESS)
	parser.add_argument('--batches', action='store_true', help=argparse.SUPPRESS)
	args = parser.parse_args()

	if args.table is not None:
		with open(args.table) as handle:
			print(table([json.loads(line) for line in handle if line.strip()]))
	elif args.one:
		time_one(args)
	else:
		os.makedirs(args.workdir, exist_ok=True)
		run(args)


if __name__ == '__main__':
	main()
