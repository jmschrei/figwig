# compare_sparse.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Time the ways of holding many tracks against each other: bigWigs, figwig
files in groups as figwig 0.1 wrote them, and figwig files in one group or
in track order read densely, as sparse entries and summed per label.

	uv run python benchmarks/compare_sparse.py --bigwig a.bw --bigwig b.bw \
		--bed peaks.narrowPeak --cuts called_cuts.npz --workdir /dev/shm/sparse \
		--out results.jsonl

Three sets of tracks:

- the bigWigs given (six genome-wide count tracks in the paper's runs);
- 96 tracks, the six plus copies rolled by a random offset per chromosome,
  made as `compare_formats.py` makes them;
- the single-cell cut sites of `--cuts`, a .npz from the figwig-singlecell
  experiment's `filter_cells.py`: tid, pos, cell, count, barcodes, chroms,
  lengths, one track per cell.

For each set the files are: the N bigWigs (not for cells); a figwig file in
groups of 8 (64 for cells) written as figwig 0.1 wrote them, with
`_MATRIX_CHUNKS` off; a figwig file in position order with every track in
one group, compressed and uncompressed; and a figwig file in track order.
The reads are `--windows` windows of `--width` bases on the BED regions'
midpoints, in one call on 1 and 8 threads and as 32 batches of 64 windows
on one thread: dense, with `sparse=True`, and with `labels` (one label per
strand for the six, six labels for the 96, ten random labels for cells), for
every track and for a subset of 500 cells; and for cells also every cut of
20 cells over the genome with `read_entries`. Each configuration runs in
its own process after a warm-up on 1,000 windows, opening included, and
the table gives the median of `--repeats`. Every configuration of one set
and selection is checked to give the same values, through the dense array
a sparse result fills.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import statistics
import subprocess

import numpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_formats import load_windows  # noqa: E402
from compare_formats import pinned_env  # noqa: E402
from compare_formats import prepare_tracks  # noqa: E402
from compare_formats import track_path  # noqa: E402


def shell(command, env):
	t0 = time.perf_counter()
	result = subprocess.run(command, env=env, capture_output=True, text=True)
	if result.returncode != 0:
		raise RuntimeError('{} failed: {}'.format(' '.join(command),
			result.stderr[-2000:]))
	return time.perf_counter() - t0


def build_track_files(args, n, records):
	"""The figwig files of n bigWig tracks: groups of 8 as figwig 0.1 wrote
	them, one group compressed, one group raw, and track order."""

	prepare_tracks(args, n)
	bigwigs = [track_path(args.workdir, i) for i in range(n)]
	files = {}
	variants = [('groups8_old', ['--groups', '8', '-l', '6'], True),
		('onegroup_zlib', ['--groups', str(n), '-l', '6'], False),
		('onegroup_raw', ['--groups', str(n), '-l', '0'], False)]
	for name, flags, old in variants:
		path = os.path.join(args.workdir, 'tracks{}_{}.fw'.format(n, name))
		files[name] = path
		if os.path.exists(path):
			continue
		env = pinned_env(8)
		if old:
			env['FIGWIG_NO_MATRIX'] = '1'
		command = [sys.executable, '-c', _BW2FW_OLD if old else _BW2FW, '-o',
			path, '-p', '8'] + flags + bigwigs
		seconds = shell(command, env)
		records.append({'kind': 'build', 'set': 'tracks{}'.format(n), 'file':
			name, 'seconds': seconds, 'bytes': os.path.getsize(path)})
		print('built', path, '{:.1f} s'.format(seconds), file=sys.stderr,
			flush=True)

	path = os.path.join(args.workdir, 'tracks{}_trackorder.fw'.format(n))
	files['trackorder'] = path
	if not os.path.exists(path):
		seconds = shell([sys.executable, '-m', 'figwig', 'reorder', files[
			'onegroup_raw'], path, '--order', 'tracks', '-p', '8'], pinned_env(8))
		records.append({'kind': 'build', 'set': 'tracks{}'.format(n), 'file':
			'trackorder', 'seconds': seconds, 'bytes': os.path.getsize(path),
			'note': 'reorder from onegroup_raw'})
	return bigwigs, files


# bw2fw through the Python API, so that the old layout can be forced.
_BW2FW = """
import sys
from figwig.bw2fw import main
main(sys.argv[1:])
"""
_BW2FW_OLD = """
import sys
import figwig.figwig
figwig.figwig._MATRIX_CHUNKS = False
from figwig.bw2fw import main
main(sys.argv[1:])
"""

_CELLS = """
import sys, time, json, os
import numpy
import figwig.figwig
from figwig import FigWigWriter
cuts, path, variant, threads = sys.argv[1:5]
threads = int(threads)
z = numpy.load(cuts)
tid, pos, cell, count = z['tid'], z['pos'], z['cell'], z['count']
chroms, lengths, barcodes = list(z['chroms']), z['lengths'], list(z['barcodes'])
tracks = [{'name': str(b), 'dtype': 'uint8'} for b in barcodes]
if variant == 'groups64_old':
	figwig.figwig._MATRIX_CHUNKS = False
	kwargs = dict(groups=64, level=1)
elif variant == 'onegroup_zlib':
	kwargs = dict(groups=len(tracks), level=1)
elif variant == 'onegroup_raw':
	kwargs = dict(groups=len(tracks), level=0)
else:
	kwargs = dict(order='tracks')
t0 = time.perf_counter()
with FigWigWriter(path, list(zip(chroms, lengths.tolist())), tracks, genome='hg38',
		n_jobs=threads, **kwargs) as writer:
	bounds = numpy.concatenate([[0], numpy.flatnonzero(tid[1:] != tid[:-1]) + 1, [len(tid)]])
	for a, b in zip(bounds[:-1], bounds[1:]):
		writer.write_sparse(chroms[int(tid[a])], pos[a:b], cell[a:b], count[a:b])
print(json.dumps({'seconds': time.perf_counter() - t0, 'bytes': os.path.getsize(path)}))
"""


def build_cell_files(args, records):
	files = {}
	for variant in ('groups64_old', 'onegroup_zlib', 'onegroup_raw',
			'trackorder'):
		path = os.path.join(args.workdir, 'cells_{}.fw'.format(variant))
		files[variant] = path
		if os.path.exists(path):
			continue
		result = subprocess.run([sys.executable, '-c', _CELLS, args.cuts, path,
			variant, '8'], env=pinned_env(8), capture_output=True, text=True)
		if result.returncode != 0:
			raise RuntimeError(result.stderr[-3000:])
		entry = json.loads(result.stdout.strip().splitlines()[-1])
		records.append({'kind': 'build', 'set': 'cells', 'file': variant,
			'seconds': entry['seconds'], 'bytes': entry['bytes']})
		print('built', path, '{:.1f} s'.format(entry['seconds']), file=sys.stderr,
			flush=True)
	return files


def time_one(args):
	"""One read configuration in its own process; prints JSON."""

	import figwig

	names, starts = load_windows(args.bed, args.width, args.windows)
	rng = numpy.random.default_rng(0)
	reader = None
	if args.mode.startswith('bigwig'):
		paths = [track_path(args.workdir, i) for i in range(args.n)]

		def open_():
			return [figwig.BigWigReader(path) for path in paths]
	else:
		def open_():
			return figwig.FigWigReader(args.path)
		reader = open_()

	n_tracks = args.n if reader is None else len(reader.tracks)
	selection = None
	if args.selection == 'subset500':
		selection = numpy.sort(rng.choice(n_tracks, 500, replace=False))
	if args.labels == 'strand':
		labels = numpy.array([i % 2 for i in range(n_tracks)])
	elif args.labels == 'six':
		labels = numpy.arange(n_tracks) % 6
	elif args.labels == 'ten':
		labels = rng.integers(0, 10, n_tracks)
	else:
		labels = None
	if labels is not None and selection is not None:
		labels = labels[selection]

	def read(handle, chroms, starts_):
		if args.mode == 'bigwig':
			files = handle if selection is None else [handle[i] for i in selection]
			y = figwig.read_bigwig(files, chroms, starts_, args.width,
				n_jobs=args.threads)
			if labels is not None:
				y = numpy.stack([y[:, labels == k].sum(axis=1) for k in range(
					labels.max() + 1)], axis=1)
			return y
		if args.mode == 'entries':
			return handle.read_entries(tracks=selection[:20] if selection is not
				None else list(range(20)), n_jobs=args.threads)
		kwargs = {}
		if args.mode == 'sparse':
			kwargs['sparse'] = True
		if labels is not None:
			kwargs['labels'] = labels
		return handle.read(chroms, starts_, args.width, tracks=selection,
			n_jobs=args.threads, **kwargs)

	handle = open_()
	if args.mode != 'entries':
		read(handle, names[:1000], starts[:1000])
	del handle

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
		if args.mode == 'entries':
			digest = hashlib.sha256(numpy.ascontiguousarray(y.values.astype(
				numpy.float32)).tobytes()).hexdigest()[:12]
		else:
			if args.mode == 'sparse':
				dense = numpy.zeros(y.shape, dtype=numpy.float32)
				dense[y.rows, y.channels, y.offsets] = y.values
				y = dense
			digest = hashlib.sha256(numpy.ascontiguousarray(numpy.nan_to_num(
				y.astype(numpy.float32))).tobytes()).hexdigest()[:12]

	print(json.dumps({'kind': 'read', 'set': args.set, 'file': args.file, 'mode':
		args.mode, 'selection': args.selection, 'labels': args.labels, 'threads':
		args.threads, 'batches': args.batches, 'seconds': seconds, 'digest':
		digest}))


def configurations(args, sets):
	"""Every (set, file, path, n, mode, selection, labels, threads, batches)."""

	out = []
	for set_name, n, bigwigs, files, label_name in sets:
		cells = set_name == 'cells'
		selections = ['all', 'subset500'] if cells else ['all']
		kinds = []
		if bigwigs:
			kinds.append(('bigwigs', None, 'bigwig'))
		for name, path in files.items():
			kinds.append((name, path, 'dense'))
			kinds.append((name, path, 'sparse'))
		for name, path, mode in kinds:
			for selection in selections:
				if cells and mode == 'dense' and selection == 'all':
					continue
				for threads in args.threads_list:
					out.append((set_name, name, path, n, mode, selection, 'none',
						threads, False))
				if mode != 'sparse':
					for threads in args.threads_list:
						out.append((set_name, name, path, n, mode, selection,
							label_name, threads, False))
				if threads_has_one(args) and selection == 'all' and mode != 'sparse' \
						and not cells:
					out.append((set_name, name, path, n, mode, selection, 'none', 1,
						True))
				if threads_has_one(args) and mode != 'sparse':
					out.append((set_name, name, path, n, mode, selection, label_name,
						1, True))
		if cells:
			for name, path in files.items():
				out.append((set_name, name, path, n, 'entries', 'subset500', 'none',
					1, False))
	return out


def threads_has_one(args):
	return 1 in args.threads_list


def run(args):
	records = []
	out = open(args.out, 'a')

	def record(entry):
		records.append(entry)
		out.write(json.dumps(entry) + '\n')
		out.flush()

	sets = []
	for n, label_name in ((len(args.bigwig), 'strand'), (96, 'six')):
		if n == 0 or (n == 96 and not args.many):
			continue
		bigwigs, files = build_track_files(args, n, records)
		sets.append(('tracks{}'.format(n), n, bigwigs, files, label_name))
	if args.cuts is not None:
		files = build_cell_files(args, records)
		sets.append(('cells', 0, [], files, 'ten'))
	for entry in records:
		out.write(json.dumps(entry) + '\n')
	out.flush()

	configs = configurations(args, sets)
	print('{} read configurations'.format(len(configs)), file=sys.stderr,
		flush=True)
	for repeat in range(args.repeats):
		order = configs[repeat:] + configs[:repeat]
		for set_name, name, path, n, mode, selection, label_name, threads, \
				batches in order:
			command = [sys.executable, os.path.abspath(__file__), '--one',
				'--workdir', args.workdir, '--set', set_name, '--file', name,
				'--path', path or '', '--n', str(n), '--mode', mode, '--selection',
				selection, '--labels', label_name, '--threads', str(threads),
				'--width', str(args.width), '--windows', str(args.windows)]
			command += ['--batches'] if batches else []
			for bed in args.bed:
				command += ['--bed', bed]
			result = subprocess.run(command, env=pinned_env(threads),
				capture_output=True, text=True)
			if result.returncode != 0:
				print(result.stderr[-1500:], file=sys.stderr, flush=True)
				continue
			record(json.loads(result.stdout.strip().splitlines()[-1]))
		print('repetition {} done'.format(repeat + 1), file=sys.stderr,
			flush=True)

	out.close()
	print(table(records))


def table(records):
	lines = []
	builds = [r for r in records if r['kind'] == 'build']
	if builds:
		lines.append('files: build seconds, MB')
		for r in builds:
			lines.append('  {:10s} {:16s} {:8.1f} s {:10.1f} MB {}'.format(r['set'],
				r['file'], r['seconds'], r['bytes'] / 1e6, r.get('note', '')))
		lines.append('')

	reads = [r for r in records if r['kind'] == 'read']
	by_key = {}
	for r in reads:
		by_key.setdefault((r['set'], r['file'], r['mode'], r['selection'],
			r['labels'], r['threads'], r['batches']), []).append(r['seconds'])
	keys = sorted(by_key, key=lambda k: (k[0], k[3], k[4], k[6], k[2], k[1], k[5]))
	lines.append('reads: median seconds ({} runs each)'.format(max(len(v) for v
		in by_key.values()) if by_key else 0))
	lines.append('  {:9s} {:16s} {:7s} {:9s} {:6s} {:7s} {:7s} {:>10s}'.format(
		'set', 'file', 'mode', 'selection', 'labels', 'threads', 'batches',
		'seconds'))
	for k in keys:
		lines.append('  {:9s} {:16s} {:7s} {:9s} {:6s} {:7d} {:7s} {:10.4f}'.format(
			k[0], k[1], k[2], k[3], k[4], k[5], 'yes' if k[6] else 'no',
			statistics.median(by_key[k])))

	digests = {}
	for r in reads:
		if r['digest'] is not None and r['mode'] != 'entries':
			digests.setdefault((r['set'], r['selection'], r['labels']), set()).add(
				r['digest'])
	lines.append('')
	lines.append('distinct outputs per (set, selection, labels): {}'.format(
		{str(k): len(v) for k, v in digests.items()}))
	return '\n'.join(lines)


def main():
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('--bigwig', action='append', default=[])
	parser.add_argument('--bed', action='append', default=[])
	parser.add_argument('--cuts', default=None)
	parser.add_argument('--workdir', required=True)
	parser.add_argument('--out', default='results.jsonl')
	parser.add_argument('--width', type=int, default=2114)
	parser.add_argument('--windows', type=int, default=8192)
	parser.add_argument('--threads-list', type=int, nargs='+', default=[1, 8])
	parser.add_argument('--prepare-threads', type=int, default=16)
	parser.add_argument('--repeats', type=int, default=3)
	parser.add_argument('--many', action='store_true', help="Include the "
		"96-track set.")
	parser.add_argument('--table', default=None)
	parser.add_argument('--one', action='store_true', help=argparse.SUPPRESS)
	parser.add_argument('--set', help=argparse.SUPPRESS)
	parser.add_argument('--file', help=argparse.SUPPRESS)
	parser.add_argument('--path', help=argparse.SUPPRESS)
	parser.add_argument('--n', type=int, help=argparse.SUPPRESS)
	parser.add_argument('--mode', help=argparse.SUPPRESS)
	parser.add_argument('--selection', help=argparse.SUPPRESS)
	parser.add_argument('--labels', help=argparse.SUPPRESS)
	parser.add_argument('--threads', type=int, help=argparse.SUPPRESS)
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
