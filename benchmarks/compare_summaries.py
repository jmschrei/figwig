"""Compare ordinary and indexed figwig summaries on identical tracks.

	python benchmarks/compare_summaries.py --workdir benchmarks/results/summaries
	python benchmarks/compare_summaries.py --bigwig counts.bw --dtype int32 \
		--chroms chr21 chr22 --workdir benchmarks/results/count-summaries
	git show 1052310:figwig/figwig.py > benchmarks/results/baseline-1052310.py
	python benchmarks/compare_summaries.py --baseline-reader \
		benchmarks/results/baseline-1052310.py --baseline-revision 1052310

The default is a deterministic sparse synthetic genome, 8 and 48 tracks, both
orders, and int32 counts and float32 signals. Real bigWigs are streamed in bounded
batches; tracks beyond the given sources are cyclic shifts of those sources.
int32 requires nonnegative integral counts in range, and never rounds a signal.

Every writer/file-reader-mode/repetition runs in a fresh subprocess, with imports
and kernel warmup outside timing and configurations rotated between repetitions.
Each read workload opens a fresh reader; navigation keeps one across its steps.
Aligned interval sums isolate the summary fast path; unaligned sums include raw
edge tiles (which can require large sparse blocks in track order).
The OS page cache is warmed by a linear file pre-read, not flushed: reader-cold
does NOT mean disk-cold. Writer timing includes open/write/close calls; pipeline
timing also includes streaming the source and hashing the exact writer inputs.
Peak RSS is the file/mode worker's cumulative highwater, including imports,
inputs, ALL-workload warmup and preceding timed/untimed passes, rather than a
per-workload allocation measurement. For isolated workload memory comparisons,
run e.g. --workloads chrom_base --selections all in a separate invocation.

Untimed second passes count Python preads, logical native raw-chunk runs, and
inflated bytes. These are library-level requests, not physical disk I/O. The
optional index-ignored control masks the bit on the SAME indexed file, measuring
ordinary decoding of its restarted streams. Exact output hashes must agree for
every off/on/ignored pair. Raw JSONL, medians/ranges, inputs, file hashes, runtime
source snapshots, dirty patch and environment all live under benchmarks/results/.
"""

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zlib

import numpy


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKLOADS = ['chrom_base', 'chrom_zoom', 'sums_local', 'sums_distant',
	'sums_local_aligned', 'sums_distant_aligned',
	'whole_base', 'whole_zoom', 'raw_windows', 'navigation']


def sha_file(path):
	hash_ = hashlib.sha256()
	with open(path, 'rb') as handle:
		for part in iter(lambda: handle.read(1 << 20), b''):
			hash_.update(part)
	return hash_.hexdigest()


def json_hash(value):
	return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def runtime_hashes():
	return {str(path.relative_to(ROOT)): sha_file(path)
		for path in sorted((ROOT / 'figwig').rglob('*.py'))}


def output_hash(arrays):
	hash_ = hashlib.sha256()
	for array in arrays:
		hash_.update(json.dumps([array.dtype.str, array.shape]).encode())
		hash_.update(numpy.ascontiguousarray(array).view(numpy.uint8))
	return hash_.hexdigest()


def peak_mb():
	"""Own VmHWM on Linux; ru_maxrss bytes on macOS, KiB elsewhere."""
	try:
		with open('/proc/self/status') as handle:
			for line in handle:
				if line.startswith('VmHWM:'):
					return int(line.split()[1]) / 1024
	except OSError:
		pass
	rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	return rss / (2**20 if sys.platform == 'darwin' else 1024)


def git(*args):
	return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True)


def provenance(workdir, args):
	files = runtime_hashes()
	for name in ['pyproject.toml', 'docs/summary_restarts.rst',
			'benchmarks/compare_summaries.py']:
		path = ROOT / name
		if path.exists():
			files[name] = sha_file(path)
	for name in files:
		path = workdir / 'source' / name
		path.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(ROOT / name, path)
	patch = git('diff', '--binary', 'HEAD')
	(workdir / 'source' / 'tracked.patch').write_text(patch)
	versions = {}
	for name in ['figwig', 'numpy', 'numba', 'llvmlite', 'deflate', 'isal']:
		try:
			versions[name] = importlib.metadata.version(name)
		except importlib.metadata.PackageNotFoundError:
			versions[name] = None
	cpu = platform.processor()
	if sys.platform == 'darwin':
		try:
			cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'],
				text=True, stderr=subprocess.DEVNULL).strip()
		except (OSError, subprocess.CalledProcessError):
			pass
	elif Path('/proc/cpuinfo').exists():
		cpu = next((line.split(':', 1)[1].strip() for line in
			Path('/proc/cpuinfo').read_text().splitlines() if line.startswith(
			'model name')), cpu)
	return {'revision': git('rev-parse', 'HEAD').strip(),
		'branch': git('branch', '--show-current').strip(),
		'status': git('status', '--porcelain'), 'source_sha256': files,
		'runtime_fingerprint': json_hash(runtime_hashes()),
		'tracked_patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
		'command': [sys.executable, *sys.argv],
		'python': sys.version, 'executable': sys.executable,
		'platform': platform.platform(), 'machine': platform.machine(),
		'cpu': cpu, 'cpu_count': os.cpu_count(), 'versions': versions,
		'zlib_build': zlib.ZLIB_VERSION, 'zlib_runtime': zlib.ZLIB_RUNTIME_VERSION,
		'threads': args.threads, 'engine': args.engine,
		'environment': {key: value for key, value in os.environ.items() if
			key.endswith('_NUM_THREADS') or key == 'NUMBA_CACHE_DIR'},
		'cache': 'Full-file linear pre-read before warmup; fresh reader per '
			'workload, one retained for navigation; OS page cache not evicted.',
		'memory': 'File/mode worker cumulative own peak RSS, including imports, '
			'inputs, ALL-workload warmup and preceding timed/untimed passes. '
			'Not per-workload RSS; isolate with e.g. --workloads chrom_base '
			'--selections all. VmHWM on Linux, ru_maxrss on macOS.',
		'counters': 'Separate untimed pass: _pread + contiguous _load_batch '
			'raw runs; _inflate_exact + _summary_index._inflate + non-stored '
			'_load_batch decoded lengths. Logical requests, not physical I/O.'}


def reader_module(config):
	if config['mode'].startswith('pre-extension-reader-'):
		name = 'figwig._summary_baseline'
		if name not in sys.modules:
			importlib.import_module('figwig')
			if sha_file(config['baseline']['path']) != config['baseline']['sha256']:
				raise RuntimeError('Baseline reader changed since manifest capture')
			spec = importlib.util.spec_from_file_location(name, config['baseline']['path'])
			module = importlib.util.module_from_spec(spec)
			sys.modules[name] = module
			spec.loader.exec_module(module)
		return sys.modules[name]
	import figwig.figwig as module
	return module


def dataset(args):
	if args.bigwig:
		from figwig import BigWigReader
		inputs = [{'path': str(Path(name).resolve()),
			'bytes': Path(name).stat().st_size, 'sha256': sha_file(name)}
			for name in args.bigwig]
		readers = [BigWigReader(item['path']) for item in inputs]
		names = args.chroms or list(readers[0].chrom_sizes)
		chroms = {}
		for name in names:
			lengths = [reader.chrom_sizes[name] for reader in readers]
			if len(set(lengths)) != 1:
				raise ValueError('Source chromosome lengths differ for ' + name)
			chroms[name] = min(lengths[0], args.chrom_bases or lengths[0])
		return {'kind': 'bigwig', 'inputs': inputs, 'chroms': chroms,
			'seed': args.seed, 'batch_bases': args.batch_bases,
			'transform': 'track i uses source i % k; first k tracks unchanged; '
				'later tracks cyclically shifted within the full source '
				'chromosome by (i//k*2654435761 + chrom_index*104729 + seed) % '
				'length. Output clipped to --chrom-bases when given. Uncovered '
				'bases use BigWigReader zero semantics.'}
	return {'kind': 'synthetic', 'version': 1, 'seed': args.seed,
		'chroms': {'chrLong': args.tiles * args.tile,
			'chrPanel': max(32, args.tiles // 32) * args.tile - 13,
			'chrFar': max(64, args.tiles // 2) * args.tile - 7},
		'batch_tiles': args.batch_tiles,
		'recipe': 'One nonzero base per track per tile, at tile//3 (clipped '
			'at chromosome end). Omit tile blocks where (tile//1024)%11==7. '
			'PCG64(seed+chrom_index); int32 uniform integers [1,128), or '
			'float32 standard_normal, generated row-major by batch.'}


def batches(config, small=False):
	"""Yield bounded exact writer arguments, reproducible for either layout."""
	source, n, dtype = config['source'], config['tracks'], config['dtype']
	tile = config['tile']
	if source['kind'] == 'synthetic':
		for ci, (name, length) in enumerate(source['chroms'].items()):
			rng = numpy.random.default_rng(source['seed'] + ci)
			count = -(-length // tile)
			if small:
				count = min(count, 128)
			for start in range(0, count, source['batch_tiles']):
				ids = numpy.arange(start, min(start + source['batch_tiles'], count))
				shape = (len(ids), n)
				values = rng.integers(1, 128, size=shape, dtype=numpy.int32) if \
					dtype == 'int32' else rng.standard_normal(shape, dtype=numpy.float32)
				keep = (ids // 1024) % 11 != 7
				positions = numpy.minimum(ids[keep] * tile + tile // 3, length - 1)
				yield 'sparse', name, numpy.repeat(positions, n), \
					numpy.tile(numpy.arange(n, dtype=numpy.int64), len(positions)), \
					values[keep].ravel()
		return
	from figwig import BigWigReader
	readers = [BigWigReader(item['path']) for item in source['inputs']]
	try:
		for ci, (name, length) in enumerate(source['chroms'].items()):
			if small:
				length = min(length, tile * 128)
			for start in range(0, length, source['batch_bases']):
				width = min(source['batch_bases'], length - start)
				values = numpy.empty((1, n, width), dtype=dtype)
				for track in range(n):
					reader = readers[track % len(readers)]
					span = reader.chrom_sizes[name]
					shift = 0 if track < len(readers) else (track // len(readers) *
						2654435761 + ci * 104729 + source['seed']) % span
					at = (start + shift) % span
					first = min(width, span - at)
					given = reader.read(name, [at], first, n_jobs=config['threads']).ravel()
					if first < width:
						given = numpy.concatenate([given, reader.read(name, [0],
							width - first, n_jobs=config['threads']).ravel()])
					if dtype == 'int32':
						checked = given.astype(numpy.float64)
						if not numpy.isfinite(checked).all() or (checked < 0).any() or \
							(checked > numpy.iinfo(numpy.int32).max).any() or \
							(checked != numpy.floor(checked)).any():
							raise ValueError('--dtype int32 requires exact nonnegative '
								'int32 counts; use --dtype float32 for signals.')
					values[0, track] = given
				yield 'dense', name, numpy.array([start]), None, values
	finally:
		# BigWigReader retains no file descriptor between operations.
		readers.clear()


def write_file(config, path, small=False):
	from figwig import FigWigWriter
	chroms = config['source']['chroms']
	if small:
		chroms = {name: min(length, config['tile'] * 128)
			for name, length in chroms.items()}
	inputs = hashlib.sha256(json.dumps(chroms).encode())
	start = time.perf_counter()
	writer = FigWigWriter(path, chroms, [{'name': 'signal' + str(i), 'dtype':
		config['dtype']} for i in range(config['tracks'])], tile=config['tile'],
		groups=8, order=config['order'], level=6, engine=config['engine'],
		n_jobs=config['threads'], zooms=2, summary_index=config['indexed'])
	opened = time.perf_counter() - start
	write_seconds = 0.0
	try:
		for kind, chrom, positions, tracks, values in batches(config, small):
			inputs.update(chrom.encode())
			for array in [positions, tracks, values]:
				if array is not None:
					inputs.update(json.dumps([array.dtype.str, array.shape]).encode())
					inputs.update(array.view(numpy.uint8))
			at = time.perf_counter()
			if kind == 'sparse':
				writer.write_sparse(chrom, positions, tracks, values)
			else:
				writer.write(chrom, positions, values)
			write_seconds += time.perf_counter() - at
	finally:
		at = time.perf_counter()
		writer.close()
		closed = time.perf_counter() - at
	return {'seconds': opened + write_seconds + closed,
		'open_seconds': opened, 'write_seconds': write_seconds,
		'close_seconds': closed, 'pipeline_seconds': time.perf_counter() - start,
		'dataset_sha256': inputs.hexdigest(), 'bytes': Path(path).stat().st_size,
		'file_sha256': sha_file(path), 'peak_mb': peak_mb()}


def query_calls(config, selection, workload):
	"""Calls take a reader, keeping all query construction outside timing."""
	chroms = config['source']['chroms']
	names = list(chroms)
	chosen = min(names, key=chroms.get)
	other = max(names, key=chroms.get)
	tile = config['tile']
	tracks = [0] if selection == 'one' else None
	jobs = config['threads']
	def profile(name, level):
		return lambda reader: reader.tile_sums(tracks=tracks, chroms=name, level=level)
	def intervals(distant=False, pan=0, aligned=False):
		used = names if distant else [chosen]
		interval_names, starts, ends = [], [], []
		for name in used:
			length = chroms[name]
			width = min(24 * tile + 17, max(tile + 2, length // 4))
			limit = max(0, length - width - 1)
			positions = numpy.linspace(0, limit, 8 if distant else 4,
				dtype=numpy.int64) if distant else numpy.full(4, limit // 3)
			positions = numpy.minimum(limit, positions + 1 + pan * tile +
				numpy.arange(len(positions)) * (3 if distant else tile))
			if aligned:
				positions = positions // tile * tile
				width = max(tile, width // tile * tile)
			interval_names.extend([name] * len(positions))
			starts.extend(positions.tolist())
			stops = numpy.minimum(positions + width, length)
			ends.extend((stops // tile * tile if aligned else stops).tolist())
		return lambda reader: reader.sums(interval_names, starts, ends,
			tracks=tracks, n_jobs=jobs)
	if workload.startswith('chrom_'):
		return [(workload, profile(chosen, int(workload == 'chrom_zoom')))]
	if workload.startswith('whole_'):
		return [(workload, profile(None, int(workload == 'whole_zoom')))]
	if workload.startswith('sums_'):
		return [(workload, intervals('distant' in workload,
			aligned=workload.endswith('_aligned')))]
	if workload == 'raw_windows':
		width = min(1024, min(chroms.values()))
		locations = [(name, int(position)) for name, length in chroms.items()
			for position in numpy.linspace(0, max(0, length - width), 16,
				dtype=numpy.int64)]
		return [(workload, lambda reader: reader.read([c for c, _ in locations],
			[p for _, p in locations], width, tracks=tracks, n_jobs=jobs))]
	return [('chrom_base', profile(chosen, 0)), ('same_chrom', profile(chosen, 0)),
		('new_chrom', profile(other, 0)), ('zoom_change', profile(chosen, 1)),
		('local_sums', intervals()), ('same_intervals', intervals()),
		('pan_intervals', intervals(pan=1024)), ('distant_intervals', intervals(True))]


def close_reader(reader):
	# FigWigReader owns no persistent fd, but a raw read may create this pool.
	if reader._pool is not None:
		reader._pool[2].shutdown(wait=True)


@contextlib.contextmanager
def counters(fw):
	"""Instrumentation stays out of latency runs, including native batch I/O."""
	import figwig._summary_index as index
	counts, lock, originals = {}, threading.Lock(), []
	def add(**values):
		with lock:
			for key, value in values.items():
				counts[key] = counts.get(key, 0) + int(value)
	def patch(module, name, function):
		originals.append((module, name, getattr(module, name)))
		setattr(module, name, function)
	pread, exact, inflate, load = fw._pread, fw._inflate_exact, index._inflate, fw._load_batch
	def read(fd, size, offset):
		result = pread(fd, size, offset)
		add(pread_calls=1, pread_bytes=len(result), read_calls=1, read_bytes=len(result))
		return result
	def legacy(*args, **kwargs):
		result = exact(*args, **kwargs)
		add(legacy_inflate_calls=1, legacy_inflated_bytes=len(result),
			inflated_bytes=len(result))
		return result
	def restart(*args, **kwargs):
		result = inflate(*args, **kwargs)
		add(extension_inflate_calls=1, extension_inflated_bytes=len(result),
			inflated_bytes=len(result))
		return result
	def batch(reader, fd, offsets, sizes, kinds, bound, crcs=None):
		order = numpy.argsort(offsets, kind='stable')
		o, s = offsets[order], sizes[order]
		calls = 0 if not len(o) else 1 + int((o[1:] != o[:-1] + s[:-1]).sum())
		result = load(reader, fd, offsets, sizes, kinds, bound, crcs)
		stored = (kinds & fw._STORED) > 0
		decoded = int(result[2][~stored].sum()) if reader._codec else 0
		add(batch_read_calls=calls, batch_read_bytes=s.sum(), read_calls=calls,
			read_bytes=s.sum(), raw_chunk_inflate_calls=(~stored).sum() if
			reader._codec else 0, raw_chunk_inflated_bytes=decoded, inflated_bytes=decoded)
		return result
	patch(fw, '_pread', read)
	patch(fw, '_inflate_exact', legacy)
	patch(index, '_inflate', restart)
	patch(fw, '_load_batch', batch)
	try:
		yield counts
	finally:
		for module, name, original in reversed(originals):
			setattr(module, name, original)


def read_once(config, calls, instrument=False):
	module = reader_module(config)
	with counters(module) if instrument else contextlib.nullcontext({}) as counts:
		start = time.perf_counter()
		reader = module.FigWigReader(config['path'])
		if config['mode'] == 'indexed-file-index-ignored':
			reader._optional &= ~2
		opened = time.perf_counter() - start
		arrays, steps = [], []
		try:
			for name, call in calls:
				before = dict(counts)
				at = time.perf_counter()
				arrays.append(call(reader))
				step = {'name': name, 'seconds': time.perf_counter() - at}
				if instrument:
					step = {'name': name, 'counters': {key: value - before.get(key, 0)
						for key, value in counts.items()}}
				steps.append(step)
			seconds = time.perf_counter() - start
		finally:
			close_reader(reader)
		result = {'sha256': output_hash(arrays), 'peak_mb': peak_mb(),
			'steps': steps, 'output_bytes': sum(a.nbytes for a in arrays)}
		for step, array in zip(steps, arrays):
			step.update(sha256=output_hash([array]), shape=list(array.shape),
				dtype=array.dtype.str)
		if instrument:
			result['counters'] = dict(counts)
		else:
			result.update(seconds=seconds, open_seconds=opened)
		return result


def worker(config):
	fingerprint = json_hash(runtime_hashes())
	if config['kind'] == 'write':
		warm = str(Path(config['path']).with_suffix('.warm.fw'))
		try:
			write_file(config, warm, small=True)
		finally:
			Path(warm).unlink(missing_ok=True)
		result = write_file(config, config['path'])
	else:
		# Imports, JIT and OS cache population are deliberately outside timing.
		with open(config['path'], 'rb') as handle:
			while handle.read(1 << 20):
				pass
		queries = [(selection, name, query_calls(config, selection, name)) for
			selection, name in itertools.product(config['selections'], config['workloads'])]
		for _, _, calls in queries:
			read_once(config, calls)
		measurements = []
		for selection, name, calls in queries:
			row = read_once(config, calls)
			counted = read_once(config, calls, instrument=True)
			if row['sha256'] != counted['sha256']:
				raise ValueError('Untimed instrumentation changed the output')
			row.update(selection=selection, workload=name,
				counters=counted['counters'], counter_steps=counted['steps'])
			measurements.append(row)
		result = {'measurements': measurements, 'peak_mb': peak_mb()}
	if fingerprint != json_hash(runtime_hashes()):
		raise RuntimeError('Runtime sources changed during the worker')
	print(json.dumps(dict(result, runtime_fingerprint=fingerprint)), flush=True)


def subprocess_one(config, args):
	env = dict(os.environ)
	for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
			'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS']:
		env[name] = str(args.threads)
	result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
		'--worker', json.dumps(config)], env=env, capture_output=True, text=True)
	if result.returncode:
		raise RuntimeError(str(config) + '\n' + result.stderr[-6000:])
	return json.loads(result.stdout.strip().splitlines()[-1])


def aggregate(records):
	groups = {}
	for record in records:
		key = tuple(record[name] for name in ['kind', 'tracks', 'dtype', 'order',
			'mode', 'selection', 'workload'])
		groups.setdefault(key, []).append(record)
	rows = []
	for key, values in groups.items():
		row = dict(zip(['kind', 'tracks', 'dtype', 'order', 'mode', 'selection',
			'workload'], key))
		row['samples'] = len(values)
		for name in ['seconds', 'pipeline_seconds', 'open_seconds', 'peak_mb',
				'bytes', 'output_bytes']:
			if name in values[0]:
				x = [value[name] for value in values]
				row[name] = {'median': statistics.median(x), 'min': min(x), 'max': max(x)}
		if 'counters' in values[0]:
			row['counters'] = {name: {'median': statistics.median([v['counters'].get(
				name, 0) for v in values]), 'min': min(v['counters'].get(name, 0)
				for v in values), 'max': max(v['counters'].get(name, 0) for v in values)}
				for name in sorted(set().union(*(v['counters'] for v in values)))}
		row['output_hashes'] = sorted({v.get('sha256', v.get('dataset_sha256'))
			for v in values})
		rows.append(row)
	return rows


def run(args):
	workdir = Path(args.workdir).resolve()
	results = (ROOT / 'benchmarks' / 'results').resolve()
	if not workdir.is_relative_to(results):
		raise ValueError('--workdir must be under ignored benchmarks/results/')
	if workdir.exists() and any(workdir.iterdir()):
		raise ValueError('--workdir must be empty; use a new directory for each run')
	workdir.mkdir(parents=True, exist_ok=True)
	(workdir / 'files').mkdir()
	source = dataset(args)
	meta = provenance(workdir, args)
	baseline = None
	if args.baseline_reader:
		path = Path(args.baseline_reader).resolve()
		baseline = {'path': str(path), 'sha256': sha_file(path),
			'revision': args.baseline_revision,
			'helpers': 'Pre-extension figwig.py imported as figwig._summary_baseline; '
				'relative imports use the snapshotted current kernels/helpers.'}
		if args.baseline_revision:
			raw = subprocess.check_output(['git', '-C', str(ROOT), 'show',
				args.baseline_revision + ':figwig/figwig.py'])
			if hashlib.sha256(raw).hexdigest() != baseline['sha256']:
				raise ValueError('--baseline-reader differs from --baseline-revision')
		shutil.copyfile(path, workdir / 'source' / 'baseline-figwig.py')
	manifest = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
		'provenance': meta, 'baseline_reader': baseline,
		'source': source, 'arguments': vars(args), 'files': []}
	manifest_path = workdir / 'manifest.json'
	manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
	configs = []
	for n, dtype, order, indexed in itertools.product(args.tracks, args.dtype,
			args.orders, [False, True]):
		mode = 'indexed' if indexed else 'ordinary'
		configs.append({'kind': 'write', 'tracks': n, 'dtype': dtype, 'order': order,
			'indexed': indexed, 'mode': mode, 'tile': args.tile, 'source': source,
			'threads': args.threads, 'engine': args.engine,
			'path': str(workdir / 'files' / f'{n}-{dtype}-{order}-{mode}.fw')})
	records, expected = [], {}
	def save(config, result):
		if result['runtime_fingerprint'] != meta['runtime_fingerprint']:
			raise RuntimeError('Runtime source changed since manifest capture; rerun')
		common = {key: config[key] for key in ['kind', 'tracks', 'dtype', 'order',
			'mode', 'rep', 'path']}
		common['runtime_fingerprint'] = result['runtime_fingerprint']
		for row in result.get('measurements', [result]):
			record = {**common, 'selection': None, 'workload': None, **row}
			key = tuple(record[k] for k in ['kind', 'tracks', 'dtype', 'selection',
				'workload'])
			# Different layouts also hold exactly the same numeric dataset.
			hash_ = record.get('sha256', record.get('dataset_sha256'))
			if expected.setdefault(key, hash_) != hash_:
				raise ValueError('Output mismatch across configurations: ' + str(key))
			records.append(record)
			with open(workdir / 'samples.jsonl', 'a') as handle:
				handle.write(json.dumps(record) + '\n')
	for kind in ['write', 'read']:
		if kind == 'read':
			read_configs = []
			for config in configs:
				item = dict(config, kind='read', selections=args.selections,
					workloads=args.workloads)
				read_configs.append(item)
				if config['indexed'] and not args.no_index_ignored:
					read_configs.append(dict(item, mode='indexed-file-index-ignored'))
				if baseline is not None:
					read_configs.append(dict(item, baseline=baseline,
						mode='pre-extension-reader-' + config['mode'] + '-file'))
			configs = read_configs
		for rep in range(args.repeats):
			shift = rep * 7 % len(configs)
			ordered = configs[shift:] + configs[:shift]
			if rep % 2:
				ordered = list(reversed(ordered))
			for config in ordered:
				config = dict(config, rep=rep)
				result = subprocess_one(config, args)
				save(config, result)
				print(kind, rep + 1, config['tracks'], config['dtype'], config['order'],
					config['mode'], f"{result.get('seconds', 0):.3f}s" if kind == 'write'
					else f"{len(result['measurements'])} workloads", flush=True)
		if kind == 'write':
			for config in configs:
				path = config['path']
				manifest['files'].append(dict(config, bytes=Path(path).stat().st_size,
					sha256=sha_file(path)))
			manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
	summary = {'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
		'manifest_sha256': sha_file(manifest_path),
		'samples_sha256': sha_file(workdir / 'samples.jsonl'), 'rows': aggregate(records)}
	(workdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
	print('\n| kind | tracks | dtype | order | mode | selection | workload | '
		'median ms (min–max) | file bytes | logical bytes | inflated bytes | peak MiB |')
	print('|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|')
	for row in summary['rows']:
		t = row['seconds']
		c = row.get('counters', {})
		print('| {} | {} | {} | {} | {} | {} | {} | {:.3f} ({:.3f}–{:.3f}) | {} | {} | '
			'{} | {:.1f} |'.format(row['kind'], row['tracks'], row['dtype'],
			row['order'], row['mode'], row['selection'] or '', row['workload'] or '',
			t['median'] * 1000, t['min'] * 1000, t['max'] * 1000,
			row.get('bytes', {}).get('median', ''),
			c.get('read_bytes', {}).get('median', ''),
			c.get('inflated_bytes', {}).get('median', ''), row['peak_mb']['median']))
	print('\nEvidence:', workdir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
	parser.add_argument('--workdir', default='benchmarks/results/summaries-' +
		datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
	parser.add_argument('--bigwig', action='append')
	parser.add_argument('--dtype', nargs='+', choices=['int32', 'float32'])
	parser.add_argument('--chroms', nargs='+')
	parser.add_argument('--chrom-bases', type=int, default=0,
		help='Real-source prefix length per chromosome; 0 keeps full chromosomes')
	parser.add_argument('--tracks', type=int, nargs='+', default=[8, 48])
	parser.add_argument('--orders', nargs='+', choices=['positions', 'tracks'],
		default=['positions', 'tracks'])
	parser.add_argument('--tiles', type=int, default=65536,
		help='Tiles of the largest synthetic chromosome; others are 1/32 and 1/2')
	parser.add_argument('--tile', type=int, default=16384)
	parser.add_argument('--batch-tiles', type=int, default=2048)
	parser.add_argument('--batch-bases', type=int, default=1 << 18)
	parser.add_argument('--seed', type=int, default=20261031)
	parser.add_argument('--threads', type=int, default=1)
	parser.add_argument('--engine', choices=['auto', 'zlib', 'libdeflate', 'isal'],
		default='auto')
	parser.add_argument('--repeats', type=int, default=5)
	parser.add_argument('--selections', nargs='+', choices=['one', 'all'],
		default=['one', 'all'])
	parser.add_argument('--workloads', nargs='+', choices=WORKLOADS, default=WORKLOADS)
	parser.add_argument('--no-index-ignored', action='store_true')
	parser.add_argument('--baseline-reader', help='Pre-extension figwig.py, imported '
		'with current helper kernels; record its exact SHA and optional git revision')
	parser.add_argument('--baseline-revision', help='Verify baseline-reader against '
		'this revision of figwig/figwig.py')
	parser.add_argument('--worker', help=argparse.SUPPRESS)
	args = parser.parse_args()
	if args.worker:
		worker(json.loads(args.worker))
	else:
		if args.bigwig and args.dtype is None:
			parser.error('--bigwig requires explicit --dtype int32 (counts) or float32')
		args.dtype = args.dtype or ['int32', 'float32']
		if min(args.tracks) < 1 or min(args.repeats, args.threads, args.tiles,
				args.batch_tiles, args.batch_bases) < 1 or args.chrom_bases < 0:
			parser.error('Track counts, repeats, threads and batch sizes must be positive')
		if args.tile < 16 or args.tile > 65536 or args.tile & (args.tile - 1):
			parser.error('--tile must be a power of two from 16 through 65536')
		run(args)
