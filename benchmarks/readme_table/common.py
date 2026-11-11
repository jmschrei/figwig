# common.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

"""Helpers shared by the rows of the README table.

A row is a function `run(ctx) -> dict` in one of the row modules. It starts
every measurement as a worker process (`worker`), so that each timing has its
own interpreter, its own memory and warm inputs, and returns:

	{'setting': markdown, 'baseline': markdown,
	 'read': [item, ...], 'write': [item, ...], 'size': [item, ...],
	 'inputs': {...}, 'raw': {...}, 'notes': [str, ...]}

where an item is made by `item()`: one comparison, figwig against its
baseline, in seconds or in MB. `render.py` turns the rows into the table.
"""

import os
import sys
import json
import time
import shutil
import platform
import subprocess
import importlib.metadata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUN = os.path.join(HERE, 'run.py')
TESTDATA = os.path.expanduser(os.environ.get('TESTDATA_ROOT', '~/datasets/testdata'))
PACKAGES = ['figwig', 'numpy', 'numba', 'pybigtools', 'pyBigWig', 'zarr', 'scipy',
	'pandas', 'pysam', 'snapatac2', 'cooler', 'hictkpy', 'h5py', 'deflate', 'isal']


def item(figwig, baseline, unit='s', label=None, baseline_name=None, baseline_text=None):
	"""One comparison. `unit` is 's' or 'MB'. `baseline` is None where the
	baseline was not measured, and then `baseline_text` says what stands in."""

	return {'label': label, 'figwig': figwig, 'baseline': baseline, 'unit': unit,
		'baseline_name': baseline_name, 'baseline_text': baseline_text}


def pinned_env(threads, extra=None):
	"""An environment that holds every library to `threads` threads."""

	t = str(threads)
	env = dict(os.environ, OMP_NUM_THREADS=t, MKL_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t,
		NUMBA_NUM_THREADS=t, RAYON_NUM_THREADS=t, PYTHONHASHSEED='0')
	env.update(extra or {})
	return env


def worker(ctx, target, threads=1, extra_env=None, timeout=None, **kwargs):
	"""Run `module:function(ctx, **kwargs)` in a fresh process and return its
	dict. The function prints nothing the caller needs: its return value is
	the last line of stdout, as JSON. A failure raises with stderr."""

	command = [sys.executable, RUN, '--worker', target, '--ctx', json.dumps(ctx),
		'--kwargs', json.dumps(kwargs)]
	result = subprocess.run(command, env=pinned_env(threads, extra_env), capture_output=True,
		text=True, timeout=timeout)
	if result.returncode != 0:
		raise RuntimeError('{} {} failed ({}):\n{}'.format(target, kwargs, result.returncode,
			result.stderr[-3000:]))
	return json.loads(result.stdout.strip().splitlines()[-1])


def repeated(ctx, target, threads=1, repeats=None, key='seconds', **kwargs):
	"""`repeats` workers one after another; returns (median of `key`, every
	result). The median is of the runs' own `key`."""

	results = [worker(ctx, target, threads=threads, **kwargs)
		for _ in range(repeats or ctx['repeats'])]
	values = sorted(r[key] for r in results)
	return values[len(values) // 2], results


def peak_rss_mb():
	"""This process's own peak resident set, from VmHWM (ru_maxrss carries
	over the parent's memory on Linux)."""

	with open('/proc/self/status') as handle:
		for line in handle:
			if line.startswith('VmHWM:'):
				return int(line.split()[1]) / 1024
	return float('nan')


def testdata(path):
	"""The absolute path of a file of the test-data collection."""

	full = os.path.join(TESTDATA, path)
	if not os.path.exists(full):
		raise FileNotFoundError('{} is not in the test-data collection ({})'.format(path, TESTDATA))
	return full


def size_mb(path):
	"""The size of a file, or of a directory's files, in MB (10**6 bytes)."""

	if os.path.isdir(path):
		total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
	else:
		total = os.path.getsize(path)
	return total / 1e6


def fresh(path):
	"""Remove a file or directory if it exists, so that a write starts empty."""

	if os.path.isdir(path):
		shutil.rmtree(path)
	elif os.path.exists(path):
		os.remove(path)


def idle_percent(seconds=2):
	"""The percentage of the machine's CPU time idle over `seconds`."""

	def read():
		with open('/proc/stat') as handle:
			fields = [int(v) for v in handle.readline().split()[1:]]
		return fields[3] + fields[4], sum(fields)

	idle0, total0 = read()
	time.sleep(seconds)
	idle1, total1 = read()
	return 100.0 * (idle1 - idle0) / max(total1 - total0, 1)


def versions():
	found = {}
	for name in PACKAGES:
		try:
			found[name] = importlib.metadata.version(name)
		except importlib.metadata.PackageNotFoundError:
			pass
	return found


def machine():
	model = ''
	try:
		with open('/proc/cpuinfo') as handle:
			for line in handle:
				if line.startswith('model name'):
					model = line.split(':', 1)[1].strip()
					break
	except OSError:
		pass
	return {'cpu': model, 'cores': os.cpu_count(), 'python': platform.python_version(),
		'platform': platform.platform()}


def commit():
	try:
		out = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
			capture_output=True, text=True, check=True)
		dirty = subprocess.run(['git', '-C', REPO, 'status', '--porcelain', '--', 'figwig'],
			capture_output=True, text=True, check=True).stdout.strip()
		return out.stdout.strip() + ('+dirty' if dirty else '')
	except (OSError, subprocess.CalledProcessError):
		return None
