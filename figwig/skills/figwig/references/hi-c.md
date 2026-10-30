# Hi-C contact matrices as tracks

One track per bin: track *i* is row *i* of the matrix and contact (*i*, *j*)
is base *j* of track *i*. Bulk Hi-C, Micro-C and single-cell Hi-C fit.

## Layout

| Choice | Use | Why |
|---|---|---|
| coordinates | each chromosome as long as its number of bins, a position per bin | in bp a contact sits at its bin's start base, and `tile` stops at 65,536, so from 64 kb bins a chunk holds one column at most; a bp file took twice the bytes (in the chunks, not the index) and a row read decodes a chunk per 64 kb of genome. Use bp only to share a file with base-level tracks, read with `bins=resolution` |
| triangles | both, plus a `diagonal` track | a row (virtual 4C) is one track; with the upper triangle only, the part of row *i* left of the diagonal is base *i* of every earlier track. The upper triangle alone halves the bytes, as in cooler: mirror a window with `m + m.T - numpy.diag(numpy.diag(m))` in a signed type |
| tile, groups | 256 bins, 1,024 tracks, then measure | a row decodes every chunk of its group along the genome, so smaller groups read rows faster; larger ones read windows faster in a smaller file |

## Writing from cooler pixels

```python
import numpy
from figwig import FigWigWriter

def write_contacts(path, names, n_bins, bin1, bin2, count, tile=256, group=1024):
	"""Upper-triangle pixels with global bin ids, as cooler stores them, to
	one track per bin, both triangles and a 'diagonal' track. `n_bins` is a
	numpy array of each chromosome's bins."""

	offsets = numpy.concatenate([[0], numpy.cumsum(n_bins)])
	n = int(offsets[-1])
	key, inverse = numpy.unique(bin1.astype(numpy.int64) * n + bin2,
		return_inverse=True)                         # repeated pixels summed
	count = numpy.bincount(inverse, weights=count).astype(numpy.int64)
	bin1, bin2 = key // n, key % n

	off, on = bin1 != bin2, bin1 == bin2
	rows = numpy.concatenate([bin1, bin2[off], numpy.full(on.sum(), n)])
	cols = numpy.concatenate([bin2, bin1[off], bin1[on]])
	vals = numpy.concatenate([count, count[off], count[on]])

	tracks = ['{}:{}'.format(name, i) for name, size in zip(names, n_bins)
		for i in range(size)]
	groups = [tracks[i:i + group] for i in range(0, n, group)] + [['diagonal']]
	tracks.append('diagonal')
	chrom = numpy.searchsorted(offsets, cols, side='right') - 1
	with FigWigWriter(path, dict(zip(names, n_bins.tolist())), tracks, tile=tile,
			groups=groups) as writer:
		for c, name in enumerate(names):             # the column's chromosome
			take = numpy.flatnonzero(chrom == c)
			take = take[numpy.argsort(cols[take], kind='stable')]
			writer.write_sparse(name, cols[take] - offsets[c], rows[take], vals[take])
```

- cooler's pixels are row-major and `write_sparse` takes a group's entries
  one chromosome at a time in position order (`references/figwig-writing.md`),
  so the entries are regrouped by the column's chromosome. At full depth,
  spill them to a file per column chromosome and write each in calls that
  end on a tile: a call takes about 90 bytes per entry, several GB for
  chr1's columns at 5 kb.
- Published `.mcool` files repeat pixels: 4DN's K562 file 4DNFI18UHVRO stores
  thousands of (bin1, bin2) pairs twice at each resolution, and
  `write_sparse` raises `entries ... both name base` on them. The
  `numpy.unique` above sums them.

## Reading

```python
from figwig import FigWigReader

fw = FigWigReader(path)
offsets = numpy.concatenate([[0], numpy.cumsum(n_bins)])
n = int(offsets[-1])                                 # 'diagonal' is track n
c, s, k = 0, 10, 20                                  # chromosome, first bin, bins
rows = numpy.arange(offsets[c] + s, offsets[c] + s + k)

window = fw.read([names[c]], [s], k, tracks=rows)[0]   # (k, k), the tracks' dtype

i = 77                                               # row i over the genome
y = fw.read(names, [0] * len(names), int(n_bins.max()), tracks=[i])[:, 0]
row = numpy.concatenate([y[j, :size] for j, size in enumerate(n_bins)])

f = 5                                                # fine bins per coarse bin
y = fw.read([names[c]], [s], k, tracks=numpy.append(rows, n),
	labels=numpy.append(numpy.arange(k) // f, k // f), bins=f)[0]
coarse, diagonal = y[:-1], y[-1]
d = numpy.arange(k // f)
coarse[d, d] = (coarse[d, d] + diagonal) / 2         # as cooler counts it
```

- `labels=` over rows and `bins=` over columns combine in one read, so one
  fine file serves every coarser resolution. With both triangles a contact
  between two fine bins of one coarse bin is in both rows, and the coarse
  diagonal holds it twice; the `diagonal` track under one more label fixes it
  in the same read. `sums(..., paired=True)` over each row's own bin gives
  the fine diagonal too, and is much slower.
- A row needs one width for every chromosome: read each at the longest
  chromosome's width and trim; an integer read is 0 past a chromosome's end.

## Single cell

| Layout | Pseudobulk window | One whole cell | Notes |
|---|---|---|---|
| one track per cell; each chromosome pair *c1* <= *c2* a pseudo-chromosome of *n1* × *n2* positions, pixel (*i*, *j*) at (*i* − *o1*) · *n2* + (*j* − *o2*), upper triangle | one read, a window per row, `labels=` per cell type | position order decodes every chunk: keep a copy from `figwig reorder --order tracks -l 6` (track order is raw without `-l`) | a pseudo-chromosome holds at most 2**32 − 1 positions (human chr1 × chr1 fits to about 4 kb bins); a window past *n2* runs on into the next row without an error |
| one track per bin, the genome copied per cell (`17:chr5`, in bins) | one read, a window per cell, summed per type after | one read of the cell's chromosomes, `sparse=True` | serves both queries from one file; its writer's memory grew far past the others' at 10 kb: measure |
| one track per (cell, bin) | | | open parses every track's record: about 10M tracks at 1 Mb took seconds and gigabytes, about 1e9 at 10 kb is out of reach |

```python
def write_cells(path, names, n_bins, cell, bin1, bin2, count, cell_names):
	"""Every cell's upper-triangle pixels, global bin ids, to one track per
	cell over chromosome-pair pseudo-chromosomes."""

	offsets = numpy.concatenate([[0], numpy.cumsum(n_bins)])
	n = int(offsets[-1])
	key, inverse = numpy.unique((cell.astype(numpy.int64) * n + bin1) * n + bin2,
		return_inverse=True)                         # repeated pixels summed
	count = numpy.bincount(inverse, weights=count).astype(numpy.int64)
	cell, bin1, bin2 = key // (n * n), key // n % n, key % n
	c1 = numpy.searchsorted(offsets, bin1, side='right') - 1
	c2 = numpy.searchsorted(offsets, bin2, side='right') - 1
	pair = c1 * len(names) + c2
	position = (bin1 - offsets[c1]) * n_bins[c2] + (bin2 - offsets[c2])   # int64
	order = numpy.argsort(pair * 2**32 + position, kind='stable')
	pair, position, cell, count = pair[order], position[order], cell[order], \
		count[order]

	sizes = {'{}|{}'.format(names[x], names[y]): int(n_bins[x] * n_bins[y])
		for x in range(len(names)) for y in range(x, len(names))}
	edges = numpy.flatnonzero(numpy.diff(pair)) + 1
	with FigWigWriter(path, sizes, list(cell_names), groups=len(cell_names)
			) as writer:
		for a, b in zip(numpy.r_[0, edges], numpy.r_[edges, len(pair)]):
			x, y = divmod(int(pair[a]), len(names))
			writer.write_sparse('{}|{}'.format(names[x], names[y]), position[a:b],
				cell[a:b], count[a:b])
```

One sort by pair and position replaces a mask per chromosome pair, which
scans every entry once per pair (hundreds of pairs). Positions pass 2**31
from about 5 kb bins on human chromosomes, so keep them int64. A `.scool`
from before cooler 0.8.9 keeps each cell as a whole cooler at the file's top
level: `cooler.Cooler(path)` raises `KeyError` and
`cooler.fileops.list_scool_cells` raises `OSError`;
`cooler.fileops.list_coolers(path)` lists the cells, each opened as
`cooler.Cooler(path + '::' + name)`.

A pseudobulk window from the first layout, `cells` its reader and
`cell_type` one label per cell:

```python
size = int(n_bins[c])
name = '{}|{}'.format(names[c], names[c])
y = cells.read([name] * k, numpy.arange(s, s + k) * size + s, k,
	labels=cell_type)                                # (k rows, types, k columns)
y = numpy.rint(y).astype(numpy.int64).transpose(1, 0, 2)
pseudobulk = numpy.stack([m + m.T - numpy.diag(numpy.diag(m)) for m in y])
```

## Limits that bite at Hi-C sizes

- `tile_sums` needs tracks × tiles × 4 bytes within 2**28 (`references/errors.md`);
  at 256-bin tiles over hg38 that fails from about 10 kb, and it raises
  `keeps no summary`. Use `read(bins=)` or `sums`.
- Opening parses a record per track, so open time and memory grow with bins.
- hictk's `fetch(range1, range2)` refuses a rectangle that crosses the
  diagonal (`overlaps with the lower-triangle`): split a row there to check
  one against it. Its `chr:start-end` strings pick the same bins as cooler's.
