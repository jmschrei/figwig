# Optional regional summary indexing

Implement an opt-in `FigWigWriter(..., summary_index=True)` accelerator, default
`False`, preserving the frozen v1 layout, stored summary bytes after inflation,
and existing reader APIs. This repository owns its extension specification,
writer and local Python summary reader. The companion browser reader plan is
`/Users/klavikka/projektit/figwig-js/SUMMARY_RESTARTS_PLAN.md`.

## Contract and simplicity review

V1 permits ignored optional bits, foreign JSON metadata keys and unused bytes in
the data region. It requires one complete zlib stream per legacy summary/zoom,
back-to-back group records ending at declared `length`, a zero reserved zoom
offset and unchanged checksum/summary semantics. Literal BGZF or concatenated
zlib streams cannot replace those summaries.

Use full-flush restart boundaries within each single legacy zlib stream. IDs
precede all sum rows, so a regional window needs separate ID and sum fragments.
Each fragment has exact bounds, row count and compressed/uncompressed CRCs; a
bounded raw inflater appends an empty final block (`03 00`) for decoding. Keep
whole-stream CRC/Adler checks for existing readers and whole-level APIs.

Publish a namespaced metadata locator to a checksummed root in the unused data
region and use optional bit 1 (`0x2`). Bind the accelerator to the
exact final 128-byte header. This is a branch-local convention,
not a claim of an upstream-assigned official feature. Do not change reserved
fields or the frozen core record layout.

The 32 KiB / 2,048-bin design is profile 2 in both the metadata
locator and root. Unsupported profiles, including the earlier 16 KiB profile 1,
fall back to ordinary v1 summaries. Keep the existing magic as a family marker.

The selected design uses directly addressed CRC-paged fixed-window descriptors,
with no B+ tree or per-chromosome block directory. A window is the existing
level-global bin ID divided by its fixed bin count; storage windows may contain
bins from neighboring chromosomes while biological summary bins remain separate.
This also avoids index padding proportional to the chromosome count. The exact
binary contract is [docs/summary_restarts.rst](docs/summary_restarts.rst): a
256-byte header-bound root, 32-byte group locators, variable 32-byte level
catalogs, and up to 102 forty-byte window descriptors per 4,096-byte page.

Keep sidecars, a second live copy of summary sums, checkpoint dictionaries,
converters, workers, new statistics and finer bin grids out of this initial
implementation. The Python reader keeps its existing public APIs and uses the
index for selected chromosomes and the exact integer interval-sum optimization.

## Writer work

1. Add and validate the boolean writer flag; the disabled path should keep
   existing output unchanged. Document both Python usage and append semantics.
2. When enabled, partition each existing summary/zoom by level-global fixed
   windows, with a target of 32 KiB decoded ID-plus-sum bytes, power-of-two window
   size capped at 2,048 bins. Never partition just by sparse row count. Budget
   at most 16 MiB for the entire directory before allocation, sharing page
   allowance deterministically among groups and considering finer levels first.
3. Leave a level on the legacy path only when both its compressed and decoded
   sizes are at most 64 KiB. Preserve absent versus available-empty summaries.
   Keep the existing global summary-memory cutoff.
4. Compress the exact existing ID and sum bytes with `Z_FULL_FLUSH` at each
   occupied window boundary, using one zlib wrapper and final Adler-32. Keep
   existing base float32 rounding and actual zoom float64 values unchanged.
5. Serialize the accelerator before the chromosome table. Compute the final
   tables/directory and header, fill the root's header binding, and write the
   header last. Raw chunks, their offsets and existing raw-index CRCs are intact.
   Avoid accumulating a second decoded pyramid.
6. Ordinary track append must preserve foreign metadata keys. Default append
   invalidates by clearing the accelerator bit while retaining a stale locator.
   Reject `append=True, summary_index=True` before opening the file. Merge/reorder
   create new files and may safely discard the extension; opt-in behavior does
   not propagate automatically.

## Python reader work

1. Discover the index lazily from the optional bit and metadata, bind it to the
   existing header identity and check each requested locator, catalog, source
   header, page and fragment. Unknown versions and valid stale roots use the
   ordinary path; recognized corruption raises, including inside `sums`.
2. Add one internal iterator over stored summary bins in a union of requested
   ranges. Read pages lazily, inflate one bounded window at a time, and keep
   only checked catalogs between calls. Retain no descriptor or decoded-window
   cache and no open file descriptor; drop catalogs on pickle.
3. Have `tile_sums(chroms=..., level=...)` allocate only the requested output,
   preserving track/chromosome order, duplicates, scalar shape, float32 base
   sums and actual float64 zoom values. Use whole streams for broad regions and
   keep the existing complete-stream path for whole-genome requests.
4. Use the same range iterator for the existing integer `sums` optimization.
   Merge whole-tile interiors without reading gaps between distant intervals;
   decode partial boundary tiles normally. Preserve the existing 2**24 and
   2**53 exactness guards. Return wholly aligned interiors directly. General
   binned reads and floating-point interval sums keep their raw-data semantics.

## Verification and companion reader

- Extend the independent specification reader with a separately written
  extension verifier. Generate both storage orders, multiple chromosomes,
  multiple groups and levels, sparse/empty windows, NaNs/zeros, cancellation sums,
  partial chromosome ends and rounded large-value cases.
- Check every raw value and every base/zoom payload with flag on/off and with
  unmodified pinned/current v1 readers and the Kaitai reader. No extension reader
  should be the oracle for its own expected values.
- Check malformed offsets/counts, page and fragment CRCs, wrong header binding,
  absent versus empty levels and mutation invalidation. Keep allocation and
  file-extent limits in every parser.
- Exercise Python's writer-to-public-reader path in both orders against the
  independent v1 stored-summary oracle and raw NumPy interval sums. Prove small
  selections avoid complete summary/raw-index reads, distant intervals read
  their union, corruption is not hidden by fallback, and pickle/concurrent
  reads/replacement preserve cache and file ownership.
- Keep a reusable local benchmark beside the existing `compare_*.py` scripts:
  paired off/on files and a same-file index-ignored control, fresh processes,
  warm-up outside timing, rotated configurations, repeated timings, exact
  output hashes, process peak RSS, file size and writer cost. Record logical
  read/decode counters separately from timing. Distinguish warm OS page cache
  from cold reader state and support public bigWig inputs as well as a seeded
  synthetic dataset. Retain source/input hashes, commands and every raw run.
- Implement regional support in figwig-js without changing whole-level APIs or
  ordinary v1 fallback. Test real Chromium cross-origin range access when CORS
  hides the physical size; verify exact ranges, cache ownership, abort and close.
- Benchmark flag off/on on identical data over a shared 50 Mbps link with 10 ms
  simulated RTT. Report initialization separately, cold and panning/zoom queries,
  payload/index/gap bytes, requests, bytes inflated, cache memory, file growth
  and writer construction cost. Preserve reproducible tooling and raw evidence.

## Acceptance

For a representative 65,536-bin eight-track level, an aligned 64-bin viewport
must avoid the whole level, decode only its fixed window (18,432 bytes with a
512-bin window), and fetch a bounded directory working set. Sparse absence must
return zero without payload reads; no-summary must retain the legacy fallback
or required-summary error. Old readers must produce unchanged values.

Keep the writer flag opt-in until complete correctness, compatibility and browser
range tests plus the specified network benchmarks justify changing that policy.
Keep all work on the requested branch without committing or publishing unless
separately requested.
