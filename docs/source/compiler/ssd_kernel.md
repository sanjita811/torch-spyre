# Mamba-2 SSD Kernel on Spyre - Design & Handover

This document is the handover for the Mamba-2 **state-space-duality
(SSD)** kernel on Spyre. It explains what the kernel computes, how it is mapped
onto the Spyre accelerator as a small set of fused `torch.compile` graphs, and
*why* each non-obvious choice was made, which backend limits forced it, and what
is still open.

| File | Role |
|---|---|
| [`torch_spyre/_inductor/customops.py`](../../../torch_spyre/_inductor/customops.py) | The device kernels *only* - four pure traced graphs (`_ssd_build_decay_matrix`, `_ssd_build_intra_decay`, `_ssd_fused_cblock`, `_ssd_fused_masked`), no module-level state. Per-kernel tiling config is on the signatures (`bh_tiles=32`, `cblock=64`); in-graph `spyre_hint` is the only layout machinery here. |
| [`tests/inductor/test_ssd.py`](../../../tests/inductor/test_ssd.py) | Everything host-side: the driver (`_ssd_spyre`), the named-dim binders (`_ssd_declare_dims` / `_ssd_bind_dims`), the device-const cache (`_ssd_device_const`), the driver-policy constants (`_SSD_INTRA_FACTORED_TOTAL_LIMIT`, `_SSD_MAX_FLAT_SCAN_CHUNKS`), the CPU oracle (`ssd_cpu`), the Mamba-2 reference (`ssd_reference`), and the rel-L2 correctness tests. |
| [`tests/inductor/test_coarse_tile_e2e.py`](../../../tests/inductor/test_coarse_tile_e2e.py) | Flash-style `run_coarse_tile_test` structural + Spyre-vs-CPU checks for the two fused kernels (Group 11). |

Code comments intentionally stay high-level and point here; the detail, measured
numbers, and backend history live in this file.

> **Note - how to run.**
>
> ```bash
> # End-to-end rel-L2-vs-Mamba correctness (factored + masked paths).
> LD_PRELOAD=/opt/ibm/spyre/deeptools/lib/libutil.so \
>   python -m pytest tests/inductor/test_ssd.py -q
>
> # Flash-style structural + Spyre-vs-CPU self-consistency for the kernels.
> LD_PRELOAD=/opt/ibm/spyre/deeptools/lib/libutil.so \
>   python -m pytest tests/inductor/test_coarse_tile_e2e.py -k ssd -q
> ```
>
> `test_ssd.py` validates the three tiers (see [Validation harness](#validation-harness));
> a clean run reports the fp32 formulation error, the fp16 numeric floor, and the
> device rel-L2 for both `Y` and the final state. The device is exclusive to one
> process (VFIO single-holder) - never run the Spyre suites in parallel.

## What the kernel computes

Mamba-2 replaces the recurrent SSM scan with the **chunked SSD** formulation: an
algebraically-equivalent form that is mostly dense matmuls, so it maps onto
matmul hardware. The sequence of length `T` is split into `C = T / L` chunks of
length `L`, and the output is the sum of two terms:

1. **Intra-chunk (diagonal) term** - an attention-like `(L, L)` interaction
   *within* each chunk, masked causally and weighted by a within-chunk decay.
2. **Inter-chunk (off-diagonal) term** - each chunk summarizes its contribution
   into a small state `(N, P)`; a **scan** propagates those states forward across
   chunks under a per-chunk decay, and each chunk reads the rolled-in state from
   all earlier chunks.

The dimension names (Mamba-2 convention) used throughout the code:

| Symbol | Meaning | Default |
|---|---|---|
| `B` | batch | 2 |
| `T` | sequence length | 4096 |
| `H` | model dim | 2048 |
| `P` | head dim | 64 |
| `N` | SSM state dim | 128 |
| `nheads` | `H // P` | 32 |
| `L` | chunk length | picked per `T` |
| `C` | chunk count `T // L` | derived |
| `G` | ngroups (B/C projection sharing) | 1 |
| `BH` | folded batch·heads `B·nheads` | 64 |

The ground truth is `ssd_reference` - the **verbatim** `ssd_minimal_discrete`
from `state-spaces/mamba`, unmodified. Everything else is validated against it.

## Mapping to Spyre: fused kernels + a thin host driver

The SSD core runs as **one fused `torch.compile` graph** on Spyre
(`_ssd_fused_cblock`, or the fp16-overflow fallback `_ssd_fused_masked`), fed by a
small separately-compiled `_ssd_build_decay_matrix` graph. The driver `_ssd_spyre`
(in `tests/inductor/test_ssd.py`) does the host-side data preparation, picks the
path, binds tensor-dim names, compiles, and un-folds the result.

Two structural decisions frame the layout:

- **Fold `B` and `nheads` into a single leading `BH` batch dim.** Every stage
  is independent across `(batch, head)`, so `BH = B·nheads` is the natural
  parallel axis. It is the *only* dim tiled by `spyre_hint` (see below): it is a
  pure batch dim - not a matmul contraction dim, not a scan/reduction dim, not a
  stick dim - so splitting it is offset-free and needs no cross-stick data
  movement.
- **Chunk to `(BH, C, L, ·)` and pad `C` up to a whole 64-element stick.** The
  scan's chunk dim `C` becomes a stick dim, and Spyre transfers memory in
  128-byte sticks (64 fp16 elements). When `C` is not a multiple of 64 the driver
  `F.pad`s the chunk dim; padded chunks carry zeros and contribute nothing, and
  the padding is dropped when un-folding `Y`. (Note: it is **`C` that must reach
  64**, not `L` - a subtlety that lets small-`L` configs work.)

Why tile only `BH`, and why *hints* at all? The kernel is a chain of batched
matmuls whose automatic work-division and layout choices repeatedly hit backend
walls (span overflow, cross-stick slicing, restickify). `spyre_hint` lets us pin
the parallelization to the one axis that is provably safe and steer the scan
matmul's core split explicitly. See
[Work Division Planning](work_division_planning.md) and
[Span-Overflow Hint Analysis](span_overflow_hint_analysis.md)
for the mechanisms these hints drive.

## The inter-chunk scan (the crux)

This is the hardest part of the kernel and the reason for most of its shape.

The scan is `decay_matrix (BH, C+1, C) @ chunk_states (BH, C, N·P)`. It is
`O(C²)` in **both** MACs and memory: the dense `decay_matrix` alone is
`(C+1)·C` per `BH`. On the device the matmul materializes a coarse-tile
read-copy whose **per-core span scales with the `C+1` row dim**. The AIU per-core
span limit is 255.996 MiB, and the work-division pass cannot split that row dim
below the limit - so the **dense flat scan hard-walls the backend at `C > 64`**
(measured on merged main: `C ≤ 64` compiles; `C ≥ 128` always fails with a span
overflow, a dxp `SIGABRT`, or an HBM-pool OOM). `_SSD_MAX_FLAT_SCAN_CHUNKS = 64`
encodes this cap.

### C-block scan - `work_div` over the output rows

The fix (`_ssd_fused_cblock`) wraps the scan matmul in
`spyre_hint(work_div={"C": NCB})`:

```python
ncb = max(1, c // cblock)                    # cblock = 64 rows (one stick)
with spyre_hint(work_div={"C": ncb}):
    scan = torch.matmul(dr, cs_np)           # (BH, C, N·P) run rows
```

`work_div` is a *per-op core split* with no loop-nest-contiguity constraint
(unlike coarse-tile `tiles=`). It fans the scan's **`C` output-row dim** across
the cores while the reduction `C` and the `N·P` columns stay whole per core, so
each core touches only `(C/NCB, N·P)` and stays under the span limit for `C` up
to 512. `C` is a batch dim of the matmul (never a stick/reduction dim), so
splitting it is offset-free - this is exactly why the split is legal here when
the dense form is not.

> **Why the `C`-row dim, and not `C+1`?** Only the `C` **run rows** go through the
> hinted scan (output `(BH, C, N·P)`). The extra **final-state row** (`C+1`) would
> tip the output back over the span wall, so it is computed as a *separate small*
> `decay_final (BH, 1, C) @ cs` matmul in the same graph. The kernel therefore
> returns `(y_grouped, scan_final)` from **one** compile - no separate
> final-state kernel and no `chunk_states` host round-trip.

The `work_div` form replaced an earlier per-block Python `for`-loop that was
retired once merged-main PRs #3530 / #3612 lifted the `work_div` restickify wall.
It is ~8–15× faster than the loop across `C = 128..512` and stays under span
across the whole shipped envelope.

## Factored intra

The within-chunk decay is `L[i, s] = exp(g_i − g_s)` for `i ≥ s` (and masked
otherwise), where `g` is the within-chunk cumulative decay. Materializing the
dense `(L, L)` decay per chunk is a `(BH, C, L, L)` tensor - expensive and, at
long `L`, span-overflowing.

The factored path (`_ssd_fused_cblock`) reconstructs that decay as an **outer
product** folded into the `C` and `B` projections instead of materializing the
matrix:

```text
exp(g_i − g_s) = exp(shifted_i) · exp(−shifted_s),   shifted = g − total/2
```

- `c_scaled = c_proj · exp(shifted)` and `b_scaled_t = (b_proj · exp(−shifted))ᵀ`,
  so `attn = (c_scaled @ b_scaled_t) · causal_mask` reproduces the decayed
  attention with no `(L, L)` intermediate.
- Centering by `total/2` (`shifted = g − total/2`) halves the fp16 exponent
  range: the peak scale becomes `exp(|total|/2)` instead of `exp(|total|)`.
- The same `half_tot = exp(total/2)` feeds both the chunk-state build
  (`b_scaled_t @ (x · half_tot)`) and the off-diagonal combine
  (`c_scaled @ rolled · half_tot`).

### fp16 guard

The factored path is fp16-safe while
`max|chunk_decay| < _SSD_INTRA_FACTORED_TOTAL_LIMIT`
(≈ 20 = `2·(ln 65504 − 1)`, a margin below the fp16 exponent ceiling). Above it,
`_ssd_spyre` falls back to the masked kernel. **In the shipped `T` envelope the
guard always keeps the factored path**; the fallback exists for robustness on
out-of-distribution decays.

> **fp16 saturation in the CPU oracle.** The un-masked upper triangle of
> `c_scaled @ b_scaled_t` is `exp(g_i − g_s)` with `i < s`, which overflows fp16
> to `+inf`; the causal mask then makes it `inf·0 = NaN`. The **device saturates
> fp16 in hardware** (clamps to 65504), so the CPU oracle explicitly clamps that
> product to the fp16 range *before* masking - the clamped entries are exactly the
> masked-away ones, so fp32 is unchanged. Without this clamp the fp16 "numeric
> floor" print is meaningless (it was `NaN` from `T = 8192` up). This is the single
> most load-bearing subtlety in making the CPU oracle a *faithful* device mirror.

## Masked fallback

`_ssd_fused_masked` uses a precomputed dense `(BH, C, L, L)` intra decay mask
plus the plain dense `(BH, C+1, C)` scan. It is unconditionally fp16-safe (the
decay is bounded by construction) but has **no C-block variant**, so it only runs
at small `C` where the dense scan still fits the span. It is not reached in the
shipped `T` envelope and exists purely as the fp16-overflow safety net selected
by the guard above.

## Decay-matrix build

`_ssd_build_decay_matrix` emits only the `(BH, C, C)` **run rows** on device. The
`(BH, 1, C)` final-state row is built on the **host** (`exp` of a host tensor +
one H2D copy) because a size-1 row mis-lays-out as a compile output. Emitting
only the run rows on device avoids three separate problems:

1. a `torch.cat` whose mutation-copies trip layout warnings,
2. a separate compiled graph just to re-slice the run rows back out, and
3. a D2H round-trip for the final row.

The build's inputs carry the stick on the `BH` dim (`SpyreTensorLayout(...,
dim_order=[1, 0])`) so the `C`-broadcast in the outer-difference
`decay_before[...,None] − decay_cumsum[...,None,:]` lands **off** the stick.

## Graph-input pre-copies (`* 1.0`)

Inside the kernels, raw graph inputs that feed a matmul or slice are pre-copied
(`a * 1.0`, `x * 1.0`, `decay_run * 1.0`, …) to turn each `InputBuffer` into a
`ComputedBuffer`. The backend normally inserts these read copy-ins itself
(coarse-tile `_insert_all_read_copy_ops`), but that pass is **skipped on the
span-overflow (post-stickify) path** this kernel takes, so the copy is manual.

Inputs whose *first use is already a real scale* - `c_proj * exp(shifted)`,
`b_proj * exp(−shifted)` - need no separate `* 1.0`: the multiply itself produces
the `ComputedBuffer` (the same idiom flash-attention uses with
`key * scaling_factor`). Removing the remaining `* 1.0` copies increases the
read-copy count and crashes codegen, so they stay. This is a workaround, tracked
against backend issue #3381.

## Named dims

The layout pass resets its dim registry **every compile**, so `declare_tensor_dim`
+ `name_tensor_dims` must be re-issued before each `torch.compile`. `_ssd_bind_dims`
does both from a `(tensor, names)` table, and the two kernel branches share a
`common` spec. Flash-attention avoids explicit naming by reusing
`spyre_hint(tiles=)` keys as dim names, but this kernel needs explicit names on
the matmul contraction dims (`Lk` / `La` / `Ca`), which are **not** tiled - so
they must be declared by hand.

**Why the binders live in the driver, not the kernels.** `name_tensor_dims`
binds names to tensors *by object identity* on the real host tensors, before the
graph is traced - a `torch.compile`d kernel only ever sees `FakeTensor`s and so
*cannot* issue these bindings for its own graph inputs. Naming is therefore an
irreducibly host-side, pre-compile step: `_ssd_declare_dims` / `_ssd_bind_dims`
sit in the driver (`_ssd_spyre`) next to the tensors they annotate, and the
kernels in `customops.py` stay pure traced graphs carrying only in-graph
`spyre_hint`. This is also why the flash approach (no naming at all) is not
reachable here rather than merely unused: flash's load-bearing dims are the ones
it tiles, so their tile keys double as names; SSD's load-bearing dims are the
untiled contraction/batch axes, which the solver cannot infer without the
explicit host-side bindings, and which cannot be tiled to earn names the flash
way (`BH` is the only offset-free tile axis; C-tiling hits the cross-stick wall).

## Config policy

The kernel cost has a **U-curve in the chunk size `L`** (with `C = T/L`):

- intra terms (`c @ bᵀ`, `attn @ x`) grow `∝ L`;
- the inter-chunk scan is `O(C²) = O((T/L)²)` in both MACs *and* memory, so it
  explodes as `L` shrinks at long `T`.

The compute optimum is `L*(T) = (2·T·N·P / (N+P))^⅓`, snapped to the nearest
64-multiple that divides `T` (64-multiple because `L` is a fp16 stick;
*nearest* not floor because the scan cost rises steeply below `L*`). Measured
example: at `T = 16384`, doubling `L` from 64 → 128 cut clean device kernel time
`741 ms → 53.6 ms` (14×). Three constraints layer on top of `L*`:

- **scan mode by span** - `C ≤ 64` → dense flat scan (`O(C²)`);
  `64 < C ≤ 512` and 64-aligned → the C-block scan above; beyond that a
  hierarchical (device-blocked, sub-quadratic) scan would be required - needed
  only when no single `L` satisfies both the scan-span *and* the
  intra-`L×L`-span limits at once (`T ≳ 32768`).
- **fp16 guard** - keep the factored intra in range; `_ssd_spyre` makes the final
  factored-vs-masked call at runtime from the actual data
  (`max|chunk_decay| < _SSD_INTRA_FACTORED_TOTAL_LIMIT`).
- **`BH` tiling** - the kernels' `bh_tiles` param, the largest tile count that
  divides `BH`.

> **Config lives where it belongs.** Driver policy (the fp16-guard limit and the
> flat-scan cap) sits in `tests/inductor/test_ssd.py` as
> `_SSD_INTRA_FACTORED_TOTAL_LIMIT` / `_SSD_MAX_FLAT_SCAN_CHUNKS`; per-kernel
> tiling (`bh_tiles=32`, `cblock=64`) sits on the kernel signatures in
> `customops.py`. For the fixed sweep shape (`B2/H2048/P64/N128`) only `L` and the
> scan mode vary with `T`; `bh_tiles` and `cblock` are held at their defaults. The
> test hardcodes the inline `L` rule directly rather than carrying a general
> analytic chooser - the analytic `L*` policy above is the design rationale, not
> shipped code. (An earlier standalone `ssd_config.py`/`pick_config` chooser was
> removed once the kernels became test-driven with no registered op to consume it.)

Measured winners (warm ms / peak GB, `bh_tiles = 32`):

| T | L | C | scan | warm | peak |
|---|---|---|---|---|---|
| 4096 | 64 | 64 | flat | 432 ms | 0.02 GB |
| 8192 | 64 | 128 | cblock | 951 ms | 0.04 GB |
| 16384 | 128 | 128 | cblock | 2053 ms | 0.11 GB |
| 32768 | 128 | 256 | cblock | 3885 ms | 0.23 GB |
| 65536 | 128 | 512 | cblock | 7918 ms | 0.50 GB |

> **Open call - `bh_tiles = 32` vs 16.** `32` wins on peak memory and roughly
> ties on speed, but its **cold compile is ~200–470 s** versus ~48 s at 16 - an
> 8–13× compile tax for a mostly-negligible memory win. The device sweep found
> ≤ 16 tiles as fast or faster at every `L`. Whether to keep 32 everywhere is
> unresolved and pending a mentor decision. (Memory: *SSD bh_tiles compile tax*.)

## Groups (ngroups / G)

`G` is Mamba-2's `ngroups`: the `B` and `C` SSM projections are shared across
`nheads/G` heads (the SSM analog of grouped-query attention). Common in real
checkpoints; default 1.

The kernel is **G-agnostic**: `b` / `c` are `repeat_interleave`-expanded to
per-head `(B, nheads, C, L, N)` on the host *before* the kernel, so the device
math is identical for any `G`. `G` appears only in the test's input generation.

> **Future optimization (not implemented - bounded win, high complexity).** For
> `G > 1` the host expansion materializes `nheads/G` redundant copies of `B` and
> `C`. A G-aware kernel could keep them `(B, G, C, L, N)` and broadcast per-group
> inside the matmuls (as the Mamba-2 GPU reference does). The catch: `a`, `x`, and
> `shifted` are all per-head, so grouped `b` / `c` must broadcast across the
> head-within-group dim *inside* the `BH` coarse-tiled loop - a backend-risky
> expand - and the saving is bounded to the input tensors plus their read-copies
> (the dominant `chunk_states` and scan buffers are unchanged). Worth revisiting
> only for a grouped-model customop where input bandwidth dominates.

## Validation harness

`tests/inductor/test_ssd.py` validates in three independent tiers so a failure
localizes to a specific layer:

1. **`ssd_reference`** - verbatim Mamba-2 `ssd_minimal_discrete`. The ground
   truth. Never modified.
2. **`ssd_cpu`** - an independent CPU oracle that runs *the same op sequence,
   routing, C-padding, and rank-1 init as the device path* but shares no code with
   it. Run in **fp32** it gates the *formulation* (`Y` rel-L2 < 1e-3); run in
   **fp16** it prints the *numeric floor* - the best any fp16 device could do
   (this is where the saturation clamp above is essential).
3. **`_ssd_spyre`** - the real device kernel, compared to the reference at the
   fp16 relative-L2 budget.

`test_ssd_factored` exercises the factored `_ssd_fused_cblock` path;
`test_ssd_masked` forces the masked fallback via `factored_limit=0.0`.
`tests/inductor/test_coarse_tile_e2e.py` (Group 11) adds the flash-style
`run_coarse_tile_test` structural check: it compiles each fused kernel once,
asserts the tile `LoopSpec` is present, and checks compiled-Spyre ==
eager-CPU-of-the-same-kernel elementwise (self-consistency, not fidelity to
Mamba - that is what the rel-L2 gate above covers).

Acceptance metric for the end-to-end tests is **relative L2**
(`‖got − ref‖ / ‖ref‖`). Gates:

| Check | Metric | Threshold | Typical |
|---|---|---|---|
| fp32 formulation | `Y` rel-L2 | < 1e-3 | ≈ 0.0002 |
| device fp16 | `Y` rel-L2 | < 0.05 | ≈ 0.004–0.006 |
| device fp16 | final-state rel-L2 | < 0.05 | ≈ 0.004–0.006 |

## Known limitations & backend asks

- **Flat scan caps at `C ≤ 64`; C-block extends to `C ≤ 512`.** Beyond that a
  hierarchical (device-blocked, sub-quadratic) scan is required. Its 4D block
  matmul now compiles on merged main, but the fully-blocked scan path is not the
  one shipped here. *Backend ask:* split the scan's `(C+1)` row dim in
  work-division, or tile the scan matmul, to lift the cap without a kernel-side
  blocked scan.
- **Manual `* 1.0` read-copies** are required because the coarse-tile read-copy
  pass is skipped on the span-overflow path (issue #3381).
- **The `(BH, 1, C)` decay-matrix final row must be built on the host** - a size-1
  compile output mis-lays-out.
- **`bh_tiles = 32` carries an 8–13× cold-compile tax** (see the open call
  above).
- **N-blocked scan is backend-blocked.** An `N`-blocked flash-accumulator scan is
  math-correct and compiles standalone (shrinking the pool from 545 MB to 46 MB),
  but `N = 128` spans two sticks and slicing/reducing across a stick boundary
  inside the multi-op hinted graph is not yet supported by the backend. The dense
  scan stays until cross-stick slice/reduce lands.
- **G-aware kernel** not implemented (bounded win - see Groups above).

## See also

- [Work Division Planning](work_division_planning.md) - the `work_div` mechanism
  the C-block scan drives.
- [Span-Overflow Hint Analysis](span_overflow_hint_analysis.md) - the per-core
  span limit and the post-stickify path this kernel takes.
- [Coarse-Tiling Loops](coarse_tiling_loops.md) - the `tiles=` hint and the
  read-copy pass that `* 1.0` stands in for.
- [Tensors and Layouts](../user_guide/tensors_and_layouts.md) - sticks,
  `dim_order`, and `SpyreTensorLayout`.
- `state-spaces/mamba` `ssd_minimal.py` - the reference this kernel is validated
  against.
