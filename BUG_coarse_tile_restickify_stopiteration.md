# Bug: coarse-tiled op reading a graph input crashes in `insert_restickify` (`StopIteration`)

**Severity:** blocks compilation of any `spyre_hint`-tiled kernel that reads a
graph input directly inside the tiling scope.
**Introduced by:** #3381 "Coarse tiling bug fixes" (commit `5e7364b`), interacting
with `insert_restickify`. Surfaced after merging main (`93300e5`) onto a
coarse-tiled Mamba-2 SSD kernel; also reproduces with a bare batched matmul.
**Env:** torch `2.13.0+cpu`, current `main` (`93300e5`).

## Symptom

```
torch._inductor.exc.InductorError: StopIteration:
  ...
  File ".../torch_spyre/_inductor/insert_restickify.py", line 232, in insert_restickify
  File ".../torch_spyre/_inductor/insert_restickify.py", line 164, in insert_restickify_on_node_inputs
  File ".../torch_spyre/_inductor/insert_restickify.py", line 116, in _create_restickify_node
```

## Minimal reproducer + the precise trigger

`repro_coarse_tile_restickify_stopiteration.py` — a single `spyre_hint`-tiled
**batched (3D)** matmul of two graph inputs, tiling the **batch** dim:

```python
def k(c, b):
    with spyre_hint(num_tiles_per_dim={"BH": 4}):   # BH = the batch dim
        return torch.matmul(c, b.transpose(-1, -2))  # c, b are graph inputs
```

`torch.compile(k, dynamic=False)(c_dev, b_dev)` → `StopIteration`.

**Why #3381's own suite doesn't catch it (the coverage gap):**
- `test_coarse_tile_e2e.py::test_hint_matmul_row_tiling` tiles the **M** dim of a
  **2D** `x @ y` of graph inputs — verified PASSES.
- Tiling the **batch** dim of a **3D batched** matmul of graph inputs (above) —
  FAILS. This exact shape isn't in the e2e suite, so #3381 went green on CI.
- Confirmed the axis matters: 2D tile-M passes; 3D tile-batch raises StopIteration.

## Root cause

1. **#3381** extended `_full_buffer_read_deps` (`wsr/coarse_tile.py:1757`) to treat
   **graph inputs** (`InputBuffer`, incl. `ConstantBuffer`) as needing a tile-sized
   *read copy* when read directly by a tiled op — previously only
   `SpyreEmptyFallback` did (`coarse_tile.py:1812`). This is correct in itself (a
   tiled op's load index is tile-scoped and would mis-index a full-size input).

2. That read copy is materialized as a **scheduler-created buffer** named
   `coarse_tile_read_copy_{tiled_op}_{dep}` (`wsr/coarse_tile.py:2367`), e.g.
   `coarse_tile_read_copy_buf5_arg2_1`. It has **no FX-graph origin node** (it is
   created after FX lowering, during the coarse-tile scheduler pass).

3. `insert_restickify` still needs to restickify that operand, and
   `_create_restickify_node` (`insert_restickify.py:116`) resolves the target by
   searching `V.graph.env` for a **`torch.fx.Node`** whose `TensorBox` name matches
   `arg_name`:

   ```python
   fx_arg_node = next(
       fx_node for fx_node, tb in graph_lowering.env.items()
       if isinstance(fx_node, torch.fx.Node)
       and isinstance(tb, TensorBox) and tb.get_name() == arg_name
   )
   ```

   For a `coarse_tile_read_copy_*` buffer this match never exists (it's not an
   FX-origin buffer), so `next(...)` raises `StopIteration`. `V.graph.get_buffer(arg_name)`
   *can* resolve it (it's a real scheduler buffer — see the debug block at
   `insert_restickify.py:459`), but the FX-node insertion path assumes every
   restickify target has an FX origin.

So: #3381 newly routes tiled graph-input reads through a scheduler-created copy
buffer, and `insert_restickify` has no path to anchor a restickify on a buffer with
no FX node. #3381's own message flags related "follow-up work" (the write-side
mutation gap), consistent with this read-side gap being unhandled.

## Suggested fix directions (for the backend team)

- In `_create_restickify_node`, when `arg_name` is a coarse-tile read-copy (or any
  scheduler-created buffer with no FX origin), insert/anchor the `spyre.restickify`
  without requiring an FX node — or unwrap to the underlying graph input's FX node
  and restickify there, then let the read-copy consume the restickified result.
- Or: have the coarse-tile read-copy carry provenance back to its source FX node so
  the existing `env` search resolves it.

## Impact on us (torch-spyre SSD kernel)

Our fused kernel (`test_ssd.py`) tiles over BH with `spyre_hint` and reads
`c_proj`/`b_proj`/`x` (graph inputs) directly inside the scope, so it now fails to
compile on merged main. Not specific to our kernel — the minimal repro above has
none of our specifics. Blocked until this is fixed (or #3381 read-copy path is
reworked).
