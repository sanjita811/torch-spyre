# Q for mentor: intra-chunk decay mask at L=256 — layout of an L×L outer-difference

**Context:** Mamba-2 uses `chunk_size=256`. Our SSD kernel's intra-chunk decay is
an `(BH, C, L, L)` matrix `exp(clamp(g_l − g_s, max=0))`, where `g = cumsum(a)` per
chunk (shape `(BH, C, L)`). I have a working fix but it costs two full `(BH,C,L,L)`
H2D transfers, and I suspect there's a cleaner way. Wanted your read.

## The core problem

The natural construction is an **outer difference of one tensor with itself**:

```python
outer = g.unsqueeze(-1) - g.unsqueeze(-2)     # (BH,C,L,1) - (BH,C,1,L) -> (BH,C,L,La)
mask  = exp(clamp(outer, max=0)) * causal
```

On Spyre this needs the single tensor `g` in **two incompatible stick orientations
at once**: the `unsqueeze(-1)` operand wants `L` on-stick, the `unsqueeze(-2)`
operand wants `La(=L)` on-stick. At **L=64** a custom `SpyreTensorLayout` with the
stick on the BH dim (so both L and La are off-stick) hides this and works. At
**L>64** it breaks — verified at L=256:

| construction | L=256 result |
|---|---|
| stick on BH (`dim_order [1,2,0]`) — the L=64 trick | **silent WRONG mask**, rel 0.9975 vs CPU |
| default layout (stick on innermost L) | compile error: `NotImplementedError: no mechanism to resolve stick incompatibility` |
| stick on C (`dim_order [...,1]`) | silent WRONG mask, rel 1.0017 |

So no single-input-tensor layout we tried builds the L×L self-outer-difference
correctly at L=256 — it either errors or silently corrupts.

(Same root issue as the *inline* intra decay, which we already know is backend-
blocked with the identical "no mechanism to resolve stick incompatibility" message.)

## The workaround I have (works, but heavy)

Pre-expand **both** operands to the full `(BH,C,L,La)` shape on the host, so the
kernel does a plain elementwise `g_row - g_col` with **no in-kernel broadcast /
outer-difference** — each operand then has one fixed, compatible layout:

```python
g_row = g.unsqueeze(-1).expand(BH,C,L,L).contiguous()   # varies along L
g_col = g.unsqueeze(-2).expand(BH,C,L,L).contiguous()   # varies along La
# kernel: exp(clamp(g_row - g_col, max=0)) * causal
```

**Verified correct at L=256** (mask rel 1.5e-3; full kernel Y=0.0059 / init 0.0058).

**The cost I dislike:** this materializes and H2D-transfers two full `(BH,C,L,L)`
fp16 tensors (2 × 134 MB at T=4096/L=256/BH=32) that are just broadcast copies of a
`(BH,C,L)` = 0.5 MB vector. It moves the outer-difference off-device entirely,
trading compute for a big transfer. It's on the fp16-fallback path (only when the
factored fast path would overflow — which at L=256 is ~0.9% of chunks by
`|total|>22`, but currently routes the whole batch), so it's not the common case,
but at L=256 it *is* the case whenever any chunk overflows.

## Questions

1. **Is there a supported way to do a self-outer-difference `v[...,:,None] −
   v[...,None,:]` for L>64 on-device** without pre-expanding both operands to L×L
   on the host? e.g. a layout/restickify pattern that lets one `g` feed both the
   row and column roles, or a `segsum`-style primitive?

2. **Is the "no mechanism to resolve stick incompatibility" limitation expected to
   be lifted** (it also blocks the cheaper inline factored form)? If it's on a
   roadmap I'd build around it differently.

3. **Better math?** The whole point of the mask is fp16 stability at large L
   (the factored 2-matmul form peaks at `exp(|total|/2)` and overflows past
   |total|≈22, which happens at L=256). Is there a numerically-stable intra-chunk
   formulation that (a) stays a matmul-friendly op sequence and (b) avoids the
   L×L self-outer-difference? Two I considered:
   - **Sub-chunk** the intra into Ls=64 blocks (each sub-block factored & safe;
     max|sub-total|=7.7 at L=256) — but the off-diagonal sub-block pairs still
     span the full |total|, so they'd need clamping too, i.e. a smaller Ls×Ls mask
     or a sub-block carried scan. More ops; and the block reshape hits the same
     4D-batched-matmul backend limit as our hierarchical scan.
   - **Per-row/col max-subtraction** (online-softmax style) — but that reduces to
     the same per-(l,s) correction, i.e. exactly the L×L mask.
   Is either the intended approach, or is there a standard trick I'm missing?

## Also (separate, already reported)

The L=256 masked kernel additionally needed the `#3381` graph-input read-copy
workaround (pre-copy operands to ComputedBuffers) to compile at all — that's the
coarse-tile / `insert_restickify` `StopIteration` regression in
`BUG_coarse_tile_restickify_stopiteration.md`.
