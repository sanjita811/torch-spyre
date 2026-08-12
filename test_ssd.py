# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mamba-2 SSD (state-space duality) kernel on Spyre, with a CPU-validated reference.

Sections below:
  * config            — shape constants (B,T,H,P,N,L); SSD_T/SSD_L env overrides.
  * reference         — ssd_reference: verbatim Mamba-2 chunked SSD (ground truth).
  * device kernels    — fused_kernel_cblock (factored intra + scanhint C-block scan +
                        combine, all one graph) and fused_kernel_masked (fp16 fallback).
  * CPU mirror        — ssd_cpu: same op sequence on CPU, to validate the formulation.
  * driver            — ssd_spyre: chunk inputs, build decay, route factored/masked,
                        compile+run; best_config/_BEST_CONFIG_BY_T pick L/bh/scan per T.
  * test              — __main__: build inputs, compare device vs reference by rel-L2.

The scan is the crux: the dense inter-chunk scan's read-copy walls the backend at C>64,
so the kernel blocks the scan's C output-row dim via spyre_hint(work_div={"C"}) (fans it
across the 32 cores, reduction whole) — letting C grow at bounded per-core memory. See
memory notes ssd_cblock_scan_breakthrough / ssd_tiling_vs_flashattn for the derivation.
"""

import dataclasses
import os

import torch
import torch.nn.functional as F
from torch_spyre._inductor import spyre_hint
try:
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )
except (ImportError, ModuleNotFoundError):
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )
from einops import rearrange, repeat

try:
    from torch.spyre import SpyreTensorLayout
except (ImportError, ModuleNotFoundError):
    from torch_spyre._C import SpyreTensorLayout

# ============================ config (Mamba-2 names) ====================
# B=batch T=seqlen H=dim P=headdim nheads=H//P N=d_state L=chunk C=T//L G=ngroups.
# T and L are env-overridable (SSD_T / SSD_L) so a C>64 config can be exercised to
# validate the C-blocked scan end-to-end (default keeps the shipped C=64 shape).
B, H, P, N = 2, 2048, 64, 128
T = int(os.environ.get("SSD_T", "4096"))
L = int(os.environ.get("SSD_L", "64"))
nheads = H // P
G = 1
C = T // L

# BH tile count for spyre_hint (cores); prefer 16, fall to 8/4/2/1 by divisibility.
_bh = B * nheads
BH_TILES = next(t for t in [16, 8, 4, 2, 1] if _bh % t == 0)


# ======================= reference (CPU, ground truth) ==================
def segsum(x):
    """Stable segment sum (verbatim, Mamba ssd_minimal.py)."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_reference(X, A, B, C, block_len, initial_states=None):
    """Reference SSD = ssd_minimal_discrete (verbatim, state-spaces/mamba)"""
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


# ========================= Spyre device kernels =========================
def _declare_dims(n_chunks=C, chunk_len=L):
    """Register the named dims before each compile (the registry resets per run)."""
    for name, size in [
        ("L", chunk_len), ("La", chunk_len), ("Lk", chunk_len),
        ("P", P), ("N", N), ("One", 1),
        ("BH", B * nheads), ("C", n_chunks), ("Ca", n_chunks), ("Cp", n_chunks + 1),
        ("PN", P * N),
    ]:
        declare_tensor_dim(name, size)


# Chunk-size / tiling policy lives in ssd_config.py (pure arithmetic, no torch).
# ssd_spyre; INTRA_FACTORED_TOTAL_LIMIT is the fp16 guard threshold reused below.
from ssd_config import (  # noqa: E402
    INTRA_FACTORED_TOTAL_LIMIT,
    MAX_FLAT_SCAN_CHUNKS,
    SSDConfig,
    pick_config,
)


# Measured-best per T (B2/H2048/P64/N128), from the (L × bh_tiles) device sweep on the
# integrated scanhint C-block kernel (warm via ssd_bench, peak=max HBM). Smaller-L/larger-C
# wins; bh_tiles=32 wins on memory (~half bh16's peak) and ~ties speed at every T — and the
# scanhint kernel's simpler graph lets bh=32 compile at C>=256 (the old for-loop kernel
# timed out there, forcing bh=16). bh=32 cold-compile is ~200-470s (vs bh16 ~55s), a
# one-time cost. Numbers (warm ms / peak GB):
#   4096  L64/C64/bh32   432ms 0.02G | 8192  L64/C128/bh32   951ms 0.04G
#   16384 L128/C128/bh32 2053ms 0.11G | 32768 L128/C256/bh32 3885ms 0.23G
#   65536 L128/C512/bh32 7918ms 0.50G
_BEST_CONFIG_BY_T = {
    4096:  SSDConfig(L=64,  bh_tiles=32, scan_mode="flat"),
    8192:  SSDConfig(L=64,  bh_tiles=32, scan_mode="cblock"),
    16384: SSDConfig(L=128, bh_tiles=32, scan_mode="cblock"),
    32768: SSDConfig(L=128, bh_tiles=32, scan_mode="cblock"),
    65536: SSDConfig(L=128, bh_tiles=32, scan_mode="cblock"),
}


def best_config(T_, B_=None, H_=None, P_=None, N_=None):
    """Measured-best SSDConfig for ``T_`` on the swept B2/H2048/P64/N128 shape;
    falls back to analytic ``pick_config`` for other T or shapes."""
    default_shape = (B_, H_, P_, N_) in ((None, None, None, None), (2, 2048, 64, 128))
    cfg = _BEST_CONFIG_BY_T.get(T_) if default_shape else None
    if cfg is not None:
        return cfg
    return pick_config(B_ or B, T_, H_ or H, P_ or P, N_ or N)


def build_decay_matrix(decay_before, decay_cumsum, final_arg, strict_mask):
    """Build the (BH, C+1, C) scan decay-matrix on device.

    Inputs carry the stick on the BH dim so the C-broadcast in the
    outer-difference is off the stick.
    """
    outer = decay_before.unsqueeze(-1) - decay_cumsum.unsqueeze(-2)      # (BH, C, C)
    decay_run = torch.exp(torch.clamp(outer, max=0.0)) * strict_mask     # (BH, C, C)
    decay_final = torch.exp(final_arg).unsqueeze(1)                      # (BH, 1, C)
    return torch.cat([decay_run, decay_final], dim=1)                    # (BH, C+1, C)


def build_intra_decay(g_row, g_col, causal_mask):
    """(BH,C,L,L) bounded intra-chunk decay mask (FALLBACK path). Asymmetric: g_row
    is host-pre-expanded to (BH,C,L,La), g_col stays (BH,C,La) and broadcasts here
    (feasible on the column side; the full self-outer-difference is not)."""
    outer = g_row - g_col.unsqueeze(-2)                   # (BH,C,L,La) - (BH,C,1,La)
    return torch.exp(torch.clamp(outer, max=0.0)) * causal_mask


# Block size for the C-row-blocked scan (whole 64-element sticks). The factored kernel
# always uses the C-block form: at C<=64 (NCB=1) it degenerates to a single block == the
# dense scan, so there is no separate flat kernel. CBLOCK env kept only as a manual override.
CBLOCK = int(os.environ.get("CBLOCK", "64"))               # rows per block, whole sticks


def fused_kernel_cblock(a, cumsum_tri, c_proj, b_proj, causal_mask, x,
                        decay_run, decay_final, init_state=None, init_col=None):
    """Factored intra + inter-chunk scan + off-diagonal combine in ONE fused graph.

    INTRA (factored): reconstruct the per-chunk decay L[i,s]=exp(g_i-g_s) as an OUTER
    PRODUCT of exp(shifted_i)·exp(-shifted_s) folded into C and B, so no (L,L) matrix is
    materialized. shifted = g - total/2 centers the exponent to halve the fp16 range
    (safe while |chunk_decay| < INTRA_FACTORED_TOTAL_LIMIT — else ssd_spyre picks the
    masked kernel).

    SCAN: a single dense matmul over the C RUN rows ``decay_run(BH,C,C) @ cs(BH,C,N*P)``
    wrapped in ``spyre_hint(work_div={"C": NCB})`` — work_div is a PER-OP core split (no
    loop-nest-contiguity constraint, unlike coarse-tile), so it fans the scan's C
    OUTPUT-ROW dim across the 32 cores while the reduction (C) and N*P stay whole per core
    — each core touches (C/NCB, N*P), staying under the 256MB span at C up to 512. Using
    only the C run rows (not the Cp=C+1 combined matrix) keeps the scan output (BH,C,N*P)
    off the span wall the extra final row would trip; the final-state row is a separate
    small ``decay_final(BH,1,C) @ cs`` matmul in the SAME graph.

    Returns ``(y_grouped, scan_final)`` from one compile — no separate final-state kernel,
    no 134MB chunk_states host transfer. Replaces the earlier per-block Python for-loop
    (retired after merged main #3530/#3612 lifted the work_div restickify wall);
    device-verified ~8-15x faster than the loop across C=128..512. See memory
    ssd_tiling_vs_flashattn.
    """
    bh_c = B * nheads
    c = decay_run.shape[-1]                                 # = n_chunks (== C dim)
    ncb = max(1, c // CBLOCK)                               # C output-row core-split factor
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        # Pre-copy graph inputs to ComputedBuffers (#3381: raw-input slices/matmuls
        # otherwise restickify badly and abort codegen).
        a_c = a * 1.0
        c_c = c_proj * 1.0
        b_c = b_proj * 1.0
        x_c = x * 1.0
        dr = decay_run * 1.0                                # (BH, C, C)
        df = decay_final * 1.0                              # (BH, 1, C)

        # --- intra (factored) ---
        intra_cumsum = torch.matmul(a_c, cumsum_tri)                  # (BH,C,L) = g
        total = a_c.sum(dim=-1, keepdim=True)                        # (BH,C,1)
        shifted = intra_cumsum - 0.5 * total
        half_tot = torch.exp(0.5 * total).unsqueeze(-1)              # (BH,C,1,1)
        c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)            # (BH,C,L,N)
        b_scaled_t = (b_c * torch.exp(-shifted).unsqueeze(-1)).transpose(-1, -2)
        attn = torch.matmul(c_scaled, b_scaled_t) * causal_mask
        y_diag = torch.matmul(attn, x_c)                             # (BH,C,L,P)
        chunk_states = torch.matmul(b_scaled_t, x_c * half_tot)      # (BH,C,N,P)
        cs_np = chunk_states.reshape(bh_c, c, N * P)                 # (BH,C,N*P)

        # --- inter-chunk scan (core-split over the C output rows) + combine ---
        with spyre_hint(work_div={"C": ncb}):
            scan = torch.matmul(dr, cs_np)                           # (BH, C, N*P) run rows
        if init_state is not None:
            scan = scan + init_col[:, :c] * init_state
        rolled = scan.reshape(bh_c, c, N, P)
        y_grouped = torch.matmul(c_scaled, rolled) * half_tot + y_diag

        # --- final-state row (separate small matmul, same graph) ---
        scan_final = torch.matmul(df, cs_np)                         # (BH, 1, N*P)
        if init_state is not None:
            scan_final = scan_final + init_col[:, c:c + 1] * init_state
    return y_grouped, scan_final


def fused_kernel_masked(a, cumsum_tri, c_proj, b_proj, decay_intra, x, decay_matrix,
                        init_state=None, init_col=None):
    """fp16-overflow fallback: precomputed (BH,C,L,L) intra-decay mask (built by
    ``build_intra_decay``) instead of the factored exp(±shifted) form. Unconditionally
    fp16-safe; chosen at runtime by ``ssd_spyre`` only when the factored fp16 guard
    (max|chunk_decay| < INTRA_FACTORED_TOTAL_LIMIT) fails. Keeps the plain DENSE scan (decay_matrix @
    chunk_states over the full C+1 rows) — NOT the scanhint C-split of the factored path;
    it only runs at small C where the dense scan is within the span limit.
    """
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        # Pre-copy graph inputs to ComputedBuffers (#3381 workaround).
        a_c = a * 1.0
        c_c = c_proj * 1.0
        b_c = b_proj * 1.0
        x_c = x * 1.0
        intra_cumsum = torch.matmul(a_c, cumsum_tri)                    # (BH, C, L)
        total = a_c.sum(dim=-1, keepdim=True)                          # (BH, C, 1)
        decay_to_end = torch.exp(total - intra_cumsum)                  # (BH, C, L)
        attn = torch.matmul(c_c, b_c.transpose(-1, -2)) * decay_intra
        y_diag = torch.matmul(attn, x_c)                                # (BH, C, L, P)
        b_decayed = b_c * decay_to_end.unsqueeze(-1)
        chunk_states = torch.matmul(b_decayed.transpose(-1, -2), x_c)   # (BH, C, N, P)

        bh, c, n, p = chunk_states.shape
        scan_out = torch.matmul(decay_matrix, chunk_states.reshape(bh, c, n * p))
        if init_state is not None:
            scan_out = scan_out + init_col * init_state               # (BH,C+1,N·P)
        rolled_states = scan_out[:, :c].reshape(bh, c, n, p)

        y_off = torch.matmul(c_c, rolled_states) * torch.exp(intra_cumsum).unsqueeze(-1)
        return y_off + y_diag, scan_out


# ===================== CPU mirror (formulation validator) ===============
def ssd_cpu(x, a, b_proj, c_proj, initial_states=None, dtype=torch.float32):
    """CPU mirror of ``ssd_spyre`` in ONE self-contained function — an INDEPENDENT oracle
    that re-derives the whole SSD (chunk, decay-matrix, intra factored/masked, scan+combine,
    final-state) inline, sharing no code with the device path so a bug can't hide in both.
    Same op sequence, routing, C-padding, and rank-1 init as ssd_spyre."""
    batch, heads, n_chunks, chunk_len, head_dim = x.shape
    state_dim = b_proj.shape[-1]
    n_bh = batch * heads

    # --- chunk + C-pad to a whole 64-stick (padded chunks contribute 0) ---
    x_flat = x.reshape(n_bh, n_chunks, chunk_len, head_dim).contiguous()
    a_flat = a.reshape(n_bh, n_chunks, chunk_len).contiguous()
    b_flat = b_proj.reshape(n_bh, n_chunks, chunk_len, state_dim).contiguous()
    c_flat = c_proj.reshape(n_bh, n_chunks, chunk_len, state_dim).contiguous()
    elem_stick = 64
    c_real = n_chunks
    if n_chunks % elem_stick != 0:
        n_chunks = ((n_chunks + elem_stick - 1) // elem_stick) * elem_stick
        pad = n_chunks - c_real
        x_flat = F.pad(x_flat, (0, 0, 0, 0, 0, pad))
        a_flat = F.pad(a_flat, (0, 0, 0, pad))
        b_flat = F.pad(b_flat, (0, 0, 0, 0, 0, pad))
        c_flat = F.pad(c_flat, (0, 0, 0, 0, 0, pad))

    assert bool((a_flat <= 1e-6).all()), "SSD kernel requires A ≤ 0"

    chunk_decay = a_flat.float().sum(-1)                               # (BH, C)
    x_d = x_flat.to(dtype); a_d = a_flat.to(dtype)
    b_d = b_flat.to(dtype); c_d = c_flat.to(dtype)
    cumsum_tri = torch.triu(torch.ones(chunk_len, chunk_len, dtype=dtype))

    # --- scan decay-matrix (BH, C+1, C): run rows exp(clamp(before_i - cumsum_s)) *
    # strict-lower, plus the final-state row exp(total - cumsum) ---
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)                   # (BH, C)
    decay_before = decay_cumsum - chunk_decay                         # exclusive cumsum
    strict = torch.tril(torch.ones(n_chunks, n_chunks, dtype=torch.bool), -1)
    outer = decay_before.unsqueeze(-1) - decay_cumsum.unsqueeze(-2)    # (BH, C, C)
    decay_run = torch.exp(torch.clamp(outer, max=0.0)) * strict.to(outer.dtype)
    decay_final = torch.exp(decay_cumsum[:, -1:] - decay_cumsum).reshape(n_bh, 1, n_chunks)
    decay_matrix = torch.cat([decay_run, decay_final], dim=1).to(dtype)  # (BH, C+1, C)

    # --- optional non-zero initial state → rank-1 scan correction ---
    init_state = init_col = None
    if initial_states is not None:
        init_col = torch.exp(
            torch.cat([decay_before, decay_cumsum[:, -1:]], dim=1)
        ).unsqueeze(-1).to(dtype)                                     # (BH, C+1, 1)
        init_state = initial_states.reshape(
            n_bh, head_dim, state_dim).transpose(-1, -2).reshape(
            n_bh, 1, state_dim * head_dim).contiguous().to(dtype)     # (BH, 1, N·P)

    # --- intra: factored (fp16-safe) fast path, or masked (BH,C,L,L) fallback ---
    intra_cumsum = torch.matmul(a_d, cumsum_tri)                      # (BH,C,L) = g
    total = a_d.sum(dim=-1, keepdim=True)                            # (BH,C,1)
    if float(chunk_decay.abs().max()) < INTRA_FACTORED_TOTAL_LIMIT:
        # mirrors fused_kernel_cblock: outer-product decay folded into C and B
        shifted = intra_cumsum - 0.5 * total
        c_scaled = c_d * torch.exp(shifted).unsqueeze(-1)
        b_scaled_t = (b_d * torch.exp(-shifted).unsqueeze(-1)).transpose(-1, -2)
        causal_mask = torch.tril(torch.ones(chunk_len, chunk_len, dtype=dtype))
        y_diag = torch.matmul(torch.matmul(c_scaled, b_scaled_t) * causal_mask, x_d)
        chunk_states = torch.matmul(b_scaled_t, x_d) * torch.exp(0.5 * total).unsqueeze(-1)
    else:
        # mirrors fused_kernel_masked: precomputed (BH,C,L,L) bounded decay mask
        g = a_flat.float().cumsum(-1)
        causal = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool))
        outer_l = g.unsqueeze(-1).expand(
            n_bh, n_chunks, chunk_len, chunk_len) - g.unsqueeze(-2)   # (BH,C,L,La)
        decay_intra = (torch.exp(torch.clamp(outer_l, max=0.0)) * causal.to(g.dtype)).to(dtype)
        y_diag = torch.matmul(torch.matmul(c_d, b_d.transpose(-1, -2)) * decay_intra, x_d)
        b_decayed = b_d * torch.exp(total - intra_cumsum).unsqueeze(-1)
        chunk_states = torch.matmul(b_decayed.transpose(-1, -2), x_d)

    # --- inter-chunk scan + off-diagonal combine ---
    bh, c, n, p = chunk_states.shape
    scan_out = torch.matmul(decay_matrix, chunk_states.reshape(bh, c, n * p))
    if init_state is not None:
        scan_out = scan_out + init_col * init_state
    rolled_states = scan_out[:, :c].reshape(bh, c, n, p)
    y_grouped = torch.matmul(c_d, rolled_states) * torch.exp(intra_cumsum).unsqueeze(-1) + y_diag

    # --- final state = scan row after the real chunks; un-fold y to (B,T,H,P) ---
    final_state = (
        scan_out[:, c_real].reshape(batch, heads, state_dim, head_dim).permute(0, 1, 3, 2)
    )
    y = y_grouped.reshape(batch, heads, n_chunks, chunk_len, head_dim)
    y = y[:, :, :c_real].permute(0, 2, 3, 1, 4).contiguous()
    return y.reshape(batch, c_real * chunk_len, heads, head_dim).half(), final_state.half()


_device_const_cache = {}


def _device_const(key, build):
    """Build a data-independent constant on host once, move to Spyre, and reuse."""
    tensor = _device_const_cache.get(key)
    if tensor is None:
        tensor = _device_const_cache[key] = build().to("spyre")
    return tensor


# ============================== Spyre driver ============================
def ssd_spyre(x, a, b_proj, c_proj, initial_states=None, config=None):
    """Run the SSD core on Spyre in ONE fused kernel. Inputs (B, nheads, C, L, *)."""
    global BH_TILES, CBLOCK
    if config is not None:
        BH_TILES = config.bh_tiles
        CBLOCK = config.cblock_size
    batch, heads, n_chunks, chunk_len, head_dim = x.shape
    state_dim = b_proj.shape[-1]
    n_bh = batch * heads

    x_flat = x.reshape(n_bh, n_chunks, chunk_len, head_dim).contiguous()
    a_flat = a.reshape(n_bh, n_chunks, chunk_len).contiguous()
    b_flat = b_proj.reshape(n_bh, n_chunks, chunk_len, state_dim).contiguous()
    c_flat = c_proj.reshape(n_bh, n_chunks, chunk_len, state_dim).contiguous()

    elem_stick = 64
    c_real = n_chunks
    if n_chunks % elem_stick != 0:
        c_pad = ((n_chunks + elem_stick - 1) // elem_stick) * elem_stick
        pad = c_pad - n_chunks
        x_flat = F.pad(x_flat, (0, 0, 0, 0, 0, pad))                   # pad chunk dim
        a_flat = F.pad(a_flat, (0, 0, 0, pad))
        b_flat = F.pad(b_flat, (0, 0, 0, 0, 0, pad))
        c_flat = F.pad(c_flat, (0, 0, 0, 0, 0, pad))
        n_chunks = c_pad

    # Per-chunk total decay (BH, C): the only host reduction; feeds the scan
    # decay-matrix build. Within-chunk g is computed on-device in the kernel.
    chunk_decay = a_flat.float().sum(-1)                               # (BH, C)

    x_dev, a_dev, b_dev, c_dev = (t.to("spyre") for t in (x_flat, a_flat, b_flat, c_flat))
    cumsum_tri = _device_const(
        f"cumsum_tri_{chunk_len}",
        lambda: torch.triu(torch.ones(chunk_len, chunk_len, dtype=torch.float16)),
    )
    causal_mask = _device_const(
        f"causal_intra_{chunk_len}",
        lambda: torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.float16)),
    )

    # --- scan decay-matrix (BH, C+1, C), built on device. Inputs carry the stick on the
    # BH dim so the C-broadcast in the outer-difference is off the stick. ---
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)                    # (BH, C)
    decay_before = decay_cumsum - chunk_decay                          # exclusive cumsum
    final_arg = (decay_cumsum[:, -1:] - decay_cumsum).half()           # (BH, C), all <= 0
    mask_dev = _device_const(
        f"strict_mask_{n_chunks}",
        lambda: torch.tril(
            torch.ones(n_chunks, n_chunks, dtype=torch.float16), -1
        ).reshape(1, n_chunks, n_chunks),
    )
    bh_stick = SpyreTensorLayout(
        [n_bh, n_chunks], [n_chunks, 1], torch.float16, [1, 0]  # dim_order [1,0] -> stick on BH
    )
    _declare_dims(n_chunks)
    decay_matrix = torch.compile(build_decay_matrix, dynamic=False)(
        decay_before.half().to(device_layout=bh_stick),
        decay_cumsum.half().to(device_layout=bh_stick),
        final_arg.to(device_layout=bh_stick),
        mask_dev,
    )                                                                   # (BH, C+1, C)

    # Optional non-zero initial state → rank-1 scan correction. init_col (BH,C+1,1) is
    # exp([decay_before ; decay_total]) (all ≤0 → ∈(0,1], fp16-safe); propagates the
    # initial state into every scan row.
    init_state = init_col = None
    if initial_states is not None:
        init_col = torch.exp(
            torch.cat([decay_before, decay_cumsum[:, -1:]], dim=1)
        ).unsqueeze(-1).half().to("spyre")                            # (BH, C+1, 1)
        init_state = (
            initial_states.reshape(n_bh, head_dim, state_dim)          # (BH, P, N)
            .transpose(-1, -2).reshape(n_bh, 1, state_dim * head_dim)  # (BH, 1, N·P)
            .contiguous().half().to("spyre")
        )

    # Factored intra is fp16-safe iff max|chunk_decay| < the limit; else masked fallback.
    factored = float(chunk_decay.abs().max()) < INTRA_FACTORED_TOTAL_LIMIT

    # Names must be re-declared right before EACH compile (registry resets per compile).
    used_cblock = factored
    if factored:
        # Split the decay matrix on the host into run rows (BH,C,C) and the final row
        # (BH,1,C) as separate device inputs; the kernel scans the run rows (hinted) and
        # does the final row as a separate matmul in the same graph. Host slice keeps the
        # storage_offset (device slice of the custom-layout decay_matrix drops it).
        decay_run = (decay_matrix[:, :n_chunks, :] * 1.0).contiguous()
        decay_final = decay_matrix.cpu()[:, n_chunks:, :].contiguous().to("spyre")
        _declare_dims(n_chunks, chunk_len)
        name_tensor_dims(a_dev, ["BH", "C", "Lk"]); name_tensor_dims(cumsum_tri, ["Lk", "L"])
        name_tensor_dims(c_dev, ["BH", "C", "L", "N"]); name_tensor_dims(b_dev, ["BH", "C", "La", "N"])
        name_tensor_dims(x_dev, ["BH", "C", "La", "P"])
        name_tensor_dims(decay_run, ["BH", "C", "Ca"])
        name_tensor_dims(decay_final, ["BH", "One", "Ca"])
        name_tensor_dims(causal_mask, ["L", "La"])
        if init_state is not None:
            name_tensor_dims(init_col, ["BH", "Cp", "One"])
            name_tensor_dims(init_state, ["BH", "One", "PN"])
        y_grouped, scan_out = torch.compile(fused_kernel_cblock, dynamic=False)(
            a_dev, cumsum_tri, c_dev, b_dev, causal_mask, x_dev, decay_run, decay_final,
            init_state, init_col
            )
    else:
        # Robust fallback: build the (BH,C,L,L) intra-decay mask, then the masked core.
        # Host pre-expands only g_row (the self-outer-difference is backend-blocked at
        # L>64); g_col stays (BH,C,La) and broadcasts in-kernel.
        g = a_flat.float().cumsum(-1)                                  # (BH,C,L)
        g_row_d = g.unsqueeze(-1).expand(
            n_bh, n_chunks, chunk_len, chunk_len).contiguous().half().to("spyre")
        g_col_d = g.half().to("spyre")                                 # (BH,C,La)
        _declare_dims(n_chunks, chunk_len)
        name_tensor_dims(g_row_d, ["BH", "C", "L", "La"])
        name_tensor_dims(g_col_d, ["BH", "C", "La"])
        decay_intra_dev = torch.compile(build_intra_decay, dynamic=False)(
            g_row_d, g_col_d, causal_mask)
        _declare_dims(n_chunks, chunk_len)
        name_tensor_dims(a_dev, ["BH", "C", "Lk"]); name_tensor_dims(cumsum_tri, ["Lk", "L"])
        name_tensor_dims(c_dev, ["BH", "C", "L", "N"]); name_tensor_dims(b_dev, ["BH", "C", "La", "N"])
        name_tensor_dims(decay_intra_dev, ["BH", "C", "L", "La"])
        name_tensor_dims(x_dev, ["BH", "C", "La", "P"])
        name_tensor_dims(decay_matrix, ["BH", "Cp", "Ca"])
        if init_state is not None:
            name_tensor_dims(init_col, ["BH", "Cp", "One"])
            name_tensor_dims(init_state, ["BH", "One", "PN"])
        y_grouped, scan_out = torch.compile(fused_kernel_masked, dynamic=False)(
            a_dev, cumsum_tri, c_dev, b_dev, decay_intra_dev, x_dev, decay_matrix,
            init_state, init_col
            )
    # Final state = scan row after the real chunks (padded rows equal it, since
    # padded chunk_states are 0). Row index c_real works whether or not we padded.
    # The C-blocked kernel already returns ONLY that final row (shape (BH,1,NP)).
    final_row = scan_out[:, 0] if used_cblock else scan_out[:, c_real]
    final_state = (
        final_row.cpu()
        .reshape(batch, heads, state_dim, head_dim)
        .permute(0, 1, 3, 2)
    )

    # Un-fold (BH, C_pad, L, P) -> (B, T, H, P), dropping any padded chunks.
    y = y_grouped.cpu().reshape(batch, heads, n_chunks, chunk_len, head_dim)
    y = y[:, :, :c_real]                                               # drop C-padding
    y = y.permute(0, 2, 3, 1, 4).contiguous()
    return y.reshape(batch, c_real * chunk_len, heads, head_dim).half(), final_state.half()


# ================================= test =================================
def rel_l2(got, ref):
    """Relative L2 (norm-based) error — the acceptance metric for this kernel."""
    got, ref = got.float(), ref.float()
    return (got - ref).norm().item() / (ref.norm().item() + 1e-12)


if __name__ == "__main__":
    torch.manual_seed(42)
    x_raw = torch.randn(B, T, nheads, P)                   # pre-discretization input
    dt_bias = torch.randn(nheads) * 0.1                    # per-head learnable bias
    dt_min, dt_max = 0.0, float("inf")                     # Mamba-2 default dt_limit (no clamp)
    dt = F.softplus(torch.randn(B, T, nheads) - 4 + dt_bias).clamp(dt_min, dt_max)
    a_log = -torch.exp(torch.rand(nheads))                 # A = -exp(A_log), per-head
    b_grp = torch.randn(B, T, G, N)
    c_grp = torch.randn(B, T, G, N)
    b_raw = b_grp.repeat_interleave(nheads // G, dim=2)                 # (B, T, nheads, N)
    c_raw = c_grp.repeat_interleave(nheads // G, dim=2)                 # (B, T, nheads, N)
    a_raw = dt * a_log                                     # discretized A = dt*A
    x_dt = x_raw * dt.unsqueeze(-1)                        # dt-scaled SSM input (ZOH)

    # Config drives the run: best_config(T) picks L/scan_mode/tiling. An explicit SSD_L
    # overrides cfg.L (manual sweep); otherwise cfg.L (compute-optimal, C-block for C>64).
    cfg = best_config(T)
    if "SSD_L" in os.environ:                  # manual L override: re-pick to match
        chunk_len = L
        cfg = dataclasses.replace(
            cfg, L=chunk_len,
            scan_mode="cblock" if T // chunk_len > MAX_FLAT_SCAN_CHUNKS else "flat")
    else:
        chunk_len = cfg.L
    print(f"config: T = {T}, L = {chunk_len}, C = {T // chunk_len}, scan = {cfg.scan_mode}, "
          f"bh_tiles = {cfg.bh_tiles}")

    y_ref, final_ref = ssd_reference(x_dt, a_raw, b_raw, c_raw, chunk_len)

    # --- chunk (B,T,nheads,*) -> (B,nheads,C,L,*); b/c already per-head expanded ---
    nc = T // chunk_len                                                  # C = T / L
    xd_c = x_dt.half().reshape(B, nc, chunk_len, nheads, P).permute(0, 3, 1, 2, 4).contiguous()
    a_c = a_raw.half().reshape(B, nc, chunk_len, nheads).permute(0, 3, 1, 2).contiguous()
    b_c = b_raw.half().reshape(B, nc, chunk_len, nheads, N).permute(0, 3, 1, 2, 4).contiguous()
    c_c = c_raw.half().reshape(B, nc, chunk_len, nheads, N).permute(0, 3, 1, 2, 4).contiguous()

    # --- CPU mirror: validate the FORMULATION (no card needed) ---
    print("Validating CPU mirror (same ops as Spyre)...")
    y_cpu32, fin_cpu32 = ssd_cpu(xd_c, a_c, b_c, c_c, dtype=torch.float32)
    ey32, ef32 = rel_l2(y_cpu32, y_ref), rel_l2(fin_cpu32, final_ref)
    print(f"  fp32 formulation:  Y={ey32:.6f}  final={ef32:.6f}")
    assert ey32 < 1e-3, f"fp32 formulation Y rel-L2 {ey32:.2e} — math diverges from reference"
    y_cpu16, fin_cpu16 = ssd_cpu(xd_c, a_c, b_c, c_c, dtype=torch.float16)
    ey16, ef16 = rel_l2(y_cpu16, y_ref), rel_l2(fin_cpu16, final_ref)
    print(f"  fp16 numeric floor: Y={ey16:.4f}  final={ef16:.4f}")
    print("PASSED (CPU mirror vs Mamba reference)")

    print("Running Spyre SSD kernel (fused)...")
    y_spyre, final_spyre = ssd_spyre(xd_c, a_c, b_c, c_c, config=cfg)
    err_y = rel_l2(y_spyre.cpu(), y_ref)
    err_final = rel_l2(final_spyre.cpu(), final_ref)
    print(f"  Y            rel-L2 error = {err_y:.4f}")
    print(f"  final_state  rel-L2 error = {err_final:.4f}")
    assert err_y < 0.05, f"Y rel-L2 {err_y:.4f} exceeds 0.05 (fp16 budget)"
    assert err_final < 0.05, f"final_state rel-L2 {err_final:.4f} exceeds 0.05"
    print("PASSED (Spyre vs Mamba reference, fp16 relative-L2 tolerance)")
