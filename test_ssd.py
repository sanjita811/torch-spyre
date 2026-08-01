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

import os

import torch
import torch.nn.functional as F
from torch_spyre._inductor import spyre_hint
try:
    # Post-WSR-refactor (#3293) location.
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
# L is the chunk size. On Spyre the optimum is SMALL (opposite of GPU Mamba's 256):
# the kernel is ~280x memory-bound, so the O(L²) intra-chunk attn intermediate — not
# the O(C²) scan — dominates once C is small. Measured device wall-clock (isolated
# bench, 2026-08-01, merged main f4fab17), B2/T4096/N128:
#   L=64/C=64  = 928ms (Y 0.0039)   <-- optimum for T=4096
#   L=128/C=32 = 1777ms (Y 0.0045)
#   L=256/C=16 = 4402ms (Y 0.0057)
# So default L=64 here. At LONG T the picture flips via a CORRECTNESS+compile wall,
# not speed: T=16384 needs C≤64 (T16384/L64/C256 => Y=0.59 GARBAGE, dense-scan fp16
# saturation; T16384/L128/C128 => dxp SIGABRT "immediate out of boundary"; only
# T16384/L256/C64 => Y=0.0057 compiles+correct). pick_config (ssd_config.py) encodes
# exactly this — grow L just enough to keep C≤MAX_FLAT_SCAN_CHUNKS(=64). All its
# picks were re-validated against these device measurements. Use pick_config for any
# non-default shape (see validate_long_t); the hardcoded L below is only the T=4096 demo.
B, T, H, P, N, L = 2, 4096, 2048, 64, 128, 64
nheads = H // P
G = 1                                
C = T // L

# BH tile count for spyre_hint (cores). 
_bh = B * nheads
BH_TILES = next(t for t in ([16, 8, 4, 2, 1] if L > 64 else [4, 2, 1]) if _bh % t == 0)

BUILD_DECAY_ON_DEVICE = True          # scan decay-matrix on device (vs host fallback)


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
        ("PN", P * N), ("G2", B * nheads * n_chunks),
    ]:
        declare_tensor_dim(name, size)


# Chunk-size / tiling policy lives in ssd_config.py (pure arithmetic, no torch).
# ssd_spyre; INTRA_FACTORED_TOTAL_LIMIT is the fp16 guard threshold reused below.
from ssd_config import (  # noqa: E402
    INTRA_FACTORED_TOTAL_LIMIT,
    SSDConfig,
    pick_config,
)


def _intra_decay_factored_safe(chunk_decay):
    """True iff the FACTORED intra-decay stays within fp16 range for every row."""
    return float(chunk_decay.abs().max()) < INTRA_FACTORED_TOTAL_LIMIT


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
    """(BH,C,L,L) bounded intra-chunk decay mask (FALLBACK path). ASYMMETRIC layout:
    ``g_row`` is PRE-EXPANDED to (BH,C,L,La) on host but ``g_col`` stays (BH,C,La)
    and is broadcast in-kernel via ``unsqueeze(-2)``. Verified on device at L=256
    (rel 1.3e-3) — the in-kernel broadcast is layout-feasible on the COLUMN side
    (off the last stick dim) even though the full self-outer-difference is not.
    Halves the host pre-expand transfer vs expanding both (134MB not 268MB @L256).
    clamp(max=0) keeps weights in (0,1] → unconditionally fp16-safe at any L."""
    outer = g_row - g_col.unsqueeze(-2)                   # (BH,C,L,La) - (BH,C,1,La)
    return torch.exp(torch.clamp(outer, max=0.0)) * causal_mask


def fused_kernel(a, cumsum_tri, c_proj, b_proj, causal_mask, x, decay_matrix,
                 init_state=None, init_col=None):
    """intra + scan + combine in ONE compiled kernel, batched over (BH, C).
    FACTORED intra-decay fast path; ``init_state``/``init_col`` optionally fold a
    non-zero initial state in as a rank-1 scan correction.
    """
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        # WORKAROUND (#3381 regression): a coarse-tiled op that reads a GRAPH INPUT
        # directly routes through a coarse_tile_read_copy buffer that
        # insert_restickify can't anchor → StopIteration. Pre-copy every graph-input
        # operand to a ComputedBuffer first (the *_c copies below), so no tiled op
        # reads a raw graph input — the pattern flash-attn uses (key*scale). Remove
        # when the backend handles the read-copy buffer. See
        # BUG_coarse_tile_restickify_stopiteration.md.
        a_c = a * 1.0
        c_c = c_proj * 1.0
        b_c = b_proj * 1.0
        x_c = x * 1.0
        # --- intra-chunk ---
        intra_cumsum = torch.matmul(a_c, cumsum_tri)                    # (BH, C, L) = g
        total = a_c.sum(dim=-1, keepdim=True)                          # (BH, C, 1)
        shifted = intra_cumsum - 0.5 * total                           # (BH, C, L)
        c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)             # (BH,C,L,N)
        b_scaled = b_c * torch.exp(-shifted).unsqueeze(-1)           # (BH,C,La,N)
        b_scaled_t = b_scaled.transpose(-1, -2)                        # (BH,C,N,La) SHARED
        attn = torch.matmul(c_scaled, b_scaled_t) * causal_mask
        y_diag = torch.matmul(attn, x_c)                                # (BH, C, L, P)
        chunk_states = torch.matmul(b_scaled_t, x_c) * torch.exp(0.5 * total).unsqueeze(-1)

        # --- inter-chunk scan (dense O(C²) matmul) ---
        # BACKEND-BLOCKED at the sub-quadratic alternative: a hierarchical O(C^1.5)
        # scan would block-reshape (BH,C,NP)<->(BH,nb,K,NP) around a 4D block matmul.
        # On merged main (f4fab17) the 4D block MATMUL now compiles (probe_blockscan.py,
        # rel 1.7e-3) but the block RESHAPE does NOT: the in-kernel C-split errors
        # "reshape split a named dim, re-annotate after the reshape" and the merge-back
        # is silently WRONG (rel 1.11, probe_blockreshape.py). A carried O(C) recurrence
        # is also blocked (indexing named C at a fixed coord). So this DENSE scan is the
        # only working form; C is kept ≤64 by pick_config to stay within its span limit
        # (see ssd_config.MAX_FLAT_SCAN_CHUNKS). BACKEND ASK: re-annotate named dims
        # across a splitting reshape → unlocks the hierarchical scan for extreme T.
        bh, c, n, p = chunk_states.shape          # derive from shapes, not module globals
        scan_out = torch.matmul(decay_matrix, chunk_states.reshape(bh, c, n * p))
        if init_state is not None:
            scan_out = scan_out + init_col * init_state               # (BH,C+1,N·P)
        rolled_states = scan_out[:, :c].reshape(bh, c, n, p)           # (BH, C, N, P)

        # --- off-diagonal + combine ---
        y_off = torch.matmul(c_scaled, rolled_states) * torch.exp(0.5 * total).unsqueeze(-1)
        return y_off + y_diag, scan_out


def fused_kernel_masked(a, cumsum_tri, c_proj, b_proj, decay_intra, x, decay_matrix,
                        init_state=None, init_col=None):
    """Fallback: same fused kernel but with a precomputed (BH,C,L,L) intra-decay
    mask (built by ``build_intra_decay``). Unconditionally fp16-safe; used only
    when ``_intra_decay_factored_safe`` is False.
    """
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        # Pre-copy graph inputs to ComputedBuffers (#3381 workaround; see fused_kernel).
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
# ssd_cpu runs the identical op sequence on plain CPU torch
def _ssd_core_cpu(a, cumsum_tri, c_proj, b_proj, x, decay_matrix,
                  causal_mask=None, decay_intra=None, init_state=None, init_col=None,
                  scan_mode="flat", block_K=0, chunk_decay=None):
    """CPU twin of fused_kernel"""
    intra_cumsum = torch.matmul(a, cumsum_tri)                          # (BH, C, L) = g
    total = a.sum(dim=-1, keepdim=True)                                 # (BH, C, 1)
    if decay_intra is None:
        # factored fast path (mirrors fused_kernel)
        shifted = intra_cumsum - 0.5 * total
        c_scaled = c_proj * torch.exp(shifted).unsqueeze(-1)
        b_scaled = b_proj * torch.exp(-shifted).unsqueeze(-1)
        b_scaled_t = b_scaled.transpose(-1, -2)
        attn = torch.matmul(c_scaled, b_scaled_t) * causal_mask
        y_diag = torch.matmul(attn, x)
        chunk_states = torch.matmul(b_scaled_t, x) * torch.exp(0.5 * total).unsqueeze(-1)
    else:
        # masked fallback (mirrors fused_kernel_masked)
        decay_to_end = torch.exp(total - intra_cumsum)
        attn = torch.matmul(c_proj, b_proj.transpose(-1, -2)) * decay_intra
        y_diag = torch.matmul(attn, x)
        b_decayed = b_proj * decay_to_end.unsqueeze(-1)
        chunk_states = torch.matmul(b_decayed.transpose(-1, -2), x)

    bh, c, n, p = chunk_states.shape
    scan_out = torch.matmul(decay_matrix, chunk_states.reshape(bh, c, n * p))
    if init_state is not None:
        scan_out = scan_out + init_col * init_state
    rolled_states = scan_out[:, :c].reshape(bh, c, n, p)
    y_off = torch.matmul(c_proj, rolled_states) * torch.exp(intra_cumsum).unsqueeze(-1)
    return y_off + y_diag, scan_out


def _build_decay_cpu(chunk_decay, n_bh, n_chunks, dtype):
    """CPU twin of _build_decay: (BH, C+1, C) scan decay-matrix, same clamp math."""
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)
    decay_before = decay_cumsum - chunk_decay
    decay_total = decay_cumsum[:, -1:]
    strict = torch.tril(torch.ones(n_chunks, n_chunks, dtype=torch.bool), -1)
    outer = decay_before.unsqueeze(-1) - decay_cumsum.unsqueeze(-2)     # (BH, C, C)
    decay_run = torch.exp(torch.clamp(outer, max=0.0)) * strict.to(outer.dtype)
    decay_final = torch.exp(decay_total - decay_cumsum).reshape(n_bh, 1, n_chunks)
    return torch.cat([decay_run, decay_final], dim=1).to(dtype)         # (BH, C+1, C)


def _build_intra_decay_cpu(a_flat, n_bh, n_chunks, chunk_len, dtype):
    """CPU twin of build_intra_decay: (BH, C, L, L) bounded mask, same clamp math."""
    g = a_flat.float().cumsum(-1)                                       # (BH,C,L)
    causal = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool))
    outer = g.unsqueeze(-1) - g.unsqueeze(-2)                           # (BH,C,L,La)
    return (torch.exp(torch.clamp(outer, max=0.0)) * causal.to(g.dtype)).to(dtype)


def ssd_cpu(x, a, b_proj, c_proj, initial_states=None, dtype=torch.float32,
            scan_mode="flat", block_K=0):
    """CPU mirror of ``ssd_spyre`` (same ops, same routing, C-pad, rank-1 init)."""
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
        x_flat = F.pad(x_flat, (0, 0, 0, 0, 0, pad))
        a_flat = F.pad(a_flat, (0, 0, 0, pad))
        b_flat = F.pad(b_flat, (0, 0, 0, 0, 0, pad))
        c_flat = F.pad(c_flat, (0, 0, 0, 0, 0, pad))
        n_chunks = c_pad

    assert bool((a_flat <= 1e-6).all()), "SSD kernel requires A ≤ 0"

    chunk_decay = a_flat.float().sum(-1)                               # (BH, C)
    x_d = x_flat.to(dtype); a_d = a_flat.to(dtype)
    b_d = b_flat.to(dtype); c_d = c_flat.to(dtype)
    cumsum_tri = torch.triu(torch.ones(chunk_len, chunk_len, dtype=dtype))
    decay_matrix = _build_decay_cpu(chunk_decay, n_bh, n_chunks, dtype)

    init_state = init_col = None
    if initial_states is not None:
        decay_cumsum = torch.cumsum(chunk_decay, dim=-1)
        decay_before = decay_cumsum - chunk_decay
        init_col = torch.exp(
            torch.cat([decay_before, decay_cumsum[:, -1:]], dim=1)
        ).unsqueeze(-1).to(dtype)                                      # (BH, C+1, 1)
        init_state = initial_states.reshape(
            n_bh, head_dim, state_dim).transpose(-1, -2).reshape(
            n_bh, 1, state_dim * head_dim).contiguous().to(dtype)      # (BH, 1, N·P)

    cd_d = chunk_decay.to(dtype)
    if _intra_decay_factored_safe(chunk_decay):
        causal_mask = torch.tril(torch.ones(chunk_len, chunk_len, dtype=dtype))
        y_grouped, scan_out = _ssd_core_cpu(
            a_d, cumsum_tri, c_d, b_d, x_d, decay_matrix,
            causal_mask=causal_mask, init_state=init_state, init_col=init_col,
            scan_mode=scan_mode, block_K=block_K, chunk_decay=cd_d)
    else:
        decay_intra = _build_intra_decay_cpu(a_flat, n_bh, n_chunks, chunk_len, dtype)
        y_grouped, scan_out = _ssd_core_cpu(
            a_d, cumsum_tri, c_d, b_d, x_d, decay_matrix,
            decay_intra=decay_intra, init_state=init_state, init_col=init_col,
            scan_mode=scan_mode, block_K=block_K, chunk_decay=cd_d)

    final_state = (
        scan_out[:, c_real]
        .reshape(batch, heads, state_dim, head_dim)
        .permute(0, 1, 3, 2)
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


def _build_init_col(chunk_decay):
    """(BH, C+1, 1) rank-1 column that propagates a non-zero initial state into
    every scan row: exp(acs_pad) where acs_pad = [0, cumsum(chunk_decay)] padded.
    Equivalently exp([decay_before ; decay_total]). All entries ≤ 0 → col ∈ (0,1],
    fp16-safe. Built on host (tiny (BH,C+1) vector), moved to device.
    """
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)                    # (BH, C)
    decay_before = decay_cumsum - chunk_decay                          # exclusive cumsum
    decay_total = decay_cumsum[:, -1:]
    col = torch.exp(torch.cat([decay_before, decay_total], dim=1))     # (BH, C+1)
    return col.unsqueeze(-1).half().to("spyre")                        # (BH, C+1, 1)


def _build_decay(chunk_decay, n_bh, n_chunks):
    """Build the (BH, C+1, C) scan decay-matrix."""
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)                    # (BH, C)
    decay_before = decay_cumsum - chunk_decay                          # exclusive cumsum
    decay_total = decay_cumsum[:, -1:]
    if BUILD_DECAY_ON_DEVICE:
        final_arg = (decay_total - decay_cumsum).half()                # (BH, C), all <= 0
        # strict-lower mask is a pure function of shape -> cache (data-independent).
        mask_dev = _device_const(
            f"strict_mask_{n_chunks}",
            lambda: torch.tril(
                torch.ones(n_chunks, n_chunks, dtype=torch.float16), -1
            ).reshape(1, n_chunks, n_chunks),
        )
        bh_stick = SpyreTensorLayout(
            [n_bh, n_chunks], [n_chunks, 1], torch.float16, [1, 0]  # dim_order [1,0] -> stick on BH
        )
        before_dev = decay_before.half().to(device_layout=bh_stick)
        cumsum_dev = decay_cumsum.half().to(device_layout=bh_stick)
        arg_dev = final_arg.to(device_layout=bh_stick)
        _declare_dims(n_chunks)
        return torch.compile(build_decay_matrix, dynamic=False)(
            before_dev, cumsum_dev, arg_dev, mask_dev
        )                                                               # (BH, C+1, C) on device
    # all-host fallback
    strict_lower = torch.tril(torch.ones(n_chunks, n_chunks, dtype=torch.bool), -1)
    outer_diff = decay_before.unsqueeze(-1) - decay_cumsum.unsqueeze(-2)
    decay_run = torch.exp(outer_diff.masked_fill(~strict_lower, float("-inf")))
    decay_final = torch.exp(decay_total - decay_cumsum).reshape(n_bh, 1, n_chunks)
    return torch.cat([decay_run, decay_final], dim=1).half().to("spyre")


def _build_intra_decay(a_flat, chunk_len, n_bh, n_chunks):
    """(BH,C,L,L) intra-decay mask (FALLBACK path), correct for ANY L (incl. 256).

    The naive in-kernel outer-difference ``g[...,:,None] - g[...,None,:]`` needs the
    single tensor ``g`` in two incompatible stick orientations at once (L on-stick
    for one operand, La on-stick for the other). At L=64 a BH-stick layout hid this;
    at L>64 every single-tensor layout we tried either errored ("no mechanism to
    resolve stick incompatibility") or silently built a WRONG mask (rel ~1.0 at
    L=256). Fix: PRE-EXPAND both copies to the full (BH,C,L,La) shape on the host so
    the kernel does a plain elementwise ``g_row - g_col`` with NO in-kernel
    broadcast/outer-difference — each operand then has one fixed, compatible layout.
    Verified correct at L=256 (rel 1.5e-3). Cost: two full (BH,C,L,L) H2D transfers
    (bounded, and only on the masked fallback path). See ssd_design.md#precision.
    """
    g = a_flat.float().cumsum(-1)                                      # (BH,C,L)
    # ASYMMETRIC: pre-expand only the ROW operand (134MB @L256); the COLUMN stays
    # (BH,C,La) and broadcasts in-kernel via unsqueeze(-2). Verified feasible+correct
    # on device at L=256 (probe_outerdiff.py expand_row: rel 1.3e-3), whereas the
    # single-g self-outer-difference is still backend-blocked ("no mechanism to
    # resolve stick incompatibility") and the BH-stick layout silently mis-builds
    # (rel 0.61). BACKEND ASK: support the in-kernel self-outer-difference
    # v[...,:,None]-v[...,None,:] at L>64 so BOTH operands can stay (BH,C,L) (would
    # drop the remaining 134MB host pre-expand to a ~0.5MB g vector).
    g_row = g.unsqueeze(-1).expand(n_bh, n_chunks, chunk_len, chunk_len).contiguous().half()
    g_col = g.half()                                                   # (BH,C,La), no expand
    causal = _device_const(
        f"causal_intra_{chunk_len}",
        lambda: torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.float16)),
    )
    g_row_d, g_col_d = g_row.to("spyre"), g_col.to("spyre")
    _declare_dims(n_chunks, chunk_len)
    name_tensor_dims(g_row_d, ["BH", "C", "L", "La"])
    name_tensor_dims(g_col_d, ["BH", "C", "La"])
    return torch.compile(build_intra_decay, dynamic=False)(g_row_d, g_col_d, causal)


# ============================== Spyre driver ============================
def ssd_spyre(x, a, b_proj, c_proj, initial_states=None, config=None):
    """Run the SSD core on Spyre in ONE fused kernel. Inputs (B, nheads, C, L, *)."""
    global BH_TILES
    if config is not None:
        BH_TILES = config.bh_tiles
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

    decay_matrix = _build_decay(chunk_decay, n_bh, n_chunks)            # (BH, C+1, C)

    # Optional non-zero initial state → rank-1 scan correction
    init_state = init_col = None
    if initial_states is not None:
        init_col = _build_init_col(chunk_decay)                        # (BH, C+1, 1)
        init_state = (
            initial_states.reshape(n_bh, head_dim, state_dim)          # (BH, P, N)
            .transpose(-1, -2).reshape(n_bh, 1, state_dim * head_dim)  # (BH, 1, N·P)
            .contiguous().half().to("spyre")
        )

    factored = _intra_decay_factored_safe(chunk_decay)

    # --- one fused kernel: intra (incl. inline decay) + scan + combine ---
    _declare_dims(n_chunks, chunk_len)
    name_tensor_dims(a_dev, ["BH", "C", "Lk"])
    name_tensor_dims(cumsum_tri, ["Lk", "L"])
    name_tensor_dims(c_dev, ["BH", "C", "L", "N"]); name_tensor_dims(b_dev, ["BH", "C", "La", "N"])
    name_tensor_dims(x_dev, ["BH", "C", "La", "P"])
    name_tensor_dims(decay_matrix, ["BH", "Cp", "Ca"])
    if init_state is not None:
        name_tensor_dims(init_col, ["BH", "Cp", "One"])
        name_tensor_dims(init_state, ["BH", "One", "PN"])
    if factored:
        # Fast path: 2 compiles (scan-decay build + this), no L×L mask kernel.
        name_tensor_dims(causal_mask, ["L", "La"])
        y_grouped, scan_out = torch.compile(fused_kernel, dynamic=False)(
            a_dev, cumsum_tri, c_dev, b_dev, causal_mask, x_dev, decay_matrix,
            init_state, init_col
            )
    else:
        # Robust fallback: build the (BH,C,L,L) mask kernel, then the fused core.
        decay_intra_dev = _build_intra_decay(a_flat, chunk_len, n_bh, n_chunks)
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
    final_state = (
        scan_out[:, c_real].cpu()
        .reshape(batch, heads, state_dim, head_dim)
        .permute(0, 1, 3, 2)
    )

    # Un-fold (BH, C_pad, L, P) -> (B, T, H, P), dropping any padded chunks.
    y = y_grouped.cpu().reshape(batch, heads, n_chunks, chunk_len, head_dim)
    y = y[:, :, :c_real]                                               # drop C-padding
    y = y.permute(0, 2, 3, 1, 4).contiguous()
    return y.reshape(batch, c_real * chunk_len, heads, head_dim).half(), final_state.half()


# ================================= test =================================
def _chunk_inputs(x, a, b_proj, c_proj):
    """(B, T, nheads, *) raw tensors -> (B, nheads, C, L, *) chunked layout.

    b_proj/c_proj are already expanded to per-head (nheads)
    (repeated from their ngroups groups).
    """
    x = x.reshape(B, C, L, nheads, P).permute(0, 3, 1, 2, 4).contiguous()
    a = a.reshape(B, C, L, nheads).permute(0, 3, 1, 2).contiguous()
    b_proj = b_proj.reshape(B, C, L, nheads, N).permute(0, 3, 1, 2, 4).contiguous()
    c_proj = c_proj.reshape(B, C, L, nheads, N).permute(0, 3, 1, 2, 4).contiguous()
    return x, a, b_proj, c_proj


def rel_l2(got, ref):
    """Relative L2 (norm-based) error — the acceptance metric for this kernel."""
    got, ref = got.float(), ref.float()
    return (got - ref).norm().item() / (ref.norm().item() + 1e-12)


def validate_long_t(B_=1, T_=16384, H_=2048, P_=64, N_=128):
    """Validate the long-sequence policy path on device. MUST run as the only
    Spyre compile in the process (see __main__ note) — call it from a fresh
    interpreter: `python -c "import test_ssd as m; m.validate_long_t()"`.

    ``pick_config`` chooses L for this (B,T,H,P,N) (T=16K → L=128, ~14× faster than
    L=64), and the SSDConfig drives the kernel. Asserts rel-L2 < 0.05 vs reference.
    """
    cfg = pick_config(B_, T_, H_, P_, N_)
    L_ = cfg.L
    # _chunk_inputs and _declare_dims read module globals; set them for this shape.
    globals().update(B=B_, T=T_, H=H_, P=P_, N=N_, nheads=H_ // P_, G=1,
                     L=L_, C=T_ // L_, BH_TILES=cfg.bh_tiles)
    _device_const_cache.clear()
    torch.manual_seed(42)
    nheads_ = H_ // P_
    xr = torch.randn(B_, T_, nheads_, P_)
    dt = F.softplus(torch.randn(B_, T_, nheads_) - 4 + torch.randn(nheads_) * 0.1
                    ).clamp(0.0, float("inf"))
    a_log = -torch.exp(torch.rand(nheads_))
    b_raw = torch.randn(B_, T_, 1, N_).repeat_interleave(nheads_, dim=2)
    c_raw = torch.randn(B_, T_, 1, N_).repeat_interleave(nheads_, dim=2)
    a_raw = dt * a_log
    x_dt = xr * dt.unsqueeze(-1)
    y_ref, f_ref = ssd_reference(x_dt, a_raw, b_raw, c_raw, L_)
    xc, ac, bc, cc = _chunk_inputs(x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half())
    y, fs = ssd_spyre(xc, ac, bc, cc, config=cfg)
    ey, ef = rel_l2(y.cpu(), y_ref), rel_l2(fs.cpu(), f_ref)
    print(f"long-T T={T_} L={L_} C={T_ // L_} (policy):  Y={ey:.4f}  final={ef:.4f}")
    assert ey < 0.05 and ef < 0.05, f"long-T failed: Y={ey:.4f} final={ef:.4f}"
    print(f"PASSED (long-T: pick_config chose L={L_})")


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

    y_ref, final_ref = ssd_reference(x_dt, a_raw, b_raw, c_raw, L)

    # --- CPU mirror: validate the FORMULATION (no card needed) ---
    xd_c, a_c, b_c, c_c = _chunk_inputs(x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half())
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
    y_spyre, final_spyre = ssd_spyre(xd_c, a_c, b_c, c_c)
    err_y = rel_l2(y_spyre.cpu(), y_ref)
    err_final = rel_l2(final_spyre.cpu(), final_ref)
    print(f"  Y            rel-L2 error = {err_y:.4f}")
    print(f"  final_state  rel-L2 error = {err_final:.4f}")
    assert err_y < 0.05, f"Y rel-L2 {err_y:.4f} exceeds 0.05 (fp16 budget)"
    assert err_final < 0.05, f"final_state rel-L2 {err_final:.4f} exceeds 0.05"
    print("PASSED (Spyre vs Mamba reference, fp16 relative-L2 tolerance)")

    # --- non-zero initial_states ---
    # print("\nRunning with non-zero initial_states...")
    # init_ref = torch.randn(B, 1, nheads, P, N) * 0.5       # ref layout (B,1,H,P,N)
    # y_ref2, final_ref2 = ssd_reference(x_dt, a_raw, b_raw, c_raw, L, initial_states=init_ref)
    # init_spyre = init_ref.squeeze(1).half()                # (B, H, P, N) for the driver
    # y_spyre2, final_spyre2 = ssd_spyre(xd_c, a_c, b_c, c_c, initial_states=init_spyre)
    # err_y2 = rel_l2(y_spyre2.cpu(), y_ref2)
    # err_final2 = rel_l2(final_spyre2.cpu(), final_ref2)
    # print(f"  Y            rel-L2 error = {err_y2:.4f}")
    # print(f"  final_state  rel-L2 error = {err_final2:.4f}")
    # assert err_y2 < 0.05, f"init_states Y rel-L2 {err_y2:.4f} exceeds 0.05"
    # assert err_final2 < 0.05, f"init_states final_state rel-L2 {err_final2:.4f} exceeds 0.05"
    # print("PASSED (non-zero initial_states)")

    # # --- C < 64 config: exercise the zero-chunk C-PADDING branch ---
    # print("\nRunning C<64 config (T small -> C=32, exercises C-padding)...")
    # T2, L2 = 2048, 64                                      # C = 2048//64 = 32 < 64
    # globals()["T"], globals()["L"], globals()["C"] = T2, L2, T2 // L2
    # globals()["BH_TILES"] = next(t for t in [4, 2, 1] if (B * nheads) % t == 0)
    # _device_const_cache.clear()   # shapes changed across configs; drop stale consts
    # xr = torch.randn(B, T2, nheads, P)
    # dt2 = F.softplus(torch.randn(B, T2, nheads) - 4 + dt_bias).clamp(dt_min, dt_max)
    # b2 = torch.randn(B, T2, G, N).repeat_interleave(nheads // G, dim=2)
    # c2 = torch.randn(B, T2, G, N).repeat_interleave(nheads // G, dim=2)
    # a2 = dt2 * a_log
    # xdt2 = xr * dt2.unsqueeze(-1)
    # yr3, fr3 = ssd_reference(xdt2, a2, b2, c2, L2)
    # xc3, ac3, bc3, cc3 = _chunk_inputs(xdt2.half(), a2.half(), b2.half(), c2.half())
    # ys3, fs3 = ssd_spyre(xc3, ac3, bc3, cc3)
    # ey3, ef3 = rel_l2(ys3.cpu(), yr3), rel_l2(fs3.cpu(), fr3)
    # print(f"  C=32 (padded to 64):  Y={ey3:.4f}  final={ef3:.4f}")
    # assert ey3 < 0.05, f"C<64 Y rel-L2 {ey3:.4f} exceeds 0.05"
    # assert ef3 < 0.05, f"C<64 final_state rel-L2 {ef3:.4f} exceeds 0.05"
    # print("PASSED (C<64 zero-chunk padding path)")

    # # Long-sequence validation (pick_config → L=128 at T=16K) is NOT run here: the
    # # Spyre backend (dxp_standalone) accumulates in-process compile state, so a 5th
    # # shape-varying device compile after the four above SIGABRTs ("Immediate value
    # # out of boundary") even though the same config compiles cleanly in a fresh
    # # process (verified Y=0.0047). Run it standalone in its OWN process:
    # #     python -c "import test_ssd as m; m.validate_long_t()"
    # print("\n(Long-T config not run in-process — one Spyre compile per process; "
    #       "run `python -c 'import test_ssd as m; m.validate_long_t()'`.)")