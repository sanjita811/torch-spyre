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
"""Detailed CPU mirror of the Spyre SSD kernel — factored into per-stage helpers
that map 1:1 onto the device functions (``_ssd_core_cpu`` ↔ ``fused_kernel``,
``_build_decay_cpu`` ↔ ``build_decay_matrix``, ``_build_intra_decay_cpu`` ↔
``build_intra_decay``). Pure CPU torch — no Spyre imports, no compile.

This is the *deep-debugging* validator: when a device result looks wrong, run the
matching helper here to localize whether the bug is in a specific stage. For the
routine formulation check, ``test_ssd.ssd_cpu`` (the compact single-function
version) is enough; both compute the identical math."""
import torch
import torch.nn.functional as F

from ssd_config import INTRA_FACTORED_TOTAL_LIMIT


def _ssd_core_cpu(a, cumsum_tri, c_proj, b_proj, x, decay_matrix,
                  causal_mask=None, decay_intra=None, init_state=None, init_col=None):
    """CPU twin of fused_kernel / fused_kernel_masked (path chosen by decay_intra)."""
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
    """CPU twin of build_decay_matrix: (BH, C+1, C) scan decay-matrix, same clamp."""
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)
    decay_before = decay_cumsum - chunk_decay
    decay_total = decay_cumsum[:, -1:]
    strict = torch.tril(torch.ones(n_chunks, n_chunks, dtype=torch.bool), -1)
    outer = decay_before.unsqueeze(-1) - decay_cumsum.unsqueeze(-2)     # (BH, C, C)
    decay_run = torch.exp(torch.clamp(outer, max=0.0)) * strict.to(outer.dtype)
    decay_final = torch.exp(decay_total - decay_cumsum).reshape(n_bh, 1, n_chunks)
    return torch.cat([decay_run, decay_final], dim=1).to(dtype)         # (BH, C+1, C)


def _build_intra_decay_cpu(a_flat, n_bh, n_chunks, chunk_len, dtype):
    """CPU twin of build_intra_decay: (BH, C, L, L) bounded mask, same asymmetric
    op sequence (g_row pre-expanded, g_col broadcast via unsqueeze(-2))."""
    g = a_flat.float().cumsum(-1)                                       # (BH,C,L)
    causal = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool))
    g_row = g.unsqueeze(-1).expand(n_bh, n_chunks, chunk_len, chunk_len)  # (BH,C,L,La)
    g_col = g                                                            # (BH,C,La)
    outer = g_row - g_col.unsqueeze(-2)                                 # (BH,C,L,La)
    return (torch.exp(torch.clamp(outer, max=0.0)) * causal.to(g.dtype)).to(dtype)


def ssd_cpu(x, a, b_proj, c_proj, initial_states=None, dtype=torch.float32):
    """CPU mirror of ``ssd_spyre`` (same ops, same routing, C-pad, rank-1 init),
    built from the per-stage helpers above. Inputs (B, nheads, C, L, *)."""
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

    if float(chunk_decay.abs().max()) < INTRA_FACTORED_TOTAL_LIMIT:
        causal_mask = torch.tril(torch.ones(chunk_len, chunk_len, dtype=dtype))
        y_grouped, scan_out = _ssd_core_cpu(
            a_d, cumsum_tri, c_d, b_d, x_d, decay_matrix,
            causal_mask=causal_mask, init_state=init_state, init_col=init_col)
    else:
        decay_intra = _build_intra_decay_cpu(a_flat, n_bh, n_chunks, chunk_len, dtype)
        y_grouped, scan_out = _ssd_core_cpu(
            a_d, cumsum_tri, c_d, b_d, x_d, decay_matrix,
            decay_intra=decay_intra, init_state=init_state, init_col=init_col)

    final_state = (
        scan_out[:, c_real]
        .reshape(batch, heads, state_dim, head_dim)
        .permute(0, 1, 3, 2)
    )
    y = y_grouped.reshape(batch, heads, n_chunks, chunk_len, head_dim)
    y = y[:, :, :c_real].permute(0, 2, 3, 1, 4).contiguous()
    return y.reshape(batch, c_real * chunk_len, heads, head_dim).half(), final_state.half()
