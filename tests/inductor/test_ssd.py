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

"""Mamba-2 SSD (state-space duality) kernel test.

The reusable device kernels live in ``torch_spyre._inductor.customops`` (the
``_ssd_*`` traced graphs). The host-side driver that chunks the inputs, declares
and binds the named dims, builds the decay matrix, routes factored-vs-masked,
and stitches the compiled kernels together lives here, next to the ground-truth
reference it is checked against:

  * ``ssd_reference`` — verbatim Mamba-2 chunked SSD (ground truth).
  * ``ssd_cpu``       — the same op sequence on CPU (validates the formulation).
  * ``_ssd_spyre``    — the device driver (compiles + runs the customops kernels).

Acceptance is relative-L2 vs the reference (the shared harness only does
elementwise atol/rtol, so the gate is a local assert). Design and backend
history: docs/source/compiler/ssd_kernel.md.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from torch_spyre._inductor import customops
from torch_spyre._inductor.wsr.propagate_named_dims import (
    declare_tensor_dim,
    name_tensor_dims,
)

try:
    from torch.spyre import SpyreTensorLayout
except (ImportError, ModuleNotFoundError):
    from torch_spyre._C import SpyreTensorLayout

# Model shapes (Mamba-2 names): B batch, H model dim, P head dim, N state dim,
# G groups; nheads = H // P.
_B, _H, _P, _N, _G = 2, 2048, 64, 128, 1
_NHEADS = _H // _P

# Driver policy (the kernels' own tiling config lives on their signatures):
_SSD_INTRA_FACTORED_TOTAL_LIMIT = 20.0  # fp16 guard; above this -> masked fallback
_SSD_MAX_FLAT_SCAN_CHUNKS = 64  # dense flat scan fits the per-core span only to C=64


# ======================= reference (CPU, ground truth) ==================
def segsum(x):
    """Stable segment sum (verbatim, Mamba ssd_minimal.py)."""
    n = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=n)
    mask = torch.tril(torch.ones(n, n, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(n, n, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_reference(X, A, B, C, block_len, initial_states=None):
    """Reference SSD = ssd_minimal_discrete (verbatim, state-spaces/mamba)."""
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    X, A, B, C = (
        rearrange(t, "b (c l) ... -> b c l ...", l=block_len) for t in (X, A, B, C)
    )
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


# ===================== CPU mirror (formulation validator) ===============
def ssd_cpu(x, a, b_proj, c_proj, initial_states=None, dtype=torch.float32,
            factored_limit=None):
    """Independent CPU oracle (no shared code with the device path): same op
    sequence, routing, C-padding, rank-1 init. fp32 checks the formulation; fp16
    the numeric floor."""
    if factored_limit is None:
        factored_limit = _SSD_INTRA_FACTORED_TOTAL_LIMIT
    batch, heads, n_chunks, chunk_len, head_dim = x.shape
    state_dim = b_proj.shape[-1]
    n_bh = batch * heads

    # --- chunk + C-pad to a whole 64-stick (padded chunks contribute 0) ---
    x_flat = x.reshape(n_bh, n_chunks, chunk_len, head_dim)
    a_flat = a.reshape(n_bh, n_chunks, chunk_len)
    b_flat = b_proj.reshape(n_bh, n_chunks, chunk_len, state_dim)
    c_flat = c_proj.reshape(n_bh, n_chunks, chunk_len, state_dim)
    elem_stick = 64
    c_real = n_chunks
    if n_chunks % elem_stick != 0:
        n_chunks = ((n_chunks + elem_stick - 1) // elem_stick) * elem_stick
        pad = n_chunks - c_real
        x_flat = F.pad(x_flat, (0, 0, 0, 0, 0, pad))
        a_flat = F.pad(a_flat, (0, 0, 0, pad))
        b_flat = F.pad(b_flat, (0, 0, 0, 0, 0, pad))
        c_flat = F.pad(c_flat, (0, 0, 0, 0, 0, pad))

    chunk_decay = a_flat.float().sum(-1)  # (BH, C)
    x_d = x_flat.to(dtype)
    a_d = a_flat.to(dtype)
    b_d = b_flat.to(dtype)
    c_d = c_flat.to(dtype)
    cumsum_tri = torch.triu(torch.ones(chunk_len, chunk_len, dtype=dtype))

    # --- scan decay-matrix (BH, C+1, C): run rows exp(clamp(before_i -
    # cumsum_s)) * strict-lower, plus the final-state row exp(total - cumsum) ---
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)  # (BH, C)
    decay_before = decay_cumsum - chunk_decay  # exclusive cumsum
    strict = torch.tril(torch.ones(n_chunks, n_chunks, dtype=torch.bool), -1)
    outer = decay_before.unsqueeze(-1) - decay_cumsum.unsqueeze(-2)  # (BH, C, C)
    decay_run = torch.exp(torch.clamp(outer, max=0.0)) * strict.to(outer.dtype)
    decay_final = torch.exp(
        decay_cumsum[:, -1:] - decay_cumsum
    ).reshape(n_bh, 1, n_chunks)
    decay_matrix = torch.cat([decay_run, decay_final], dim=1).to(dtype)  # (BH,C+1,C)

    # --- optional non-zero initial state -> rank-1 scan correction ---
    init_state = init_col = None
    if initial_states is not None:
        init_col = torch.exp(
            torch.cat([decay_before, decay_cumsum[:, -1:]], dim=1)
        ).unsqueeze(-1).to(dtype)  # (BH, C+1, 1)
        init_state = initial_states.reshape(
            n_bh, head_dim, state_dim
        ).transpose(-1, -2).reshape(
            n_bh, 1, state_dim * head_dim
        ).contiguous().to(dtype)  # (BH, 1, N*P)

    # --- intra: factored (fp16-safe) fast path, or masked (BH,C,L,L) fallback ---
    intra_cumsum = torch.matmul(a_d, cumsum_tri)  # (BH,C,L) = g
    total = a_d.sum(dim=-1, keepdim=True)  # (BH,C,1)
    if float(chunk_decay.abs().max()) < factored_limit:
        # factored (mirrors _ssd_fused_cblock); clamp the pre-mask product to fp16
        # range so the masked-away upper triangle can't overflow to inf*0 = NaN.
        shifted = intra_cumsum - 0.5 * total
        half_tot = torch.exp(0.5 * total).unsqueeze(-1)
        c_scaled = c_d * torch.exp(shifted).unsqueeze(-1)
        b_scaled_t = (b_d * torch.exp(-shifted).unsqueeze(-1)).transpose(-1, -2)
        causal_mask = torch.tril(torch.ones(chunk_len, chunk_len, dtype=dtype))
        attn = torch.clamp(torch.matmul(c_scaled, b_scaled_t), -65504.0, 65504.0)
        y_diag = torch.matmul(attn * causal_mask, x_d)
        chunk_states = torch.matmul(b_scaled_t, x_d * half_tot)
    else:
        # mirrors _ssd_fused_masked: precomputed (BH,C,L,L) bounded decay mask
        g = a_flat.float().cumsum(-1)
        causal = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool))
        outer_l = g.unsqueeze(-1).expand(
            n_bh, n_chunks, chunk_len, chunk_len
        ) - g.unsqueeze(-2)  # (BH,C,L,La)
        decay_intra = (
            torch.exp(torch.clamp(outer_l, max=0.0)) * causal.to(g.dtype)
        ).to(dtype)
        y_diag = torch.matmul(
            torch.matmul(c_d, b_d.transpose(-1, -2)) * decay_intra, x_d
        )
        b_decayed = b_d * torch.exp(total - intra_cumsum).unsqueeze(-1)
        chunk_states = torch.matmul(b_decayed.transpose(-1, -2), x_d)

    # --- inter-chunk scan + off-diagonal combine ---
    bh, c, n, p = chunk_states.shape
    scan_out = torch.matmul(decay_matrix, chunk_states.reshape(bh, c, n * p))
    if init_state is not None:
        scan_out = scan_out + init_col * init_state
    rolled_states = scan_out[:, :c].reshape(bh, c, n, p)
    y_grouped = (
        torch.matmul(c_d, rolled_states) * torch.exp(intra_cumsum).unsqueeze(-1)
        + y_diag
    )

    # --- final state = scan row after the real chunks; un-fold y to (B,T,H,P) ---
    final_state = scan_out[:, c_real].reshape(
        batch, heads, state_dim, head_dim
    ).permute(0, 1, 3, 2)
    y = y_grouped.reshape(batch, heads, n_chunks, chunk_len, head_dim)
    y = y[:, :, :c_real].permute(0, 2, 3, 1, 4).contiguous()
    return (
        y.reshape(batch, c_real * chunk_len, heads, head_dim).half(),
        final_state.half(),
    )


# ============================== Spyre driver ============================
_ssd_device_const_cache: dict = {}


def _ssd_device_const(key, build):
    """Build a data-independent constant on host once, cache on Spyre, reuse."""
    tensor = _ssd_device_const_cache.get(key)
    if tensor is None:
        tensor = _ssd_device_const_cache[key] = build().to("spyre")
    return tensor


def _ssd_declare_dims(n_chunks, chunk_len, n_bh, head_dim, state_dim):
    """Register named dims before each compile (the registry resets per run)."""
    for name, size in [
        ("L", chunk_len), ("La", chunk_len), ("Lk", chunk_len),
        ("P", head_dim), ("N", state_dim), ("One", 1),
        ("BH", n_bh), ("C", n_chunks), ("Ca", n_chunks), ("Cp", n_chunks + 1),
        ("PN", head_dim * state_dim),
    ]:
        declare_tensor_dim(name, size)


def _ssd_bind_dims(n_chunks, chunk_len, n_bh, head_dim, state_dim, specs):
    """Declare dims and annotate each (tensor, names); both reset per compile."""
    _ssd_declare_dims(n_chunks, chunk_len, n_bh, head_dim, state_dim)
    for tensor, names in specs:
        name_tensor_dims(tensor, names)


def _ssd_spyre(x, a, b_proj, c_proj, initial_states=None, factored_limit=None):
    """Run the SSD core on Spyre. Inputs (B, nheads, C, L, *). Compiles and runs
    the customops device kernels; host prep + dim binding + routing here."""
    if factored_limit is None:
        factored_limit = _SSD_INTRA_FACTORED_TOTAL_LIMIT
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
        x_flat = F.pad(x_flat, (0, 0, 0, 0, 0, pad))  # pad chunk dim
        a_flat = F.pad(a_flat, (0, 0, 0, pad))
        b_flat = F.pad(b_flat, (0, 0, 0, 0, 0, pad))
        c_flat = F.pad(c_flat, (0, 0, 0, 0, 0, pad))
        n_chunks = c_pad

    chunk_decay = a_flat.float().sum(-1)  # (BH, C)

    x_dev, a_dev, b_dev, c_dev = (
        t.to("spyre") for t in (x_flat, a_flat, b_flat, c_flat)
    )
    cumsum_tri = _ssd_device_const(
        f"cumsum_tri_{chunk_len}",
        lambda: torch.triu(torch.ones(chunk_len, chunk_len, dtype=torch.float16)),
    )
    causal_mask = _ssd_device_const(
        f"causal_intra_{chunk_len}",
        lambda: torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.float16)),
    )

    # Scan decay-matrix (BH, C+1, C), built on device. Inputs carry the stick on
    # the BH dim so the C-broadcast in the outer-difference is off the stick.
    decay_cumsum = torch.cumsum(chunk_decay, dim=-1)  # (BH, C)
    decay_before = decay_cumsum - chunk_decay  # exclusive cumsum
    final_arg = (decay_cumsum[:, -1:] - decay_cumsum).half()  # (BH, C), all <= 0
    mask_dev = _ssd_device_const(
        f"strict_mask_{n_chunks}",
        lambda: torch.tril(
            torch.ones(n_chunks, n_chunks, dtype=torch.float16), -1
        ).reshape(1, n_chunks, n_chunks),
    )
    bh_stick = SpyreTensorLayout(
        [n_bh, n_chunks], [n_chunks, 1], torch.float16, [1, 0]  # stick on BH
    )
    _ssd_declare_dims(n_chunks, chunk_len, n_bh, head_dim, state_dim)
    decay_run = torch.compile(customops._ssd_build_decay_matrix, dynamic=False)(
        decay_before.half().to(device_layout=bh_stick),
        decay_cumsum.half().to(device_layout=bh_stick),
        mask_dev,
    )  # (BH, C, C)
    decay_final = torch.exp(final_arg).unsqueeze(1).half().to("spyre")  # (BH,1,C)

    init_state = init_col = None
    if initial_states is not None:
        init_col = torch.exp(
            torch.cat([decay_before, decay_cumsum[:, -1:]], dim=1)
        ).unsqueeze(-1).half().to("spyre")  # (BH, C+1, 1)
        init_state = initial_states.reshape(
            n_bh, head_dim, state_dim
        ).transpose(-1, -2).reshape(
            n_bh, 1, state_dim * head_dim
        ).contiguous().half().to("spyre")  # (BH, 1, N*P)

    # Factored intra is fp16-safe iff max|chunk_decay| < the limit; else masked.
    factored = float(chunk_decay.abs().max()) < factored_limit

    common = [
        (a_dev, ["BH", "C", "Lk"]), (cumsum_tri, ["Lk", "L"]),
        (c_dev, ["BH", "C", "L", "N"]), (b_dev, ["BH", "C", "La", "N"]),
        (x_dev, ["BH", "C", "La", "P"]),
    ]
    if init_state is not None:
        common += [
            (init_col, ["BH", "Cp", "One"]),
            (init_state, ["BH", "One", "PN"]),
        ]

    if factored:
        _ssd_bind_dims(
            n_chunks, chunk_len, n_bh, head_dim, state_dim,
            common + [
                (decay_run, ["BH", "C", "Ca"]),
                (decay_final, ["BH", "One", "Ca"]),
                (causal_mask, ["L", "La"]),
            ],
        )
        y_grouped, scan_out = torch.compile(
            customops._ssd_fused_cblock, dynamic=False
        )(
            a_dev, cumsum_tri, c_dev, b_dev, causal_mask, x_dev,
            decay_run, decay_final, init_state, init_col,
        )
    else:
        decay_matrix = torch.cat(
            [decay_run.cpu(), decay_final.cpu()], dim=1
        ).to("spyre")
        g = a_flat.float().cumsum(-1)  # (BH,C,L)
        g_row_d = g.unsqueeze(-1).expand(
            n_bh, n_chunks, chunk_len, chunk_len
        ).contiguous().half().to("spyre")
        g_col_d = g.half().to("spyre")  # (BH,C,La)
        _ssd_bind_dims(
            n_chunks, chunk_len, n_bh, head_dim, state_dim,
            [(g_row_d, ["BH", "C", "L", "La"]), (g_col_d, ["BH", "C", "La"])],
        )
        decay_intra_dev = torch.compile(
            customops._ssd_build_intra_decay, dynamic=False
        )(g_row_d, g_col_d, causal_mask)
        _ssd_bind_dims(
            n_chunks, chunk_len, n_bh, head_dim, state_dim,
            common + [
                (decay_intra_dev, ["BH", "C", "L", "La"]),
                (decay_matrix, ["BH", "Cp", "Ca"]),
            ],
        )
        y_grouped, scan_out = torch.compile(
            customops._ssd_fused_masked, dynamic=False
        )(
            a_dev, cumsum_tri, c_dev, b_dev, decay_intra_dev, x_dev,
            decay_matrix, init_state, init_col,
        )

    # The C-blocked kernel returns ONLY the (BH,1,NP) final row; masked returns
    # the full (BH,C+1,NP) scan.
    final_row = scan_out[:, 0] if factored else scan_out[:, c_real]
    final_state = final_row.cpu().reshape(
        batch, heads, state_dim, head_dim
    ).permute(0, 1, 3, 2)

    # Un-fold (BH, C_pad, L, P) -> (B, T, H, P), dropping any padded chunks.
    y = y_grouped.cpu().reshape(batch, heads, n_chunks, chunk_len, head_dim)
    y = y[:, :, :c_real].permute(0, 2, 3, 1, 4).contiguous()
    return (
        y.reshape(batch, c_real * chunk_len, heads, head_dim).half(),
        final_state.half(),
    )


def rel_l2(got, ref):
    """Relative L2 (norm-based) error — the acceptance metric for this kernel."""
    got, ref = got.float(), ref.float()
    return (got - ref).norm().item() / (ref.norm().item() + 1e-12)


def _build_inputs(seq_len, chunk_len, seed=42):
    """Build discretized Mamba-2 inputs and the reference outputs.

    Returns the chunked device-shaped tensors (B, nheads, C, L, *) alongside the
    reference (Y, final_state) computed by ``ssd_reference``.
    """
    torch.manual_seed(seed)
    x_raw = torch.randn(_B, seq_len, _NHEADS, _P)  # pre-discretization input
    dt_bias = torch.randn(_NHEADS) * 0.1  # per-head learnable bias
    dt = F.softplus(torch.randn(_B, seq_len, _NHEADS) - 4 + dt_bias)
    a_log = -torch.exp(torch.rand(_NHEADS))  # A = -exp(A_log), per-head
    b_grp = torch.randn(_B, seq_len, _G, _N)
    c_grp = torch.randn(_B, seq_len, _G, _N)
    b_raw = b_grp.repeat_interleave(_NHEADS // _G, dim=2)  # (B, T, nheads, N)
    c_raw = c_grp.repeat_interleave(_NHEADS // _G, dim=2)
    a_raw = dt * a_log  # discretized A = dt*A
    x_dt = x_raw * dt.unsqueeze(-1)  # dt-scaled SSM input (ZOH)

    y_ref, final_ref = ssd_reference(x_dt, a_raw, b_raw, c_raw, chunk_len)

    # Chunk (B,T,nheads,*) -> (B,nheads,C,L,*); b/c already per-head expanded.
    nc = seq_len // chunk_len  # C = T / L
    xd_c = x_dt.half().reshape(
        _B, nc, chunk_len, _NHEADS, _P
    ).permute(0, 3, 1, 2, 4).contiguous()
    a_c = a_raw.half().reshape(
        _B, nc, chunk_len, _NHEADS
    ).permute(0, 3, 1, 2).contiguous()
    b_c = b_raw.half().reshape(
        _B, nc, chunk_len, _NHEADS, _N
    ).permute(0, 3, 1, 2, 4).contiguous()
    c_c = c_raw.half().reshape(
        _B, nc, chunk_len, _NHEADS, _N
    ).permute(0, 3, 1, 2, 4).contiguous()
    return xd_c, a_c, b_c, c_c, y_ref, final_ref


def test_ssd_factored():
    """Factored (fp16-safe) path: fp32 formulation is exact; device is within the
    fp16 relative-L2 budget vs the Mamba-2 reference."""
    seq_len, chunk_len = 4096, 64
    xd_c, a_c, b_c, c_c, y_ref, final_ref = _build_inputs(seq_len, chunk_len)

    # fp32 CPU mirror validates the formulation (card-free).
    y_cpu32, fin_cpu32 = ssd_cpu(xd_c, a_c, b_c, c_c, dtype=torch.float32)
    assert rel_l2(y_cpu32, y_ref) < 1e-3
    assert rel_l2(fin_cpu32, final_ref) < 1e-3

    y_spyre, final_spyre = _ssd_spyre(xd_c, a_c, b_c, c_c)
    assert rel_l2(y_spyre, y_ref) < 0.05
    assert rel_l2(final_spyre, final_ref) < 0.05


def test_ssd_masked():
    """Masked fallback path (forced via factored_limit=0.0), valid at C<=64: same
    fp16 relative-L2 budget vs the reference."""
    seq_len, chunk_len = 4096, 64
    assert seq_len // chunk_len <= _SSD_MAX_FLAT_SCAN_CHUNKS
    xd_c, a_c, b_c, c_c, y_ref, final_ref = _build_inputs(seq_len, chunk_len)

    y_masked, final_masked = _ssd_spyre(xd_c, a_c, b_c, c_c, factored_limit=0.0)
    assert rel_l2(y_masked, y_ref) < 0.05
    assert rel_l2(final_masked, final_ref) < 0.05
