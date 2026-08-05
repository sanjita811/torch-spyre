# CPU-only formulation sweep for the SSD kernel. No Spyre card, no compile.
# Validates ssd_cpu (the op-for-op mirror of ssd_spyre) vs the Mamba reference
# across chunk sizes L and the two intra paths, in fp32 and fp16.
import os
import sys

import torch
import torch.nn.functional as F

import test_ssd as m
from ssd_config import INTRA_FACTORED_TOTAL_LIMIT


def build_inputs(B, T, nheads, P, N, G=1, seed=42):
    torch.manual_seed(seed)
    x_raw = torch.randn(B, T, nheads, P)
    dt_bias = torch.randn(nheads) * 0.1
    dt = F.softplus(torch.randn(B, T, nheads) - 4 + dt_bias).clamp(0.0, float("inf"))
    a_log = -torch.exp(torch.rand(nheads))
    b_grp = torch.randn(B, T, G, N)
    c_grp = torch.randn(B, T, G, N)
    b_raw = b_grp.repeat_interleave(nheads // G, dim=2)
    c_raw = c_grp.repeat_interleave(nheads // G, dim=2)
    a_raw = dt * a_log
    x_dt = x_raw * dt.unsqueeze(-1)
    return x_dt, a_raw, b_raw, c_raw


def run(B, T, H, P, N, L):
    nheads = H // P
    C = T // L
    # set the module globals ssd_cpu / _chunk_inputs read
    m.B, m.T, m.H, m.P, m.N, m.L, m.C, m.nheads, m.G = B, T, H, P, N, L, C, nheads, 1

    x_dt, a_raw, b_raw, c_raw = build_inputs(B, T, nheads, P, N)
    y_ref, fin_ref = m.ssd_reference(x_dt, a_raw, b_raw, c_raw, L)
    xc, ac, bc, cc = m._chunk_inputs(x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half())

    # per-chunk |total| — decides factored fp16 safety
    a_flat = ac.reshape(B * nheads, C, L)
    chunk_decay = a_flat.float().sum(-1)
    maxtot = float(chunk_decay.abs().max())
    factored_safe = maxtot < INTRA_FACTORED_TOTAL_LIMIT

    row = f"B{B} T{T} H{H} P{P} N{N} L{L:4d} C{C:4d} | max|tot|={maxtot:6.2f} safe={factored_safe}"

    for dt in (torch.float32, torch.float16):
        y, fs = m.ssd_cpu(xc, ac, bc, cc, dtype=dt)
        ey = m.rel_l2(y, y_ref)
        ef = m.rel_l2(fs, fin_ref)
        tag = "fp32" if dt == torch.float32 else "fp16"
        row += f"  [{tag} {'MASK' if not factored_safe else 'FACT'}: Y={ey:.4f} f={ef:.4f}]"
    print(row)


if __name__ == "__main__":
    print("=== CPU formulation sweep (ssd_cpu vs Mamba reference) ===")
    # default shape at several chunk sizes
    for L in (64, 128, 256):
        run(2, 4096, 2048, 64, 128, L)
    print("--- longer T ---")
    run(2, 8192, 2048, 64, 128, 128)
    run(2, 16384, 2048, 64, 128, 128)
    run(2, 16384, 2048, 64, 128, 256)
