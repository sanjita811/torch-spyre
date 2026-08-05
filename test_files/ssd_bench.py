# One-config SSD device benchmark (one Spyre compile per process).
# Usage: python ssd_bench.py T L bh_tiles [N]
# Prints a parseable line: BENCH T=.. L=.. bh=.. | Y=.. final=.. | compile=..s warm=..ms
import os
import sys
import time

import torch
import torch.nn.functional as F

import test_ssd as m


def main():
    T = int(sys.argv[1]); L = int(sys.argv[2]); bh = int(sys.argv[3])
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    B, H, P = 2, 2048, 64
    nheads = H // P; C = T // L
    m.B, m.T, m.H, m.P, m.N, m.L, m.C, m.nheads, m.G = B, T, H, P, N, L, C, nheads, 1
    m.BH_TILES = bh
    m._device_const_cache.clear()

    torch.manual_seed(42)
    x_raw = torch.randn(B, T, nheads, P)
    dt = F.softplus(torch.randn(B, T, nheads) - 4 + torch.randn(nheads) * 0.1).clamp(0.0, float("inf"))
    a_log = -torch.exp(torch.rand(nheads))
    b_raw = torch.randn(B, T, 1, N).repeat_interleave(nheads, dim=2)
    c_raw = torch.randn(B, T, 1, N).repeat_interleave(nheads, dim=2)
    a_raw = dt * a_log
    x_dt = x_raw * dt.unsqueeze(-1)

    y_ref, f_ref = m.ssd_reference(x_dt, a_raw, b_raw, c_raw, L)
    xc, ac, bc, cc = m._chunk_inputs(x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half())

    t0 = time.time()
    y, fs = m.ssd_spyre(xc, ac, bc, cc)      # cold: includes compile
    t_cold = time.time() - t0
    ey = m.rel_l2(y.cpu(), y_ref); ef = m.rel_l2(fs.cpu(), f_ref)

    # warm timed runs (compile cached; still includes H2D/D2H — isolated-bench,
    # but the delta across L is dominated by the kernel per memory's L-sweep)
    warm = []
    for _ in range(3):
        t0 = time.time()
        m.ssd_spyre(xc, ac, bc, cc)
        warm.append((time.time() - t0) * 1000)
    warm_ms = min(warm)
    print(f"BENCH T={T} L={L} C={C} bh={bh} N={N} | Y={ey:.4f} final={ef:.4f} "
          f"| cold={t_cold:.1f}s warm={warm_ms:.1f}ms")


if __name__ == "__main__":
    main()
