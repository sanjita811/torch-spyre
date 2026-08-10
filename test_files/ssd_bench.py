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
"""One-config SSD device benchmark (one Spyre compile per process).

Config-DRIVEN: builds an explicit SSDConfig(L, bh_tiles, scan_mode) and passes it to
ssd_spyre so C>64 routes through fused_kernel_cblock (the dense flat path walls at C>64).

Usage: python ssd_bench.py T L bh_tiles [N]
Prints: SWEEP T=.. L=.. C=.. bh=.. | Y=.. final=.. | peak=..GB warm=..ms | OK/WRONG
(peak GB is filled by the runner from the HBM_POOL_PLANNING log; here it prints n/a).
"""
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_ssd as m  # noqa: E402
from ssd_config import SSDConfig  # noqa: E402


def main():
    T = int(sys.argv[1]); L = int(sys.argv[2]); bh = int(sys.argv[3])
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    B, H, P = 2, 2048, 64
    nheads = H // P
    C = T // L
    m.B, m.T, m.H, m.P, m.N, m.L, m.C, m.nheads, m.G = B, T, H, P, N, L, C, nheads, 1
    m.CBLOCK = 64
    m._device_const_cache.clear()

    cfg = SSDConfig(
        L=L, bh_tiles=bh,
        scan_mode="cblock" if C > 64 else "flat",
        intra="factored", cblock_size=64,
    )

    torch.manual_seed(42)
    x_raw = torch.randn(B, T, nheads, P)
    dt = F.softplus(
        torch.randn(B, T, nheads) - 4 + torch.randn(nheads) * 0.1
    ).clamp(0.0, float("inf"))
    a_log = -torch.exp(torch.rand(nheads))
    b_raw = torch.randn(B, T, 1, N).repeat_interleave(nheads, dim=2)
    c_raw = torch.randn(B, T, 1, N).repeat_interleave(nheads, dim=2)
    a_raw = dt * a_log
    x_dt = x_raw * dt.unsqueeze(-1)

    y_ref, f_ref = m.ssd_reference(x_dt, a_raw, b_raw, c_raw, L)
    xc, ac, bc, cc = m._chunk_inputs(
        x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half(), L)

    t0 = time.time()
    y, fs = m.ssd_spyre(xc, ac, bc, cc, config=cfg)     # cold: includes compile
    t_cold = time.time() - t0
    ey = m.rel_l2(y.cpu(), y_ref); ef = m.rel_l2(fs.cpu(), f_ref)

    warm = []
    for _ in range(5):
        t0 = time.time()
        m.ssd_spyre(xc, ac, bc, cc, config=cfg)
        warm.append((time.time() - t0) * 1000)
    warm_ms = statistics.median(warm)

    ok = ey < 0.05 and ef < 0.05
    print(f"SWEEP T={T} L={L} C={C} bh={bh} N={N} | Y={ey:.4f} final={ef:.4f} "
          f"| cold={t_cold:.1f}s warm={warm_ms:.1f}ms | {'OK' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
