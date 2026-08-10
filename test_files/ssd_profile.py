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
"""Profile the SSD kernel with the kineto/AIUPTI Spyre profiler.

The rebuilt _C.so (Aug 2026) registers ProfilerActivity.SPYRE and links
libaiupti.so, so torch.profiler captures Spyre kernel launches (and, with the
aiupti activity trace, device-side timing). Run e.g.:

  LD_PRELOAD=/opt/ibm/spyre/deeptools/lib/libutil.so \
  SSD_T=8192 SSD_L=64 CBLOCK_SCAN=1 \
  .venv/bin/python test_files/ssd_profile.py [trace_out.json]

Imports test_ssd.py's ssd_spyre + input builders, warms up (compile), then
profiles a few steady-state iterations. Prints the op table sorted by device
time and optionally writes a chrome trace.
"""
import os
import sys

import torch  # noqa: F401
import torch_spyre  # noqa: F401
from torch.profiler import ProfilerActivity, profile

# test_ssd.py lives one dir up; import it as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_ssd as ssd  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TRACE_OUT = sys.argv[1] if len(sys.argv) > 1 else None
# AIUPTI device-trace buffers cap at 5; keep iters small so the trace doesn't overflow.
ITERS = int(os.environ.get("PROF_ITERS", "2"))


def _build_inputs():
    """Mirror test_ssd.__main__ input construction at the env-configured T/L."""
    torch.manual_seed(42)
    B, T, nheads, P, N, G = ssd.B, ssd.T, ssd.nheads, ssd.P, ssd.N, ssd.G
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
    # Config drives the run (as __main__ does): chunk with cfg.L so C>64 routes to cblock.
    cfg = ssd.best_config(ssd.T)
    chunk_len = ssd.L if "SSD_L" in os.environ else cfg.L
    chunked = ssd._chunk_inputs(
        x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half(), chunk_len)
    return chunked, cfg, chunk_len


def main():
    (xd_c, a_c, b_c, c_c), cfg, chunk_len = _build_inputs()
    print(f"SSD profile: T={ssd.T} L={chunk_len} C={ssd.T // chunk_len} "
          f"scan={cfg.scan_mode} iters={ITERS}")

    # warmup / compile (not profiled)
    y, fs = ssd.ssd_spyre(xd_c, a_c, b_c, c_c, config=cfg)
    y.cpu()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.SPYRE],
                 record_shapes=False) as prof:
        for _ in range(ITERS):
            y, fs = ssd.ssd_spyre(xd_c, a_c, b_c, c_c, config=cfg)
        y.cpu()

    # Spyre device time shows in the "Self SPYRE" column; sort by CPU time (the valid
    # sort key — self_spyre_time_total is not a FunctionEventAvg sort attribute).
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=20))
    if TRACE_OUT:
        prof.export_chrome_trace(TRACE_OUT)
        print(f"chrome trace -> {TRACE_OUT}")


if __name__ == "__main__":
    main()
