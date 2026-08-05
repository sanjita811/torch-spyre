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
"""Chunk-size / tiling policy for the SSD kernel — pure arithmetic, no torch.

Separated from ``test_ssd.py`` so the kernel file stays about kernels. The kernel
consults this via ``pick_config(...)`` (analytic) or the ``ssd_sweep.autotune``
cache (measured), then passes the resulting ``SSDConfig`` to ``ssd_spyre``.

The kernel's cost has a U-curve in the chunk size L (with C = T/L chunks):
  * intra terms (c@bᵀ, attn@x) grow ∝ L
  * the inter-chunk scan is O(C²) = O((T/L)²) in both MACs AND memory (the
    (BH,C+1,C) decay-matrix), so it EXPLODES as L shrinks at long T.
Measured (clean device kernel_ms): T=16K went 741ms @L=64 → 53.6ms @L=128 (14×)
just by doubling L. The compute-optimal chunk size is L*(T) = (2·T·N·P/(N+P))^⅓,
which matches the measured optima (T=4K→64, T=16K→128). So pick L per shape.
"""
import dataclasses
import math

# fp16 exponent ceiling: exp(FP16_EXP_MAX) is the largest representable fp16
# (ln 65504 ≈ 11.09). The factored intra-decay's peak scale is exp(|total|/2), so
# it is fp16-safe iff max|total| < 2·11.09 ≈ 22.2. We keep a margin.
FP16_EXP_MAX = 11.0
INTRA_FACTORED_TOTAL_LIMIT = 2.0 * (FP16_EXP_MAX - 1.0)   # ≈ 20.0 (with margin)

# HARD scan-span constraint (measured on merged main f4fab17, 2026-07-31). The
# dense O(C²) inter-chunk scan materializes a coarse-tile read-copy whose per-core
# span scales with the (C+1) row dim, which the work-division pass cannot split
# below the AIU hardware limit (MAX_SPAN = 65535·4096 = 256 MiB, see
# work_division.py). Measured envelope at N=128 (NP=8192): C≤64 compiles; C≥128
# ALWAYS fails ("per-core tensor span … exceeds hardware limit" / dxp SIGABRT /
# HBM pool OOM). Verified: T4096/L64/C64 ✓, T8192/L128/C64 ✓, T16384/L256/C64 ✓
# vs T8192/L64/C128 ✗, T16384/L64/C256 ✗ (16.5GB), T16384/L128/C128 ✗ (344MB span).
# So pick_config MUST keep C = T/L ≤ this until a sub-quadratic (hierarchical)
# scan lands. BACKEND ASK: split the scan's (C+1) row dim in work-division, or
# tile the scan matmul, to lift this cap without a kernel-side blocked scan.
MAX_FLAT_SCAN_CHUNKS = 64

# Largest L whose intra (BH,C,L,L) attn fits the per-core span limit. Measured
# 2026-08-01: L=512 OK (T=32768/C64, Y=0.0109); L=1024 exceeds 256MB/core + HBM OOM.
# So flat-scan ceiling = MAX_FLAT_SCAN_CHUNKS·MAX_INTRA_L = 64·512 = 32768.
MAX_INTRA_L = 512


def _l_star(T, N, P):
    """Compute-optimal chunk size (continuous): minimizes a·L + b/L²."""
    return (2.0 * T * N * P / (N + P)) ** (1.0 / 3.0)


def _snap_L(L_cont, T):
    """64-multiple NEAREST to L_cont that divides T (so C = T/L is integral).

    L must be a multiple of 64 (fp16 stick) for the intra matmuls to stay
    stick-aligned. NEAREST (not floor): the U-curve is shallow near L* but the
    scan cost rises steeply below it, so rounding L*≈112 up to 128 (measured 14×
    faster at T=16K) beats rounding down to 64. Falls back to 64.
    """
    lo = max(64, int(L_cont // 64) * 64)
    hi = lo + 64
    order = (lo, hi) if (L_cont - lo) <= (hi - L_cont) else (hi, lo)
    for L in order:
        if L >= 64 and T % L == 0:
            return L
    for L in range(max(64, (int(L_cont // 64) + 2) * 64), 63, -64):
        if T % L == 0:
            return L
    return 64


def _bh_tiles_for(n_bh, L):
    """Largest divisor of n_bh in {16,8,4,2,1}. Device sweep (2026-08-01) found more
    tiles faster up to ~16 at every L (bh32 only ties bh16 at 2-5x compile)."""
    return next(t for t in [16, 8, 4, 2, 1] if n_bh % t == 0)


def _pick_block_K(C):
    """Hierarchical-scan block size: a 64-multiple that divides C, nearest to √C
    (which balances the local O(C·K) and top O((C/K)²) costs). Returns 0 if no
    64-aligned divisor exists (→ caller keeps flat). K and C/K both stay
    stick-aligned so the block/top matmuls avoid the sub-stick out-of-bounds bug."""
    target = math.sqrt(C)
    best, best_err = 0, None
    K = 64
    while K <= C:
        if C % K == 0 and (C // K) >= 1:
            err = abs(K - target)
            if best_err is None or err < best_err:
                best, best_err = K, err
        K += 64
    return best


@dataclasses.dataclass(frozen=True)
class SSDConfig:
    """How to run one (B,T,H,P,N) SSD shape on Spyre. Chosen by ``pick_config``
    (analytic) or the ``ssd_sweep.autotune`` cache (measured). ``scan_mode`` is
    'flat' (one dense O(C²) matmul) or 'hierarchical' (two-level O(C^1.5), for
    large C); ``intra`` is 'factored' (fast, fp16-bounded) or 'masked' (the
    unconditionally-safe (BH,C,L,L) fallback for large per-chunk decay)."""
    L: int
    bh_tiles: int
    scan_mode: str = "flat"          # 'flat' | 'hierarchical'
    block_K: int = 0                 # hier block size (0 = N/A)
    intra: str = "factored"          # 'factored' | 'masked'


def pick_config(B, T, H, P, N, mean_abs_a=0.06, C_hier_threshold=None):
    """Analytic policy: choose L, tiling, scan mode, and intra path for a shape.

    ``mean_abs_a`` estimates the mean |per-step decay| (a = dt·A); the per-chunk
    total is ~mean_abs_a·L, and the factored intra path is fp16-safe while that
    stays under INTRA_FACTORED_TOTAL_LIMIT. Defaults to 0.06 (this session's data).
    At the default T=4K shape this returns L=64/flat/factored — today's behavior.

    ``scan_mode`` is 'flat' by default. The hierarchical (O(C^1.5)) scan is
    CORRECT (CPU-validated) but currently BACKEND-BLOCKED on device (the 4D-batched
    block matmul can't restickify), so ``C_hier_threshold=None`` disables it for the
    device path. Growing L via ``_snap_L`` already shrinks C enough that flat is the
    right choice within T≤64K (e.g. T=64K → L=128 → C=512, flat = 349ms, correct).
    Pass an integer threshold only when experimenting with the hier path (CPU, or a
    future backend that supports block matmuls).
    """
    n_bh = B * (H // P)
    L = _snap_L(_l_star(T, N, P), T)

    # HARD scan-span constraint: the flat dense scan requires C = T/L ≤
    # MAX_FLAT_SCAN_CHUNKS or it fails to compile (see the constant's note). The
    # compute-optimal L* often violates this at long T (e.g. T=16384 → L*≈112 →
    # L=128 → C=128 ✗). Raise L to the smallest 64-multiple that keeps C ≤ cap.
    # This DOMINATES the L* preference — a slower-but-compiling L beats an OOM.
    L_scan_min = ((T // MAX_FLAT_SCAN_CHUNKS) + 63) // 64 * 64      # smallest L s.t. C≤cap
    if L < L_scan_min:
        L = L_scan_min if T % L_scan_min == 0 else _snap_L(float(L_scan_min), T)

    # fp16 guard: the factored intra path peaks at exp(|total|/2)≈exp(mean·L/2) and
    # overflows past INTRA_FACTORED_TOTAL_LIMIT. Below the scan-imposed L we could
    # cap L for safety; but when the scan constraint FORCES a large L we can't lower
    # it, so switch to the unconditionally-safe masked intra path instead.
    intra = "factored"
    limit_L = int(INTRA_FACTORED_TOTAL_LIMIT / max(mean_abs_a, 1e-6))
    if L > limit_L:
        capped = _snap_L(float(limit_L), T)
        if capped >= L_scan_min and mean_abs_a * capped < INTRA_FACTORED_TOTAL_LIMIT:
            L = capped                      # lowering L stays scan-legal → keep factored
        else:
            intra = "masked"                # forced large L (scan) → bounded masked path
    C = T // L

    # Hierarchical (sub-quadratic) scan: needed only when NO single L satisfies both
    # constraints at once — i.e. keeping C≤cap needs an L so large the intra L×L attn
    # itself overflows the span limit (measured: L≥512 → attn≥1GB). That happens at
    # T≥~32768. There the flat scan can't be made legal by L alone; the blocked scan
    # (small L for cheap attn + block the C dim so the scan never materializes C+1)
    # is the only path. Its 4D block matmul now compiles on merged main (probe:
    # rel 1.7e-3), unlike the pre-merge "can't restickify" block. Auto-select when C
    # would still exceed the cap after the intra-attn L ceiling, or when a threshold
    # is passed explicitly (CPU experiments).
    # The intra L×L attn intermediate (BH,C,L,L) grows as L²; past L≈256 it nears
    # the same 256 MiB per-core span limit. So when the scan constraint forces
    # L > MAX_INTRA_L, neither flat-scan L choice is legal (small L → C≥128 scan
    # OOM; large L → attn OOM) and only the blocked scan works.
    auto_hier = (C > MAX_FLAT_SCAN_CHUNKS) or (L > MAX_INTRA_L)
    scan_mode = "hierarchical" if (auto_hier or (
        C_hier_threshold is not None and C > C_hier_threshold)) else "flat"
    block_K = _pick_block_K(C) if scan_mode == "hierarchical" else 0
    # If no clean 64-aligned block size exists, hierarchical can't be stick-safe;
    # fall back to flat (still correct, just O(C²) — and may not compile at C≥128).
    if scan_mode == "hierarchical" and block_K == 0:
        scan_mode = "flat"
    return SSDConfig(L=L, bh_tiles=_bh_tiles_for(n_bh, L),
                     scan_mode=scan_mode, block_K=block_K, intra=intra)