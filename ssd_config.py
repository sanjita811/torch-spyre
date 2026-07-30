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
    """Divisor-based BH tile count (matches the historical BH_TILES rule)."""
    return next(t for t in ([16, 8, 4, 2, 1] if L > 64 else [4, 2, 1]) if n_bh % t == 0)


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
    # fp16 guard: if the factored path would overflow at this L, cap L to the
    # largest 64-multiple that stays safe; if even L=64 is unsafe, keep L=64 and
    # switch to the masked fallback (bounded, always correct).
    intra = "factored"
    limit_L = int(INTRA_FACTORED_TOTAL_LIMIT / max(mean_abs_a, 1e-6))
    if L > limit_L:
        capped = _snap_L(float(limit_L), T)
        if capped >= 64 and mean_abs_a * capped < INTRA_FACTORED_TOTAL_LIMIT:
            L = capped
        else:
            L = 64
            intra = "masked"
    C = T // L
    # Hierarchical scan disabled by default (backend-blocked on device); only when
    # a threshold is explicitly passed (CPU experiments / future backend support).
    scan_mode = "hierarchical" if (C_hier_threshold is not None and C > C_hier_threshold) else "flat"
    block_K = _pick_block_K(C) if scan_mode == "hierarchical" else 0
    # If no clean 64-aligned block size exists, hierarchical can't be stick-safe;
    # fall back to flat (still correct, just O(C²)).
    if scan_mode == "hierarchical" and block_K == 0:
        scan_mode = "flat"
    return SSDConfig(L=L, bh_tiles=_bh_tiles_for(n_bh, L),
                     scan_mode=scan_mode, block_K=block_K, intra=intra)