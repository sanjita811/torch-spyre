# CPU math prototypes for SSD kernel improvements. No Spyre, no compile.
#  (A) Saturation hypothesis: does clamping exp to fp16-max (modeling Spyre
#      saturation) turn the factored-path NaN at L>=128 into a small error?
#      -> tells us whether the CPU mirror can be trusted as a device oracle.
#  (B) Sequential carried scan (GPU-production O(C) recurrence) vs our dense
#      O(C^2) decay-matrix scan -- validate the math is identical on CPU.
import torch
import torch.nn.functional as F

import test_ssd as m


def build_inputs(B, T, nheads, P, N, G=1, seed=42):
    torch.manual_seed(seed)
    x_raw = torch.randn(B, T, nheads, P)
    dt_bias = torch.randn(nheads) * 0.1
    dt = F.softplus(torch.randn(B, T, nheads) - 4 + dt_bias).clamp(0.0, float("inf"))
    a_log = -torch.exp(torch.rand(nheads))
    b_raw = torch.randn(B, T, G, N).repeat_interleave(nheads // G, dim=2)
    c_raw = torch.randn(B, T, G, N).repeat_interleave(nheads // G, dim=2)
    return x_raw * dt.unsqueeze(-1), dt * a_log, b_raw, c_raw


FP16_MAX = 65504.0


def sat_exp(x):
    """exp that saturates to fp16-max like Spyre (instead of inf), then rounds
    to fp16. Models device behavior so masked-out overflow entries stay finite."""
    return torch.exp(x.float()).clamp(max=FP16_MAX).half()


def satclamp(t):
    """Clamp a matmul result to +-fp16max (models Spyre saturation, not inf)."""
    return t.float().clamp(min=-FP16_MAX, max=FP16_MAX).half()


def factored_intra_cpu(a, cumsum_tri, c_proj, b_proj, x, causal_mask, expf,
                       clamp_matmul=False):
    """Factored intra path (mirrors fused_kernel), parametrized by exp fn.
    clamp_matmul models device saturation of the attn matmul output."""
    intra_cumsum = torch.matmul(a, cumsum_tri)
    total = a.sum(dim=-1, keepdim=True)
    shifted = intra_cumsum - 0.5 * total
    c_scaled = c_proj * expf(shifted).unsqueeze(-1)
    b_scaled = b_proj * expf(-shifted).unsqueeze(-1)
    b_scaled_t = b_scaled.transpose(-1, -2)
    attn = torch.matmul(c_scaled, b_scaled_t)
    if clamp_matmul:
        attn = satclamp(attn)
    attn = attn * causal_mask
    y_diag = torch.matmul(attn, x)
    chunk_states = torch.matmul(b_scaled_t, x) * expf(0.5 * total).unsqueeze(-1)
    return y_diag, chunk_states, intra_cumsum, total


def expA_saturation(B, T, H, P, N, L):
    """(A) Compare torch.exp (inf->NaN) vs sat_exp (fp16max, device-like)."""
    nheads = H // P; C = T // L
    m.B, m.T, m.H, m.P, m.N, m.L, m.C, m.nheads, m.G = B, T, H, P, N, L, C, nheads, 1
    x_dt, a_raw, b_raw, c_raw = build_inputs(B, T, nheads, P, N)
    y_ref, _ = m.ssd_reference(x_dt, a_raw, b_raw, c_raw, L)
    xc, ac, bc, cc = m._chunk_inputs(x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half())
    n_bh = B * nheads
    x_f = xc.reshape(n_bh, C, L, P).half()
    a_f = ac.reshape(n_bh, C, L).half()
    b_f = bc.reshape(n_bh, C, L, N).half()
    c_f = cc.reshape(n_bh, C, L, N).half()
    tri = torch.triu(torch.ones(L, L, dtype=torch.float16))
    causal = torch.tril(torch.ones(L, L, dtype=torch.float16))
    chunk_decay = a_f.float().sum(-1)
    dm = m._build_decay_cpu(chunk_decay, n_bh, C, torch.float16)
    for name, expf, clampmm in [
        ("torch.exp            ", lambda t: torch.exp(t).half(), False),
        ("sat_exp              ", sat_exp, False),
        ("sat_exp+satclamp(mm) ", sat_exp, True)]:
        y_diag, cs, ic, tot = factored_intra_cpu(
            a_f, tri, c_f, b_f, x_f, causal, expf, clamp_matmul=clampmm)
        scan = torch.matmul(dm, cs.reshape(n_bh, C, N * P))
        rolled = scan[:, :C].reshape(n_bh, C, N, P)
        y_off = torch.matmul(c_f, rolled) * expf(ic).unsqueeze(-1)
        y = (y_off + y_diag).reshape(B, nheads, C, L, P)[:, :, :C]
        y = y.permute(0, 2, 3, 1, 4).contiguous().reshape(B, C * L, nheads, P).half()
        finite = torch.isfinite(y).all().item()
        ey = m.rel_l2(y, y_ref) if finite else float("nan")
        print(f"  L={L:4d} factored via {name}: finite={finite}  Y={ey:.4f}")


if __name__ == "__main__":
    print("=== (A) saturation hypothesis: torch.exp vs device-like sat_exp ===")
    for L in (64, 128, 256):
        expA_saturation(2, 4096, 2048, 64, 128, L)
