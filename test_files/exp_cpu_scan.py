# CPU-only validation of alternative inter-chunk scans for the SSD kernel.
# The current kernel uses a DENSE O(C^2) decay-matrix scan (explodes at long T).
# GPU-production Mamba uses an O(C) sequential recurrence. Here we validate that
#   (1) a carried recurrence,
#   (2) a two-level hierarchical (blocked) O(C^1.5) scan
# both reproduce the flat dense scan EXACTLY on CPU (fp32) and stay accurate in
# fp16 with device-like saturation — the prerequisite before any device probe.
import torch
import torch.nn.functional as F

import test_ssd as m


def build_inputs(B, T, nheads, P, N, G=1, seed=42):
    torch.manual_seed(seed)
    x_raw = torch.randn(B, T, nheads, P)
    dt = F.softplus(torch.randn(B, T, nheads) - 4 + torch.randn(nheads) * 0.1).clamp(0.0, float("inf"))
    a_log = -torch.exp(torch.rand(nheads))
    b_raw = torch.randn(B, T, G, N).repeat_interleave(nheads // G, dim=2)
    c_raw = torch.randn(B, T, G, N).repeat_interleave(nheads // G, dim=2)
    return x_raw * dt.unsqueeze(-1), dt * a_log, b_raw, c_raw


def flat_scan(chunk_decay, states, dtype):
    """Current kernel: dense (BH,C+1,C) decay matrix @ states. O(C^2)."""
    n_bh, C = chunk_decay.shape
    n, p = states.shape[-2:]
    dm = m._build_decay_cpu(chunk_decay, n_bh, C, dtype)          # (BH,C+1,C)
    scan = torch.matmul(dm, states.reshape(n_bh, C, n * p).to(dtype))
    rolled = scan[:, :C].reshape(n_bh, C, n, p)
    final = scan[:, C]
    return rolled, final


def carried_scan(chunk_decay, states, dtype):
    """GPU-production O(C) sequential recurrence:
       h[0]=0 ; h[i] = exp(chunk_decay[i-1])*h[i-1] + states[i-1]
       rolled[i]=h[i], final=h[C]. Sequential (C python steps here)."""
    n_bh, C = chunk_decay.shape
    n, p = states.shape[-2:]
    st = states.reshape(n_bh, C, n * p).to(dtype)
    dec = torch.exp(chunk_decay.to(dtype))                        # (BH,C) in (0,1]
    h = torch.zeros(n_bh, n * p, dtype=dtype)
    rolled = torch.empty(n_bh, C, n * p, dtype=dtype)
    for i in range(C):
        rolled[:, i] = h
        h = dec[:, i:i + 1] * h + st[:, i]
    final = h
    return rolled.reshape(n_bh, C, n, p), final


def hier_scan(chunk_decay, states, K, dtype):
    """Two-level blocked scan. nb blocks of size K.
       1. local: within-block prefix (dense K-scan per block).
       2. block totals -> top-level prefix over nb block-carries.
       3. add each block's incoming carry (decayed) to its local prefix.
       O(C*K + (C/K)^2). Validated to equal flat/carried."""
    n_bh, C = chunk_decay.shape
    n, p = states.shape[-2:]
    assert C % K == 0
    nb = C // K
    st = states.reshape(n_bh, C, n * p).to(dtype)
    cd = chunk_decay.to(dtype)

    # reshape to blocks: (BH, nb, K, D) and (BH, nb, K)
    stb = st.reshape(n_bh, nb, K, n * p)
    cdb = cd.reshape(n_bh, nb, K)

    # --- local prefix within each block (exclusive) + block total ---
    # local exclusive cumsum of decay within block
    cd_cs = torch.cumsum(cdb, dim=-1)                            # (BH,nb,K) inclusive
    cd_before = cd_cs - cdb                                      # exclusive
    # local dense: rolled_local[b,i] = sum_{j<i} exp(cd_before[i]-cd_cs[j]) st[j]
    diff = cd_before.unsqueeze(-1) - cd_cs.unsqueeze(-2)         # (BH,nb,K,K)
    strict = torch.tril(torch.ones(K, K, dtype=torch.bool), -1)
    Lrun = torch.exp(diff).masked_fill(~strict, 0.0).to(dtype)
    rolled_local = torch.matmul(Lrun, stb)                      # (BH,nb,K,D)
    # block total state: carry_out[b] = sum_j exp(blocktot - cd_cs[j]) st[j]
    blocktot = cd_cs[:, :, -1:]                                  # (BH,nb,1)
    Lend = torch.exp(blocktot - cd_cs).to(dtype)                # (BH,nb,K)
    block_state = torch.matmul(Lend.unsqueeze(-2), stb).squeeze(-2)  # (BH,nb,D)
    block_decay = cd_cs[:, :, -1]                                # (BH,nb) total per block

    # --- top-level prefix over blocks (carry entering each block) ---
    top_rolled, _ = flat_scan(block_decay, block_state.reshape(n_bh, nb, n, p), dtype)
    carry_in = top_rolled.reshape(n_bh, nb, n * p)              # (BH,nb,D) state entering block b

    # --- combine: rolled[b,i] = rolled_local[b,i] + exp(cd_before[i])*carry_in[b] ---
    decay_in = torch.exp(cd_before).to(dtype)                   # (BH,nb,K)
    rolled = rolled_local + decay_in.unsqueeze(-1) * carry_in.unsqueeze(-2)
    rolled = rolled.reshape(n_bh, C, n, p)
    # final = carry after last block
    _, top_final = flat_scan(block_decay, block_state.reshape(n_bh, nb, n, p), dtype)
    final = top_final
    return rolled, final


def run(B, T, H, P, N, L, K=None):
    nheads = H // P; C = T // L
    m.B, m.T, m.H, m.P, m.N, m.L, m.C, m.nheads, m.G = B, T, H, P, N, L, C, nheads, 1
    x_dt, a_raw, b_raw, c_raw = build_inputs(B, T, nheads, P, N)
    xc, ac, bc, cc = m._chunk_inputs(x_dt.half(), a_raw.half(), b_raw.half(), c_raw.half())
    n_bh = B * nheads
    a_f = ac.reshape(n_bh, C, L).half()
    chunk_decay = a_f.float().sum(-1)                            # (BH,C)
    # random-ish states standing in for the real chunk_states (scan is linear in them)
    torch.manual_seed(1); states = torch.randn(n_bh, C, N, P) * 0.1

    for dt in (torch.float32, torch.float16):
        rf, ff = flat_scan(chunk_decay, states, dt)
        rc, fc = carried_scan(chunk_decay, states, dt)
        ec = m.rel_l2(rc, rf); efc = m.rel_l2(fc, ff)
        line = f"L={L:4d} C={C:4d} {('fp32' if dt==torch.float32 else 'fp16')}: carried rolled={ec:.2e} final={efc:.2e}"
        if K:
            rh, fh = hier_scan(chunk_decay, states, K, dt)
            eh = m.rel_l2(rh, rf); efh = m.rel_l2(fh, ff)
            line += f" | hier(K={K}) rolled={eh:.2e} final={efh:.2e}"
        print(line)


if __name__ == "__main__":
    print("=== alternative scans vs flat dense (CPU) ===")
    run(2, 4096, 2048, 64, 128, 64, K=8)     # C=64
    run(2, 16384, 2048, 64, 128, 128, K=None) # C=128
    run(2, 16384, 2048, 64, 128, 64, K=16)   # C=256, hier K=16 (256=16*16)
    run(2, 65536, 2048, 64, 128, 128, K=None) # C=512 (carried only; hier K needs 64-mult for device)
