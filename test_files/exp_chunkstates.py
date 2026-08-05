# Experiment: is the current chunk_states computation the cheapest on the CURRENT
# backend, or does a different decay-placement / sharing win? Target:
#   states[n,p] = sum_l B[l,n] * exp(total - g_l) * X[l,p]
# Variants (all compute the SAME thing; differ in ops/layout):
#   v0 current : b_scaled = B*exp(total/2-g) SHARED with attn; states=(b_scaled^T@X)*exp(total/2)
#   v1 bfold   : b_decayed = B*exp(total-g) (separate); states = b_decayed^T @ X  (no share)
#   v2 xfold   : x_dec = X*exp(total-g); states = B_raw^T @ x_dec
#   v3 scalarX : share b_scaled^T@X, apply exp(total/2) scalar to X pre-matmul instead of after
# Metric (profiling-independent): #restickify + #matmul in generated code, + correctness.
# One compile per process. Usage: python exp_chunkstates.py <v0|v1|v2|v3>
import sys
import torch
import torch_spyre  # noqa: F401
from torch_spyre._inductor import spyre_hint

try:
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim, name_tensor_dims)
except (ImportError, ModuleNotFoundError):
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim, name_tensor_dims)

BH, C, L, N, P = 64, 64, 64, 128, 64
for nm, s in [("BH", BH), ("C", C), ("L", L), ("La", L), ("Lk", L),
              ("N", N), ("P", P)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
a = (torch.randn(BH, C, L, dtype=torch.float16) * 0.02).clamp(max=0)
b = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
c = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
x = torch.randn(BH, C, L, P, dtype=torch.float16) * 0.1
tri = torch.triu(torch.ones(L, L, dtype=torch.float16))
causal = torch.tril(torch.ones(L, L, dtype=torch.float16))

variant = sys.argv[1] if len(sys.argv) > 1 else "v0"


def cpu_ref():
    g = a.float().cumsum(-1)
    total = a.float().sum(-1, keepdim=True)
    decay = torch.exp(total - g)                        # (BH,C,L)
    states = torch.einsum("bln,bl,blp->bnp",
                          b.float().reshape(BH * C, L, N),
                          decay.reshape(BH * C, L),
                          x.float().reshape(BH * C, L, P)).reshape(BH, C, N, P)
    return states


REF = cpu_ref()


def k(a, c, b, x, tri, causal):
    with spyre_hint(num_tiles_per_dim={"BH": 16}):
        a_c = a * 1.0; c_c = c * 1.0; b_c = b * 1.0; x_c = x * 1.0
        g = torch.matmul(a_c, tri)                      # (BH,C,L)
        total = a_c.sum(dim=-1, keepdim=True)
        shifted = g - 0.5 * total
        c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)
        b_scaled = b_c * torch.exp(-shifted).unsqueeze(-1)
        b_scaled_t = b_scaled.transpose(-1, -2)
        attn = torch.matmul(c_scaled, b_scaled_t) * causal   # keep attn to preserve sharing context
        y_diag = torch.matmul(attn, x_c)
        if variant == "v0":
            states = torch.matmul(b_scaled_t, x_c) * torch.exp(0.5 * total).unsqueeze(-1)
        elif variant == "v1":
            b_dec = (b_c * torch.exp(total - g).unsqueeze(-1)).transpose(-1, -2)
            states = torch.matmul(b_dec, x_c)
        elif variant == "v2":
            x_dec = x_c * torch.exp(total - g).unsqueeze(-1)
            states = torch.matmul(b_c.transpose(-1, -2), x_dec)
        else:  # v3
            x_s = x_c * torch.exp(0.5 * total).unsqueeze(-1)
            states = torch.matmul(b_scaled_t, x_s)
        return y_diag, states


ad = a.to("spyre"); cd = c.to("spyre"); bd = b.to("spyre"); xd = x.to("spyre")
trid = tri.to("spyre"); causd = causal.to("spyre")
name_tensor_dims(ad, ["BH", "C", "Lk"]); name_tensor_dims(trid, ["Lk", "L"])
name_tensor_dims(cd, ["BH", "C", "L", "N"]); name_tensor_dims(bd, ["BH", "C", "La", "N"])
name_tensor_dims(xd, ["BH", "C", "La", "P"]); name_tensor_dims(causd, ["L", "La"])
_, states = torch.compile(k, dynamic=False)(ad, cd, bd, xd, trid, causd)
rel = (states.cpu().float() - REF).norm().item() / (REF.norm().item() + 1e-12)
print(f"VARIANT={variant}  states rel-L2={rel:.4e}  {'OK' if rel < 0.02 else 'WRONG'}")
