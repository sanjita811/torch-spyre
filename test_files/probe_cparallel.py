# C-parallelism study for the SSD intra stage (the dominant, per-chunk-independent
# compute). At C=128 today we only tile BH=64; the GPU tiles BH*C. Test 4 methods for
# exposing C as parallel work, measure compile success + correctness. Intra-only (the
# scan couples C and stays separate).
#
#   base   : spyre_hint({"BH": bt})                      -- today (BH-only)
#   fold   : reshape (BH,C,..)->(BH*C,..), tile the folded batch dim G=BH*C
#   nested : with hint({"BH":bt}):  with hint({"C":ct}): ...   (two-level tiling)
#   conly  : spyre_hint({"C": ct})                        -- tile C alone
#
# Usage: python probe_cparallel.py [method] [C]
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

import time
METHOD = sys.argv[1] if len(sys.argv) > 1 else "base"
C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
BH = int(sys.argv[3]) if len(sys.argv) > 3 else 64
L, N, P = 64, 128, 64
G = BH * C
BH_TILES = min(16, BH)
C_TILES = min(16, C)

for nm, s in [("BH", BH), ("C", C), ("L", L), ("La", L), ("N", N), ("P", P),
              ("Lk", L), ("G", G)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
a = (torch.rand(BH, C, L, dtype=torch.float16) * -0.06)
cumsum_tri = torch.triu(torch.ones(L, L, dtype=torch.float16))
c_proj = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
b_proj = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
causal = torch.tril(torch.ones(L, L, dtype=torch.float16))
x = torch.randn(BH, C, L, P, dtype=torch.float16) * 0.1


def _intra_body(a_c, cumsum_tri, c_c, b_c, causal_mask, x_c):
    """intra stage: y_diag + chunk_states, batched over the leading dim(s)."""
    intra_cumsum = torch.matmul(a_c, cumsum_tri)
    total = a_c.sum(dim=-1, keepdim=True)
    shifted = intra_cumsum - 0.5 * total
    c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)
    b_scaled = b_c * torch.exp(-shifted).unsqueeze(-1)
    b_scaled_t = b_scaled.transpose(-1, -2)
    attn = torch.matmul(c_scaled, b_scaled_t) * causal_mask
    y_diag = torch.matmul(attn, x_c)
    chunk_states = torch.matmul(b_scaled_t, x_c * torch.exp(0.5 * total).unsqueeze(-1))
    return y_diag, chunk_states


def base(a, ct, c_p, b_p, cm, x):
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        return _intra_body(a * 1.0, ct, c_p * 1.0, b_p * 1.0, cm, x * 1.0)


def conly(a, ct, c_p, b_p, cm, x):
    with spyre_hint(num_tiles_per_dim={"C": C_TILES}):
        return _intra_body(a * 1.0, ct, c_p * 1.0, b_p * 1.0, cm, x * 1.0)


def nested(a, ct, c_p, b_p, cm, x):
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        with spyre_hint(num_tiles_per_dim={"C": C_TILES}):
            return _intra_body(a * 1.0, ct, c_p * 1.0, b_p * 1.0, cm, x * 1.0)


def fold(a, ct, c_p, b_p, cm, x):
    # Fold (BH,C,...) -> (G=BH*C, ...) so a single batch dim G carries all parallel work.
    with spyre_hint(num_tiles_per_dim={"G": BH_TILES}):
        a_f = (a * 1.0).reshape(G, L)
        c_f = (c_p * 1.0).reshape(G, L, N)
        b_f = (b_p * 1.0).reshape(G, L, N)
        x_f = (x * 1.0).reshape(G, L, P)
        yd, cs = _intra_body(a_f, ct, c_f, b_f, cm, x_f)
        return yd.reshape(BH, C, L, P), cs.reshape(BH, C, N, P)


# fp32 reference
af = a.float(); ctf = cumsum_tri.float()
ic = torch.matmul(af, ctf); tot = af.sum(-1, keepdim=True); sh = ic - 0.5 * tot
csf = c_proj.float() * torch.exp(sh).unsqueeze(-1)
bsf = b_proj.float() * torch.exp(-sh).unsqueeze(-1); bstf = bsf.transpose(-1, -2)
yd_ref = torch.matmul(torch.matmul(csf, bstf) * causal.float(), x.float())
cs_ref = torch.matmul(bstf, x.float() * torch.exp(0.5 * tot).unsqueeze(-1))

fn = {"base": base, "conly": conly, "nested": nested, "fold": fold}[METHOD]
ad = a.to("spyre"); ctd = cumsum_tri.to("spyre"); cd = c_proj.to("spyre")
bd = b_proj.to("spyre"); cmd = causal.to("spyre"); xd = x.to("spyre")
if METHOD == "fold":
    name_tensor_dims(ad, ["BH", "C", "Lk"]); name_tensor_dims(cd, ["BH", "C", "L", "N"])
    name_tensor_dims(bd, ["BH", "C", "La", "N"]); name_tensor_dims(xd, ["BH", "C", "La", "P"])
else:
    name_tensor_dims(ad, ["BH", "C", "Lk"]); name_tensor_dims(cd, ["BH", "C", "L", "N"])
    name_tensor_dims(bd, ["BH", "C", "La", "N"]); name_tensor_dims(xd, ["BH", "C", "La", "P"])
name_tensor_dims(ctd, ["Lk", "L"]); name_tensor_dims(cmd, ["L", "La"])

cfn = torch.compile(fn, dynamic=False)
yd, cs = cfn(ad, ctd, cd, bd, cmd, xd)          # warmup/compile
yd.cpu(); cs.cpu()
t0 = time.perf_counter()
for _ in range(10):
    yd, cs = cfn(ad, ctd, cd, bd, cmd, xd)
yd.cpu(); cs.cpu()
ms = (time.perf_counter() - t0) / 10 * 1e3
ry = (yd.cpu().float() - yd_ref).norm().item() / (yd_ref.norm().item() + 1e-12)
rc = (cs.cpu().float() - cs_ref).norm().item() / (cs_ref.norm().item() + 1e-12)
ok = ry < 0.02 and rc < 0.02
print(f"CPARALLEL method={METHOD} C={C} BH={BH}  y_diag={ry:.4e} cs={rc:.4e}  "
      f"{ms:.1f}ms/iter  {'OK' if ok else 'WRONG'}")
