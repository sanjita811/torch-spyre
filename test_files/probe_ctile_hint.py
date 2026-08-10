# Probe A: express the C-block scan as a COARSE-TILE HINT on C (like Spyre flash-attn
# tiles Q/KV), NOT a Python for-loop. The dense body is written ONCE; nested
# spyre_hint(num_tiles_per_dim={"C": ncb}) asks the compiler to tile the C output-row
# dim. This should (a) kill the per-block intra recompute (one intra, compiler-looped),
# (b) dodge the storage-offset-drop bug (compiler tiles, no Python slice of an input),
# (c) let the planner distribute C-blocks across cores.
#
# The scan matmul dm(BH,Cp,C)@cs(BH,C,NP) REDUCES over C, so its C is a reduction dim,
# not an output row — the hint tiles the OUTPUT C of intra/combine. We compare:
#   dense  : no C hint (baseline, C=128 ~2.2GB)
#   ctile  : nested BH + C coarse-tile hints, dense body (target)
# Usage: python probe_ctile_hint.py [C] [ncb] [mode=ctile|dense]
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

BH, L, N, P = 64, 64, 128, 64
C = int(sys.argv[1]) if len(sys.argv) > 1 else 128
NCB = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MODE = sys.argv[3] if len(sys.argv) > 3 else "ctile"
NP = N * P
CBLK = C // NCB
BH_TILES = 8

for nm, s in [("BH", BH), ("C", C), ("Ca", C), ("Ck", C), ("L", L), ("La", L),
              ("N", N), ("P", P), ("Lk", L), ("NP", NP), ("CBLK", CBLK), ("NCB", NCB)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
a = (torch.rand(BH, C, L, dtype=torch.float16) * -0.06)
cumsum_tri = torch.triu(torch.ones(L, L, dtype=torch.float16))
c_proj = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
b_proj = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
causal = torch.tril(torch.ones(L, L, dtype=torch.float16))
x = torch.randn(BH, C, L, P, dtype=torch.float16) * 0.1
_db = (torch.rand(BH, C, dtype=torch.float32) * -0.5).cumsum(-1)
_outer = _db.unsqueeze(-1) - _db.unsqueeze(-2)
_strict = torch.tril(torch.ones(C, C), diagonal=-1)
dm = (torch.exp(torch.clamp(_outer, max=0.0)) * _strict).to(torch.float16)   # (BH,C,C)


def _body(a_c, c_c, b_c, x_c, dm, cumsum_tri, causal_mask):
    """Dense intra+scan+combine, written ONCE (no Python C-loop)."""
    intra_cumsum = torch.matmul(a_c, cumsum_tri)
    total = a_c.sum(dim=-1, keepdim=True)
    shifted = intra_cumsum - 0.5 * total
    c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)
    b_scaled = b_c * torch.exp(-shifted).unsqueeze(-1)
    b_scaled_t = b_scaled.transpose(-1, -2)
    attn = torch.matmul(c_scaled, b_scaled_t) * causal_mask
    y_diag = torch.matmul(attn, x_c)
    half_tot = torch.exp(0.5 * total).unsqueeze(-1)
    chunk_states = torch.matmul(b_scaled_t, x_c * half_tot)
    cs_np = chunk_states.reshape(BH, C, NP)
    scan = torch.matmul(dm, cs_np)                          # reduces over C
    rolled = scan.reshape(BH, C, N, P)
    return torch.matmul(c_scaled * half_tot, rolled) + y_diag


def dense(a, cumsum_tri, c_proj, b_proj, causal_mask, x, dm):
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        return _body(a * 1.0, c_proj * 1.0, b_proj * 1.0, x * 1.0,
                     dm * 1.0, cumsum_tri, causal_mask)


def ctile(a, cumsum_tri, c_proj, b_proj, causal_mask, x, dm):
    # C is an OUTPUT dim of intra/combine (independent per chunk -> tile-able) but the
    # scan REDUCES over C (needs all of it). So use SEPARATE hint scopes: intra+combine
    # under BH+C tiling, the scan under BH-only. One body, no Python C-loop.
    # Whole body under one BH coarse-tile scope (proven to compile as 'dense'); add a
    # work_div={"C": NCB} ONLY on the combine (the biggest per-block matmul c@rolled) so
    # its C output is split across cores without a second loop nest or touching the scan.
    a_c = a * 1.0; c_c = c_proj * 1.0; b_c = b_proj * 1.0; x_c = x * 1.0; dm_c = dm * 1.0
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        intra_cumsum = torch.matmul(a_c, cumsum_tri)
        total = a_c.sum(dim=-1, keepdim=True)
        shifted = intra_cumsum - 0.5 * total
        half_tot = torch.exp(0.5 * total).unsqueeze(-1)
        c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)
        b_scaled = b_c * torch.exp(-shifted).unsqueeze(-1)
        b_scaled_t = b_scaled.transpose(-1, -2)
        attn = torch.matmul(c_scaled, b_scaled_t) * causal_mask
        y_diag = torch.matmul(attn, x_c)
        chunk_states = torch.matmul(b_scaled_t, x_c * half_tot)
        cs_np = chunk_states.reshape(BH, C, NP)
        scan = torch.matmul(dm_c, cs_np)
        rolled = scan.reshape(BH, C, N, P)
        with spyre_hint(work_div={"C": NCB}):
            return torch.matmul(c_scaled * half_tot, rolled) + y_diag


# fp32 reference
af = a.float(); ct = cumsum_tri.float()
ic = torch.matmul(af, ct); tot = af.sum(-1, keepdim=True); sh = ic - 0.5 * tot
csf = c_proj.float() * torch.exp(sh).unsqueeze(-1)
bsf = b_proj.float() * torch.exp(-sh).unsqueeze(-1); bstf = bsf.transpose(-1, -2)
ydf = torch.matmul(torch.matmul(csf, bstf) * causal.float(), x.float())
htf = torch.exp(0.5 * tot).unsqueeze(-1)
cst = torch.matmul(bstf, x.float() * htf)
scanf = torch.matmul(dm.float(), cst.reshape(BH, C, NP)).reshape(BH, C, N, P)
ref = (torch.matmul(csf * htf, scanf) + ydf).half()

fn = {"ctile": ctile, "dense": dense}[MODE]
ad = a.to("spyre"); ctd = cumsum_tri.to("spyre"); cd = c_proj.to("spyre")
bd = b_proj.to("spyre"); cmd = causal.to("spyre"); xd = x.to("spyre"); dd = dm.to("spyre")
name_tensor_dims(ad, ["BH", "C", "Lk"]); name_tensor_dims(ctd, ["Lk", "L"])
name_tensor_dims(cd, ["BH", "C", "L", "N"]); name_tensor_dims(bd, ["BH", "C", "La", "N"])
name_tensor_dims(cmd, ["L", "La"]); name_tensor_dims(xd, ["BH", "C", "La", "P"])
name_tensor_dims(dd, ["BH", "Ca", "C"])
out = torch.compile(fn, dynamic=False)(ad, ctd, cd, bd, cmd, xd, dd)
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"CTILE-HINT mode={MODE} C={C} NCB={NCB}  Y rel-L2={rel:.4e}  {'OK' if rel < 0.02 else 'WRONG'}")
