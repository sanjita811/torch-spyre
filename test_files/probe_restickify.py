# Restickify/recompute reduction study for fused_kernel_cblock. output_8192.log showed
# the intra (shifted/total/c_scaled/b_scaled/y_diag) is computed full-C for chunk_states
# AND recomputed per block from input slices (storage-offset workaround). Test whether
# computing it ONCE full-C and slicing the produced intermediates per block is (a) legal
# and (b) fewer restickifies / faster.
#
#   recompute : current kernel — per-block recompute from input slices (baseline)
#   shared    : full-C intra once; per block SLICE produced c_scaled/y_diag (offset test)
#
# Usage: python probe_restickify.py [method] [C]
import sys
import time
import torch
import torch_spyre  # noqa: F401
from torch_spyre._inductor import spyre_hint

try:
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim, name_tensor_dims)
except (ImportError, ModuleNotFoundError):
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim, name_tensor_dims)

METHOD = sys.argv[1] if len(sys.argv) > 1 else "recompute"
C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
BH, L, N, P = 64, 64, 128, 64
CBLK = 64
NCB = C // CBLK
BH_TILES = 16
for nm, s in [("BH", BH), ("C", C), ("Ca", C), ("L", L), ("La", L), ("N", N),
              ("P", P), ("Lk", L)]:
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
dm = (torch.exp(torch.clamp(_outer, max=0.0)) * torch.tril(torch.ones(C, C), -1)).half()


def recompute(a, ct, c_p, b_p, cm, x, dm):
    """Baseline: per-block recompute from input slices (the current kernel)."""
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        a_c = a * 1.0; c_c = c_p * 1.0; b_c = b_p * 1.0; x_c = x * 1.0
        dr = dm * 1.0
        total = a_c.sum(dim=-1, keepdim=True)
        shifted = torch.matmul(a_c, ct) - 0.5 * total
        b_scaled_t = (b_c * torch.exp(-shifted).unsqueeze(-1)).transpose(-1, -2)
        cs_np = torch.matmul(b_scaled_t, x_c * torch.exp(0.5 * total).unsqueeze(-1)
                             ).reshape(BH, C, N * P)
        blks = []
        for j in range(NCB):
            sl = slice(j * CBLK, (j + 1) * CBLK)
            a_b = a_c[:, sl]; c_b = c_c[:, sl]; b_b = b_c[:, sl]; x_b = x_c[:, sl]
            tot_b = a_b.sum(dim=-1, keepdim=True)
            sh_b = torch.matmul(a_b, ct) - 0.5 * tot_b
            csc_b = c_b * torch.exp(sh_b).unsqueeze(-1)
            bsct_b = (b_b * torch.exp(-sh_b).unsqueeze(-1)).transpose(-1, -2)
            ydiag_b = torch.matmul(torch.matmul(csc_b, bsct_b) * cm, x_b)
            scan_b = torch.matmul(dr[:, sl, :], cs_np).reshape(BH, CBLK, N, P)
            yoff_b = torch.matmul(csc_b, scan_b) * torch.exp(0.5 * tot_b).unsqueeze(-1)
            blks.append(yoff_b + ydiag_b)
        return blks[0] if NCB == 1 else torch.stack(blks, dim=1).reshape(BH, C, L, P)


def shared(a, ct, c_p, b_p, cm, x, dm):
    """Compute the intra ONCE full-C (c_scaled, y_diag, half_tot), slice per block."""
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        a_c = a * 1.0; c_c = c_p * 1.0; b_c = b_p * 1.0; x_c = x * 1.0
        dr = dm * 1.0
        total = a_c.sum(dim=-1, keepdim=True)
        shifted = torch.matmul(a_c, ct) - 0.5 * total
        half_tot = torch.exp(0.5 * total).unsqueeze(-1)
        c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)          # (BH,C,L,N) full
        b_scaled_t = (b_c * torch.exp(-shifted).unsqueeze(-1)).transpose(-1, -2)
        attn = torch.matmul(c_scaled, b_scaled_t) * cm
        y_diag = torch.matmul(attn, x_c)                           # (BH,C,L,P) full
        cs_np = torch.matmul(b_scaled_t, x_c * half_tot).reshape(BH, C, N * P)
        blks = []
        for j in range(NCB):
            sl = slice(j * CBLK, (j + 1) * CBLK)
            csc_b = c_scaled[:, sl]                                 # SLICE produced
            ht_b = half_tot[:, sl]
            ydiag_b = y_diag[:, sl]                                 # SLICE produced
            scan_b = torch.matmul(dr[:, sl, :], cs_np).reshape(BH, CBLK, N, P)
            yoff_b = torch.matmul(csc_b, scan_b) * ht_b
            blks.append(yoff_b + ydiag_b)
        return blks[0] if NCB == 1 else torch.stack(blks, dim=1).reshape(BH, C, L, P)


# fp32 ref
af = a.float(); ctf = cumsum_tri.float()
tot = af.sum(-1, keepdim=True); sh = torch.matmul(af, ctf) - 0.5 * tot
csf = c_proj.float() * torch.exp(sh).unsqueeze(-1)
bstf = (b_proj.float() * torch.exp(-sh).unsqueeze(-1)).transpose(-1, -2)
htf = torch.exp(0.5 * tot).unsqueeze(-1)
ydf = torch.matmul(torch.matmul(csf, bstf) * causal.float(), x.float())
csnp = torch.matmul(bstf, x.float() * htf).reshape(BH, C, N * P)
scanf = torch.matmul(dm.float(), csnp).reshape(BH, C, N, P)
ref = (torch.matmul(csf, scanf) * htf + ydf).half()

def scanblock(a, ct, c_p, b_p, cm, x, dm):
    """Block ONLY the scan (the 2.2GB read-copy); everything else full-C. Write each
    block's rolled_states into a preallocated (BH,C,N,P) buffer via copy_f, then do the
    combine c_scaled@rolled and y_off*half_tot as ONE full-C matmul — NO torch.stack
    (kills the 67MB stack terminal). Combines shared-intra + flash-style copy_f carry."""
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        a_c = a * 1.0; c_c = c_p * 1.0; b_c = b_p * 1.0; x_c = x * 1.0
        dr = dm * 1.0
        total = a_c.sum(dim=-1, keepdim=True)
        shifted = torch.matmul(a_c, ct) - 0.5 * total
        half_tot = torch.exp(0.5 * total).unsqueeze(-1)
        c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)
        b_scaled_t = (b_c * torch.exp(-shifted).unsqueeze(-1)).transpose(-1, -2)
        y_diag = torch.matmul(torch.matmul(c_scaled, b_scaled_t) * cm, x_c)
        cs_np = torch.matmul(b_scaled_t, x_c * half_tot).reshape(BH, C, N * P)
        rolled = None
        for j in range(NCB):
            sl = slice(j * CBLK, (j + 1) * CBLK)
            scan_b = torch.matmul(dr[:, sl, :], cs_np).reshape(BH, CBLK, N, P)
            part = scan_b if NCB == 1 else torch.nn.functional.pad(
                scan_b, (0, 0, 0, 0, j * CBLK, C - (j + 1) * CBLK))
            rolled = part if rolled is None else torch.ops.spyre.copy_f(rolled + part, rolled)
        y_off = torch.matmul(c_scaled, rolled) * half_tot          # ONE full-C combine
        return y_off + y_diag


fn = {"recompute": recompute, "shared": shared, "scanblock": scanblock}[METHOD]
ad = a.to("spyre"); ctd = cumsum_tri.to("spyre"); cd = c_proj.to("spyre")
bd = b_proj.to("spyre"); cmd = causal.to("spyre"); xd = x.to("spyre"); dd = dm.to("spyre")
name_tensor_dims(ad, ["BH", "C", "Lk"]); name_tensor_dims(ctd, ["Lk", "L"])
name_tensor_dims(cd, ["BH", "C", "L", "N"]); name_tensor_dims(bd, ["BH", "C", "La", "N"])
name_tensor_dims(cmd, ["L", "La"]); name_tensor_dims(xd, ["BH", "C", "La", "P"])
name_tensor_dims(dd, ["BH", "Ca", "C"])
cfn = torch.compile(fn, dynamic=False)
out = cfn(ad, ctd, cd, bd, cmd, xd, dd)
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"RESTICKIFY method={METHOD} C={C} NCB={NCB}  rel-L2={rel:.4e}  "
      f"{'OK' if rel < 0.02 else 'WRONG'}")
