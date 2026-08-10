# C-row-blocked scan, factored form. Block the scan OUTPUT-ROW dim (chunk index C):
# a BATCH dim of both the scan matmul (dm's output row) and the combine matmul
# (c_scaled's chunk) -- never a stick dim, never a reduction dim -> slicing it is
# offset-free and dodges the N-stick wall that kills all N-blocking.
#
# To keep named-dim propagation happy, C is FACTORED on the inputs as (NCB, CBLK)
# with BOTH dims declared+named, so a block is a clean `select` on the NCB dim
# (yielding a CBLK-named dim) rather than a size-mismatched slice of C. CBLK is a
# whole number of sticks (mult of 64). This is exactly the form that lets C exceed
# the dense-scan 64-cap: run this at C=128 (dense scan hard-walls there).
#
# Usage: python probe_cblock_scan.py [C] [CBLK] [mode=slicescatter|cat|dense]
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
CBLK = int(sys.argv[2]) if len(sys.argv) > 2 else 64        # rows per block, mult of 64
MODE = sys.argv[3] if len(sys.argv) > 3 else "stack"
NP = N * P
NCB = C // CBLK
BH_TILES = 8
assert C % CBLK == 0 and CBLK % 64 == 0

for nm, s in [("BH", BH), ("C", C), ("Ca", C), ("L", L), ("La", L),
              ("N", N), ("P", P), ("Lk", L), ("NP", NP),
              ("CBLK", CBLK), ("CBa", CBLK), ("NCB", NCB)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
a = (torch.rand(BH, C, L, dtype=torch.float16) * -0.06)
cumsum_tri = torch.triu(torch.ones(L, L, dtype=torch.float16))
c_proj = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
b_proj = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1
causal = torch.tril(torch.ones(L, L, dtype=torch.float16))
x = torch.randn(BH, C, L, P, dtype=torch.float16) * 0.1
# Structured decay matrix like the real kernel: exp(clamp(outer,max=0)) * strict_lower,
# bounded to (0,1], strictly lower-triangular. (Random dm is unrepresentative of the
# real scan and can exaggerate fp16 device error.)
_db = (torch.rand(BH, C, dtype=torch.float32) * -0.5).cumsum(-1)       # decreasing cumsum
_outer = _db.unsqueeze(-1) - _db.unsqueeze(-2)                          # (BH,C,C)
_strict = torch.tril(torch.ones(C, C), diagonal=-1)
dm = (torch.exp(torch.clamp(_outer, max=0.0)) * _strict).to(torch.float16)  # (BH,C,C)


def core_fn(a, cumsum_tri, c_proj, b_proj, causal_mask, x, dm):
    with spyre_hint(num_tiles_per_dim={"BH": BH_TILES}):
        a_c = a * 1.0; c_c = c_proj * 1.0; b_c = b_proj * 1.0; x_c = x * 1.0
        dm_c = dm * 1.0   # pre-copy: slicing a raw graph INPUT drops storage_offset
        intra_cumsum = torch.matmul(a_c, cumsum_tri)
        total = a_c.sum(dim=-1, keepdim=True)
        shifted = intra_cumsum - 0.5 * total
        c_scaled = c_c * torch.exp(shifted).unsqueeze(-1)          # (BH,C,L,N)
        b_scaled = b_c * torch.exp(-shifted).unsqueeze(-1)
        b_scaled_t = b_scaled.transpose(-1, -2)
        attn = torch.matmul(c_scaled, b_scaled_t) * causal_mask
        y_diag = torch.matmul(attn, x_c)
        half_tot = torch.exp(0.5 * total).unsqueeze(-1)            # (BH,C,1,1)
        # fold half_tot into OPERANDS (never scale a matmul output by a broadcast):
        x_scaled = x_c * half_tot
        chunk_states = torch.matmul(b_scaled_t, x_scaled)          # (BH,C,N,P)
        c_scaled_ht = c_scaled * half_tot                          # (BH,C,L,N)
        cs_np = chunk_states.reshape(BH, C, NP)                    # (BH,C,NP)

        if MODE == "dense":
            scan = torch.matmul(dm, cs_np)                         # (BH,C,NP) read-copy (C,NP,C)
            rolled = scan.reshape(BH, C, N, P)
            return torch.matmul(c_scaled_ht, rolled) + y_diag

        # Block the scan output-row dim C. Slicing PRODUCED intermediates at a
        # nonzero C-offset drops the storage_offset (customops.py: block 1 read
        # the storage base -> wrong). So recompute each block's c_scaled_ht and
        # y_diag from INPUT slices (a,c_proj,b_proj,x,dm sliced on C) instead;
        # cs_np stays full-C (the scan couples all chunks).
        blks = []
        for j in range(NCB):
            s = slice(j * CBLK, (j + 1) * CBLK)
            a_b = a_c[:, s]; c_b = c_c[:, s]; b_b = b_c[:, s]; x_b = x_c[:, s]
            ic_b = torch.matmul(a_b, cumsum_tri)
            tot_b = a_b.sum(dim=-1, keepdim=True)
            sh_b = ic_b - 0.5 * tot_b
            csc_b = c_b * torch.exp(sh_b).unsqueeze(-1)           # (BH,CBLK,L,N)
            bsc_b = b_b * torch.exp(-sh_b).unsqueeze(-1)
            bsct_b = bsc_b.transpose(-1, -2)
            ht_b = torch.exp(0.5 * tot_b).unsqueeze(-1)
            attn_b = torch.matmul(csc_b, bsct_b) * causal_mask
            ydiag_b = torch.matmul(attn_b, x_b)                   # (BH,CBLK,L,P)
            csc_ht_b = csc_b * ht_b

            dm_blk = dm_c[:, s, :]                                  # (BH,CBLK,C) input slice
            scan_blk = torch.matmul(dm_blk, cs_np)               # (BH,CBLK,NP)
            rolled_blk = scan_blk.reshape(BH, CBLK, N, P)
            yoff_blk = torch.matmul(csc_ht_b, rolled_blk)        # (BH,CBLK,L,P)
            y_blk = yoff_blk + ydiag_b
            blks.append(y_blk)
    # ---- reassembly OUTSIDE the spyre_hint block ----
    if NCB == 1:
        return blks[0]
    if MODE.startswith("blk"):
        return blks[int(MODE[3:])]                        # debug: return one block only
    if MODE == "cat":
        return torch.cat(blks, dim=1)
    return torch.stack(blks, dim=1).reshape(BH, C, L, P)  # dim MERGE (NCB,CBLK)->C


# fp32 reference
af = a.float(); ct = cumsum_tri.float()
ic = torch.matmul(af, ct); tot = af.sum(-1, keepdim=True); sh = ic - 0.5 * tot
csf = c_proj.float() * torch.exp(sh).unsqueeze(-1)
bsf = b_proj.float() * torch.exp(-sh).unsqueeze(-1)
bstf = bsf.transpose(-1, -2)
ydf = torch.matmul(torch.matmul(csf, bstf) * causal.float(), x.float())
ht_f = torch.exp(0.5 * tot).unsqueeze(-1)
cst = torch.matmul(bstf, x.float() * ht_f)
scanf = torch.matmul(dm.float(), cst.reshape(BH, C, NP))
rollf = scanf.reshape(BH, C, N, P)
ref = (torch.matmul(csf * ht_f, rollf) + ydf).half()

ad = a.to("spyre"); ctd = cumsum_tri.to("spyre"); cd = c_proj.to("spyre")
bd = b_proj.to("spyre"); cmd = causal.to("spyre"); xd = x.to("spyre"); dd = dm.to("spyre")
name_tensor_dims(ad, ["BH", "C", "Lk"]); name_tensor_dims(ctd, ["Lk", "L"])
name_tensor_dims(cd, ["BH", "C", "L", "N"]); name_tensor_dims(bd, ["BH", "C", "La", "N"])
name_tensor_dims(cmd, ["L", "La"]); name_tensor_dims(xd, ["BH", "C", "La", "P"])
name_tensor_dims(dd, ["BH", "Ca", "C"])
out = torch.compile(core_fn, dynamic=False)(ad, ctd, cd, bd, cmd, xd, dd)
if MODE.startswith("blk"):
    j = int(MODE[3:])
    ref = ref[:, j * CBLK:(j + 1) * CBLK]                 # compare only that block
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"CBLOCK-SCAN mode={MODE} C={C} CBLK={CBLK} NCB={NCB}  Y rel-L2={rel:.4e}  {'OK' if rel < 0.02 else 'WRONG'}")
