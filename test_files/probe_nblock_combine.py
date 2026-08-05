# Device probe: N-blocked scan + y_off accumulation (flash reduction pattern).
# Mirrors the FULL scan->rolled->y_off dataflow of fused_kernel, but blocks the
# N dim (of NP) and accumulates y_off over N-blocks via copy_f — avoiding the
# 545MB (BH,C+1,NP,C) scan read-copy AND the 'cannot reorder unhinted cat' error
# (accumulation is hint-safe; cat is not, inside the group).
#
# y_off = c_scaled(BH,C,L,N) @ rolled(BH,C,N,P) contracts N -> each N-block is a
# partial sum. Nblk*P must be stick-aligned (mult of 64); P=64 so Nblk mult of 1.
import sys
import torch
import torch_spyre  # noqa: F401
from torch_spyre._inductor import spyre_hint
try:
    from torch_spyre._inductor.wsr.propagate_named_dims import (declare_tensor_dim, name_tensor_dims)
except (ImportError, ModuleNotFoundError):
    from torch_spyre._inductor.propagate_named_dims import (declare_tensor_dim, name_tensor_dims)

BH, C, L, N, P = 64, 64, 64, 128, 64
Cp = C + 1
NBLK = int(sys.argv[1]) if len(sys.argv) > 1 else 64      # N-block (32/64); Nblk*P stick-aligned
nb = N // NBLK
for nm, s in [("BH", BH), ("C", C), ("Cp", Cp), ("L", L), ("N", N), ("P", P)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
dm = torch.randn(BH, Cp, C, dtype=torch.float16) * 0.1        # decay_matrix
cs = torch.randn(BH, C, N, P, dtype=torch.float16) * 0.1      # chunk_states (BH,C,N,P)
csc = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1     # c_scaled (BH,C,L,N)

# reference: dense scan + full y_off
scan_ref = torch.matmul(dm.float(), cs.reshape(BH, C, N * P).float())   # (BH,Cp,NP)
rolled_ref = scan_ref[:, :C].reshape(BH, C, N, P)
yoff_ref = torch.matmul(csc.float(), rolled_ref).half()                 # (BH,C,L,P)


def dense(dm, cs, csc):
    with spyre_hint(num_tiles_per_dim={"BH": 8}):
        dm_c = dm * 1.0; cs_c = cs * 1.0; csc_c = csc * 1.0
        scan = torch.matmul(dm_c, cs_c.reshape(BH, C, N * P))
        rolled = scan[:, :C].reshape(BH, C, N, P)
        return torch.matmul(csc_c, rolled)


def nblock(dm, cs, csc):
    with spyre_hint(num_tiles_per_dim={"BH": 8}):
        dm_c = dm * 1.0; cs_c = cs * 1.0; csc_c = csc * 1.0
        yoff = None
        for b in range(nb):
            n0 = b * NBLK
            cs_blk = cs_c[:, :, n0:n0 + NBLK, :].reshape(BH, C, NBLK * P)  # (BH,C,Nblk*P)
            scan_blk = torch.matmul(dm_c, cs_blk)                          # (BH,Cp,Nblk*P) small
            rolled_blk = scan_blk[:, :C].reshape(BH, C, NBLK, P)
            csc_blk = csc_c[:, :, :, n0:n0 + NBLK]                         # (BH,C,L,Nblk)
            part = torch.matmul(csc_blk, rolled_blk)                       # (BH,C,L,P) partial
            if yoff is None:
                yoff = part
            else:
                yoff = torch.ops.spyre.copy_f(yoff + part, yoff)
        return yoff


mode = sys.argv[2] if len(sys.argv) > 2 else "nblock"
dd = dm.to("spyre"); cd = cs.to("spyre"); cscd = csc.to("spyre")
name_tensor_dims(dd, ["BH", "Cp", "C"])
name_tensor_dims(cd, ["BH", "C", "N", "P"])
name_tensor_dims(cscd, ["BH", "C", "L", "N"])
fn = {"nblock": nblock, "dense": dense}[mode]
out = torch.compile(fn, dynamic=False)(dd, cd, cscd)
rel = (out.cpu().float() - yoff_ref.float()).norm().item() / (yoff_ref.float().norm().item() + 1e-12)
print(f"NBLOCK-COMBINE mode={mode} NBLK={NBLK} nb={nb}  y_off rel-L2={rel:.4e}  {'OK' if rel<0.02 else 'WRONG'}")
