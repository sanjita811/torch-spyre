# Device probe: produce chunk_states ALREADY N-blocked, so it's never re-sliced.
# The in-kernel wall was: chunk_states = b_scaled_t @ x_c (matmul output) then
# sliced on N -> "Reduction: no mechanism to resolve stick incompatibility".
# Fix attempt: slice b_scaled_t on its N OUTPUT-ROW *before* the matmul, so each
# block's chunk_states is its own fresh matmul output (clean layout, no result-slice).
# Then scan + y_off per block, accumulate y_off via copy_f. Mirrors full-kernel
# dependency chain (b_scaled_t and x_c as matmul-produced, like fused_kernel).
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
NBLK = int(sys.argv[1]) if len(sys.argv) > 1 else 64
nb = N // NBLK
for nm, s in [("BH", BH), ("C", C), ("Cp", Cp), ("L", L), ("N", N), ("La", L), ("P", P)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
# inputs that the intra stage would have produced:
bst = torch.randn(BH, C, N, L, dtype=torch.float16) * 0.1     # b_scaled_t (BH,C,N,La)
x = torch.randn(BH, C, L, P, dtype=torch.float16) * 0.1       # x_c (BH,C,La,P)
dm = torch.randn(BH, Cp, C, dtype=torch.float16) * 0.1        # decay_matrix
csc = torch.randn(BH, C, L, N, dtype=torch.float16) * 0.1     # c_scaled (BH,C,L,N)

# reference
cs_ref = torch.matmul(bst.float(), x.float())                          # (BH,C,N,P)
scan_ref = torch.matmul(dm.float(), cs_ref.reshape(BH, C, N * P))
rolled_ref = scan_ref[:, :C].reshape(BH, C, N, P)
yoff_ref = torch.matmul(csc.float(), rolled_ref).half()


def produce_blocked(bst, x, dm, csc):
    with spyre_hint(num_tiles_per_dim={"BH": 8}):
        bst_c = bst * 1.0; x_c = x * 1.0; dm_c = dm * 1.0; csc_c = csc * 1.0
        y_off = None
        for b in range(nb):
            n0 = b * NBLK
            bst_blk = bst_c[:, :, n0:n0 + NBLK, :]                     # slice N BEFORE matmul
            cs_blk = torch.matmul(bst_blk, x_c)                        # (BH,C,NBLK,P) fresh output
            scan_blk = torch.matmul(dm_c, cs_blk.reshape(BH, C, NBLK * P))
            rolled_blk = scan_blk[:, :C].reshape(BH, C, NBLK, P)
            csc_blk = csc_c[:, :, :, n0:n0 + NBLK]
            part = torch.matmul(csc_blk, rolled_blk)
            y_off = part if y_off is None else torch.ops.spyre.copy_f(y_off + part, y_off)
        return y_off


bd = bst.to("spyre"); xd = x.to("spyre"); dd = dm.to("spyre"); cscd = csc.to("spyre")
name_tensor_dims(bd, ["BH", "C", "N", "La"])
name_tensor_dims(xd, ["BH", "C", "La", "P"])
name_tensor_dims(dd, ["BH", "Cp", "C"])
name_tensor_dims(cscd, ["BH", "C", "L", "N"])
out = torch.compile(produce_blocked, dynamic=False)(bd, xd, dd, cscd)
rel = (out.cpu().float() - yoff_ref.float()).norm().item() / (yoff_ref.float().norm().item() + 1e-12)
print(f"PRODUCE-BLOCKED NBLK={NBLK} nb={nb}  y_off rel-L2={rel:.4e}  {'OK' if rel<0.02 else 'WRONG'}")
