# Device probe: flash-style C-blocked scan with a copy_f accumulator.
# Goal: compute scan_out = decay_matrix @ chunk_states  (BH,C+1,C)@(BH,C,NP)
# WITHOUT materializing the (BH,C+1,NP,C) product (the 545MB buf17 in output_4096).
# Pattern copied from decompositions.py flash SDPA: block the REDUCTION dim (C),
# accumulate into a running buffer via spyre.copy_f. No block-RESHAPE (that's the
# blocked backend op) — just slicing C into K-chunks inside nested spyre_hint.
#
# Usage: python probe_flashscan.py [K]   (K = C-block size, default 32)
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

BH, NP = 64, 128 * 64
K = int(sys.argv[1]) if len(sys.argv) > 1 else 32
C = int(sys.argv[3]) if len(sys.argv) > 3 else 64
Cp = C + 1
nblk = C // K
for nm, s in [("BH", BH), ("C", C), ("Cp", Cp), ("NP", NP), ("K", K)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
dm = torch.randn(BH, Cp, C, dtype=torch.float16) * 0.1
st = torch.randn(BH, C, NP, dtype=torch.float16) * 0.1
ref = torch.matmul(dm.float(), st.float()).half()


def dense(dm, st):
    with spyre_hint(num_tiles_per_dim={"BH": 16}):
        return torch.matmul(dm * 1.0, st * 1.0)


def flash(dm, st):
    # block the reduction dim C into nblk slices of K; accumulate with copy_f.
    with spyre_hint(num_tiles_per_dim={"BH": 16}):
        dm_c = dm * 1.0
        st_c = st * 1.0
        acc = torch.matmul(dm_c[:, :, 0:K], st_c[:, 0:K])          # first block
        for b in range(1, nblk):
            c0 = b * K
            part = torch.matmul(dm_c[:, :, c0:c0 + K], st_c[:, c0:c0 + K])
            acc = torch.ops.spyre.copy_f(acc + part, acc)
        return acc


def flash_np(dm, st):
    # block the OUTPUT free dim NP (stick-aligned) instead of the reduction dim.
    # each block is a full matmul dm@st[:,:,np0:np0+NB] -> no reduction-dim slice,
    # no giant (Cp,NP,C) product; concat the column blocks.
    NB = NP // 4                                    # 4 column blocks, each 2048 (stick-aligned)
    with spyre_hint(num_tiles_per_dim={"BH": 16}):
        dm_c = dm * 1.0
        st_c = st * 1.0
        outs = [torch.matmul(dm_c, st_c[:, :, j:j + NB]) for j in range(0, NP, NB)]
        return torch.cat(outs, dim=-1)


mode = sys.argv[2] if len(sys.argv) > 2 else "flash"
dd = dm.to("spyre"); sd = st.to("spyre")
name_tensor_dims(dd, ["BH", "Cp", "C"])
name_tensor_dims(sd, ["BH", "C", "NP"])
fn = {"flash": flash, "dense": dense, "flash_np": flash_np}[mode]
out = torch.compile(fn, dynamic=False)(dd, sd)
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"FLASH-SCAN mode={mode} K={K} nblk={nblk}  rel-L2={rel:.4e}  {'OK' if rel < 0.02 else 'WRONG'}")
