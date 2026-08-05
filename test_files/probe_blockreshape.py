# Device probe: does the block RESHAPE the hierarchical scan needs compile?
# The 4D block MATMUL already compiles (probe_blockscan.py, rel 1.7e-3). Remaining
# unknown for end-to-end O(C^1.5): the in-kernel reshape splitting the chunk dim
# (BH,C,NP)->(BH,nb,K,NP), matmul, then merge back ->(BH,C,NP). Two variants:
#   variant A: split IN-KERNEL from (BH,C,NP)  [harder]
#   variant B: input already blocked (BH,nb,K,NP), only merge back  [safer]
# Stale memory: "lower_pad_sequence" blocked it pre-merge. Re-test both.
# Usage: python probe_blockreshape.py [A|B]
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

BH, C, K, NP = 32, 256, 64, 128 * 64
nb = C // K
for nm, s in [("BH", BH), ("C", C), ("nb", nb), ("K", K), ("Ka", K), ("NP", NP)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
st = torch.randn(BH, C, NP, dtype=torch.float16) * 0.05
Lrun = torch.randn(BH, nb, K, K, dtype=torch.float16) * 0.05
ref = torch.matmul(Lrun.float(),
                   st.float().reshape(BH, nb, K, NP)).reshape(BH, C, NP).half()

variant = sys.argv[1] if len(sys.argv) > 1 else "A"


def kA(states, Lrun):
    with spyre_hint(num_tiles_per_dim={"BH": 4}):
        s_c = states * 1.0
        L_c = Lrun * 1.0
        stb = s_c.reshape(BH, nb, K, NP)          # in-kernel C-split
        out = torch.matmul(L_c, stb)
        return out.reshape(BH, C, NP)


def kB(stb, Lrun):
    with spyre_hint(num_tiles_per_dim={"BH": 4}):
        s_c = stb * 1.0
        L_c = Lrun * 1.0
        out = torch.matmul(L_c, s_c)              # (BH,nb,K,NP)
        return out.reshape(BH, C, NP)             # merge back only


ld = Lrun.to("spyre")
name_tensor_dims(ld, ["BH", "nb", "K", "Ka"])
if variant == "A":
    sd = st.to("spyre"); name_tensor_dims(sd, ["BH", "C", "NP"])
    out = torch.compile(kA, dynamic=False)(sd, ld)
else:
    sd = st.reshape(BH, nb, K, NP).to("spyre"); name_tensor_dims(sd, ["BH", "nb", "K", "NP"])
    out = torch.compile(kB, dynamic=False)(sd, ld)
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"BLOCK-RESHAPE variant={variant}  rel-L2={rel:.4e}  {'OK' if rel<0.02 else 'WRONG'}")
