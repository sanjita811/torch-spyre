# Device probe: does the block-structured reshape + batched matmul that a
# hierarchical O(C^1.5) scan needs compile on merged main? Stale memory says
# 'lower_pad_sequence' / '4D block matmul cannot restickify' — predates
# #3344/#3350/#3403/#3429. Re-test the CORE op only (not the full scan):
#   states (BH, C, NP) -> blocks (BH, nb, K, NP) -> per-block dense K-scan
#   Lrun (BH, nb, K, K) @ stb (BH, nb, K, NP) -> (BH, nb, K, NP)
# This is a 4D-batched matmul (two batch dims BH,nb). If it compiles+correct,
# hierarchical is reachable.
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

BH, C, K, NP = 32, 256, 64, 128 * 64     # C=256 (T=16K/L=64), nb=4 blocks of K=64
nb = C // K
for nm, s in [("BH", BH), ("nb", nb), ("K", K), ("Ka", K), ("NP", NP)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
Lrun = torch.randn(BH, nb, K, K, dtype=torch.float16) * 0.05
stb = torch.randn(BH, nb, K, NP, dtype=torch.float16) * 0.05

ref = torch.matmul(Lrun.float(), stb.float()).half()


def block_matmul(Lrun, stb):
    with spyre_hint(num_tiles_per_dim={"BH": 4}):
        Lc = Lrun * 1.0          # #3381 pre-copy
        sc = stb * 1.0
        return torch.matmul(Lc, sc)


ld = Lrun.to("spyre"); sd = stb.to("spyre")
name_tensor_dims(ld, ["BH", "nb", "K", "Ka"])
name_tensor_dims(sd, ["BH", "nb", "Ka", "NP"])
out = torch.compile(block_matmul, dynamic=False)(ld, sd)
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"BLOCK-MATMUL 4D (BH,nb,K,K)@(BH,nb,K,NP)  rel-L2={rel:.4e}  {'OK' if rel<0.02 else 'WRONG'}")
