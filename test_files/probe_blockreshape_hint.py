# Device probe: can a named_dims HINT re-annotate the split output so the block
# reshape the hierarchical scan needs compiles? The named-dim pass
# (propagate_named_dims.py:426-466) uses a per-op `named_dims` hint DIRECTLY,
# bypassing input-propagation (which raises "reshape split a named dim,
# re-annotate after the reshape"). spyre_hint(**kwargs) forwards arbitrary kwargs
# into that hint dict. So wrap the post-split ops in spyre_hint(named_dims=[...]).
#
# Usage: python probe_blockreshape_hint.py [split|both]
#   split : add named_dims hint only where needed
#   both  : also hint the merge-back
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


def k(states, Lrun):
    with spyre_hint(num_tiles_per_dim={"BH": 4}):
        s_c = states * 1.0
        L_c = Lrun * 1.0
        # in-kernel C-split; annotate the split output via a named_dims hint scope
        with spyre_hint(named_dims=["BH", "nb", "K", "NP"]):
            stb = s_c.reshape(BH, nb, K, NP)
            out = torch.matmul(L_c, stb)          # (BH,nb,K,NP)
        # merge back
        with spyre_hint(named_dims=["BH", "C", "NP"]):
            merged = out.reshape(BH, C, NP)
        return merged


sd = st.to("spyre"); ld = Lrun.to("spyre")
name_tensor_dims(sd, ["BH", "C", "NP"])
name_tensor_dims(ld, ["BH", "nb", "K", "Ka"])
out = torch.compile(k, dynamic=False)(sd, ld)
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"BLOCK-RESHAPE-HINT  rel-L2={rel:.4e}  {'OK' if rel<0.02 else 'WRONG'}")
