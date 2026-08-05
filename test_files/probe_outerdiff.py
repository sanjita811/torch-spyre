# Device probe: can we build the (BH,C,L,L) intra-decay mask ON-DEVICE from a
# single g=(BH,C,L) via an in-kernel outer difference at L>64, on merged main?
# Stale memory says "no mechanism to resolve stick incompatibility" / silent wrong
# mask — but that predates #3344/#3350/#3403/#3429 tiling commits. Re-test.
#
# Usage: python probe_outerdiff.py <strategy>
#   strategy in {default, bhstick, expand_row, expand_both}
# One compile per process (backend one-Spyre-compile-per-process limit).
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
try:
    from torch.spyre import SpyreTensorLayout
except (ImportError, ModuleNotFoundError):
    from torch_spyre._C import SpyreTensorLayout

BH, C, L = 32, 16, 256          # T=4096, L=256 -> C=16 (real Mamba-2 chunk)
for nm, s in [("BH", BH), ("C", C), ("L", L), ("La", L)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
a = (torch.randn(BH, C, L, dtype=torch.float16) * 0.02).clamp(max=0)
g = a.float().cumsum(-1).half()                     # (BH,C,L)
causal = torch.tril(torch.ones(L, L, dtype=torch.float16))


def cpu_mask(g):
    outer = g.unsqueeze(-1).float() - g.unsqueeze(-2).float()
    return (torch.exp(torch.clamp(outer, max=0.0)).half()
            * torch.tril(torch.ones(L, L, dtype=torch.float16)))


ref = cpu_mask(g)
strat = sys.argv[1] if len(sys.argv) > 1 else "default"


def build_from_single(g_one, causal):
    outer = g_one.unsqueeze(-1) - g_one.unsqueeze(-2)   # (BH,C,L,La)
    return torch.exp(torch.clamp(outer, max=0.0)) * causal


def build_from_both(g_row, g_col, causal):
    outer = g_row - g_col
    return torch.exp(torch.clamp(outer, max=0.0)) * causal


cd = causal.to("spyre")
name_tensor_dims(cd, ["L", "La"])

if strat == "default":
    gd = g.to("spyre")
    name_tensor_dims(gd, ["BH", "C", "L"])
    out = torch.compile(build_from_single, dynamic=False)(gd, cd)
elif strat == "bhstick":
    # custom layout: stick on BH (dim_order puts BH innermost-in-memory)
    stl = SpyreTensorLayout([BH, C, L], [C * L, L, 1], torch.float16, [1, 2, 0])
    gd = g.to(device_layout=stl)
    name_tensor_dims(gd, ["BH", "C", "L"])
    out = torch.compile(build_from_single, dynamic=False)(gd, cd)
elif strat == "expand_row":
    # only row pre-expanded on host; col stays (BH,C,L) broadcast in-kernel
    g_row = g.unsqueeze(-1).expand(BH, C, L, L).contiguous().to("spyre")
    name_tensor_dims(g_row, ["BH", "C", "L", "La"])
    def bp(g_row, g_col, causal):
        return torch.exp(torch.clamp(g_row - g_col.unsqueeze(-2), max=0.0)) * causal
    gc = g.to("spyre"); name_tensor_dims(gc, ["BH", "C", "La"])
    out = torch.compile(bp, dynamic=False)(g_row, gc, cd)
else:  # expand_both = current shipping approach (baseline sanity)
    g_row = g.unsqueeze(-1).expand(BH, C, L, L).contiguous().to("spyre")
    g_col = g.unsqueeze(-2).expand(BH, C, L, L).contiguous().to("spyre")
    name_tensor_dims(g_row, ["BH", "C", "L", "La"])
    name_tensor_dims(g_col, ["BH", "C", "L", "La"])
    out = torch.compile(build_from_both, dynamic=False)(g_row, g_col, cd)

rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"STRATEGY={strat}  rel-L2={rel:.4e}  {'OK' if rel < 0.02 else 'WRONG'}")
