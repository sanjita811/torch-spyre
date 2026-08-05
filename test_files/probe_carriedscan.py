# Device probe: carried O(C) scan via flash-style copy_f accumulator, tiling BH
# (the batch dim, like flash tiles batch), carrying the running state across a
# PYTHON loop over C chunks. No block reshape. Recurrence:
#   h = 0; for i in C: rolled[i]=h; h = exp(chunk_decay[i])*h + states[i]
# States (BH,C,NP). We slice per-chunk states[:, i] (BH,NP) inside the loop.
# Question: does per-chunk slice + copy_f accumulate compile at C=128 tiling BH?
import torch
import torch_spyre  # noqa: F401
from torch_spyre._inductor import spyre_hint

try:
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim, name_tensor_dims)
except (ImportError, ModuleNotFoundError):
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim, name_tensor_dims)

BH, C, NP = 32, 128, 128 * 64
for nm, s in [("BH", BH), ("C", C), ("NP", NP), ("One", 1)]:
    declare_tensor_dim(nm, s)

torch.manual_seed(0)
states = torch.randn(BH, C, NP, dtype=torch.float16) * 0.02
dec = torch.exp(torch.randn(BH, C, dtype=torch.float16).clamp(max=0) * 0.1)  # (BH,C) in (0,1]

# reference: sequential recurrence (exclusive prefix)
h = torch.zeros(BH, NP, dtype=torch.float32)
ref = torch.empty(BH, C, NP, dtype=torch.float32)
for i in range(C):
    ref[:, i] = h
    h = dec[:, i:i+1].float() * h + states[:, i].float()
ref = ref.half()


def carried(states, dec):
    with spyre_hint(num_tiles_per_dim={"BH": 4}):
        s_c = states * 1.0
        d_c = dec * 1.0
        rows = []
        h = torch.zeros(BH, NP, device=states.device, dtype=states.dtype)
        for i in range(C):
            rows.append(h.unsqueeze(1))
            h = d_c[:, i:i+1] * h + s_c[:, i]
        return torch.cat(rows, dim=1)          # (BH,C,NP)


sd = states.to("spyre"); dd = dec.to("spyre")
name_tensor_dims(sd, ["BH", "C", "NP"])
name_tensor_dims(dd, ["BH", "C"])
out = torch.compile(carried, dynamic=False)(sd, dd)
rel = (out.cpu().float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-12)
print(f"CARRIED-SCAN C={C} loop tiling BH  rel-L2={rel:.4e}  {'OK' if rel<0.02 else 'WRONG'}")
