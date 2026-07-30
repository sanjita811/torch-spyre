import torch
import torch_spyre  
from torch_spyre._inductor import spyre_hint

try:
    from torch_spyre._inductor.wsr.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )
except (ImportError, ModuleNotFoundError):
    from torch_spyre._inductor.propagate_named_dims import (
        declare_tensor_dim,
        name_tensor_dims,
    )

BH, L, N = 64, 64, 128
for nm, s in [("BH", BH), ("L", L), ("La", L), ("N", N)]:
    declare_tensor_dim(nm, s)

c = torch.randn(BH, L, N, dtype=torch.float16) * 0.01
b = torch.randn(BH, L, N, dtype=torch.float16) * 0.01


def k(c, b):
    # 3D batched matmul; tile the BATCH dim (BH). c and b are graph inputs.
    with spyre_hint(num_tiles_per_dim={"BH": 4}):
        return torch.matmul(c, b.transpose(-1, -2))


cd, bd = c.to("spyre"), b.to("spyre")
name_tensor_dims(cd, ["BH", "L", "N"])
name_tensor_dims(bd, ["BH", "La", "N"])

out = torch.compile(k, dynamic=False)(cd, bd)   # <-- raises StopIteration
