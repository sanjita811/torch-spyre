import torch
import torch.spyre  # noqa: F401 -- triggers the spyre monkey-patch (to(device_layout=))
from torch.spyre import SpyreTensorLayout
from torch_spyre._inductor.propagate_named_dims import declare_tensor_dim

BH, C = 64, 64
STICK = 64
for nm, v in [("BH", BH), ("C", C)]:
    declare_tensor_dim(nm, v)

# dim_order=[1, 0]: stick goes on dim 0 (BH), so both reads of acs in the
# outer-product kernel share stick=Mod(i0,64) regardless of which C index they use.
acs_layout = SpyreTensorLayout([BH, C], [C, 1], torch.float16, [1, 0])

def build_decay(acs):
    a = acs.unsqueeze(-1)
    b = acs.unsqueeze(-2)
    return torch.exp(a - b)

acs = torch.randn(BH, C, dtype=torch.float16)

cpu_out = build_decay(acs)
spyre_out = torch.compile(build_decay, dynamic=False)(acs.to(device_layout=acs_layout)).cpu()
torch.testing.assert_close(cpu_out, spyre_out, rtol=1e-2, atol=1e-2)
print(f"build_decay: {list(cpu_out.shape)} OK")