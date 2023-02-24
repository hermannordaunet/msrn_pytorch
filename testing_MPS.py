import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.rand(size=(4, 5), device=mps_device, dtype=torch.float32)
    print(x)
else:
    print("MPS device not found.")
