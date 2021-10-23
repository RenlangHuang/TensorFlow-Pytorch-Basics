import torch

a = torch.tensor([1.,-1.])
a = torch.exp(a)
a[1] = -a[0]
print(a)
print(a.floor())
print(a.ceil())
print(a.trunc())
print(a.frac())
print(a.round())