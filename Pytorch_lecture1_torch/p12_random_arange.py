import torch

# random [0,1)
x = torch.rand(5,3)
print(x)

# standard normal (mean=0,stdd=1.0)
print(torch.randn(3,5))

# arange
print(torch.arange(1,6))

# normal(means,std,out=None)
print(torch.normal(mean=-1.0,std=torch.arange(1,6).float()))

# linspace(start,end,steps,out=None)
print(torch.linspace(0,10,10))
print(torch.linspace(0,10,11))