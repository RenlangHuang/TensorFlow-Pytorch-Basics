import torch
import numpy as np

a = torch.ones(3)
print(a,a.dtype)

b = torch.from_numpy(np.arange(3))
print(b,b.dtype)

print(torch.add(a,b), a+b)
print(torch.sub(a,b), a-b)
print(torch.mul(a,b), a*b)
print(torch.div(a,b+1),a/(b+1))
# a.add_(b) equals to a = a+b

# broadcast
print('broadcast mechanism:')
c = torch.from_numpy(np.arange(12).reshape((-1,3)))
print(c,c.dtype)
print(c-b)
print(c+a)

# matrix multiplication
print('matrix mutiplication:')
a = torch.from_numpy(np.arange(4).reshape((2,2))).float()
b = torch.from_numpy(np.arange(4).reshape((2,2))+1).float()
print('a=',a,'\nb=',b)
print(a @ b)
print(torch.matmul(a,b))
print(torch.mm(a,b)) # only for 2D
print(a @ torch.FloatTensor((1.,2.)))

# complex math operation
print('complex math operation:')
print(a.pow(2)); print(b**2)
print(a.sqrt()); print(a.pow(0.5)); print(b**0.5)
print(torch.exp(torch.tensor([1.])),torch.exp(a))
print(torch.log(torch.tensor([2.])),torch.log(b))
