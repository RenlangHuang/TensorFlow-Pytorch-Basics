import torch
import numpy as np

a = np.arange(10)
np.random.shuffle(a)
a = torch.from_numpy(a)
print(a)

print(a.max(),a.min())
print(a.argmax(),a.argmin())
print(a.median())

# a = a.contiguous()
print('b=',a.reshape(2,-1))
print(a.view(-1,2))
print('a=',a) # not changed
b = a.reshape(2,-1)
print('global:',b.max(),b.argmax())
print('\ncol:',torch.max(b,dim=0)) #col
print('\nrow:',torch.min(b,dim=1)) #row
print('\ncol:',b.max(dim=0)) #col
print('\nargmin of row:',b.argmin(dim=1)) #row
res = b.max(dim=0)
print('decomposite:',res.values,res.indices)

# can dispose of some elements
print(a.resize_(2,2))
print(a) # changed