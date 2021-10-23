import torch
import numpy as np

print('torch and numpy:')
a = np.array([[1,2],[2,3]])
print(a,a.dtype)
b = torch.from_numpy(a) #numpy to tensor
print(b,b.dtype) #dtype as np
print(b.numpy()) #tensor to numpy

print('dtype given:')
a = torch.FloatTensor((3,5))
print(a,a.dtype) #float32
b = torch.IntTensor((1,3))
print(b) #int32
c = torch.DoubleTensor((1.,2.))
print(c) #float64

# dtype setting:
# .IntTensor(), int32
# .ShortTensor(), int16
# .LongTensor(), int64
# .FloatTensor(), float32
# .DoubleTensor(), float64
# .CharTensor(), .ByteTensor()

# for GPU!!! (.cuda.)
print('tensors on GPU:')
x = torch.cuda.FloatTensor(2,2,2)
# different from .FloatTensor((2,2,2))
print(x,x.shape,x.dtype)

print('dtype cast:')
print(c.int()) #cast
print(b.float())
print(a.double())
print(c,c.dtype) #not changed
c = c.half()
print(c,c.dtype) #changed

# dtype cast:
# int16: .short(), 
# int32; .int(), 
# int64: .long(), 
# float16: .half(), 
# float32: .float(),
# float64: .double(),
# .char(), .byte()