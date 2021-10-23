import torch

# index and segmentation are the same as numpy and tensorflow
a = torch.arange(12).reshape(3,-1)
b = torch.arange(12,0,-1).reshape(3,-1)
print('a =',a)
print('b =',b)

print(torch.stack([a,b]))
print(torch.stack([a,b],dim=0))
print(torch.stack([a,b],dim=1))
print(torch.stack([a,b],dim=2))

# images
a = torch.randn(3, 32, 32)
b = torch.randn(3, 32, 32)
# 堆叠合并为2个图片张量，批量的维度插在最末尾
stack_ab = torch.stack([a, b], dim = -1)
print(stack_ab.size())
# torch.Size([3, 32, 32, 2])
stack_ab = torch.stack([a, b]) # dim = 0
print(stack_ab.size())
# torch.Size([2, 3, 32, 32])

# concatentation
a = torch.arange(12).reshape(3,-1)
b = torch.arange(12,0,-1).reshape(3,-1)
print(torch.cat((a,b))) # default dim = 0
print(torch.cat((a,b),1))

# transpose
print(a.t()) # not changed, a = a.t()
# exchange axises
a = torch.stack([torch.ones((2,2)),
    torch.ones((2,2))*2.0,
    torch.ones((2,2))*3.0],dim=-1)
print(a.size(),a)
b = a.permute([2,0,1])
print(b.size(),b)
b = a.transpose(2,0)
print(b.size(),b)