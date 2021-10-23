import torch
import torchvision

print('torch:',torch.__version__)
print('torchvision:',torchvision.__version__)
print('CUDA version:',torch.version.cuda)
print('CUDA available?',torch.cuda.is_available())
print('devices:',torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i,',',torch.cuda.get_device_name(i))

zeros_x = torch.zeros(3,4,dtype=torch.long)
print(zeros_x)

tensorx = torch.tensor([-1,1])
print(tensorx,tensorx.shape,tensorx.dtype)

tensorx = torch.tensor([3.0,1.0])
print(tensorx,tensorx.shape,tensorx.dtype)

tensorx = torch.randn_like(tensorx,dtype=torch.float)
print(tensorx,tensorx.size()) # tuple

tensorx = zeros_x.new_ones(2,3,dtype=torch.double)
print(tensorx,tensorx.shape,tensorx.dtype)

ones_x = torch.ones(2,2,2,dtype=torch.int32)
print(ones_x,ones_x.shape,ones_x.dtype)

x = torch.full((2,2,2),3,dtype=torch.int16)
print(x,x.shape,x.dtype)