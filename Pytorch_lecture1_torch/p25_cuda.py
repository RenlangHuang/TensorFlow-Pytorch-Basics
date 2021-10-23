import torch

x = torch.tensor([1.,2.])
if torch.cuda.is_available():
    device = torch.device("cuda")          # 定义一个 CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 显示创建在 GPU 上的一个 tensor
    x = x.to(device)                       # 也可以采用 .to("cuda") 
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to() 方法也可以改变数值类型