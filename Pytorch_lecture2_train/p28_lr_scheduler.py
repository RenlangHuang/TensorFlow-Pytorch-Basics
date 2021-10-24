import torch
import torchvision
import matplotlib.pyplot as plt

model = torchvision.models.resnet101()
optimizer = torch.optim.Adam(params=model.parameters(),lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)

lr = list()
for epoch in range(20):
    lr.append(lr_scheduler.get_last_lr()[0])
    optimizer.step()
    lr_scheduler.step()
plt.plot(lr,label='StepLR')


optimizer = torch.optim.Adam(params=model.parameters(),lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,15],gamma=0.5)

lr = list()
for epoch in range(20):
    lr.append(lr_scheduler.get_last_lr()[0])
    optimizer.step()
    lr_scheduler.step()
plt.plot(lr,label='MultiStepLR')


optimizer = torch.optim.Adam(params=model.parameters(),lr=0.1)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)

lr = list()
for epoch in range(20):
    lr.append(lr_scheduler.get_last_lr()[0])
    optimizer.step()
    lr_scheduler.step()
plt.plot(lr,label='ExponentialLR')

ax = plt.gca()
x_major_locator = plt.MultipleLocator(1)
y_major_locator = plt.MultipleLocator(0.01)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.title('learning rate scheduler')
plt.xlim(0,19)
plt.legend()
plt.grid()
plt.show()