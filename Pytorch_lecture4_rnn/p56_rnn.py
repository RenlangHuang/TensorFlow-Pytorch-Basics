import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader


# generate a dataset
t_ = np.linspace(0, 199, 200)
t = np.linspace(-np.pi*2, np.pi*2, dtype=np.float32)
x, y = list(), list()
for i in range(200):
    t = t + t_[i]
    x.append(np.sin(t))
    y.append(np.cos(t))
x = torch.from_numpy(np.array(x)[...,np.newaxis])
y = torch.from_numpy(np.array(y)[...,np.newaxis])
ds = TensorDataset(x, y)
loader = DataLoader(dataset=ds,batch_size=32,shuffle=True)


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.dense = torch.nn.Linear(32, 1)
        self.dense.weight.data.normal_(0, 0.1)
    
    def forward(self, x, prev_hidden):
        # x (batch, time_step, input_size)
        # prev_hidden (num_layers, batch, hidden_size)
        # y_out (batch, time_step, output_size)
        result = list()
        y_out, h_state = self.rnn(x,prev_hidden)
        for t in range(y_out.shape[1]):
            result.append(torch.tanh(self.dense(y_out[:,t,:])))
        y_out = torch.stack(result,dim=1)
        return y_out, h_state

model = RNN()
optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
Loss = torch.nn.MSELoss()

h_state = None
losses = []
for epoch in range(50):
    batch_loss = []
    for step,data in enumerate(loader):
        inputs, labels = data
        inputs, labels = Variable(inputs),Variable(labels)
        out,_ = model(inputs, h_state)
        loss = Loss(out, labels)
        batch_loss.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(np.mean(np.array(batch_loss)))
    print('epoch: %d: loss = %.4f'%(epoch+1, losses[-1]))

# save the model parameters
torch.save(model.state_dict(),'./checkpoint/rnn.pth')

# visualization
plt.plot(losses)
plt.grid()
plt.show()

# application
t_ = np.random.random()*10.0
t = np.linspace(-np.pi*2, np.pi*2, dtype=np.float32)+t_
x = torch.from_numpy(np.array(np.sin(t))[np.newaxis,:,np.newaxis])
y = torch.from_numpy(np.array(np.cos(t))[np.newaxis,:,np.newaxis])
model.load_state_dict(torch.load('./checkpoint/rnn.pth'))
ybar,_ = model(x, None)
ybar = ybar.detach().numpy()
plt.plot(t,x.squeeze(),'r')
plt.plot(t,y.squeeze(),'b')
plt.scatter(t,ybar.squeeze(),marker='*')
plt.grid()
plt.show()