import os
import torch
import numpy as np
from sklearn import datasets
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset,DataLoader

# load iris dataset
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target
print('iris dataset:')
print('features:',type(x_data),x_data.shape)
print('labels:',type(y_data),y_data.shape)

# shuffle the dataset randomly
np.random.seed(681)
np.random.shuffle(x_data)
np.random.seed(681)
np.random.shuffle(y_data)

x_data = torch.FloatTensor(x_data)
y_data = torch.LongTensor(y_data)
ds = TensorDataset(x_data,y_data)
loader = DataLoader(dataset=ds,batch_size=30,shuffle=True)#,num_workers=2)

# class style
'''class MLP(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(MLP,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.output = torch.nn.Linear(n_hidden,n_output)
    def forward(self,inputs):
        x = F.relu(self.hidden(inputs))
        return self.output(x)

model = MLP(4,4,3)'''

# Sequential style
model = torch.nn.Sequential(
    torch.nn.Linear(4,4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,3)
)
print(model)

### parameter 'weight_decay' is the weight of the L2 regularization item!
#optimizer = torch.optim.SGD(model.parameters(),lr=0.02)
#optimizer = torch.optim.SGD(model.parameters(),lr=0.02,momentum=0.2,nesterov=True)
#optimizer = torch.optim.Adagrad(model.parameters(),lr=0.05)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.005)
optimizer = torch.optim.Adam(model.parameters(),lr=0.005)#,betas=(0.8,0.9))

Loss = torch.nn.CrossEntropyLoss()
'''if os.path.exists('./checkpoint/mlp.pth'):
    print('------------load the model----------------')
    model.load_state_dict(torch.load('./checkpoint/mlp.pth'))'''

def evaluate(model,inputs,labels):
    accuracy = 0
    outputs = model(inputs)
    outputs = torch.argmax(outputs,dim=1)
    for i in range(outputs.shape[0]):
        if outputs[i]==labels[i]:
            accuracy += 1
    return float(accuracy)/float(outputs.shape[0])*100.0

losses = []
accuracy_train = []
accuracy_test = []
for epoch in range(200):
    batch_loss = []
    for step,data in enumerate(loader):
        inputs, labels = data
        if step == 4: # K-fold cross validation
            accuracy_test.append(evaluate(model,inputs,labels));break
        inputs, labels = Variable(inputs),Variable(labels)
        out = model(inputs)
        loss = Loss(out,labels)
        batch_loss.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(np.mean(np.array(batch_loss)))
    accuracy_train.append(evaluate(model,x_data,y_data))

# save the hole model
# torch.save(model,'./checkpoint/mlp_model.pth')
# save the model parameters
# torch.save(model.state_dict(),'./checkpoint/mlp.pth')

# visualization
plt.subplot(1,2,1)
plt.plot(losses)
plt.grid()
plt.subplot(1,2,2)
plt.plot(accuracy_train,label='Training Accuracy')
plt.plot(accuracy_test,label='Testing Accuracy')
plt.legend()
plt.grid()
plt.show()