import os
import torch
import numpy as np
from sklearn import datasets
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt

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

y_train = torch.LongTensor(y_data[:-30])
y_test = torch.LongTensor(y_data[-30:])
x_train = torch.FloatTensor(x_data[:-30])
x_test = torch.FloatTensor(x_data[-30:])

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

#optimizer = torch.optim.SGD(model.parameters(),lr=0.05)
#optimizer = torch.optim.SGD(model.parameters(),lr=0.05,momentum=0.8,nesterov=True)
#optimizer = torch.optim.Adagrad(model.parameters(),lr=0.08)
#optimizer = torch.optim.RMSprop(model.parameters(),lr=0.02,alpha=0.9)
optimizer = torch.optim.Adam(model.parameters(),lr=0.05,betas=(0.9,0.99))

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
for i in range(200):
    out = model(x_train)
    loss = Loss(out,y_train)
    losses.append(loss)
    accuracy_train.append(evaluate(model,x_train,y_train))
    accuracy_test.append(evaluate(model,x_test,y_test))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# save the hole model
'''torch.save(model,'./checkpoint/mlp_model.pth')
model = torch.load('./checkpoint/mlp_model.pth')'''
# save the model parameters
#torch.save(model.state_dict(),'./checkpoint/mlp.pth')
#model.load_state_dict(torch.load('./checkpoint/mlp.pth'))

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