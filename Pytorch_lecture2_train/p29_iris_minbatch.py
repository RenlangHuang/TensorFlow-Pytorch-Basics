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

y_train = torch.LongTensor(y_data[:-30])
y_test = torch.LongTensor(y_data[-30:])
x_train = torch.FloatTensor(x_data[:-30])
x_test = torch.FloatTensor(x_data[-30:])

ds = TensorDataset(x_train,y_train)
loader = DataLoader(dataset=ds,batch_size=32,shuffle=True)#,num_workers=2)

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
best_acc = -100.0
for epoch in range(500):
    batch_loss = []
    for step,data in enumerate(loader):
        inputs, labels = data
        inputs, labels = Variable(inputs),Variable(labels)
        out = model(inputs)
        loss = Loss(out,labels)
        batch_loss.append(loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(np.mean(np.array(batch_loss)))
    accuracy_train.append(evaluate(model,x_train,y_train))
    accuracy_test.append(evaluate(model,x_test,y_test))
    current_acc = accuracy_train[-1]+1.5*accuracy_test[-1]-loss*80
    if current_acc>best_acc: # only save the best model
        best_epoch = epoch
        best_acc = current_acc
        best_model = model.state_dict().copy()

model.load_state_dict(best_model)
print('best epoch: %d: Training_accuracy = %.4f, Validation_accuracy = %.4f'%\
    (best_epoch,accuracy_train[best_epoch],accuracy_test[best_epoch]))

# save the whole model
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

