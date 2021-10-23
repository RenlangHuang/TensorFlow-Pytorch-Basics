import os
import torch
import numpy as np
from sklearn import datasets
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
x_data = torch.FloatTensor(x_data)
y_data = torch.FloatTensor(y_data)

model = torch.nn.Sequential(
    torch.nn.Linear(4,4),
    torch.nn.ReLU(),
    torch.nn.Linear(4,3)
)

'''if os.path.exists('./checkpoint/mlp_model.pth'): # the whole model
    print('------------load the model----------------')
    model = torch.load('./checkpoint/mlp_model.pth')'''
if os.path.exists('./checkpoint/mlp.pth'): # model parameters
    print('------------load the model parameters----------------')
    model.load_state_dict(torch.load('./checkpoint/mlp.pth'))

def evaluate(model,inputs,labels):
    accuracy = 0
    with torch.no_grad():
        outputs = model(inputs)
    outputs = torch.argmax(outputs,dim=1)
    for i in range(outputs.shape[0]):
        if outputs[i]==labels[i]:
            accuracy += 1
    return float(accuracy)/float(outputs.shape[0])*100.0

print('accuracy on the train dataset: ',evaluate(model,x_train,y_train))
print('accuracy on the test dataset: ',evaluate(model,x_test,y_test))
print('accuracy on the whole dataset: ',evaluate(model,x_data,y_data))
