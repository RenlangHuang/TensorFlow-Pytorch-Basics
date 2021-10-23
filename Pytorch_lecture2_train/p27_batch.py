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

for epoch in range(3):
    for i,data in enumerate(loader):
        inputs,labels = data
        inputs, labels = Variable(inputs),Variable(labels)
        print('Epoch:',epoch,'| Step:',i,'| batch_x:',inputs.data.size(),'batch_y:',labels.data.size())

'''for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        print('Epoch: %d | Step: %d | batch x: '%(epoch,step),batch_x,'| batch y: ',batch_y)'''
