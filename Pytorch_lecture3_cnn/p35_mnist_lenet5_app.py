import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from torch.autograd.grad_mode import no_grad

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)
print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)
print("categories:",set(y_train))
np.set_printoptions(precision=2)

x_train = x_train.reshape(x_train.shape[0], 1, 28, 28) # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
print("x_train.shape", x_train.shape)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)


class LeNet5(torch.nn.Module):
    def __init__(self,n_output):
        super(LeNet5,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1,6,kernel_size=(5,5)),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2),

            torch.nn.Conv2d(6,16,kernel_size=(5,5)),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16*4*4, 120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(120, 84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(84, n_output)
        )
    
    def forward(self,inputs):
        outputs = self.conv(inputs)
        outputs = outputs.view(outputs.shape[0],-1)
        return self.dense(outputs)

model = LeNet5(10)

if os.path.exists('./checkpoint/LeNet5_mnist.pth'):
    print('------------load the model----------------')
    model.load_state_dict(torch.load('./checkpoint/LeNet5_mnist.pth'))

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU detected! Loading the model to CUDA...')
gpu_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(gpu_device)


def evaluate(model,inputs,labels,batch_size=32):
    accuracy = 0
    gpu_available = torch.cuda.is_available()
    for i in range(0,inputs.shape[0],batch_size):
        if i+batch_size>=inputs.shape[0]:
            batch = inputs[i:]
        else: batch = inputs[i:i+batch_size]
        if gpu_available:
            batch = batch.cuda()
        with torch.no_grad():
            outputs = model(batch)
        outputs = torch.argmax(outputs,dim=1)
        for k in range(outputs.shape[0]):
            if outputs[k]==labels[k+i]:
                accuracy += 1
    return float(accuracy)/float(inputs.shape[0])*100.0


if torch.cuda.is_available():
    accuracy_train = evaluate(model,x_train.cuda(),y_train.cuda())
    accuracy_test = evaluate(model,x_test.cuda(),y_test.cuda())
else:
    accuracy_train = evaluate(model,x_train,y_train)
    accuracy_test = evaluate(model,x_test,y_test)
accuracy = (accuracy_train*x_train.shape[0] + accuracy_test*x_test.shape[0])/(x_train.shape[0]+x_test.shape[0])
print('Training_accuracy = %.4f, Validation_accuracy = %.4f, Total_accuracy = %.4f'%\
    (accuracy_train,accuracy_test,accuracy))


for i in range(5):
    sample = np.random.randint(0,x_train.shape[0])
    plt.imshow(x_train[sample].squeeze(), cmap='gray') # 绘制灰度图
    print('the label of the sample:',y_train[sample])
    sample = torch.FloatTensor(x_train[sample].reshape(1, 1, 28, 28))
    with torch.no_grad():
        if torch.cuda.is_available():
            predict = model(sample.cuda())
        else: predict = model(sample.cuda())
    print(predict); print('predicted by LeNet5:',torch.argmax(predict))
    plt.show()