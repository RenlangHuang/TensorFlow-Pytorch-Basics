import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.squeeze(), y_test.squeeze()

categories = ('airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck')
np.set_printoptions(precision=2)

x,y = [],[] #application
sample = np.random.randint(0,x_train.shape[0],size=10).tolist()
for i in sample:
    x.append(x_train[i])
    y.append(categories[y_train[i]])
images = x.copy()
x = torch.FloatTensor(np.array(x)).permute([0,3,1,2])


class AlexNet8(torch.nn.Module):
    def __init__(self,n_output):
        super(AlexNet8,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,96,kernel_size=(3,3)),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3,3),stride=2),

            torch.nn.Conv2d(96,256,kernel_size=(3,3)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3,3),stride=2),

            torch.nn.Conv2d(256,384,kernel_size=(3,3),padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,384,kernel_size=(3,3),padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(384,256,kernel_size=(3,3),padding=(1,1)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(3,3),stride=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256*2*2, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, n_output)
        )
    
    def forward(self,inputs):
        outputs = self.conv(inputs)
        outputs = outputs.reshape(outputs.shape[0],-1)
        return self.dense(outputs)

model = AlexNet8(10)

if os.path.exists('./checkpoint/AlexNet8_cifar10.pth'):
    print('------------load the model----------------')
    model.load_state_dict(torch.load('./checkpoint/AlexNet8_cifar10.pth'))

print(torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU detected! Loading the model to CUDA...')
print('number of GPU: ',torch.cuda.device_count())
#if torch.cuda.device_count()>1: # the first device id = cuda:id
#    model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]) # default:using all
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

y_train = torch.LongTensor(y_train).squeeze()
y_test = torch.LongTensor(y_test).squeeze()
x_train = torch.FloatTensor(x_train).permute([0,3,1,2])
x_test = torch.FloatTensor(x_test).permute([0,3,1,2])
print("x_train.shape", x_train.shape)
print("y_train.shape", y_train.shape)
print("x_test.shape", x_test.shape)
print("y_test.shape", y_test.shape)

#accuracy_train = evaluate(model,x_train.cuda(),y_train.cuda())
#accuracy_test = evaluate(model,x_test.cuda(),y_test.cuda())
#print('Training_accuracy = %.4f, Validation_accuracy = %.4f'%\
#    (accuracy_train,accuracy_test))


# application
print(x.shape)
with torch.no_grad():
    pred = model(x.cuda())
pred = torch.argmax(pred,dim=1)
for k in range(10):
    print(y[k],categories[pred[k]])
    plt.subplot(2,5,k+1)
    plt.title(categories[pred[k]])
    plt.imshow(images[k], cmap='gray')
plt.show()