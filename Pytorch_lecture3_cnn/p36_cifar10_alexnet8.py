import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
import tensorflow as tf
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset,DataLoader

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
np.set_printoptions(precision=2)

# 可视化训练集输入
sample = np.random.randint(0,x_train.shape[0])
plt.imshow(x_train[sample], cmap='gray') # 绘制灰度图
print('the label of the sample:',y_train[sample])
plt.show()

y_train = torch.LongTensor(y_train).squeeze()
y_test = torch.LongTensor(y_test).squeeze()
x_train = torch.FloatTensor(x_train).permute([0,3,1,2])
x_test = torch.FloatTensor(x_test).permute([0,3,1,2])
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)

train = TensorDataset(x_train,y_train)
loader = DataLoader(dataset=train,batch_size=32,shuffle=True)#,num_workers=2)


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
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

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

losses = []
accuracy_train = []
accuracy_test = []
best_acc = -100.0
for epoch in range(5): #15
    batch_loss = []
    for step,data in enumerate(loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()),Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs),Variable(labels)
        out = model(inputs)
        loss = Loss(out,labels)
        batch_loss.append(loss.data.item())
        if step % 5 == 4:
            print('epoch %d | batch %d: loss = %.4f'%(epoch+1,step+1,batch_loss[-1]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses.append(np.mean(np.array(batch_loss)))
    if torch.cuda.is_available():
        accuracy_train.append(evaluate(model,x_train.cuda(),y_train.cuda()))
        accuracy_test.append(evaluate(model,x_test.cuda(),y_test.cuda()))
    else:
        accuracy_train.append(evaluate(model,x_train,y_train))
        accuracy_test.append(evaluate(model,x_test,y_test))
    print('epoch %d: loss = %.4f, mean_loss = %.4f, Training_accuracy = %.4f, Validation_accuracy = %.4f'%\
        (epoch+1,batch_loss[-1],losses[-1],accuracy_train[-1],accuracy_test[-1]))
    print('----------------------------------------------------------------------------')
    current_acc = accuracy_train[-1]+1.5*accuracy_test[-1]-losses[-1]*80
    if current_acc>best_acc: # only save the best model
        best_epoch = epoch
        best_acc = current_acc
        best_model = model.state_dict().copy()

model.load_state_dict(best_model)
print('best epoch: %d: Training_accuracy = %.4f, Validation_accuracy = %.4f'%\
    (best_epoch+1,accuracy_train[best_epoch],accuracy_test[best_epoch]))

# save the model parameters
torch.save(model.state_dict(),'./checkpoint/AlexNet8_cifar10.pth')

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
