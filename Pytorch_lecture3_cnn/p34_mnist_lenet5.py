import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
from p33_training_classification import training


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


def main():
    # dataset preparation
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)
    print("categories:",set(y_train))
    np.set_printoptions(precision=2)

    # visualize a sample in the dataset
    sample = np.random.randint(0,x_train.shape[0])
    plt.imshow(x_train[sample], cmap='gray') # 绘制灰度图
    print('the label of the sample:',y_train[sample])
    plt.show()

    # data type conversion
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28) # 给数据增加一个维度，使数据和网络结构匹配
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    print("x_train.shape", x_train.shape)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)

    # construct the tensor dataset and data loader
    train = TensorDataset(x_train,y_train)
    loader = DataLoader(dataset=train,batch_size=32,shuffle=True)#,num_workers=2)

    # initialize the model and training modules
    model = LeNet5(10)
    Loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)#,betas=(0.8,0.9))

    # initialize the devices
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU detected! Loading the model to CUDA...')
    gpu_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(gpu_device)

    # training
    training(model,Loss,optimizer,loader,5,(x_train,y_train),(x_test,y_test),'./checkpoint/LeNet5_mnist.pth')


if __name__=='__main__':
    main()