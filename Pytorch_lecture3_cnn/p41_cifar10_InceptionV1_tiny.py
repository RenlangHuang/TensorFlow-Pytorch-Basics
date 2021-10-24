import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
import tensorflow as tf
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
from p33_training_classification import training


def ConvBNReLU(in_channels,out_channels,kernel_size):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU()
    )

class InceptionV1Module(torch.nn.Module):
    def __init__(self, in_channels,out_channels1, out_channels2reduce,out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV1Module, self).__init__()
        self.branch1 = ConvBNReLU(in_channels, out_channels1, kernel_size=1)
        self.branch2 = torch.nn.Sequential(
            ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1),
            ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=3)
        )
        self.branch3 = torch.nn.Sequential(
            ConvBNReLU(in_channels, out_channels3reduce, kernel_size=1),
            ConvBNReLU(out_channels3reduce, out_channels3, kernel_size=5)
        )
        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            ConvBNReLU(in_channels, out_channels4, kernel_size=1)
        )
    def forward(self,x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class InceptionV1tiny(torch.nn.Module):
    def __init__(self,n_output):
        super(InceptionV1tiny,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            torch.nn.Conv2d(64,64,kernel_size=1,stride=1),
            torch.nn.BatchNorm2d(64),

            torch.nn.Conv2d(64,192, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(192),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception = torch.nn.Sequential(
            InceptionV1Module(in_channels=192,out_channels1=64, out_channels2reduce=96, out_channels2=128, out_channels3reduce = 16, out_channels3=32, out_channels4=32),
            InceptionV1Module(in_channels=256, out_channels1=128, out_channels2reduce=128, out_channels2=192,out_channels3reduce=32, out_channels3=96, out_channels4=64),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionV1Module(in_channels=480, out_channels1=192, out_channels2reduce=96, out_channels2=208, out_channels3reduce=16, out_channels3=48, out_channels4=64)
        )
        self.GlobalAveragePooling2D = torch.nn.AvgPool2d(kernel_size=2, stride=1)
        #self.last_conv = ConvBNReLU(in_channels, out_channels=128, kernel_size=1)
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            torch.nn.Linear(512, n_output)
        )
    
    def forward(self,inputs):
        outputs = self.conv(inputs)
        outputs = self.inception(outputs)
        outputs = self.GlobalAveragePooling2D(outputs)
        outputs = outputs.reshape(outputs.shape[0],-1)
        return self.dense(outputs)


def main():
    # load the dataset
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    np.set_printoptions(precision=2)

    # visualize a sample of the dataset
    sample = np.random.randint(0,x_train.shape[0])
    plt.imshow(x_train[sample], cmap='gray') # 绘制灰度图
    print('the label of the sample:',y_train[sample])
    plt.show()

    # prepare for the tensor dataset
    y_train = torch.LongTensor(y_train).squeeze()
    y_test = torch.LongTensor(y_test).squeeze()
    x_train = torch.FloatTensor(x_train).permute([0,3,1,2])
    x_test = torch.FloatTensor(x_test).permute([0,3,1,2])
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    # construct tensor dataset and data loader
    train = TensorDataset(x_train,y_train)
    loader = DataLoader(dataset=train,batch_size=32,shuffle=True)#,num_workers=2)

    # initialize the model
    model = InceptionV1tiny(10)
    inputs = torch.randn(10, 3, 32, 32)
    print(model(inputs).shape)

    Loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    # initialize the devices
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU detected! Loading the model to CUDA...')
    print('number of GPU: ',torch.cuda.device_count())
    #if torch.cuda.device_count()>1: # the first device id = cuda:id
    #    model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]) # default:using all
    gpu_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(gpu_device)

    # training
    training(model,Loss,optimizer,loader,5,(x_train,y_train),(x_test,y_test),'./checkpoint/InceptionV1tiny_cifar10.pth')

if __name__=='__main__':
    main()