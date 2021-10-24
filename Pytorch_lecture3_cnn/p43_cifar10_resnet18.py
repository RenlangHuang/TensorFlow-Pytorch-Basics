import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torchvision
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
from p33_training_classification import training


def ResNet18(n_output):
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, n_output)
    #print(model)
    return model


def main():
    # dataset preparation
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    np.set_printoptions(precision=2)

    # visualize a sample in the dataset
    sample = np.random.randint(0,x_train.shape[0])
    plt.imshow(x_train[sample], cmap='gray') # 绘制灰度图
    print('the label of the sample:',y_train[sample])
    plt.show()

    # data type conversion
    y_train = torch.LongTensor(y_train).squeeze()
    y_test = torch.LongTensor(y_test).squeeze()
    x_train = torch.FloatTensor(x_train).permute([0,3,1,2])
    x_test = torch.FloatTensor(x_test).permute([0,3,1,2])
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_test.shape:", x_test.shape)
    print("y_test.shape:", y_test.shape)

    # construct the tensor dataset and data loader
    train = TensorDataset(x_train,y_train)
    loader = DataLoader(dataset=train,batch_size=32,shuffle=True)#,num_workers=2)

    # initialize the model and training modules
    model = ResNet18(10)
    Loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0002)

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
    training(model,Loss,optimizer,loader,10,(x_train,y_train),(x_test,y_test),'./checkpoint/ResNet18_cifar10.pth')

if __name__=='__main__':
    main()
