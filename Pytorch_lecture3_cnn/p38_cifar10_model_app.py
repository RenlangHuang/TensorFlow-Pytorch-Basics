import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from p33_training_classification import evaluate
from p37_cifar10_alexnet8 import AlexNet8
from p39_cifar10_vgg16 import VGG16
from p41_cifar10_InceptionV1_tiny import InceptionV1tiny
from p43_cifar10_resnet18 import ResNet18


def load_pretrained_model(name):
    if name=='AlexNet':
        model = AlexNet8(10)
        if os.path.exists('./checkpoint/AlexNet8_cifar10.pth'):
            print('------------load the model----------------')
            model.load_state_dict(torch.load('./checkpoint/AlexNet8_cifar10.pth'))
    elif name=='VGG':
        model = VGG16(10)
        if os.path.exists('./checkpoint/VGG16_cifar10.pth'):
            print('------------load the model----------------')
            model.load_state_dict(torch.load('./checkpoint/VGG16_cifar10.pth'))
    elif name=='Inception':
        model = InceptionV1tiny(10)
        if os.path.exists('./checkpoint/InceptionV1tiny_cifar10.pth'):
            print('------------load the model----------------')
            model.load_state_dict(torch.load('./checkpoint/InceptionV1tiny_cifar10.pth'))
    else:
        model = ResNet18(10)
        if os.path.exists('./checkpoint/ResNet18_cifar10.pth'):
            print('------------load the model----------------')
            model.load_state_dict(torch.load('./checkpoint/ResNet18_cifar10.pth'))
    return model


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

y_train = torch.LongTensor(y_train).squeeze()
y_test = torch.LongTensor(y_test).squeeze()
x_train = torch.FloatTensor(x_train).permute([0,3,1,2])
x_test = torch.FloatTensor(x_test).permute([0,3,1,2])
print("x_train.shape:", x_train.shape)
print("y_train.shape:", y_train.shape)
print("x_test.shape:", x_test.shape)
print("y_test.shape:", y_test.shape)


model = load_pretrained_model('VGG')
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU detected! Loading the model to CUDA...')
print('number of GPU: ',torch.cuda.device_count())
#if torch.cuda.device_count()>1: # the first device id = cuda:id
#    model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]) # default:using all
gpu_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(gpu_device)
model.eval()


# evaluate the pre-trained model
accuracy_train = evaluate(model,x_train.cuda(),y_train.cuda())
accuracy_test = evaluate(model,x_test.cuda(),y_test.cuda())
print('Training_accuracy = %.4f, Validation_accuracy = %.4f'%\
    (accuracy_train,accuracy_test))


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
