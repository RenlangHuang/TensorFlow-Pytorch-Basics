import torch
from torch import nn

### refer to https://codechina.csdn.net/mirrors/shanglianlm0525/pytorch-networks/-/tree/master/

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class InceptionV2ModuleA(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleA, self).__init__()
        self.branch1 = ConvBNReLU(in_channels,out_channels1,kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1),
            ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels3reduce, kernel_size=1),
            ConvBNReLU(out_channels3reduce, out_channels3, kernel_size=3, padding=1),
            ConvBNReLU(out_channels3, out_channels3, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels4, kernel_size=1),
        )
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class InceptionV2ModuleB(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleB, self).__init__()
        self.branch1 = ConvBNReLU(in_channels,out_channels1,kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1),
            ConvBNReLU(out_channels2reduce, out_channels2reduce, kernel_size=[1,3],padding=[0,1]),
            ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=[3,1],padding=[1,0])
        )
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels3reduce,kernel_size=1),
            ConvBNReLU(out_channels3reduce, out_channels3reduce, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(out_channels3reduce, out_channels3reduce, kernel_size=[3, 1], padding=[1, 0]),
            ConvBNReLU(out_channels3reduce, out_channels3reduce, kernel_size=[1, 3], padding=[0, 1]),
            ConvBNReLU(out_channels3reduce, out_channels3, kernel_size=[3, 1], padding=[1, 0]),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels4, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)

class InceptionV2ModuleC(nn.Module):
    def __init__(self, in_channels,out_channels1,out_channels2reduce, out_channels2, out_channels3reduce, out_channels3, out_channels4):
        super(InceptionV2ModuleC, self).__init__()

        self.branch1 = ConvBNReLU(in_channels,out_channels1,kernel_size=1)

        self.branch2_conv1 = ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1)
        self.branch2_conv2a = ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=[1,3], padding=[0,1])
        self.branch2_conv2b = ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=[3,1], padding=[1,0])

        self.branch3_conv1 = ConvBNReLU(in_channels, out_channels3reduce,kernel_size=1)
        self.branch3_conv2 = ConvBNReLU(out_channels3reduce, out_channels3, kernel_size=3,stride=1,padding=1)
        self.branch3_conv3a = ConvBNReLU(out_channels3, out_channels3, kernel_size=[3, 1],padding=[1, 0])
        self.branch3_conv3b = ConvBNReLU(out_channels3, out_channels3, kernel_size=[1, 3],padding=[0, 1])

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, out_channels4, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        x2 = self.branch2_conv1(x)
        out2 = torch.cat([self.branch2_conv2a(x2), self.branch2_conv2b(x2)],dim=1)
        x3 = self.branch3_conv2(self.branch3_conv1(x))
        out3 = torch.cat([self.branch3_conv3a(x3), self.branch3_conv3b(x3)], dim=1)
        out4 = self.branch4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionV3ModuleD(nn.Module):
    def __init__(self, in_channels,out_channels1reduce,out_channels1,out_channels2reduce, out_channels2):
        super(InceptionV3ModuleD, self).__init__()
        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels1reduce, kernel_size=1),
            ConvBNReLU(out_channels1reduce, out_channels1, kernel_size=3,stride=2,padding=1)
        )
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, out_channels2reduce, kernel_size=1),
            ConvBNReLU(out_channels2reduce, out_channels2, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(out_channels2, out_channels2, kernel_size=3, stride=2,padding=1),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out


class InceptionV2(nn.Module):
    def __init__(self, num_classes=1000, stage='train'):
        super(InceptionV2, self).__init__()
        self.stage = stage

        self.block1 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=64, kernel_size=7,stride=2,padding=3),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.block2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
        )
        self.block3 = nn.Sequential(
            InceptionV2ModuleA(in_channels=192,out_channels1=64,out_channels2reduce=64, out_channels2=64, out_channels3reduce=64, out_channels3=96, out_channels4=32),
            InceptionV2ModuleA(in_channels=256, out_channels1=64, out_channels2reduce=64, out_channels2=96,out_channels3reduce=64, out_channels3=96, out_channels4=64),
            InceptionV3ModuleD(in_channels=320, out_channels1reduce=128, out_channels1=160, out_channels2reduce=64,out_channels2=96),
        )
        self.block4 = nn.Sequential(
            InceptionV2ModuleB(in_channels=576, out_channels1=224, out_channels2reduce=64, out_channels2=96,out_channels3reduce=96, out_channels3=128, out_channels4=128),
            InceptionV2ModuleB(in_channels=576, out_channels1=192, out_channels2reduce=96, out_channels2=128,out_channels3reduce=96, out_channels3=128, out_channels4=128),
            InceptionV2ModuleB(in_channels=576, out_channels1=160, out_channels2reduce=128, out_channels2=160,out_channels3reduce=128, out_channels3=128, out_channels4=128),
            InceptionV2ModuleB(in_channels=576, out_channels1=96, out_channels2reduce=128, out_channels2=192,out_channels3reduce=160, out_channels3=160, out_channels4=128),
            InceptionV3ModuleD(in_channels=576, out_channels1reduce=128, out_channels1=192, out_channels2reduce=192,out_channels2=256),
        )
        self.block5 = nn.Sequential(
            InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160,out_channels3reduce=160, out_channels3=112, out_channels4=128),
            InceptionV2ModuleC(in_channels=1024, out_channels1=352, out_channels2reduce=192, out_channels2=160,
                               out_channels3reduce=192, out_channels3=112, out_channels4=128)
        )
        self.conv = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4, self.block5,
            nn.MaxPool2d(kernel_size=7, stride=1),
            nn.Dropout(0.5)
        )
        self.dense = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out


if __name__=='__main__':
    model = InceptionV2(10)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)
