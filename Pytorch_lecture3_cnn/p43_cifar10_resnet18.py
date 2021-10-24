import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import numpy as np
import tensorflow as tf
from torch.autograd import Variable
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset,DataLoader


class ResNet18(torch.nn.Module):
    def __init__(self,n_output):
        super(ResNet18,self).__init__()

