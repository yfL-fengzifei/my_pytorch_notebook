#NiN网络中的网络
"""
LeNet,AlexNet,VGG共同点：先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结构
其中AlexNet和VGG对LeNet的改进主要在于如何对这两个模块加宽(增加通道数)和加深

NiN中的网络是另一个思路，即串联多个由卷积层和全连接层构成的小网络来构建一个深层网络
NiN块，如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为4维，
1*1卷积层可以看成是全连接层，其中空间维度(高和宽)上的每个元素相当于样本，通道相当关于特征，因此NiN使用1*1卷积层来代替全连接层，从而使空间信息能够自然传递到后面的层中
NiN块，是由一个卷积层加上两个1*1卷积层串联而成的，其中一个卷积层的超参数可以自行设置，而第二和第三个卷积层的超参数一般是固定的

1*1的卷积也是nn.Conv2d(in_channels,out_channels,kernel_size=1)

NiN模型
NiN使用狷急窗口形状为11*11,5*5,3*3的卷积层，相应的输出通道数与AlexNet一致，每个NiN块后接一个步长2，窗口形状3*3的最大池化层
除了NiN块以外，NiN与Alex显著不同的是，NiN去掉了AlexNet网络最后的3个全连接层，取而代之的是NiN使用了输出通道数等于标签类别数的NiN块，然后使用**全局平均池化层对每个通道中所有元素求平均**并直接用于分类。这里的全局平局池化层即窗口形状等于输入空间维形状的平均池化层，
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim
from collections import OrderedDict
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    blk=nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels,out_channels,kernel_size=1),
        nn.ReLU()
    )
    return blk

#NiN模型

#全局平均池化层
class GlobalAveragePool2d(nn.Module):
    #全局平均池化层可通道将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAveragePool2d,self).__init__()
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:])

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):  # X shape:(batch,*,*,...)
        return X.view(X.shape[0], -1)

net=nn.Sequential(
    nin_block(1,96,kernel_size=11,stride=4,padding=0),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(96,256,kernel_size=5,stride=1,padding=2),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nin_block(256,384,kernel_size=3,stride=1,padding=1),
    nn.MaxPool2d(kernel_size=3,stride=2),
    nn.Dropout(0.5),
    nin_block(384,10,kernel_size=3,stride=1,padding=1),
    GlobalAveragePool2d(),
    FlattenLayer(),

)
print(net)
for name,param in net.named_parameters():
    print(name,param.shape)

X=torch.rand(1,1,224,224)
for name,blk in net.named_children():
    X=blk(X)
    print(name,'output shape: ',X.shape)