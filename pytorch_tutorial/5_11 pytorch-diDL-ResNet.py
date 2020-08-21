#ResNet
"""
question:
对神经网络模型添加新的层，充分训练后的模型是否只可能更有效的降低训练误差？
理论上，原模型解的空间只是新模型解的空间的子空间。也就是说，如果将新添加的层训练成恒等映射f(x)=x,新模型的原模型将同样有效，
由于新模型可能得出更优的解来拟合数据集，因此添加层似乎更容易降低训练误差。
然而，在实践中，添加过多的层后训练误差往往不降反升。即使利用BN带来的数值稳定性使训练深层模型更加容易，但是该问题仍然存在。
针对上述问题提出了ResNet

设输入为x,假设希望学出的理想映射为f(x),然后将f(x)作为激活函数的输入。
一种方法：x->[加权运算->激活函数->加权运算]->[f(x)]->激活函数
一种方法：x->[加权运算->激活函数->加权运算]->[f(x)-x]->   -->[f(x)]->激活函数
         x->--------------------------------------->(+)
第一种方法需要直接拟合该映射f(x)
第二中方法需要拟合出有关恒等映射的残差映射[f(x)-x]，残差映射在实际中往往更容易优化

现在假设希望学到的理想映射为恒等映射[f(x)=x],则此时在第二种方法中只需要两第二个加权运算的权重和偏差参数学习成0，那么f(x)即为恒等映射

实际中，当理想映射f(x)极接近恒等映射时，残差映射也易于捕捉恒等映射的细微波动。在残差块中，输入可通过跨层的数据线路更快地前向传播


ResNet沿用了VGG全3*3卷积层的设计，残差块里首先有2个相同输出通道数的3*3卷积层。每个卷积层后接一个批量归一化层和ReLU激活函数。然后将输入跳过这两个卷积运算后直接加载最后的ReLU激活函数前，这样的设计要求两个卷积层的输出与输入形状一样，从而可以相加，如果想改变通道数，就需要引入一个额外的1*1卷积来讲输入变成需要的形状后再做相加运算

ResNet模型的前两层和之前介绍的GoogleNet中的一样，在输出通道数为64，步长为2的7*7卷积层后接步长为2个3*3的最大池化层，不同之处在于ResNet每个卷积层后增加了BN层

googleNet在后面接了4由inception块组成的模块
ResNet则使用4个残差块组成的模块，每个模块使用若干个同样输出通道的残差块，第一个模块的通道数同输入通道数一致，由于之前使用了步长为2的最大池化层，所以无需减小高和宽，轴的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半


...???...没看完
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

class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1):
        super(Residual,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        return F.relu(Y+X)

#简单测试
# blk=Residual(3,3)
# print(blk)
#
# X=torch.rand((4,3,6,6))
# print(blk(X).shape)
#
# blk2=Residual(3,6,use_1x1conv=True,stride=2)
# print(blk2(X).shape)

#ResNet模型
net=nn.Sequential(
    nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

def resnet_block(in_channels,out_channels,num_residuals,fist_block=False):
    if fist_block:
        assert in_channels==out_channels
    blk=[]
    for i in range(num_residuals):
        if i==0 and not fist_block:
            blk.append(Residual(in_channels,out_channels,use_1x1conv=True,stride=2))
        else:
            blk.append(Residual(out_channels,out_channels))
    return nn.Sequential(*blk)





