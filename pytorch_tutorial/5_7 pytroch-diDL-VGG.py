#VGG
"""
VGG_block 连续使用数个相同的填充为1，窗口为3*3的卷积层后接上一个步长为2，窗口形状为2*2的最大池化层，卷积层保持输入的高和宽不变，而池化层对其减半

卷积层模块串联数个VGG_block，其超参数由变量conv_arch定义，该变量指定了每个VGG块里卷积层个数和输入输出通道数

VGG有5个卷积块，前2块使用单层卷积，后3块使用双卷积层。第一块的输入输出通道分别为1和64，之后每次对输出通道数翻倍，直到变为512
上述网络使用了8个卷积层和3个全连接层，故叫做VGG11

named_children获取一级子模块及其名字(named_modules会返回所有子模块，包括子模块的子模块)

记载数据集和训练与之前一样，见文档
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

def vgg_block(num_convs,in_channels,out_channels): #指定卷积层的个数、输入输出通道
    blk=[]
    for i in range(num_convs):
        if i==0:
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blk) #为什么是*blk

conv_arch=((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2,512,512)) #这里是提前定义的网络层的结构
fc_features=512*7*7 #c*w*h
fc_hidden_units=4096 #任意


# 直接定义一个卷积拉伸成单一向量的函数
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):  # X shape:(batch,*,*,...)
        return X.view(X.shape[0], -1)

def VGG(conv_arch,fc_features,fc_hidden_units=4096):
    net=nn.Sequential()
    for i,(num_convs,in_channels,out_channels) in enumerate(conv_arch): #注意这种方法，元组的遍历
        net.add_module("vgg_block_"+str(i+1),vgg_block(num_convs,in_channels,out_channels))
    net.add_module("fc",nn.Sequential(
        FlattenLayer(),
        nn.Linear(fc_features,fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units,fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units,10)
    ))
    return net

#网络实例化
net=VGG(conv_arch,fc_features,fc_hidden_units)
print(net)

X=torch.rand(1,1,224,224)
#named_children获取一级子模块及其名字(named_modules会返回所有子模块，包括子模块的子模块)
for name,blk in net.named_children():
    X=blk(X)
    print(name,'output shape:',X.shape)
# for name,parma in net.named_parameters():
#     print(name,'parameters:',parma.shape)

#构造一个通道数更小，或者说更窄的网络在fashion-MNIST数据集上进行训练
ratio=8
small_conv_arch=[(1,1,64//ratio),(1,64//ratio,128//ratio),(2,128//ratio,256//ratio),(2,256//ratio,512//ratio),(2,512//ratio,512//ratio)]
net2=VGG(small_conv_arch,fc_features//ratio,fc_hidden_units//ratio)
print(net2)
X2=torch.rand(1,1,224,224)
for name,blk in net2.named_children():
    X2=blk(X2)
    print(name,'output shape of the net2',X2.shape)



