# googleNet and inception
"""
googleNet借鉴了NiN串联网络的思想，并在此基础上做了很大的改进

Inception 块
输入：
    1*1卷积----------------------->
    1*1卷积-3*3卷积--------------->
    1*1卷积-5*5卷积--------------->
    3*3最大池化------------------->
=>通道合并层
inception块里有四条并行的路线，前3条路线使用窗口大小分别为1*1,3*3，5*5卷积层来抽取不同空间尺寸下的信息，其中中间线路会对输入先做1*1卷积来减少输入通道数，以降低模型复杂度，第4条线路则使用3*3最大池化层，后接1*1卷积来改变通道数。4条线路都使用了合适的填充来使输入与输出的高和宽一致。
最后将每条路线的输出在通道维上连结，并输入接下来的层中去

googleNet模型
每个模块之间使用步长为2的383最大池化层来减小输出高宽
1.第一个模块是用一个64通道的7*7卷积层
2.第二个模块使用两个卷积层，首先是64通道的1*1卷积，然后将通道增大3倍的3*3卷积层，对应inception块中的第二条路线
3.第三个模块，串联两个完整的inception块，第一个inception块的输出通道数为64+128+32+32=256，其中4条路线的输出通道数比例为64:128:32:32=2:4:1:1。
其中第二、第三条线路先分别为将输入通道数减小至96/192=1/2,16/192=1/12后，再接上第二层卷积层。第二个inception块输出通道数增至128+192+96+64=480,每条线路的输出通道数之比为128:192:96:64=4:6:3:2，其中第二、第三条线路先分别将输入通道数减少至128/256=1/2和32/256=1/8
4.第四个模块，串联了5个inception块，其输出通道数分别是
192+208+48+64=512，
160+224+64+64=512
128+256+64+64=512
112+288+64+64=528
256+320+128+128=832
这些线路的通道上述分配和第三模块中的类似，首先3*3卷积层的第二条线路输出最多通道，其次仅含1*1卷积层的第一条路线，之后是含5*5卷积层的第三路线和含3*3最大池化层的第四条路线，其中第二条、第三条都会先按比例减少通道数，这些比例在inception块中都略有不同
5.第五模块，有输出通道数为256+320+128+128=832和384+384+128+128=1024的两个Inception块，其中每条路线的通道数的分配思路和第三、四模块中的一致，只是在具体数值上有所不同，
需要注意的是，第五模块的很后面紧跟输出层，该模块同NiN一样使用全局平均池化层来将每个通道的高和宽变成1，最后将输出变成二维数组后接上一个输出个数为标签类别数的全连接层
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


# 全局平均池化
class GlobalAveragePool2d(nn.Module):
    # 全局平均池化层可通道将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):  # X shape:(batch,*,*,...)
        return X.view(X.shape[0], -1)


# inception
class Inception(nn.Module):
    # c1-c4为每条线路中层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1*1卷积
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2,1*1卷积层+3*3卷积
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3,1*1卷积层+5*5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4,3*3最大池化层+1*1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


# GoogleNet模型
# 第一个模块
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第二个模块
b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第三模块
b3 = nn.Sequential(
    Inception(192, 64, (96, 128), (16, 32), 32),
    Inception(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第四模块
b4 = nn.Sequential(
    Inception(480, 192, (96, 208), (16, 48), 64),
    Inception(512, 160, (112, 224), (24, 64), 64),
    Inception(512, 128, (128, 256), (24, 64), 64),
    Inception(512, 112, (144, 288), (32, 64), 64),
    Inception(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 第五模块
b5 = nn.Sequential(
    Inception(832, 256, (160, 320), (32, 128), 128),
    Inception(832, 384, (192, 384), (48, 128), 128),
    GlobalAveragePool2d()
)

# 整个网络模型
net = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024, 10))
print(net)

x=torch.rand(1,1,224,224)
for blk in net.children():
    x=blk(x)
    print(x.shape)
