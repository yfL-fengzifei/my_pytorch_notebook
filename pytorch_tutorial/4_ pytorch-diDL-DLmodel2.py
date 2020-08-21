#模型参数的访问、初始化和共享
"""
nn中的init模块，包含了多种模型初始化方法

访问模型参数
对于Sequential实例中含模型参数的层，可以通过Module类的parameters()或者named_parameters方法来访问所有参数(以迭代器的形式访问)，后者除了返回参数Tensor外还会返回其名字

对于Sequential实例中含模型参数的层：
for name,param in net.named_parameters()
for param in net.parameters()
for param in net[0].parameters()
print(net[0].parameters(),net[0].weight.size())
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

net=nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))
print(net)
X=torch.rand(2,4)
Y=net(X).sum()

#模型参数访问
print(type(net.named_parameters()))
for name,param in net.named_parameters():
    print(name,param.size())

for param in net.parameters():
    print(param.size())

print(net[0].parameters(),net[0].weight.size())


