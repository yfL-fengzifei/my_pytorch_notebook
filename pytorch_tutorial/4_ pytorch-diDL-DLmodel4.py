# save and load DL_model
"""
可以直接使用save函数和load函数分别存储和读取Tensor,save使用python的pickle使用程序将对象进行序列化，然后将序列化的对象保存早disk,使用save可以保存各种对象，包括模型、张量、字典等。load函数使用pickle unpickle工具将pickle的对象文件反序列化为内存

在pytorch中，module的可学习参数(即权重和偏差)，模块模型包含在参数中(通过module.parameters()访问)，state_dict是一个从参数名称隐射到参数tensor的字典对象
只有具有可学习参数的层(卷积层、线性层)才有state_dict中的条目，优化器(optim)也有一个state_dict,其中包含关于优化器状态以及所使用的超参数的信息
net.state_dict()
optimizer.state_dict()

保存和加载模型
pytorch中保存和加载训练模型有两种方法：
1.仅保存和加载模型参数(state_dict): 推荐方式
保存：
torch.save(model.state_dict(),PATH) #推荐的文件后缀名是pt和pth
加载：
model=TheModelClass()
model.load_state_dict(torch.load(PATH))
2.保存和加载整个模型
保存：
torch.save(model,PATH)
加载：
model=torch.laod(PATH)
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

# x=torch.ones(3)
# torch.save(x,'x.pt')
# x2=torch.load('x.pt')
# print(x2)

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden=nn.Linear(3,2)
        self.act=nn.ReLU()
        self.output=nn.Linear(2,1)
    def forward(self,x):
        a=self.act(self.hidden(x))
        return self.output(a)

net=MLP()
net.state_dict()
print(net.state_dict())

optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
optimizer.state_dict()
print(net.state_dict())

X=torch.randn(2,3)
Y=net(X)

PATH="./net.pt"
torch.save(net.state_dict(),PATH)

net2=MLP()
net2.load_state_dict(torch.load(PATH))
Y2=net2(X)
print(Y2)
print(Y2==Y)
