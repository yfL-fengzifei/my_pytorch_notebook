#模型构造
"""
Module类是nn模块里提供的一个模型构造类，是所有神经网络模块的基类，从而可以继承它来定义想要的模型。
用户定义的类需要重载Module类的__int__函数和forward函数，分别用于创建模型参数和定义前向计算

实例化网络后，如net=MLP()
net(X)会自动调用MLP继承自Module类的__call__函数，这个函数将自动调用MLP类定义的forward函数来完成前向计算

Module类是一个通用的部件，事实上，pytorch还实现了继承自Module的可以方便构建模型的类，如Sequential,ModuleList,ModuleDict等

Sequential类
当模型的前向计算为简单串联各个层的计算时，Sequential类可以通过更加简单的方式定义模型。Sequential类，可以接收一个子模块的有序字典(OrderedDict)或者一系列子模块作为参数来逐一添加Module的实例，而模型的前向计算就是讲这些实例按添加的顺序逐一计算

ModuleList类，ModuleList接收一个子模块的列表作为输入，然后也可以类似list那样进行append和extend操作

Sequential和ModuleList都可以进行列表化构造网络，区别：
ModuleList仅仅是一个存储各种模块的列表，这些模块之间没有联系没有顺序(所以不同保证相邻层的输入输出维度匹配)，而且没有实现forward功能需要自己实现，所以当调用net(torch.zeros(1,784))会报错误(NotImplementedError)，利用certain_net.append()外部增加模块（来添加）
Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现，利用certain_net.add_model('linear', nn.Linear())来添加
#ModuleList只是为了让网络定义前向传播时更加灵活,ModuleList可以是一个可迭代 或是可索引
#同时，ModuleList不同于一般的python的list，加入到ModuleList里面的所有模块的参数会被自动添加到整个网络中，

ModuleDict接收一个子模块的字典作为输入，然后也可以类似字典那样进行添加访问操作,利用certain_net['layer_name':nn.Linear()]来添加
ModuleDict与ModuleList,ModuleDict实例仅仅是存放了一些模块的字典，并没有定义forward函数需要自己定义，凡是ModuleDict也与python的Dict有所不同，ModuleDict里的所有模块的参数会自动添加到整个网络中

构造更复杂的模型
get_constant函数创建训练中不被迭代的参数，即常数参数

还可以进行网络的嵌套
见文档
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

#定义MLP
class MLP(nn.Module):
    # def __init__(self,**kwargs):
    # 调用MLP父类Module的构造函数来进行必要的初始化，这样在构造实例时还可以指定其他函数参数，
    #     super(MLP,self).__init__()
    def __init__(self):
       super(MLP,self).__init__()
       self.hidden=nn.Linear(784,256)
       self.act=nn.ReLU()
       self.output=nn.Linear(256,10)

    def forward(self,X):
        a=self.act(self.hidden(X))
        return self.output(a)
net=MLP()
X=torch.rand(2,784)
print(net)
print(net(X))
print(net.parameters()) #只会打印类别
# for p in net.parameters():
#     print(p)

#ModelList
# net2=nn.ModuleList([nn.Linear(784,256),nn.ReLU()])
# net2.append(nn.Linear(256,10)) #类似list的append操作
# print(net2[-1])
# print(net2)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.linears=nn.ModuleList([nn.Linear(10,10) for i in range(10)])

    def forward(self,X):
        for i,l in enumerate(self.linears):
            X=self.linears[i//2](X)+l(X)
        return X

net2=MyModel()
print(net2)
print(net2.parameters()) #只会打印类别
# for p2 in net2.parameters():
#     print(p2)

#ModuleDict
net3=nn.ModuleDict({'linear':nn.Linear(784,256),'act':nn.ReLU()})
net3['output']=nn.Linear(256,10)
print(net3)
print(net3['linear'])
print(net3.output)

#稍复杂的网络
class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP,self).__init__()
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        self.linear=nn.Linear(20,20)
    def forward(self,x):
        x=self.linear(x)
        x=F.relu(torch.mm(x,self.rand_weight.data)+1)

        x=self.linear(x)
        while x.norm().item()>1:
            x/2
        if x.norm().item()<0.8:
            x*=10
        return x.sum()


