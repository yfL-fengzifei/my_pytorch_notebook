#dropout
"""
应对过拟合的另一个方法是dropout
dropout方法有一些不同的变种，这里说的是倒置丢弃法

1.对网络的隐藏层应用丢弃法时，该层的隐藏单元将有一定概率被丢弃，设丢弃概率为p,那么有p的概率隐藏单元会被清零，1-p的概率隐藏单元会除以1-p做拉伸。
2.dropout概率p是dropout的超参数。具体来说，设随机变量(xi)为0和1的概率为p和1-p,使用dropout时，计算新的隐藏单元h'i，h'i=（(xi)/(1-p)）*h'i
3.随机dropout之后，输出层的计算无法过度依赖h1-h5中的任意一个，从而在训练模型时起到正则化的作用，并可以用来应对过拟合。
4.测试模型时，为了拿到更加确定性的结果，一般不使用丢弃法
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

#定义dropout函数
def dropout(X,drop_prob):
    X=X.float()
    assert 0<=drop_prob<=1
    keep_prob=1-drop_prob

    #说明这是全部丢弃的
    if keep_prob==0:
        return torch.zeros_like(X) #创建一个为X相同维度的全0tensor

    #随机dropout
    mask=(torch.randn(X.shape)<keep_prob).float() #比较之后返回的是逻辑值tensor，使用float函数返回的是0,1二值tensor

    return mask*X/keep_prob #隐藏层的神经元的个数不变，只是将其为设为0,或这保持不变，同时这里采用了拉伸的方法，保持dropout后的期望不变

# #简单测试dropout
# X=torch.arange(16).view(2,8)
# print(dropout(X,0))
# print(dropout(X,0.5))
# print(dropout(X,1.0))

#定义模型参数
num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256

W1 = torch.tensor(np.random.normal(0, 0.01, size=(num_inputs, num_hiddens1)), dtype=torch.float, requires_grad=True)
b1 = torch.zeros(num_hiddens1, requires_grad=True)
W2 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens1, num_hiddens2)), dtype=torch.float, requires_grad=True)
b2 = torch.zeros(num_hiddens2, requires_grad=True)
W3 = torch.tensor(np.random.normal(0, 0.01, size=(num_hiddens2, num_outputs)), dtype=torch.float, requires_grad=True)
b3 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2, W3, b3]

#定义模型
drop_prob1,drop_prob2=0.2,0.5
def net(X,is_training=True):
    X=X.view(-1,num_inputs)
    H1=(torch.matmul(X,W1)+b1).relu()
    if is_training: #只在训练模型时使用丢弃法
        H1=dropout(H1,drop_prob1)
    H2=(torch.matmul(H1,W2)+b2).relu()
    if is_training: #只在训练模型时使用丢弃法
        H2=dropout(H2,drop_prob2)
    return torch.matmul(H2,W3)+b3

#评估模型，在测试时，不使用dropout
def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for X,y in data_iter:
        if isinstance(net,nn.Module): #isinstance来判断一个对象是否是一个已知的类型，类似type,值得注意的是type不会认为子类是一种父类类型，不考虑继承关系，isinstance会认为子类是一种父类类型，考虑继承关系，如果p判断两个类型是否相同推荐使用isinstance
            net.eval() #评估模型，会自动关闭dropout
            acc_sum+=(net(X).argmax(dim=1)==y).float().sum.item()
            net.train() #改回训练模型,...???...为什么还要改回来
        else: #针对自定义模型
            if('is_training' in net.__code__.co_varnames): #表示如果有is_training这个参数
                acc_sum+=(net(X,is_training=False).argmax(dim=1)==y).float().sum().item()
            else:
                acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n

num_epochs,lr,batch_size=5,100.0,256

mnist_train = tv.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=True, download=True,
                                       transform=transforms.ToTensor())
test_train = tv.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=False, download=True,
                                      transform=transforms.ToTensor())
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4
train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)

loss=torch.nn.CrossEntropyLoss()


def train_ch3(net, train_iter, test_iter, loss_fun, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0  # 训练损失，训练准确率
        for X, y in train_iter:
            y_hat = net(X)  # forward
            loss = loss_fun(y_hat, y).sum()  # compute loss,并转换成标量损失

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:  # ...???...这是什么意思
                for param in params:
                    param.grad.data.zero_()

            loss.backward()  # backward

            # 需要再研究优化算法
            if optimizer is None:
                for param in params:  # 手动执行sgd优化算法
                    param.data -= lr * param.grad / batch_size  # 注意batch_size
            else:
                optimizer.step()

            train_l_sum += loss.item()

            #训练准确率，这两行如evaluate_accuracy函数类似
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        #每遍历一次完整的训练集，进行一次测试集的准确度检测
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

# 开始训练
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params,lr=lr,optimizer=None)









