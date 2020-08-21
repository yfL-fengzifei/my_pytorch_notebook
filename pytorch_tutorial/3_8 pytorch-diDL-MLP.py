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

"""
小批量样本 X.size()=n*d，样本数为n,输入个数为d
设多层感知机只有一个隐藏层，隐藏单元个数为h，记隐藏层H，H.size()=d*h
权重参数Wh.size()=d*h
"""
batch_size = 256

# 下载数据集并加载数据集
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

# 因为单个样本图像大小为28*28,则在全连接层中一共有28*28=784个神经元，也就是输入向量的长度为784，因为最后输出对应的是10个类别所以，神经网络的输出层神经元个数为10
num_imputs = 784
num_outputs = 10
num_hiddens=256
W1=torch.tensor(np.random.normal(0,0.01,(num_imputs,num_hiddens)),dtype=torch.float)
b1=torch.zeros(num_hiddens,dtype=torch.float) #标量tensor
W2=torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_outputs)),dtype=torch.float)
b2=torch.zeros(num_outputs,dtype=torch.float) #标量tensor

params=[W1,b1,W2,b2]
# print(params)
for param in params:
    param.requires_grad_(True)

#手动实现relu函数
def relu(X):
    return torch.max(input=X,other=torch.tensor(0.0))

#定义网络模型
def net(X):
    X=X.view(-1,num_imputs)
    H=relu(torch.matmul(X,W1)+b1)
    return torch.matmul(H,W2)+b2

#定义损失函数
loss=torch.nn.CrossEntropyLoss()

#定义循环模型
num_epochs,lr=5,100
"""
...???...什么意思
注：由于原书的mxnet中的SoftmaxCrossEntropyLoss在反向传播的时候相对于沿batch维求和了，而PyTorch默认的是求平均，所以用PyTorch计算得到的loss比mxnet小很多（大概是maxnet计算得到的1/batch_size这个量级），所以反向传播得到的梯度也小很多，所以为了得到差不多的学习效果，我们把学习率调得成原书的约batch_size倍，原书的学习率为0.5，这里设置成100.0。(之所以这么大，应该是因为d2lzh_pytorch里面的sgd函数在更新的时候除以了batch_size，其实PyTorch在计算loss的时候已经除过一次了，sgd这里应该不用除了)
"""
# 评估模型
# 对于评估网络模型，在数据集data_iter上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:  # 这里data_iter是一个batch，X表示的是一个batch的图像，y表示的是一个batch的label,注意这里的label 也就是y,只是一个标量tensor
        acc_sum += (net(X).argmax(
            dim=1) == y).float().sum().item()  # 这里与上面那个accuracy()函数的不同是，并没有利用mean()函数，而是值利用了sum()函数，XXX.float()是将逻辑值转换成0.或1.的二值
        n += y.shape[0]
    return acc_sum / n  # 这里执行的是一个batch的准确率

def train_ch3(net, train_iter, test_iter, loss_fun, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0  # 训练损失，训练准确率
        for X, y in train_iter:
            y_hat = net(X)  # forward
            loss = loss_fun(y_hat, y).sum()  # compute loss

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:  # ...???...这是什么意思
                for param in params:
                    param.grad.data.zero_()

            loss.backward()  # backward

            # ...???...需要再研究优化算法
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

train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
















