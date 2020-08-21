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

num_imputs = 784
num_outputs = 10
num_hiddens=256

# 直接定义一个卷积拉伸成单一向量的函数
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):  # X shape:(batch,*,*,...)
        return X.view(X.shape[0], -1)

net=nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_imputs,num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens,num_outputs)
)
#print(list(net.parameters())) #显示所有参数
#print(net[1].weights)
#print(net[1].bias)

#参数初始化
for params in net.parameters(): #一共有4个按顺序是权重1，偏置1，权重2，偏置2
    init.normal_(params,mean=0,std=0.01) #全都初始化为正态分布

#定义损失
loss=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.SGD(net.parameters(),lr=0.5)

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

            # # ...???...需要再研究优化算法
            # if optimizer is None:
            #     for param in params:  # 手动执行sgd优化算法
            #         param.data -= lr * param.grad / batch_size  # 注意batch_size
            # else:
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
num_epochs=5
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

