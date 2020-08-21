#easy dropout
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

#下载数据集
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

#定义模型
num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):  # X shape:(batch,*,*,...)
        return X.view(X.shape[0], -1)

#定义模型
drop_prob1,drop_prob2=0.2,0.5
net=nn.Sequential(
    FlattenLayer(),
    nn.Linear(num_inputs,num_hiddens1),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hiddens1,num_hiddens2),
    nn.ReLU(),
    nn.Dropout(drop_prob2),
    nn.Linear(num_hiddens2,num_outputs)
)

num_epochs,lr=5,100.0

#模型参数初始化
for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)

#定义损失函数
loss_fn=nn.CrossEntropyLoss()

#优化器
optimizer=optim.SGD(net.parameters(),lr=0.05)

#评估模型，在测试时，不使用dropout
def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for X,y in data_iter:
        if isinstance(net,nn.Module): #isinstance来判断一个对象是否是一个已知的类型，类似type,值得注意的是type不会认为子类是一种父类类型，不考虑继承关系，isinstance会认为子类是一种父类类型，考虑继承关系，如果p判断两个类型是否相同推荐使用isinstance
            net.eval() #评估模型，会自动关闭dropout
            acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
            net.train() #改回训练模型,...???...为什么还要改回来,
        else: #针对自定义模型
            if('is_training' in net.__code__.co_varnames): #表示如果有is_training这个参数
                acc_sum+=(net(X,is_training=False).argmax(dim=1)==y).float().sum().item()
            else:
                acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
        n+=y.shape[0]
    return acc_sum/n

#定义训练
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

            # # 需要再研究优化算法
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

train_ch3(net,train_iter,test_iter,loss_fn,num_epochs,batch_size,None,None,optimizer)