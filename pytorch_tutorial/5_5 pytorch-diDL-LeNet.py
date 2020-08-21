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

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc=nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )

    def forward(self,img):
        feature=self.conv(img)
        output=self.fc(feature.view(img.shape[0],-1))
        return output

net=LeNet()
print(net)

# for name,param in net.named_parameters():
#     print(name,param.shape)

num_epochs,lr=5,0.001
loss=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=lr)

def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    with torch.no_grad():
        for X,y in data_iter:
            #if isinstance(net,nn.Module):
            net.eval()
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train()
            n+=y.shape[0]
    return acc_sum/n

def train_ch5(net,train_iter,test_iter,batch_size,optimizer,num_epochs):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,batch_count,start=0.0,0.0,0,0,time.time()
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).float().sum().item() #貌似加不加float()都行
            n+=y.shape[0]
            batch_count+=1
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

train_ch5(net,train_iter,test_iter,batch_size,optimizer,num_epochs)
