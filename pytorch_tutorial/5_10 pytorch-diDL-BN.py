#BN
"""
标准化，处理后的任意一个特征在数据集中所有样本上的均值为0，标准差为1，标准化处理输入数据使各个特征的分布相近（使得更容易训练有效的模型）

一般来说，数据标准化预处理对于浅层模型足够有效，随着模型训练的进行，当每层中参数更新时，靠近输出层的输出较难出现剧烈的变化，但对深层神经网络来说，即使输入数据已做标准化，训练中模型参数的更新依然很容易造成靠近输出层输出的剧烈变化。这种计算数值的不稳定性通常难以训练出有效的深度模型

BN(批量归一化)的提出正是为了应对深度模型训练的挑战，在模型训练时，BN利用小批量的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定

**对全连接层做BN
1.通常将BN置于全连接层中的仿射变化和激活函数之间，
设全连接层的输入为u,权重参数和偏差参数分别为W和b，激活函数为phi
批量归一化的运算符为BN
则，使用批量归一化的全连接层的输出为：
phi(BN(x)),其中
x=Wu+b,此时为对输入u的仿射变换

2.令m个样本组成一个小批量，仿射变换的输出为一个新的小批量：
B={x^(1),x^(2),...,x^(m)}
B为BN层的输入，
对于小批量B中的任意样本x^(i)，其BN层的输出同样是相同维度的向量
y^(i)=BN(x^(i))
y^(i)的计算过程如下：
1）.计算小批量B的均值  (mu)_B = (1/m)*sum[x^(i)],(1<=i<=m)
2).计算小批量B的方差  (sigma)^2_B = (1/m)*sum[(x^(i)-(mu)_B)^2] ，这里的平方是按元素求平方
3).按元素对每个样本进行标准化 hat(x)^(i)=[x^(i)-(mu)_B]/[sqrt(sigma^2_B+(epsilon)]，其中epsilon>0是一个很小的常数，保证分母大于0,在上面标准化的基础上，BN层引入了两个可以学习的模型参数，拉伸(scale)参数gamma,和偏移(shift)beta
    这两个参数和x^(i)形状相同，维度相同，两个参数分别于x^(i)按元素进行惩罚和加法计算
4).y^(i)=(gamma).*hat(x)^(i)+(beta)
5).最终得到了x^(i)的批量归一化的输出y^(i),可学习的拉伸和偏移参数保留了不对hat(x)^(i)做批量归一化的可能，
如果批量归一化无益，理论上，学出的模型可以不使用批量归一化
"""
"""
**对卷积层做BN
对卷积层来说，批量归一化发生在卷积计算之后，应用激活函数之前。如果卷积计算输出多个通道，需要对这些通道的输出分别做BN，且每个通道都拥有独立的大神和偏移参数，并均为标量。
这小批量中有m个样本，在但个通道上，假设卷积计算输出的高和宽分别为p和q,就需要对该通道中的m*p*q个元素同时最批量归一化BN，对这些元素做标准化计算时，使用相同的均值和方差，即该通道中m*p*q个元素的均值和方差
"""
"""
**预测时的BN
使用BM训练时，可以量批量大小设置的大一点，从而使批量内样本的均值和方差的计算都较为准确，将训练好的模型用于预测时，希望模型对于任意输入都有确定的输出，因此当样本的输出不应该取决于批量归一化所需要的随机小批量中的均值和方差，一种常用的方法就是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用他们确定的输出，...???...没理解
因此，和dropout一样，批量归一化在训练模式和预测模型下的计算结果也是不一样的

"""
"""
手动实现没看，见文档
简洁实现: nn.BatchNorm1d();nn.BatchNorm2d

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

#LeNet+BN

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, X):  # X shape:(batch,*,*,...)
        return X.view(X.shape[0], -1)

net=nn.Sequential(
    nn.Conv2d(1,6,5),
    nn.BatchNorm2d(6),
    nn.Sigmoid(),
    nn.MaxPool2d(2,2), #kernel_size,stride
    nn.Conv2d(6,16,5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2,2),
    FlattenLayer(),
    nn.Linear(16*4*4,120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120,84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84,10)

)


