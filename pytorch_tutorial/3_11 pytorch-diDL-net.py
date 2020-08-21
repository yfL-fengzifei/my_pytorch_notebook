# net model
"""
一般假设训练集和测试集里的每一个样本都是从同一个概率分布中相互独立地生成的。基于该独立同分布假设，给定一个任意一个机器学习模型（含参数），他的训练误差的期望和泛化误差都是一样的

#模型选择
通常㤇评估若干候选模型的表现并丛总选择模型，这一过程称为模型选择，可供选择的候选模型可以是有着不同超参数的同类模型

#验证集
从严格意义上讲，测试集只能在所有超参数和模型参数选定后使用一次。不可以使用测试数据选择模型，如调参。由于无法从训练误差估计泛化误差，因此也不应只依赖训练数据选择模型。鉴于此，我们可以预留一部分在训练数据集和测试数据集以外的数据来进行模型选择。
这部分数据被称为验证数据集，简称验证集（validation set）。
例如，我们可以从给定的训练集中随机选取一小部分作为验证集，而将剩余部分作为真正的训练集
然而在实际应用中，由于数据不容易获取，测试数据极少只使用一次就丢弃。因此，实践中验证数据集和测试数据集的界限可能比较模糊。

#K 折交叉验证
由于验证数据集不参与模型训练，当训练数据不够用时，预留大量的验证数据显得太奢侈。一种改善的方法是KK折交叉验证（KK-fold cross-validation）。
在KK折交叉验证中，我们把原始训练数据集分割成K个不重合的子数据集，然后我们做K次模型训练和验证。每一次，我们使用一个子数据集验证模型，并使用其他K−1个子数据集来训练模型。在这KK次训练和验证中，每次用来验证模型的子数据集都不同。最后，我们对这KK次训练误差和验证误差分别求平均。

#过拟合和欠拟合
...没看完...

#权重衰减-L2正则化
权重衰减就是L2正则化，正则化通过为模型损失函数增加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段
L2正则化，在模型原算是函数基础上添加L2范数惩罚项，从而得到训练所需最小化的函数，L2范数惩罚项：指的是模型权重参数每个元素的平方和与一个正的常数的乘积
如：线性回归
1.损失函数为 l(w1,w2,b)=(1/n)*sum[(x^i_1*W1+x^i_2*W2+b)-y^i];(i|1<=i<=n)
其中w1，w2是权重参数，b是偏差参数，样本i的输入为x^i_1,x^i_2,标签为y^i,样本数为n,将权重参数用向量W=[w1,w2]表示，带有L2范数惩罚项的新损失函数为
l(w1,w2,b)+(lambda/2n)*||W||**2，||W||**2表示向量的2范数
2.超参数: lambda>0, 当权重参数均为0时，惩罚项最小，当lambda较大时，惩罚项在损失函数中的比重较大，这通常会使学到的权重参数的元素较接近0，当lambda设为0时，惩罚项完全不起作用，||W||**2=w1**2+w2**2
3.在使用L2正则化后，在小批量随机梯度下降中，将线性回归中权重w1和w2的更新参数为
w1=(1-(eta*lambda/|B|))*w1-(eta/|B|)*sum[(x^i_1*w1+x^i_2*w2+b)-y^i],(i in B)
4.由上式可见，L2正则化，令权重参数w1和w2先乘以一个小于1的数，在减去不含惩罚项的梯度，因此，L2正则化又叫权重衰减；权重衰减通过惩罚绝对值较大的模型参数为需要学习的模型增加了限制，从而防止过拟合
5.有时，在惩罚项中添加偏差元素的平方和

6.总结
正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段
权重衰减等价于L2范数正则化，通常会使学到的权重参数的元素接近0
权重衰减可以通过优化器中的weight_decay超参数指定
可以定义多个优化器实例对不同的模型使用不同的迭代方法

"""

# 高位线性回归实验
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


# 画图
def set_figsize(figsize=(3.5, 2.5)):
    plt.rcParams['figure.figsize'] = figsize  # 这相当于字典的形式，设置figigure_size尺寸


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)  # 这是调用的是plt库里的函数semilogy,而不是自定义的semilogy函数
    if x2_vals and y_vals:
        plt.semilogy(x2_vals, y_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


n_train, n_test, num_imputs = 20, 100, 200  # 数量：训练、测试、样本特征
true_w, true_b = torch.ones(num_imputs, 1) * 0.01, 0.05
features = torch.randn(n_train + n_test, num_imputs)  # 样本特征
labels = torch.matmul(features, true_w) + true_b  # 标签
# print(labels)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)  # 加入噪声项
train_features, test_features = features[:n_train, :], features[:n_test,
                                                       :]  # 样本特征 features[:n_train,:]等价于features[:n_train]
train_labels, test_labels = labels[:n_train], labels[n_train:]  # 标签


# 定义随机初始化模型参数
def init_params():
    w = torch.randn((num_imputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义L2正则项
def l2_penalty(w):
    return (w ** 2).sum() / 2


# 定义训练
batch_size, num_epochs, lr = 1, 100, 0.003


# 网络
def linreg(X, w, b):
    return torch.mm(X, w) + b


net = linreg  # 注意这种写法,在没传递参数前不要加括号


# 损失
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2  # **2是按元素相乘的，值得注意的是这里的1/2只是为了将来求梯度的时候好运算


loss = squared_loss  # 注意这种写法

# 定义数据集
dataset = Data.TensorDataset(train_features, train_labels)
train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_params()  # 随机参数化
    train_ls, test_ls = [], []  # 训练误差，测试误差
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)  # 注意这里只对权重参数引入了正则化，但是在不是正则项的时候有偏置参与运算
            l = l.sum()  # 值得注意的是，变成标量损失，以前在没有加入正则化的时候是loss(net(X, w, b), y).sum()

            # 梯度清零
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            # backward
            l.backward()

            # 参数更新
            for param in [w, b]:
                param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data,...???...为什么呢
                # 这里的param包括了偏置和权重参数

        # 每完全遍历一次数据集，利用更新过的参数计算一次网络损失
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())  # 这里计算的是所有样本的损失的平均值，item()函数是只取数字
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())

    # 最终画图，训练完后，画出每个历元后的验证误差和测试误差
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])

    print('L2 norm of w:', w.norm().item())


# 测试一下，没有使用权重衰减的例子
fit_and_plot(3)
