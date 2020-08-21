# start with softmax regression

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

# 因为单个样本图像大小为28*28,则在全连接层中一共有28*28=784个神经元，也就是输入向量的长度为784，因为最后输出对应的是10个类别所以，神经网络的输出层神经元个数为10
num_imputs = 784
num_outputs = 10
# 定义权重参数，对于单层网络来说一共有784*10个权重参数(最好说成10*784)
W = torch.tensor(np.random.normal(0, 0.01, (num_imputs, num_outputs)), requires_grad=True)  # 权重被初始化为符合正态分布的参数
b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)  # 偏置参数被初始化为0


# 如果上述创建的时候不添加requires_grad，那么需要下面手动设置属性
# W.requires_grad_(True)
# b.requires_grad_(True)


# softmax运算的实现,这是个例子
# X=torch.tensor([[1,2,3],[4,5,6]])
# print(X.sum(dim=0,keepdim=True)) #keepdim=True表示tensor维度操作后，在结果中保留行和列这两个维度
# print(X.sum(dim=1,keepdim=True))


# X行数为样本数，列数为输出个数，X.exp()先对每个元素做指数运算，在对每行元素求和，最后令矩阵每行个元素与该元素之和相除，最终输出的矩阵中的任意一行元素代表了样本在各个输出类别上的预测概率
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 值得注意的是，这里本质上作用的也是batch数据，但是X经过net()函数处理后，输入到softmax时已经是2维的tensor

# # 伪数据测试,测试一下softmax函数
# X = torch.rand((2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))


# 定义模型
# 特别注意，这里没有问题，这里考虑了batch的因因素，X.view((-1,num_outputs))，形状的X.shape=torch.Size([batch,784])
def net(X):
    return softmax(torch.mm(X.view((-1, num_imputs)).float(), W.float()) + b)  # 这里的参数被随机初始化了


# 调用softmax()函数, 返回的是, 每一行表示一个样本的输出结果，一行元素代表了样本在各个输出类别上的预测概率
# X.view(-1,num_imputs)将每张原始图像改成长度为num_imputs的向量，X.view(-1,num_imputs)返回的是一个tensor向量，然后torch.mm来计算tensor的之间的乘法
# 这里有一个bug,如果报错就将上面改成torch.mm(X.view((-1, num_imputs)).float(), W.float)...???...不知道是为什么


# 定义损失函数
# 为了得到标签的预测概率,可以使用gather函数，
# 变量y_hat是2个样本在3个类的预测概率，变量y是2个样本的标签类别。通过使用gather函数，可以得到2个样本的预测概率，
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])  # 这里假设ground truth的类别是0


# y_hat.gather(1, y.view(-1, 1))  # 索引特定维度上的数据, tensor([[0.1000],[0.5000]])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))  # 每一行表示一个损失函数计算的结果


# 计算分类准确率
"""
给定一个类别的预测概率分布y_hat，把预测概率最大的类别作为输出类别，如果与真实类别y一直，说明这次预测是正确的。分类准确率即正确预测数量与预测数量之比
"""


# # 定义准确率函数
# #这里评估的是伪数据的表现
# def accuracy(y_hat, y):
#     return (y_hat.argmax(
#         dim=1) == y).float().mean().item()  # argmax()函数返回在指定维度下的最大值索引号,逻辑值转换成float数据类型，返回的是0值和1值,当转换成0和1之后，计算均值就相当于准确率

# 利用上述的微数据，计算准确率
# print(accuracy(y_hat, y))


# 对于评估网络模型，在数据集data_iter上的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:  # 这里data_iter是一个batch，X表示的是一个batch的图像，y表示的是一个batch的label,注意这里的label 也就是y,只是一个标量tensor
        acc_sum += (net(X).argmax(
            dim=1) == y).float().sum().item()  # 这里与上面那个accuracy()函数的不同是，并没有利用mean()函数，而是值利用了sum()函数，XXX.float()是将逻辑值转换成0.或1.的二值
        n += y.shape[0]
    return acc_sum / n  # 这里执行的是一个batch的准确率


# 调用函数
# 因为随机初始化了网络net,所以这个随机模型的准确率应该接近于类别个数10的导数即0.1
print(evaluate_accuracy(test_iter, net=net))  # 调用函数的时候，不用加上()

"""
注意这列的训练模型咱暂时不能运行
"""
'''
# 训练模型
num_epochs, lr = 5, 0.1
#定义训练模型
def train_ch3(net, train_iter, test_iter, loss_fun, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
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
            if optimizer is not None:
                for param in params:  # 手动执行sgd优化算法
                    param.data -= lr * param.grad / batch_size #注意batch_size
            else:
                optimizer.step()

            train_l_sum+=loss.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            
            pass #函数还没有写完

#调用函数，执行训练
train_ch3(net=net,train_iter=train_iter,test_iter=test_iter,loss_fun=cross_entropy,num_epochs=num_epochs,batch_size=batch_size,[W,b],lr=lr)
'''


# 进行预测
# 定义函数，将数值标签转换成文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]  # 以列表的形式返回,传入的labels也是列表，是可迭代对象


# 可以在一行里画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # 在一行上输出一个batch
    # figsize设置窗口的尺寸...???...
    # ...???...我猜返回的是窗口
    for f, img, lbl in zip(figs, images,
                           labels):  # 值得注意的是zip将对象中对应的元素打包成一个个元祖，然后返回这些由元素组成的对象，并可以用list()函数生成列表，对应元素的尺寸应该相等
        f.imshow(img.view(28, 28).numpy())  # 转换图像尺寸，并显示，显示是以numpy的形式
        f.set_title(lbl)  # 设置标题
        f.axes.get_xaxis().set_visible(False)  # 设置x轴
        f.axes.get_yaxis().set_visible(False)  # 设置y轴
    plt.show()  # 显示，...???... f.imshow()与plotshow的区别


X, y = iter(test_iter).next()
true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())  # argmax返回的是最大概率值下的索引号，0,1,2....等
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])
