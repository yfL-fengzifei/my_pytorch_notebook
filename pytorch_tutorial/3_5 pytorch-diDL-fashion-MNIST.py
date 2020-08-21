# a multi-classed datset
"""
介绍一个多类图像分类数据集
"""

# torchvision
"""
torchvision主要用于构建计算机视觉模型
torchvision.datasets 一些加载数据的函数及常用的数据集接口
torchvision.models 包含常用的模型结构(含预训练模型)，AlexNet,VGG,ResNet
torchvision.transforms 常用的图像变换，裁剪、旋转
torchision.utils 其他一些有用的方法
"""
# torchvision.transform
"""
transform=transforms.ToTensor()将所有数据转换成tensor,如果不进行转换则返回的是PIL图片，transforms.ToTensor将(H*W*C)且数据位于[0,255]的PIL的图片或者数据类型为np.uint8的numpy数组转换成尺寸为(C*H*w)且数据类型为torch.float32且位于[0.0,1.0]的tensor

值得注意的是因为像素值在[0,255]，所以刚好是uint8所能表示的范围，包括transforms.ToTensor在内的一些关于图片的含少数就默认输入是uint8类型，若不是，可能不会报错，但是可能得不到想要的效果。因此，如果用像素值(0-255整数)表示图片数据，那么一律将其类型设置成uint8，避免不必要的bug，见文档中的博客
"""

# batch-images
"""
在训练数据集上训练模型，并在训练好的模型在测试数据集上评价模型的表现
mnist_train和torch.utils.data.Dataset的子类，因此可以将其传入torch.utils.data.DataLoader来创建一个读取小批量数据样本的DataLoader实例

在实践中，数据读取经常是训练的性能瓶颈，特别当模型叫简单或者计算硬件性能较高时。DataLoader中一个很方面的功能是允许使用多进程来加速数据读。
(貌似在Windos中不能使用多进程)
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

# 下载数据集
mnist_train = tv.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=True, download=True,
                                       transform=transforms.ToTensor())
mnist_test = tv.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=False, download=True,
                                      transform=transforms.ToTensor())


# 由torchvision.datasets.FashionMNIST创建的数据集是torch.utils.data.Dataset的子类，因此可以用len()来获取该数据集的大小，还可以用下表来获取具体的一个样本。训练集中和测试集中的每个类别的图像分别有60000和10000
# print(type(mnist_train)) #数据集的类别
# print(len(mnist_train),len(mnist_test)) #数据集的数量

# # 访问一个样本
# feature, label = mnist_train[0]  # 数据集返回的是(feature,label)，这里不是一个batch,而是单一图像样本,
# 值得注意的是mnist_train[0]返回的是一个元祖，(tensor(features),int(label_)
# print(feature.shape, label)  # channel*height*width,可以发现一个样本返回的label是一个数
# # 数据集单一样本返回的feature，因为使用了transforms.ToTensor()，所以每个像素的数值为[0.0,1.0]的32浮点数。

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


# # 显示训练集中的前10个样本的图像内容和文本标签
# X, y = [], []  # 生成一个空列表
# for i in range(10):
#     X.append(mnist_train[i][0])  # 手动将10个图像组成一个batch,在同一行进行显示,mnisst_train[i][0]相当于只提取图片，将元组中的第一个元素放在列表中，最终组成一个长列表，这列表中的每一个都是一个tensor
#     y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))  # 调用get_fashion_mnist_labels()函数，和show_fashion_mnist()函数

# 读取小批量
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不同额外的进程来加速读取数据，这里表示windows不能用多进程读取数据
else:
    num_workers = 4

# 加载数据，创建batch迭代
train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取一遍训练数据需要的时间
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
