# easy implementation of linear regression

"""
概览
torch.utils.data 该模块提供了有关数据处理的工具
torch.nn 该模块定义了大量神经网络的层
torch.nn.init 该模块定义了各种初始化方法
torch.optim 该模块提供了很多常用的优化算法

定义网络模型
troch.nn就是利用autograd来定义模型，torch.nn的核心数据结构是Module，这一个抽象的概念，即可以表示神经网络的某个层(layer)，也可以表示包含层的神经网络。在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层，一个nn.Module实例应该包含一些层以及返回输出的前向传播方法forward

模型初始化
from torch.nn import init 该Module提供了多种初始化的方法，这里init是initializer

优化器及其优化方法
optimizer=optim.SGD()为不同的子网络设置不同的学习率,利用的是字典的数据结构
有时候，不想让学习率固定成一个常数，则调整学习率的方法为主要有两种：
方法1：就该optimizer.param_groups中对应的学习率，（不推荐）
方法2：新建一个优化器，由于optimizer十分轻量级，构建开销很小，因此可以创建一个optimizer，但是后者对于使用动量的优化器Adam,会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况
"""

import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim
import numpy as np
from collections import OrderedDict

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = torch.tensor([2, -3.4]).unsqueeze(1)  # 权重
true_b = torch.tensor([4.2])  # 偏置
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)),
                        dtype=torch.float)  # 生成标准正态分布，features.size([1000,2])
labels = torch.mm(features, true_w) + true_b  # 加法遵循boardcast
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()))

# 读取数据
batch_size = 10  # 定义一个批量的大小
dataset = Data.TensorDataset(features, labels)  # 将训练数据的特征和标签组合,组合之后，每个dataset[i]返回的就是(data,label)形式
dataiter = Data.DataLoader(dataset, batch_size,
                           shuffle=True)  # dataset返回的是一个可利用下标进行索引的对象，DataLoader返回的是一个可迭代的对象，并根据参数将数据组合成一个batch


# 打印第一个batch查看结果
# for data, label in dataiter:
#     print(data,label)
#     break

# 定义网络模型,method one
class LinearNet(nn.Module):
    def __init__(self, n_features):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


# # 定义网络模型,method two
# # 利用certain_net=nn.Sequential()
# net_two = nn.Sequential(
#     nn.Linear(num_inputs, 1)
# )
# # 定义网络模型,method three
# # 利用，certain_net.add_model()
# net_three = nn.Sequential()
# net_three.add_module('linear', nn.Linear(num_inputs, 1))
# # 定义网络模型，method four
#
# net_four = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))

# 实例化网络
net = LinearNet(num_inputs)
print('net one: ', net)
# print('net two', net_two)
# print('net three', net_three)
# print('net four', net_four)
# print(net_four[0])  # 因为net_two、net_three、net_four是同时Sequential(或ModuleList)实例化得来的
print(net.linear)
# 值得注意的是，print(net[0])只适用于net_tow[0],net_three[0]，net_four[0],因为net[0]这样根据下表访问子模块的写法只有当net时个ModuleList或是Sequential实例时才可以，后面章节还会提到
# 而利用class来实例化对象时,只能是net.linear
# 查看网络参数，查看网络所有的可学习参数
# for param in net.parameters():
#     print(param)

# 模型初始化
init.normal_(net.linear.weight, mean=0, std=0.01)  # 如果是net_two\three\four则直接写成net_two[0].weights
# 利用init.normal()函数讲权重参数的每个元素初始化为随机采样与均值为0，标准差为0.01的正态分布,
init.constant_(net.linear.bias, val=0)  # 或者直接使用net[0].bias.data.fill_(0)

# 定义损失函数
loss_fn = nn.MSELoss()

# 定义优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
# #为不同的子网络设置不同的学习率
# #如果对某个参数不指定学习率，就使用最外层的默认学习率
# optimizer=optim.SGD([{'parmas':net.subnet1.parameters()},{'parmas':net.subnet2.parameters(),'lr':0.01}],lr=0.03)
# #net.subnet1.parameters的学习率使用的是最外层的学习率lr=0.03
# (感觉上述模型只有一个subnet，不知道怎么使用)
# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1  # 学习率是之前的0.1倍，使用的是字典的索引方式

# 定义训练方法
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    for data, label in dataiter:  # 这是batch操作，值得注意的是data的维度是batch*sample_features 因为定义的x只是一个两特征的数据点
        outputs = net(data)  # forward
        loss = loss_fn(outputs, label.view(-1, 1))  # 计算损失函数，#其实不用调整，但是为了以防万一
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        loss.backward()  # backward
        optimizer.step()  # 更新参数
    print('epoch %d: ,loss: %f' % (epoch, loss.item()))

#比较一下网络层的真实参数
#dense=net.linear
print(true_w,net.linear.weight)
print(true_b,net.linear.bias)