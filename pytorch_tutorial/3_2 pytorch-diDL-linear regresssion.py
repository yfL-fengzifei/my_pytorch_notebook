import torch
from matplotlib import pyplot as plt
import numpy as np
import random

# 人工生成一个训练样本
# 样本特征数为2
# 假定真实权重为[2,-3.4]_transpose, 偏差b=4.2
# 同时加入一个随机噪声项，服从均值为0，标准差为0.01的正太分布，噪声代表数据集中无意义的干扰

# 样本特征数
num_inputs = 2

# 样本数
num_examples = 1000

# 真值权重
true_w = [2, -3.4]
true_w = torch.tensor(true_w)  # 转换成tensor
true_w = torch.reshape(true_w, (2, 1))  # 转换维度,那么下面就可以不用.t()
# 直接定义
# true_w=torch.tensor([[2,-3.4]]) #则下面直接写成true_w.t()

# 真值偏差
true_b = 4.2

# 生成1000*2的样本
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
# 等价于features=torch.randn([num_examples,num_inputs],dtype=torch.float32)
print(features)

# 定义真值标签,1000*1
# 文档中利用公式相加
# labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
# print(labels)
labels = (torch.mm(features, true_w)) + true_b
print(labels)

# 加上随机干扰项
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
# for i in range(1000):
#     print(features[i],labels[i])

# 画图，第二特征与标签之间的关系
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()


# plt.close()


# 读取数据
# 在训练模型的时候，需要遍历数据集并不断读取小批量数据样本，
# 因此可以定义一个函数，用来返回batch_size大小各随机样本的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # features的长度
    indices = list(range(num_examples))  # 生成[0,999]的列表
    random.shuffle(indices)  # 将列表中的数字随机排序,执行的本地操作，返回的还是indices

    # 下面的indices[i:pass]相当于从i开始索引
    for i in range(0, num_examples, batch_size):  # range，开始\结束\步长
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch_size

        # torch.index_select   certain_tensor.index_select(0,j)
        # 因为j是一个tensor,在index_select中表示的tensor对象的索引号，0表示按行索引，1表示案列索引
        # 所以下面表示的是生成一个生成器，返回一个迭代器，对features和labels按行进行索引50个
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10  # 定义批为10个样本和标签
# 调用函数
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 模型初始化
# 将权重初始化为0，标准差为0.01的正态分布随机数，偏差初始化为0
# 因为之后的模型训练中，需要对这些参数求梯度来迭代参数的值，因此requires_grad=True
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32,
                 requires_grad=True)  # 这里只生成一组，因为是初始化，后面需要迭代进行更新修改
b = torch.tensor([0], dtype=torch.float32, requires_grad=True)


# 也可以写成 b=torch.zeros(1,dtype=torch.float32)
# 上述也可以写成下面的样子
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
# b = torch.tensor([0], dtype=torch.float32)
# w.requires_grad_(requires_grad=True)
# b.requires_grad_(True)  # 缺省写应该也行


# 定义线性回归模型
def linreg(X, w, b):
    return torch.mm(X, w) + b  # torch.mm表示矩阵乘法


# 定义损失函数
def suqare_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
# 下面自动求梯度模块计算得到梯度是一个批量样本的梯度和，将其除以批量大小来得到平均值
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 训练模型


pass
