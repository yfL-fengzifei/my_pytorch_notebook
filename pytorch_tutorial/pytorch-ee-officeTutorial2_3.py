# learning scheme
"""
学习如何获取数据、选择模型并评估模型的参数，以便对新数据给出良好的预测
"""
# 注意注释中一行太长也会爆出编码的错误
'''
一般来说，给定输入数据和相应的期望输出grond truth 以及权重的初始值，模型输入数据（前向传播）,然后通过把结果输出与ground truth进行比较来评估误差，为了优化模型的参数，其权重(即单位权重变化引起的误差变化，也即误差相
对于参数的梯度)通过使用对复合函数求导的链式法则进行计算(反向传播)，然后，权重的值沿导致误差减小的方向更新，不断重复过程直到在新数据上的评估误差降至可接受的水平下。
'''

# a sample thermometer
import torch

# 认为定义输入值及其对应真值
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# using linear model to fit the data
# 定义模型
# t_u为输入，w为权重，b为偏置
def model(t_u, w, b):
    return w * t_u + b  # 返回模型的计算结果


# 注意pytorch标量表示零维张量，并且乘机运算将使用广播来返回张量

# 定义误差平方损失函数
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2  # 计算模型的输出与模型标签之间的区别
    return squared_diffs.mean()  # 返回标量损失


# 最后需要通过对所得张量中的所有元素平均来得到标量损失函数，最后得到的是均方差
# 模型初始化
# 这里给出一个简单的前向传播的例子
# w = torch.ones(1)
# b = torch.ones(1)
# t_p = model(t_u, w, b)  # 得到模型的输出
# # 计算损失函数
# loss = loss_fn(t_p, t_c)  # 模型输出与标签之间的关系，返回的是标量损失
# print(loss)

# 手动计算变化率
# delta = 0.1
# loss_rate_of_change_w = \
#     (loss_fn(model(t_u, w + delta, b), t_c) - loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)
# # 如果变化为负，则需要增加w以使损失减少，如果变化为正，则需要减少w
# # 但是减少多少w呢，对w施加于损失的变化率成正比的变化是一个好主意，尤其是在损失具有多个参数的情况下：将变化应用于对损失有重大变化对的参数。通常缓慢的更改参数也是一个明智的选择
# learning_rate = 1e-2
# w = w - learning_rate * loss_rate_of_change_w


# 在实际操作的时候，对于具有两个或多个参数的模型中，应该计算损失相对于每个参数的导数，并将它们放在导数向量中，即梯度
# 为什么要计算梯度，实际上就是因为要在参数空间中搜索，参数空间在邻域内的变化，会对损失函数有何影响。但是一般来讲对于多参数模型，往往不知道需要变化的邻域应该多大。因此对参数空间进行无限小邻域内搜索可以准确的分析需要改变参数的方向，最终引出了梯度的概念
# 要计算损失函数相对于参数的倒数，可以应用链式法则，
# 即链式法则可以是：(计算损失函数县相对于其输入（即模型的输出)的导数）乘以(模型相对于参数的导数),即(loss_fn0/d(w) = [d(loss_fn)/d(t_p)]*[d(t_p)/dw]


# 损失相对于模型输出的导数
def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c)
    return dsq_diffs


# 模型相对于参数w的导数
def dmodel_dw(t_u, w, b):
    return t_u


# 模型相对与偏置的导数
def dmodel_db(t_u, w, b):
    return 1.0


# 将上述这些函数放在一起，返回损失相对于w和b的梯度的函数
def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dw = dloss_fn(t_p, t_c) * dmodel_dw(t_u, w, b)
    # (模型输出与真值标签，然后计算损失函数相对于模型输出的导数)*(模型的输入与权重和偏置，然后计算模型相对于权重的导数)，
    # 最终得到损失相对于权重参数的导数，dsq_differs*tu,dsq_differs为一个经过相减后tensor，t_u然后是一个与前面相同的维度的tensor，
    # 两相同维度的tensor相乘，按照python的内部机制，tensor*tensor是按元素相乘
    dloss_db = dloss_fn(t_p, t_c) * dmodel_db(t_u, w, b)
    # 结果是dsq_differs*1.0,相当于运用了广播机制了
    return torch.stack([dloss_dw.mean(), dloss_db.mean()])
    # stack是堆叠的意思，首先将各自损失loss相对于模型权重和偏置的导数，变成标量损失tensor，然后将这两个堆叠成向量，


# training in loop
"""
从参数的暂定值开始，可以迭代的对其应用更新一进行固定次数的迭代或直到参数停止改变位置。其中可以使用多个停止条件

历元 epoch
在所有训练样本上的一次参数更新迭代称为一个epoch
"""


# 定义循环，更新参数
def training_loop(n_epochs, learnig_rate, params, t_u, t_c, print_params=True,
                  verbose=1):  # print_params默认的参数True,verbose默认参数为1
    for epoch in range(1, n_epochs + 1):
        w, b = params  # 将所有的权重和偏置当做是参数，这里应该传入用户定义的初始化参数
        t_p = model(t_u, w, b)  # 前向传播
        loss = loss_fn(t_p, t_c)  # 计算损失函数
        grad = grad_fn(t_u, t_c, t_p, w, b)  # 多所有参数计算一次梯度，即计算损失函数相对于参数的梯度，返回的是由所有参数梯度组成的梯度向量

        params = params - learnig_rate * grad  # 对所有参数以相同的学习率更新参数

        if epoch % verbose == 0:
            print('Epoch %d, Loss%f' % (epoch, float(loss)))

            if print_params:
                print('\tparams：', params)
                print('\tgrad:', grad)
    return params


# 调用循环，开始进行训练
params = training_loop(n_epochs=5000, learnig_rate=1e-2, params=torch.tensor([1.0, 0.0]), t_u=t_u * 0.1, t_c=t_c)
print(params)  # 注意因为一直是在Tensor上进行计算的，因此输出的也是Tensor
# 改进方法，对输入数据进行规范化，调整学习率

import matplotlib.pyplot as plt

t_p = model(t_u * 0.1, *params)  # 得出最优的拟合模型
fig = plt.figure(dpi=600)
plt.xlabel = 'x'
plt.ylabel = 'y'
# 绘图的时候利用的是numpy数据
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.show() #show的时候很大