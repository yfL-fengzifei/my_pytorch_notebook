# learning scheme part2
# tensor and torch.autograd
"""
tensor可以记住他们自己来自什么运算，以及，其起源的父张量，并且提供相对于输入的导数链，因此无需手动对模型求导
不管如何嵌套，只要给出前向传播表达式，pytorch都会自动提供该表达式相对于其参数的梯度

在定义tensor的时候，required_grad=True,表示，pytorch需要追踪在params上进行运算而产生的所有tensor，换句话说，任何以params为祖先的Tensor都可以访问从params到该tensor所调用的函数链，如果这些函数是可微的，如果这些函数是可微的(大多数pytorch的tensor运算都是可微的)，则导数的值会自动存储在参数tensor的grad属性中，存储在哪个tensor的grad属性中？？？
一般来说，所有pytorch的tensor都有一个初始化为空的grad属性

一般需要做的就是将required_grad设置为True,然后调用模型，计算损失值，然后对损失tensor:loss调用backward

torch Module中的optim subModule，可以在其中找到实现不同优化算法的类
查看算法
import torch.optim as optim
dir(optim)

overfiting模型过拟合：
方式过拟合的方法：
假设有足够多的数据，则应确保能够拟合训练数据的模型在数据点之间尽可能正则化。
有几种方法实现此目标：
一种是在损失函数中添加所谓的惩罚项，以使模型的行为更平稳，变得更慢(到一定程度)
另一种方法是向输入样本添加噪声，在训练数据样本之间人为创建新的数据，并迫使模型也尝试拟合他们
处上述方法外，可选择正确大小(就参数而言)的神经网络模型基于两个步骤：增大模型大小直到成功拟合数据，然后逐渐缩小直到不再过拟合
(其中一种理论就是，在拟合模型时，在训练集评估一次损失，然后再验证集上评估一次损失(但不进行参数更新))
在拟合和过拟合之间的平衡，可以通过以相同的方式对t_u和t_c进行打乱，然后生成的数据随机分为两部分从而得到训练集和验证集
"""

# review the part1 in 'pytorch-officeTutorial2_3'
import torch

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)


# 定义模型
def model(t_u, w, b):
    return w * t_u + b


# 定义损失函数
def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c) ** 2
    return squared_diffs.mean()


# # 初始化参数
# params = torch.tensor([1.0, 0.0], requires_grad=True)
#
# # 计算损失函数
# loss = loss_fn(model(t_u, *params), t_c)
# # 调用反向传播
# loss.backward()
# # 查看计算得到的梯度
# print(params.grad)  # 这里params的grad属性，包含损失loss关于parmas的每个元素的导数
"""
见文档，图解很详细
"""

"""
你可以将(包含任意数量的张量)的required_grad设置为True,以及组合任何函数。在这种情况下，pytorch会沿整个函数链(即计算图)计算损失的导数，并在这些张量(即计算图的叶节点)的grad属性中将这些导数值累积起来
注意，导数值是累积的，？？？
调用backward()函数会导致导数值在叶节点处累积，所以将其用于参数更新后，需要将梯度显式清零
重复调用backward会导致导数在叶节点处累积，因此，如果调用了backward，然后再次计算损失并再次调用backward()如在训练循环中一样，那么在每个叶节点上的梯度会被累积(求和)在前一次迭代计算出的那个叶节点上，导致梯度值得不正确，因此为防止这种情况发生，需要再每次迭代时将梯度显式清零，直接使用zero_()函数
if params.grad is not None:
    params.grad.zero_()

"""

# # 定义训练策略
# def training_loop(n_epochs, learning_rate, params, t_u, t_c):
#     for epoch in range(1, n_epochs + 1):
#         if params.grad is not None:  # 一开始创建的时候，因为没有执行model即没有执行前向传播，那么会自动生成一个grad的空属性
#             params.grad.zero_()  # 每次epoch将梯度清零
#
#         t_p = model(t_u, *params)  # 执行前向传播
#         loss = loss_fn(t_p, t_c)  # 计算损失函数
#
#         # 自动计算梯度
#         loss.backward()  # 执行反向传播，得到损失相对于参数的梯度值，并放在params.grad属性中
#
#         # 手动更新参数
#         params = (params - learning_rate * params.grad).detach().requires_grad_()  # 更新参数
#
#         if epoch % 500 == 0:
#             print('Epoch%d,Loss%f' % (epoch, float(loss)))
#     return params


'''
.detch().requires_grad_()
p1=(p0*lr*p0.grad)
其中p0用于初始化模型的随机权重，p0.grad是通过损失函数根据p0和训练数据计算出来的
现在进行第二次迭代p2=(p1*lr*p1.grad)
.detatch()将新的params张量从与其更新表达式关联的计算图中分离出来。这样，params就会丢失关于生成它的相关运算的记忆。然后，你可以调用.requires_grad_()，这是一个就地（in place）操作(注意下标“_”)
，以重新启用张量的自动求导。现在，你可以释放旧版本params所占用的内存，并且只需通过当前权重进行反向传播,???
'''
# # 开始训练
# t_un = 0.1 * t_u
# training_loop(
#     n_epochs=5000,
#     learning_rate=1e-2,
#     params=torch.tensor([1.0, 0.0], requires_grad=True),
#     t_u=t_un,
#     t_c=t_c)


# 优化器
"""
优化器的目的就是要更新参数 optimizer=optim.certain_optimMethod(para1,para2...)
最常用的是SGD随机梯度下降法
momentum优化算法是基于SGD的，只需将momentum参数传递给SGD算法，就能实现momentum算法
t_p=model(t_u,*params)
loss=loss_fn(t_p,t_c)
loss.backward()
optimizer.step()

调用step后params值就会更新，无需亲自更新它，调用step发生的事情是：优化器通过将params减去learning_rate与grad的乘积来更新params,这与之前手动编写的更新过程完全相同
"""


# 利用pytorch中有的torch.optim来重新定义
def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print('epoch%d,loss%f' % (epoch, float(loss)))
    return params


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = torch.optim.SGD([params], lr=learning_rate)

# training_loop(n_epochs=5000,optimizer=optimizer,params=params,t_c=t_u*0.1,t_c=t_c) #???

# 对张量的元素进行打算等价于重新排列其索引，使用randperm函数
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)
