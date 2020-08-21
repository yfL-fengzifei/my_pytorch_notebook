# neuron and activation
"""
in fact,神经元不过是输入的线性变换，然后再通过一个固定的非线性函数(激活函数)
Note that the non-linear function is called activation, not the neuron
neuron include (the learned(learned parameters) and the activation)

没看完...
"""

# torch.nn Module
"""
torch.nn subModule 包含创建各种神经网络体系结构所需的构建块，这些构建块在pytorch称为module，在其他框架中称为layer层
这些构建块(module)(层)都是从基类nn.Module继承而来的python类，构建块(module)(层)可以具有一个或多个参数实例作为属性，这些参数就是在训练过程中需要优化的张量
(也就是说，定义一个继承自nn.Module的类(网络)，然后就可以定义一些具有多个参数的module(构建块)(层)，这些构建块(层)可以具有多个参数，也就是具有多个可学习的参数，这些参数的类型是张量，需要利用反向传播进行优化)

小朋友你是否有很多问号...？？？...
模块(还可以具有一个或多个子模块，并且也可以追踪其参数，什么意思？？？
注：子模块必须是顶级属性（top-level attributes），而不能包含在list或dict实例中！否则，优化器将无法找到子模块（及其参数）。(我猜这里的子模块指的是，从nn.Module中继承的各种层(构建块))。
注：对于需要子模块列表或字典的情况，PyTorch提供有nn.ModuleList和nn.ModuleDict，设么意思？？？

(我的理解是，self.conv1=nn.sequential(nn.conv2d(),nn.relu(),nn.conv3d))也就是模块还可以具有一个或多个子模块(子类)的意思，并且可以追踪其参数
"""
"""
在定义的网络(继承自nn.Module),中应该定义forward函数，但是在对网络实例化后，应该直接调用网络，而不是直接调用网络内定义的forward函数
可以查看特定层的权重和偏置
certain_layer.weight
cetain_layer.bias

Note that:
nn.Module 及其子类(层)被设计为可以同时处理多个样本，为了容纳多个样本，模型希望输入的第0维为这个批次中的样本数目
nn中的任何模块都被编写成能够同时产生一个批次(即多个输入)的输出，因此鸡舍需要对10个样本运行nn.Linear,可以创建大小为B*Nin的输入张量，其中B为批次的大小，而Nin是每个样本输入特征的数量，然后再模型中同时运行

批处理的意义：
1）充分利用计算资源，特别是在GPU上，因为GPU是高度并行化的，通过提供成批的输入，可以计算分散到其他闲置的计算单元上，意味着成批的结果就像单个结果一样能够很快的返回，
2）再有，整个网络模型，将使用整个批次的统计信息，而当批次大小较大时，那些统计数据将变得更准确

损失函数
损失函数是nn.Modele中的子类，因此可以创建一个实例并将其作为函数进行调用

定义模型
method one:
certain_model=nn.Sequential()

method two:
将nn.Module子类化，要实现nn.Module的子类，至少需要定义一个forward()函数，该函数将接受模型输入并返回输出


nn.Module有些细节没看
"""

# can not run

import torch
import torch.nn as nn
import torch.optim as optim

t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

# unsqueeze()用来扩充维度
t_c = torch.tensor(t_c).unsqueeze(1)  # <1>,在第1维度上扩充1，利用unsqueeze函数就可以不利用reshape函数，此时变成的是10*1的tensor
t_u = torch.tensor(t_u).unsqueeze(1)  # <1>

linear_model = nn.Linear(1, 1)
optimizer = optim.SGD(linear_model.parameters(),
                      lr=1e-2)  # certain_module.parameters()可以获取任何模型(包括nn.Module,或子模块或子类)的参数列表，


# certain_module.parameters返回的是一个生成器
# list(certain_parameters())是什么意思...???...
# 在训练循环中调用optimizer.step(),将循环访问每个参数，并按与存储在参数grad属性中的值成比例的量对其进行更改


# 验证集的数据与标签也没有定义
def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs + 1):
        t_p_train = model(t_un_train)  # 训练集
        loss_train = loss_fn(t_p_train, t_c_train)  # 计算训练集的损失

        t_p_val = model(t_un_val)  # 测试集
        loss_val = loss_fn(t_p_val, t_c_val)  # 计算验证集的损失

        optimizer.zero_grad()
        loss_train.backward()  # 对训练集执行反向传播
        optimizer.step()  # 利用训练集对模型进行参数更新，
        # 因为optimizer传入的就是linear_model的参数，即linear_model.parameters()，因此，step更新的就是该模型的参数

        if epoch == 1 or epoch % 1000 == 0:
            print('epoch %d, training loss%.4f, valdidation loss %.4f' % (
                epoch, float(loss_train), float(loss_val)))  # 输出训练集和验证集的损失


# 这里的loss_fn不再进行手动计算，而是调用nn.Module中定义好的函数
# 调用training_loop函数执行训练
# 这里并没有进行数据切分，can not run
training_loop(n_epochs=3000, optimizer=optimizer, model=linear_model, loss_fn=nn.MSELoss(), t_u_train=t_u_train,
              t_u_val=t_u_val, t_c_train=t_u_train, t_c_val=t_c_val)
print(linear_model.weight)
print(linear_model.bias)

# 重新定义模型
# this is one of creating the model
# 利用nn.Sequential容器串联模块的简单方法
seq_model = nn.Sequential(nn.Linear(1, 13), nn.Tanh(), nn.Linear(13, 1))
# (我的理解在创建完模型后，就会自动生成可学习参数的位置)
# 查看参数形状，[param.shape for param in seq_model.parameters()] #注意这种写法
# seq_model.parameters()参数都是优化器所需的参数张量，(我的理解是，这里的参数张量都是以层组织的(seq_model.parameters()[i]，返回的是某层的参数张量))\
# 检查几个子模块组成的模型的参数时，可以方便的通过其名称识别参数，利用的是named_parameters(); (值得注意的是，每个层的名称都是该层在参数中出现的顺序)即\
# for name,param in seq_model.named_parameters():
#    print(name,param.shape)
# 另一种user定义层的方法，利用OrderdDict函数
# from collections import OrderedDict
#
# seq_model = nn.Sequential(OrderedDict(
#     [('hiddern_linear', nn.Linear(1, 8)), ('hidden_activation', nn.Tanh()), ('output_linear', nn.Linear(8, 1))]))
# 查看特定层的参数seq_model.output_linear.bias
# 定义模型后，得到优化器所需的参数张量。在调用model.backward()知乎，所有参数 都将被计算其grad，然后优化器会在调用optimizer.step()期间更新参数的值

# 重新调用training_loop,执行训练
training_loop(n_epochs=3000, optimizer=optimizer, model=seq_model, loss_fn=nn.MSELoss(), t_u_train=t_u_train,
              t_u_val=t_u_val, t_c_train=t_u_train, t_c_val=t_c_val)
print('hidden', seq_model.hidden_linear.weight.grad)

# create network
"""
there are three method of creating the same network
"""
# method one: 直接创建法
seq_model = nn.Sequential(nn.Linear(1, 11), nn.Tanh(), nn.Linear(11, 1))
# 并未上述各层添加名称
from collections import OrderedDict

namedseq_model = nn.Sequential(OrderedDict(
    [('hidden_linear', nn.Linear(1, 12)), ('hidden_activation', nn.Tanh()), ('output_linear', nn.Linear(12, 1))]))


# method two: 通过定义类来实例化法
class SubclassModel(nn.Module):
    def __init__(self):
        super.__init__()
        # super.__init__(SubclassModel,self)
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t


# 调用
subclass_model = SubclassModel()


# method three
class SubclassFunctionModel(nn.Module):
    def __init__(self):
        super.__init__()
        # super.__init__(SubclassModel,self)
        self.hidden_linear = nn.Linear(1, 14)
        # self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(14, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        # activated_t = self.hidden_activation(hidden_t)
        activated_t = torch.tanh(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t


# 调用
func_model = SubclassFunctionModel()
