#nn
import torch
import torch.nn as nn


"""
=================================================================================================
torch.nn
"""
"""
nn.Parameter 张量子类，表示可学习参数
nn.Module 所有网络层基类，管理网络属性
nn.functional 函数具体实现
nn.init 参数初始化方法
"""

"""
=================================================================================================
nn.Nodule
"""
"""
torch.nn的核心数据结构是Module,其是一个抽象的概念，即可以表示某个层，也可以表示包含很多层的神经网络
nn.Module:
parameters:存储管理nn.Parameters类
modules:存储管理nn.Module类
# buffers:存储管理缓冲属性，如BN层中的running_mean
# ***_hooks：存储管理钩子函数
一般是net_modules_parameters
每个module都有8个字典管理他的属性

#卷积
1D，2D，3D
transposeconv 逆卷积

#池化
avgpool 平均池化
maxpool 最大池化
adaptiveavgpool 自适应池化
maxunpool2d

#BN层
1D、2D、3D

#dropout
1D、2D、3D

#Relu
ReLU函数有个inplace参数，如果设为True，它会把输出直接覆盖到输入中，这样可以节省内存/显存。之所以可以覆盖是因为在计算ReLU的反向传播时，只需根据输出就能够推算出反向传播的梯度。但是只有少数的autograd操作支持inplace操作（如tensor.sigmoid_()），除非你明确地知道自己在做什么，否则一般不要使用inplace操作。


"""


"""
=================================================================================================
查看参数
"""
"""
net=Net()
params=list(net.parameters())

named_parameters() 或parameters()返回迭代器
for name,parameter in net.named_parameters():
    pass
for parameter in net.parameters()

对于 子Module 中的parameter，会其名字之前加上当前Module的名字。如对于`self.sub_module = SubModel()`，SubModel中有个parameter的名字叫做param_name，
那么二者拼接而成的parameter name 就是`sub_module.param_name`


list(layer.parameters())

查看参数的长度
print(len(params))

"""


"""
=================================================================================================
前向传播-Squential
"""
"""
Squential

net1=nn.Squential()
net.add_module('conv',nn.Conv2d(3,3,3))
net.add_module('bn',nn.BatchMormal2d(3))
net.add_module('name',nn.Relu())

net2=nn.Squential(
    nn.Conv2d(3,3,3),
    nn.BatchNormal(3),
    nn.Relu()
    )
    
from collections import OrderedDict
net=nn.Squential(OrderedDict([
    ('conv',nn.Conv2d(3,3,3)),
    ('bn',nn.BatchNorm2d(3)),
    ('relu',nn.Relu())    
    ]))

#查看参数
#可以根据名字和序号取出子module
net1[0] 等价于net1.conv

net2[0]

net3.conv 等价于net3[0]
"""


"""
=================================================================================================
前向传播-ModuleList
"""
"""
ModuleList

modulelist=nn.ModuleList([nn.l=Linear(3,4),nn.ReLU(),nn.Linear(4,2)])

值得注意是modulelist并没有实现forward方法；所以不能直接作用于输入

#查看参数
modulelist[0]

"""


"""
=================================================================================================
nn.functional 和nn.Module
"""
"""
如果模型有可学习的参数，最好用nn.Module,否则既可以用nn.functional也可以使用nn.Mddule，二者在性能上没有太大的差异
如果对于可学习参数的魔窟开，利用functional来代替时，实现起来比较繁琐，需要手动定义参数parameter,（见'python-book'）
值得注意的是一般使用nn.Dropout而不是nn.functional.dropout,因为dropout在训练和测试两个阶段的行为有所差别，使用nn.Module对象能够通过model.eval操作加以区别
"""



"""
=================================================================================================
损失函数
"""
"""
见官方文档
nn.certain_loss_function()

一般计算出来的loss是一个标量，可以用loss.item()来获取loss的值
"""


"""
=================================================================================================
反向传播
"""
"""
output = net(input)
# 执行反向传播，注意这里是out是torch.size([1,10]),并不是一个标量，也不是一个损失，所以在执行反向传播的时候，要传入与out大小相同的tensor
# output.backward(torch.ones(1,10))

这点没懂...
"""


"""
=================================================================================================
查看梯度
"""
"""
print(net.conv1.bias.grad)
print(net.certain_Sequential.certain_layer.bias.grad)

线性传播，梯度消失，爆炸见'bilibili-pytorch框架'
"""


"""
=================================================================================================
优化器
"""
"""
import torch.optim as optim

#学习率设置
为不同的子网络设置不同的学习率，在finetune中经常用到
如果对某个参数不指定学习率，那么久使用最外层的学习率
optimizer=optim.SGD([
    {'params':net.features.parameters()},
    {'params':net.classifier.parameters(),'lr':1e-2},
    ],lr=1e-5
)

#梯度清零
optimizer.zero_grad() 等价于net.zero_grad()

#参数组的概念
optimizer有一个参数组的性质
optimizer.param_groups
只有在nn.Sequential()和nn.ModuleList的情况下才会可能有多个组
"""

"""
=================================================================================================
初始化
"""
"""
良好的初始化能够让模型更快的收敛，并达到更高的水平，
pytorch,都采用了较合理的初始化策略，一般不需要考虑，
初始化策略用nn.init函数提供
import torch.nn import init
init.xavier_normal_(linear.weight)
"""



"""
=================================================================================================
模型保存与加载
"""
"""
法一：
# class pass
# training_loop pass
# 已经训练好的net
torch.save(net,'net.pkl')
net2=torch.load('net.pkl')

法二：
# 只保留计算图中节点的参数，效果更快
torch.save(net.state_dict(),'net_param.pkl')
# 首先先建立与训练时一模一样的网络
class pass
net3=Net()
net3.load_state_dict(torch.load('net_param.pkl'))
"""

"""
=================================================================================================
计算准确率
"""
"""
# 显示在整个测试集上的结果
correct = 0  # 预测正确的图片数
total = 0  # 总共的图片数

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
"""

"""
=================================================================================================
深入理解...没看
"""


# from PIL import Image
# import matplotlib.pyplot as plt
# lena=Image.open('./lena.png')
# plt.imshow(lena)
# plt.show()
# # lena.show()
# print('pass')


# net1 = nn.Sequential()
# net1.add_module('conv', nn.Conv2d(3, 3, 3))
# net1.add_module('bn', nn.BatchNorm2d(3))
# net1.add_module('name', nn.ReLU())
#
# net2 = nn.Sequential(
#     nn.Conv2d(3, 3, 3),
#     nn.BatchNorm2d(3),
#     nn.ReLU()
# )
#
# from collections import OrderedDict
#
# net3 = nn.Sequential(OrderedDict([
#     ('conv', nn.Conv2d(3, 3, 3)),
#     ('bn', nn.BatchNorm2d(3)),
#     ('relu', nn.ReLU())
# ]))

# modulelist=nn.ModuleList([nn.Linear(3,4),nn.ReLU(),nn.Linear(4,2)])


import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier=nn.Sequential(
            nn.Linear(16*5*5,20,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )

    def forward(self,x):
        x=self.features(x)
        x=x.view(-1,16*5*5)
        x=self.classifier(x)
        return x

net=Net()


# optimizer=optim.SGD(params=net.parameters(),lr=1)
# optimizer.zero_grad()
#
# input=torch.randn(1,3,32,32)
# output=net(input)
# output.backward(output) #？？？？
# optimizer.step()

optimizer=optim.SGD([
    {'params':net.features.parameters()},
    {'params':net.classifier.parameters(),'lr':1e-2},
    ],lr=1e-5
)
print('pass')

