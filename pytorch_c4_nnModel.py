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
https://zhuanlan.zhihu.com/p/88712978(好多没看)
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

.named_parameters() 或.parameters()返回迭代器
for name,parameter in net.named_parameters():
    pass
for parameter in net.parameters()
值得注意的是，模型的可学习参数包含在model.parameters()


对于 子Module 中的parameter，会其名字之前加上当前Module的名字。如对于`self.sub_module = SubModel()`，SubModel中有个parameter的名字叫做param_name，
那么二者拼接而成的parameter name 就是`sub_module.param_name`

list(layer.parameters())

#查看参数的长度
print(len(params))

对于参数parameters,可以使用for param in module.parameters()来遍历网络模型中的参数，因为该函数返回的是一个迭代器iterator。在使用优化算法的时候就是讲model.parameters()传给Optimizer，与之类似的还有函数buffers、children、modules
for name,param in model.named_parameters()
for name,buf in model.named_buffers()
for name,module in model.named_chlidren()
for name,modle in model.named_modules()


#使用state_dict来访问所有的参数
state_dict的作用是返回一个包含module的所有state的dictionary，而这个字典的keys对应的是parameter和buffer的名字names。(该函数的源码部分有一个循环可以递归遍历Module中所有的SubModule)
state_dict是一个简化版的python字典对象，其将每一层映射为对应的参数tensor。值得注意的是，只有包含可学习参数的网络层(卷积层、线性层等)或者是bn,会出现在state_dict中。
优化器也有一个state_dict,其包含优化器的状态信息，以及使用的超参数
因为state_dict是一个python字典，可以轻松的被保存、更新、迭代、恢复、并被pytorch模型和优化器添加大量的模块。

import torch
import torch.nn as nn
#net=nn.certain_module()
net=nn.Linear(2,2)
#查看该module中的所有state,返回的是一个字典OrderedDict()
print(net.state_dict())
#查看键值
print(net.state_dict().keys())

net2=nn.Sequential(nn.Linear(2,2),nn.Linear(2,2))
#查看该module的所有state
print(net2.state_dict())
#查看键值
print(net2.state_dict().keys())
#循环查看
for name in model.state_dict():
    print(name,model.state_dict()[name],model.state_dict()[name].size())
"""

"""
=================================================================================================
前向传播: __init__

"""
"""
def __init__(self,XXX):
一种是自定义将参数封装成parameter
parameter是一种特殊的tensor,但其默认是需要求导的，即默认requires_grad=Ture

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
net3=nn.Squential(OrderedDict([
    ('conv',nn.Conv2d(3,3,3)),
    ('bn',nn.BatchNorm2d(3)),
    ('relu',nn.Relu())    
    ]))

#查看参数
#可以根据名字和序号取出子module
net1[0] 等价于net1.conv

net2[0]

net3.conv 等价于net3[0]
net[i].weights
net[i].bias

值得注意的是，只有是利用nn.Sequential()或nn.ModuleList()声明的网络，才可以利用net[i]索引的方式访问具体的层或子网络，而利用class声明的网络，只能利用net.certain_layer的方式进行访问,即net.certian_layer.weights、net.certain_layer.bias

使用state_dict来访问所有的参数
state_dict的作用是返回一个包含module的所有state的dictionary(OrderDict)，而这个字典的keys对应的是parameter和buffer的名字names。(该函数的源码部分有一个循环可以递归遍历Module中所有的SubModule)
import torch
import torch.nn as nn
#net=nn.certain_module()
net=nn.Linear(2,2)
#查看该module中的所有state,返回的是一个字典OrderedDict()
print(net.state_dict())
#查看键值
print(net.state_dict().keys())

net2=nn.Sequential(nn.Linear(2,2),nn.Linear(2,2))
#查看该module的所有state
print(net2.state_dict())
#查看键值
print(net2.state_dict().keys())
#循环查看
for name in model.state_dict():
    print(name,model.state_dict()[name],model.state_dict()[name].size())
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
moduleList[i].weights
moduleList[i].bias

值得注意的是，只有是利用nn.Sequential()或nn.ModuleList()声明的网络，才可以利用net[i]索引的方式访问具体的层或子网络，而利用class声明的网络，只能利用net.certain_layer的方式进行访问
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
softmax
"""
"""
本质上，softmax是一个全连接层
对于分类问题，softmax回归输入值为pi,当做预测类别i的置信度，并将值伟大的输出所对应的类作为预测输出，即输出为argmax(pi)
但是直接使用输出层的输出有两个问题，一方面，由于输出值得范围不确定，因此很难直观上判断这些值得意义(就是什么样的值叫做高)，另一方面，由于真实标签是离散值，这些离散值与不确定范围的输出值之前的误差难以衡量

针对上述的问题，softmax运算符就可以有效的解决上面的问题，即softmax函数将输出值转变成：值为正，且，和为1的概率分布,hat(y1),hat(y2),hat(y3)=softmax(O1,O2,O3),这里不列出公式
"""


"""
=================================================================================================
batch
"""
"""
batch_size=n
features=d # features in one sample, the inputs of network's layer
num_labels=q # the number of ground truth classess
W| W.size()=[d,q] # weights of the layer
b| b.size()=[1,q] #bias of the layer
X| X.size()=[n,d] #features of the batch samples 

O=X*W+b #实际上更常用的是O=W*X #W.size()=[q,d], X.size()=[d,n]
hat(y)=softmax(O)

"""


"""
=================================================================================================
损失函数
"""
"""
见官方文档
nn.certain_loss_function()

一般计算出来的loss是一个标量，可以用loss.item()来获取loss的值

cross entropy函数可以衡量两个概率分布差异。交叉熵只关心正确类别的预测概率，只要其值足够大，就可以确保分类结果的正确
需要注意的损失对于batch的交叉熵函数与单一样本的交叉熵函数有所不同，这里没有列出函数，需要进一步查看
最小化交叉熵函数等价于最大化训练数据集所有标签类别的联合预测概率

nn.CrossEntropyLoss() #这个损失函数包含了softmax运算和交叉熵损失计算
#内置交叉熵函数的运行机制需要再看,'3_7'在训练的时候有一个loss.sum()的操作，没看懂
"""


"""
=================================================================================================
反向传播
"""
"""
output = net(input)
# 执行反向传播，注意这里是out是torch.size([1,10]),并不是一个标量，也不是一个损失，所以在执行反向传播的时候，要传入与out大小相同的tensor
# 注意在y.backword()时，如果y是标量，则不需要为backword()传入任何参数，否则，需要传入一个与y同形的tensor
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
#调整学习率
for param_group in optimizer.param_groups:
    param_group['lr']*=0.1

#梯度清零
注意grad在反向传播过程中是累加的，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需要把梯度清零
optimizer.zero_grad() 等价于net.zero_grad()
#梯度清零的判断的例子，见'3_7 pytorch-diDL-softmax-regression2.py',...???...没看懂


#参数组的概念
optimizer有一个参数组的性质
optimizer.param_groups；只有在nn.Sequential()和nn.ModuleList的情况下才会可能有多个组,在class中也可以使用sequential 和ModuleList
方法1：就该optimizer.param_groups中对应的学习率，（不推荐）
方法2：新建一个优化器，由于optimizer十分轻量级，构建开销很小，因此可以创建一个optimizer，但是后者对于使用动量的优化器Adam,会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况，...???...没懂

#优化器的保存
优化器也有一个state_dict,其包含优化器的状态信息，以及使用的超参数
optimizer=optim.SGD()
for var_name in optimizer.state_dict()
    print(var_name,'\t',optimizer.state_dict()[var_name])
#这里还有一个参数组的概念
optimizer.state_dict()['param_groups']
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

init.normal_(net.certain_layer.learnalble_param,mean=,std=) 将可学习参数的每个元素初始化为随机采样与均值为=，标准差=，的正态分布
init.constant_(net.certain_layer.learnable_param,val=) 将参数设为常数，或直接使用net.certain_layer.weights(or bias).data.fill_(certain_value)
"""


"""
=================================================================================================
超参数
"""
"""
人为设定得参数为超参数，而不是由模型训练得到的。
一般调参调的就是这些超参数，在少数情况下可以通过模型训练学出来。
"""


"""
=================================================================================================
模型保存与加载
"""
"""
模型保存的格式是，.pkl,.pt,.pth 这几种格式都没有什么区别

法一：
# class pass
# training_loop pass
# 已经训练好的net
torch.save(net,'net.pkl')
net2=torch.load('net.pkl')
#尽量不要使用这种方式，因为这种方式并没有保存模型本身，而是保存的是路径


法二：
# 只保留计算图中节点的参数，效果更快
#state_dict()的作用见上述’查看参数一节‘
torch.save(net.state_dict(),'net_param.pkl')
# 首先先建立与训练时一模一样的网络
class pass
net3=Net()
net3.load_state_dict(torch.load('net_param.pkl')) #load_state_dict的作用于state_dict的作用相反，将parameter和buffer加载到module和SubModule中
#必须是先torch.load()，然后在load_state_dict，直接使用load_state_dict(path)不行

#加载预训练模型
import torchvision.models as models
pretrained_network=models.certian_network(pretrained=True)
#预训练的模型的参数
pretrained_dict=pretrained_network.state_dict()

#从网络上下载
import torch.utils.model_zoo as model_zoo
state_dict=model.zoo.load_ulr() #从给定URL处下载torch序列化对象，如果要下载的文件是压缩包，将自动解压缩

#加载部分预训练的模型
#这里只是给个例子
my_net=Net()
pretrained_network=models.certain_network(pretrained=True)
pretrained_dict=pretrianed_network.state_dict()
my_net_dict=my_net.state.dict()
#将预训练模型中的参数不属于自定义模型结构的键值删除
pretrained_dict={k:v for k,v in pretrained_dict() if k in my_net_dict} #这里完全是利用键名进行约束的
#更新自定的网络参数
my_net_dict.update(pretrained_dict)
#加载真正的参数
my_net._dictload_state_dict(my_net_dict)


#保存和加载优化器
model=Net()
optimizer=optim.SGD()
1、假设在某个epoch,要保存模型参数，优化器参数以及epoch
先建立一个字典，保存三个参数
state={'net':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch}
torch.save(state,dir) #dir='path/model_save.pth'

2、当想恢复某一阶段的训练(或者进行测试时)，那么就可以读取之前保存的网络模型参数等
checkpint=torch.load(dir)
model.load_state_dict(checkponit['net'])
optimizer.load_state_dict(checkpoint['optimizer'])
state_epoch=checkpoint['epoch']+1

state_dict是一个简化版的python字典对象，其将每一层映射为对应的参数tensor。值得注意的是，只有包含可学习参数的网络层(卷积层、线性层等)或者是bn,会出现在state_dict中。
优化器也有一个state_dict,其包含优化器的状态信息，以及使用的超参数

3、测试
值得注意的是，在进行推理之前，必须调用model.eval()来设置dropout和bn层

#在推理和恢复训练的时候进行节点保存和加载
torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict,
            'loss':loss},path)
model=certain_model()
optimizer=certain_optimizer()
checkpoint=torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch=checkpoint['epoch']
loss=checkpoint['loss']

model.eval() 或者是model.train()
在保存模型节点的时候，不仅仅是保存模型的state_dict,保存优化器的state_dict也是很重要的，因为其包含着缓存和参数。其他要保存的可能包括epoch,loss或其他等等
将这些要保存的内容组织成字典，一般将节点保存为.tar的格式(官方说的是.tar而没说是.pth.tar)
当要加载的时候，先使用torch.load,然后使用.load_state_dict

#保存和加载多模型没看
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
深入理解nn.Moduel-后面有点没看完
"""
"""
nn.Module基类的构造函数
def __init__(self):
    self._parameters=
    self._module=
    self._buffers=
    self._back_hooks=
    self._forward_hooks=
    self.training

_parameters: 字典，保存用户直接定义的参数，self.param=nn.Parameter(torch.tensor([1,2,3]))
值得注意的是，只有自己定义的parameter会被加入到_parameters字典中，而self.fc=nn.Linear(3,4)则不会加入到_parameters字典中
_modules:子module，通过self.submodule=nn.Linear(3,4)指定的子module会保存于此
其他参数没看...???...
training BN和dropout在训练阶段和测试阶段中采取的策略不同，通过判断training值来决定前向传播策略

值得注意的是：
1.net=Net()打印出来的只有_modules中定义的层，相当于certain_net._modules()
2.而net._parameters 打印出来的之后只有自己定义的parameters，也就是在_parameters字典中的参数.
3.net.parameters()或net.named_parameters()打印出来是整个网络定义的参数
4.net.named_modules() 如上述一样打印出来的只是submodule中的参数 值得注意的是，这里是递归返回，从外层module到内层module

for name,buf in model.named_buffers()
for name,module in model.named_chlidren() #没有递归操作
"""


"""
=================================================================================================
测试阶段
"""
"""
#提前设置循环参数
module.training=False
module(test_input)
值得注意的是，虽然可以通过直接设置training属性，来讲子module设为train和eval模式，但是这种方式较为繁琐，因为如果一个模型具有多个dropout层，就需要为每个dropout层指定training。

一般来说：
调用module.train()函数，将当前module及其子module中的所有training属性设为True
调用module.eval()函数将training属性都是设置为False

值得注意的是，在进行推理之前，必须调用model.eval()来设置dropout和bn层
"""

"""
=================================================================================================
预训练的模型
"""
"""
pytorch模型子包中，包含了不同模型的定义，对应于不同的任务：分类、检测等

#分类任务
import torvision.modules as modules
#利用随机权重构建模型
certian_model=models.certain_model()

使用预训练的模型
#预训练的模型使用的是torch.utils.model_zoo
certain_model=models.certain_model(pretrained=True)

实例化一个预训练的模型，将会将其权重下载到缓存目录中。这个目录可以进行更改设置
所有的预训练模型，要求输入图像以相同的方式进行归一化，batch*3*H*W,其中H和W至少是224。图像应该加载到[0,1]范围之间，然后利用均值mean=[0.485, 0.456, 0.406],标准差 std=[0.229, 0.224, 0.225]进行归一化
如何获得这些均值和方法，是利用的ImageNet data数据集进行计算的，具体背后的计算过程见https://pytorch.org/docs/stable/torchvision/models.html
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

# optimizer=optim.SGD([
#     {'params':net.features.parameters()},
#     {'params':net.classifier.parameters(),'lr':1e-2},
#     ],lr=1e-5
# )
print('pass')

