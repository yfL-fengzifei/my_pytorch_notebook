# pytorch提供的autograd包能够根据输入和前向传播过程自动构建计算图，并执行反向传播
# 如果将tensor的属性.requires_grad=True，则就开始追踪track,在其上的所有操作（这样就可以利用链式法则进行梯度传播了.完成计算后，可以调用.backword(),来完成所有梯度计算。此tensor的梯度将累积到.grad属性中
#注意grad在反向传播过程中是累加的，这意味着每一次运行反向传播，梯度都会累加之前的梯度，所以一般在反向传播之前需要把梯度清零
#注意在y.backword()时，如果y是标量，则不需要为backword()传入任何参数，否则，需要传入一个与y同形的tensor
#如果不想被继续追踪track，可以调用.detach()，将其从追踪记录中分离出来，这样可以防止将来的计算被追踪，这样梯度就传不过去了。此外，还可以用with torch.no_grad()将不想被追踪的操作代码块包裹起来，这种方法在评估模型的时候很常用。因为在评估模型时，并不需要计算可训练参数（requires_grad）的梯度
#function是另外一个很重要的类。tensor与function相结合可以构建一个记录有整个计算过程的有向无环图DAG，每个tensor都有一个.grad_fn属性1，该属性即创建该tensor的function，就是说该tensor是不是通过某些运算得到的，若是，则grad_fn返回一个与这些运算相关的对象，否则为None.（我的理解是grad_fn记录的是这个tensor是利用什么方法得到的）

import torch
import numpy as np
x=torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn) #因为x是直接创建的，所以没有grad_fn

y=x+2
print(y)
print(y.grad_fn) #y是通过加法创建的，所以有grad_fn会有记录

#综上，x是直接创建的，被称为叶子节点，叶子节点对应的grad_fn是None
print(x.is_leaf,y.is_leaf) #判断是否为叶子节点


z=y*y*3
out=z.mean()
print(z,out)

#对于requires_grad可以后期就地修改
a=torch.randn(2,2) #不指明的情况下，requires_grad=False
a=((a*3)/(a-1))
print(a)
print(a.requires_grad) #此时a是否执行自动梯度
a.requires_grad_(True) #就地操作，指定执行自动梯度
print(a.requires_grad) #查看属性
b=(a*a).sum()
print(b)
print(b.requires_grad) #查看b的属性
print(b.grad_fn) #查看构建b的操作

#当requires_grad定义为True是，所有的操作会被追踪，并自动执行梯度计算
#当执行.backword时，会完整所有梯度的计算,
#然后所有的梯度会累积到tensor的grad属性中

#因为out是标量，所以调用backward时不需要指定求导变量
#out关于x的梯度，为d(out)/dx
"""
貌似此时可能不许要
#根据上面的说法，因为梯度是累加的所以，需要进行清零
#x.grad.data.zero_()
"""
print(x.grad)
out.backward() #等价于out.backward(torch.tensor(1.0))，这里指定的是从out执行backward,相当于out为输出
print(x.grad)
#print(y.grad) #默认情况下，backward只会保留叶子节点的梯度

pass