#autograd

import torch

"""
=================================================================================================
有几个问题待解决
"""

"""
=================================================================================================
计算图
"""
"""
1.计算图是一种有向无环图，记录算子与变量之间的关系
2.一般用矩形表示算子，椭圆表示变量
3.pytorch的autograd会随着用户的操作，记录生成当前variable的所有操作，并由此建立一个有向无环图。
4.用户每进行一个操作，相应的计算图就会发生改变。
5.在更低层的实现中，计算图中记录了操作Function,每个变量在图中的位置可通过其grad_fn属性在图中的位置推测得到，
6.在反向传播中，autograd沿着这个图总当前变量（根节点）溯源，可以利用链式法则计算所有叶子节点的梯度
7.每个前向传播操作的函数都有与之对应的反向传播函数用来计算输入的各个变量的梯度，这个函数的函数名通常以Backward结尾，如addBackward
8.动态图：计算图在每次前向传播时都是从头开始构建
9.反向传播中，非叶子节点的导数计算完后即被清空
*10*.叶子节点中需要求导的变量variable，具有AccumulateGrad标识，因为其梯度是累加的，...???...
*11*.多次反向传播时，梯度是累加的，反向传播的中间缓存会被清空，为进行多次反向传播需要指定retain_graph来保存这些缓存，...???...
*12*.反向传播backward函数的参数grad_variables可以看成是链式求导的中间结果，如果是标量，可以省略，默认为1，...???...
*13*.variable的`volatile`属性默认为False，如果某一个variable的`volatile`属性被设为True，那么所有依赖它的节点`volatile`属性都为True。volatile属性为True的节点不会求导，volatile的优先级比`requires_grad`高，...???...

"""

"""
=================================================================================================
指定requires_grad
"""
"""
.requires_grad=True 表示表示开始追踪所有在tensor上的操作，完成所有的计算后，就可以调用.backward()，然后所有的梯度都会自动被计算。tensor上的梯度会被累积到.grad性质上。
值得注意的是dtype是float才可以有requires_grad

#法1
certain_tensor=torch.randn(3,4,requires_grad=True)
#法2
certain_tensor=torch.randn(3,4).requires_grad_(True)
#法3
certain_tensor=torch.randn(3,4)
certain_tensor.requires_grad=True
"""

"""
=================================================================================================
tensor梯度属性
"""
"""
#查看自动求导属性
certain_tensor.requires_grad

#查看反向传播对应的函数
#值得注意的是由用户创建的variable没有grad_fn,一般叶子节点没有grad_fn,然后在根据用户定义查看时候有requires_grad,有requires_grad就有grad
#注意.grad_fn有next_functions的属性，也就是反向溯源的下一个Backward函数，certain_tensor.grad_fn.next_function
certain_tensor.grad_fn #即该tensor是如何得来的,记录创建该张量时所用的方法

#查看叶子节点
certain_tensor.is_leaf
"""


"""
=================================================================================================
backward
"""
"""
保留非叶子节点的梯度grad
x=torch.tensor([1.],requires_grad=True)
y=torch.tensor([2.],requires_grad=True)
z=torch.add(x,y)
z.retain_grad()
z.backward()


#多次执行backward
torch.autograd.backward(tensor,grad_tensor=None,retain_grad=None,create_graph=False)
create_graph 创建导数计算图，用于高阶求导
grad_tensor 用于多个梯度之间权重的设置
值得注意的是，当loss不为标量的时候，loss.backward(troch.ones(loss.size()))

计算一次backward后，直接在计算一次计算图会报错，因为pytorch计算完一次计算图后会将计算图释放掉
如果想再次调用，则需要retain_graph
例子：y=w*x+b
计算w的梯度的时候，需要用到x的数值，这些数值在前向过程中会保存成buffer，在计算完梯度之后会自动清空。为了能够多次反向传播需要指定`retain_graph`来保留这些buffer
即：
z.backward(retain_gragh=True)


#计算梯度
为了停止tensor追踪操作的历史，可以调用.detch()，从而从计算历史中隔离出来，并防止后续的操作被追踪
d(f)/d(x)=
[[d(f1)/x1,d(f1)/x2,...,d(f1)/xm],
 [d(f2)/x1,d(f2)/x2,...,d(f2)/xm],
 ...
 [d(fn)/x1,d(fn)/x2,...,d(fn)/xm],]
 这是一个n*m的雅克比矩阵
 
 如果l=g(y)是一个标量函数，则
 d(l)/d(y)=
[[d(l)/d(y1),d(l)/d(y2),...,d(l)/d(yn)]]
这是一个1*n的雅克比矩阵

根据链式法则
d(l)/d(x)=[d(l)/d(y)]*[d(y)/d(x)]
(1*n)*(x,m)=1*m的矩阵
"""


"""
=================================================================================================
grad(torch.autograd.grad())
"""
"""
也是用于求导梯度，得到的是想要的tensor的梯度
torch.autograd.gard(outputs,inputs,grad_outputs=None,retain_graph=None,create_graph=False)
certain_grad=torch.autograd.grad(loss,leaf_tensor,ceate_graph=True)
"""


"""
=================================================================================================
关闭自动求导
"""
"""
在inference,即测试推理时，不需要计算梯度
即：
方法一：
#常用于在模型评估时
with torch.no_grad():
    pass
方法二;
torch.set_gard_enabled(False)
pass
torch.set_grad_enabled(True)
"""

"""
=================================================================================================
tensor.data和tensor.detach()的用途
"""
"""
如果想修改tensor数值，但是不希望被autograd记录，那么可以用tensor.data进行操作
即：
certain_tensor=torch.tensor([pass],requires_grad=True)
#修改，但不需要被记录,但是原来的tensor会改变
certain_tensor.data=pass_operation

#.detch() ...没懂???...
为了停止tensor追踪操作的历史，可以调用.detch()，从而从计算历史中隔离出来，并防止后续的操作被追踪

尽量用tensor.data
"""

"""
=================================================================================================
非叶子节点的梯度属性
"""
"""
查看非叶子节点变量的梯度

还可以参考上面的retain_grad()

方法一:
假设z为根节点，y为中间节点，但是z,y都不是叶子节点，在z.backward()后会被清空，
解决:隐式调用backward()函数
torch.autograd.grad(z,y)
法二：
利用hook函数，但是没看懂...???...
"""







# def abs(x):
#     if x.data[0]>0:
#         return x
#     else:
#         return -x
#
# x=torch.ones(1,requires_grad=True)
# y=abs(x)
# print('grad_fn:',y.grad_fn)
# y.backward()
# print(x.grad)

# # 第二种方法：使用hook
# # hook是一个函数，输入是梯度，不应该有返回值
# def variable_hook(grad):
#     print('y的梯度：',grad)
#
# x = torch.ones(3, requires_grad=True)
# w = torch.rand(3, requires_grad=True)
# y = x * w
# # 注册hook
# hook_handle = y.register_hook(variable_hook)
# z = y.sum()
# z.backward()
#
# # 除非你每次都要用hook，否则用完之后记得移除hook
# hook_handle.remove()

x=torch.tensor([1.],requires_grad=True)
y=torch.tensor([2.],requires_grad=True)
z=torch.add(x,y)
z.retain_grad()
w=z.add(1)
w.backward()

print('pass')