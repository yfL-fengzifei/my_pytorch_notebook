#torch tensor and autograd

import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""
总体上来说
torch.function 等价于 tensor.function

存储上
.function() 返回新的tensor
.function_() 就地inplace操作，修改本tensor
"""

"""
=================================================================================================
tensor的类型及其转换
"""
"""
torch.float - torch.FloatTensor #float32(float) float64(double) float16(half) 
torch.double -torch.DoubleTensor #int16(short) int32(int) int64(long)
torch.int8 -torch.CharTensor
torch.uint8 - torch.ByteTensor
torch.Tensor 是 torch.FloatTensor 的别名

转换法1
# certain_tensor.double()
# certain_tensor.float()
# certain_tensor.half() #and so on
转换法2
# certain_tensor.to(torch.float32)
转换法3
#certain_tensor.type(torch.float32)
"""


"""
=================================================================================================
tensor的创建
"""
"""
#创建函数
Tensor() 注意与tensor的不同 相当于torch.FloatTensor
tensor()
ones()
zeros()
eye()
arrange() 有步长， 左闭右开
linsapce() 均分切份，左闭右闭
logspace() log数列，均匀切分，左闭右闭，base为log的底（默认为10）
rand()/randint()/randn() 均匀[0,1)/[a,b)/标准正态分布

normal/uniform() 正态/均匀分布  （四种模式，张量/标量）
randperm() 随机排列0~n-1
full() 自定义元素数值
torch.*_like(certain_tensor) #创建类型，形状 device都相同，但是元素不同的tensor

空tensor 就是不输入元素就行
certian_tensor=torch.tensor()

克隆
certain_tensor=old_tensor.clone() #不再共享内存


#同时执行类型和运行部署
dtype=torch.XXX
device=toch.XXX
"""


"""
=================================================================================================
tensor的性质
"""
"""
#tensor的维度大小
certain_tensor.size() #返回的是tuple的形式
certain_tensor.shape

#tensor元素
certain_tensor.item() #只能用在只有一个元素的tensor上面

#tensor的元素个数
certain_tensor.numel()

#打印tensor 
print - %s

#控制精度
%
.format
torch.set_printoptions

#查看内存地址
print(id(certain_tensor))


#查看类型
a.type() 
a.dtype 
certain_tensor.type(new_type)

#内存管理
数值分配在连续的内存块中，由torch.Storage实例管理。存储（Storage）是一个一维的数值数据数组，例如一块包含了指定类型（可能是float或int32）数字的连续内存块。PyTorch的张量（Tensor）就是这种存储（Storage）的视图（view），我们可以使用偏移量和每一维的跨度索引到该存储中。
多个张量可以索引同一存储，即使它们的索引方式可能不同
由于基础内存仅分配一次，所以无论Storage实例管理的数据大小如何，都可以快速地在该数据上创建不同的张量视图

certain_tensor.storage()表明，即使创建的tensor是多维的，但是在内存中存储的依然是连续的数组，从这个意义上讲，张量知道如何将一对索引转换为存储中的某个位置。

"""


"""
=================================================================================================
tensor的数据结构类型转换
"""
"""
a=torch.tensor([[1,2,3],[1,2,3]])

#tensor 与 numpy
#注意互相转换后是共享内存的
certain_tensor=torch.from_numpy(certain_ndarray)
certain_ndarray=certain_tensor.numpy()

#tensor 与 list
#转换成list,可直接写（先转成numpy在tolist也行）
b=a.tolist()

python-list在内存中是被独立分配的，并不连续，而
pytorch-tensor and numpy-ndarray是连续内存块上的视图(view)
"""


"""
=================================================================================================
tensor的修改形状与索引
"""
"""
#修改形状
torch.view() 调整形状，共享内存

#添加和减少维度
#注意维度的顺序，[X,X,...,X,H,W]
#可以指定维度或不指定维度
torch.squeeze() #减少形状
torch.unsqueeze() #添加形状
a.view(1,a.shape[0],a.shape[1]) 等价于 a.unsqueeze(0)

#resize
#值得注意的是resize不能改变整体的形状，resize_会可以超过或少于原来的维度

#None
#None在所有哪个索引位置，就会在哪个位置增加一个维度
a[None] 等价于 a[None,:,:]

#transpose
转换维度
"""


"""
=================================================================================================
tensor的数学逻辑运算与索引
"""
"""
ceratin_tensor > 1 #返回的是一个逻辑tensor
a[a>1] #只把满足要求的返回成一个list

#certain_tensor.mask_select()
a[a>1] 等价于 a.mask_select(a>1)

#certain_tensor.index_select()
#在指定维度上选取，如选取某行某列,见CSDN
#在'dL-intopytorch'中‘3_2’有活用index_select和yield的例子

#certain_tensor.nonzero()
#返回非零元素的下标

#gather...没看("python-book")

#逐元素操作
abs,squrt,div,exp,fmod,log,pow,...
cos,sin,asin,atan2,cosh
ceil round,floor,trunc
clamp #截断
sigmod,tanh 激活函数
** #表示逐元素平方
* #表示逐元素相乘


#归并操作,有的可指定维度
#注意keepdim=True 表示保留归并下的维度
mean,sum,median,mode
norm,dist
std,var
cumsum,cumprod

#逻辑运算
gt,lt,ge,le,eq,ne
topk
sort
max,min

"""


"""
=================================================================================================
tensor的操作
"""
"""
#拼接
cat() 不会增加维度
stack() 会增加维度

#切分
chunk()
split() #更强大
"""


"""
=================================================================================================
tensor的运算平台转换 cpu gpu
"""
"""
certain_tensor.cuda.XXX
certain_tensor.cuda.cpu
certain_tensor.to(device=XXX)

torch.set_num_threads 可以设置pytorch进行cpu多线程并行计算时候所占用的线程数，这个可以用来限制Pytorch所占用的cpu数

# 第一种创建方法
# certain_tensor_gpu=torch.tensor([1,2,3],device='cuda')
# 还可以利用to将在CPU上创建的tensor复制到GPU
# certain_tensor_gpu=certain_tensor.to(device='cuda')
# 将certain_tensor放在指定的gpu上，当设备有多个gpu时
# certain_tensor_gpu=certain_tensor.to(device='cuda:0') #0表示下标

# gpu的tensor回流到cpu上
# certrain_tensor_cpu=certain_tensor_gpu.to(device='cpu')

# 上述更简单的使用方式，使用to方法可以同时改变device和dtype,而下面的方法不能同时改变
# points_gpu = points.cuda() # 默认为GPU0
# points_gpu = points.cuda(0)
# points_cpu = points_gpu.cpu()
"""


"""
=================================================================================================
tensor的矩阵运算
"""
"""
tracw 迹
diag 对角线元素
triu tril 上三角，下三角
mm bmm 矩阵乘法，batch的矩阵乘法，正经的矩阵乘法
mul 对应按位相乘
addmm addbmm addnv addr矩阵运算
transpose 维度变换
t 转置 #注意使用contiguous
dot cross 内积 外积
inverse 求逆
svd svd分解
"""


"""
=================================================================================================
tensor的矩阵运算
"""
"""
所有数组都向其中shape中最长的数组看起，见"python-book"
建议用view,unsqueeze,None
expand,expand_as来重复数组，注意repeat的区别，expand不是真正的复制数组，repeat是复制会额外占用内存
"""


"""
=================================================================================================
tensor的持久化
"""
"""
torch.save()
torch.laod()
"""


"""
=================================================================================================
tensor的向量化
"""
"""
向量化计算是一种特殊的并行计算方式，相对于一般程序在同一时间执行多个操作，通常是对不同的数据执行同样的一个或一批指令，或者说把指令应用在一个数组、向量上，向量化。
"""

"""
=================================================================================================
tensor的加载与保存
"""
"""
# 将tensor保存到文件中
# torch.save(points,'../../data/chapter/ourpoints.t') #note that points is a tensor created by users
# 另一种方法
# with open('../../data/chapter2/ourpoints.t','wb') as f:
#     torch.save(points, f)

# 将tensor加载回来
# certain_tensor=torch.load('../../data/chapter/ourpoints.t')
# 加载的另一种方法是
# with open('../../data/chapter2/ourpoints.t','rb') as f:
#     points = torch.load(f)
"""


"""
=================================================================================================
小demo
"""
"""
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
等价于features=torch.randn([num_examples,num_inputs],dtype=torch.float32)
"""



