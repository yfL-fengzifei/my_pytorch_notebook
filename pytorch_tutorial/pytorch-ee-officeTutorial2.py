# INTRODUCTION
"""
1. torch Module: 提供各种对tensor的操作
2. tensor and torch.autograd Module: pytorch允许张量跟踪对其所执行的操作，并通过反向传播来计算输出相对于任何输入的导数。注意：此功能由张量自身提供。并通过torch.autograd模块进一步扩展完善。
3. torch.nn Moduel:是pytorch用于构建神经玩过的核心模块，该module提供了常见的神经网络层和其他架构组件。全连接层、激活函数、损失函数都在这个module。这些组件可用于构建和初始化一个还未经训练的模型。
4. torch.util.data Module: 能够找到适用于数据加载和处理的工具。主要用到两个类：Dataset和DataLoader。Dataset(torch.util.data.Dataset)执行自定义数据（可以是任何一种形式）与标准tensor之前的转换。DataLoader(torch.util.data.DataLoader)可以在后台生成子进程来从Dataset中加载数据，使数据准备就绪并在循环可以使用后立即等待训练循环。
5. 针对专用的硬件（多GPU）或利用多台计算资源来训练模型，可以使用torch.nn.DataParallel和torch.distributed
6. torch.optim Module:当模型根据训练数据得到输出结果后，使用该module可以提供更新模型的标准方法，从而使输出更接近与训练数据中的标签。
7. PyTorch内部使用pickle来序列化张量对象和实现用于存储的专用序列化代码。
"""

import torch

# = Tensor
"""
注意在索引tensor的时候，并不是重新赋值给一个新的tensor，而是索引到tensor在内存中对应位置的视图
"""
# a=torch.ones(3)
# print(type(a)) #tensor
# print(type(a[1])) #tensor
# print(type(float(a[1]))) #float; not a tensor


# compared python-list with pytorch-tesnor/numpy-ndarray
"""
python-list在内存中是被独立分配的，并不连续，而
pytorch-tensor and numpy-ndarray是连续内存块上的视图(view)
"""

# tensor's storage
"""
数值分配在连续的内存块中，由torch.Storage实例管理。存储（Storage）是一个一维的数值数据数组，例如一块包含了指定类型（可能是float或int32）数字的连续内存块。PyTorch的张量（Tensor）就是这种存储（Storage）的视图（view），我们可以使用偏移量和每一维的跨度索引到该存储中。
多个张量可以索引同一存储，即使它们的索引方式可能不同
由于基础内存仅分配一次，所以无论Storage实例管理的数据大小如何，都可以快速地在该数据上创建不同的张量视图。

"""
# 利用tensor.storage()来查看张量的存储
# points = torch.tensor([[1, 2], [3, 4]])
# points.storage()
# print(points.storage()) #表明，即使创建的tensor是多维的，但是在内存中存储的依然是连续的数组，从这个意义上讲，张量知道如何将一对索引转换为存储中的某个位置。
# 手动索引内存
# points_storage = points.storage()
# points_storage[0]
# OR points.storage()[0]


# tensor size尺寸，storage offset存储偏移，stride步长
"""
size,shape是一个元组，表示张量每个维度上有多少个元素,(rows,columns)
存储偏移是存储中与张量中的第一个元素相对应的索引。步长是在存储中为了沿每个维度获取下一个元素而需要跳过的元素数量。
stride也是一个元组，对于2维tensor,stride[0]在内存中沿行方向（维度）索引，需要跨多少个元素索引到下一个元素，stride[1]反之亦然。
"""

# tensor's opertaion
# clone克隆，创建新的tensor
# certain_tensor.clone()
# transpose 转置
# certain_tensor.t() #值得注意的是，a.storage和a.t().storage在内存中的存储数据顺序是一样的，可以看出即使是转置了，两个tensor仍然共享一个内存，即是两个不同的tensor视图，仅仅是尺寸和步长不同
"""...没看完..."""

# tensor's dtype
"""
类型，类
torch.float - torch.FloatTensor #float32(float) float64(double) float16(half) 
torch.double -torch.DoubleTensor #int16(short) int32(int) int64(long)
torch.int8 -torch.CharTensor
torch.uint8 - torch.ByteTensor
torch.Tensor 是 torch.FloatTensor 的别名
"""
# 读取tensor的数据类型
# certain_tensor.dtype
# 数据类型转换
# certain_tensor.double()
# certain_tensor.float()
# certain_tensor.half() #and so on
# 另一种刚简洁的数据类型转换方法
# certain_tensor.to(torch.float32)
# 第三种数据类型转换方法
# certain_tensor.type(torch.float32)


# tensor and numpy
"""
#tensor to numpy
certain_tensor.numpy()

#numpy to tensor
output_tensor=torch.from_numpy(certain_numpy)
"""

# sequence tensor
"""
动态创建张量是很不错的，但是如果其中的数据对你来说具有价值，那么你可能希望将其保存到文件中并在某个时候加载回去。毕竟你可不想每次开始运行程序时都从头开始重新训练模型！
PyTorch内部使用pickle来序列化张量对象和实现用于存储的专用序列化代码。

#数据的互通性，...没看完...
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


# tensor computered on GPU
"""
tensor除了有dtype属性外，还有device的属性，用来设置，该tensor被放置在计算机的什么位置上
除了通过device和to的方式，将tensor放置在GPU上，还可以直接利用troch.cuda.FloatTensor(certain_tensor)
请注意，当计算结果产生后，points_gpu的张量并不会返回到CPU，只要定义了device=‘cuda’后，之后左右针对该certain_tensor的计算都在gpu上，如果想要回流到cpu上，需要certrain_tensor_cpu=certain_tensor_gpu.to(device='cuda')
"""
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


# tensor API: other operations
"""
docs in office :
创建操作——构造张量的函数
索引、切片、连接、变异——改变形状、步长、张量内容
数学操作——通过计算来操作张量内容的函数
    按点操作——函数应用于每个元素，abs\cos
    简化操作——通过张量迭代计算合计值的函数，mean\std\norm
    比较操作——用于比较张量，equal\max
    频谱操作——在频域中转换和运行的函数，stft\hamming_window
    其他操作——特殊函数，cross,对于矩阵的trace
    BLAS和LAPACK
随机采样操作——从概率分布中随机采样值，randn,normal
序列化操作——用于保存和加载张量的函数，save,load
并行操作——控制并行cpu执行的线程数，set_sum_threads
"""
# certain_tensor.method_() #下划线表示就地操作，即直接修改输入而不是创建新的tensor