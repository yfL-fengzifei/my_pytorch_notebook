# using tensor in real world, especially to the image data

import torch

# use tensor to represent real data
"""
将不同类型的现实世界数据表示为pytorch张量
处理何种数据类型，其中电子表格、文本、图像均为数据类型
从文件加载数据
将数据转换为张量
调整张量以便于他们可以作为神经网络模型的输入
"""
"""
tensor是pytorch数据的基础，在神经网络中，输入的是tensor，输出的也是tensor。
神经网络内部和优化期间的所有操作都是tensor之间的操作
神经网络中的所有参数(如权重和偏差)都是tensor
神经网络使用的关键：tensor上的操作并进行有效索引
"""

# conver images data to tensor
"""
从常见的图像格式中载入图像，然后将数据转换为tensor表示,该tensor以pytorch所期望的方式排列图像的各个部分
"""

# images data
"""
图像：表示为按规则网格排列的标量集合，并且具有高度和宽度（以像素为单位）。每个网格点（像素）可能只有一个标量，这种图像表示为灰度图像；或者每个网格点可能有多个标量，它们通常代表不同的颜色或不同的特征（features），例如从深度相机获得的深度
(每个图像都是以网格标量形式组成的，其中网格表示的是像素，一个网格可以只有一个标量，或多个标量)
代表当个像素值得标量通常使用的是8位编码，在特殊的场景下更多位的编码可以提供更大的范围或更高的灵敏度
"""
# 加载图像
"""
利用imageio Module
"""
import imageio

# img_arr = imageio.imread(
#     './bobby.jpg')
# print(img_arr.shape)  # 注意不同于直接定义的numpy的ndarray对象，载入的jpg图像是w*h*c
# # img_arr加载的是一个numpy数组对象，有2个空间维度和第三个颜色维度,即w*h*c
# # 因此将numpy转换成tensor，利用的是torch.from_numpy()
# # 值得注意的是，图像和图像tensor的维度设置是不同的，pytorch模块处理图像数据需要将张量设置为c*h*w(即通道、高度、宽度)
# # 因此可以使用转置transpose获得正确的维度设置
# img = torch.from_numpy(img_arr)
# out = torch.transpose(img, 0, 2)  # 注意转换成tensor的时候，需要利用transpose函数获得争取的维度设置
# print(out.shape)
# # 这里需要注意的是，转置操作不会复制张量数据，与原始tensor数据使用的是相同的内部存储，只是修改了张量和尺度和步长信息，
# # 但是需要注意的是img的改变会使得out的改变


# batch images
"""
上述描述的是一张图像，现在可以创建包含多个图像的数据集以用作神经网络的输入，然后沿着第一维度将这些图像按照批量存储，以获得N*c*h*w张量
其中一个高效的选择是使用堆叠stack来构建张量，你可以预先分配适当尺寸的tensor,并用从文件中加载图像填充它
"""
batch_size = 100
batch = torch.zeros(batch_size, 3, 256, 256,
                    dtype=torch.uint8)
# 定义每个批次包含100个RGB图像，分别为3*256*256。需要注意的是张量的类型，这里希望每种颜色都以8位整数表示，就想大多数标准消费相机照出的相机格式一样

# 现在可以从输入的文件夹中加载所有的png图像并将其存储在张量中
import os

data_dir = './image-cats/'
filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name) == 'png']  # 好像是迭代器的意思，需要再看？？？！！！
for i, filename in enumerate(filenames):
    img_arr = imageio.imread(filename)  # 读取数据
    batch[i] = torch.transpose(torch.from_numpy(img_arr), 0, 2)  # 将numpy-image转换成tensor，注意维度的转换，然后赋值给batch[i]对应的位置

# note
"""
神经网络通常使用浮点张量作为输入
当输入数据的范围为0到1，或-1到1的时候，神经王阔表现出最佳的训练性能(影响来自于如何构造模块的定义)
因此通常的操作就是将张量转换为浮点数并归一化像素。强制转换成浮现数很容易，但是归一化比较麻烦，因为其取决于用户决定的输入的哪个范围应该落在0到1(或-1到1)之间。一种可能的选择是将像素的值除以255

"""
# # one idea
# batch = batch.float()
# batch /= 255.0

# the second idea
# 计算输入数据的均值和标准差并对其进行缩放，以便与每个通道上均值和单位标准差输出为0
n_channels = batch.shape[1]
for c in range(n_channels):
    mean = torch.mean(batch[:,
                      c])
    # 这里有一个需要注意的点，这也是在tensor索引时容易忽略的点，
    # tensor[1:,:]索引tensor第一个维度的位置1处及其后面位置的数据，索引tensor所有第二维度的数据，对于多维tensor，如果索引的参数不等于维度数，那么按照顺序进行索引，缺省参数对应的维度默认全部索引。
    # 因此这里边索引的就是就相当与batch[:,c,:,:],表示的就是将c通道的所有数据(包括batch,h和w)
    # 这里的mean计算的是每个batch批次下的单一通道下的所有数据的均值
    std = torch.std(batch[:, c])  # 同理，这里计算的是标准差
    batch[:, c] = (batch[:, c] - mean) / std  # 同理，对数据进行归一化，这里的加减和处罚

# the third idea
# 这里还可以对输入执行其他几种操作，包括旋转、缩放、裁切等几何变换，这些操作可能有助于训练，或者可能需要进行这些操作使任意输入符合，例如图像的尺寸大小，
