#数据集

"""
=================================================================================================
image 维度的区别
tensor\PIL\opencv(一般图像) 中维度是不同的
"""


"""
=================================================================================================
内置数据集
"""
"""
torch.utils.data 该模块提供了有关数据处理的工具

import torchvision

torchvision 包含三个部分
.models:提供各种经典的网络结构以及预训练好的模型
.datasets:提供常用的数据集加载，继承于torch.utils.data.Dateset;例子：torchvision.datasets.certain_datasets
.transforms：提供常用的预处理皂搓，主要包括对tensor以及PILImage对象的操作
.utils 其他一些有用的方法

certain_dataset=torchvision.dataset.certainset() #返回的是一个Dataset对象，后面还需要跟上DataLoader

resnet34=models.squeezenet1_1(pretrained=True,num_classes=1000)
#加载预训练好的模型，如果不存在会进行下载
#预训练好的模型保存在~/.torch/models下面

例子:
data_train=torchvision.datasets.certain_datasets()
data_test=torchvision.datasets.certain_datasets()
访问：
feature,lable=data_train[0] 数据集返回的是(feature,label)，这里不是一个batch,而是单一图像样本，这里相当于ImageLodaer或自己创建的数据集，后面还需要进行DataLoader
值得注意的是data_train[0]返回的是一个元组，(tensor(features),int(label_)
数据集单一样本返回的feature，因为使用了transforms.ToTensor()，所以每个像素的数值为[0.0,1.0]的32浮点数
"""


"""
=================================================================================================
创建自己的数据集-dataset
"""
"""
Dataset是一个抽象类，所有自定义的Dataset需要继承他并且复写__getitem__()函数，即接受一个索引，返回一个样本
    
数据加载，可通过自定义的数据集对象。数据集对象被抽象为Dataset类，实现自定义的数据集需要继承Dataset,并实现两个python 魔法方法
__getitem__：返回一条数据或一个样本，obj[index]等价于obj.__getitem(index)
__len__:返回样本的数量，len(obj)等价于obj.__len__()

例子：
# import torch
import torch.utils.data as data
# import os
# from PIL import Image
# import numpy as np
#
# class DogCat(data.Dataset):
#     def __init__(self,root):
#         imgs=os.listdir(root)
#         # 所有图片的绝对路径
#         # 这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
#         self.imgs=[os.path.join(root,img) for img in imgs]
#
#     def __getitem__(self, index):
#         img_path=self.imgs[index]
#         label = 1 if 'dog' in img_path.split('/')[-1] else 0
#         pil_img=Image.open(img_path)
#         array=np.array(pil_img) #不会占用新的内存
#         data=torch.from_numpy(array)
#         return data,label
#
#     def __len__(self):
#         return len(self.imgs)
#
# dataset=DogCat('./data/dogcat/')
# img,label=dataset[0] #相当于调用dataset.__getitem__(0)
# for img,label in dataset:
#     print(img.size(),img.float().mean(),label)

*值得注意的是：
上面的代码显示，可以自己定义自己的数据集，并依次获取，但这里返回的数据不适合实际使用，因为：
返回样本的形状大小不一，因为每张图片的大小不一，这对需要batch训练的神经网络来说不好
返回样本的数据较大，未归一化至[-1,1]
"""


"""
=================================================================================================
创建自己的数据集-数据增强
"""
"""
transform主要接受的是PIL图像，opencv呢

import torchvision.transforms as transforms

transforms.FiveCrop(size)
transforms.TenCrop(size,vertical_flip=False)
注意上面返回的是tuple的形式，注意返回的维度

transforms.Lambda封装自定义的转换策略，
trans=transforms.Lambda(lambda img:img.rotate(random()*360))

transforms.ToTensor() 将数据转换为tensor,会自动归一化自动将[0,255]归一化至[0,1],显示的时候要返回归一化,
#transforms.ToTensor将(H*W*C)且数据位于[0,255]的PIL的图片或者数据类型为np.uint8的numpy数组转换成尺寸为(C*H*w)且数据类型为torch.float32且位于[0.0,1.0]的tensor
transform.ToPILImage 将tensor转换成PILImage对象
#值得注意的是上述的操作定义后都是以函数的形式存在的，需要二次调用
#值得注意的是因为像素值在[0,255]，所以刚好是uint8所能表示的范围，包括transforms.ToTensor在内的一些关于图片的函数就默认输入是uint8类型，若不是，可能不会报错，但是可能得不到想要的效果。因此，如果用像素值(0-255整数)表示图片数据，那么一律将其类型设置成uint8，避免不必要的bug，见文档中的博客

transforms.Normalize(mean,std,inplace_Fasle) 逐channel的对图像进行标准化，（归一化RGB的均值和方差），最终变成的是[-1,1]之间的数值
output=(input-mean)/std 加速模型收敛

‘bilibili-pytorch框架’有一个逆transforms_invert的函数

"""
"""
数据增强的策略：
让训练集与测试集更接近，逼近分布
"""

"""
=================================================================================================
创建数据集-ImageFolder
"""
"""
#值得注意的是，ImageFolder是一个常用的Dataset,注意与上述自己实现的方法进行比较

ImageFolder 假设所有的文件按文件夹保存，每个文件夹下存储同一类别的图片，文件夹名为类名，构造函数如下
ImageFolder(root,transform=None,target_transform=None,loader=default_loader)
root 在root指定路径下寻找图片
transform 对PILImage进行转换，transform的输入时使用loader读取图片的返回对象
target_transform 对label进行转换
loader 给定路径后如何读取图片，默认读取为RGB格式的PIL对象

label是按照文件夹名顺序排列后存成字典，即{类名：序列号(从0开始)}
要想知道label的具体意义，应该事先定义一个tuple,classes=('class_name1',...,'class_namen')
print(classes[label]) #得到label指代的意义

#属性
dataset=ImageFloader(root)
dataset.classes #查看类别,list形式
dataset.class_to_index #查看类别和映射的标签，dict形式
dataset.imgs #查看图像，list[tuple()],[(path,lable),(...)]

data,label=dataset[idx]
"""

"""
=================================================================================================
创建数据集-TensorDataset
"""
"""
import torch.utils.data as Data
torch_dataset=Data.TensorDataset(data_tensor=,target_tensor=) #就是将tensor作为数据和标签，而不是img
将训练数据的特征和标签组合，组合之后，每个dataset[i]返回的就是(data,label)形式

"""


"""
=================================================================================================
创建数据集-DataLoader
"""
"""
torch.utils.data.DataLoader
Dataset只负责数据的抽象，一次调用__getitem__只返回一个样本，但是实际情况是利用batch做训练，同时需要对数据进行shuffle和并行加速，
因此有了DataLoader,即dataset也是可以迭代的，只不过一次返回一个数据样本，data返回的是一个batch

DataLoader函数
DataLoader(dataset,batch_size=1,shuffle=False,sampler=None,num_workers=0,collate_fn=defalut_collate,pin_memory=Flase,drop_last=False)

dataset:加载的数据集(dataset对象)
batch_size:batch_size
shuffle:数据打乱
sampler:样本抽样，定义从数据集中抽取样本的策略,如果指定则忽略shuffle
num_workers：使用多进程加载的进程数，0表示不适用多进程，(貌似在Windos中不能使用多进程)
collate_fn:如何将多个样本数据拼接成一个batch,一般使用默认的拼接方式即可，综合一系列的样本从而形成一个mini-batch张量
pin_memory:是否将数据保存在pin_memory,pin_memory中的数据转到GPU会快一些
drop_last:不足,batch丢弃

dataloader是一个可迭代的对象，意味着可以像使用迭代器一样使用它
for batch_datas,batch_labels in dataloader:
    train

dataiter=iter(dataloader)
batch_datas,batch_labels=next(dataiter)
即用iter()函数返回一个迭代器
dateriter.next()或next(dataiter)，就相当于batch(dataset[idx]),返回的是batch的data和label


在数据处理中，有时会出现某个样本无法读取等问题，比如某张图片损坏。这时在`__getitem__`函数中将出现异常，此时最好的解决方案即是将出错的样本剔除。
如果实在是遇到这种情况无法处理，则可以返回None对象，然后在`Dataloader`中实现自定义的`collate_fn`，将空对象过滤掉。但要注意，在这种情况下dataloader返回的batch数目会少于batch_size。

from torch.utils.data.Dataloader import default_collate #导入默认的拼接方式
def my_collate_fn(batch)
    batch=list(filter(lamda x:x[0] is not None, batch)) #batch相当于x
    if len(batch)==0:
        return torch.Tensor()
    return default_collate(batch)
dataloader=Data.DataLoader(dataset,2,collate_fn=my_collate_fn,num_workers=0,shuffle=True)
for batch_datas,batch_labels in dataloader:
    print(batch_data.size(),batch_labels.size())

不丢弃的方法：没看懂（见'python-book'）
多进程的问题：没看（见'python-book'），(貌似在Windos中不能使用多进程)

sampler模块：
用于对数据进行采样。常用的有随机采样器：randomSampler,当dadalodar的shuffle参数为True时，系统会自动调用这个采样器，实现打乱数据。
shuffle=False时，默认的是采用SequentialSampler，他会按顺序一个一个进行采样
还有一种采样方法是，weightedRandomSampler，会根据每个样本的权重选取数据，在样本比例不均衡的问题中，可用它进行重采样
构建weightedRandomSampler时需提供两个参数：每个样本的权重weights,供选取的样本总数num_samples,以及一个可选参数replacement，权重越大的样本被选中的概率越大，带选取的样本数目一般小于全部的样本数目，replacement用于指定是否可以重复选取某一个样本，默认为True,即允许在一个epoch中重复采样一个数据，如果设为False,则当某一类的样本被全部选取完，但其样本数目仍为达到num_samples时，sampler将不会再从该类中选择数据，此时可能到时权重weights参数失效
没懂...
"""



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as Data
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
#
# # dataset=ImageFolder('./dogcat_data/dogcat_2')
# # print(dataset.class_to_idx)
#
# # tran=transforms.ToTensor()
# # dataset=ImageFolder('./dogcat_data/dogcat_2',transform=transforms.ToTensor())
#
# normalize = transforms.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
# transform  = transforms.Compose([
#          transforms.RandomResizedCrop(224),
#          transforms.RandomHorizontalFlip(),
#          transforms.ToTensor(),
#          normalize,
# ])
# dataset=ImageFolder('./dogcat_data/dogcat_2',transform=transform)
#
# weights = [2 if label == 1 else 1 for data, label in dataset]
#
# dataloder=Data.DataLoader(dataset,batch_size=3,shuffle=True,num_workers=0,drop_last=False)
#
#
# print('pass')

# import os
# from PIL import Image
# import numpy as np
# import torchvision.transforms as transform
#
# transformer=transform.Compose([
#     transform.Resize(224),#缩放图片(Image)，保持长宽比不变，最短边为224像素
#     transform.CenterCrop(224),# 从图片中间切出224*224的图片
#     transform.ToTensor(), #图片(Image)转成Tensor，归一化至[0, 1]
#     transform.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5]) ## 标准化至[-1, 1]，规定均值和标准差
# ])
#
# class DogCat(data.Dataset):
#     def __init__(self,root):
#         imgs=os.listdir(root)
#         # 所有图片的绝对路径
#         # 这里不实际加载图片，只是指定路径，当调用__getitem__时才会真正读图片
#         self.imgs=[os.path.join(root,img) for img in imgs]
#         self.transforms=transformer
#
#     def __getitem__(self, index):
#         img_path=self.imgs[index]
#         label = 1 if 'dog' in img_path.split('/')[-1] else 0
#         data=Image.open(img_path)
#
#         if self.transforms:
#             data=self.transforms(data)
#         return data,label
#
#     def __len__(self):
#         return len(self.imgs)
#
# dataset=DogCat('./data/dogcat/')
# # img,label=dataset[0] #相当于调用dataset.__getitem__(0)
# for img,label in dataset:
#     print(img.size(),label)