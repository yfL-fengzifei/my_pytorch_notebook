#image augmenstation图像增强
"""
图像增强(增广)通过对训练图像做一系列随机改变，来产生相似但不同的训练样本，从而扩大训练数据集的规模
图像增强的的另一种解释：随机改变训练样本可以降低模型对某些属性的依赖，从而提高模型的泛化能力
如：对图像进行不同方式的裁剪，是感兴趣的物体出现在不同位置，从而减轻模型对物体出现位置的依赖性
调整亮度、色彩等因素来来降低模型对色彩的敏感度

...具体显示图像的例子没看...

翻转
torchvision.transforms.RandomHorizontalFlip() 一半概率的图像进行左右翻转
torchvision.trandforms.RandomVerticalFlip() 一半概率的图像进行上下翻转

随机裁剪
让物体以不同的比例出现在图像的不同位置，这样能够降低模型对目标位置的敏感性
如：随机裁剪出一块面积为原面积10%~100%的区域，且该区域的宽和高之比为0.5~2之间，然后再讲该区域的宽和高分别缩放到200像素
torchvision.transforms.RandomResizedCrop(200,scale(0.1,1),ratio=(0.5,2))


改变颜色
亮度brihgtness、对比度contrast、饱和度saturation、色调hue
如：将图像的亮度随机变化为原图亮度的50%(1-0.5)~150%(1+0.5)
torchvision.transforms.ColorJitter(brightness=0.5)
如：随机变化图像的色调
torchvision.transforms.ColorJitter(hue=0.5)
如：随机变化图像的对比度
torchvision.transforms.ColorJitter(contrast=0.5)
如：同时设置随机变化的亮度、对比度、饱和度、色调
torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)


叠加多个图像增广方法
torchvision.transforms.Compose([torchvision.transforms.RandHorizontalFlip(),torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),torchvision.transforms.RandomResizedCrop(200,scale(0.1,1),ratio=(0.5,2))])

为了在预测时得到确定的结果，通常只将图像增广应用在训练样本山，而不再预测时使用含随机操作对的图像增广
再有，使用ToTensor架构小批量图像转成pytorch需要的格式，即形状为(批量大小、通道数、高、宽)，值域在0到1之间且类型为32位浮点数
torchvision.transforms.ToTensor()
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

img=Image.open('./cat1.jpg ')
img.show()
# plt.imshow(img)
# plt.show()



