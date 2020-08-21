#IOU
"""
书接上文,'g_4_2 pytorch-diDL-cv-IOU'

首先回归一下，目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域时候包含感兴趣的目标，并调整区域边界从而更准确的预测目标的真实边界框
不同的模型使用的区域采样方法可能不同，其中一种就是anchor方法

其中生成的某个anchor,较好的覆盖了图像中的狗，如果该目标的ground truth已知，
（'较好'是如何量化是一个问题）,一种直观的方法是衡量anchor和真实边界框之间个ground truth的相似度，jaccard系数可以衡量两个集合的相似度，给定集合A和B，jaccard系数为J(A,B)=(|A交B|)/(|A并B|)

实际上可以将边界框内的像素区域看成是像素的集合，因此可以利用两个边界框的像素集合得J系数来衡量两个边界框的相似度，此时将J系数称为IOU，即两个边界框相交面积与相并面积之比
IOU的取值为[0,1]
"""

"""
编程注意事项，
导入自定义函数是：
1. form file_name import *
   form file_name import function_name
   这里注意的是 filename 貌似不能用数字开头命名

2. 在导入的自定义函数中，除了定义的函数，尽量不要有可直接运行的命令，
因为这样，在import阶段就会运行这些命令
如果不想直接删除或注释掉这些code，那么使用下面的语句
if __name__ == '__main__'
name表示当前模块，如果该模块被直接运行，那么那么name就是main; 间接import name就不是main 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

#IOU的实现
def compute_intersection(set_1,set_2):
    """
    计算anchor之间的交集
    :param set_1: (n1,4)大小的tensor，anchor表示为(xmin,ymin,xmax,ymax)
    :param set_2: (n2,4)大小的tensor，anchor表示为(xmin,ymin,xmax,ymax)
    :return: set_1中每个box相对于set_2中每个box的交集
    """

    lower_bounds=torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0))
    upper_bounds=torch.min(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0))

    intersection_dims=torch.clamp(upper_bounds-lower_bounds,min=0)
    return intersection_dims[:,:,0]*intersection_dims[:,:,1]

def compute_jaccard(set_1,set_2):
    """
    计算anchor之间的IOU
    :param set_1: 同上
    :param set_2: 同上
    :return: set_1中每个box相对于set_2中每个box的IOU
    """

    intersection=compute_intersection(set_1,set_2)

    areas_set_1=(set_1[:,2]-set_1[:,0]*(set_1[:,3]-set_1[:,1]))
    areas_set_2=(set_2[:,2]-set_2[:,0]*(set_2[:,3]-set_2[:,1]))

    union=areas_set_1.unsuqeeze(1)+areas_set_2.unsqueeze(0)-intersection

    return intersection/union

