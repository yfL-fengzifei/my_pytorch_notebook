#object detection and boundingbox and anchor
"""
bounding box边界框，来描述目标位置。边界框是一个矩形框，可以由矩形左上角x和y坐标与右下角的x和y坐标确定

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import os
import matplotlib.pyplot as plt

from PIL import Image
img=Image.open('./catdog.jpg')
plt.imshow(img)
plt.show()

#这里定义的左上和右下角
dog_bbox,cat_bbox=[60,45,378,516],[400,112,655,493]

#在图中将边界框画出来，以检查其准确

fig=plt.imshow(img)

#常用的图像可以用patches。Ellipse来实现或者是直接用plt.rectangle和plt.circle实现
#值得注意的是plt.只实现了常用的几个，复杂的图像需要用到patches
#光创建一个图形对象是不够的，还需要添加进axes对象里面去，使用的是axes.add_patch()方法
#构建patches集合，patches[],patches.append(),PatchCollection(patches),axes.add_collection()
fig.axes.add_patch(
    plt.Rectangle(
        xy=(dog_bbox[0], dog_bbox[1]),
        width=dog_bbox[2] - dog_bbox[0],
        height=dog_bbox[3] - dog_bbox[1],
        fill=False,
        edgecolor='blue',
        linewidth=2
    )
)
fig.axes.add_patch(
    plt.Rectangle(
        xy=(cat_bbox[0], cat_bbox[1]),
        width=cat_bbox[2] - cat_bbox[0],
        height=cat_bbox[3] - cat_bbox[1],
        fill=False,
        edgecolor='red',
        linewidth=2
    )
)

plt.show()