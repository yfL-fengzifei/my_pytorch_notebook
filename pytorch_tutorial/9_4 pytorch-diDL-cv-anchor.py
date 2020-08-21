#object detection based on anchor
"""
目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域中是包含我们感兴趣的目标，并调整区域边缘从而更准确地预测目标的ground-truth bounding box
不同的模型使用的区域采样方法可能不同，其中一种方法是：
以每个像素为中心生成多个大小和高宽比不同的边界框，即为anchor,
"""
#生成多个anchor
"""
假设输入图像高为h,宽为w，分别以图像的每个像素为中心生成不同形状的anchor，设大小s属于(0,1],且宽高比r>0,那么anchor的宽和高分别为w*s*sqrt(r)和h*s*sqrt(r),当中心位置给定时，已知宽和高的anchor是确定的。
假设分别设定一组大小为s1,...sn和一组宽高比r1,...rm，如果以每个像素为中心时使用所有的大小与宽高比的组合，输入图像将一共得到w*h*n*m个anchor(因为一共有w*h像素，n*m个s和宽高比组合)，
虽然这些anchor可能覆盖了所有的真实边界框，但计算复杂度容易过高，因此，通常只对包含s1或r1的大小与宽高比的组合感兴趣，即(s1,r1),(s1,r2),...,(s1,rm),(s2,r1),(s3,r1),...,(sn,r1),
也就是说，以相同像素为中心的anchor的数量为n+m-1,因此对于整个输入图像，会一共生成w*h*(n+m-1)个anchor.
函数为MultiBoxPrior
在上述函数中，anchor表示成(xmin,ymin,xmax,ymax)
feature_map:torch tensor,Shape:[N,C,H,W]
sizes: list of size (0~1) of generated MultiBoxPriores
ratios: list of aspect ratios (no negative) of generated MultiBoxPriors
return: anchors of shape (1,num_anchors,4) 由于batch里每个都一样，所以第一维为1
其中s和r与最终生成的anchor有关


值得注意的是：torchvision.models.detection.rpn里面有一个AnchorGenerator类可以生成anchor，但是与上述不一样
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

img=Image.open('./catdog.jpg')
# plt.imshow(img)
# plt.show(img)

#图像尺寸
w,h=img.size
print(img.size)


#生成anchor
def MultiBoxPrior(feature_map,sizes=[0.75,0.5,0.25],ratios=[1,2,0.5]):
    pairs=[] #(size,sqrt(r))
    for r in ratios:
        pairs.append([sizes[0],math.sqrt(r)]) #(s1,r1),...,(s1,rm)
    for s in sizes[1:]:
        pairs.append([s,math.sqrt(ratios[0])]) #(s2,r1),...,(s2,rn)
    pairs=np.array(pairs) #转换成ndarray形式

    ss1=pairs[:,0]*pairs[:,1] #计算anchor的宽
    ss2=pairs[:,0]/pairs[:,1] #计算anchor的高

    base_anchors=np.stack([-ss1,-ss2,ss1,ss2],1)/2
    #每一行表示同一个s和r下计算出来的[负宽，负高，正宽，正高]
    #每一列表示不同s和r计算出来的宽或者是高

    h,w=feature_map.shape[-2:] #因为一般feature_map是4Dtensor,取最后的高和宽

    shifts_x=np.arange(0,w)/w #x坐标，计算偏移量,估计是因为，读入的时候就讲图像值标定在了[0,1]之间
    shifts_y=np.arange(0,h)/h #y坐标，计算偏移量
    shift_x,shift_y=np.meshgrid(shifts_x,shifts_y)  #meshgrid将两个矩阵元素的内积分别生成两个对应元素的矩阵,将x和y坐标关联起来，形成矩阵的偏移量
    #结果是shift_x每一行都相同(每一行都是shifts_x中的元素)，shift_y每一列都相同(每一列都是shifts_y中的元素)，(相当于x中的所有元素都和y中的每个元素凑对)
    #相当于上述计算的是每个像素左上角的坐标值

    shift_x=shift_x.reshape(-1) #变成行向量,所以是按行拼接 shift_x[728:728+728]==shift_x[0:728]
    shift_y=shift_y.reshape(-1) #变成行向量
    shifts=np.stack((shift_x,shift_y,shift_x,shift_y),1)

    anchors=shifts.reshape((-1,1,4))+base_anchors.reshape((1,-1,4)) #遵循广播机制
    #这里就是加法，相当于分别以shifts为中心，左右减去上述计算出的宽和高的1/2,上述的1/2也可以解释了，从而形成了左上和右下四个坐标

    return torch.tensor(anchors,dtype=torch.float32).view(1,-1,4)
    #...为什么转换成tensor不会报错，但是直接对anchor.view(1,-1,4)（anchor本身是ndarray）就会显示.view的错误

"""
 #这里就是加法，相当于分别以shifts为中心，左右减去上述计算出的宽和高的1/2,上述的1/2也可以解释了，从而形成了左上和右下四个坐标
返回的anchor的形状为(1,num_anchors,4),将anchor变量y的形状变为(图像高，图像宽，以相同像素为中心的anchor个数，4)后，就可以通过指定像素位置来获取所有该像素中心的anchor了
"""


X=torch.Tensor(1,3,h,w) #torch.tensor不同于torch.Tensor,这里创建的是一个4Dtensor
#...???...为什么这里是0,而不是随机数
#这里是0是随机数并没有太大的用处，只是得到图像的宽和高，然后生成anchor
Y=MultiBoxPrior(X,sizes=[0.75,0.5,0.25],ratios=[1,2,0.5])
print(Y.shape)

boxes=Y.reshape((h,w,5,4))
# boxes=Y.view((h,w,5,4)),#这里定义了一共生成5个anchor，一共有四个坐标，分别为左上和右下
# print(boxes[250,250,0,:]) #为什么会有错...???...

#绘制多个边界框
def show_bboxes(axes,bboxes,labels=None,colors=None):
    def _make_list(obj,default_values=None):
        if obj is None:
            obj=default_values
        elif not isinstance(obj,(list,tuple)):
            obj=[obj]
        return obj
    labels=_make_list(labels)
    colors=_make_list(colors,['b','g','r','m','c'])

    for i,bbox in enumerate(bboxes): #索引一整行
        color=colors[i%len(colors)]

        rect=plt.Rectangle(
            xy=(bbox[0],bbox[1]),
            width=bbox[2]-bbox[0],
            height=bbox[3]-bbox[1],
            fill=False,
            linewidth=2
        )
        axes.add_patch(rect)
        if labels and len(labels)>i:
            text_color='k' if color=='w' else 'w' #相当于三目运算符，如果color==‘w',那么，text_color='k',否则，test_color='w'
            axes.text(rect.xy[0],rect.xy[1],labels[i],
                      va='center',ha='center',fontsize=6,
                      color=text_color,bbox=dict(facecolor=color,lw=0))

#创建要绘制的图像对象
fig=plt.imshow(img)

#anchor尺度
bbox_scale=torch.tensor([[w,h,w,h]],dtype=torch.float32)
"""
注意上述shifts_x和shifts_y中除以了图像的宽和高，因此在绘图的时候，需要恢复anchor的原始坐标值，因此定义了变量bbox_scale
"""

#调用函数
"""
注意这里要恢复anchor的尺度
"""
show_bboxes(
    fig.axes,bboxes=boxes[250,250,:,:]*bbox_scale,
    labels=['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 's=0.5, r=1', 's=0.25, r=1']
)
plt.show()



