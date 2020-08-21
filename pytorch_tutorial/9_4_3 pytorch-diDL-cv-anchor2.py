#标注训练集的anchor
"""
在训练集中，将每个anchor作为一个训练样本，为了训练目标检测模型，需要为每个anchor标注两类标签：
1.anchor所含目标的类别，简称类别
2.真实边界框ground truth相对anchor的偏移量，偏移量offset,

在目标检测时，
1.首先生成多个anchor,
2.然后为每个anchor预测类别以及偏移量，
3.接着根据预测的偏移量调整anchor位置从而得到预测边界框，
4.最终筛选需要输出的预测边界框

在目标检测的训练集中，每个图像已经标注了真实边界框ground truth的位置以及所含目标的类别；
在生成anchor之后，主要依据与anchor相似的真实边界框ground truth的位置和类别信息为anchor标注，
问题：该如何为anchor分配与其相似的真实边界框ground truth呢

假设图像中的anchor分别为A1，A2,...,Ana；真实边界框ground truth分别为B1,B2,...,Bnb,且na>=nb(即anchor的数量大于ground truth的数量)
定义矩阵X属于R-na*nb,其中第i行第j列的元素xij为anchor Ai与ground-truth Bj 的IOU
1.首先，找出矩阵X中最大元素，并将该元素的行索引和列索引分别记为i1,j1
为anchor Ai1分配ground-truth Bj1; 显然，anchor Ai1 和ground-truth Bj1 在所有'anchor and ground-truth'中相似度最高
2.接下来，将矩阵X中第i1行和第j1列上所有的元素丢弃，(我觉得这里是因为Bj1和Ai1互为最佳匹配，因此不需要再进行选择)
找出矩阵X中剩余的最大元素，并将该元素的行索引和列索引分别记为i2,j2，
为anchor Ai2分配ground-turth Bj2 再讲矩阵中第i2行和第j2列所有元素丢弃
3.此时矩阵X中已有两行、两列元素被丢弃，
以此类推，知道矩阵X中所有nb列元素全部被丢弃，(因为一般anchor的数量大于ground-truth)
这个时候，已经为nb个anchor各分配了一个 ground-truth,
4.接下来，只需要遍历剩余的na-nb个anchor：
给定其中的anchor Ai ,根据矩阵X的第i行找到与Ai IOU最大的ground-truth Bj(值得注意的是，这里X和Bj用的应该是未丢弃元素之前的矩阵和元素),
且只有当该IOU大于预先设定的阈值时，才为anchor Ai分配真实边界框Bj

现在可以标注anchor的类别class和偏移量offset
如果一个anchor A 被分配了ground-truth B，将anchor A的类别设为B的类别，并根据B和A的中心坐标的相对位置以及两个框的相对大小为anchor A标注偏移量
有数据集中各个框的位置和大小各异，因此这些相对位置和相对大小通常需要一些特殊变换，才能使偏移量的分布更加均匀，从而更容易拟合
设anchor A及其分配的ground-truth B的中心坐标分别为(xa,ya)和(xb,yb); A 和 B的宽分别为wa和wb; 高分别为ha和hb； 一个常用的技巧是将A的偏移量标注为：
公式见文档，其中常数的默认值为 (mu)_x=(mu)_y=(mu)_w=(mu)_h=0, (sigma)_x=(sigma)_y=0.1, (sigma)_w=(sigma)_h=0.2
如果一个anchor没有被分配真实边界框ground-truth ,那么只需要将该anchor的类别设为背景，类别为背景的anchor通常被称为父类anchor,其余被称为正类anchor


"""

#sample
"""
读取图像中猫和狗的真实边界框，其中第一个二元素为类别(0为狗，1为猫)，剩余四个元素分别为左上角的x和y，以及右下角的x和y,(值域在[0,1])
此时通过左上角和右下角的坐标够早了5个需要标注的anchor,分别记为A0,...,A4

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

#加载和显示图像
img=Image.open('./catdog.jpg')
plt.imshow(img)
plt.show()
plt.close()

print(img.size) #值得注意的是，这里返回的是w*h
w,h=img.size

#画出这些anchor与ground truth在图像中的位置
bbox_scale=torch.tensor((w,h,w,h),dtype=torch.float32)

#ground_truth
#每一行表示的是图像中一个目标的ground-truth:[class,(bbox_coordinate)],注意需要尺度的恢复
ground_truth=torch.tensor([[0,0.1,0.08,0.52,0.92],
                           [1,0.55,0.2,0.9,0.88]])


#定义anchor
#anchor只是坐标
anchors=torch.tensor(
    [[0,0.1,0.2,0.3],
     [0.15,0.2,0.4,0.4],
     [0.63,0.05,0.88,0.98],
     [0.66,0.45,0.8,0.8],
     [0.57,0.3,0.92,0.9]]
)

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

fig=plt.imshow(img)

#绘制ground-truth and anchor
show_bboxes(fig.axes,ground_truth[:,1:]*bbox_scale,['dog','cat'],'k')
show_bboxes(fig.axes,anchors*bbox_scale,['0','1','2','3','4'])

plt.show()


