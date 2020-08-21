# #为anchor标注类别和偏移量
# """
# 将背景设为0，并令从零开始的目标类别的整数索引自加1，(1为狗，2为猫)
#
# anchor and ground-truth的分配，见'9_4_3 pytorch-diDL-cv-anchor2'
# """
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data as Data
# import torchvision.transforms as transforms
# import numpy as np
# import math
# from PIL import Image
# import matplotlib.pyplot as plt
#
#
# #IOU的实现
# def compute_intersection(set_1,set_2):
#     """
#     计算anchor之间的交集
#     :param set_1: (n1,4)大小的tensor，anchor表示为(xmin,ymin,xmax,ymax)
#     :param set_2: (n2,4)大小的tensor，anchor表示为(xmin,ymin,xmax,ymax)
#     :return: set_1中每个box相对于set_2中每个box的交集
#     """
#
#     lower_bounds=torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0))
#     upper_bounds=torch.min(set_1[:,:2].unsqueeze(1),set_2[:
#     intersection_dims=torch.clamp(upper_bounds-,:2].unsqueeze(0))
# lower_bounds,min=0)
#     return intersection_dims[:,:,0]*intersection_dims[:,:,1]
#
# def compute_jaccard(set_1,set_2):
#     """
#     计算anchor之间的IOU
#     :param set_1: 同上
#     :param set_2: 同上
#     :return: set_1中每个box相对于set_2中每个box的IOU
#     """
#
#     intersection=compute_intersection(set_1,set_2)
#
#     areas_set_1=(set_1[:,2]-set_1[:,0]*(set_1[:,3]-set_1[:,1]))
#     areas_set_2=(set_2[:,2]-set_2[:,0]*(set_2[:,3]-set_2[:,1]))
#
#     union=areas_set_1.unsuqeeze(1)+areas_set_2.unsqueeze(0)-intersection
#
#     return intersection/union
#
#
# def assign_anchor(bb,anchor,jaccard_threshold=0.5):
#     """
#     #为每个anchor分配真实的bb,即分配ground-truth
#     :param bb: ground_truth, shape:(nb,4),一共nb个ground-truth
#     :param anchor: 待分配的anchor, shape(na,4) 一共生成了na个anchor
#     :param jaccard_threshold: IOU预先设定的阈值
#     :return: assigned_idx shape:(na,) 每个anchor分配的真实bb(ground-truth)对应的索引，若为分配任何bb(ground-truth)则为-1
#     """
#
#     na=anchor.sape[0]
#     nb=bb.shape[0]
#
#     jaccard=compute_jaccard(anchor,bb).detch().cpu().numpy()