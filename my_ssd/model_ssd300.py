from model_VGGbase import *
from model_aux import *
from model_pre import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torchvision

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSD300(nn.Module):
    """
    封装VGGBase,辅助网络，预测卷积
    """
    def __init__(self,n_classes):
        super(SSD300,self).__init__()
        self.n_classes=n_classes

        self.base=VGGBase() #相当于实例化的基础网络
        self.aux_convs=AuxiliaryConvolutions() #相当于实例化了辅助网络
        self.pred_convs=PredictionConvolutions(n_classes) #相当于实例化了预测网络 #操作的时候就是硬卷积

        #因为低层次特征图(conv4_3)有相当大的尺度，因此利用L2正则化和重缩放，重缩放因子初始设置为20，但是在反向传播中每个通道中的重缩放因子是可以学习的
        self.rescale_factors=nn.Parameter(torch.FloatTensor(1,512,1,1)) #因为要学习，所以在__init__中预测
        nn.init.constant_(self.rescale_factors,20)

        #创建先验框
        self.priors_cxcy=self.create_prior_boxes()


    def create_prior_boxes(self):
        """
        对模型创建8732个先验框(默认框)
        :return: 以中心坐标形式返回的先验框, tensor (8732,4)
        """
        fmap_dims={'conv4_3':38,
                   'conv7':19,
                   'conv8_2':10,
                   'conv9_2':5,
                   'conv10_2':3,
                   'conv11_2':1}

        obj_scales= {'conv4_3': 0.1,
                     'conv7': 0.2,
                     'conv8_2': 0.375,
                     'conv9_2': 0.55,
                     'conv10_2': 0.725,
                     'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps=list(fmap_dims.keys())

        prior_boxes=[]

        for k,fmap in enumerate(fmaps): #某一特征图
            for i in range(fmap_dims[fmap]): #高(列)
                for j in range(fmap_dims[fmap]): #宽(行)

                    #中心坐标形式
                    cx=(j+0.5)/fmap_dims[fmap] #除以特征图维度是因为将坐标转换为分数的形式
                    cy=(i+0.5)/fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]: #特征图对应的横纵比,list[]
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio),
                                            obj_scales[fmap] / sqrt(ratio)])  # w=s*sqrt(ratio),h=s/sqrt(ratio)
                        #每个特征图，从左到右，从上到下，在每个位置上生成多个具有不同横纵比的先验框

                        #额外的先验框
                        if ratio==1:
                            #捕捉异常使用的是try/except语句，用来检测try语句中的错误，从而让except语句补货异常信息并处理
                            try: #先执行，异常时交给expect处理
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            #对于最后一个特征图，没有下一个特征图
                            except IndexError:
                                additional_scale=1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes=torch.FloatTensor(prior_boxes).to(device)

        prior_boxes.clamp_(0,1) #加紧到0,1之间，即超出图像边界的被修剪

        #值得注意的是，从一开始就是以小数形式表示的先验框坐标(中心点的形式)
        return prior_boxes #(8732,4)


    def forward(self,image):
        """
        前向传播
        :param image: tensor (n,3,300,300)
        :return: 每个图像生成8732个位置和得分
        """

        #执行VGG网络，得到低层特征图
        conv4_3_feats,conv7_feats=self.base(image)

        #对conv4_3进行操作
        #这么做的意义没懂...???...
        norm=conv4_3_feats.pow(2).sum(dim=1,keepdim=True).sqrt() #(n,512,38,38)-> (n,1,38,38)
        conv4_3_feats=conv4_3_feats/norm #(n,512,38,38)
        conv4_3_feats=conv4_3_feats*self.rescale_factors #(n,512,38,38) 广播机制

        #执行辅助卷积，得到高层特征图
        conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats=self.aux_convs(conv7_feats)

        #执行预测卷积（对每个结果位置框预测相对于先验框的偏移量和类别）
        #就是硬卷积
        locs,classes_scores=self.pred_convs(conv4_3_feats,conv7_feats,conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats)

        return locs,classes_scores #(n,8732,4) (n,8732,n_classes)


    def detect_objects(self,predicted_locs,predicted_scores,min_score,max_overlap,top_k):
        """
        将SSD300的输出，即8732个位置输出(相对于先验框的偏移量)和类别得分，将其解码，从而检测目标

        对于每个类别，执行NMS

        :param predicted_locs: 预测到的相对于先验框的位置框(偏移量，其实就是4个值)，tensor (n,8732,4)
        :param predicted_scores: 每个进过编码后的位置框(偏移量)下的类别得分，tensor (n,8732,n_classes)
        :param min_score: 对于某个特定类别的，box与之匹配的最小得分阈值
        :param max_overlap: 两个boxes之间的最大IOU，低于这个值得不会被进行NMS抑制
        :param top_k: 如果所有的类别都出现了，那么值保留前k类目标
        :return: 检测结果(位置框，标签，得分) list
        """
        batch_size=predicted_locs.size(0)
        n_priors=self.priors_cxcy.size(0)
        predicted_scores=F.softmax(predicted_scores,dim=2)
        pass

# if __name__=='__main__':
# ssd300 = SSD300(20)
# print(ssd300)
# # for name,param in ssd300.named_parameters():
# #     print(name)
# # 查看参数
# biases = list()
# not_biases = list()
# for param_name, param in ssd300.named_parameters():
#     # print(param_name,param.size())
#     if param.requires_grad:
#         if param_name.endswith('.bias'):
#             biases.append(param)
#         else:
#             not_biases.append(param)
#
# lr = 1e-3
#
# momentum = 0.9
#
# weight_decay = 5e-4
#
# # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr,
# #                             momentum=momentum, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(params=ssd300.parameters(), lr=lr,
#                             momentum=momentum, weight_decay=weight_decay)
#
# # 优化器是按组输出的，根据你的设置有关
# print(optimizer)
