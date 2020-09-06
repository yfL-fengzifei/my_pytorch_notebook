import torch
import torch.nn as nn
# import torch.nn.functional as F
# from math import sqrt
# import torchvision

class PredictionConvolutions(nn.Module):
    """
    利用低层和高层特征图来预测类别得分和Bbox

    bbox(位置)以编码后的偏移量的方式(给出)进行预测，该偏移量相对于8732个先验框(默认框)中的每一个
    ‘cxcy_to_gcxgcy’

    类别得分表示的是，8732个定位到的bbox的每个目标的类别得分,注意这里不是8732个先验框中的目标而是，经过预测偏移量后的Bbox中的目标的类别
    """
    def __init__(self,n_classes):
        """
        :param n_classes: 目标的类别数量
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes=n_classes

        #每个特征图下每个位置定义的先验框的数量
        #number表示使用n个横纵比
        n_boxes={'conv4_3':4,
                 'conv7':6,
                 'conv8_2':6,
                 'conv9_2':6,
                 'conv10_2':4,
                 'conv11_2':4}

        #位置预测卷积，预测的是偏移量，
        self.loc_conv4_3=nn.Conv2d(512,n_boxes['conv4_3']*4,kernel_size=3,padding=1) #->16*38*38
        self.loc_conv7=nn.Conv2d(1024,n_boxes['conv7']*4,kernel_size=3,padding=1) #->24*19*19
        self.loc_conv8_2=nn.Conv2d(512,n_boxes['conv8_2']*4,kernel_size=3,padding=1)
        self.loc_conv9_2=nn.Conv2d(256,n_boxes['conv9_2']*4,kernel_size=3,padding=1)
        self.loc_conv10_2=nn.Conv2d(256,n_boxes['conv10_2']*4,kernel_size=3,padding=1)
        self.loc_conv11_2=nn.Conv2d(256,n_boxes['conv11_2']*4,kernel_size=3,padding=1)

        #分类预测卷积，预测位置框(先验框)中的目标的类别
        self.cl_conv4_3=nn.Conv2d(512,n_boxes['conv4_3']*n_classes,kernel_size=3,padding=1) #->(4*n_classes)*38*38
        self.cl_conv7=nn.Conv2d(1024,n_boxes['conv7']*n_classes,kernel_size=3,padding=1)
        self.cl_conv8_2=nn.Conv2d(512,n_boxes['conv8_2']*n_classes,kernel_size=3,padding=1)
        self.cl_conv9_2=nn.Conv2d(256,n_boxes['conv9_2']*n_classes,kernel_size=3,padding=1)
        self.cl_conv10_2=nn.Conv2d(256,n_boxes['conv10_2']*n_classes,kernel_size=3,padding=1)
        self.cl_conv11_2=nn.Conv2d(256,n_boxes['conv11_2']*n_classes,kernel_size=3,padding=1)

        #初始化卷积参数
        self.init_conv2d()

    def init_conv2d(self):
        """
        初始化卷积参数
        """
        for c in self.children():
            if isinstance(c,nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias,0.)

    def forward(self,conv4_3_feats,conv7_feats,conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats):
        """
        前向传播
        :param conv4_3_feats: a tensor of dimensions (N, 512, 38, 38)
        :param conv7_feats: a tensor of dimensions (N, 1024, 19, 19)
        :param conv8_2_feats: a tensor of dimensions (N, 512, 10, 10)
        :param conv9_2_feats: a tensor of dimensions (N, 256, 5, 5)
        :param conv10_2_feats: a tensor of dimensions (N, 256, 3, 3)
        :param conv11_2_feats: a tensor of dimensions (N, 256, 1, 1)
        :return: 8732个Bbox和对应的得分，即，每个图像中相对于每个先验框的偏移量(预测后的Bbox以偏移量的方式给出，以及Bbox中的目标的得分)
        """

        batch_size=conv4_3_feats.size(0)

        #这里边预测位置框(相对于先验框的偏移量)没什么规则，就是硬(传统)卷积

        #预测定位框(bbox)的边界(即相对于先验框的偏移量)
        l_conv4_3=self.loc_conv4_3(conv4_3_feats)
        l_conv4_3=l_conv4_3.permute(0,2,3,1).contiguous() #tensor.permute()转换维度，tensor.contiguous()保证tensor是一块连续的内存,便于后面的.view(); 原来是(n,16*38*38),现在是(n,38*38*16)
        l_conv4_3=l_conv4_3.view(batch_size,-1,4) #(N,5776,4) 一共有5776个boxes

        l_conv7=self.loc_conv7(conv7_feats) #(N,24*19*19)
        l_conv7=l_conv7.permute(0,2,3,1).contiguous() #(N,19*19*24)
        l_conv7=l_conv7.view(batch_size,-1,4) #(n,2166,4) 一共有2166个boxes

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        #预测位置框中的目标类别
        c_conv4_3=self.cl_conv4_3(conv4_3_feats) #(n,4*n_classes,38,38)
        c_conv4_3=c_conv4_3.permute(0,2,3,1).contiguous() #(n,38,38,4*n_classes)
        c_conv4_3=c_conv4_3.view(batch_size,-1,self.n_classes)

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1,self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        #一共8732个boxes
        #用特定的顺序连接 (必须与先验框的顺序进行匹配)
        locs=torch.cat([l_conv4_3,l_conv7,l_conv8_2,l_conv9_2,l_conv10_2,l_conv11_2],dim=1) #(N,8732,4)
        class_scores=torch.cat([c_conv4_3,c_conv7,c_conv8_2,c_conv9_2,c_conv10_2,c_conv11_2],dim=1) #(N,8732,n_classses)

        return locs,class_scores

if __name__=='__main__':
    preConv=PredictionConvolutions(20)
    print(preConv)