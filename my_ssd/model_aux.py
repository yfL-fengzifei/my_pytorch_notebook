# import torch
import torch.nn as nn
import torch.nn.functional as F
# from math import sqrt
# import torchvision

class AuxiliaryConvolutions(nn.Module):
    """
    辅助卷积来生成高层次特征
    """
    def __init__(self):
        super(AuxiliaryConvolutions,self).__init__()

        #注意默认stride=1
        self.conv8_1=nn.Conv2d(1024,256,kernel_size=1,padding=0) #1024*19*19 -> 256*19*19
        self.conv8_2=nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1) #256*19*19 -> 512*10*10

        self.conv9_1=nn.Conv2d(512,128,kernel_size=1,padding=0) #512*7*7 -> 128*10*10
        self.conv9_2=nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1) #128*10*10 -> 256*5*5

        self.conv10_1=nn.Conv2d(256,128,kernel_size=1,padding=0) #256*5*5-> 256*5*5
        self.conv10_2=nn.Conv2d(128,256,kernel_size=3,padding=0) #256*5*5 -> 256*3*3

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0) #256*3*3 -> 128*3*3
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) #128*3*3 -> 256*1*1

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

    def forward(self,conv7_feats):
        """
        前向传播
        :param conv7_feats: 将conv7_feats特征图作为辅助网络的参数,低层次特征图，tensor (N,1024,19,19)
        :return: 更高层次的特征图，conv8_2,conv9_2,conv10_2,conv11_2
        """
        out=F.relu(self.conv8_1(conv7_feats))
        out=F.relu(self.conv8_2(out))
        conv8_2_feats=out

        out=F.relu(self.conv9_1(out))
        out=F.relu(self.conv9_2(out))
        conv9_2_feats=out

        out=F.relu(self.conv10_1(out))
        out=F.relu(self.conv10_2(out))
        conv10_2_feats=out

        out=F.relu(self.conv11_1(out))
        conv11_2_feats=F.relu(self.conv11_2(out))

        return conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats

if __name__=='__main__':
    auxConv=AuxiliaryConvolutions()
    print(auxConv)
