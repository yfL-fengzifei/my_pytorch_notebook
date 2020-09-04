import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import torchvision

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
基础卷积操作，output=(Dinput+2pad-d(Dkernel-1)-1)/stride + 1
#nn.Conv2d()
in_channels:
out_channels:
kernel_size:
stride:默认等于1
padding:默认等于0(both side)
padding_mode: 默认为‘zeros’
dilation:默认为1
bias:增加一个可学习的参数，默认为True
groups：实现类似于分组卷积的工作，given groups=n,要求groups能被in_channels和out_channels整除，相当于将in_channels/groups,将out_channels/groups, given input:[n,32,H1,W1],kernel_size=3*3,out=[n,48,H2,W2],则当groups=1是，weights参数为(n)*48*32*3*3; 当groups=2时，weights参数为48*16*3*3 因为32/2=16,实际上这里每组in_channels被重复利用了out_channels/groups次
"""

"""
加载预训练的模型
只需要网络结构，不需要用预训练的参数来初始化，则model=torchvision.models.certain_network(pretrained=False)

对于具体的任务，很难保证模型和公开的模型完全一样，但是预训练模型的参数确实有助于提高训练的准确率，为了结合二者的有点，需要加载部分预训练模型
pretrained_model=torchvision.models.certain_network(pretrained=True)
model=Net(pass) #自己定义的网络

读取参数
pretrained_dict=pretrained_model.state_dict()
model_dict=model.state_dict()

将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict={k:v for k,v in pretrained_dict.items() if k in model_dict}

#更新现有的module_dict
model_dict.update(pretrained_dict)

#加载真正需要的state_dict
model.load_state_dict(model_dict)

"""


class VGGBase(nn.Module):
    """
    VGG 基础网络 卷积来生成低层次特征图
    """
    def __init__(self):
        super(VGGBase,self).__init__()

        #VGG16中标准的卷积层配置

        self.conv1_1=nn.Conv2d(3,64,kernel_size=3,padding=1) #3*300*300 -> 64*300*300
        self.conv1_2=nn.Conv2d(64,64,kernel_size=3,padding=1) #64*300*300 -> 64*300*300
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2) #64*300*300 -> 64*150*150

        self.conv2_1=nn.Conv2d(64,128,kernel_size=3,padding=1) #64*150*150 -> 128*150*150
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3,padding=1) #128*150*150 ->128*150*150
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2) #128*150*150 -> 128*75*75

        self.conv3_1=nn.Conv2d(128,256,kernel_size=3,padding=1) #128*75*75 -> 256*75*75
        self.conv3_2=nn.Conv2d(256,256,kernel_size=3,padding=1) #256*75*75 -> 256*75*75
        self.conv3_3=nn.Conv2d(256,256,kernel_size=3,padding=1) #256*75*75 -> 256*75*75
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=,ceil_mode=True) #对偶数维度使用向上取整的操作 #256*75*75 -> 256*38*38

        self.conv4_1=nn.Conv2d(256,512,kernel_size=3,padding=1) #256*38*38 -> 512*38*38
        self.conv4_2=nn.Conv2d(512,512,kernel_size=3,padding=1) #512*38*38 -> 512*38*38
        self.conv4_3=nn.Conv2d(512,512,kernel_size=3,padding=1) #512*38*38 -> 512*38*38
        self.pool4=nn.MaxPool2d(kernel_size=2,stride=2) #512*38*38 -> 512*19*19

        self.conv5_1=nn.Conv2d(512,512,kernel_size=3,padding=1) #512*19*19 -> 512*19*19
        self.conv5_2=nn.Conv2d(512,512,kernel_size=3,padding=1) #512*19*19 -> 512*19*19
        self.conv5_3=nn.Conv2d(512,512,kernel_size=3,padding=1) #512*19*19 -> 512*19*19
        self.pool5=nn.MaxPool2d(kernel_size=3,stride=1,padding=1) #512*19*19 -> 512*19*19

        #替换VGG16中FC6和FC7层
        self.conv6=nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6) #512*19*19 -> 1024*19*19
        self.conv7=nn.Conv2d(1024,1024,kernel_size=1) #1024*19*19 -> 1024*19*19

        #加载预训练的层
        self.load_pretrained_layers()


    def load_pretained_layers(self):
        """
        如原文一样，将在ImageNet任务中预训练的VGG16网络作为基础网络
        其在pytorch中是可获得的,并在其上进行了修改
        """
        #当前基础网络的参数
        state_dict=self.state_dict()
        param_names=list(state_dict.keys()) #列出(当前基础网络(自定义网络))参数键值的名字

        #预训练的VGG网络
        pretrained_state_dict=torchvision.models.vgg16(pretrained=True).state_dict() #预训练网络的参数
        pretrained_param_names=list(pretrained_state_dict.keys()) #列出预训练网络的参数的键值的名字

        #从预训练的网络中的参数迁移到当前的模型中
        for i,param in enumerate(param_names[:-4]): #除了6和7参数
            state_dict[param]=pretrained_state_dict[pretrained_param_names[i]]