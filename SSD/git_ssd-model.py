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
        self.pool3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True) #对偶数维度使用向上取整的操作 #256*75*75 -> 256*38*38

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


    def forward(self,image):
        """
        前向传播
        :param image: image tensor (n,3,300,300)
        :return: conv4_3,和 conv7下的低层特征图
        """
        out=F.relu(self.conv1_1(image))
        out=F.relu(self.conv1_2(out))
        out=self.pool1(out)

        out=F.relu(self.conv2_1(out))
        out=F.relu(self.conv2_2(out))
        out=self.pool2(out)

        out=F.relu(self.conv3_1(out))
        out=F.relu(self.conv3_2(out))
        out=F.relu(self.conv3_3(out))
        out=self.pool3(out)

        out=F.relu(self.conv4_1(out))
        out=F.relu(self.conv4_2(out))
        out=F.relu(self.conv4_3(out))
        conv4_3_feats=out #(N,512,38,38)
        out=self.pool4(out)

        out=F.relu(self.conv5_1(out))
        out=F.relu(self.conv5_2(out))
        out=F.relu(self.conv5_3(out))
        out=self.pool5(out)

        out=F.relu(self.conv6(out))
        conv7_feats=F.relu(self.conv7(out)) #(N,1024,19,19)

        return conv4_3_feats,conv7_feats


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
        #想原始预训练过的参数，赋值给自定网络的参数
        for i,param in enumerate(param_names[:-4]): #除了6和7参数
            state_dict[param]=pretrained_state_dict[pretrained_param_names[i]]
            #值得注意的是，虽然在自定的VGG基础网络中的self.pool3使用的是向上取整，改变了输出的2D维度，但是并没有改变参数量(weights,bias)

            #将fc6和fc7准换为卷积层
            #fc6
            conv_fc6_weight=pretrained_state_dict['classifier.0.weight'].view(4096,512,7,7) #(4096,512,7,7)
            conv_fc6_bias=pretrained_state_dict['classifier.0.bias'] #(4096)
            state_dict['conv6.weight']=decimate(conv_fc6_weight,m=[4,None,3,3]) #(1024,512,3,3)
            state_dict['conv6.bias']=decimate(conv_fc6_bias,m=[4]) #(1024)

            #fc7
            conv_fc7_weight=pretrained_state_dict['classifier.3.weight'].view(4096,4096,1,1)
            conv_fc7_bias=pretrained_state_dict['classifier.3.bias']
            state_dict['conv7.weight']=decimate(conv_fc7_weight,m=[4,4,None,None])
            state_dict['conv7.bias']=decimate(conv_fc6_bias,m=[4])


            #经过上述操作，已经修改了self.state_dict, 相当于my_net_dict.update(pretrained_dict);
            self.load_state_dict(state_dict) #相当于my_net.load_state_dict(my_net_dict)

            print("\nLoaded base model\n")


def decimate(tensor,m):
    """
    利用因子 m 修改tensor,也就是通过保留每个第m个值来进行降采样

    当将全连接层Fc转换为等价的全卷积层时使用的，但是尺寸更小
    :param tensor: 要修建的tensor
    :param m: #修剪因子列表，对应于tensor的每个维度，None表示对该维度不进行修剪
    :return: 修剪后的tensor
    """
    assert tensor.dim()!=len(m) #tensor.dim()表示tensor的维度总数(这里应该是不等于吧)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor=tensor.index_select(dim=d,index=torch.arange(start=0,end=tensor.size(d),step=m[d].long()))
        #这里传入的tensor的维度是tensor.dim()=4
        #例子, state_dict['conv6.weight']=decimate(conv_fc6_weight,m=[4,None,3,3]) #(4096,512,7,7)。d=0(num_kernel),1(channel),2(h),3(w); m[0]=4,m[1]=None,m[2]=3,m[3]=3
        #d=0; m[0]=4; tensor.size(0)=4096;
        #tensor.index_select(dim=0,index=torch.arange(strat=0,end=4096,step=4)) 就是对num_kernel每四个取一个,4096/4=1024,最终得到1024个num_kernel
        #最终得到的是(1024,512,3,3),因为对原始的tensor已经view过了

    #返回修改过的tensor，这些tensor将作为自定义网络的conv6和conv7的权重和偏置参数
    return tensor

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

        def int_conv2d(self):
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
        self.cl_conv4_3=nn.Conv2d(512,n_boxes['conv11_2']*n_classes,kernel_size=3,padding=1) #->(4*n_classes)*38*38
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


class SSD300(nn.Module):
    """
    封装VGGBase,辅助网络，预测卷积
    """
    def __init__(self,n_classes):
        super(SSD300,self).__init__()
        self.n_classes=n_classes

        self.base=VGGBase() #相当于实例化的基础网络
        self.aux_convs=AuxiliaryConvolutions() #相当于实例化了辅助网络
        self.pred_convs=PredictionConvolutions() #相当于实例化了预测网络 #操作的时候就是硬卷积

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

        aspect_ratios= {'conv4_3': [1.,2.,0.5],
                     'conv7': [1.,2.,3.,0.5,.333],
                     'conv8_2': [1.,2.,3.,0.5,.333],
                     'conv9_2': [1.,2.,3.,0.5,.333],
                     'conv10_2': [1.,2.,0.5],
                     'conv11_2': [1.,2.,0.5]}

        fmaps=list(fmap_dims.keys())

        prior_boxes=[]

        for k,fmap in enumerate(fmaps): #某一特征图
            for i in range(fmap_dims[fmap]): #高(列)
                for j in range(fmap_dims[fmap]): #宽(行)
                    #中心坐标
                    cx=(j+0.5)/fmap_dims[fmap] #除以特征图维度是因为将坐标转换为分数的形式
                    cy=(i+0.5)/fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]: #特征图对应的横纵比,list[]
                        prior_boxes.append([cx,cy,obj_scales[fmap]*sqrt(ratio),obj_scales[fmap]/sqrt(ratio)]) #w=s*sqrt(ratio),h=s/sqrt(ratio)
                        #每个特征图，从左到右，从上到下，在每个位置上生成多个具有不同横纵比的先验框

                        #额外的先验框
                        if ratio==1:
                            #捕捉异常使用的是try/except语句，用来检测try语句中的错误，从而让except语句补货异常信息并处理
                            try: #先执行，异常时交给expect处理
                                additional_scale=sqrt(obj_scales[fmap]*obj_scales[fmaps[k+1]])
                            #对于最后一个特征图，没有下一个特征图
                            except ImportError:
                                additional_scale=1.
                            prior_boxes.append([cx,cy,additional_scale,additional_scale])

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
        norm=conv4_3_feats.pow(2).sum(dim=1,keepdim=True).sqrt() #(n,512,38,38)-> (n,1,38,38)
        conv4_3_feats=conv4_3_feats/norm #(n,512,38,38)
        conv4_3_feats=conv4_3_feats*self.rescale_factors #(n,512,38,38) 广播机制

        #执行辅助卷积，得到高层特征图
        conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats=self.aux_convs(conv7_feats)

        #执行预测卷积（对每个结果位置框预测相对于先验框的偏移量和类别）
        #就是硬卷积
        locs,classes_scores=self.pred_convs(conv4_3_feats,conv8_2_feats,conv9_2_feats,conv10_2_feats,conv11_2_feats)

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







