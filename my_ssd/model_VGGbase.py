import torch
import torch.nn as nn
import torch.nn.functional as F
# from math import sqrt
import torchvision

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    def load_pretrained_layers(self):
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
    assert tensor.dim()==len(m) #tensor.dim()表示tensor的维度总数,当异常的时候报错

    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor=tensor.index_select(dim=d,index=torch.arange(start=0,end=tensor.size(d),step=m[d]).long())
        #这里传入的tensor的维度是tensor.dim()=4
        #例子, state_dict['conv6.weight']=decimate(conv_fc6_weight,m=[4,None,3,3]) #(4096,512,7,7)。d=0(num_kernel),1(channel),2(h),3(w); m[0]=4,m[1]=None,m[2]=3,m[3]=3
        #d=0; m[0]=4; tensor.size(0)=4096;
        #tensor.index_select(dim=0,index=torch.arange(strat=0,end=4096,step=4)) 就是对num_kernel每四个取一个,4096/4=1024,最终得到1024个num_kernel
        #最终得到的是(1024,512,3,3),因为对原始的tensor已经view过了

    #返回修改过的tensor，这些tensor将作为自定义网络的conv6和conv7的权重和偏置参数
    return tensor

if __name__=='__main__':
    vggbase=VGGBase()
    print(vggbase)