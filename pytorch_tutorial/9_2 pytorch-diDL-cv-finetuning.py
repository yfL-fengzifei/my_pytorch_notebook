#fine-tuning
"""
微调是迁移学习中一种常用技术，微调步骤如下：
1.在源数据集（如ImageNet数据集）上预训练一个网络模型，即源模型
2.创建一个新的模型，即目标模型，该目标模型复制了源模型上除了输出层外的所有模型设计及其参数，
  假设这些模型参数包含了数据集上学习到的知识，且这些知识同样适用于目标数据集。
  还假设源模型的输出层跟源数据集的标签紧密相关，因此在目标模型中不予采用
3.为目标模型添加一个输出大小为目标数据集类别个数的输出层，并堆积初始化该层的模型参数
4.在目标数据集上训练目标模型，从头训练输出层（见文档图），而其余层的参数都是基于源模型的参数微调得到的

*当目标数据集远小于源数据集时，微调有助于提升模型的泛化能力

"""
"""
一些模型在训练和评估时使用不同的模块，如BN，为了可以开关这些模块利用model.train(),model.eval(),(见train(),eval()函数)

"""

#pytoch models and pretrainedmodel，见官方文档
"""
torchvision.models
包括预定义的模型，包括图像分类、像素级语义分割、目标检测、实例分割、人员关键点检测、视频分类

所有预训练模型需要以相同的方式对输入图像进行归一化，也就是3通道RGB图像小批量的数量，(3*H*W)，其中H和W的最小是224,

图像一定要加载到[0,1]范围内，然后利用mean=[0.485,0.456,0.406]和std=[0.229,0.224,0.225]进行正则化
如: normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]) #这应该是是对RGB颜色通道进行归一化操作

**图像分类：
AlexNet,VGG,ResNet,SqueezeNet,DenseNet,InceptionV3,GoogleNet,ShuffleNetv2,MobileNetv2,ResNeXt,Wide ResNet,MNASNet

例子：
import torviszion.models as models

1.利用随机权重创建一个模型，调用的是模型构建器
resnet18=models.resnet18()

2.预训练模型，(利用的是torch.utils.models_zoo)
实例化一个预训练的面模型将会将他的权重下载到缓存目录中，这个目录可以利用TORCH_MODEL_ZOO环境变量来设置(见torch.utils.model_zoo.load_url())
resnet18=models.resnet18(pretrained=True)


**语义分割
FCN ResNet50,FCN ResNet101,DeepLabV3 ResNet50,DeepLavV3 ResNet101
如图像分类一样，所有的预训练模型都需要用想用的方式进行归一化，值得注意的是被训练的图像最小尺寸为520
预训练的模型是在COCO2017上训练的，一共有20个类别(Pascal VOC datasets)，值得注意的是里边有train


**目标检测、实例分割
FasterRCNN_ResNet-50_FPN
MaskRCNN_ResNet-50_FPN

"""

"""
预训练模型
import pretrainedmodels
print(pretrainedmodels.model_names)
"""
"""
不管是torchvision.models 还是pretrainedmodels 默认都会将预训练好的模型参数下载到home目录下.torch文件夹，可以通过TORCH_MODEL_ZOO

注意查看对应模型源码中其定义部分
"""

#ImagFolder
"""
torchvision.datasets.ImageFolder
transform对图像进行预处理
target_transform 对图像类别进行预处理，默认为0,1,2
loader 数据集加载方式，通常默认既可
.classes 返回类别(list形式)
.class_to_idx 返回类别+标签(list形式)
.imgs (img-path,class)tuple的list
.[][] [图像索引]；[0表示图像，1表示标签]
plt.inshow(imgs[0][0])
plt.show()
"""

#pre-process
"""
在训练时，先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入。测试时，将图像的高和宽均缩放为256像素，然后从中裁剪出高宽为224像素的中心区域作为输入，此外对RGB三个颜色通道的数值做标准化：每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出
"""
"""
值得注意的是
在使用与预训练模型时，一定要和预训时使用相同的预处理，
1.如果使用torchvision.models,那就要求: 以相同的方式正则化，也就是mini-batch_size最小为224，图像载入的范围为[0,1]，然后利用mean=[0.485,0.456,0.406]和std=[0.229,0.224,0.225]进行归一化，这里应该是对颜色通道进行归一化
2.如果使用是pretrained-models 见readme, 查看预处理
"""

#transformer learning迁移学习
"""
根据用户数据，将最后全连接层转换成定义的类别
这是fc层就被随机初始化了，但是其他层保存着预训练得到的参数。由于是在很大的ImageNet数据集上预训练的，所以参数已经足够好，因此一般只需要使用较小的学习率来微调这些参数，而fc中的随机初始化参数一般需要更大的学习率来从头训练
pytorch可以方便地对模型的不同部分设置不同的学习参数，
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

#读取所有的图像文件
# train_imgs=ImageFolder('./hotdog/train')
# # print('stop')
# test_imgs=ImageFolder('./hotdog/test')


#执行预处理
normalize=transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
train_augs=transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
test_augs=transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize
])

#
#模型定义和初始化
pretrained_net=models.resnet18(pretrained=True)
#这里模型，将最终的全局平均池化层输出变化成ImageNet数据集上1000类的输出
#查看最后的全连接层
print(pretrained_net.fc)

#fc层被随机初始化了
pretrained_net.fc=nn.Linear(512,2)
print(pretrained_net.fc)
print(pretrained_net.fc.parameters())


output_param=list(map(id,pretrained_net.fc.parameters())) #id函数返回对象对的内存地址，map对对象按元素执行函数
feature_param=filter(lambda p:id(p) not in output_param,pretrained_net.parameters()) #根据函数，对序列执行函数


#因为fc层被随机初始化了，所以一般需要更大的学习率从头训练
#下面对模型的不同部分设置不同的学习参数
#下面将fc的学习率设为以训练过的部分的10倍
lr=0.01
optimizer=optim.SGD([
    {'params':feature_param},
    {'params':pretrained_net.fc.parameters(),'lr':lr*10}],
    lr=lr,weight_decay=0.001
)

#定义评估模型
def evaluate_accuracy(data_iter, net,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

#定义训练函数、
def train(train_iter,test_iter,net,loss,optimizer,num_epochs):
    batch_count=0
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
            batch_count+=1
        test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc))

def train_fine_tuning(net,optimizer,batch_size=128,num_epochs=5):
    train_iter=Data.DataLoader(ImageFolder('./hotdog/train',transform=train_augs),batch_size,shuffle=True)
    test_iter=Data.DataLoader(ImageFolder('./hotdog/test',transform=test_augs),batch_size)
    loss=nn.CrossEntropyLoss() #定义损失函数
    train(train_iter,test_iter,net,loss,optimizer,num_epochs)

train_fine_tuning(pretrained_net,optimizer)

# """
# 作为对比，定义一个相同的模型，但将他的所有模型参数都初始化为随机值，由于整个模型都需要从头训练，因此可以使用较大的学习率
# """
# scratch_net=models.resnet18(pretrained_net=False,num_classes=2)
# lr=0.1
# optimizer=optim.SGD(scratch_net.parameters(),lr=lr,weight_decay=0.001)
# train_fine_tuning(scratch_net,optimizer)




