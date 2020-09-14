#注意不要使用test_作为开头进行文件命名

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os

#定义数据增强
#注意这里边已经归一化，totensor() -> [0,1]; normalize() ->[-1,1]
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

trainlodaer=Data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
testlodaer=Data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# #========================测试mean/std广播机制，plt.imshow和ToPILImage
# dataiter=iter(trainlodaer)
# images,labels=dataiter.next()
#
# #执行反归一化
# img=images[0]/2+0.5 # y=(x-mean)/std -> x=y*std+mean
# #因为这里标准差和均值都一样，所以用标量也是一样的
# #注意这里利用广播机制也可以进行相乘和加法，在每个通道乘以和加上不同的方法
# #相当于下面的
# mean=[0.5,0.5,0.5]
# std=[0.5,0.5,0.5]
# mean=torch.tensor(mean).unsqueeze(1).unsqueeze(2)
# std=torch.tensor(std).unsqueeze(1).unsqueeze(2)
# img2=images[0]*std+mean
#
# np_img=img.numpy()
# np_img2=img2.numpy()
#
# plt.figure(1)
# plt.subplot(1,2,1)
# plt.imshow(np.transpose(np_img,(1,2,0)))
# plt.subplot(1,2,2)
# plt.imshow(np.transpose(np_img2,(1,2,0)))
# plt.show()
# print('pass')
#
# # #test 0ne
# # #强烈注意，plt.imshow载入的是[-1,1]之间的数据，*255之后就不对了！！！
# # img=img*255
# # img2=img*255
# # np2_img=img.numpy()
# # np2_img2=img2.numpy()
# #
# # plt.figure(2)
# # plt.subplot(1,2,1)
# # plt.imshow(np.transpose(np2_img,(1,2,0)))
# # plt.subplot(1,2,2)
# # plt.imshow(np.transpose(np2_img2,(1,2,0)))
# # plt.show()
# # print('pass')
#
# #test_three
# #这里也需要强烈注意，ToPILImage()传入的也是[-1,1]之间的tensor
# transform_topil=transforms.ToPILImage()
# pil_img=transform_topil(img)
# pil_img2=transform_topil(img2)
#
# plt.figure(3)
# plt.subplot(1,2,1)
# plt.imshow(pil_img)
# plt.subplot(1,2,2)
# plt.imshow(pil_img2)
# plt.show()
# # print('pass')

#========================继续测试checkpoint
#定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu((self.conv1(x))))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5) #这里-1表示的是Batch
        x=F.relu((self.fc1(x)))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

net=Net()
print(net)
print(net.parameters())
# for param in net.parameters():
#     print(param)
# for name,param in net.named_parameters():
#     print(name)
# for state in net.state_dict():
#     print(state)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(5):

    running_loss=0.0

    #enumerate(data,strat=0) start为返回的下标的起始位置
    for i,data in enumerate(trainlodaer,0):
        inputs,labels=data

        #梯度清零
        optimizer.zero_grad()

        #前向传播
        outputs=net(inputs)
        loss=criterion(outputs,labels)

        #反向传播
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

        if i%2000==1999: #因为这里i是从0开始的，所以要==1999
            print('[%d,%5d] loss: %.3f'%(epoch+1,i+1,running_loss/2000)) #2000次迭代下的平均损失
            running_loss=0.0

    #保存训练节点
    checkpoint_state={'model_state_dict':net.state_dict(),
                      'optimizer':optimizer.state_dict(),
                      'epoch':epoch}
    if not os.path.isdir('./checkpoint_save'):
        # print('can not be open')
        os.mkdir('./checkpoint_save')
    torch.save(checkpoint_state,'./checkpoint_save/checkpoint_node_%s.pth'%(str(epoch)))
    print('checkpoint_node_%s.pth finished'%(str(epoch)))

print('finished training')

# #===================完全训练结束后，保存模型
# PATH='./checkpoint_net.pth'
# torch.save(net.state_dict(),PATH)
#===================完全训练结束后，另一种保存模型的方式
# PATH='./chechpoint_net_v2.pth'
# state={'model_state_dict':net.state_dict()}
# torch.save(state,PATH)

#===================完全训练结束后，保存节点(这段代码是伪代码)
# if not os.path.isdir('./checkpoint_save'):
#     # print('can not be open')
#     os.mkdir('./checkpoint_save')
#     torch.save(state,'./checkpoint_save/xxx')

print('pass')