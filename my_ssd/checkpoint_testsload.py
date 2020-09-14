import torch
import torch.nn as nn
import torch.nn.functional as  F
import torch.optim as optim
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

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

#很重要，在继续训练的时候，线实例化网络、优化器等
net=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
# PATH='./checkpoint_net.pth'
# checkpoint=torch.load(PATH)
# net.load_state_dict(checkpoint)
#加载第二个版本
# PATH='./chechpoint_net_v2.pth'
# checkpoint=torch.load(PATH)
# net.load_state_dict(checkpoint['model_state_dict'])
#加载第三个版本
PATH='./checkpoint_save/checkpoint_node_1.pth'
checkpoint=torch.load(PATH)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_epoch=checkpoint['epoch']+1

#将网络设置为测试模型
net.eval()

#加载测试数据集
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
trainlodaer=Data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=0)
testlodaer=Data.DataLoader(testset,batch_size=4,shuffle=False,num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# #=========================================下面是前两个加载版本的测试
# testiter=iter(testlodaer)
# # images,labels=next(testiter)
# images,labels=testiter.next()
#
# #取一各样本
# img=images[1]
#
# #反归一化
# mean=[0.5,0.5,0.5]
# std=[0.5,0.5,0.5]
# mean=torch.tensor(mean).unsqueeze(1).unsqueeze(2)
# std=torch.tensor(std).unsqueeze(1).unsqueeze(2)
# img2=img*std+mean
# # img2_pil=transforms.ToPILImage(img2) #注意，直接这样写是不行的，要写成下面得到形式
# trans_topil=transforms.ToPILImage()
# img2_pil=trans_topil(img2)
# plt.imshow(img2_pil)
# plt.show()
#
# #测试
# output=net(img.unsqueeze(0))
# print(output)
#
# value,predicted=torch.max(output,1)
# print('value:',value,'predicted:',predicted)
# print('class:',classes[predicted])

#=========================================下面是继续训练
for epoch in range(start_epoch,6):

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

testiter=iter(testlodaer)
# images,labels=next(testiter)
images,labels=testiter.next()

#取一各样本
img=images[1]

#反归一化
mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]
mean=torch.tensor(mean).unsqueeze(1).unsqueeze(2)
std=torch.tensor(std).unsqueeze(1).unsqueeze(2)
img2=img*std+mean
# img2_pil=transforms.ToPILImage(img2) #注意，直接这样写是不行的，要写成下面得到形式
trans_topil=transforms.ToPILImage()
img2_pil=trans_topil(img2)
plt.imshow(img2_pil)
plt.show()

#测试
output=net(img.unsqueeze(0))
print(output)

value,predicted=torch.max(output,1)
print('value:',value,'predicted:',predicted)
print('class:',classes[predicted])

print('pass')



