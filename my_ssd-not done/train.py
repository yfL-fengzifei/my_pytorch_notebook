import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model_ssd300 import SSD300
from datasets import PascalVOCDataset
from train_strategy import *


def create_label_map():
    #lable map
    voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable','dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    #将标签变成整数，值得注意的是label_map是字典的形式
    label_map={k:v+1 for v,k in enumerate(voc_labels)}
    label_map['background']=0

    return voc_labels,label_map #返回原始标签和label字典

#数据参数
data_folder='./'
keep_difficult=True

##模型参数
#类别
_,label_map=create_label_map()
n_classes=len(label_map)
#设备
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

##学习参数
#模型结点的路径
checkpoint=None

#batch_size
batch_size=8

#迭代次数
iterations=120000

#加载数据的进程，好像Windows只能是0
#workers=4
workers=0

#每__batches打印训练的状态
print_freq=200

#学习率
lr=1e-3

#在达到一些迭代后衰减学习率
decay_lr_at=[80000,100000]
#衰减学习率为当前的多少被
decay_lr_to=0.1

#动量
momentum=0.9

#权重衰减
weight_decay=5e-4

#当梯度爆炸是进行一系列的操作，尤其是当在更大的batch_sizes
grad_clip=None

#...???...
# cudnn.benchmark=True

def main():
    """
    训练
    """
    global start_epoch,label_map,epoch,checkpoint,decay_lr_at

    #初始化模型，或这是加载chekpoint(训练到一半的时候)
    if checkpoint is None:
        start_epoch=0
        #(属于初始化的范畴)
        model=SSD300(n_classes=n_classes) #实例化网络 (实例化了VGGBase,aux,pre网络，以及创建了先验框)
        #只有在执行完model(images)返回的才是下面的
        #return locs, classes_scores  # (n,8732,4) (n,8732,n_classes)

        #初始化优化器，后面怎么翻译...???...
        biases=list()
        not_biases=list()
        for param_name,param in model.named_parameters(): #查看参数
            if param.requires_grad:
                #偏置和权重可学习参数分开
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        #为不同的可学习参数，设置不同的学习率
        optimizer=torch.optim.SGD(params=[{'params':biases,'lr':2*lr},{'params':not_biases}],lr=lr,momentum=momentum,weight_decay=weight_decay)
    else:
        #如果要继续训练(属于初始化的范畴)
        checkpoint=torch.load(checkpoint,map_location='cpu') #因为此时没有cuda,所以要这样写
        start_epoch=checkpoint['epoch']+1
        print('\nLoaded checkpoint from epoch %d.\n'%start_epoch)
        model=checkpoint['model']
        optimizer=checkpoint['optimizer']

    #移动到默认的装置
    model=model.to(device)
    criterion=MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device) #这里传入的是创建的先验框
    # priors_cxcy 传入的是之前 return prior_boxes #(8732,4)，中心坐标形式
    #这里将MultiboxLoss也当做是一个网络模型来进行实例化

    #创建dataloaders
    #data_folder='./'
    # train_dataset=PascalVOCDataset(data_folder,split='train',keep_difficult=keep_difficult) #前面定义的是keep_difficult=True
    #我不想保存难识别的样本
    train_dataset=PascalVOCDataset(data_folder,split='train') #此时使用默认的False
    train_loader=Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=workers,pin_memory=True)
    #pin_memory设置锁页内存，当计算机内存充足是设置为True,内存不足时设置为False
    #这里dataloader返回的是batch_size tuple, image为batch_size tensor,boxes\labels\difficulties为batch_size list,list中每个元素都是batch_size,tensor

    #计算训练的epoch的总数，...???...
    #文章设置的batch_size=32, 训练了120000次迭代，在80000和100000次迭代之后衰减
    #//整数除法
    epochs=iterations//(len(train_dataset)//32) #len(train_dataset)=16651，120000次迭代，计算一共有多少次epochs历元
    decay_lr_at=[it//(len(train_dataset)//32) for it in decay_lr_at] #在达到一些迭代后衰减学习率 decay_lr_at=[80000,100000],即在[154, 193]历元下，进行权重衰减

    #历元
    #epochs是历元的总数(不管有没有训练完，一共就这么多次，即如果是中间不训练了即checkpoint，继续开始)
    for epoch in range(start_epoch,epochs):

        #在特定的epochs历元下执行学习率衰减
        #在[154, 193]历元下，先进行权重衰减，在进行训练
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer,decay_lr_to) #还没看

        #如果每达到需要调整学习率的地方直接进行训练
        train(train_loader=train_loader,model=model,criterion=criterion,optimizer=optimizer,epoch=epoch)

        #保存节点
        save_checkpoint(epoch,model,optimizer) #感觉这里是将模型、优化器直接进行保存，感觉这样不好
        #这里试着自己写一版
        # save_checkpoint2(epoch,model,optimizer)

def train(train_loader,model,criterion,optimizer,epoch):
    """
    一个epoch
    :param train_loader: 训练数据
    :param model: 模型
    :param criterion: multibox 损失
    :param optimizer: 优化器
    :param epoch: epoch 数量
    """
    model.train() #在训练模式下可以使用dropout
    batch_time=AverageMeter() #前项传播时间+反向传播时间
    data_time=AverageMeter() #数据加载的时间
    losses=AverageMeter() #loss

    start=time.time()

    #对batch执行操作
    for i,(images,boxes,labels,_) in enumerate(train_loader):
        data_time.update(time.time()-start) #即每个batch执行一次更新,其实就是加载一个batch所用的时间

        #移动到默认的设备上
        images=images.to(device)
        boxes=[b.to(device) for b in boxes] #因为boxes是一个list 一共有batch_size个元素，每个元素是一个tensor
        labels=[l.to(device) for l in labels]

        #执行前项传播
        predicted_locs,predicted_scores=model(images) #最终是经过硬卷积得来的

        #计算损失
        loss=criterion(predicted_locs,predicted_scores,boxes,labels) #最终得到的是标量，见Multiboxloss 相当于执行了MultiboxLoss.forward

        #反向传播
        optimizer.zero_grad()
        loss.backward()

        #如果必要的时候，进行梯度修剪 还没看...???...
        #pass

        #更新模型
        optimizer.step()

        losses.update(loss.item(),images.size(0)) #对损失标量进行更新,images.size(0)=batch_size,实际上求的是当前loss和之前loss的平均值
        batch_time.update(time.time()-start) #整个batch下运行的时间

        start=time.time()

        #打印状态
        #每print_freq个batch打印一次状态
        if i % print_freq==0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch,i,len(train_loader),batch_time=batch_time,data_time=data_time,loss=losses))

    del predicted_locs,predicted_scores,images,boxes,labels #释放内存


def save_checkpoint(epoch,model,optimizer):
    """
    保存模型节点
    :param epoch: epoch数量
    :param model: 模型
    :param optimizer: 优化器
    """
    state={'epoch':epoch,'model':model,'optimizer':optimizer}
    filename='checkpoint_ssd300.pth.tar'
    torch.save(state,filename) #以字典得到形式进行保存


# def save_checkpoint2(epoch,model,optimizer):
#     state={'model_state_dict':model.state_dict(),
#            'optimizer_state_dict':optimizer.state_dict(),
#            'epoch':epoch}
#     PATH='checkpoint_ssd300.tar'
#     torch.save(state,PATH)

if __name__=='__main__':
    main()


