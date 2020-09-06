import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from model_ssd300 import SSD300
from datasets import PascalVOCDataset


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

#在一些迭代后衰减学习率
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
        model=SSD300(n_classes=n_classes)

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
    # criterion=MultiBosLoss()

    #创建dataloaders
    #data_folder='./'
    # train_dataset=PascalVOCDataset(data_folder,split='train',keep_difficult=keep_difficult) #前面定义的是keep_difficult=True
    #我不想保存难识别的样本
    train_dataset=PascalVOCDataset(data_folder,split='train') #此时使用默认的False
    train_loader=Data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_fn,num_workers=workers,pin_memory=True)


if __name__=='__main__':
    main()


