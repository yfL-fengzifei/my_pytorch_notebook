import time
import torch
import torch.optim as optim
import torch.utils.data as Data
from git_ssd-datasets import PascalVOCDataset
from git_ssd-transform import SSD300,MultiBosLoss

#数据参数
data_folder='./'
keep_difficult=True

def main():
    """
    训练
    """
    global start_epoch,label_map,epoch,checkpoint,decay_lr_at

    #初始化模型或加载检测点
    if checkpoint is None:
        start_epoch=0
        model=SSD300(n_classes=n_classes)

        #初始化优化器，
        biases=list()
        not_biases=list()
        for param_name,param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer=optim.SGD(params=[{'param':biases,'lr':2*lr},{'param':not_biases}],lr=lr,momentum=momentum,weight_decay=weight_decay)



def trian(train_loader,model,criterion,optimizer,epoch):
    """
    一次历元的训练过程
    :param train_loader: 训练数据的dataloader
    :param model: 模型
    :param criterion: multibox loss
    :param optimizer: 优化器
    :param epoch: 历元次数
    """
    pass


if __name__=='__main__':
    main()

