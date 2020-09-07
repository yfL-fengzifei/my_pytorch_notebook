import torch
import torch.utils.data as Data
import warnings

"""
测试一下创建的数据集有没有问题
"""
warnings.filterwarnings("ignore",category=UserWarning)
from datasets import PascalVOCDataset

train_dataset=PascalVOCDataset(data_folder='./',split='train')
#这里返回的是tuple的形式，每个tuple有image,boxes,labels,difficulties #返回的是一张图像、及其图像中对应的objects
#注意查看debug中的变量对应起来，这是对的，没有问题
#len(train_datasets)=16551
#len(train_datasets[0])=4 #mage,boxes,labels,difficulties

train_loader = Data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn,num_workers=0, pin_memory=True)
#one_batch=iter(train_loader)
#batch_image,boxes,labels,difficulties=one_batch.next()

print('pass')