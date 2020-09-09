import torch
import torch.utils.data as Data
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from datasets import PascalVOCDataset

train_dataset = PascalVOCDataset(data_folder='./', split='train')
# 这里返回的是tuple的形式，每个tuple有image,boxes,labels,difficulties #返回的是一张图像、及其图像中对应的objects (再次强调，这里返回的仅仅是一个样本)
# 注意查看debug中的变量对应起来，这是对的，没有问题
# len(train_datasets)=16551
# len(train_datasets[0])=4 #image,boxes,labels,difficulties

train_loader = Data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn,
                               num_workers=0, pin_memory=True)

# #=========测试1===========================
# # one_batch=iter(train_loader)
# # batch_image,boxes,labels,difficulties=one_batch.next()
# iter_loader = iter(train_loader)
# batch_traindata = next(iter_loader)
# # type(batch_traindata):tuple
# # type(batch_traindata[0]): tensor
# # type(batch_traindata[1])、type(batch_traindata[2])、type(batch_traindata[3]): list
# # len(batch_traindata)=4
# # batch_traindata[0].size()=8 #因为是tensor
# # len(batch_traindata[1])=8 #因为是list
# # len(batch_traindata[2])=8
# # len(batch_traindata[3])=8
#
# # 再来一个batch
# batch_traindata2 = next(iter_loader)
#
# images, boxes, labels, _ = batch_traindata
# images2, boxes2, labels2, _2 = batch_traindata2
#
# print('boxes：', len(boxes))
# print('boxes2：', len(boxes2))
#
# set_1 = boxes2[1][0].unsqueeze(0)  # (1,4)
# set_2 = torch.cat(boxes, 0)  # (22,4)
#
# # 测试一下计算IOU交集
# print(set_1[:, :2], set_1[:, :2].unsqueeze(1), set_1[:, :2].unsqueeze(1).size())
# print(set_2[:, :2], set_2[:, :2].unsqueeze(0), set_2[:, :2].unsqueeze(0).size())
# lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
# print('lower_bounds：', lower_bounds.size())
# upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
# print('upper_bounds：', upper_bounds.size())
# sub = upper_bounds - lower_bounds
# print('sub', sub.size())
# print(sub[:, :, 0], sub[:, :, 0].size())
# print(sub[:, :, 1], sub[:, :, 1].size())


# #=========测试2===========================
#
# import time
#
# class AverageMeter(object):
#     """
#     以data_time为例，每次epoch时都实例化一次data_time=AverageMeter(),自动调用reset
#     然后当对每个Batch执行操作的时候，执行data_time.update(time.time()-start)
#     """
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val=0
#         self.avg=0
#         self.sum=0
#         self.count=0
#
#     def update(self,val,n=1):
#         self.val=val
#         self.sum+=val*n
#         self.count+=n
#         self.avg=self.sum/self.count
#         print(self.val,self.avg)
#
# data_time=AverageMeter() #实例化，执行reset
# start=time.time()
#
# for i,(images,boxes,labels,_) in enumerate(train_loader):
#     data_time.update(time.time()-start)
#     break


print('pass')
