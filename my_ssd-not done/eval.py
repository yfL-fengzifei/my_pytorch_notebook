import torch
import torch.optim as optim
import torch.utils.data as Data
from model_ssd300 import SSD300
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

pp=PrettyPrinter()

#====================第一个版本,对应于save_checkpoint()
#====================第二个版本见最下面
# def save_checkpoint(epoch,model,optimizer):
#     """
#     保存模型节点
#     :param epoch: epoch数量
#     :param model: 模型
#     :param optimizer: 优化器
#     """
#     state={'epoch':epoch,'model':model,'optimizer':optimizer}
#     filename='checkpoint_ssd300.pth.tar'
#     torch.save(state,filename) #以字典得到形式进行保存
data_folder='./'
keep_difficult=True #pass

batch_size=64
# workers=4
workers=0
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载节点
checkpoint='./checkpoint_ssd300.pth.tar'
checkpoint=torch.load(checkpoint,map_location='cpu')
model=checkpoint['model'] #这里能直接加载吗...???...
model=model.to(device)

#将模型转换为训练模式
model.eval()

#加载测试集
test_dateset=PascalVOCDataset(data_folder,split='test',keep_difficult=keep_difficult)
test_loader=Data.DataLoader(test_dateset,batch_size=batch_size,shuffle=True,collate_fn=test_dateset.collate_fn,num_workers=workers,pin_memory=True) #collect_fn为自定义的打包方式。dataloader返回的是batch_size tuple, image为batch_size tensor,boxes\labels\difficulties为batch_size list,list中每个元素都是batch_size,tensor

def evalute(test_loader,model):
    """
    评估
    :param test_loader: 测试集
    :param model: 模型
    """
    #确实这里是eval模式
    model.eval()

    #创建List 存储检测到的和真值下的boxes、labels、scores
    det_boxes=list()
    det_labels=list()
    det_scores=list()
    true_boxes=list()
    true_labels=list()
    true_difficulties=list() #pass

    #在评估时不需要计算梯度
    with torch.no_grad():

        for i,(images,boxes,labels,diffifulties) in enumerate(test_loader)

















# #====================第二个版本,对应于save_checkpoint2()，感觉有点不太对...???...
# # def save_checkpoint2(epoch,model,optimizer):
# #     state={'model_state_dict':model.state_dict(),
# #            'optimizer_state_dict':optimizer.state_dict(),
# #            'epoch':epoch}
# #     PATH='checkpoint_ssd300.tar'
# #     torch.save(state,PATH)
#
# data_folder='./'
# keep_difficult=True #pass
#
# batch_size=64
# # workers=4
# workers=0
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# #加载节点
# checkpoint='./checkpoint_ssd300.tar'
#
# #下面是直接粘贴的'train.py'中的文件
# model = SSD300(n_classes=20)  # 实例化网络 (实例化了VGGBase,aux,pre网络，以及创建了先验框)
# # 只有在执行完model(images)返回的才是下面的
# # return locs, classes_scores  # (n,8732,4) (n,8732,n_classes)
#
# # 初始化优化器，后面怎么翻译...???...
# biases = list()
# not_biases = list()
# for param_name, param in model.named_parameters():  # 查看参数
#     if param.requires_grad:
#         # 偏置和权重可学习参数分开
#         if param_name.endswith('.bias'):
#             biases.append(param)
#         else:
#             not_biases.append(param)
# # 为不同的可学习参数，设置不同的学习率
# lr=1e-3
# momentum=0.9
# weight_decay=5e-4
# optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr, momentum=momentum,weight_decay=weight_decay)
#
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch=checkpoint['epoch']+1
#
# model.eval()



