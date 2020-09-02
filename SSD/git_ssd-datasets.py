import torch
import torch.utils.data as Data
import json
import os
from PIL import Image

"""
创建自己的数据集
需要定义__len__方法，返回的是dataset的数量
需要定义__getitem__方法，返回的是第i个图像，bboxes、labels.基于的是json文件

Dataset是一个抽象类，所有自定义的Dataset需要继承它并复习__getitem__()函数，即接收一个索引，返回一个样本
__getitem__:返回一条数据或一个样本
__len__:返回样本的数量
"""

class PascalVOCDataset(Data.Dataset):
    """
    定义一个pytorch 数据集，然后再pytorch DataLoader中使用，来创建bctahes
    """
    def __init__(self,data_folder,split,keep_difficult=False):
        """
        :param data_folder: 存储数据文件的文件夹
        :param split: split,TRAIN或TEST中的一个
        :param keep_difficult: 保留或抛弃被定义为难检测的目标
        """

        #实例化某个数据集的时候，传入保存的文件夹，并定义是训练还是测试
        #一共五个文件：
        # train_images.json
        # train_objects.json
        # label_map.json
        # test_images.json
        # test_objects.json
        self.split=split.upper() #大写
        assert self.split in {'TRAIN','TEST'} #检查并抛出异常

        self.data_folder=data_folder
        self.keep_difficult=keep_difficult

        #读取数据文件
        #json文件中是由list得来的
        with open(os.path.join(data_folder,self.split+'_images.json'),'r') as j:
            self.images=json.load(j) #json.load用来读取文件，json.laods用来读取字符串
        with open(os.path.join(data_folder,self.split+'_objects.json'),'r') as j:
            self.objects=json.load(j)

        assert len(self.images)==len(self.objects)

    def __getitem__(self,i):
        #读取图像
        image=Image.open(self.images[i],mode='r')
        image=image.convert('RGB') #读取模式

        #读取objects中的ground-truth数据
        objects=self.objects[i]
        boxes=torch.FloatTensor(objects['boxes']) #(n_objects,4) 一张图像中的目标数*4坐标
        labels=torch.LongTensor(objects['labels']) #(n_objects) 一张图像中的目标数，在tensor中size是[]维，这里应该是整数
        difficulties=torch.ByteTensor(objects['difficulties']) #(n_objects)，这里应该是0或1

        #如果想要忽略难识别的目标，则执行下面的操作
        if not self.keep_difficult:
            boxes=boxes[1-difficulties] #这里的索引时利用的表达式的方法,注意这里只有numpy和tensor才能这么做
            labels=labels[1-difficulties]
            difficulties=difficulties[1-difficulties]

        #应用转换

