import os
import torch
import json
import xml.etree.ElementTree as ET
from git_ssd-parse_annotations import create_label_map,parse_annotation

def create_data_lists(voc07_path,voc12_path,output_folder):
    """
    创建图像、Bbox、标签列表，并保存到特定的文件中
    :param voc07_path: voc07文件夹的路径
    :param voc12_path: voc12文件夹的路径
    :param output_folder: folder where the JSONs must be saved
    """
    voc12_path=os.path.abspath(voc12_path) #voc07的绝对路径
    voc07_path=os.path.abspath(voc07_path)
    #值得注意的是，这里返回的路径实际上是：当前工程目录的绝对路径+参数文件夹的路径(即传入的是文件夹的路径)
    # dir='ceratin_path'
    # for f in os.listdir(dir):
    #     path=os.path.abspath(os.path.join(dir,f)) #这有可能是从当前工作目录开始的绝对路径

    # 创建空列表的方式有两个，list()或[]
    train_images=list() #图像列表
    train_objects=list() #每个图像对应的ground-truth
    n_objects=0

    #训练数据
    for path in [voc07_path,voc12_path]: #_path应该是文件夹的名称，len[voc07_path,voc12_path]=2

        # 得到lable_map
        _, label_map = create_label_map()

        #查找训练数据中图像的ID
        #.join=path(文件夹的绝对路径)+'ImageSets/Main/trainval.txt'
        #with语句(官方推荐)，关闭文件的操作会被自动执行，不必调用close()方法
        with open(os.path.join(path,'ImageSets/Main/trainval.txt')) as f:
            #fileObject.read(size) 读取指定的字节数，不指定或指定为负表示读取所有
            #str.splitlines([keepends]) 按照\r,\n,\r\n分隔，返回一个包含各行作为元素的列表，keepens=False默认不保留末尾的换行符
            ids=f.read().splitlines() #这是一个list ...???...

        for id in ids:
            #解析标注的XML文件
            #'git_ssd-parse_annotation.py'
            #返回的是{'boxes':boxes,'labels':labels,'difficulties':difficulties},对应于一个图像id
            objects=parse_annotation(os.path.join(path,'Annotations',id+'.xml'))

            if len(objects)==0: #如果该图像中没有图像
                continue
            n_objects+=len(objects) #图像中一共包含几个目标
            train_objects.append(objects) #最终返回list,每个元素都是一个图像中对应的多个目标的ground-truth(ground-truth是字典的形式)
            train_images.append(os.path.join(path,'JPEGImages',id+'.jpg')) #最终返回的list,每个元素都是一个图像

    #判断，抛出异常
    assert len(train_objects)==len(train_images)

    #保存成json的形式
    with open(os.path.join(output_folder,'TRAIN_images.json'),'w') as j:
        json.dump(train_images,j)
    with open(os.path.join(output_folder,'TRAIN_objects.json'),'w') as j:
        json.dump(train_objects,j)
    with open(os.path.join(output_folder,'label_map.json'),'w') as j:
        json.dump(label_map,j)






