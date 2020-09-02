import os
import torch
import json
import xml.etree.ElementTree as ET
import git_ssd-parse_annotations

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

    train_images=list() #创建空列表的方式有两个，list()或[]
    train_objects=list()
    n_objects=0

    #训练数据
    for path in [voc07_path,voc12_path]:

        #查找训练数据中图像的ID
        #.join=path(文件夹的绝对路径)+'ImageSets/Main/trainval.txt'
        #with语句(官方推荐)，关闭文件的操作会被自动执行，不必调用close()方法
        with open(os.path.join(path,'ImageSets/Main/trainval.txt')) as f:
            #fileObject.read(size) 读取指定的字节数，不指定或指定为负表示读取所有
            #str.splitlines([keepends]) 按照\r,\n,\r\n分隔，返回一个包含各行作为元素的列表，keepens=False默认不保留末尾的换行符
            ids=f.read().splitlines() #这是一个list

        for id in ids:
            #解析标注的XML文件
            objects=parse_annotation(os.path.join(path,'Annotations',id+'.xml'))






