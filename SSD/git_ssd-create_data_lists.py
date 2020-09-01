import os
import torch

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

    train_images=list() #创建空列表的方式有两个，list()或[]
    train_objects=list()
    n_objects=0



