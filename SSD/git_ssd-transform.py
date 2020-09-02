import torch
import torchvision
import torchvision.transforms.functional as FT
import random

#这里有一个问题，数据标注是在原始图像上，而进行数据增强后，对应的数据标注怎么办...???...很重要，下满的第3条有所说明

"""
对图像和object的ground-truth进行转换

1、随机调整亮度、对比度、饱和度和色度，每个都有50%的机会，且按随机的顺序
2、50%的机会，对图像执行放大操作，这有助于学习检测小目标。放大后的图像必须是原始图像的1到4倍。周围的空间必须利用ImageNet data数据的均值来填充...这句话什么意思
3、随机裁切图像，也就是执行缩小操作。这有助于学习检测大目标或部分目标。一些目标可能被完全裁减掉。裁切的维度应该在原始维度的0.3到1之间。横纵比应该在0.5到2之间。每个裁切后的图像，应该保证至少有一个bbox存在(保留)，且Bbox与裁切后的图像之间的IOU是0,0.1,0.3,0.5,0.7或0.9中的一个(这是被随机选择的)。再有，任何保留下来的bbox的中心不再属于裁切后的图像中时，将会被抛弃
4、50%的机会，对图像执行水平翻转
5、将图像缩放到300*300
6、将所有的Bbox的绝对边界坐标转换成分数边界坐标。在模型的所有阶段，边界和其以中心坐标表示的Bbox都会以小数的形式存在
7、利用ImageNet图像数据(这些数据被用来预训练基础的VGG模型)的均值和方差进行图像归一化
"""

def photometric_distort(image):
    """
    随机改变亮度、对比度、饱和度、色度，概率为50%，
    image:a PIL image
    return: 变化后的图像
    """
    new_image=image
    distortions=[
        FT.adjust_brightness,
        FT.adjust_contrast,
        FT.adjust_saturation,
        FT.adjust_hue
    ]
    #distortions相当于实例化了一个函数

    random.shuffle(distortions)

    #对每一张图像，亮度、对比度、饱和度、色度调整的几率都是50%
    for d in distortions:
        if random.random()<0.5: #random.random()返回一个随机生成的[0,1]之间的实数
            if d.__name__ is 'adjust_hue': #__name__内置名字
                #对于hue_delta
                adjust_factor=random.uniform(-18/255.,18/255) #random.uniform随机生成一个(x,y)之间的实数
            else:
                #对于亮度、对比度、饱和度
                adjust_factor=random.uniform(0.5,1.5)

            #执行亮度变化
            new_image=d(new_image,adjust_factor)

    return new_image


def transform(image,boxes,labels,difficulties,split):
    """
    image:image,a PIL Image
    boxes: 边界坐标形式，维度为(n_objects,4)的tensor
    labels: 目标的标签，维度为[]，数量为(n_objects)的tensor
    difficulties: 目标检测的难易度，维度为[],数量为(n_objects)的tensor
    split: “TRAIN" 或 "TEST"之一，因为会应用到不同的转换结合
    return: 返回转换后的图像、bbox坐标、标签、难易度
    """

    #均值、标准差
    #均值和标准差是利用ImageNet计算得来的，文章使用的VGG基础网络，就是使用的ImageNet data进行的预训练
    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

    new_image=image
    new_box=boxes
    new_labels=labels
    new_difficulties=difficulties

    #当评估或预测时，不进行下述的操作
    if split=='TRAIN':
        #随机进行光度改变，概率为50%,
        new_image=photometric_distort(new_image)

        #将PIL图像转换为tensor
        new_image=FT.to_tensor(new_image)










