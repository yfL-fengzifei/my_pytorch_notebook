import torch
import torchvision
import torchvision.transforms.functional as FT
import random

"""
==================================================================================================
import torchvision.transforms as transforms

#在PIL图像上转换
transforms.CenterCrop() #中心裁切
transforms.ColorJitter() #改变亮度、对比度、饱和度、色度
transforms.FiveCrop() #四个角落+中心裁切
transforms.Grayscale() #转变为灰度图
transforms.pad() #边缘填充
transforms.RandomAffine() #仿射变换
transforms.RandomApply() #随机选取变换中的一个
transforms.RandomSizedCrop()、.RandomHorizotalFlip() #各种随机变换
transforms.Resize() #改变尺寸

#在torch.tensor上转换
transforms.Normalize() #对每个通道进行归一化

#类型转换
transforms.ToPILImage() #转换为PIL图像
transforms.ToTensor() #转换为tenosr

#一般变换
transforms.Lambda() #用户定义的转换
transforms.functional.adjust_brightness() #调整图像亮度
transforms.functional.adjust_contrast() #调整对比度
transforms.functional.adjust_gamma() #gamma矫正
transforms.functional.adjust_hue() #调整色相
transforms.functional.adjust_saturation() #调整饱和度
transforms.functional.affine() #放射变换
transforms.functional.crop() #剪裁
....等等，有的与transforms.certain_fun一样

transforms.functional.to_pil_image() #将tensor或ndarray转换成PIL图像
transforms.functional.to_tensor() #将PIL图片或ndarray转换成tensor
"""

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


def expand(image,boxes,filler):
    """
    将图像放在一个更大的填充材料画布上，来执行方法操作。有助于学习小目标
    image: tensor,(3,original_h,original_w)
    boxes: 边界坐标形式的Bbox，tensor,(n_objects,4)
    filler: 填充材料的RGB值，[R,G,B]列表
    return: 扩充后的图像，更新bbox坐标

    该图像有50%的概率，进行方法，并将其中对应的object ground-truth进行更改(只要图像一更改，对应的Bbox就必须更改)
    """

    #计算原始图像维度
    original_h=image.size(1)
    original_w=image.size(2)

    #计算要新生成的图像维度
    max_scale=4 #设置最大放大尺度
    scale=random.uniform(1,max_scale) #生成随机数
    new_h=int(scale*original_h)
    new_w=int(scale*original_w)

    #创建具有填充的新图像
    filler=torch.FloatTensor(filler) #filler=torch.FloatTensor([0.485,0.456,0.406]) #mean=[0.485,0.456,0.406]
    new_image=torch.ones((3,new_h,new_w),dtype=torch.float)*filler.unsqueeze(1).unsqueeze(1) #filler.shape=[3,1,1],相当于每个通道都乘以一个对应的均值,相当于对每个通道分别进行填充
    #不要用new_image=filler.unsqueeze(1).unsqueeze(1).expand()

    #将原始的图像随机放在新的图像上
    left=random.randint(0,new_w-original_w)
    right=left+original_w
    top=random.randint(0,new_h-original_h)
    bottom=top+original_h
    new_image[:,top:bottom,left:right]=image

    #根据新图像调整bbox坐标
    new_boxes=boxes+torch.FloatTensor([left,top,left,top]).unsqueeze(0) #bbox原始边界形式为[xmin,ymin,xmax,ymax]，最终得到new_boxes还是(n_objects,4)

    return new_image,new_boxes


def random_crop(image,boxes,labels,difficulties):
    """
    有助于学习检测更大和部分目标。有些目标可能被完全裁切掉

    image: tensor,(3,original_h,original_w)
    boxes: 边界坐标形式得到Bbox，tensor,(n_objects,4)
    labels: 目标标签，(n_objects)
    difficulties: 目标检测的难易程度，tensor (n_objects)
    return: 返回裁切后的图像，更新Bbox坐标、标签、难易程度
    """
    #计算原始图像维度
    original_h=image.size(1)
    original_w=image.size(2)

    #持续选择一个最小的IOU，直到作出正确的裁切
    while True:
        #最最小的IOU随机取值
        min_overlap=random.choice([0.,.1,.3,.5,.7,.9,None]) #None表示不进行裁切 #random.choice() 从元组或列表中随机选择一项

        #如果不进行裁切
        if min_overlap==None:
            return image,boxes,labels,difficulties

        #在该经过选择后的min_overlap下记性五十次的尝试
        #该过程在文章中并没有设计，但是在作者的源码中有所呈现

        #对于一个给定得到(在预先定义的IOU中随机选择的一个)IOU，先执行裁切，然后计算横纵比，然后计算裁切后的图像与原始Bbox之间的IOU，然后正式裁切图像，然后计算原始Bbox得到中点是否落在裁切后的图像上，任何一个条件不满足阈值，则不再继续执行，重新循环
        max_trials=50
        for _ in range(max_trials):
            #裁切维度必须是原始维度的[0.3,1]之间
            #实际上在文章中出现的是[0.1,1],但是作者给的源码是[0.3,1]
            #值得注意的是，在放大操作中'def expand()'中，是将长宽同时进行缩放，利用的是相同的调整因子，此时，进行缩小是，是对长宽分别进行调整
            min_scale=0.3
            scale_h=random.uniform(min_scale,1)
            scale_w=random.uniform(min_scale,1)
            new_h=int(scale_h*original_h)
            new_w=int(scale_w*original_w)

            #横纵比应该在[0.5,2]之间
            aspect_ratio=new_h/new_w #这里是高比宽
            if not 0.5<aspect_ratio<2:
                continue

            #裁切维度
            left=random.randint(0,original_w-new_w)
            right=left+new_w
            top=random.randint(0,original_h-new_h)
            bottom=top+new_h
            crop=torch.FloatTensor([left,top,right,bottom])
            #这里返回的[left,top,right,bottom]的坐标基准还是原始图像

            #计算crop和bbox之间的IOU
            overlop=find_jaccard_overlop(crop.unsqueeze(0),boxes) #crop.unsqueeze(0)变成(1,4),然后与该原始图像下的bboxes计算IOU，find_jaccard_overlap返回的维度是(n1,n2)，这里实际上是(1,n_objects)
            overlop=overlop.squeeze(0) #去掉维度为1的相应维度，得到n_objects个IOU

            #如果上述，缩小后的图像与原始图像中的IOU，小于随机选择的(预先定义)的IOU，则不再向下执行
            if overlop.max().item()<min_overlap:
                continue

            #正式裁切图像
            new_image=image[:,top:bottom,left:right] #即从原始图像中直接索引新的裁切图像，[3,new_h,new_w]

            #计算原始Bbox的中心
            bb_centers=(boxes[:,:2]+boxes[:,2:])/2 #(n_objects,2)

            #查找裁切后的图像中，中点落在该裁切图像上的Bbox
            certer_in_crop=(bb_centers[:,0]>left)*(bb_centers[:,0]<right)*(bb_centers[:,1]>top)*(bb_centers[:,1]<bottom) #(n_objects)
            if not certer_in_crop.any():
                continue

            #抛弃不满足条件的Bbox
            new_boxes=boxes[certer_in_crop,:] #ground-truth 一直就是tensor,在n_objects维度中，索引满足要求的，利用的是比较表达式索引法
            new_labels=labels[certer_in_crop]
            new_difficulties=difficulties[certer_in_crop]

            #计算裁切图像中bbox新的坐标
            #这里同时将超出裁切后的图像的Bbox边界做了裁切
            new_boxes[:,:2]=torch.max(new_boxes[:,:2],crop[:2])
            new_boxes[:,:2]-=crop[:2]
            new_boxes[:,2:]=torch.min(new_boxes[:,:2],crop[:2])
            new_boxes[:,2:]-=crop[:2]

            return new_image,new_boxes,new_labels,new_difficulties


def find_jaccard_overlop(set_1,set_2):
    """
    计算IOU
    set_1: tensor (n1,4)
    set_2: tensor (n2,4)
    return: set1中的每个box相对于set2中每个box的IOU，tensor (n1,n2)
    """
    #找到交集
    intersection=find_intersection(set_1,set_2) #(n1,n2)

    #找到每个box的面积
    areas_set_1=(set_1[:,2]-set_1[:,0])*(set_1[:,3]-set_1[:,1]) #(n1)
    areas_set_2=(set_2[:,2]-set_2[:,0])*(set_2[:,3]-set_2[:,1]) #(n1)

    #找到并集
    union=areas_set_1.unsqueeze(1)+areas_set_2.unsqueeze(0)-intersection #(n1,n2) 相当于x+z+y+z-z

    #计算IOU
    return intersection/union


def find_intersection(set_1,set_2):
    """
    计算交集
    set_1: tensor,(n1,4)
    set_2: tensor(n2,4)
    return: 返回set1中每个box中与set2中每个box的交集，tensor (n1,n2)
    """
    lower_bounds=torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0)) #(n1,n2,2)
    upper_bounds=torch.min(set_1[:,2:].unsqueeze(1),set_2[:,2:].unsqueeze(0)) #(n1,n2,2)
    #torch.max() 当传入两个tensor的时候，按维度和元素进行比较，且两个tensor的维度不需要完全匹配，但是要遵循广播机制，最终的输出也是经过广播后的tensor
    #注意广播机制，对于二维tensor来说，0表示列维度，就是从一行变成了两行;1表示行维度，就是从一列变成了两列
    #不要扣具体的维度，很容易混乱，(暂时先不要扣)，本质上是用最小的中的最大的，和，最大的中的最小的相减，相当于得到了宽和高，最终相乘可以得到面积

    intersection_dims=torch.clamp(upper_bounds-lower_bounds,min=0) #(n1,n2,2) torch.clamp()将输入tensor中的元素，限制在指定扰动范围内，该函数中是将所有元素都限定到大于0
    return intersection_dims[:,:,0]*intersection_dims[:,:,1] #(n1,n2)


def flip(image,boxes):
    """
    水平翻转
    image: PIl image
    boxes: 边界坐标形式，tensor (n_objects,4)
    return: 翻转后的图像，更新bbox坐标
    """
    #翻转图像
    new_image=FT.hflip(image)

    #翻转bbox
    #减一可能是跟算法有关，不懂...???...
    new_boxes=boxes
    new_boxes[:,:2]=image.width-boxes[:,0]-1
    new_boxes[:,2:]=image.height-boxes[:,2]-1
    new_boxes=new_boxes[:,[2,1,0,3]]

    return new_image,new_boxes


def resize(image,boxes,dim=(300,300),return_percent_coords=True):
    """
    将图像缩放到(300,300)
    image: PIL image
    boxes: 边界坐标形式，tensor (n_objects,4)
    return: 返回缩放后的图像，更新Bbox坐标
    """
    #缩放图像
    new_image=FT.resize(image,dim)

    #缩放Bbox
    old_dims=torch.FloatTensor([image.width,image.height,image.width,image.height]).unsqueeze(0) #(1,4)
    new_boxes=boxes/old_dims #n_objects中的美一个都除以上面的维度，变成分数形式

    if not return_percent_coords:
        new_dims=torch.FloatTensor([dim[1],dim[0],dim[1],dim[0]]).unsqueeze(0)
        new_boxes=new_boxes*new_dims
        #这里边，不管图像缩放不缩放，Bbox的分数形式都是不会变的

    return new_image,new_boxes


#值得注意的是整个transform作用的都是一张图像
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
    new_boxes=boxes
    new_labels=labels
    new_difficulties=difficulties

    #当评估或预测时，不进行下述的操作
    if split=='TRAIN':
        #随机进行光度改变，概率为50%,
        new_image=photometric_distort(new_image)

        #将PIL图像或ndarray转换为tensor
        new_image=FT.to_tensor(new_image)

        #缩放图像，概率50%，有助于检测小目标
        #利用ImageNet data数据的均值来填充周围的空间
        if random.random<0.5:
            new_image,new_boxes=expand(new_image,boxes,filler=mean) #这里的mean是上述ImageNet data归一化时使用的通道均值

        #值得注意的是，该图像可能进行放大，但是必经过裁切
        #随机裁切图像()缩小
        new_image,new_boxes,new_labels,new_difficulties=random_crop(new_image,new_boxes,new_labels,new_difficulties)

        #翻转图像，概率50%
        if random.random()<0.5:
            new_image,new_boxes=flip(new_image,new_boxes)

        #将图像缩放到网络定义的需要的尺寸，并将bbox从绝对边界坐标形式，变成分数形式
        new_image,new_boxes=resize(new_image,new_boxes,dims=(300,300))

        #转换为PIL图像
        new_image=FT.to_pil_image(new_image)

        #利用ImageNet data数据中的均值的标准差进行归一化
        #上述预先定义的均值和标准差，仅在放大的pad中使用到了mean
        #归一化时在PIL图像上进行得到
        new_image=FT.normalize(new_image,mean=mean,std=std)

        return new_image,new_boxes,new_labels,new_difficulties









