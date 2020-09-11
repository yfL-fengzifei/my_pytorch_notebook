import torch
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiBoxLoss(nn.Module):
    """
    多框损失，目标检测损失函数
    组合：
    预测框(也就是偏移量)的定位损失；预测到的类别得分的置信度损失
    """
    #priors_cxcy=return prior_boxes #(8732,4) 中心坐标形式
    #其他的参数选择的是默认值
    def __init__(self,priors_cxcy,thresold=0.5,neg_pos_ratio=3,alpha=1.):
        super(MultiBoxLoss,self).__init__()
        self.priors_cxcy=priors_cxcy #(8732,4) 中心坐标形式
        self.priors_xy=cxcy_to_xy(priors_cxcy) #传入先验框，返回的是边界坐标形式
        self.threshold=thresold
        self.neg_pos_ratio=neg_pos_ratio
        self.alpha=alpha

        self.smooth_l1=nn.L1Loss()
        self.cross_entropy=nn.CrossEntropyLoss(reduction=False) #...???

    def forward(self,predicted_locs,predicted_scores,boxes,lables):
        """
        前向传播，传入的是return locs, classes_scores =model(image)  # (n,8732,4) (n,8732,n_classes)
        :param predicted_locs: 预测到的相对于8732个先验框的位置(偏移量)，tensor (n,8732,4)
        :param predicted_scores: 对应于每个预测框下的类别得分， tensor (n,8732,n_classes)
        :param boxes: ground-truth 边界框(预测框)(以边界形式) N tensor List，(batch_size)boxes(来自于dataloader)
        :param lables: ground-truth 标签，N tensor list
        :return: 多框损失，标量
        """
        batch_size=predicted_locs.size(0) #batch_size (n,8732,4)
        n_priors=self.priors_cxcy.size(0) #8732
        n_classes=predicted_scores.size(2) #(n,8732,n_classes)

        assert n_priors==predicted_locs.size(1)==predicted_scores.size(1)

        true_locs=torch.zeros((batch_size,n_priors,4),dtype=torch.float).to(device) #(n,8732,4)
        true_classes=torch.zeros((batch_size,n_priors),dtype=torch.long).to(device) #(n,8732)

        #对于batch中的每一张图像
        for i in range(batch_size):
            #值得注意的是，这里的boxes是真值ground-truth
            #这里的boxes是一个batch下的,是一个list,其中每个元素都是一个tensor，
            n_objects=boxes[i].size(0) #boxes中的每个元素对应的是batch中的每个图像下对应的boxes,.size(0)表示每个图片下有多少个n_objects
            overlop=find_jaccard_overlop(boxes[i],self.priors_xy) #boxes[i]是一个样本(图像)下的，(n_objects,4),self.priors_xy (8732,4) #(n_objects,8732)

            #对于每个先验框，找到与之IOU最大的目标
            #dim=0表示沿着列方向(tensor是二维的情况)寻找，最大值，在没有keepdim参数的时候，返回两个列表
            #第一个list返回的是包含每列最大值的一维list,第二个list返回的是该最大值在该列的索引
            overlap_for_each_prior,object_for_each_prior=overlop.max(dim=0) #(8732),其中object_for_each_prior为索引，元素范围值最大为n_objects (实际上是n_objects-1,因为是从0开始索引的)

            #...???...
            #实在是没懂

            #首先为每个目标ground-truth找到具有最大IOU的先验框
            _,prior_for_each_object=overlop.max(dim=1) #(n,objects) 索引值从(0,8732-1)

            #然后...???...
            object_for_each_prior[prior_for_each_object]=torch.LongTensor(range(n_objects)).to(device)


        return pass #(conf_loss+self.alpha*loc_loss)


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
    return intersection/union #(n1,n2)


def find_intersection(set_1,set_2):
    """
    计算交集
    set_1: tensor,(n1,4) 这里是(n_objects,4)
    set_2: tensor(n2,4) 这里是(8732,4)
    return: 返回set1中每个box中与set2中每个box的交集，tensor (n1,n2)
    """
    lower_bounds=torch.max(set_1[:,:2].unsqueeze(1),set_2[:,:2].unsqueeze(0)) #(n1,n2,2)
    #(n_objects,1,2) ; (1,8732,2) -> (n_objects,8732,2) #x_min,y_min下的最大值
    upper_bounds=torch.min(set_1[:,2:].unsqueeze(1),set_2[:,2:].unsqueeze(0)) #(n1,n2,2)
    #torch.max() 当传入两个tensor的时候，按维度和元素进行比较，且两个tensor的维度不需要完全匹配，但是要遵循广播机制，最终的输出也是经过广播后的tensor
    #注意广播机制，对于二维tensor来说，0表示列维度，就是从一行变成了两行;1表示行维度，就是从一列变成了两列
    #不要抠具体的维度，很容易混乱，(暂时先不要抠)，本质上是用最小的中的最大的，和，最大的中的最小的相减，相当于得到了宽和高，最终相乘可以得到面积
    #(n_objects,1,2) ; (1,8732,2) -> (n_objects,8732,2) #x_max,y_max下的最小值

    intersection_dims=torch.clamp(upper_bounds-lower_bounds,min=0) #(n1,n2,2) torch.clamp()将输入tensor中的元素，限制在指定扰动范围内，该函数中是将所有元素都限定到大于0。 [0]：(x_max-x_min); [1]:(y_max-y_min)
    return intersection_dims[:,:,0]*intersection_dims[:,:,1] #(n1,n2)


def cxcy_to_xy(cxcy):
    """
    将bbox从中心坐标形式转换为边界坐标形式，(c_x,c_y,w,h) -> (x_min,y_min,x_max,y_max)
    :param cxcy: 中心坐标形式的Bbox，tensor (n_boxes,4)
    :return: 边界坐标形式的Bbox，tensor (n_boxes,4)
    """
    #传入(8743,4):(c_x,c_y,w,h)
    # x_min=c_x-w/2,y_min=cy-h/2
    # x_max=c_x+w/2,y_max=cy+h/2
    #在第2维度进行拼接
    return torch.cat([cxcy[:,:2]-(cxcy[:,2:]/2),
                      cxcy[:,:2]+(cxcy[:,2:]/2)],1)
    #返回的是边界坐标形式


def adjust_learning_rate(optimizer,scale):
    """
    利用特定因子调整学习率
    :param optimizer: 优化器的学习率必须是衰减的
    :param scale: 学习率的乘数
    """
    for param_group in optimizer.param_groups: #optimizer根据当初初始化的情况进行分布，这里应该有两组
        param_group['lr']=param_group['lr']*scale
    print("decaying learning rate.\n The new LR is %f\n" %(optimizer.param_groups[1]['lr']))


class AverageMeter(object):
    """
    以data_time为例，每次epoch时都实例化一次data_time=AverageMeter(),自动调用reset
    然后当对每个Batch执行操作的时候，执行data_time.update(time.time()-start)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

