﻿reference https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection.git
 #  概念

 - 目标检测
 - SSD：single-shot detection
 早期的目标检测包括两个阶段：区域建议网络（进行目标定位）、分类器（检测建议区域中的目标类别）。从计算角度看，是相当耗时，且并不适用于实际场景的实时应用。SSD模型将定位和检测任务封装进一个单一的前项网络中，从而使得当部署在轻量化的硬件时，能够进行更快的检测
 
 - 多尺度特征图
 在图像分类任务中，预测是基于最终的卷积特征图-原始图像最小但最深的表示。在目标检测中，从中间卷积层中得到的特征图也是能够直接使用的，因为这些是原始图像上不同尺度的表示。因此一个固定尺寸的滤波器在不同特征图上进行卷积，能够检测不同大小的目标
 
 - 先验
 在特定特征图上的特定位置上定义的预计算的框boxes，即为先验。这些boxes有特定的横纵比和尺度（尺寸）。这些是通过仔细选择得到的，以便能够匹配数据集中目标bbox（ground-truth）的性质
 
 - multibox
 这是一个技巧，其将目标bbox的预测作为一个回归问题进行处理。其中被检测到的目标的坐标与其真值坐标进行回归。再有，对于每个预测到的box，为不同的类别生成不同的得分。先验（上述）作为预测阶段的初始点，因为先验是在真值上建模得到的。因此一共有和先验一样多的预测框box，其中的大部分是不包含目标的。
其预测分为两个部分：1）包含或不包含目标的box的坐标，这是一个回归任务；2）这个box的对于不同目标类别的得分，包括背景（表示box中没有背景类），这是一个分类任务
 
 - hard negative mining
 没翻译
 
 - NMS非极大值抑制
 在任何一个给定的位置，多个先验可能重叠。因此产生于这些先验得到预测box可能实际上是同一个目标的重叠box，NMS就是通过抑制除了最大得分的那个box的剩下的所有box来去除冗余的预测

#  overview

 - 边界坐标
 - 中心坐标
 - IOU
 - multibox
 见上面
 
 - SSD
 SSD是一个全卷积网络，主要包含三个部分
 1）基础卷积：由存在的图像分类框架得到，主要提供低层次的特征图
 2）辅助卷积：连接在基础网络的顶部，提供高层次的特征图
 3）预测卷积：在这些特征图中定位和识别
 文章提出了两个模型SSD300和SSD512，数字表示的身世输入图像的尺寸。
##  基础卷积-part1
作者使用的是VGG16框架作为基础网络。文章建议使用在ILSVRC分类任务上预训练的模型。**其在pytorch中已经有实现**，选择其他的也行，但是要考虑计算量的需求。

该git-tutorial中，在预训练的网络中做作出了一些改变，来适应自己的目标检测任务。这些是出于逻辑性和必要性考虑的，以及一些方便性和偏好性。

 - 输入图像尺寸：300*300
 - 第3个poolinglayer，即进行维度减半的操作，使用的是ceil（向上取整）机制而取代默认的floor（向下）取整机制，来决定输出的尺寸。只要前面的特征图是奇数而不是偶数，那么就是有意义的。
 - 改进第5个poolinglayer，从原有的2*2kernel,2stride，编程3*3kernel,1stride。这使得对特征图不在进行维度减半操作
 - 并不需要全连接层（即分类），抛弃fc8，并将fc6和fc7变成全卷积层conv6和conv7

fc到卷积的变换
**值得注意的是当RGB拉伸变成1D时，组织顺序如git上所示，要特别看一下**

**这里可能要注意，虽然是一个2D卷积核，一般在定义的时候只定义不同卷积核的个数，单个卷积核的通道数自动与输入的特征图的通道相同，但是该卷积核的不同通道数上的参数都是不一样的**

##  基础卷积-part2
**这里关于改进的部分没看懂...???...**
常规操作就是讲原来的全连接层转换为全卷积层，
但是转换的卷积核个数太多且大，及其消耗计算
为了改善这个问题，原文作者选择减少卷积核的数量和尺寸，即通过对参数进行子采样...???...这点没懂见; （'github原文'）
github文档中使用了一种像素的方法，但是没懂...???...
针对卷积核尺寸不一的问题，使用的是膨胀卷积的方法。


##  辅助卷积
在上述基础网络行堆叠更多的卷积层。这些卷积层提供额外的特征图，且尺寸逐渐减小。
一共引进了4个卷积block，每个bock包含2层。尽管在基础网路中，利用pooling层尺寸已经减少，在辅助卷积中的每个block中的第2层都使用stride=2的卷积核操作。

##  a detour
**有一个很大的问题，通道的问题在先验框和预测计算的时候是怎么处理的**
在进行预测卷积之前，必须理解要预测的东西是什么。就是目标及其位置，但是目标和位置都是以什么形式存在的呢？
这里必须了解先验priors及其在SSD中扮演的关键角色

###  priors先验
目标检测是相当复杂的，不仅仅值得是其种类。目标可以出现在人任何位置，具有任意的尺寸和形状。（值得注意的是，这里并不打算阐述目标出现在哪里和以怎样形式出现的无线可能性。虽然从数学角度上式可能的，但是许多选择是不可能的或无趣的。...???...）

**实际上，可以将可能的预测的数学空间离散成数千种可能**
**先验priors是预先进行计算的，且具有固定大小的boxes，后面这句话怎么翻译...???...**
**先验框是手动的，但是认真选择的，其基于数据集中真值目标的尺寸和形状来选择的。通过将这些先验框放在一个特征图中的每个可能位置，且考虑了位置上的变化。**

**原作者对先验框的定义如下：**
 - 这些先验框被应用到各种高层和低层次特征图上，也就是来自conv4_3,conv7,conv8_2,conv9_2,conv10_2,conv11_2的特征图。
 - 如果一个先验框有一个尺度s（用来根据面积和后续定义的横纵比，来计算具体的先验框的大小），那么其面积等于边长为s的正方形，最大的特征图即conv4_3，其对应的先验框的尺度为0.1，即图像维度的10%，而剩下的先验框的尺度（对应的是不同特征图（来自于不同的层）下）从0.2线性增加到0.9（0.2；0.375；0.55；0.725；0.9）。可以看出，更大的特征图具有更小尺度的先验框，因此能够更好的检测小目标
 - 在一个特征图上每个位置，其具有不同横纵比的先验框。所有的特征图中的先验框的横纵比为1:1、2:1、1:2。而中间的特征图conv7,conv8_2,conv9_2还具有横纵比为3:1、1:3的先验框。再有所有的特征图，都具有一个额外的先验框，该先验框的横纵比为1:1，尺度为当前特征图和后面一个特征图的尺度的几何平均值。（这里有一个问题，最后一个特征图没有后续特征图，怎么计算尺度的几何平均...???...）
 - **见github中的表格**

###  先验框的可视化
先验框的尺度：w*h=s^2
先验框的横纵比(宽高比)：w/h=a
先验框的宽和高：w=s*sqrt(a),h=s/sqrt(a)
值得注意的是当先验框的边界超出特征图时，被裁切(修剪)

###  预测(框)与先验框
**bbox\先验框\预测框**
利用回归问题来找到目标Bbox的坐标。值得注意的是，先验框不能表示最终预测的boxes

**值得注意的损失，先验框近似的表示预测(框)的概率，这句话什么意思...???...**
这就意味着，可以将每个先验框作为近似起点，然后判断需要调整多少来获得一个更精确的Bbox的预测
因此如果每个预测后的Bbox与先验框有点偏差，那么目标就是计算这个偏差，且需要有一种方式来度量或量化这种偏差
通过计算预测后的Bbox和先验框之间的偏差，并将这些偏差(偏移)进行编码，这个编码后的偏移向量表示的是先验框需要调整多少来生成一个Bbox。
即每个先验框通过调整得到一个更精确的预测框，其上述的四个偏移量就是要进行Bbox坐标回归的形式。

##  预测卷积
上述在不同特征图（6个）上定义了不同尺度的先验框。然后对于每个特征图上的每个位置上的每个先验框，想要预测：
 - Bbox的偏移量(具有四个偏移元素的偏移向量)
 - 对于Bbox上的一系列的n_classes得分，其中classes表示的是全部类别的数量(包括背景类)
为了以尽可能简单的方式执行上述的操作，需要对每个特征图执行两个卷积层操作
 - 一个定位预测卷积层:3*3卷积核,stride=1,padding=1。在该位置上得到每个先验框上使用4个滤波器 **...???...什么意思**。对应于一个先验框的这四个卷积核滤波器计算四个编码的偏移向量，这个四个偏移向量的对应的是由该先验预测的Bbox **...???...没懂**
 - 一个分类预测卷积层：3*3卷积核,stride=1,padding=1。在该位置上得到的每个先验框上使用n_classes个滤波器。对于一个先验框，这n_classes个滤波器计算对应于这个先验框的n_classes个得分。
 
所有的滤波器使用的是3*3的尺寸。实际上，并不需要滤波器的形状与先验框一样，因为不同的滤波器会学着根据不同的先验框形状进行预测。**...???...什么意思**

特征图经过预测卷积(定义卷积和分类卷积):
定位预测：定位预测后的该位置下(每个位置)的通道值表示的是，相对于该位置下的多个先验框下的被编码的偏移量，num_channels=num_priors*4
分类预测：分类预测后的该位置下(每个位置)的通道值表示的是，相对于该位置下的多个先验框下的类别得分,这里的类别包括背景类
最终将所有特征图下的预测卷积结果堆叠起来

##  multibox loss损失
有几个问题需要回到：
1）回归后的Bbox的损失函数
2）是否对分类得分使用multiclass cross-entropy
3）如何组合定位和分类损失
4）如何将预测后的box匹配到ground-truth
5）已经有个很多预测（每个特征图上的每个位置上的多个先验框对应的预测结果），如何考虑不包含目标的box

###  将预测box与ground-truth进行匹配
依据的就是先验框
 - 计算所有先验框与真值目标之间的IOU，tensor：num_priors*num_ground-truth
 - 每个先验框与最大的IOU对应的目标进行匹配
 - 如果一个先验框与其匹配的目标之间的IOU阈值小于0.5，那么将其定义为不包含目标，因此称其为假匹配，大多数的先验框都是假的，(即IOU小于0.5的先验框的类别对应的是背景)
 - 少数的先验框与其匹配的目标之间的IOU大于0.5，将其定义为包含目标，因此称其为正匹配。
现在，将每个先验框匹配到一个真值ground-truth，实际上，也将对应的（计算于先验框）的预测box（预测box与先验框的数量相同）匹配到一个真值
 
'github tutorial'中给出了一个例子：
每个先验框有一个匹配，或真或假。相关的，每个预测也有一个匹配，或真或假。
与真值目标具有正匹配的预测(框)具有ground-truth真值坐标，其被用来进行回归任务。那些假匹配就没有对应的坐标
所有的预测(框)都有一个真值标签，无论其属于正匹配或负匹配，其被用于进行分类任务

###  定位损失
值得注意的是，对于假匹配没有真值坐标。那么为什么还要训练这个模型来在空的空间(无目标的位置)画box呢？
因此，定位损失仅表示为：如何精确的将正确匹配的预测框boxes回归到对应的真值坐标
因为，以偏移向量的形式预测的定位boxes，因此在计算损失之前，也应该需要对真值坐标进行编码
定位损失是一个平均的平滑L1损失，表示的是编码后的正确匹配的定位boxes的偏移量，与其真值之间的损失

###  置信度损失
每一个预测，无论是正匹配还是假匹配，都有一个与之关联的真值标签，**后边一句话怎么翻译...???..**
然而，考虑到图像中仅有少量的目标，大部分的作出的预测(框)实际上并不包含目标。如果假匹配的数量比正匹配的数量多很多，那么最终得到的模型不太可能能够检测目标，其多半是被训练成检测背景类。
结果这个问题的办法就是，限制假匹配的数量，其将在损失函数中被评估。
...???...没看完

###  总体损失
multibox损失：综合了置信度损失和定位损失，其比例为\alpha,一般来讲\alpha是一个可学习的参数，但是在原文中，\alpha=1。

##  预测过程
模型训练以后，可以将其应用到图像中。然而，预测仍然是原始的形式：两个tensor:所有先验框对应的偏移向量和类别得分。其需要经过处理来得到最终的、具有标签得到可解释性的Bbox。即：
 - 现在有，以对应于先验框的偏移向量表示的预测框。将其解码为边界坐标，
 - 对于每一个非背景类：
 1）对所有的boxes中的每一个box，提取该类对应的得分
 2）消除那些得分不满足阈值的boxes
 3）这些经过保留的boxes，为特定目标类的候选box
此时如果在原始图像上画出这些候选框，会发现有许多高度重叠的box，其明显是冗余的。这是因为从数千个先验框中，很有可能有多个预测对应于相同的目标。
为解决这个问题：
1）首先对每个类别，按照概率得分对候选box进行排序；
2）然后找到哪些个候选是冗余的，IOU就是一个工具，能够判断两个box的一致性；
3）因此，建立box对，计算IOU，如果候选box的IOU大于0.5，那么二者就可能是相同的目标。并且抑制得分小的那个候选box
上述的方法叫做NMS非极大值抑制，从算法上将：
1）对一个特定类别的候选box，根据得分进行降序排列
2）考虑得分最高的候选box，消除其他与该候选框(得分最大的)IOU大于0.5的候选box
3）消除之后，在剩下的候选box中选择得分最大的一个，按照2）中的方法消除冗余候选box
4）重复上述操作，直到遍历了整个候选box

===========================================
在特征图上利用卷积核进行
预测，预测出来输出的就是与默认框相对的Bbox的偏移量，即对于每个特征图cell，预测的是每个cell中相对于(原始)默认框的偏移量，和每个默认框中有目标存在的类别得分。值得注意的是每个默认框相对于特征图cell的位置是固定的。
具体来说，对于一个给定的位置(特征图中的一个cell)一共定义k个默认框，每个默认框计算c个类别得分，和相对于该默认框的4个偏移量。因此，在一个特征图上的每个位置(cell)一共应用(c+4)*k个滤波器，最终对于m*n的特征图，一共产生(c+4)*m*n个输出。

在训练的时候仅需要一个输入图像和每个目标对应的ground-truth框。在训练的时候，首先将这些默认框与真值框进行匹配，有正有负。

对于默认框来说，其余faster-RCNN中使用的anchor boxes相似，不同点就将其应用在不同分辨率的特征图中，从而将输出Bbox的可能潜在空间离散化

在训练阶段，需要定义那些默认框对应于真值，并根据此进行训练。(**文章提到的匹配策略没看懂...???...**)

定位损失是预测框(不是默认框)与真值框参数之间的平滑L1损失
文章指出的是对bbox(检测框？？？)的中心和宽高的偏移进行回归

在文章中，文章利用公式计算不同特征图对应的尺度和横纵比，默认框的中心位置也是根据公式计算出来的，与cell相关

判断为负的例子没懂；难训练样本的处理没懂，数据增强没懂

文章对VGG16模型的改进，相似与[16]的方法
文章对conv4_3尺度的改进没懂，参考的是[12]
检测分析工具没看懂,参考的是[19]
经过分析，SSD有更低的定位误差、但是对相似目标检测不佳、对Bbox尺寸敏感(即对小目标检测不佳)

================================
 #  实现
 ##  模型输入
 ###  图像
基础网络利用的是pytorch中内置的在ImageNet上预训练的VGG-16，
网络的输入图像应该为[N,3,300,300],dtype=float,像素值范围为0-1，RGB通道应该归一化到mean = [0.485, 0.456, 0.406];std = [0.229, 0.224, 0.225]
 ###  目标的Bbox
目标的真值Bbox坐标表示为(x_min,y_min,x_max,y_max)
网络的输入ground-truth bbox应该为一个长度为N(N=num_imgs)的list,其中list中的每个元素都是一个[N_o,4]的float tensor,其中N_o是特定图像中存在的目标数量
 ###  目标标签
 每个标签应该被编码成从1到20打的整数，来表示20个不同的目标类。同时应该增加一个为索引为0的背景类，表示Bbox中没有目标的存在
(后面有一句话怎么翻译)
网络输入的ground-truth标签应该是一个长度为N(N=num_imgs)的list，其中每个元素都是一个具有N_o维度的Longtensor，其中N_o是特定图像中目标存在的数量