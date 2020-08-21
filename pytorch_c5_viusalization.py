#visualization

"""
=================================================================================================
Tensorboard
"""
"""
python脚本，记录可视化的数据 --> 放入到硬盘中 envent file --> 在终端使用tensorboard --> web端然后执行可视化
"""

"""
SummaryWriter
提供创建event file的高级接口
属性：
log_dir：event file输出文件夹
comment：不指定log_dir时，文件夹后缀
filename_suffix: event file 文件名后缀

例子：
log_dir='./train_log/test_log_dir'
# writer=SummaryWriter(log_dir=log_dir,comment='_scalars',filename_suffix='12345678')
#设置了log_dir后，comment就不会起作用了

writer=SummaryWriter(comment='_scalars',filename_suffix='12345678')

for x in range(100):
    writer.add_scalar('y=pow_2_x',2**x,x)

writer.close()
"""

"""
add_scalar(tag,scalar_value,global_step=None,walltime=None)
记录标量
tag:图像的标签名，图的唯一标识(title)
scalar_value:要记录的标量，（y轴）
global_step:x轴 （根据不同的迭代情况）


add_scalars(main_tag,tag_scalar_dict,global_step=None,walltime=None)
绘制多条曲线
main_tag:该图的标签
tag_scalar_dict:key是变量的tag(标题),value是变量的值（相当于scalar_value）

"""

"""
add_histogram(tag,values,global_step=None,bins='tensorflow',walltime=None)
统计直方图与多分位数折线图
tag:图像title
values:要统计的参数
global_step:y轴
bins:取直方图的bins
"""

"""
=================================================================================================
visdom
"""
"""
两个概念
env:环境，不同的环境的可视化结果相互隔离，互补影响，周期使用时如果不指定env,默认使用main,不同的用户，不同程序一般使用不同的env
pane:窗格，窗格可用于可视化图像、数值和打印文本，其可以拖动、缩放、保存和关闭，一个程序可使用同一个env中的不同pane,每个pane可视化或记录某一信息

visdom下,点击save保存json文件，保存路径位于~/.visdom目录下，也可以修改env的名字后电机fork，保存env的状态至更名后的env

vis=visdom.Visdom(env=u'test1') 用于构建一个客户端，客户端除指定env之外，还可以指定host和port等参数
vis作为一个客户端对象，可以使用常见的画图函数，包括：
    line：类似于plot操作
    image:可视化图片，可以是输入的图片，可以是卷积核的信息
    text：用于记录日志等文字信息
    histgram：可视化分布，主要是查看数据、参数的分布
    scatter:散点图
    bar\pie等

visdom 只支持tensor和ndarray的形式，
win：指定pane的名字，如果不指定，自动分配一个新的pane,
opts:选项，接收一个字典，常见的option 包括title,xlabel,ylabel,width等，主要用于设置pane的显示格式

每次操作都会覆盖之前的数值，但是在训练网络的过程中需要不断更新数值，因此需要指定参数uddate='append'来避免覆盖之前的数值，除了使用update参数以外，还可以使用vis.updateTrace方法来更新图，但是updaTrace不仅能在指定pane上新增一个和已有数据相互独立的Trace,还能像update='append'那样在同一条trace上追加数据


vis.image() 可视化一张图像
vis.images() 可视化batch图像
注意输入的图像应该是ndarray的形式


例子：
#commned
pip install visdom
python -m visdom.server

import torch
import visdom
vis=visdom.Visdom(env='test1',use_incoming_socket=False)
x=torch.arange(1,30,0.01)
y=torch.sin(x)
vis.line(X=x,Y=y)
print('sinx')

...还得练
"""