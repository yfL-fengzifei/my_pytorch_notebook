# knowledge about softmax
"""
对于离散值得预测问题，可以使用包括softmax回归的在内的分类模型，
softmax回归的输出单元从一个变成了多个，且引入了softmax运算，使得输出更适合离散值的预测和训练
"""

# classification problem and softmax regression 分类问题 与 softmax回归
"""
一般使用离散的数值来表示类别，例如y1=1,y2=2...,最不可取的方法就是利用回归模型将预测值就近定点化到1,2,3这三个离散值中之一

softmax回归于线性回归一个最主要的不用就是，softmax回归的输出值个数等于标签里的类别数，本质上来讲，softmax回归层也是一个全连接层

对于上述分类问题，softmax回归输出值Oi，当做预测类别i的置信度，并将值最大的输出所对应的类作为预测输出，即输出arg max(Oi)
但是直接使用输出层的输出有两个问题，一方面，由于输出值得范围不确定，因此很难直观上判断这些值得意义(就是什么样的值叫做高)，另一方面，由于真实标签是离散值，这些离散值与不确定范围的输出值之前的误差难以衡量

针对上述的问题，softmax运算符就可以有效的解决上面的问题，即softmax函数将输出值转变成：值为正，且，和为1的概率分布,hat(y1),hat(y2),hat(y3)=softmax(O1,O2,O3),这里不列出公式

"""

# batch 计算
"""
batch_size=n
features=d # features in one sample, the inputs of network's layer
num_labels=q # the number of ground truth classess
W| W.size()=[d,q] # weights of the layer
b| b.size()=[1,q] #bias of the layer
X| X.size()=[n,d] #features of the batch samples 

O=X*W+b #实际上更常用的是O=W*X #W.size()=[q,d], X.size()=[d,n]
hat(y)=softmax(O)
"""

# crossentropy_loss_fn and prediction 交叉熵损失含函数 与 预测
"""
上述softmax的最终输出是一个类别预测分布
真实类别标签页也可以用类别预测分布，[0,0,...,1,0,0,...]

想要预测分类结果正确，并不需要续页概率完全等于标签概率，如果使用线性归回中的MSELoss损失函数归于严格
cross entropy函数可以衡量两个概率分布差异。交叉熵只关心正确类别的预测概率，只要其值足够大，就可以确保分类结果的正确

需要注意的损失对于batch的交叉熵函数与单一样本的交叉熵函数有所不同，这里没有列出函数，需要进一步查看
最小化交叉熵函数等价于最大化训练数据集所有标签类别的联合预测概率
"""