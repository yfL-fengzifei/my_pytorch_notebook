# 3.1 linear regrassion

>线性回归假设输出与输入之间是线性关系
>
>模型输出是线性回归对真值的预测或估计，允许有一定的误差

## 3.1.1 model training
>目标：通过数据寻找特定的模型参数值，使模型在数据上的误差尽可能的小
>
>three elements

### one / elements - training data set or training set
>sample 样本
>
>label 标签
>
>feature 特征 ； 用来预测标签的两个因素， 用来表征样本的特点
>
>一个样本对应一个标签（真实价格）
>
>n 样本数； i 第i个样本的索引

### loss function / cost function - square loss function
>线性回归一般使用的是平方损失函数
>
>如果给定训练集training set，则loss function 至于模型参数有关
>
>更一般的情况，evaluate the model 使用的是mean square loss function，即所有样本的误差平均来衡量

### optimization method
>当模型和损失函数比较简单的时候上述误差最小化的问题可以直接用公式表达出来，即具有解析解
>
>大多数的深度学习模型，只能通过优化算法有限次迭代来尽可能降低损失函数的值，即求数值解

优化算法其中之一就是 小批量随机梯度下降 mini-batch stochastic gradient descent
>先选取一组模型参数的初始值，如随机选取，然后对参数进行多次迭代，是每次迭代都可能降低损失函数的值，
>在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量mini-batch，然后求小批量中
>数据样本中的平均损失函数有关模型参数的倒数-梯度，最后用此结果与预先设定的一个正数乘积
>作为模型参数在本次迭代的减小量

hyper-parameter超参数 fine tune微调
>人为设定得参数为超参数，而不是由模型训练得到的。
>
>一般调参调的就是这些超参数，在少数情况下可以通过模型训练学出来。

##  3.1.2 model prediction
>模型预测、模型推理、模型测试

**fully connected layer 也叫作dense layer 稠密层**

```python
import torch
from time import time # 从time module中导入time类

# 定义1000维的向量
a = torch.ones(1000)
b = torch.ones(1000)

# 计算运行的时间
# start=time()
# print(time() - start)
# 利用循环实现向量的相加
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
# c = a + b
# print(c) # 这种方法是错误的，不能得到tensor,必须先开辟一个内存
print(time() - start)

# 向量化
start2=time()
c_2 = a + b # 这种属于直接复制，不需要提前声明变量和开辟内存
print(time() - start2)

# 广播机制
# 即某些情况下可以将大小不同的两个tensor进行相加
a3 = torch.ones(3)
b = 10
print(a3+b)
```

