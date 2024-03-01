import torch
from torch import nn


class Mynn(nn.Module):

    def __init__(self):
        super().__init__()

    # 前向传播
    def forward(self, x):
        x = x + 1
        return x


# 实例化神经网络类
mynn = Mynn()

# 创建一个输入张量
x = torch.tensor(1.0)
x = mynn(x)
print(x)  # tensor(2.)

"""极为简单

为了构造了一个简单的神经网络
    1、调用父类的构造函数
    2、需要重写forward——定义调用网络时执行的运算

"""
