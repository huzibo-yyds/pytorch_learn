import torch
from torch import nn
import torch.nn.functional as m

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

print(input.shape)  # torch.Size([5, 5])
print(kernel.shape) # torch.Size([3, 3])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
# reshape作用：改变张量的形状（shape），以适应卷积操作的输入要求
print(input) # 4维？
print(kernel)

ouput=m.conv2d(input,kernel,stride=1)
print(ouput)
ouput=m.conv2d(input,kernel,stride=2)
print(ouput)
ouput=m.conv2d(input,kernel,stride=1,padding=1)
print(ouput)


"""conv2d参数说明
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) 
    + input、weight 指定形状的输入张量，与滤波器
    + stride 卷据和的步幅，可以是单数字，或元组（分别表示横向，纵向）
    + stride 填充
"""


# With square kernels and equal stride
filters = torch.randn(8, 4, 3, 3)
inputs = torch.randn(1, 4, 5, 5)
print(filters)
print(inputs)

print(m.conv2d(inputs, filters, padding=1))