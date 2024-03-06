import torchvision
import torch

# 方式1
from torch import nn

model1 = torch.load("./model/vgg16_model1")
print(model1)

# 方式2
model2 = torchvision.models.vgg16(weights=None)
model2.load_state_dict(torch.load("./model/vgg16_model2"))
print(model2)

""" 注意：自己网络的保存与加载(采用方式2时)
    需要先将网络引进来"""


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


model3 = Mynn()
model3.load_state_dict(torch.load("./model/mynn_save"))
print(model3)
