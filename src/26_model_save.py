import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(weights="VGG16_Weights.DEFAULT")

# 方式1
torch.save(vgg16, "./model/vgg16_model1")

# 方式2
torch.save(vgg16.state_dict(), "./model/vgg16_model2")

""""注意 自己构造的神经网络使用方式2保存时问题"""


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


mynn = Mynn()
torch.save(mynn.state_dict(), "./model/mynn_save")
