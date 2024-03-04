import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data/dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


mynn = Mynn()
"""Mynn(
  (conv1): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
)
"""
# print(mynn)

writer = SummaryWriter("../log/18_logs");

step = 0
for data in dataloader:
    img, target = data
    output = mynn(img)
    # torch.Size([64, 3, 32, 32])
    print(img.shape)
    # torch.Size([64, 6, 30, 30])
    print(output.shape)
    writer.add_images("in", img, step)
    # 格式转换，-1部分由自动计算得出
    output_reshape=torch.reshape(output,(-1,3,30,30))
    writer.add_images("out",img_tensor=output_reshape,global_step=step)
    step = step + 1

writer.close()