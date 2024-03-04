import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data/dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.maxpooling1=MaxPool2d(kernel_size=3,ceil_mode=False)
        # 参数说明 https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

    def forward(self, input):
        output = self.maxpooling1(input)
        return output

mynn = Mynn()
"""Mynn(
  (maxpooling1): MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
)
"""
# print(mynn)

writer = SummaryWriter("../log/19_logs_maxpooling");
step = 0

for data in dataloader:
    img,target=data
    writer.add_images("in",img_tensor=img,global_step=step)
    output=mynn(img)
    writer.add_images("out",img_tensor=output,global_step=step)
    step=step+1

writer.close()