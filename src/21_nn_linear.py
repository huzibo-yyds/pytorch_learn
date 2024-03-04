import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data/dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
# drop_last=True,当最后一页不满足batch_size，则丢弃
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        # 64 × 3 × 32 × 32 = 196608
        self.leaner = Linear(196608, 10)

    def forward(self, input):
        output = self.leaner(input)
        return output


mynn = Mynn()
"""Mynn(
  (leaner): Linear(in_features=196608, out_features=10, bias=True)
)
"""
# print(mynn)

for data in dataloader:
    imgs, targets = data
    # torch.Size([64, 3, 32, 32])
    print(imgs.shape)
    # 展开返回一维张量 https://pytorch.org/docs/stable/generated/torch.flatten.html#torch-flatten
    output = torch.flatten(imgs)
    # torch.Size([196608])
    print(output.shape)
    output = mynn(output)
    # torch.Size([10])
    print(output.shape)
