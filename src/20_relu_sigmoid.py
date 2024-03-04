import torch
import torchvision
# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])
#
# input = torch.reshape(input, (-1, 1, 2, 2))
# print(input.shape)
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data/dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        # self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        # output = self.relu(input)
        output = self.sigmoid(input)
        return output


mynn = Mynn()
"""Mynn(
  (relu): ReLU()
  (sigmoid): Sigmoid()
)
"""
# print(mynn)

writer = SummaryWriter("../log/20_logs");
step = 0

for data in dataloader:
    img, target = data
    writer.add_images("in", img_tensor=img, global_step=step)
    output = mynn(img)
    writer.add_images("out", img_tensor=output, global_step=step)
    step = step + 1

writer.close()
