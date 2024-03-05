import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../data/dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model(input)
        return output

mynn = Mynn()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    outputs = mynn(imgs)
    result_loss = loss(outputs, targets)
    # print("ok")

    # 经过nn后预测结果，与真实结果
    print(outputs)
    print(targets)
    print(result_loss)