import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# 检查是否有可用的 GPU，并将模型和数据移动到 GPU 上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# 将模型移动到 GPU 上
mynn = Mynn().to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(mynn.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data[0].to(device), data[1].to(device)  # 将数据移动到 GPU 上
        outputs = mynn(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss.item()  # 记得使用 result_loss.item() 获取标量损失值
    print(running_loss)

print("GPU train finish!")