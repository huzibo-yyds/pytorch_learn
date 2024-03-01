import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 输入层到隐藏层的全连接层，输入大小为28*28=784，隐藏层大小为128
        self.relu = nn.ReLU()  # 隐藏层激活函数使用ReLU
        self.fc2 = nn.Linear(128, 10)  # 隐藏层到输出层的全连接层，输出大小为10（代表10个数字类别）

    # 前向传播
    def forward(self, x):
        x = x.view(-1, 784)  # 将输入数据展平成一维向量
        x = self.fc1(x)  # 输入层到隐藏层的线性变换
        x = self.relu(x)  # 隐藏层的ReLU激活函数
        x = self.fc2(x)  # 隐藏层到输出层的线性变换
        return x


# 加载MNIST数据集并进行预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像数据转换为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 对图像进行归一化
])

train_dataset = datasets.MNIST(root='../data/data_MNIST', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 实例化神经网络模型、损失函数和优化器
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

# 训练神经网络模型
epochs = 5
train_losses = []  # 用于存储每个epoch的训练损失
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # 梯度清零
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

print('Training finished.')
