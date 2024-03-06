# 完整训练过程
import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

device = torch.device("cuda")

# dataset
train_data = torchvision.datasets.CIFAR10(root="../data/dataset_CIFAR10",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data/dataset_CIFAR10",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))

# dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# model
model = Mynn().to(device)

# loss_function
loss_fun = nn.CrossEntropyLoss()

# optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_train_step = 0
# total_test_step = 0
epoch = 100
writer = SummaryWriter("../log/27_train_gpu")

for i in range(epoch):
    print("-----------第 {} 轮训练开始-------------".format(i + 1))

    # train
    model.train()
    for data in train_dataloader:
        imgs, targets = data[0].to(device), data[1].to(device)
        outputs = model(imgs)  # 前向传播
        loss = loss_fun(outputs, targets)  # 计算loss

        # 反向传播
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 计算梯度
        optimizer.step()  # 参数更新

        total_train_step = total_train_step + 1

        if total_train_step % 100 == 0:
            print("训练次数：{},loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # test
    model.eval()  # 将模型设为评估模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 不会浪费计算资源跟踪梯度grad
        for data in test_dataloader:
            imgs, targets = data[0].to(device), data[1].to(device)
            outputs = model(imgs)
            loss = loss_fun(outputs, targets)

            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() # argmax(1) 返回样本预测最大值的索引 参数表示方向
            total_accuracy = total_accuracy + accuracy

    print("整体测试集loss：{}".format(total_test_loss))
    print("整体测试集上accuracy：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, i)
    writer.add_scalar("test_accuracy", (total_accuracy / test_data_size), i)

    # save
    if (total_accuracy / test_data_size) > 0.9:
        torch.save(model, "./model/gpu/mynn_{}.pth".format(i))
        print("模型 {} 已保存".format(i + 1))

writer.close()
