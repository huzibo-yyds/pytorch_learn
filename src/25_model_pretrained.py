"""" 使用现有模型训练 """

# ImageNet 训练集100G+，且不支持公开访问
# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
import torchvision
from torch import nn
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(weights=None)
vgg16_ture = torchvision.models.vgg16(weights="DEFAULT")

print(vgg16_ture)

# 在现有网络中添加新层
vgg16_ture.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_ture)

# 更改现有网络中的指定层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

dataset = torchvision.datasets.CIFAR10("../data/dataset_CIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

"""获取VGG16的模型做前置网络，供CIFAR10使用
1、如何获得 训练好/未训练 的模型
2、如何修改现有模型，供己使用
"""
