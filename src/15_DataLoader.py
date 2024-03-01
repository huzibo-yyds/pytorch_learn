import torchvision
from torch.utils.data import DataLoader

# 使用pytorch官方提供数据集
train_dataset = torchvision.datasets.CIFAR10("../data/dataset_CIFAR10", train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)

"""DataLoader参数说明 https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"""
test_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
img, target = train_dataset[0]
print(img.shape)
print(target)

# dataloader第一个。错误！！！ DataLoader不可索引，仅可迭代
# imgs,targets=test_loader[0]
# print(imgs.shape)
# print(targets)
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)

"""DataLoader总结
    相当于对DataSet进行了一步打包，和预处理
    
    ！注意
        1、 DataLoader不可索引，仅可迭代
"""
