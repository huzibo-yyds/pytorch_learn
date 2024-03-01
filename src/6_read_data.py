from torch.utils.data import Dataset
from PIL import Image
import os


# 继承自Dataset需要重写__getitem__、__len__
class MyData(Dataset):

    # 构造函数，路径准备
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir) # 合并路径
        self.img_path = os.listdir(self.path) # 以列表的形式获取路径下所有文件名

    # ctrl + O 快捷键重写
    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = '../data/dataset1/train'
ant_label = 'ants'
bee_label = 'bees'

ant_dataSet = MyData(root_dir, ant_label)
bee_dataSet = MyData(root_dir, bee_label)

# 合并训练数据集
train_dataSet=ant_dataSet+bee_dataSet

# test
print(ant_dataSet[0])
img, label = ant_dataSet[0]
img.show()
print(label)
print(len(ant_dataSet))
print(len(bee_dataSet))
print(len(train_dataSet))

"""torch.utils.data.Dataset
展示如何通过重写工具类Dataset将数据加载进来

1、Dataset需要重写 __getitem__、__len__
2、数据与label的关系 文件名是label
"""
