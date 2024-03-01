from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# preparation
writer = SummaryWriter("../log/12_log")
img = Image.open("../data/img/wallhaven-pkx32e.jpg")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
print(type(img_tensor))
writer.add_image("ToTensor", img_tensor, 1)

# Normalize 归一化
# output[channel] = (input[channel] - mean[channel]) / std[channel]
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(type(img_norm))
writer.add_image("Normalize", img_norm, 1)

# Resize
# Resize the input image to the given size
trans_resi = transforms.Resize((200, 100))
img_resize = trans_resi(img)
# Resize输入输出均为 PIL.Image
print(img_resize)
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resize", img_resize_tensor, 1)

# RandomCrop 随机裁剪
trans_randomCrop = transforms.RandomCrop(512)

writer.close()

""""useful_transform
    1、ToTensor——转换为tensor
    2、Normalize——归一化
    3、Resize
    4、RandomCrop
    5、Compose
        torchvision.transforms.Compose([...]) 将多个数据变换组合在一起的函数

"""
