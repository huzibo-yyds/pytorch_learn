from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

# preparation
img_path = "../data/dataset1/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

"""将PIL img转换为tensor"""
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
# Tensor是神经网络所需的数据类型，包含需要的属性

"""将ndarray转换为tensor"""
cv_img = cv2.imread(img_path)
tensor_img_cv = tensor_trans(cv_img)
# 同一张图片，由不同格式转移到tensor


# 使用Tensor类型传入 SummaryWriter展示
writer = SummaryWriter("../log/10_log")
writer.add_image("img_test_1", tensor_img, 1)
writer.add_image("img_test_2", tensor_img_cv, 1)  # 变色
writer.close()

"""transforms
用于对数据预处理
"""
