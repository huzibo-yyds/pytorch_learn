What I cannot create, I do not understand
不闻不若闻之，闻之不若见之，见之不若知之，知之不若行之
共勉

| 序号 | 资源名 | 资源地址 |
| --- | --- | --- |
| 1 | PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】 | [https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.999.0.0](https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.999.0.0) |
| 2 | 代码地址 | [https://github.com/huzibo-yyds/pytorch_learn](https://github.com/huzibo-yyds/pytorch_learn) |
| 3 | 样例数据集-蚂蚁蜜蜂 | [https://download.pytorch.org/tutorial/hymenoptera_data.zip](https://download.pytorch.org/tutorial/hymenoptera_data.zip) |
| 4 | pytoch官网 | [https://pytorch.org/](https://pytorch.org/) |
| 5 | pytoch官方文档 | [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html) |
| 6 | 工具经验：关闭pycharm补全时大小写匹配 | [https://jingyan.baidu.com/article/8cdccae986787f705513cd45.html](https://jingyan.baidu.com/article/8cdccae986787f705513cd45.html) |




:::warning
【主要内容】

1. 环境配置
anaconda、pytorch、pycharm、jupyter
2. Pytorch加载数据（以蚂蚁图片为例）。
   1. Dataset、Dataloader使用
   2. 如何使用pytorch官方数据集
3. 数据预处理相关函数、库的使用
   1. TensorBoard——工具，用于tensor的可视化
   2. transform——对数据预处理
4. 神经网络 neural network
   1. 输入层
   2. 隐藏层
      1. 卷积层、池化层、归一层、线性层
   3. 输出层
5. 如何定义自己的神经网络
6. 如何使用现有网络，及其修改(网络迁移)
7. 完整神经网络训练过程
   1. CPU
   2. GPU
:::

---


### 环境配置

- anaconda
- Cuda
- pytorch
- jupyter（在conda环境下再次安装）
```powershell
# 创建虚拟环境
conda create -n pytorch_2.1.2_GPU python=3.9
# 启用环境
conda activate pytorch_2.1.2_GPU
# 下载pytorch|CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# 查看nvidia驱动版本
nvidia-smi
```

面对陌生库2个常用函数
```python
# dir列出对象的所有属性及方法
dir(torch.cuda.is_available)
# help
help(torch.cuda.is_available)
```

### pytorch加载数据Dataset

- Dataset
为数据，编号、label
- Dataloader
为网络提供不同的数据形式

> 【样例数据集-蚂蚁蜜蜂】
[https://download.pytorch.org/tutorial/hymenoptera_data.zip](https://download.pytorch.org/tutorial/hymenoptera_data.zip)


【数据与label的几种方式】

1. 文件名是label
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709174904625-46f62e72-d769-4015-918c-3a38dc07fc6e.png#averageHue=%23353e45&clientId=uf406e2d9-da11-4&from=paste&height=77&id=uef9cc2bd&originHeight=96&originWidth=245&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=6115&status=done&style=stroke&taskId=u07ef7787-c63f-4d88-9b74-44ea264085d&title=&width=196)
2. 一个目录存放数据，另一个目录存放数据对应label
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709174934362-48f7a5f0-cd12-4c0b-a52c-89119086d787.png#averageHue=%233f4346&clientId=uf406e2d9-da11-4&from=paste&height=97&id=uf1b2aad3&originHeight=121&originWidth=281&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=8388&status=done&style=stroke&taskId=ucbdd2f01-2cbb-4931-b9e5-96af579b4d1&title=&width=224.8)
3. label直接标注在数据上

### TensorBoard
训练模型时的**可视化工具 | **[TensorBoard使用举例](https://github.com/huzibo-yyds/pytorch_learn/blob/master/src/8_tensorboard.py)
TensorBoard是一个由TensorFlow提供的**可视化工具**，用于帮助机器学习工程师和研究人员可视化、理解和调试他们的模型训练过程。它提供了各种可视化功能，可以帮助用户深入了解模型的性能、结构和训练过程。通过TensorBoard提供的这些可视化功能，用户可以更直观地了解模型的训练过程和性能，从而更好地调试模型、优化超参数，并作出更加合理的决策。
`from torch.utils.tensorboard import SummaryWriter`

【如何启动】

1. 确保安装TensorFlow
2. 在python脚本中使用torch.utils.tensorboard.SummaryWriter类来将数据写入到TensorBoard中
3. 在命令行中（此处为conda中），进入到你的Python脚本所在的目录，然后执行以下命令启动TensorBoard
`tensorboard --logdir=logs`📍
4. 打开TensorBoard的Web界面，查看可视化信息


说明，同一tag下的数据被叠加

```
add_scalar(self,
    tag,
    scalar_value,
    global_step=None,
    walltime=None,
    new_style=False,
    double_precision=False,
)
```

【效果】
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709178034607-dddd8377-2762-4493-84f2-076fad8e2f3a.png#averageHue=%23443a32&clientId=uf406e2d9-da11-4&from=paste&height=770&id=ub00fc6a8&originHeight=963&originWidth=1920&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=119108&status=done&style=stroke&taskId=u751eb4ce-0a7a-455b-9703-3af4f3a15c3&title=&width=1536)

tensorboard 功能汇总

- Scalars：用于可视化训练过程中的标量数据，如损失函数值、准确率等。
- Graphs：用于可视化神经网络模型的计算图结构。
- Histograms：用于可视化张量的分布情况，如权重、偏置项等。
- Images：用于可视化图像数据。
- Embeddings：用于可视化高维数据的降维结果。
- Profiler：用于性能分析和调试。
- Projector：用于可视化嵌入向量。

【常用函数】

- `SummaryWriter`：用于创建一个 TensorBoard 日志文件，将需要可视化的数据写入到日志文件中。🟥
- `add_scalar`：用于添加标量数据到 TensorBoard 中，例如损失函数值、准确率等。🟥
- `add_graph`：用于添加计算图到 TensorBoard 中，可视化神经网络模型的结构。
- `add_histogram`：用于添加张量的直方图到 TensorBoard 中，可视化参数的分布情况。
- `add_image`：用于添加图像数据到 TensorBoard 中，可视化图像。🟥
- `add_embedding`：用于添加嵌入向量到 TensorBoard 中，进行降维可视化



### transforms
用于对数据预处理🟥| [transforms举例](https://github.com/huzibo-yyds/pytorch_learn/blob/master/src/10_transforms.py)

`transforms`是一个用于对**数据进行预处理和数据增强的模块**。它包含了一系列用于处理图像、文本、音频等数据的转换函数，可以在数据加载过程中动态地对输入数据进行变换，以满足模型训练的需求。
在计算机视觉任务中，transforms模块通常用于对图像数据进行预处理和增强，包括裁剪、缩放、旋转、翻转、归一化等操作。通过对输入数据进行适当的变换，可以提高模型的鲁棒性和泛化能力，同时还可以减少模型的过拟合风险。

![形象化transfroms使用](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709186794102-9ca46260-a9ce-456e-994c-8ec6cae790e2.png#averageHue=%23f7f7f7&clientId=uf406e2d9-da11-4&from=paste&height=676&id=u4bad3d4e&originHeight=845&originWidth=1202&originalType=binary&ratio=1.25&rotation=0&showTitle=true&size=158689&status=done&style=stroke&taskId=ufd87776d-7a96-451d-bf0a-729562809d1&title=%E5%BD%A2%E8%B1%A1%E5%8C%96transfroms%E4%BD%BF%E7%94%A8&width=961.6 "形象化transfroms使用")

【举例，将PIL导入的img转换为tensor，并在tensorboard展示】
```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

"""对数据预处理"""

img_path = "dataset1/train/ants/0013035.jpg"
img = Image.open(img_path) #利用PIL


"""transforms 用法"""
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

"""Tensor是神经网络所需的数据类型，包含需要的属性"""

cv_img = cv2.imread(img_path) #利用opencv
tensor_img_cv = tensor_trans(cv_img)
# 同一张图片，由不同格式转移到tensor


"""使用Tensor类型传入 SummaryWriter展示"""
writer = SummaryWriter("logs_img")
writer.add_image("img_test_1", tensor_img, 1)
writer.add_image("img_test_2", tensor_img_cv, 1)  # 变色

writer.close()
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709188187739-cd4ec464-1f82-4dc8-aee2-1aad6f837b46.png#averageHue=%23e39855&clientId=uf406e2d9-da11-4&from=paste&height=654&id=uce6c9a77&originHeight=818&originWidth=502&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=226386&status=done&style=stroke&taskId=u6a6352a4-aeae-4ff9-a97b-f83d0bea0e5&title=&width=401.6)

【transforms 常用类】

- ToTensor（将图片转换为tensor类型）
Convert a PIL Image or ndarray to** tensor** and scale the values accordingly.
- Normalize 归一化（对每个通道上的像素变换）
# output[channel] = (input[channel] - mean[channel]) / std[channel]

### torchvison官方数据集使用

- 视觉相关数据集
[https://pytorch.org/vision/stable/datasets.html#](https://pytorch.org/vision/stable/datasets.html#)
- CIFAR-10 数据集 示例
[https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)

> 1. 获得官方数据集，并预处理为tensor
> 2. 使用tensorboard展示训练集中的图片

```python
import torchvision
# 包含视觉相关的数据集
from torch.utils.tensorboard import SummaryWriter

dataset_transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

""" https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html """
train_dataset=torchvision.datasets.CIFAR10("./dataset_CIFAR10",train=True,transform=dataset_transform,download=True)
test_dataset=torchvision.datasets.CIFAR10("./dataset_CIFAR10",train=False,transform=dataset_transform,download=True)

# print(train_dataset[0])

wirter=SummaryWriter("p10")
for i in range(len(train_dataset)):
    img,label=train_dataset[i]
    wirter.add_image("train_dataset",img,i)
wirter.close()
```

### Dataloder使用
[https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

❓dataset与dataloader分别是什么？有什么区别？

1. _**Dataset
**_**表示**数据集，可以包含多个样本
- Dataset是一个抽象类，用于_表示数据集_。它定义了数据集的基本操作，包括数据的加载、预处理和索引等
- 你可以创建自定义的Dataset类，继承自torch.utils.data.Dataset，并实现`__len__()`和`__getitem__()`方法，以便能够对数据集进行索引和加载——需要重写2个方法
2. **_Dataloader
_**训练模型时，用于**加载**训练数据集
- DataLoader是一个用于批量加载数据的类，它封装了Dataset并提供了多线程 数据加载、数据打乱、数据批处理等功能。
- 你可以创建一个DataLoader对象，将Dataset作为参数传入，然后通过设置batch_size、shuffle等参数来对数据进行批处理和打乱。


```python
import torchvision
from torch.utils.data import DataLoader

train_dataset = torchvision.datasets.CIFAR10("./dataset_CIFAR10", train=True,
                                             transform=torchvision.transforms.ToTensor(), download=True)

"""DataLoader参数说明 https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"""
test_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# DataLoader不可索引，仅可迭代
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)

"""DataLoader总结
相当于对DataSet进行了一步打包，和预处理
"""
```

### nn.Model
(Neural Network 简单介绍入门)
:::info
简单理解nn
神经元(节点)，由感知机引入。
是一种由神经元 （或称为节点）组成的计算模型，它受到人类大脑中神经元之间相互连接的启发。神经网络由多层神经元组成，分为**输入层**、**隐藏层**和**输出层**，每一层都由多个神经元组成。神经网络的基本单位是神经元，它_接收来自前一层的输入信号，进行加权求和并通过激活函数处理，然后将输出传递给下一层_。

【重要概念】

1. **输入层（Input Layer）**：接收输入数据的层，每个输入神经元对应输入数据的一个特征。
2. **隐藏层（Hidden Layer）**：位于输入层和输出层之间的层，用于提取输入数据的特征并进行非线性变换。神经网络可以包含多个隐藏层，每个隐藏层可以包含多个神经元。
3. **输出层（Output Layer）**：产生神经网络的输出结果的层，通常对隐藏层的输出进行加权求和并通过激活函数处理，得到最终的输出结果。
4. **权重（Weights）**：连接神经元之间的边上的参数，用于调整输入信号的重要性。
5. **偏置（Bias）**：每个神经元都有一个偏置参数，用于调整神经元的激活阈值，影响神经元是否被激活。
6. **激活函数（Activation Function）**：对神经元的输入进行非线性变换的函数，常见的激活函数包括_Sigmoid、ReLU、Tanh_等。
7. **损失函数（Loss Function）**：衡量神经网络输出与真实标签之间的差异的函数，用于评估模型的性能。
8. **反向传播（Backpropagation）**：一种**训练神经网络的方法**，通过计算损失函数对模型参数的梯度，并利用梯度下降算法来更新模型参数，使得模型的预测结果逐渐接近真实标签。
:::
[https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)——官方文档

![神经网络由感知机发展而来](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709884354942-538207ca-ce04-4c69-a3cc-b36842feef8e.png#averageHue=%23fafafa&clientId=u90251f36-a9ed-4&from=paste&height=922&id=u362fdf75&originHeight=1152&originWidth=2081&originalType=binary&ratio=1.25&rotation=0&showTitle=true&size=265896&status=done&style=stroke&taskId=uaa15f4e4-6391-47ce-87a3-a3188673814&title=%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%94%B1%E6%84%9F%E7%9F%A5%E6%9C%BA%E5%8F%91%E5%B1%95%E8%80%8C%E6%9D%A5&width=1664.8 "神经网络由感知机发展而来")

### 神经网络
#### 卷积运算
【卷积是什么】输出 = 输入 * 系统

卷积操作可以看作是一种**滤波器**在**输入数据上滑动**并与输入数据进行逐元素相乘后求和的过程。具体而言，对于二维输入数据（如图像），卷积操作可以通过一个称为卷积核（或滤波器）的小矩阵来实现，卷积核的大小通常是3x3或5x5等。卷积核在输入数据上滑动，对每个位置的局部区域与卷积核进行相乘并求和，得到输出数据的一个值。如下图：
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709275766652-aa355fd1-15cf-4654-b795-d17c63a74700.png#averageHue=%23fcfcfc&clientId=uada05f33-b3e1-4&from=paste&height=447&id=u625532b5&originHeight=559&originWidth=1071&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=132394&status=done&style=stroke&taskId=u79178576-a280-4d72-96b1-a65453305e2&title=&width=856.8)
```python
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

ouput=m.conv2d(input,kernel,stride=1)
print(ouput)

"""
tensor([[[[10, 12, 12],
          [18, 16, 16],
          [13,  9,  3]]]])
""""
```


【conv2d】对由多个输入平面组成的输入图像应用 2D 卷积
**torch.nn.functional.conv2d(**_**input**_**, **_**weight**_**, **_**bias=None**_**, **_**stride=1**_**, **_**padding=0**_**, **_**dilation=1**_**, **_**groups=1**_**) **
[https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch.nn.functional.conv2d)


:::tips
卷积操作的输入形式🟥：
_**input**_输入张量的形状通常为 `(batch_size, channels, height, width)`

- batch_size 是批量大小，表示一次处理的样本数量。
- channels 是通道数，表示输入数据的通道数量，例如彩色图像有 RGB 三个通道。
- height 是图像的高度，表示图像的行数。
- width 是图像的宽度，表示图像的列数。

**_weight_**卷积核张量的形状通常为 `(out_channels, in_channels, kernel_height, kernel_width)`

- out_channels 是输出通道数，表示卷积核的个数。
- in_channels 是输入通道数，表示每个卷积核的输入通道数。
- kernel_height 是卷积核的高度，表示卷积核的行数。
- kernel_width 是卷积核的宽度，表示卷积核的列数。
:::

#### 卷积层-convolution

- [Convolution Layers](https://pytorch.org/docs/stable/nn.html#convolution-layers)
- 【卷积运算——详细解释了stride、padding】
[conv_arithmetic/README.md at master · vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md)

> - **卷积操作**的基本原理是将一个卷积核（或滤波器）与输入数据进行逐元素相乘，并将结果相加，得到输出特征图中的一个单个值。
>    - **卷积核的参数**是需要_学习_的，通过反向传播算法进行优化
> - **卷积层**，属于神经网络中（隐含层）的一种，用于提取输入数据中的局部特征，并保留空间结构


📍2024年3月1日15点58分 18P
举例：_**CLASS **_**torch.nn.Conv2d**
**神经网络中，进行卷积计算，只需要定义好卷积层即可。注意输入输出的数据尺寸要符合。**

##### 【卷积前后尺寸计算】：🟥
[官方-torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709517558182-f8a6c300-6420-477b-9716-56b2483108a4.png#averageHue=%23fbf9f8&clientId=u8e74c415-d8b1-4&from=paste&height=253&id=u87a9f2c7&originHeight=253&originWidth=712&originalType=binary&ratio=1&rotation=0&showTitle=false&size=31036&status=done&style=stroke&taskId=u94350b89-6804-48e5-83bf-9b218afdf5a&title=&width=712)

感性理解计算：
padding时，是上，下，左，右四个方向填充。根据卷积核的滑动，来判断卷积后的尺寸。

---

#### 池化层-pooling

- [官方-pooling layer](https://pytorch.org/docs/stable/nn.html#pooling-layers)

> - 池化层，与卷积层一样都属于隐藏层的一种
> - 池化操作，对输入特征图的局部进行聚合来**降低空间维度**



【池化】
:::tips
Pooling定义：一种常用神经网络操作。通常用于**减少特征图的空间维度**，从而降低模型复杂度、减少计算量，并且提取特征的位置不变性。
池化操作通常在卷积神经网络的卷积层之后进行，它通过对输入特征图的局部区域进行**聚合**操作来**降低特征图的空间维度**。常用的池化操作有最大池化（Max Pooling）和平均池化（Average Pooling）两种。

- 最大池化中，对于每个局部区域，取该区域内的最大值作为输出
- 平均池化中，对于每个局部区域，取该区域内的平均值作为输出

池化操作通常通过设置池化窗口大小和步幅来控制输出特征图的大小。
作用：降维、减少计算量、提取特征不变性
🟥池化操作通常与卷积操作交替使用，用于逐步提取和降维输入数据的特征。
:::
> 鲁棒性（Robustness）指的是系统在面对异常情况或不良输入时仍能保持稳定性和正确性的能力

![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709519241255-c260dddc-b721-46c2-942f-67480a023ee2.png#averageHue=%23efefec&clientId=u8e74c415-d8b1-4&from=paste&height=306&id=ubeed904f&originHeight=306&originWidth=742&originalType=binary&ratio=1&rotation=0&showTitle=false&size=58659&status=done&style=stroke&taskId=ub93bec29-c195-453d-9463-e7ff377f760&title=&width=742)

#### 非线性激活

- [【官方】non-linear-activations（weighted-sum、nonlinearity）](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

> 为什么是非线性？
> 
> 非线性i激活的


:::tips
【定义】：
非线性激活函数（**Non-linear Activation Function**）是指一类函数，它们引入了非线性变换，使得神经网络能够学习和表示更加复杂的非线性关系。
在神经网络的每一层中，通常会在线性变换（如全连接层或卷积层）后添加一个非线性激活函数，以增加网络的表达能力。没有非线性激活函数的神经网络将由一系列线性变换组成，无法表示复杂的非线性关系，因此引入非线性激活函数可以让神经网络具备更强的表达能力。

【常见】
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709520673744-b3f41f69-7438-428f-a821-5f40fb8154a0.png#averageHue=%232f2d2a&clientId=u8e74c415-d8b1-4&from=paste&height=460&id=ufe01b9d7&originHeight=460&originWidth=618&originalType=binary&ratio=1&rotation=0&showTitle=false&size=67453&status=done&style=stroke&taskId=u32cc7148-0a8b-4ee1-bb0d-92ff4b3cc71&title=&width=618)

【作用】高了模型的表达能力和性能
:::

:::danger
【常见非线性激活函数】

- [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU)
- [Sigmoid](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html#torch.nn.Sigmoid)
:::
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709532271345-c0e51d12-601b-478c-a993-28246e6105d2.png#averageHue=%234d4a46&clientId=u8e74c415-d8b1-4&from=paste&height=357&id=u22b2ea66&originHeight=357&originWidth=810&originalType=binary&ratio=1&rotation=0&showTitle=false&size=235015&status=done&style=stroke&taskId=u03c4081f-2562-41e8-89fe-c02deea069f&title=&width=810)

#### 归一化层（Normalization Layers）
[normalization-layers](https://pytorch.org/docs/stable/nn.html#normalization-layers)
:::tips
【定义】
用于在神经网络中对输入数据或隐藏层的输出进行标准化处理，以使得数据的分布更稳定、更易于训练
通常在神经网络的隐藏层之后、激活函数之前使用。以确保输入数据或隐藏层的输出在经过激活函数之前已经进行了标准化处理

- 批量归一化（Batch Normalization）
- 层归一化（Layer Normalization）

【作用】加快训练速度、提高模型鲁棒性，改善激活函数效果
:::

📍归一化Normalization ≠ 正则化Regularization

- 归一化
   - 对输入数据或隐藏层的输出进行标准化处理，使得数据的分布更稳定、更易于训练
- 正则化
   - 正则化主要是用于控制模型的复杂度，防止模型过拟合训练数据，提高模型的泛化能力

#### 线性层-linear layers
[linear-layers](https://pytorch.org/docs/stable/nn.html#linear-layers)
$\text { output }=\text { activation }(\text { input } \times \text { weight }+ \text { bias })$

[_**CLASS**_**torch.nn.Linear**](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)


> 📍对过拟合的说明
> 过拟合（Overfitting）是指机器学习模型在训练过程中过度拟合训练数据，导致模型在训练集上表现良好，但**_在测试集或未见过的数据上表现较差_**的现象



#### Sequential
[https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential)
Sequential 是一个用于构建神经网络模型的**容器**。它允许你**_按顺序堆叠一系列的神经网络层_**，构建一个顺序模型。Sequential 容器提供了一个简单而直观的方式来构建神经网络，特别是对于那些由一系列层依次组成的简单模型。

——将一系列网络堆叠起来

#### 损失函数-Loss Functions
> 简言之，计算预测结果与实际标签之间的差异

- [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
:::tips
定义：于衡量模型预测结果与实际标签之间的差异或误差，并且是优化算法的目标函数之一。

常见损失函数

- [均方误差](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)（Mean Squared Error，MSE）/ 平方差
- [交叉熵损失](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)（Cross-Entropy Loss）
- 对数损失（Log Loss）
- Hinge损失

作用
评估模型性能、指导模型优化（优化算法目标是最小化损失函数）、指导模型学习（损失函数为模型提供反馈信号）
:::

![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709626115722-9ea6f71b-653a-4b1e-95ee-a2fe9524d4fd.png#averageHue=%23fafafa&clientId=u50b636d6-f044-4&from=paste&height=319&id=uf8bfe909&originHeight=400&originWidth=752&originalType=binary&ratio=1&rotation=0&showTitle=false&size=82334&status=done&style=stroke&taskId=u56a42184-681f-4343-8474-12805c19e2e&title=&width=600)


#### 反向传播
:::tips
定义：深度学习中用于训练神经网络的一种优化算法。它通过计算损失函数对模型参数的梯度，并使用梯度下降或其变种来更新参数，从而使得模型能够逐渐优化和拟合训练数据。
基本思想是利用**链式法则（Chain Rule）**来计算损失函数对模型参数的梯度
反向传播算法步骤

1. 前向传播（Forward Propagation）
从输入数据开始，通过神经网络前向计算，_逐层计算_出输出，并最终得到模型的**预测结果**
2. 计算损失函数（compute Loss）
将模型预测结果与真实标签值比较，**计算损失函数**，用于衡量模型预测结果与真实标签的差异
3. 反向传播（Backward Propagation）
从损失函数开始，利用**链式法则**计算损失函数对模型参数的**梯度**。通过逆向遍历神经网络计算图，将损失函数的梯度沿着网络的每一条边传播回去，并更新每个参数的值
4. 参数更新（Update Parameters）
根据计算得到的**参数梯度**，使用梯度下降或其变种的算法来更新模型的参数，使**_损失函数逐渐减小，模型逐渐优化和拟合训练数据_**。

关键：利用链式法则来计算梯度，从而实现了高效的参数更新和模型优化
:::


:::info
🟥[链式法则](https://baike.baidu.com/item/%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99/3314017)（Chain Rule）
微积分中的一个基本概念，它描述了复合函数的导数如何计算的规则。在数学上，如果一个函数是另一个函数的复合，那么它们的导数之间存在特定的关系
——微积分中的求导法则
$h(x)=f(g(x))$ 复合函数导函数->$\frac{d h}{d x}=\frac{d h}{d u} \cdot \frac{d u}{d x}$

- 链式法则在反向传播算法中扮演了重要角色，通过_计算损失函数对每个参数的梯度_，并将梯度沿着网络反向传播，以便调整参数值以最小化损失函数
- 在神经网络中，每一层的输出都是通过对输入进行一系列的线性变换和非线性激活函数的计算得到的。通过链式法则，可以计算出损失函数对每一层输出的**梯度**，然后利用梯度下降法来更新网络参数，以减小损失函数
- 链式法则在反向传播算法中扮演了至关重要的角色，它使得我们能够有效地计算复杂网络结构中每个参数的梯度，并用于优化网络参数，从而提高神经网络的性能。

📍**梯度grand**：梯度是损失函数相对于模型参数的偏导数
梯度告诉我们，如果我们稍微改变模型参数的值，损失函数会如何改变。梯度的方向指示了损失函数增加最快的方向，而梯度的大小则表示了损失函数增加的速率。
:::
#### 优化器 optimizer

- [optimizer](https://pytorch.org/docs/stable/optim.html)
:::tips
定义：机器学习和深度学习中用于优化模型参数的算法
沿着损失函数的负梯度方向移动参数值，从而使得损失函数逐渐减小
![20001976tHmwPv6YYG.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709626704263-4d619d3a-2f83-4484-9c4c-cf0f29d05ba8.png#averageHue=%23f6f6f6&clientId=u50b636d6-f044-4&from=paste&height=215&id=u696771e0&originHeight=377&originWidth=700&originalType=binary&ratio=1&rotation=0&showTitle=false&size=64996&status=done&style=stroke&taskId=ua0bef300-db5d-463c-81b6-41c0cf1176a&title=&width=400)
作用：根据模型参数的梯度信息，更新模型参数，以最小化[损失函数](# 损失函数-Loss Functions)
工作流程：

1. 计算梯度
首先，使用反向传播算法（backword）计算损失函数相对于模型参数的**梯度**。
2. 更新参数
根据计算得到的梯度信息，使用**优化算法**来更新模型参数。
3. 重复迭代
重复以上步骤直到满足停止条件，例如达到一定的迭代次数或损失函数收敛到一个稳定值。

常见优化器：

- 随机梯度下降（SGD）
- 动量优化（Momentum）
- Adam
- Adagrad
:::

### 使用现有网络模型及修改
[【B站视频】使用+修改网络模型](https://www.bilibili.com/video/BV1hE411t7RN?p=25&vd_source=b8e4081a996779a4645bea84fdcd2a69)
Downloading: "[https://download.pytorch.org/models/vgg16-397923af.pth"](https://download.pytorch.org/models/vgg16-397923af.pth")


#### 示例：vgg16 model
[computer-vision-vgg16](https://datagen.tech/guides/computer-vision/vgg16/)
![](https://cdn.nlark.com/yuque/0/2024/jpeg/21887339/1709534643690-fc049a3e-9f07-4d2e-a890-719262760cfd.jpeg#averageHue=%23f1f0ef&clientId=u8e74c415-d8b1-4&from=paste&id=WaUmw&originHeight=660&originWidth=1501&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=stroke&taskId=u0b80b114-dd5c-4d5e-bcba-81429de5fee&title=)
#### 示例：CIFAR 10 Model
![Structure-of-CIFAR10-quick-model.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709542652795-592802b1-7c19-4881-8325-998986f2855f.png#averageHue=%23dcdcdc&clientId=u8e74c415-d8b1-4&from=paste&height=201&id=PJ008&originHeight=201&originWidth=850&originalType=binary&ratio=1&rotation=0&showTitle=false&size=37315&status=done&style=stroke&taskId=u9041b238-7ade-4504-ab56-e9798fc8e9d&title=&width=850)


### 完整模型训练
[[demo] train](https://github.com/huzibo-yyds/pytorch_learn/blob/master/src/27_train.py)

1. 准备数据集
2. 准备DataLoader
3. 创建网络模型
4. 损失函数
5. 优化器
6. 预处理-设置训练过程中的一些参数
7. 训练
   1. 取数据
   2. 前向传播计算
   3. 计算loss
   4. 反向传播
   5. 更新参数-优化
8. 测试
9. 后处理-（保存模型|输出展示信息）


### GPU训练
[[demo] trian_gpu](https://github.com/huzibo-yyds/pytorch_learn/blob/master/src/27_trian_gpu.py)

【谷歌的服务，可以免费使用GPU训练，一周一定额度】
[https://colab.research.google.com/](https://colab.research.google.com/)


### 完整模型验证
使用训练好的模型来验证

### 如何看开源项目
[【up视频】看开源项目](https://www.bilibili.com/video/BV1hE411t7RN?p=33&vd_source=b8e4081a996779a4645bea84fdcd2a69)


【完结撒花】
![image.png](https://cdn.nlark.com/yuque/0/2024/png/21887339/1709714857628-ec8b7fea-43a6-4f2f-9887-efe03861ecfb.png#averageHue=%23bab69e&clientId=u33820a42-6950-4&from=paste&height=648&id=ued757127&originHeight=648&originWidth=1138&originalType=binary&ratio=1&rotation=0&showTitle=false&size=357752&status=done&style=stroke&taskId=ua1526ade-35ec-441a-aae3-36ab2b1566f&title=&width=1138)
