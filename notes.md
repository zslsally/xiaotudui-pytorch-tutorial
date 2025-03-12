# Notes

从根目录运行文件，e.g. `xiaotudui-pytorch-tutorial %    python src/test.py`

## 使用pytorch

`dir()`：查看可用工具

`help()`：说明书

## 加载数据

`Dataset`: get data and labels, need to overwrite `__getitem__`

`Dataloader`: feed data into neural network in mini-batch

## TensorBoard

检查数据和训练结果

## Transforms

对图片数据进行变换

常用：`ToTensor`, `Normalize`, `Resize`, `Compose`

tensor数据类型：打包了一些神经网络需要的参数

`ToTensor`: 支持从PIL.Image或者opencv读（类型为numpy.ndarray）的图片

`Compose`：combine multiple transforms, 一步的输出需要和后一步输入类型匹配

多看官网文档。

不知道返回值的时候：

- `print()`
- `print(type())`
- debug

## torchvision中的数据集

`torchvision.dataset`

## DataLoader

`batch_size, shuffle=True, num_workers=0, drop_last=True`

## nn.Module

overwrite `__init__`, `forward()`

## Convolution Layer

`torch.nn.functional`

调整`input`和`kernel`的shape，`padding`, `stride`

`in_channel`: 输入图像的channel

`out_channel`: 叠加kernel

## Polling Layer

又称为下采样，作用是保留特征但减少数据量

`stride`: 默认为`kernel_size`

`ceil_mode`: True保留不足kernel_size的cell的polling结果

## Non-linear Activation

给模型引入非线性特征

## Normalization Layer

加快训练速度

## Linear Layer

线性变换

## Dropout Layer

防止过拟合

## Sequential

## Loss Function

1. 计算实际输出和目标之间的差距
2. 为更新输出提供一定依据（反向传播）

需要output和target shape对应

## Optimizer

## 现有网络模型的使用及修改

1. `torch.save()`
2. `torch.load()`

## 完整的模型训练套路

1. `model.py`: define model
2. `train.py`:
    - Load data
    - Define hyperparameter (learning rate, epoch, etc) 
    - Define loss function + optimizer
    - Setup epoch
    - Calculate loss and grad (`with torch.no_grad()`)
    - Use tensorboard to display results, save model for each epoch
    - Test model performance with test data
    - accurary: 主要用于分类问题

## 利用GPU训练

没有GPU可以用google colab

1. 模型，数据，损失函数：都调用`.cuda()`，optimizer不能移到GPU
2. 【更常用】
    - `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
    - 模型，数据，损失函数：都调用`.to(device)`
    - 有多个GPU可以用`device = torch.device("cuda：0")`

## 模型验证和测试

[test.py](src/test.py)

## 开源项目结构
