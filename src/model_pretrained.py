# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 在数据集上训练好的权重

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10(
    "../dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True
)

vgg16_true.classifier.add_module(
    "add_linear", nn.Linear(1000, 10)
)  # 给模型的classifier section加一个线性层，输出维度是10
print(vgg16_true)

print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 更换一个module
print(vgg16_false)
