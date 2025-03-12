# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）,只保存模型参数，不保存模型结构，内存更小
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
## 加载时需要提前定义好模型结构,然后加载模型参数


# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")
