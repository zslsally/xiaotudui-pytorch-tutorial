# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "imgs/airplane.png"
image = Image.open(image_path)
print(image)
image = image.convert("RGB")  # png默认是4通道（还有个透明通道），转换成3通道
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()]
)

image = transform(image)
print(image.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load(
    "src/tudui_29_gpu.pth", map_location=torch.device("cpu"), weights_only=False
)  # 加载模型，映射到cpu
print(model)
image = torch.reshape(image, (1, 3, 32, 32))  # 增加batch_size维度
model.eval()
with torch.no_grad():  # 可以节约内存
    output = model(image)
print(output)

print(output.argmax(1))
