import torch

print(torch.cuda.is_available())

print(dir(torch))

print(dir(torch.cuda))

help(torch.cuda.is_available)  # 没有()
