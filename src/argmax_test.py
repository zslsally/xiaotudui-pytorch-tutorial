import torch

outputs = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

preds = outputs.argmax(1)
print(preds)

targets = torch.tensor([1, 0, 1])
print(preds == targets)
