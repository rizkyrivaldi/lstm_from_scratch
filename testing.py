import torch

model_weights = torch.load("weight.pt")

print(model_weights[3])