import torch
from torch import nn
from Data import test_loader
from Net import ConvNet
model = torch.load("model.ckpt", map_location=torch.device('cpu'), weights_only=False)
loss_fn = nn.CrossEntropyLoss()
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        loss = loss_fn(output, labels)
        print(loss.item())
