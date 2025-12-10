import torch
from torch import nn
from torch import optim
from Net import ConvNet
from Data import train_loader, valid_loader
import matplotlib.pyplot as plt

model = ConvNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.2)
epoch = 50
train_loss = []
model.train()
for epoch in range(epoch):
    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if i % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")

valid_loss = []
model.eval()
with torch.no_grad():
    for i, (image, label) in enumerate(valid_loader):
        output = model(image)
        loss = loss_fn(output, label)
        valid_loss.append(loss.item())
        if i % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")
plt.figure(figsize=(10, 8))
plt.plot(train_loss)
plt.plot(valid_loss)
plt.legend(['Training loss', 'Validation loss'])
plt.show()
torch.save(model.state_dict(), 'model.ckpt')
