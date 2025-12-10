import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, random_split


def preprocessing(x, y):
    x = x.type(torch.float32)
    y = y.type(torch.float32)
    x = (x - x.mean()) / x.std()
    x = x.unsqueeze(1)
    y = y.unsqueeze(1)
    return x, y


t = transforms.Compose([
    transforms.ToTensor(),
])

train = MNIST(root='./data', train=True, transform=t, download=False)
test = MNIST(root='./data', train=False, transform=t, download=False)

x_train, y_train = train.data, train.targets
x_test, y_test = test.data, test.targets

x_train, y_train = preprocessing(x_train, y_train)
x_test, y_test = preprocessing(x_test, y_test)

train_set = TensorDataset(x_train, y_train)
train_set, valid_set = random_split(train_set, [50000, 10000])

test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
