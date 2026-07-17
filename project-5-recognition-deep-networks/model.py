import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

## Uses the MNIST dataset from torchvision

##training dataset
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

##test dataset
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

## Map used for labelling the images when plotting
labels_map = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}

## From the example: plots the first 6 digits of the test dataset. 
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 3
j = 0
for i in range(1, cols * rows + 1):
    img, label = test_data[j]
    j+=1
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.drop1 = nn.Dropout(0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.linear1 = nn.Linear(in_features=20 * 4 * 4, out_features=50)
        self.linear2 = nn.Linear(in_features=50, out_features=10)


    def forward(self, x):
        x = nn.functional.relu(self.pool1(self.conv1(x)))
        x = nn.functional.relu(self.pool2(self.drop1(self.conv2(x))))
        x = x.view(-1, 20 * 4 * 4)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.log_softmax(self.linear2(x), dim=1)
        return x

model = NeuralNetwork()