"""
    Drew Hill & Abhiram Banda
    CS5330 Pattern Recognition & Computer Vision

    model.py

    Defines the CNN architecture for MNIST digit recognition,
    trains the model, and saves the weights to a file.
"""
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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    """CNN for MNIST digit classification with two conv layers, dropout, and two fully connected layers."""
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
        """Passes input through conv->pool->relu layers, then fully connected layers with log_softmax output."""
        x = nn.functional.relu(self.pool1(self.conv1(x)))
        x = nn.functional.relu(self.pool2(self.drop1(self.conv2(x))))
        x = x.view(-1, 20 * 4 * 4)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.log_softmax(self.linear2(x), dim=1)
        return x

if __name__ == "__main__":
    model = NeuralNetwork()

    ##Hyperparameters
    learning_rate = 1e-2
    batch_size = 64
    epochs = 8

    ## Using NLLLoss because its better for classification
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    train_counter = []


    def training_loop(dataloader, model, loss_fn, optimizer):
        """Runs one epoch of training: forward pass, loss computation, backpropagation, and weight updates."""
        size = len(dataloader.dataset)
        ## Set the model to training mode - important for batch normalization and dropout layers
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            ##Back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ## Gthering the losses and #of examples for the plot
            train_losses.append(loss.item())
            train_counter.append(examples_seen[0])
            examples_seen[0] += len(X)

            if batch % 100 == 0:
                current = batch * batch_size + len(X)
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader, model, loss_fn):
        """Evaluates the model on a dataset, returning the average loss. Prints accuracy and loss."""
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return test_loss

    ## Gathering the losses and #of examples for the plot
    test_losses = []
    test_counter = []
    examples_seen = [0]

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loop(train_dataloader, model, loss_fn, optimizer)
        print("Test set: ", end="")
        test_loss = test_loop(test_dataloader, model, loss_fn)
        test_losses.append(test_loss)
        test_counter.append(examples_seen[0])
    print("Done!")

    ##Task D - Save the Model
    torch.save(model.state_dict(), 'model_weights.pth')