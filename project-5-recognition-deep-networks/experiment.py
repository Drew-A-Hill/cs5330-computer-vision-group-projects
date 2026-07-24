import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
import time

## Fashion MNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

## Configurable CNN
class ConfigurableCNN(nn.Module):
    def __init__(self, num_filters=10, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, 5)
        self.drop1 = nn.Dropout(dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.num_filters = num_filters
        self.linear1 = nn.Linear(num_filters * 2 * 4 * 4, 50)
        self.linear2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.drop1(self.conv2(x))))
        x = x.view(-1, self.num_filters * 2 * 4 * 4)
        x = F.relu(self.linear1(x))
        x = F.log_softmax(self.linear2(x), dim=1)
        return x


def train_and_evaluate(num_filters, dropout_rate, batch_size, epochs=5):
    """Train a model with given params and return test accuracy and training time."""
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = ConfigurableCNN(num_filters=num_filters, dropout_rate=dropout_rate)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.time()

    ## Training
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    train_time = time.time() - start_time

    ## Evaluate on test set
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    test_loss = 0
    num_batches = len(test_loader)
    with torch.no_grad():
        for X, y in test_loader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    accuracy = correct / total * 100
    test_loss /= num_batches

    return accuracy, test_loss, train_time


## Experiment parameters
filter_options = [5, 10, 20, 40]
dropout_options = [0.0, 0.25, 0.5, 0.75]
batch_options = [16, 32, 64, 128]

## Defaults
default_filters = 10
default_dropout = 0.5
default_batch = 64
epochs = 5

results = []

## Round 1: Vary number of filters (hold dropout=0.5, batch=64)
print("=" * 60)
print("ROUND 1: Varying number of conv filters")
print("=" * 60)
best_filters = default_filters
best_acc = 0
for nf in filter_options:
    acc, loss, t = train_and_evaluate(nf, default_dropout, default_batch, epochs)
    print(f"  Filters={nf:3d}  |  Accuracy={acc:.2f}%  |  Loss={loss:.4f}  |  Time={t:.1f}s")
    results.append({"dim": "filters", "value": nf, "accuracy": acc, "loss": loss, "time": t,
                     "filters": nf, "dropout": default_dropout, "batch": default_batch})
    if acc > best_acc:
        best_acc = acc
        best_filters = nf

print(f"  Best filters: {best_filters} ({best_acc:.2f}%)\n")

## Round 2: Vary dropout rate (use best filters, hold batch=64)
print("=" * 60)
print("ROUND 2: Varying dropout rate")
print("=" * 60)
best_dropout = default_dropout
best_acc = 0
for dr in dropout_options:
    acc, loss, t = train_and_evaluate(best_filters, dr, default_batch, epochs)
    print(f"  Dropout={dr:.2f}  |  Accuracy={acc:.2f}%  |  Loss={loss:.4f}  |  Time={t:.1f}s")
    results.append({"dim": "dropout", "value": dr, "accuracy": acc, "loss": loss, "time": t,
                     "filters": best_filters, "dropout": dr, "batch": default_batch})
    if acc > best_acc:
        best_acc = acc
        best_dropout = dr

print(f"  Best dropout: {best_dropout} ({best_acc:.2f}%)\n")

## Round 3: Vary batch size (use best filters + best dropout)
print("=" * 60)
print("ROUND 3: Varying batch size")
print("=" * 60)
best_batch = default_batch
best_acc = 0
for bs in batch_options:
    acc, loss, t = train_and_evaluate(best_filters, best_dropout, bs, epochs)
    print(f"  Batch={bs:4d}   |  Accuracy={acc:.2f}%  |  Loss={loss:.4f}  |  Time={t:.1f}s")
    results.append({"dim": "batch", "value": bs, "accuracy": acc, "loss": loss, "time": t,
                     "filters": best_filters, "dropout": best_dropout, "batch": bs})
    if acc > best_acc:
        best_acc = acc
        best_batch = bs

print(f"  Best batch size: {best_batch} ({best_acc:.2f}%)\n")

## Round 2 pass: Re-optimize filters and dropout with best batch
print("=" * 60)
print("ROUND 4: Re-optimizing filters with best dropout & batch")
print("=" * 60)
for nf in filter_options:
    acc, loss, t = train_and_evaluate(nf, best_dropout, best_batch, epochs)
    print(f"  Filters={nf:3d}  |  Accuracy={acc:.2f}%  |  Loss={loss:.4f}  |  Time={t:.1f}s")
    results.append({"dim": "filters_r2", "value": nf, "accuracy": acc, "loss": loss, "time": t,
                     "filters": nf, "dropout": best_dropout, "batch": best_batch})
    if acc > best_acc:
        best_acc = acc
        best_filters = nf

print(f"  Best filters: {best_filters} ({best_acc:.2f}%)\n")

print("=" * 60)
print("ROUND 5: Re-optimizing dropout with best filters & batch")
print("=" * 60)
for dr in dropout_options:
    acc, loss, t = train_and_evaluate(best_filters, dr, best_batch, epochs)
    print(f"  Dropout={dr:.2f}  |  Accuracy={acc:.2f}%  |  Loss={loss:.4f}  |  Time={t:.1f}s")
    results.append({"dim": "dropout_r2", "value": dr, "accuracy": acc, "loss": loss, "time": t,
                     "filters": best_filters, "dropout": dr, "batch": best_batch})
    if acc > best_acc:
        best_acc = acc
        best_dropout = dr

print(f"  Best dropout: {best_dropout} ({best_acc:.2f}%)\n")

print("=" * 60)
print("ROUND 6: Re-optimizing batch size with best filters & dropout")
print("=" * 60)
for bs in batch_options:
    acc, loss, t = train_and_evaluate(best_filters, best_dropout, bs, epochs)
    print(f"  Batch={bs:4d}   |  Accuracy={acc:.2f}%  |  Loss={loss:.4f}  |  Time={t:.1f}s")
    results.append({"dim": "batch_r2", "value": bs, "accuracy": acc, "loss": loss, "time": t,
                     "filters": best_filters, "dropout": best_dropout, "batch": bs})
    if acc > best_acc:
        best_acc = acc
        best_batch = bs

print(f"  Best batch size: {best_batch} ({best_acc:.2f}%)\n")

## Final best config
print("=" * 60)
print(f"BEST CONFIG: filters={best_filters}, dropout={best_dropout}, batch={best_batch}")
print(f"BEST ACCURACY: {best_acc:.2f}%")
print("=" * 60)

## Save results to CSV
with open("experiment_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["dim", "value", "accuracy", "loss", "time", "filters", "dropout", "batch"])
    writer.writeheader()
    writer.writerows(results)
print("Results saved to experiment_results.csv")

## Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

## Plot 1: Filters vs accuracy (round 1)
r1 = [r for r in results if r["dim"] == "filters"]
axes[0].plot([r["value"] for r in r1], [r["accuracy"] for r in r1], marker='o')
axes[0].set_xlabel("Number of Conv Filters")
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].set_title("Effect of Conv Filters")

## Plot 2: Dropout vs accuracy (round 2)
r2 = [r for r in results if r["dim"] == "dropout"]
axes[1].plot([r["value"] for r in r2], [r["accuracy"] for r in r2], marker='o', color='orange')
axes[1].set_xlabel("Dropout Rate")
axes[1].set_ylabel("Test Accuracy (%)")
axes[1].set_title("Effect of Dropout Rate")

## Plot 3: Batch size vs accuracy (round 3)
r3 = [r for r in results if r["dim"] == "batch"]
axes[2].plot([r["value"] for r in r3], [r["accuracy"] for r in r3], marker='o', color='green')
axes[2].set_xlabel("Batch Size")
axes[2].set_ylabel("Test Accuracy (%)")
axes[2].set_title("Effect of Batch Size")

plt.tight_layout()
plt.savefig("experiment_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plots saved to experiment_results.png")