"""
    Drew Hill & Abhiram Banda
    CS5330 Pattern Recognition & Computer Vision

    task3.py

    Transfer learning: reuses the pretrained MNIST CNN to classify
    Greek letters (alpha, beta, gamma) by freezing weights and
    replacing the last layer.
"""
from matplotlib import pyplot as plt
import torch
from model import NeuralNetwork
from torch import nn
import torchvision

model = NeuralNetwork() 
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

training_set_path = "./greek_train"

# freezes the parameters for the whole network
for param in model.parameters():
    param.requires_grad = False

##Replaces the last linear layer with one that has 3 nodes for the output instead of 10
model.linear2 = nn.Linear(50, 3)

##Hyperparameters
learning_rate = 1e-2
batch_size = 5
epochs = 180

## Using NLLLoss because its better for classification
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# greek data set transform
class GreekTransform:
    """Transforms Greek letter images to match MNIST format: grayscale, scaled, cropped to 28x28, and inverted."""
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
    
train_losses = []
train_counter = []
examples_seen = [0]

def training_loop(dataloader, model, loss_fn, optimizer):
        """Runs one epoch of training on the Greek letter dataset, tracking losses for plotting."""
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


# DataLoader for the Greek data set
greek_train_dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder( 
        training_set_path, 
        transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
            GreekTransform(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,) ) ] ) 
        ),
        batch_size = batch_size,
        shuffle = True 
)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    training_loop(greek_train_dataloader, model, loss_fn, optimizer)
    print("Done!")

print(model)

# ## Plot training error
# plt.figure(figsize=(8, 5))
# plt.plot(train_counter, train_losses, color='blue', label="Train loss")
# plt.xlabel("number of training examples seen")
# plt.ylabel("negative log likelihood loss")
# plt.legend()
# plt.savefig("greek_training_error.png", dpi=150, bbox_inches='tight')
# plt.show()


torch.save(model.state_dict(), 'greek_model_weights.pth')

## Test on custom Greek letter images
labels_map = {0: "alpha", 1: "beta", 2: "gamma"}
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    GreekTransform(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

model.eval()
from PIL import Image
test_images = [("alpha", "./greek_test/alpha.png"), ("beta", "./greek_test/beta.png"), ("gamma", "./greek_test/gamma.png")]

figure = plt.figure(figsize=(9, 3))
for i, (true_label, path) in enumerate(test_images):
    img = Image.open(path)
    img_tensor = test_transform(img).unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)

    pred_label = labels_map[pred.argmax().item()]
    values = [round(x, 2) for x in pred.data.tolist()[0]]
    print(f"True: {true_label}, Predicted: {pred_label}, Output: {values}")

    figure.add_subplot(1, 3, i + 1)
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_tensor.squeeze(), cmap="gray")

plt.tight_layout()
plt.savefig("./results/greek_test_results.png", dpi=150, bbox_inches='tight')
plt.show()