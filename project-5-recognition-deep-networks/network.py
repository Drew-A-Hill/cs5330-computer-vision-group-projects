from model import NeuralNetwork
import torch
import matplotlib.pyplot as plt

model = NeuralNetwork() 
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
#model.eval()

weights = model.conv1.weight

print(model)                                                                                                   
print(weights.shape)

figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 3
for i in range(1, 11):
    filter = weights[i-1, 0]
    figure.add_subplot(rows, cols, i)
    plt.title(f"filter {i-1}")
    plt.xticks([])                                                                                                 
    plt.yticks([])
    plt.imshow(filter.detach(), cmap="gray")
plt.show()