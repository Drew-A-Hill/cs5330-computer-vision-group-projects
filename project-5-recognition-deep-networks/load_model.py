from matplotlib import pyplot as plt
import torch
from model import NeuralNetwork, test_data, test_dataloader
from torch import nn

model = NeuralNetwork() 
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

# for i in range(10):
#     img, label = test_data[i]


#     pred = model(img.unsqueeze(0))
#     pred_digit = pred.argmax()
#     tensors = pred.data.tolist()[0]
#     tensors = [round(x, 2) for x in tensors]

    
#     print(tensors, label, pred_digit)


# From the example: plots the first 6 digits of the test dataset. 
figure = plt.figure(figsize=(9, 9))
cols, rows = 3, 3
for i in range(1, 10):
    img, label = test_data[i]
    pred = model(img.unsqueeze(0))
    pred_digit = pred.argmax()

    figure.add_subplot(rows, cols, i)
    plt.title(pred_digit.item())
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
