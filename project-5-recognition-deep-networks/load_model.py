from matplotlib import pyplot as plt
import torch
from model import NeuralNetwork, test_data
from torch import nn

# Creates an instance of our NeuralNetwork class and loads it with the weights we saved previously
model = NeuralNetwork() 
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

# print out (on the command line) the 10 network output values (use only 2 decimal places), 
# the index of the max output value, and the correct label of the digit.
for i in range(10):
    # Extracts the image and the true label from the test data
    img, label = test_data[i]

    # Predicts the the digit
    pred = model(img.unsqueeze(0))
    pred_digit = pred.argmax().item()
    tensors = pred.data.tolist()[0]
    tensors = [round(x, 2) for x in tensors]

    print(tensors, label, pred_digit)


# From the example: plots the first 6 digits of the test dataset. 
figure = plt.figure(figsize=(9, 9))
cols, rows = 3, 3
for i in range(1, 10):
    img, label = test_data[i]
    pred = model(img.unsqueeze(0))
    pred_digit = pred.argmax().item()

    figure.add_subplot(rows, cols, i)
    plt.title(f" Prediction: {pred_digit}")
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()