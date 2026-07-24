"""
    Drew Hill & Abhiram Banda
    CS5330 Pattern Recognition & Computer Vision

    task2b.py

    Applies the 10 learned conv1 filters to the first training image
    using OpenCV's filter2D and plots the filtered results.
"""
import torch
import cv2
import matplotlib.pyplot as plt
from model import NeuralNetwork, training_data

model = NeuralNetwork()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

## Get the first training image as a numpy array
img, label = training_data[0]
img_numpy = img.squeeze().numpy()

## Get the 10 filters from conv1 and apply each to the image
weights = model.conv1.weight

figure = plt.figure(figsize=(8, 10))
with torch.no_grad():
    for i in range(10):
        filter = weights[i, 0]
        filter_numpy = filter.numpy()
        filtered_img = cv2.filter2D(img_numpy, -1, filter_numpy)

        ## Column 1 or 3: filter, Column 2 or 4: filtered image
        row = i // 2
        col = (i % 2) * 2

        ## Plot the filter
        figure.add_subplot(5, 4, row * 4 + col + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filter_numpy, cmap="gray")

        ## Plot the filtered image
        figure.add_subplot(5, 4, row * 4 + col + 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filtered_img, cmap="gray")

plt.tight_layout()
plt.savefig("./results/filtered_images.png", dpi=150, bbox_inches='tight')
plt.show()
