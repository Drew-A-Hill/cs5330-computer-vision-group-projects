"""
    Drew Hill & Abhiram Banda
    CS5330 Pattern Recognition & Computer Vision
    
    handwritten.py

    Runs the trained network on photos of handwritten numbers.
"""

import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import NeuralNetwork
import cv2

def load(weights_path):
    """
        Loads the trained network. Creates a new empty network and fills it with the
        saved weights.

        weights_path: str - the path of the saved weights.
        returns the model.
    """
    model = NeuralNetwork()
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    return model

def read_num_imgs(path):
    """
        Reads the image of the handwritten number and converts it to gray scale.

        path: str - Location of the img.
        returns the grayscale image as a numpy array.
    """
    img = cv2.imread(path)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return grey_img

def convert(grey_img):
    """
        Converts a grayscale photo into a tensor. Inverts it to white writing on
        black page, crops away the blank background so only the number is left, 
        then scales it to 20x20 and centers it in a 28x28 image.

        grey_img: numpy array - the grayscale img.
        returns a tensor.
    """
    # Inverts colors so theres a black background and white numbers
    inverted_img = cv2.bitwise_not(grey_img)

    # Finds the the number and crops out the background
    _, thresh = cv2.threshold(inverted_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(thresh))
    cropped_img = inverted_img[y:y + h, x:x + w]

    # Shrinks the number 
    scale = 20.0 / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    small_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Ensures background is black and writting is white
    small_img = cv2.normalize(small_img, None, 0, 255, cv2.NORM_MINMAX)

    # Ensures number is in the middle of a 28x28 frame
    canvas = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small_img

    # Converts to tensor
    tensor = torch.from_numpy(canvas).float().div(255.0)

    return tensor.unsqueeze(0).unsqueeze(0)   

def run_number_imgs(model, image_paths):
    """
        Runs all of the images of handwritten numbers through the model and gets the
        guesses and adds them to a list. 

        model: model -  the trained network.
        image_paths: list - List of all the image paths.
        returns a list of guesses.
    """
    guesses = []
    
    with torch.no_grad():
        # Iterates trhough all the images and gets the number and img path.
        for label, path in image_paths:
            num = read_num_imgs(path)
            tensor = convert(num)
            out = model(tensor)

            # Gets the index of the guess for the number.
            guess = out.argmax(1).item()

            # Adds guess to list 
            guesses.append(guess)
        
    return guesses

def display_results(guesses, image_paths):
    """
        Displays all ten images of the handwritten numbers with the 
        actual number and guess.

        guesses: list - List of guesses made by model.
        image_paths: list - List of all the image paths.
    """
    fig = plt.figure(figsize=(9,9))

    # Plots for each number 0-9
    for i in range(10):
        # Gets the image path
        path = image_paths[i][1]

        # Reads and converts the image
        img = convert(read_num_imgs(path)).squeeze()

        # Plots the images
        fig.add_subplot(3, 4, i + 1)
        plt.title(f"Number: {i} - Guess: {guesses[i]}")
        plt.axis("off")
        plt.imshow(img, cmap="gray")

    plt.tight_layout()
    plt.savefig("./results/handwritten_results.png", dpi=150, bbox_inches="tight")
    plt.show()

def main():
    """
        Loads the trained model and tests it on the handwritten numbers
        provided.
    """
    model = load("model_weights.pth")
    image_paths = []

    for i in range(10):
        path = f"./handwritten_imgs/{i}.png"

        # Checks path exists
        if not os.path.exists(path):
            print("Image path does not exist")
            return

        # Adds all of the paths to the list
        image_paths.append((i, path))

    # Runs each handwritten image through the model and gets the guesses
    guesses = run_number_imgs(model, image_paths)

    correct = 0

    # Determines how many numbers the model correctly guessed
    for j, guess in enumerate(guesses):
        if j == guess:
            correct += 1

    print(f"\nModel correctly identified {correct}/{len(guesses)} of the handwritten numbers")

    display_results(guesses, image_paths)
    return

if __name__ == "__main__":
    main()