"""
    Drew Hill & Abhiram Banda
    CS5330 Pattern Recognition & Computer Vision

    ext_pretrained.py

    Loads a pretrained model looks at the first two convolution layers.
"""
import sys
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision.models import resnet18, ResNet18_Weights

def load_pretrained():
    """
        Loads ResNet18 from torchvision.

        returns the model.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    print(model)

    return model

def get_first_layer_filters(model):
    """
        Gets the filters from the first layer.

        model: model - the pre trained ResNet18 model.
        returns the filter weights as a tensor.
    """
    weights = model.conv1.weight

    print(f"\nFirst layer filter shape: {weights.shape}")
    print(f"\nFirst filter, red channel:\n{weights[0, 0]}")

    return weights


def get_second_layer_filters(model):
    """
        Gets the filters from the second layer.

        model: model - the pre trained ResNet18 model.
        returns the filter weights as a tensor.
    """
    weights = model.layer1[0].conv1.weight

    print(f"\nSecond layer filter shape: {weights.shape}")

    return weights


def display_first_layer(weights):
    """
        Displays the filters of the first layer.

        weights: tensor - the first layer filter weights.
    """
    fig = plt.figure(figsize=(8, 8))

    with torch.no_grad():
        # Plots the filters.
        for i in range(16):
            filter = weights[i]
            filter = (filter - filter.min()) / (filter.max() - filter.min())

            # Creates a subplot for the filters.
            fig.add_subplot(4, 4, i + 1)
            plt.title(f"Filter {i}")
            plt.axis("off")

            # Moves the color channel to the end, the order imshow wants.
            plt.imshow(filter.permute(1, 2, 0))

    fig.suptitle("First Layer Filters")
    plt.tight_layout()
    plt.savefig("./results/pretrained_filters1.png", dpi=150, bbox_inches="tight")
    plt.show()


def display_second_layer(weights):
    """ 
        Displays the filters of the second layer.

        weights: tensor - the first layer filter weights.
    """
    fig = plt.figure(figsize=(8, 8))

    with torch.no_grad():
        # Plots the filters
        for i in range(16):
            filter = weights[i, 0]

            # Creates a subplot for the filters.
            fig.add_subplot(4, 4, i + 1)
            plt.title(f"Filter {i}")
            plt.axis("off")
            plt.imshow(filter, cmap="gray")
    
    fig.suptitle("Second Layer Filters")
    plt.tight_layout()
    plt.savefig("./results/pretrained_filters2.png", dpi=150, bbox_inches="tight")
    plt.show()


def display_filtered_images(weights, img):
    """
        Applies the first 10 filters of the first layer to an image.

        weights: tensor - the first layer filter weights.
        img: numpy array - the grayscale image to filter.
    """
    fig = plt.figure(figsize=(8, 10))

    with torch.no_grad():
        # Applies the first 10 filters to the image.
        for i in range(10):
            # Averages the 3 color channels down into one gray filter.
            filter = weights[i].mean(0).numpy()

            # Applies filter
            filtered_img = cv2.filter2D(img, -1, filter)

            # Creates a subplot for the filters.
            fig.add_subplot(5, 2, i + 1)
            plt.title(f"Filter {i}")
            plt.axis("off")
            plt.imshow(filtered_img, cmap="gray")

    plt.tight_layout()
    plt.savefig("./results/pretrained_filtered_images.png", dpi=150, bbox_inches="tight")
    plt.show()


def main(argv):
    """
        Loads a pre trained model, shows the filters of the first two layers.
    """
    model = load_pretrained()

    # Looks at the first layer.
    first_weights = get_first_layer_filters(model)
    display_first_layer(first_weights)

    # Looks at the second layer.
    second_weights = get_second_layer_filters(model)
    display_second_layer(second_weights)

    # Reads one of the handwritten numbers to run the first layer filters over
    img = cv2.imread("./handwritten_imgs/3.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

    display_filtered_images(first_weights, img)
    return
    
if __name__ == "__main__":
    main(sys.argv)